"""Semantic audio-event classification (YAMNet-equivalent).

Unlike ``audio_events.py`` (structural librosa analysis: silence / music-noise),
this names *what a sound is* using an AudioSet-trained classifier — the same 527
-class ontology YAMNet uses (Speech, Music, Applause, Laughter, Dog, Typing, ...).

We use the MIT Audio Spectrogram Transformer (AST) fine-tuned on AudioSet, run
via ``transformers`` on PyTorch. That avoids a TensorFlow dependency (classic
YAMNet is TF-Hub, which has no usable aarch64 + CUDA 13 build) while giving the
same labels, GPU-accelerated on the GB10.

Because torch + transformers live in the GPU ASR environment (not the lean main
venv), classification is delegated to that interpreter via subprocess — the same
pattern as ``audio.py``. Set ``audio.asr_python`` (or ``VIDEO_ANALYSIS_ASR_PYTHON``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

ASR_PYTHON_ENV = "VIDEO_ANALYSIS_ASR_PYTHON"

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet labels that are ambient/uninformative for long-form talking content.
# These dominate speech recordings and add no signal, so we drop them from the
# event stream (they're still what the model sees — we just don't emit them).
_IGNORED_LABELS = frozenset({
    "Speech", "Narration, monologue", "Conversation", "Speech synthesizer",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Inside, small room", "Inside, large room or hall", "Silence",
})


@dataclass
class SemanticEvent:
    """A detected semantic sound event spanning [start, end)."""
    start_seconds: float
    end_seconds: float
    label: str
    confidence: float

    @property
    def timestamp_seconds(self) -> float:
        return self.start_seconds


def classify_sound_events(audio_path: str,
                          model: str = DEFAULT_MODEL,
                          window_seconds: float = 2.0,
                          hop_seconds: float = 1.0,
                          score_threshold: float = 0.4,
                          asr_python: str | None = None) -> list[SemanticEvent]:
    """Classify semantic sound events in an audio file.

    Slides a window over the audio, classifies each window with AST, keeps
    labels above ``score_threshold`` (excluding ambient speech labels), and
    merges consecutive windows carrying the same label into one event.

    Args:
        audio_path: Path to a WAV file (loaded at 16 kHz mono).
        model: HF model id of an AudioSet audio-classification model.
        window_seconds: Analysis window length.
        hop_seconds: Hop between windows.
        score_threshold: Minimum class probability to emit.
        asr_python: Interpreter with torch + transformers. When set (or via
            ``VIDEO_ANALYSIS_ASR_PYTHON``) and different from the current one,
            classification runs there via subprocess.

    Returns:
        Merged SemanticEvent list sorted by start time.
    """
    asr_python = asr_python or os.environ.get(ASR_PYTHON_ENV)
    if asr_python and os.path.realpath(asr_python) != os.path.realpath(sys.executable):
        return _classify_via_subprocess(
            asr_python, audio_path, model, window_seconds, hop_seconds, score_threshold)
    return _classify_inprocess(
        audio_path, model, window_seconds, hop_seconds, score_threshold)


# How much source audio to hold in memory (as float32 samples) at once. Windows
# are still streamed into the classifier lazily within a segment, but capping
# the segment itself bounds peak host memory on very long recordings and gives
# us a point to release the CUDA allocator cache between segments — this
# matters on unified-memory hardware (e.g. GB10) where GPU memory is shared
# with other resident processes.
_SEGMENT_SECONDS = 600.0  # 10 minutes


def _cuda_status() -> tuple[bool, str]:
    """Return (usable, reason). Distinguishes a truly absent GPU from a present
    GPU whose CUDA context can't initialize.

    ``torch.cuda.is_available()`` swallows *all* errors and returns False —
    including the case that bites us on unified-memory hardware (GB10), where the
    GPU exists but ``cudaGetDeviceCount()`` fails with "out of memory" because the
    shared pool is already full (resident llama.cpp models, etc.). Conflating that
    with "no GPU" is exactly what made AST silently fall back to CPU and OOM the
    whole box. Here we force CUDA init and report the real reason so it's loud.
    """
    try:
        import torch
    except Exception as e:  # torch not installed in this interpreter
        return False, f"torch import failed: {e}"
    try:
        torch.cuda.init()
        if torch.cuda.device_count() > 0:
            return True, "ok"
        return False, "no CUDA device present"
    except Exception as e:
        # GPU may well be there; the context just couldn't be created (usually
        # OOM on the shared pool). Surfacing this points at the real fix: free
        # GPU/host memory before this runs, not "the GPU disappeared".
        return False, f"CUDA present but init failed ({type(e).__name__}: {e})"


def _classify_inprocess(audio_path: str, model: str, window_seconds: float,
                        hop_seconds: float, score_threshold: float) -> list[SemanticEvent]:
    """Run AST classification in the current interpreter (needs torch/transformers).

    Requires a usable CUDA device. AST on CPU is not a graceful degradation: each
    window pads to a fixed ~1024-frame spectrogram through a ViT-sized transformer,
    and on a long recording the CPU forward pass exhausts memory (fatal on
    unified-memory boxes) while pinning the CPU at ~10x the GPU runtime. We refuse
    rather than fall back — the caller drops to the cheap librosa detector instead.
    """
    import librosa
    import torch
    from transformers import pipeline

    cuda_ok, reason = _cuda_status()
    if not cuda_ok:
        raise RuntimeError(
            f"Semantic audio classification (AST) requires a usable GPU but "
            f"none is available: {reason}. Refusing to run on CPU — it would "
            f"exhaust memory on long audio. Free GPU/host memory (e.g. unload "
            f"resident models) or set audio_events_backend=librosa."
        )
    device = 0
    clf = pipeline("audio-classification", model=model, device=device)

    sr = 16000
    win = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if win <= 0 or hop <= 0:
        return []

    total_duration = librosa.get_duration(path=audio_path)
    raw: list[tuple[float, str, float]] = []

    seg_start = 0.0
    while seg_start < total_duration:
        seg_end = min(seg_start + _SEGMENT_SECONDS, total_duration)
        # Overlap by one window so events spanning a segment boundary aren't missed.
        y, _ = librosa.load(audio_path, sr=sr, mono=True,
                             offset=seg_start, duration=seg_end - seg_start + window_seconds)
        if len(y) >= win:
            starts = range(0, len(y) - win + 1, hop)
            windows = ({"array": y[s:s + win], "sampling_rate": sr} for s in starts)
            offsets = list(starts)
            results = clf(windows, top_k=3, batch_size=16)

            for s, preds in zip(offsets, results):
                for p in preds:
                    label = p["label"]
                    if label in _IGNORED_LABELS:
                        continue
                    if p["score"] >= score_threshold:
                        raw.append((seg_start + s / sr, label, float(p["score"])))
                    break  # only the top non-ignored label per window

        del y
        # Release the CUDA allocator cache between segments — matters on
        # unified-memory hardware where this pool is shared with host RAM.
        torch.cuda.empty_cache()
        seg_start += _SEGMENT_SECONDS

    return _merge_runs(raw, window_seconds, hop_seconds)


def _merge_runs(raw: list[tuple[float, str, float]],
                window_seconds: float, hop_seconds: float) -> list[SemanticEvent]:
    """Merge consecutive windows sharing a label into single events."""
    events: list[SemanticEvent] = []
    for start_t, label, score in raw:
        end_t = start_t + window_seconds
        contiguous = events and events[-1].label == label and (
            start_t - events[-1].end_seconds <= hop_seconds + 1e-6)
        if contiguous:
            # Extend the current run; keep the max confidence seen.
            events[-1].end_seconds = round(end_t, 3)
            events[-1].confidence = round(max(events[-1].confidence, score), 3)
        else:
            events.append(SemanticEvent(
                start_seconds=round(start_t, 3),
                end_seconds=round(end_t, 3),
                label=label,
                confidence=round(score, 3),
            ))
    return events


# Hard address-space ceiling (bytes) for the classifier subprocess. This is a
# last-resort backstop: with the GPU-required guard above, AST should never run
# on CPU and blow up host memory — but if some future path regresses, this turns
# a whole-box OOM/hang into a clean subprocess failure we catch and report,
# rather than something that forces a power-cycle. 24 GiB comfortably fits GPU
# CUDA context + model + one audio segment while leaving the box responsive.
_SUBPROCESS_MEM_LIMIT_BYTES = 24 * 1024**3


def _limit_address_space() -> None:  # pragma: no cover - runs in child, pre-exec
    """preexec_fn: cap the child's virtual address space (POSIX only)."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = _SUBPROCESS_MEM_LIMIT_BYTES
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except Exception:
        # Best-effort; if we can't set it, proceed without the backstop.
        pass


def _classify_via_subprocess(asr_python: str, audio_path: str, model: str,
                             window_seconds: float, hop_seconds: float,
                             score_threshold: float) -> list[SemanticEvent]:
    """Run classification under a different interpreter, parse back JSON."""
    cmd = [
        asr_python, "-m", "video_analysis.stage0.semantic_events",
        "--audio", audio_path, "--model", model,
        "--window", str(window_seconds), "--hop", str(hop_seconds),
        "--threshold", str(score_threshold),
    ]
    logger.info("Delegating semantic audio classification to %s", asr_python)
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    # Use an expandable CUDA allocator. On this unified-memory box (GB10) the
    # default caching allocator fragments and raises a spurious "CUDA out of
    # memory" mid-forward-pass even with tens of GiB free (a 228 MiB alloc failed
    # with 81 GiB free). expandable_segments avoids that fragmentation so AST
    # actually completes on GPU instead of erroring out to the librosa fallback.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          preexec_fn=_limit_address_space)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Semantic audio-event subprocess failed:\n{proc.stderr[-2000:]}")

    payload = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(
            f"Semantic audio-event subprocess produced no JSON:\n{proc.stdout[-2000:]}")

    return [SemanticEvent(**e) for e in payload]


def _main() -> None:
    """Worker entrypoint: classify and print events as a JSON list (last line)."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic audio-event worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    events = _classify_inprocess(
        args.audio, args.model, args.window, args.hop, args.threshold)
    print(json.dumps([asdict(e) for e in events]))


if __name__ == "__main__":
    _main()
