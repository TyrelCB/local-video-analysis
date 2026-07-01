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


def _classify_inprocess(audio_path: str, model: str, window_seconds: float,
                        hop_seconds: float, score_threshold: float) -> list[SemanticEvent]:
    """Run AST classification in the current interpreter (needs torch/transformers)."""
    import librosa
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=model, device=device)

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    win = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if win <= 0 or hop <= 0 or len(y) < win:
        return []

    # Build windows, then classify in batches for GPU efficiency.
    starts = list(range(0, len(y) - win + 1, hop))
    batch = [{"array": y[s:s + win], "sampling_rate": 16000} for s in starts]
    results = clf(batch, top_k=3, batch_size=16)

    # Per-window top label (excluding ambient speech), then merge runs.
    raw: list[tuple[float, str, float]] = []
    for s, preds in zip(starts, results):
        for p in preds:
            label = p["label"]
            if label in _IGNORED_LABELS:
                continue
            if p["score"] >= score_threshold:
                raw.append((s / sr, label, float(p["score"])))
            break  # only the top non-ignored label per window

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
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
