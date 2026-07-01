"""Audio extraction and transcription.

Extracts audio from video using FFmpeg (16kHz mono WAV) and transcribes it
with word-level timestamps. Two backends are supported:

- ``faster_whisper``: CTranslate2 engine. Fast on x86 CUDA and on CPU, but the
  aarch64 PyPI wheel ships *without* CUDA support, so on ARM boxes (e.g. the
  GB10) it silently falls back to CPU — the historical bottleneck here.
- ``torch_whisper``: reference openai-whisper on PyTorch. Runs on CUDA wherever
  a CUDA-enabled torch wheel is installed (including aarch64 + Blackwell),
  giving a ~3x speedup over faster-whisper-on-CPU on this hardware.

When ``faster_whisper`` is requested but cannot reach a GPU while a CUDA torch
build is available, we transparently switch to ``torch_whisper`` so the GPU is
actually used.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Optional path to a Python interpreter that has the GPU ASR stack (torch +
# openai-whisper and/or NeMo). When set and different from the running
# interpreter, GPU backends are delegated to it via subprocess so the main
# project venv doesn't need the multi-GB CUDA dependencies. Defaults to the
# config value, overridable by this env var.
ASR_PYTHON_ENV = "VIDEO_ANALYSIS_ASR_PYTHON"


@dataclass
class TranscriptionSegment:
    """A single segment from the transcription."""
    text: str
    start_seconds: float
    end_seconds: float
    words: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class TranscriptionResult:
    """Complete transcription result for a video/audio."""
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0
    model_used: str = ""

    @property
    def full_text(self) -> str:
        """Join all segment texts."""
        return " ".join(seg.text for seg in self.segments)


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from a video file as 16kHz mono WAV.

    Args:
        video_path: Path to the input video.
        output_path: Path for the output WAV file.

    Returns:
        Path to the extracted audio file.
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH") from None

    return output_path


def _ct2_has_cuda() -> bool:
    """True if the installed CTranslate2 build can see a CUDA device."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _torch_has_cuda() -> bool:
    """True if a CUDA-enabled torch build with a visible device is installed."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def transcribe_audio(audio_path: str, model_size: str = "medium",
                     language: str = "auto",
                     backend: str = "faster_whisper",
                     device: str | None = None,
                     asr_python: str | None = None) -> TranscriptionResult:
    """Transcribe audio with word-level timestamps.

    Args:
        audio_path: Path to the WAV audio file.
        model_size: Whisper model size (tiny, base, small, medium, large,
            large-v3, ...). ``large-v3`` is cheap on GPU and more accurate.
        language: Language code or 'auto'.
        asr_python: Path to a Python interpreter that has the GPU ASR stack.
            When set (or via the ``VIDEO_ANALYSIS_ASR_PYTHON`` env var) and it
            differs from the running interpreter, ``parakeet`` / ``torch_whisper``
            are run there via subprocess. Lets the main venv stay lean.
        backend: One of:
            - ``parakeet``: NVIDIA NeMo Parakeet (English-only). ~70x realtime
              on this GPU — by far the fastest, but only for English audio.
            - ``torch_whisper``: openai-whisper on PyTorch. Multilingual (99
              languages), GPU on aarch64. The right multilingual default.
            - ``faster_whisper``: CTranslate2. Multilingual; fast on x86 CUDA /
              CPU, but the aarch64 wheel is CPU-only. When it can't reach a GPU
              and a CUDA torch build is present, it auto-falls back to
              ``torch_whisper``.
        device: Force a device ('cuda' or 'cpu'). Auto-detected when None.
    """
    # Delegate GPU backends to a dedicated ASR interpreter if one is configured
    # and we're not already running under it.
    asr_python = asr_python or os.environ.get(ASR_PYTHON_ENV)
    if (backend in ("parakeet", "torch_whisper") and asr_python
            and os.path.realpath(asr_python) != os.path.realpath(sys.executable)):
        return _transcribe_via_subprocess(
            asr_python, audio_path, model_size, language, backend, device)

    if backend == "faster_whisper" and not _ct2_has_cuda() and _torch_has_cuda():
        logger.info(
            "faster_whisper has no CUDA device (aarch64 wheel is CPU-only) but a "
            "CUDA torch build is available — using torch_whisper backend for GPU "
            "acceleration."
        )
        backend = "torch_whisper"

    if backend == "parakeet":
        return _transcribe_parakeet(audio_path, model_size, device)
    if backend == "torch_whisper":
        return _transcribe_torch_whisper(audio_path, model_size, language, device)
    return _transcribe_faster_whisper(audio_path, model_size, language, device)


def _transcribe_via_subprocess(asr_python: str, audio_path: str, model_size: str,
                               language: str, backend: str,
                               device: str | None) -> TranscriptionResult:
    """Run a GPU backend under a different interpreter and parse back JSON.

    This module is invoked as ``python -m video_analysis.stage0.audio`` (see the
    ``__main__`` block) which prints a single JSON object on stdout.
    """
    cmd = [
        asr_python, "-m", "video_analysis.stage0.audio",
        "--audio", audio_path, "--model", model_size,
        "--language", language, "--backend", backend,
    ]
    if device:
        cmd += ["--device", device]

    logger.info("Delegating %s transcription to %s", backend, asr_python)
    env = dict(os.environ)
    # Ensure the worker can import this package.
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ASR subprocess ({backend}) failed:\n{proc.stderr[-2000:]}"
        )
    # The worker may emit framework logs on stdout; the JSON is the last line.
    payload = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(
            f"ASR subprocess ({backend}) produced no JSON result:\n{proc.stdout[-2000:]}"
        )

    result = TranscriptionResult(
        language=payload.get("language", ""),
        duration_seconds=payload.get("duration_seconds", 0.0),
        model_used=payload.get("model_used", ""),
    )
    result.segments = [
        TranscriptionSegment(
            text=s["text"],
            start_seconds=s["start_seconds"],
            end_seconds=s["end_seconds"],
            words=s.get("words", []),
            confidence=s.get("confidence", 0.0),
        )
        for s in payload.get("segments", [])
    ]
    return result


def _transcribe_faster_whisper(audio_path: str, model_size: str,
                               language: str,
                               device: str | None) -> TranscriptionResult:
    """Transcribe using faster-whisper (CTranslate2)."""
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if _ct2_has_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Load the model — faster-whisper caches models on first load
    model = WhisperModel(model_size, device=device, compute_type=compute_type,
                         download_root=os.path.expanduser("~/.cache/whisper"))

    segments, info = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        word_timestamps=True,
        log_progress=False,
    )

    result = TranscriptionResult()
    result.language = info.language
    result.duration_seconds = round(getattr(info, "duration", 0.0), 3)
    result.model_used = f"faster-whisper-{model_size}"

    for seg in segments:
        words = []
        if hasattr(seg, "words") and seg.words:
            words = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 3) if hasattr(w, "probability") else 0.0,
                }
                for w in seg.words
            ]

        result.segments.append(TranscriptionSegment(
            text=seg.text.strip(),
            start_seconds=round(seg.start, 3),
            end_seconds=round(seg.end, 3),
            words=words,
            confidence=round(seg.probability, 3) if hasattr(seg, "probability") else 0.0,
        ))

    return result


def _transcribe_torch_whisper(audio_path: str, model_size: str,
                              language: str,
                              device: str | None) -> TranscriptionResult:
    """Transcribe using openai-whisper on PyTorch (GPU-capable on aarch64)."""
    import whisper

    if device is None:
        device = "cuda" if _torch_has_cuda() else "cpu"
    use_fp16 = device == "cuda"

    model = whisper.load_model(model_size, device=device)
    out = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        word_timestamps=True,
        fp16=use_fp16,
        verbose=False,
    )

    result = TranscriptionResult()
    result.language = out.get("language", "")
    result.model_used = f"openai-whisper-{model_size}"

    last_end = 0.0
    for seg in out.get("segments", []):
        words = []
        for w in seg.get("words", []) or []:
            # openai-whisper uses "probability"; key/word names carry whitespace
            words.append({
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0.0), 3),
                "end": round(w.get("end", 0.0), 3),
                "confidence": round(w.get("probability", 0.0), 3),
            })

        end = round(seg.get("end", 0.0), 3)
        last_end = max(last_end, end)
        # openai-whisper reports avg_logprob (log-domain); convert to a 0–1
        # confidence so it matches the faster-whisper segment scale.
        avg_logprob = seg.get("avg_logprob")
        confidence = round(math.exp(avg_logprob), 3) if avg_logprob is not None else 0.0

        result.segments.append(TranscriptionSegment(
            text=seg.get("text", "").strip(),
            start_seconds=round(seg.get("start", 0.0), 3),
            end_seconds=end,
            words=words,
            confidence=confidence,
        ))

    result.duration_seconds = last_end
    return result


def _result_to_json(result: TranscriptionResult) -> str:
    """Serialize a TranscriptionResult to a one-line JSON string."""
    payload = {
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "model_used": result.model_used,
        "segments": [asdict(s) for s in result.segments],
    }
    return json.dumps(payload)


def _main() -> None:
    """Worker entrypoint: run a backend and print the result as JSON.

    Invoked as ``python -m video_analysis.stage0.audio`` by
    ``_transcribe_via_subprocess`` under the ASR interpreter.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Transcription worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--backend", default="faster_whisper")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Call the backend directly (no asr_python) to avoid re-delegating.
    if args.backend == "parakeet":
        result = _transcribe_parakeet(args.audio, args.model, args.device)
    elif args.backend == "torch_whisper":
        result = _transcribe_torch_whisper(args.audio, args.model, args.language, args.device)
    else:
        result = _transcribe_faster_whisper(args.audio, args.model, args.language, args.device)

    # Emit JSON as the final stdout line (frameworks may log above this).
    print(_result_to_json(result))


# Whisper model-size names that don't apply to NeMo — used to decide whether a
# configured ``whisper_model`` value is actually a NeMo model id.
_WHISPER_SIZES = {
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3", "turbo",
    "tiny.en", "base.en", "small.en", "medium.en",
}

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"

# Cache the loaded NeMo model — load is ~40s, so reuse it across calls.
_parakeet_model = None
_parakeet_model_name = None


def _transcribe_parakeet(audio_path: str, model_size: str,
                         device: str | None) -> TranscriptionResult:
    """Transcribe using NVIDIA NeMo Parakeet (English-only, very fast on GPU).

    Note: Parakeet only handles English audio. For multilingual content use the
    ``torch_whisper`` backend instead.
    """
    global _parakeet_model, _parakeet_model_name
    import nemo.collections.asr as nemo_asr

    # A Whisper size string ("medium", ...) is meaningless to NeMo — fall back
    # to the default Parakeet model. A full id ("nvidia/...") is used as-is.
    model_name = DEFAULT_PARAKEET_MODEL if model_size in _WHISPER_SIZES else model_size

    if _parakeet_model is None or _parakeet_model_name != model_name:
        _parakeet_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        _parakeet_model_name = model_name
        if device:
            _parakeet_model = _parakeet_model.to(device)

    out = _parakeet_model.transcribe([audio_path], timestamps=True)
    hyp = out[0]

    result = TranscriptionResult()
    result.language = "en"
    result.model_used = model_name

    ts = getattr(hyp, "timestamp", None) or {}
    segments = ts.get("segment") or []
    words_all = ts.get("word") or []

    last_end = 0.0
    if segments:
        for seg in segments:
            start = round(seg.get("start", 0.0), 3)
            end = round(seg.get("end", 0.0), 3)
            last_end = max(last_end, end)
            # Attach the word-level timestamps that fall inside this segment.
            words = [
                {
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "confidence": 0.0,
                }
                for w in words_all
                if seg.get("start", 0.0) <= w.get("start", 0.0) < seg.get("end", 0.0)
            ]
            result.segments.append(TranscriptionSegment(
                text=seg.get("segment", "").strip(),
                start_seconds=start,
                end_seconds=end,
                words=words,
                confidence=0.0,
            ))
    else:
        # No segment timestamps — emit a single segment with the full text.
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        result.segments.append(TranscriptionSegment(
            text=text.strip(), start_seconds=0.0, end_seconds=0.0, words=[],
        ))

    result.duration_seconds = last_end
    return result


if __name__ == "__main__":
    _main()
