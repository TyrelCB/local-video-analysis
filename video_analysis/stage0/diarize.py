"""Speaker diarization — assigns speaker labels to transcript segments.

Uses pyannote.audio (speaker-diarization-3.1). pyannote lives in the GPU ASR
environment (torch + the model), not the lean main venv, so diarization is
delegated to that interpreter via subprocess — the same pattern as ``audio.py``
and ``semantic_events.py``. Set ``audio.asr_python`` (or
``VIDEO_ANALYSIS_ASR_PYTHON``).

Long audio is windowed: diarizing a full 2h file in one pass exhausts memory and
crashes (the embedding + agglomerative clustering scales poorly with duration).
We diarize in windows and offset the turn times back to absolute. Speaker labels
are window-local (pyannote's SPEAKER_xx don't correspond across independent
windows), so each window's speakers are namespaced (``w0_SPEAKER_00``); a scene's
dialogue almost always falls within one window, which is what downstream
attribution needs. A whole-video global speaker identity would require
cross-window embedding matching — deliberately out of scope here.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ASR_PYTHON_ENV = "VIDEO_ANALYSIS_ASR_PYTHON"
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

# How much audio to diarize at once. 20 min is comfortably within pyannote's
# memory/clustering limits on this box (a 5-min slice runs in ~14s; the full 2h
# file crashes) while keeping most scenes intact within a single window.
_WINDOW_SECONDS = 1200.0


@dataclass
class SpeakerTurn:
    """A span [start, end) attributed to one speaker label."""
    start_seconds: float
    end_seconds: float
    speaker: str


def diarize(audio_path: str, model: str = DEFAULT_MODEL,
            asr_python: str | None = None) -> list[SpeakerTurn]:
    """Diarize an audio file into speaker turns.

    Delegates to the ASR interpreter (which has pyannote) when configured and
    different from the current one; otherwise runs in-process.
    """
    asr_python = asr_python or os.environ.get(ASR_PYTHON_ENV)
    if asr_python and os.path.realpath(asr_python) != os.path.realpath(sys.executable):
        return _diarize_via_subprocess(asr_python, audio_path, model)
    return _diarize_inprocess(audio_path, model)


def assign_speakers(segments: list, turns: list[SpeakerTurn]) -> None:
    """Set each transcript segment's ``speaker`` to the best-overlapping turn.

    Mutates ``segments`` in place (each must have ``start_seconds`` /
    ``end_seconds`` and a settable ``speaker`` attribute). A segment is assigned
    the speaker whose turn overlaps it most; ties/gaps fall back to the nearest
    turn. No-op when there are no turns.
    """
    if not turns:
        return
    turns = sorted(turns, key=lambda t: t.start_seconds)
    for seg in segments:
        s = getattr(seg, "start_seconds", 0.0)
        e = getattr(seg, "end_seconds", s)
        best_spk, best_overlap, nearest = None, 0.0, None
        nearest_dist = None
        for t in turns:
            overlap = min(e, t.end_seconds) - max(s, t.start_seconds)
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, t.speaker
            dist = min(abs(s - t.start_seconds), abs(s - t.end_seconds))
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist, nearest = dist, t.speaker
        try:
            seg.speaker = best_spk or nearest or ""
        except AttributeError:
            pass


def _diarize_inprocess(audio_path: str, model: str) -> list[SpeakerTurn]:
    """Run windowed pyannote diarization in the current interpreter."""
    import tempfile
    import wave

    import torch
    from pyannote.audio import Pipeline

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    pipe = Pipeline.from_pretrained(model, token=token)
    if pipe is None:
        raise RuntimeError(
            f"Diarization pipeline '{model}' could not load — the gated model "
            "may not be accepted for this HF token (visit the model page and "
            "accept the user conditions).")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))

    with wave.open(audio_path, "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
    total_duration = nframes / float(framerate or 1)

    turns: list[SpeakerTurn] = []
    win = 0
    seg_start = 0.0
    win_frames = int(_WINDOW_SECONDS * framerate)
    while seg_start < total_duration:
        # Slice this window to a temp WAV so we never hold the whole file.
        start_frame = int(seg_start * framerate)
        with wave.open(audio_path, "rb") as wf:
            wf.setpos(min(start_frame, nframes))
            chunk = wf.readframes(min(win_frames, nframes - start_frame))
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="diar_win_")
        os.close(fd)
        try:
            with wave.open(tmp, "wb") as out_wf:
                out_wf.setnchannels(nchannels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)
                out_wf.writeframes(chunk)
            out = pipe(tmp)
            ann = getattr(out, "speaker_diarization", out)
            for turn, _, spk in ann.itertracks(yield_label=True):
                turns.append(SpeakerTurn(
                    start_seconds=round(seg_start + turn.start, 3),
                    end_seconds=round(seg_start + turn.end, 3),
                    speaker=f"w{win}_{spk}",
                ))
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        win += 1
        seg_start += _WINDOW_SECONDS

    return turns


def _diarize_via_subprocess(asr_python: str, audio_path: str,
                            model: str) -> list[SpeakerTurn]:
    """Run diarization under the ASR interpreter, parse back JSON."""
    cmd = [asr_python, "-m", "video_analysis.stage0.diarize",
           "--audio", audio_path, "--model", model]
    logger.info("Delegating diarization to %s", asr_python)
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Diarization subprocess failed:\n{proc.stderr[-2000:]}")
    payload = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(
            f"Diarization subprocess produced no JSON:\n{proc.stdout[-2000:]}")
    return [SpeakerTurn(**t) for t in payload]


def _main() -> None:
    """Worker entrypoint: diarize and print turns as a JSON list (last line)."""
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description="Diarization worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    turns = _diarize_inprocess(args.audio, args.model)
    print(json.dumps([asdict(t) for t in turns]))


if __name__ == "__main__":
    _main()
