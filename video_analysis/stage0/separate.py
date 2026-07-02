"""Vocal source separation (MelBandRoFormer via ComfyUI).

Splits speech (vocals) from music/SFX (instruments) so downstream speech stages
run on clean vocals. This markedly improves diarization — on mixed film audio,
music/SFX bleed causes pyannote to over-segment a 2-person scene into many
spurious speakers; on isolated vocals the same scene resolves to a coherent set
of speakers. Transcription also benefits (fewer music-induced ASR errors).

Runs on a ComfyUI backend (default http://localhost:8188) that has the
ComfyUI-MelBandRoFormer node + model. ComfyUI rejects large uploads (HTTP 413),
so long audio is separated in windows (FLAC-encoded to stay under the limit) and
the vocal windows are concatenated back into one full-length WAV.

Best-effort: any failure (backend down, node missing) is surfaced to the caller,
which should fall back to using the original mixed audio.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

COMFYUI_URL_ENV = "COMFYUI_URL"
DEFAULT_COMFYUI_URL = "http://localhost:8188"
DEFAULT_MODEL = "MelBandRoformer_fp16.safetensors"
_DEFAULT_WORKFLOW = str(Path.home() / "ComfyUI" / "user" / "default" / "workflows"
                        / "hermes_known_good"
                        / "melband_roformer_vocals_instruments_preview.json")
WORKFLOW_ENV = "AUDIOSEP_WORKFLOW_PATH"

# ComfyUI's default upload cap rejects a full 2h FLAC (~126 MB); a 20-min FLAC
# window is ~21 MB and uploads fine. Separation is independent per window (no
# cross-window state), so windowing is lossless here.
_WINDOW_SECONDS = 1200.0


def _comfy_url() -> str:
    return os.environ.get(COMFYUI_URL_ENV, DEFAULT_COMFYUI_URL)


def is_available() -> bool:
    """True if the ComfyUI backend is reachable."""
    try:
        with urllib.request.urlopen(f"{_comfy_url()}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def separate_vocals(audio_path: str, out_dir: str,
                    model: str = DEFAULT_MODEL) -> str:
    """Separate speech from music/SFX; return a path to a full-length vocals WAV.

    Windows the audio, separates each window on ComfyUI, and concatenates the
    vocal stems back to one 16 kHz mono WAV (the format the ASR/diarization
    stages consume). Raises on failure so the caller can fall back to the mix.
    """
    os.makedirs(out_dir, exist_ok=True)
    with wave.open(audio_path, "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
    total_duration = nframes / float(framerate or 1)

    vocal_windows: list[str] = []
    win_idx = 0
    seg_start = 0.0
    while seg_start < total_duration:
        dur = min(_WINDOW_SECONDS, total_duration - seg_start)
        # Encode this window to FLAC (small enough to upload).
        win_flac = os.path.join(out_dir, f"win_{win_idx:03d}.flac")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(seg_start),
             "-i", audio_path, "-t", str(dur), "-ac", "1", "-ar", "44100",
             "-c:a", "flac", win_flac],
            check=True, capture_output=True)
        stems = _separate_one(win_flac, out_dir, model, win_idx)
        vocals = stems.get("vocals")
        if not vocals:
            raise RuntimeError(f"separation window {win_idx} produced no vocals")
        vocal_windows.append(vocals)
        try:
            os.remove(win_flac)
        except OSError:
            pass
        win_idx += 1
        seg_start += _WINDOW_SECONDS

    # Concatenate vocal windows into one 16 kHz mono WAV.
    out_wav = os.path.join(out_dir, "vocals.wav")
    _concat_to_wav(vocal_windows, out_wav)
    for p in vocal_windows:
        try:
            os.remove(p)
        except OSError:
            pass
    return out_wav


def _separate_one(audio_file: str, out_dir: str, model: str, idx: int) -> dict:
    """Separate one window on ComfyUI; return {stem: path}."""
    comfy = _comfy_url()
    with open(audio_file, "rb") as f:
        up = _post_multipart(f"{comfy}/upload/image",
                             {"type": "input", "overwrite": "true"},
                             {"image": (os.path.basename(audio_file), f.read())})
    name = up.get("name", os.path.basename(audio_file))
    prompt = _build_prompt(name, model)
    pid = _post_json(f"{comfy}/prompt",
                     {"prompt": prompt, "client_id": uuid.uuid4().hex})["prompt_id"]

    deadline = time.time() + 500
    hist = None
    while time.time() < deadline:
        with urllib.request.urlopen(f"{comfy}/history/{pid}", timeout=30) as r:
            h = json.loads(r.read())
        if pid in h:
            hist = h[pid]
            break
        time.sleep(2)
    if hist is None:
        raise RuntimeError(f"separation window {idx} timed out")

    nodes = hist.get("prompt", [None, None, {}])[2]
    sampler_ids = {n for n, v in nodes.items()
                   if v.get("class_type") == "MelBandRoFormerSampler"}
    slot = {0: "vocals", 1: "instruments"}
    stem_by_node = {}
    for n, v in nodes.items():
        ai = v.get("inputs", {}).get("audio")
        if isinstance(ai, list) and len(ai) == 2 and ai[0] in sampler_ids:
            stem_by_node[n] = slot.get(ai[1])

    result = {}
    for nid, out in hist.get("outputs", {}).items():
        stem = stem_by_node.get(nid)
        if not stem:
            continue
        for key in ("audio", "images"):
            for e in out.get(key, []):
                q = urllib.parse.urlencode({"filename": e.get("filename", ""),
                                            "subfolder": e.get("subfolder", ""),
                                            "type": e.get("type", "output")})
                with urllib.request.urlopen(f"{comfy}/view?{q}", timeout=120) as r:
                    data = r.read()
                p = os.path.join(out_dir,
                                 f"w{idx}_{stem}{Path(e['filename']).suffix or '.wav'}")
                with open(p, "wb") as fp:
                    fp.write(data)
                result[stem] = p
    return result


def _concat_to_wav(paths: list[str], out_wav: str) -> None:
    """Concatenate audio files into one 16 kHz mono WAV via ffmpeg concat."""
    list_file = out_wav + ".txt"
    with open(list_file, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-ac", "1", "-ar", "16000", out_wav],
        check=True, capture_output=True)
    try:
        os.remove(list_file)
    except OSError:
        pass


def _build_prompt(audio_filename: str, model: str) -> dict:
    """Convert the saved ComfyUI graph workflow to an API prompt."""
    wf_path = os.environ.get(WORKFLOW_ENV, _DEFAULT_WORKFLOW)
    wf = json.load(open(wf_path))
    nodes = {str(n["id"]): {"class_type": n["type"], "inputs": {},
                            "wv": n.get("widgets_values", [])}
             for n in wf["nodes"]}
    links = {l[0]: (str(l[1]), l[2]) for l in wf.get("links", [])}
    for nid, node in nodes.items():
        orig = next((n for n in wf["nodes"] if str(n["id"]) == nid), None)
        if not orig:
            continue
        for inp in orig.get("inputs", []):
            lid = inp.get("link")
            if lid is not None and links.get(lid):
                node["inputs"][inp["name"]] = [links[lid][0], links[lid][1]]
        widget_inputs = [i["name"] for i in orig.get("inputs", [])
                         if i.get("link") is None]
        for i, val in enumerate(node["wv"] or []):
            if (i < len(widget_inputs) and val is not None
                    and widget_inputs[i] not in node["inputs"]):
                node["inputs"][widget_inputs[i]] = val
    for nid, node in nodes.items():
        if node["class_type"] == "LoadAudio":
            node["inputs"] = {"audio": audio_filename}
        elif "ModelLoader" in node["class_type"]:
            node["inputs"]["model_name"] = model
    return {nid: {"class_type": v["class_type"], "inputs": v["inputs"]}
            for nid, v in nodes.items()}


def _post_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _post_multipart(url: str, fields: dict, files: dict) -> dict:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, (fn, content) in files.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; '
                   f'name="{k}"; filename="{fn}"\r\n\r\n'.encode())
        body.write(content)
        body.write(b"\r\n")
    for k, v in fields.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; '
                   f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())
