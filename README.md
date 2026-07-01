# Local Long-Form Video Analysis Engine

A fully local pipeline that ingests long-form videos and produces structured,
timestamped outputs: transcripts, speaker labels, chapters, summaries, key
moments, searchable metadata, and exportable reports.

## Performance

Transcription backends, measured on the same 94s clip (NVIDIA GB10 / ARM64 / CUDA 13.0):

| Backend | Time (94s clip) | Realtime factor | Languages |
|---------|-----------------|-----------------|-----------|
| `parakeet` (NeMo, GPU) | **~1.3 s** | **~70x** | English only |
| `torch_whisper` (openai-whisper, GPU) | ~57 s (medium) / ~59 s (large-v3) | ~1.6x | 99 languages |
| `faster_whisper` (CTranslate2, CPU) | ~3 min 4 s | ~0.5x | 99 languages |

`parakeet` is ~140x faster than the old CPU path; `torch_whisper` is ~3x faster and
multilingual. faster-whisper falls back to CPU on aarch64 (no CUDA wheel for ctranslate2);
when that happens and a CUDA torch build is present, it auto-upgrades to `torch_whisper`.
See `config.yaml` (`transcription_backend`) to choose. English content → `parakeet`;
mixed/non-English → `torch_whisper`.

Full-pipeline breakdown, measured on a real **13.5-minute** video (809 s,
`parakeet` backend), quick vs deep mode:

| Stage | Quick | Deep |
|-------|-------|------|
| Transcription (parakeet, GPU) | ~27 s | ~29 s |
| Visual captioning (Gemma4 E4B) | ~14 min (162 frames) | ~35 min (405 frames) |
| Pass 1 chunk reasoning | ~53 s (3 chunks) | ~2.7 min (9 chunks) |
| Pass 2 global synthesis | ~34 s | ~85 s |
| **Total** | **~15.5 min** | **~40 min** |

Vision and reasoning use the unified `localhost:9090` model manager. Transcription
went from the #1 bottleneck (~60% of wall time on CPU) to ~3% — the old CPU path
would have spent ~26 min just transcribing this clip. Deep mode's extra cost is
almost entirely the 2.5x frame count for captioning; in return it produces
finer-grained chapters (90 s vs 5 min chunks) and catches themes the coarse pass
misses.

Remaining bottleneck:
- **Visual captioning** — ~5 s per frame; frame count scales with mode and video length

### GPU ASR setup

`parakeet` and `torch_whisper` need a Python interpreter with the GPU ASR stack
(torch + openai-whisper, and NeMo for parakeet). If the main venv lacks those heavy
CUDA deps, point `audio.asr_python` in `config.yaml` (or `VIDEO_ANALYSIS_ASR_PYTHON`)
at an env that has them — transcription is then run there via subprocess.

## Architecture

```
Video Input
  ├── Stage 0: Extraction + Alignment
  │     ├── Audio extraction (FFmpeg)
  │     ├── Transcription (parakeet / whisper, GPU)
  │     ├── Scene detection (FFmpeg)
  │     ├── Frame sampling (FFmpeg)
  │     ├── Visual captioning (llama.cpp multimodal)
  │     └── Timeline alignment
  │
  ├── Pass 1: Segment Analysis
  │     ├── Chunk building (hybrid strategy)
  │     └── Chunk-level reasoning (llama.cpp text)
  │
  └── Pass 2: Global Synthesis
        ├── Full summary
        ├── Chapters
        ├── Merged key moments
        └── Search tags
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run analysis via CLI (either form works)
video-analysis analyze /path/to/video.mp4 --mode quick
python -m video_analysis.cli analyze /path/to/video.mp4 --mode quick

# Show system info and connected models
video-analysis info

# Start the Gradio UI
video-analysis ui

# Start the MCP server
video-analysis mcp
```

> The GPU transcription backends (`parakeet`, `torch_whisper`) need a Python
> interpreter with the ASR stack installed. See [GPU ASR setup](#gpu-asr-setup).

## Modes

Wall time is now dominated by visual captioning (~5 s/frame), so it scales with
frame count rather than transcription:

- **quick**: Fast preview — 0.2 fps frame sampling, 5-min chunks, no diarization
- **deep**: Standard quality — balanced frame rate, scene-based chunking, audio events
- **forensic**: Detailed review — dense sampling (2 fps), small chunks, 15s overlap, diarization + OCR (much slower)

*Audio events* (deep/forensic) are **structural acoustic analysis** — silence and
music/noise regions detected with librosa. This is not semantic sound-event
classification; it does not label sounds like applause or laughter.

## Configuration

Edit `config.yaml` to customize model servers, chunking parameters, and
enabled features.

### Transcription Backend

```yaml
audio:
    transcription_backend: parakeet   # parakeet | torch_whisper | faster_whisper
    whisper_model: medium             # whisper size, or a NeMo model id for parakeet
    asr_python: /path/to/gpu/venv/bin/python   # interpreter with the ASR stack
```

- **`parakeet`** — NVIDIA NeMo Parakeet. English-only, ~70x realtime on GPU. Best for English content.
- **`torch_whisper`** — openai-whisper on PyTorch. Multilingual (99 languages), GPU on aarch64. Use for non-English/mixed audio; `whisper_model: large-v3` is ~free on GPU.
- **`faster_whisper`** — CTranslate2. Multilingual; CPU-only on aarch64 (no CUDA wheel). Auto-upgrades to `torch_whisper` when it can't reach a GPU.

### Model Server Setup

The pipeline expects a llama.cpp model manager on `localhost:9090` serving:
- **Vision**: `Huihui-gemma-4-E4B-it-abliterated-Q4_K_M` (multimodal, image+text)
- **Reasoning**: `Huihui-gemma-4-E4B-it-abliterated-Q4_K_M` (text)

Both can point to the same model. Configure in `config.yaml`:

```yaml
reasoning_server:
    url: http://localhost:9090
    model: Huihui-gemma-4-E4B-it-abliterated-Q4_K_M

vision_server:
    url: http://localhost:9090
    model: Huihui-gemma-4-E4B-it-abliterated-Q4_K_M
```

## Privacy

All processing runs locally. No videos, audio, frames, transcripts, or
embeddings are sent to external services.

## Project Structure

```
video_analysis/
├── cli.py              # CLI entry point
├── pipeline.py         # Core orchestrator
├── stage0/             # Extraction + alignment
├── pass1/              # Segment analysis
├── pass2/              # Global synthesis
├── models/             # Data models (Pydantic)
├── storage/            # SQLite + FTS5 search
├── reasoning/          # llama.cpp / Ollama clients
├── export/             # JSON, Markdown, SRT, CSV
├── mcp/                # FastMCP tools
├── ui/                 # Gradio web UI
```

See `PRD-REFINEMENT.md` for architecture decisions and resolved open questions.
