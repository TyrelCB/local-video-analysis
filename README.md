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
`parakeet` backend). "Before" is the original fully-serial path; "now" adds
pre-inference frame dedup and 8-way concurrency for both captioning and reasoning
(shared Gemma-4 E4B preset, `--parallel 8`). See [DECISIONS.md](DECISIONS.md) for
why each knob is set the way it is.

**Deep mode** (0.5 fps → 405 frames, 90 s chunks):

| Stage | Before (serial) | Now (dedup + 8-way) |
|-------|-----------------|---------------------|
| Transcription (parakeet, GPU) | ~29 s | ~29 s |
| Visual captioning (Gemma-4 E4B) | ~35 min (405 frames) | **~48 s** (83 unique, ~80% deduped) |
| Pass 1 chunk reasoning | ~2.7 min (9 chunks, serial) | **~58 s** (up to 8 concurrent) |
| Pass 2 global synthesis | ~85 s | ~48 s |
| **Total** | **~40 min** | **~3.5 min** |

**Quick mode** (0.2 fps → 162 frames, 5 min chunks) lands at **~3 min** total, with
captioning ~62 s (59 unique, ~64% deduped).

The two wins compound: dedup removes most frames *before* inference (biggest on
screen recordings / static footage), then concurrency runs the survivors in
parallel. Captioning went from ~86% of wall time to a minority stage; on the
serial path this clip would have spent over half an hour just describing frames.
Transcription, once the #1 bottleneck on CPU (~60% of wall time), is now ~15%.

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

Modes trade detail for wall time mainly via frame-sampling rate and chunk size.
With dedup + concurrency (below), captioning no longer dominates, so deep mode's
2.5x frame count costs little extra:

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

Pointing both at the **same** model id makes the router load one E4B instance and
serve both stages from it — no duplicate weights. Start that preset with as many
parallel slots as you want concurrency for (`parallel = 8` in the llama.cpp
`presets.ini`). Rationale in [DECISIONS.md](DECISIONS.md).

### Throughput tuning

```yaml
video:
    caption_dedup: true          # skip near-duplicate frames before captioning
    caption_max_concurrent: 8    # concurrent vision requests (match server slots)
analysis:
    reasoning_max_concurrent: 8  # concurrent Pass 1 chunk-reasoning requests
```

- **`caption_dedup`** — perceptual-hash (dHash + brightness) dedup *before*
  inference; near-identical frames reuse the previous caption at zero cost.
  Lower `DEDUP_HAMMING_THRESHOLD` in `stage0/vision.py` for denser coverage.
- **`caption_max_concurrent` / `reasoning_max_concurrent`** — set to the model
  server's `--parallel` slot count. Higher values just queue server-side; the
  throughput ceiling is the GPU, so expect ~3x from 8 slots, not 8x.

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
