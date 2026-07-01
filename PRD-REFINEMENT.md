# PRD Refinement: Local Long-Form Video Analysis with Audio

## Status: APPROVED
## Date: 2026-06-29

---

## 1. Hardware & Software Survey (Current State)

### GPU
- **NVIDIA GB10**, 130 GB VRAM — enterprise-class, extremely capable
- CUDA 13.0
- Currently running llama.cpp with Qwen3.6-35B-A3B (multimodal) on port 40285
- Currently running Ollama with Qwen3-VL 30B (19 GB GGUF)
- ComfyUI running (port 8188, partially accessible)

### Installed Python Packages
- `faster-whisper 1.2.1` ✓
- `gradio 6.15.2` ✓
- `fastmcp 3.3.1` ✓
- `torch 2.14.0 (cu130)` ✓
- `transformers 5.9.0` ✓
- `librosa 0.11.0` ✓
- `opencv-python 4.13.0` ✓
- `pillow 12.1.1` ✓
- `scipy 1.17.1` ✓
- `torchcodec 0.15.0` ✓

### Missing (need installation)
- `whisperx` — not installed
- `pyannote.audio` — not installed
- `pytesseract` / `Pillow-tesseract` — not installed
- `paddleocr` / `easyocr` — not installed
- `PySceneDetect` — not installed
- SQLite FTS5 — available via Python stdlib
- `chromadb` or `lancedb` — not installed (deferred to Phase 2)

### Available Local Servers
- **llama.cpp server** — Qwen3.6-35B-A3B-UD-Q4_K_XL, multimodal, OpenAI-compatible API (port 40285)
- **Ollama** — Qwen3-VL 30B (vision), Qwen3.5 35B (text), multiple other models (port 11434)

---

## 2. Resolved Open Questions

### 2.1 Transcription Backend: faster-whisper vs WhisperX

**Recommendation: faster-whisper as default, WhisperX as optional upgrade**

**Why:**
- faster-whisper is already installed and proven. It produces word-level timestamps, supports multiple languages, and is significantly faster than OpenAI Whisper.
- WhisperX adds forced alignment + built-in diarization, but requires installing WhisperX, pyannote, and additional dependencies. It's also more fragile to set up.
- For MVP, faster-whisper + separate pyannote diarization (installed later) is cleaner.
- WhisperX can be the "deep mode" upgrade path — it gives you alignment + diarization in one call, but only when the user opts in.

**Decision: APPROVED** — Default to faster-whisper. WhisperX is a Phase 2 optional upgrade.

### 2.2 Visual Model Choice

**Recommendation: llama.cpp Qwen3.6-35B-A3B as primary, Ollama Qwen3-VL 30B as fallback**

**Why:**
- With 130 GB VRAM, you can load the largest, most capable vision model available. The Qwen3.6-35B-A3B multimodal via llama.cpp is already running and supports multimodal input.
- This model is ~35B parameters with an active expert substructure (A3B = 3B active). It's capable of detailed visual captioning.
- Ollama's Qwen3-VL 30B (19 GB GGUF) is a good backup — slightly smaller but specifically fine-tuned for vision-language tasks.
- Neither requires downloading new models — both are already cached locally.

**Decision: APPROVED** — Use llama.cpp Qwen3.6-35B-A3B-UD-Q4_K_XL as the default visual model. Fall back to Ollama Qwen3-VL if needed.

### 2.3 Reasoning Model Server

**Recommendation: llama.cpp as default, Ollama as fallback**

**Why:**
- The llama.cpp server is already running with a large model (35B-A3B), supports tool use, and has an OpenAI-compatible API. This is ideal for Pass 1 and Pass 2 reasoning.
- The same model (Qwen3.6-35B-A3B) is multimodal, meaning it can do both visual captioning AND text reasoning from the same server.
- Ollama is a good fallback for when the llama.cpp server is busy or the model is too large for a given task (e.g., small Qwen3 0.6B for quick classification).

**Decision: APPROVED** — Use llama.cpp OpenAI-compatible API as the primary reasoning backend. Ollama as fallback for specialized tasks.

### 2.4 Diarization: Required vs Optional

**Recommendation: Optional in MVP, enabled by default in "deep" mode**

**Why:**
- Diarization adds significant complexity (pyannote.audio installation, GPU memory for speaker models, alignment logic).
- For MVP, the core pipeline (transcription + visual analysis + chunking + synthesis) works without it.
- For "deep" mode (podcast/interview use cases), diarization is critical. Install pyannote.audio as a Phase 2 dependency.
- The system should gracefully degrade: if diarization fails, fall back to a single-speaker transcript.

**Decision: APPROVED** — Optional. Faster-whisper transcript segments have timestamps but no speaker labels in MVP. Add pyannote.audio in Phase 2.

### 2.5 OCR Backend

**Recommendation: Optional in MVP, enable in Phase 2**

**Why:**
- OCR requires either Tesseract (system dependency), PaddleOCR (heavy Python dependency), or EasyOCR (vision model dependency).
- None are installed. Installing any adds significant complexity for MVP.
- OCR is specifically useful for screen recordings, tutorials, and presentations — not all video types benefit.
- The system can produce value without OCR in MVP.

**Decision: APPROVED** — OCR is optional in MVP. Enable in Phase 2 with PaddleOCR (most accurate for mixed text/UI). For MVP, visual captions from the vision model serve as a proxy for on-screen content.

### 2.6 Scene Detection

**Recommendation: Enable in MVP using FFmpeg scene filter**

**Why:**
- FFmpeg's built-in `select='gt(scene,threshold)'` filter requires no extra installation and works well for basic scene detection.
- PySceneDetect is more accurate but requires `pip install scenedetect[opencv]`.
- Scene boundaries are critical for intelligent chunking — they prevent splitting mid-scene.
- The FFmpeg approach is lightweight and produces reasonable results for MVP.

**Decision: APPROVED** — Use FFmpeg scene detection in MVP. PySceneDetect as Phase 2 upgrade.

### 2.7 Embeddings & Search

**Recommendation: SQLite FTS5 for MVP, local vector DB in Phase 2**

**Why:**
- SQLite FTS5 is built into Python's stdlib, requires zero extra dependencies, and supports full-text search with ranking.
- For MVP, keyword search over transcripts, OCR text, visual captions, and summaries is sufficient.
- Vector search (Chroma, LanceDB, Qdrant) adds complexity and is most valuable for semantic search across a growing library.
- The PRD already recommends this progression.

**Decision: APPROVED** — SQLite FTS5 for MVP. Defer vector search to Phase 2.

### 2.8 Max Video Duration

**Recommendation: No hard limit, but with streaming/chunked processing**

**Why:**
- You answered "no hard limit" — the architecture should support hours of content.
- Stage 0 must stream audio and sample frames in batches (never load entire video into memory).
- Pass 1 chunking must be independent and resumable (each chunk processes in isolation).
- Pass 2 must be able to handle 50+ chunk summaries without context overflow.
- With a 35B-parameter reasoning model, Pass 2 needs to read all chunk summaries. At ~500 tokens per chunk summary, 50 chunks = 25,000 tokens — well within the 262K context window of the llama.cpp server.

**Decision: APPROVED** — No hard limit. Implement streaming extraction and resumable chunking.

---

## 3. Architecture Recommendations

### 3.1 CLI-First Design

The PRD says "CLI first, both UI and MCP on top." This is the right approach.

**Recommended structure:**
```
video-analysis/
├── README.md
├── pyproject.toml              # Project metadata, dependencies
├── config.yaml                 # Default configuration
├── video_analysis/             # Python package
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Click/typer CLI commands
│   ├── pipeline.py             # Core pipeline orchestrator
│   ├── stage0/                 # Stage 0: Extraction + Alignment
│   │   ├── __init__.py
│   │   ├── metadata.py         # FFprobe extraction
│   │   ├── audio.py            # Audio extraction + transcription
│   │   ├── diarization.py      # Speaker diarization (optional)
│   │   ├── scenes.py           # Scene detection
│   │   ├── frames.py           # Frame sampling
│   │   ├── ocr.py              # OCR (optional)
│   │   ├── vision.py           # Visual captioning
│   │   ├── audio_events.py     # Audio event detection
│   │   └── align.py            # Timeline alignment
│   ├── pass1/                  # Pass 1: Segment Analysis
│   │   ├── __init__.py
│   │   ├── chunker.py          # Chunk building logic
│   │   ├── analyzer.py         # Chunk-level reasoning
│   │   └── prompts.py          # Pass 1 prompt templates
│   ├── pass2/                  # Pass 2: Global Synthesis
│   │   ├── __init__.py
│   │   ├── synthesizer.py      # Global synthesis
│   │   └── prompts.py          # Pass 2 prompt templates
│   ├── models/                 # Data models (Pydantic)
│   │   ├── __init__.py
│   │   ├── video.py
│   │   ├── chunk.py
│   │   ├── key_moment.py
│   │   ├── chapter.py
│   │   └── analysis.py
│   ├── storage/                # Database layer
│   │   ├── __init__.py
│   │   ├── database.py         # SQLite connection
│   │   ├── schema.py           # SQL schema
│   │   └── search.py           # FTS5 search
│   ├── reasoning/              # Model inference
│   │   ├── __init__.py
│   │   ├── server.py           # Base reasoning client
│   │   ├── llama_cpp.py        # llama.cpp client
│   │   ├── ollama.py           # Ollama client
│   │   └── vision_client.py    # Vision model client
│   ├── export/                 # Export formats
│   │   ├── __init__.py
│   │   ├── json_export.py
│   │   ├── markdown_export.py
│   │   ├── srt_export.py
│   │   └── csv_export.py
│   ├── mcp/                    # MCP tool interface
│   │   ├── __init__.py
│   │   └── tools.py            # FastMCP tool definitions
│   └── ui/                     # Gradio UI
│       ├── __init__.py
│       └── app.py              # Gradio application
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_chunker.py
│   └── test_search.py
├── outputs/                    # Generated analysis outputs (gitignored)
├── .gitignore
└── requirements.txt            # Or use pyproject.toml dependencies
```

### 3.2 Model Routing Strategy

```
┌─────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 0:                                               │
│    Audio extraction  → FFmpeg (binary)                  │
│    Transcription     → faster-whisper (Python)          │
│    Scene detection   → FFmpeg scene filter (binary)     │
│    Frame sampling    → FFmpeg (binary)                  │
│    Visual captioning → llama.cpp multimodal API         │
│    OCR               → [deferred to Phase 2]            │
│    Audio events      → librosa (Python, rule-based)     │
│                                                         │
│  Pass 1:                                                │
│    Chunk analysis    → llama.cpp text API               │
│                                                         │
│  Pass 2:                                                │
│    Global synthesis  → llama.cpp text API               │
│                                                         │
│  Search:                                                │
│    FTS5              → SQLite FTS5                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Chunking Strategy Refinement

The PRD's chunking rules are good. Here's a refined implementation plan:

```
1. Load transcript segments from faster-whisper
2. Load scene boundaries from FFmpeg
3. Load speaker change points (when diarization is enabled)
4. Build candidate split points at:
   - Each scene boundary
   - Each speaker change
   - Every N words of transcript (target chunk length)
5. Score each candidate: prefer scene boundaries > speaker changes > fixed intervals
6. Create chunks at best split points within target length range
7. Add overlap region (last M seconds of previous chunk, first M seconds of next)
8. Validate: no chunk exceeds max_tokens limit for reasoning model
```

### 3.4 Resumable Processing

The pipeline must be resumable. Each stage writes its outputs and updates a status table:

```sql
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,
  video_id TEXT,
  status TEXT DEFAULT 'pending',  -- pending, running, completed, failed, paused
  current_stage TEXT,              -- stage0, pass1, pass2
  current_step TEXT,               -- audio_extraction, transcription, ..., chunking, ...
  started_at TEXT,
  completed_at TEXT,
  progress_percent REAL DEFAULT 0,
  error_message TEXT,
  config_json TEXT
);

CREATE TABLE stage0_artifacts (
  id TEXT PRIMARY KEY,
  video_id TEXT,
  artifact_type TEXT,  -- audio, transcript, scenes, frames, vision_captions, audio_events
  status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
  output_path TEXT,
  error_message TEXT,
  created_at TEXT
);
```

### 3.5 Configuration

The YAML config from the PRD is comprehensive. Two refinements:

1. **Add model server URLs** (configurable, defaults to localhost):
   ```yaml
   models:
     reasoning_server:
       type: llama_cpp          # llama_cpp, ollama, openai
       url: http://127.0.0.1:40285
       model: Qwen3.6-35B-A3B-UD-Q4_K_XL
     vision_server:
       type: llama_cpp
       url: http://127.0.0.1:40285
       model: Qwen3.6-35B-A3B-UD-Q4_K_XL
     embedding_server:
       type: none               # none, llama_cpp, ollama
   ```

2. **Add hardware detection**: On first run, auto-detect GPU, available VRAM, and installed tools. Generate a recommended config.

---

## 4. MVP Scope Tightening

### What to Build First (Phase 1 MVP)

| Component | Decision |
|-----------|----------|
| CLI library | ✅ Core of everything |
| FFmpeg extraction | ✅ Already available |
| Transcription | ✅ faster-whisper |
| Scene detection | ✅ FFmpeg scene filter |
| Frame sampling | ✅ FFmpeg |
| Visual captioning | ✅ llama.cpp multimodal |
| Chunking | ✅ Hybrid strategy |
| Pass 1 analysis | ✅ llama.cpp text |
| Pass 2 synthesis | ✅ llama.cpp text |
| SQLite storage | ✅ Built-in |
| SQLite FTS5 search | ✅ Built-in |
| JSON export | ✅ |
| Markdown export | ✅ |
| SRT export | ✅ From transcript |
| Gradio UI | ✅ Thin wrapper around CLI |
| MCP tools | ✅ FastMCP wrapper |
| Config system | ✅ YAML |
| Resumable processing | ✅ Status tracking |

### What to Defer to Phase 2

| Component | Reason |
|-----------|--------|
| Speaker diarization (pyannote) | Complex install, GPU memory heavy |
| OCR (PaddleOCR) | Heavy install, optional benefit |
| Audio event detection (librosa rules) | Nice-to-have, not core |
| Vector search (Chroma/LanceDB) | Overkill for MVP |
| Forensic mode | Requires OCR + dense sampling |
| Prompt adherence scoring | Requires Phase 4 |
| Video library view | Phase 5 |
| Batch processing | Phase 4 |
| EDL/XML export | Niche use case |

---

## 5. Risk Assessment

### High Risk
1. **Context overflow in Pass 2**: If a 2-hour video produces 80+ chunks, Pass 2 must summarize 80 summaries. At 500 tokens each = 40,000 tokens. This is fine with the 262K context window, but the synthesis prompt needs to be careful about token budget.
2. **llama.cpp multimodal performance**: The Qwen3.6-35B-A3B is running in text mode right now. Confirming image input works for visual captioning is a critical early test.

### Medium Risk
3. **FFmpeg scene detection accuracy**: The scene filter threshold tuning matters. Need a good default (0.3) and allow override.
4. **faster-whisper word timestamps**: Word-level timestamps are more accurate but require the model to support them (Systran models do).

### Low Risk
5. **SQLite FTS5 search**: Well-tested, built into Python.
6. **Gradio UI**: Straightforward once CLI is stable.
7. **MCP tools**: FastMCP is already installed and working.

---

## 6. Prompt Templates (Draft)

### Pass 1: Chunk Analysis Prompt

```
You are analyzing a segment of a video. Here is the aligned multimodal data for this segment:

TRANSCRIPT:
{transcript}

SPEAKERS:
{speaker_labels}

VISUAL DESCRIPTIONS:
{visual_captions}

OCR TEXT (if any):
{ocr_text}

AUDIO EVENTS:
{audio_events}

TASK:
Analyze this segment and return structured data:
1. Summary: 2-3 sentence summary of what happens in this segment
2. Key moments: List important timestamps with descriptions
3. Tags: List relevant topic/action tags
4. Quotes: Notable quotes with speaker attribution
5. Issues: Errors, warnings, or problems mentioned/shown
6. Detected actions: Physical actions or UI interactions observed
```

### Pass 2: Global Synthesis Prompt

```
You are synthesizing analysis results from {num_chunks} video segments into a coherent full-video report.

SEGMENT SUMMARIES:
{chunk_summaries}

KEY MOMENTS (all):
{all_key_moments}

TAGS (all):
{all_tags}

TASK:
1. Executive summary: 1 paragraph
2. Detailed summary: 3-5 paragraphs by theme
3. Chapters: 5-15 chapters with start time, title, and 1-sentence summary
4. Merged key moments: Deduplicate and rank by importance (1-5)
5. Speaker summary: Topics discussed by each speaker
6. Visual progression: How visuals change throughout the video
7. Action items: What the viewer should do or learn
8. Search tags: Comprehensive keyword list
```

---

## 7. Next Steps

After approving this refinement:

1. **Project scaffold** — `pyproject.toml`, directory structure, `.gitignore`
2. **Database schema** — SQLite tables from the PRD
3. **Core pipeline** — CLI entry point, pipeline orchestrator, config loading
4. **Stage 0** — FFmpeg tools, faster-whisper transcription, scene detection, frame sampling
5. **Vision inference** — llama.cpp multimodal client for visual captioning
6. **Pass 1** — Chunker + analyzer
7. **Pass 2** — Synthesizer
8. **Storage** — SQLite + FTS5 search
9. **Exports** — JSON, Markdown, SRT
10. **Gradio UI** — Thin wrapper
11. **MCP tools** — FastMCP wrapper
