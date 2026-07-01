# Engineering Decisions

A running log of non-obvious design decisions — what was chosen, what was
rejected, and why. Read this before changing performance-sensitive paths;
the reasoning behind a knob is rarely visible from the code alone.

Format: one entry per decision, newest first. Each records the context, the
decision, alternatives considered, and consequences.

---

## 2026-07-01 — GPU transcription, semantic audio, and net performance

Context: transcription was the pipeline's #1 bottleneck — ~60% of wall time —
because `faster-whisper` (CTranslate2) has no aarch64 CUDA wheel and silently
ran on CPU on the GB10 (a 94 s clip took ~3 min). The GPU was fully capable;
only that one backend couldn't reach it.

### D7 — Pluggable transcription backend, Parakeet default

**Decision:** a `audio.transcription_backend` knob with three options:
`parakeet` (NVIDIA NeMo, English-only, ~70x realtime on GPU), `torch_whisper`
(openai-whisper on PyTorch, 99 languages, GPU on aarch64), and the original
`faster_whisper`. Default is `parakeet` (content here is mostly English).

**Rejected — WhisperX:** it rides CTranslate2 for its core ASR, the *same*
thing that has no aarch64 CUDA wheel, so it would be CPU-bound here. Its
batching/diarization run on torch, but the transcription core defeats the point.

**Rejected — keeping faster_whisper on CPU:** the reason it was on CPU (missing
ct2 wheel) does not apply to PyTorch — a CUDA torch wheel for aarch64/Blackwell
exists and is installed. So torch-based ASR was always viable; the project just
picked the one backend that wasn't.

**Auto-upgrade:** if `faster_whisper` is requested but can't see a CUDA device
while a CUDA torch build is present, it transparently switches to
`torch_whisper` — the GPU shouldn't sit idle because of a wheel gap.

**Measured (94 s clip):** parakeet ~1.3 s, torch_whisper ~57 s (medium) / ~59 s
(large-v3), faster_whisper-CPU ~184 s. On a real 13.5-min video, transcription
fell from ~26 min (CPU) to ~27 s — from ~60% of wall time to ~3%.

### D8 — Delegate GPU ASR/audio to a separate interpreter via subprocess

**Decision:** the heavy CUDA stack (torch, NeMo, transformers) lives in a
dedicated env (`~/comfyui-env`, Python 3.12), not the lean main venv (3.13).
`transcribe_audio` and `classify_sound_events` shell out to `audio.asr_python`
(or `VIDEO_ANALYSIS_ASR_PYTHON`) via `python -m ...`, exchanging JSON on stdout.

**Rejected — installing torch+CUDA into the main venv:** ~4 GB of CUDA wheels,
and it couples a 3.13 project to a torch build proven only on 3.12. The
subprocess boundary keeps the main venv importable without any GPU deps (backends
lazy-import inside the worker), at the cost of one process spawn per call.

### D9 — Semantic audio events via AST/AudioSet, not TensorFlow YAMNet

**Decision:** `audio.audio_events_backend: librosa | yamnet`. The `yamnet` path
names sounds (Applause, Laughter, Music, Keyboard, ...) using the MIT Audio
Spectrogram Transformer fine-tuned on AudioSet, run through `transformers` on
PyTorch. Same 527-class AudioSet ontology as YAMNet; different runtime.

**Rejected — actual TF-Hub YAMNet:** classic YAMNet is TensorFlow, which has no
usable aarch64 + CUDA 13 build; TF isn't installed and is painful here. AST gives
the same labels on the torch stack already present. **Rejected — librosa-only:**
it's structural (silence/music), never *names* a sound; kept as the default and
the fallback, not the semantic answer.

**Tuning:** ambient speech labels (Speech, Conversation, "Inside, small room")
are dropped — they dominate talking-head audio and carry no signal. Score
threshold is 0.4: below that the model emits low-confidence runner-ups (spurious
"Sliding door" at 0.09) once speech is filtered. On a speech-only video the
honest result is *few* events, and that's fine — no invented "dog bark." Falls
back to librosa on error, but an empty (successful) semantic result is kept as-is.

**Speed:** ~0.3 s per 10 s of audio on the GB10 GPU (batched windows); model
load ~17 s one-time. Runs via D8 delegation.

### Net performance gain this session

On the 13.5-min token-burn screen recording (deep mode), combining D1–D9:

- **Transcription:** ~26 min (CPU) → ~27 s (Parakeet GPU) — ~60x on that stage.
- **Captioning:** dedup (~80% frames skipped on this static content) + 8-way
  concurrency; the dominant remaining cost but massively reduced vs. serial.
- **Reasoning:** Pass 1 chunks now dispatched 8-concurrent instead of serial.

Transcription went from the largest slice to a rounding error; visual captioning
is now the sole bottleneck (see the 2026-07-01 throughput section below).

---

## 2026-07-01 — Captioning & reasoning throughput

Context: on a 13.5-min screen recording (deep mode), visual captioning was
~86% of wall time (~35 min for 405 frames at ~5 s/frame), and Pass 1 chunk
reasoning was the next bottleneck. Both stages ran fully serially against a
llama.cpp server (Gemma-4 E4B) on `localhost:9090`. Hardware is a single
NVIDIA GB10 (130 GB unified memory, CUDA 13.0).

### D1 — Dedup frames *before* inference, not after

**Decision:** compute a perceptual hash per sampled frame and skip captioning
any frame within `DEDUP_HAMMING_THRESHOLD` bits of the last *captioned* frame;
skipped frames inherit that frame's caption.

**Rejected:** the prior approach captioned every frame and then dropped
near-duplicate *captions* (`_deduplicate_captions`). That paid full inference
cost for frames it threw away — the opposite of what we want when inference is
the bottleneck. Deleted.

**Why this matters most:** on screen recordings / static footage, dedup removes
the majority of frames outright (measured: ~64% in quick mode, ~80% in deep).
No model change, no quality knob — just don't do redundant work.

**Consequence / trade-off:** a static UI element present across 100 near-identical
frames is captioned from fewer samples, so the visual summary names specific
elements slightly less often. Judged an acceptable trade for ~5x; tunable via
`DEDUP_HAMMING_THRESHOLD` (lower = denser coverage).

### D2 — Hash = dHash + thermometer-coded brightness prefix

**Decision:** the frame fingerprint is a 64-bit difference hash (dHash) with a
16-bit thermometer-coded mean-brightness prefix.

**Rejected:** plain dHash alone. dHash is a *gradient* fingerprint, so flat
frames (an all-black intro, an all-white slide) all hash to zero and would be
wrongly merged. **Also rejected:** a small (3-bit) brightness bucket and Gray
coding — neither produced enough Hamming distance for a black→white swing to
clear the dedup threshold. Thermometer coding makes the prefix's Hamming
distance equal the brightness-bucket difference: black↔white = 15 bits (kept),
noise-level shifts = 0 bits (merged). Verified in `tests`.

**Rejected:** adding the `imagehash` dependency — dHash is ~15 lines and Pillow
is already a dependency.

### D3 — Real concurrency via semaphore + gather

**Decision:** `caption_frames_batch` and `analyze_chunks` dispatch requests
concurrently, bounded by an `asyncio.Semaphore`, order-preserving via `gather`.

**Rejected:** the status quo. `caption_frames_batch` already took a
`max_concurrent` parameter but **never used it** — it was a serial `for` loop.
`analyze_chunks` had a "batch" concept but `await`ed each chunk one at a time
inside the batch. Both were serial in practice.

**Sizing:** concurrency is capped at the server's `--parallel` slot count.
Benchmarked scaling on the E4B server: 1→4 slots gave ~2.8x, 4→8 gave ~3.6x,
8→16 gave nothing (plateaus at slot count). The ceiling is single-GPU compute,
not request dispatch — 8 slots buys ~3x, not 8x. We chose **8** as the point of
diminishing returns; 16 added latency and KV-cache VRAM for no throughput.

### D4 — One shared E4B preset for both stages, not two

**Decision:** vision captioning and Pass 1/2 reasoning point at the *same*
router preset (`Huihui-gemma-4-E4B-it-abliterated-Q4_K_M`, `parallel = 8`), so
the model loads once and serves both.

**Rejected:** a separate `...-Q4_K_M-vision` preset (tried first). The llama.cpp
router keys presets by model id, so a distinct id means a **second model load** —
the same E4B weights resident twice. On the GB10's 130 GB that fit fine, but it's
wasteful. Reasoning is sequential relative to captioning (Pass 1 runs after
Stage 0), so the 8 slots are shared in time, not contended; the only cost of
8 slots for the reasoning workload is idle KV-cache VRAM, which is cheap for E4B.

**Consequence:** captioning and reasoning share `parallel = 8`. If a workload
ever needs them isolated (e.g. running two videos concurrently), revisit.

### D5 — Concurrency knobs live in config, per stage

**Decision:** `video.caption_max_concurrent` / `video.caption_dedup` and
`analysis.reasoning_max_concurrent` (all default 8/true). The old
`analysis.max_chunks_per_batch` was dead config; repurposed into
`reasoning_max_concurrent` with back-compat for the old key name.

**Note / defect fixed alongside:** `resolve_video_config` and
`resolve_analysis_config` were silently dropping most fields (they only forwarded
the mode-overridable ones), so `caption_*` and the reasoning knob never reached
the pipeline — runs used dataclass defaults regardless of `config.yaml`. Fixed to
carry all non-overridden fields through.

### D6 — `parse_key_moment_time` tolerates every LLM shape

**Decision:** a single helper coerces a key-moment `"time"` to seconds from any
of: number, unit string (`"12.5s"`), range (`"30s - 45s"`), or a **list** of
those (`["0.0s", "5.0s"]`), with a safe fallback to the chunk start.

**Why:** the list-of-unit-strings shape crashed Pass 2 with
`could not convert string to float: '0.0s'`. It only surfaced in deep mode,
where 90 s chunks generate ~3x more key moments than quick mode's 5 min chunks,
raising the odds of hitting the bad shape. LLM output is not a stable contract;
parse defensively. Covered by `tests/test_analyzer.py`.

---

## Open items (not yet decided)

- **Scene detection finds 0 scenes at threshold 0.3 on screen recordings**, so
  deep mode's scene-based chunking silently falls back to fixed 90 s chunks. A
  lower `video.scene_threshold` (~0.1) likely fits screen-capture content, but
  the right default per content type is unresolved.
- **README performance table** reflects the token-burn screen recording, a
  favorable dedup case. A high-motion video (little to dedup) would lean entirely
  on the ~3x concurrency win; no such measurement is recorded yet.
