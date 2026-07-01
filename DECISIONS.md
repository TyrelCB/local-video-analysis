# Engineering Decisions

A running log of non-obvious design decisions — what was chosen, what was
rejected, and why. Read this before changing performance-sensitive paths;
the reasoning behind a knob is rarely visible from the code alone.

Format: one entry per decision, newest first. Each records the context, the
decision, alternatives considered, and consequences.

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
