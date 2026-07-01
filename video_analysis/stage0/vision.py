"""Visual captioning using a local vision model.

Sends sampled frames to the vision model (llama.cpp multimodal) to
generate descriptions of objects, UI elements, scenes, and visible actions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..reasoning.llama_cpp import LlamaCppClient, LlamaCppConfig
from ..reasoning.ollama import OllamaConfig, OllamaClient
from ..reasoning.server import ChatImage, ChatMessage, CompletionResult

logger = logging.getLogger(__name__)

# Perceptual-hash dedup: frames whose dHash differs by <= this many bits from
# the last *captioned* frame are treated as near-identical and reuse its caption
# instead of hitting the vision model. 0 = exact match only; ~5 tolerates minor
# noise/compression while still catching real scene changes.
DEDUP_HAMMING_THRESHOLD = 5

# Prompt templates for visual captioning
VISION_SYSTEM_PROMPT = (
    "You are an AI assistant analyzing video frames. Describe what you see in "
    "detail: objects, people, text, UI elements, scenes, actions, and context. "
    "Be specific and objective. For screen recordings, describe commands, code, "
    "terminal output, and UI interactions. If no meaningful content is visible "
    "(blank, black, identical to previous), say so briefly."
)

VISION_USER_PROMPT = (
    "Describe this video frame in 2-4 sentences. Cover the most important: "
    "objects, people, on-screen text, UI elements, and any action. "
    "Prioritize specifics over completeness; do not pad."
)


@dataclass
class VisualCaption:
    """A caption generated for a frame or frame group."""
    timestamp_seconds: float
    caption: str
    frame_path: str | None = None


def _dhash(frame_path: str, hash_size: int = 8) -> int | None:
    """Compute a difference hash (dHash) for a frame.

    dHash is robust to minor compression/lighting changes but sensitive to real
    content changes, which is exactly what we want for skipping near-duplicate
    frames before spending inference on them. Returns an int bit-vector, or None
    if the frame can't be read (caller should then not treat it as a duplicate).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(frame_path) as img:
            # Grayscale, resized to (hash_size+1) x hash_size; compare adjacent
            # columns to build a hash_size*hash_size bit fingerprint.
            small = img.convert("L").resize(
                (hash_size + 1, hash_size), Image.LANCZOS
            )
            pixels = list(small.getdata())
    except Exception as e:
        logger.debug("dHash failed for %s: %s", frame_path, e)
        return None

    width = hash_size + 1
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)

    # dHash alone is a gradient fingerprint: flat/featureless frames all hash to
    # zero regardless of brightness (an all-black frame == an all-white one).
    # Prefix a thermometer-coded mean-brightness bucket so a pure luminance shift
    # on featureless frames still registers. Thermometer coding (bucket b -> low
    # b bits set) makes the Hamming distance between two brightness prefixes equal
    # the difference in buckets: a small brightness delta stays under the dedup
    # threshold, while black->white spans all BRIGHTNESS_BUCKETS bits.
    BRIGHTNESS_BUCKETS = 16
    mean = sum(pixels) / len(pixels)
    bucket = min(BRIGHTNESS_BUCKETS, int(mean / 256 * BRIGHTNESS_BUCKETS))
    thermometer = (1 << bucket) - 1
    return (thermometer << (hash_size * hash_size)) | bits


def _hamming(a: int, b: int) -> int:
    """Bit-count of the XOR — number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


async def caption_frame(frame_path: str, client: LlamaCppClient,
                        prompt: str = VISION_USER_PROMPT) -> str:
    """Generate a visual caption for a single frame.

    Args:
        frame_path: Path to the image file.
        client: Vision model client.
        prompt: User prompt for captioning.

    Returns:
        The generated caption text, or an error message.
    """
    try:
        with open(frame_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        logger.error("Frame not found: %s", frame_path)
        return "ERROR: frame not found"

    user_msg = ChatMessage(role="user", content=prompt)
    messages = [
        ChatMessage(role="system", content=VISION_SYSTEM_PROMPT),
        user_msg,
    ]

    result = await client.chat_with_images(
        messages,
        images=[ChatImage(data=image_bytes, format="image/jpeg")],
        temperature=0.3,
        max_tokens=160,
    )

    if result.is_error:
        logger.error("Vision captioning failed: %s", result.error)
        return f"CAPTION_FAILED: {result.error}"

    return result.text.strip()


async def caption_frames_batch(frame_paths: list[str], timestamps: list[float],
                                client: LlamaCppClient,
                                max_concurrent: int = 4) -> list[VisualCaption]:
    """Generate captions for a batch of frames, up to max_concurrent in flight.

    Requests are dispatched concurrently (bounded by a semaphore) so a GPU-backed
    server with multiple slots stays busy. Results preserve input order.

    Args:
        frame_paths: List of frame image paths.
        timestamps: Corresponding timestamps for each frame.
        client: Vision model client.
        max_concurrent: Max concurrent captioning requests. Set to the server's
            parallel-slot count; values above that just queue on the server.

    Returns:
        List of VisualCaption objects, in the same order as the inputs.
    """
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def _one(frame_path: str, ts: float) -> VisualCaption:
        async with sem:
            caption = await caption_frame(frame_path, client)
        return VisualCaption(
            timestamp_seconds=ts,
            caption=caption,
            frame_path=frame_path,
        )

    tasks = [_one(fp, ts) for fp, ts in zip(frame_paths, timestamps)]
    return list(await asyncio.gather(*tasks))


async def caption_video_frames(video_path: str, frame_paths: list[str],
                                timestamps: list[float],
                                llm_client: LlamaCppClient | None = None,
                                vision_config: LlamaCppConfig | None = None,
                                deduplicate: bool = True,
                                max_concurrent: int = 4,
                                max_frames: int | None = None) -> list[VisualCaption]:
    """Generate visual captions for sampled video frames.

    Two cost controls are applied *before* any inference happens:

    - ``deduplicate``: consecutive frames whose perceptual hash is within
      ``DEDUP_HAMMING_THRESHOLD`` bits of the last captioned frame reuse that
      caption instead of calling the model. This is the big win on screen
      recordings / static footage — near-duplicate frames cost nothing.
    - ``max_frames``: hard cap on how many *unique* frames get captioned; if the
      deduped set still exceeds it, frames are evenly subsampled across the
      timeline so coverage stays uniform.

    Args:
        video_path: Video path (for context).
        frame_paths: Paths to sampled frames (assumed ordered by time).
        timestamps: Timestamps for each frame.
        llm_client: Optional pre-initialized vision client.
        vision_config: Optional vision server config.
        deduplicate: Whether to skip near-duplicate frames before captioning.
        max_concurrent: Max concurrent captioning requests.
        max_frames: Optional cap on number of frames actually captioned.

    Returns:
        List of VisualCaption objects, ordered by timestamp. Deduped frames
        carry the caption of the frame they matched.
    """
    if llm_client is None:
        cfg = vision_config or LlamaCppConfig()
        llm_client = LlamaCppClient(cfg)

    frames = list(zip(frame_paths, timestamps))
    if not frames:
        return []

    # 1. Pre-inference dedup: partition into "to caption" vs "reuse prior".
    #    reuse_of[i] = index into `unique` whose caption frame i should inherit.
    if deduplicate:
        unique_idx, reuse_of = _select_unique_frames(frame_paths)
    else:
        unique_idx = list(range(len(frames)))
        reuse_of = {i: i for i in range(len(frames))}

    # 2. Hard cap: evenly subsample the unique set if it's still too large.
    if max_frames is not None and len(unique_idx) > max_frames:
        unique_idx = _evenly_subsample(unique_idx, max_frames)

    captioned_set = set(unique_idx)
    logger.info(
        "Captioning %d of %d frames (%d skipped via dedup/cap)",
        len(unique_idx), len(frames), len(frames) - len(unique_idx),
    )

    # 3. Caption only the selected frames, concurrently.
    sel_paths = [frame_paths[i] for i in unique_idx]
    sel_ts = [timestamps[i] for i in unique_idx]
    captioned = await caption_frames_batch(
        sel_paths, sel_ts, llm_client, max_concurrent=max_concurrent
    )
    by_index = {idx: cap for idx, cap in zip(unique_idx, captioned)}

    # 4. Reassemble full ordered list, letting skipped frames inherit a caption.
    results: list[VisualCaption] = []
    for i, (fp, ts) in enumerate(frames):
        if i in captioned_set:
            results.append(by_index[i])
            continue
        # Find the caption this frame should reuse; walk back to a captioned one.
        src = reuse_of.get(i, i)
        while src not in by_index and src in reuse_of and reuse_of[src] != src:
            src = reuse_of[src]
        source_cap = by_index.get(src)
        text = source_cap.caption if source_cap else "(no caption)"
        results.append(VisualCaption(timestamp_seconds=ts, caption=text, frame_path=fp))

    return results


def _select_unique_frames(frame_paths: list[str]) -> tuple[list[int], dict[int, int]]:
    """Partition frames into unique (to caption) vs. near-duplicates.

    Walks frames in order, keeping a running "reference" frame. A frame whose
    dHash is within DEDUP_HAMMING_THRESHOLD bits of the reference is a duplicate
    and maps to the reference's index; otherwise it becomes the new reference.

    Returns (unique_indices, reuse_of) where reuse_of maps every duplicate index
    to the reference index whose caption it should inherit. Frames whose hash
    can't be computed are treated as unique (never silently dropped).
    """
    unique_idx: list[int] = []
    reuse_of: dict[int, int] = {}

    ref_idx: int | None = None
    ref_hash: int | None = None

    for i, path in enumerate(frame_paths):
        h = _dhash(path)
        if ref_hash is None or h is None or _hamming(h, ref_hash) > DEDUP_HAMMING_THRESHOLD:
            unique_idx.append(i)
            reuse_of[i] = i
            ref_idx, ref_hash = i, h
        else:
            reuse_of[i] = ref_idx if ref_idx is not None else i

    return unique_idx, reuse_of


def _evenly_subsample(indices: list[int], k: int) -> list[int]:
    """Pick k items evenly spaced across `indices`, always keeping the first."""
    if k >= len(indices) or k <= 0:
        return indices
    step = len(indices) / k
    return [indices[int(i * step)] for i in range(k)]
