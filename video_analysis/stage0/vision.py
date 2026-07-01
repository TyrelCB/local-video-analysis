"""Visual captioning using a local vision model.

Sends sampled frames to the vision model (llama.cpp multimodal) to
generate descriptions of objects, UI elements, scenes, and visible actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..reasoning.llama_cpp import LlamaCppClient, LlamaCppConfig
from ..reasoning.ollama import OllamaConfig, OllamaClient
from ..reasoning.server import ChatImage, ChatMessage, CompletionResult

logger = logging.getLogger(__name__)

# Prompt templates for visual captioning
VISION_SYSTEM_PROMPT = (
    "You are an AI assistant analyzing video frames. Describe what you see in "
    "detail: objects, people, text, UI elements, scenes, actions, and context. "
    "Be specific and objective. For screen recordings, describe commands, code, "
    "terminal output, and UI interactions. If no meaningful content is visible "
    "(blank, black, identical to previous), say so briefly."
)

VISION_USER_PROMPT = (
    "Describe this video frame in detail. Include: objects, people, text, UI "
    "elements, scene context, actions, and any visible content. Be thorough but concise."
)


@dataclass
class VisualCaption:
    """A caption generated for a frame or frame group."""
    timestamp_seconds: float
    caption: str
    frame_path: str | None = None


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
        max_tokens=256,
    )

    if result.is_error:
        logger.error("Vision captioning failed: %s", result.error)
        return f"CAPTION_FAILED: {result.error}"

    return result.text.strip()


async def caption_frames_batch(frame_paths: list[str], timestamps: list[float],
                                client: LlamaCppClient,
                                max_concurrent: int = 1) -> list[VisualCaption]:
    """Generate captions for a batch of frames.

    Args:
        frame_paths: List of frame image paths.
        timestamps: Corresponding timestamps for each frame.
        client: Vision model client.
        max_concurrent: Max concurrent captioning requests.

    Returns:
        List of VisualCaption objects.
    """
    captions = []
    for frame_path, ts in zip(frame_paths, timestamps):
        caption = await caption_frame(frame_path, client)
        captions.append(VisualCaption(
            timestamp_seconds=ts,
            caption=caption,
            frame_path=frame_path,
        ))
    return captions


async def caption_video_frames(video_path: str, frame_paths: list[str],
                                timestamps: list[float],
                                llm_client: LlamaCppClient | None = None,
                                vision_config: LlamaCppConfig | None = None,
                                deduplicate: bool = True) -> list[VisualCaption]:
    """Generate visual captions for all sampled video frames.

    Optionally deduplicates captions for similar/identical frames.

    Args:
        video_path: Video path (for context).
        frame_paths: Paths to sampled frames.
        timestamps: Timestamps for each frame.
        llm_client: Optional pre-initialized vision client.
        vision_config: Optional vision server config.
        deduplicate: Whether to skip captions for near-duplicate frames.

    Returns:
        List of VisualCaption objects.
    """
    if llm_client is None:
        cfg = vision_config or LlamaCppConfig()
        llm_client = LlamaCppClient(cfg)

    captions = await caption_frames_batch(frame_paths, timestamps, llm_client)

    if deduplicate and len(captions) > 1:
        captions = _deduplicate_captions(captions)

    return captions


def _deduplicate_captions(captions: list[VisualCaption]) -> list[VisualCaption]:
    """Deduplicate captions by removing consecutive near-identical descriptions.

    Simple heuristic: if a caption is very similar to the previous one,
    skip it (assuming frames are ordered by time and consecutive frames
    are similar).
    """
    if len(captions) <= 1:
        return captions

    result = [captions[0]]
    for caption in captions[1:]:
        prev = result[-1].caption
        curr = caption.caption

        # Simple length comparison: very short captions are likely identical frames
        if len(curr) < len(prev) * 0.7:
            continue

        # Skip if the caption is essentially the same (simple substring check)
        if curr in prev or prev in curr:
            continue

        result.append(caption)

    return result
