"""Base reasoning client interface.

All model clients implement this protocol. The pipeline calls
the same interface for both Pass 1 (text reasoning) and Pass 2
(global synthesis), while the vision client additionally supports
image inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str           # "system", "user", "assistant"
    content: str


@dataclass
class ChatImage:
    """An image to include with a message."""
    data: bytes         # Raw image bytes (JPEG/PNG)
    format: str = "image/jpeg"  # MIME format


@dataclass
class CompletionResult:
    """Result from a model completion call."""
    text: str
    model: str = ""
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


class ReasoningClient(ABC):
    """Base class for model inference clients."""

    @abstractmethod
    async def chat(self, messages: list[ChatMessage],
                   temperature: float = 0.3,
                   max_tokens: int = 4096,
                   **kwargs) -> CompletionResult:
        """Send a chat completion request.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            **kwargs: Additional model-specific parameters.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Check if the model server is reachable and healthy."""


class VisionClient(ReasoningClient):
    """Client that supports multimodal (image + text) input.

    Used for Stage 0 visual captioning.
    """

    @abstractmethod
    async def chat_with_images(self, messages: list[ChatMessage],
                                images: list[ChatImage],
                                temperature: float = 0.3,
                                max_tokens: int = 512,
                                **kwargs) -> CompletionResult:
        """Send a multimodal completion with images."""
