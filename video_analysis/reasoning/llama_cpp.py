"""llama.cpp server client.

Connects to a running llama.cpp server via its OpenAI-compatible API.
Supports both text-only and multimodal (image + text) conversations.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .server import ChatImage, ChatMessage, CompletionResult, ReasoningClient, VisionClient

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """llama.cpp server configuration."""
    url: str = "http://127.0.0.1:40285"
    model: str = "Qwen3.6-35B-A3B-UD-Q4_K_XL"
    timeout: int = 300
    temperature: float = 0.3
    max_tokens: int = 4096


class LlamaCppClient(VisionClient):
    """OpenAI-compatible client for llama.cpp servers."""

    def __init__(self, config: LlamaCppConfig | None = None):
        self.config = config or LlamaCppConfig()
        self._base_url = self.config.url.rstrip("/")
        self._timeout = self.config.timeout
        self._default_temperature = self.config.temperature
        self._default_max_tokens = self.config.max_tokens

    def _messages_to_openai(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert our ChatMessage format to OpenAI API format."""
        result = []
        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "assistant":
                result.append({"role": "assistant", "content": msg.content})
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content})
            result[-1]["name"] = msg.role  # metadata for tracking
        return result

    def _format_user_content(self, message: ChatMessage) -> str | list:
        """Format user message content, handling image attachments."""
        if not hasattr(message, "images") or not message.images:  # type: ignore[attr-defined]
            return message.content

        parts: list[dict] = [{"type": "text", "text": message.content}]
        for img in message.images:  # type: ignore[attr-defined]
            b64 = base64.b64encode(img.data).decode("ascii")
            media_type = img.format.split("/")[1] if "/" in img.format else "jpeg"
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.format};base64,{b64}"},
            })
        return parts

    async def chat(self, messages: list[ChatMessage],
                   temperature: float | None = None,
                   max_tokens: int | None = None,
                   **kwargs) -> CompletionResult:
        """Send a text-only chat completion to the llama.cpp server."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": self._messages_to_openai(messages),
            "temperature": temperature if temperature is not None else self._default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "stream": False,
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                return CompletionResult(text="", error="No choices in response")

            message = choices[0].get("message", {})
            text = message.get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")
            usage = data.get("usage", {})

            return CompletionResult(
                text=text,
                model=data.get("model", self.config.model),
                finish_reason=finish_reason,
                usage=usage,
            )

        except httpx.HTTPError as e:
            logger.error("llama.cpp API error: %s", e)
            return CompletionResult(text="", error=str(e))
        except Exception as e:
            logger.error("llama.cpp unexpected error: %s", e)
            return CompletionResult(text="", error=str(e))

    async def chat_with_images(self, messages: list[ChatMessage],
                                images: list[ChatImage],
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                **kwargs) -> CompletionResult:
        """Send a multimodal chat completion with images."""
        url = f"{self._base_url}/v1/chat/completions"

        # Build messages with image attachment
        openai_messages = []
        for msg in messages:
            if msg.role == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == "assistant":
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "user":
                content = self._format_user_content(msg)
                openai_messages.append({"role": "user", "content": content})

        # Attach images directly if not already in message content
        if not openai_messages or not isinstance(
            openai_messages[-1].get("content"), list
        ):
            # No images in user message, append them
            parts: list[dict] = []
            for img in images:
                b64 = base64.b64encode(img.data).decode("ascii")
                media_type = img.format.split("/")[1] if "/" in img.format else "jpeg"
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.format};base64,{b64}"},
                })
            if openai_messages and openai_messages[-1]["role"] == "user":
                current = openai_messages[-1].get("content", "")
                if isinstance(current, str):
                    parts.insert(0, {"type": "text", "text": current})
                openai_messages[-1]["content"] = parts
            else:
                openai_messages.append({"role": "user", "content": parts})

        payload = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": temperature if temperature is not None else self._default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "stream": False,
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                return CompletionResult(text="", error="No choices in response")

            message = choices[0].get("message", {})
            text = message.get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")
            usage = data.get("usage", {})

            return CompletionResult(
                text=text,
                model=data.get("model", self.config.model),
                finish_reason=finish_reason,
                usage=usage,
            )

        except httpx.HTTPError as e:
            logger.error("llama.cpp API error: %s", e)
            return CompletionResult(text="", error=str(e))
        except Exception as e:
            logger.error("llama.cpp unexpected error: %s", e)
            return CompletionResult(text="", error=str(e))

    async def health(self) -> bool:
        """Check if the llama.cpp server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def unload(self) -> bool:
        """Ask a llama.cpp router server to unload this client's model.

        Best-effort: only the router build (``--models-preset``) exposes
        ``POST /models/unload``; a plain single-model server returns 404, and
        "model is not running" is also a no-op success. Callers should treat
        any False return as advisory, not an error — freeing memory here is
        an optimization, not a correctness requirement.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base_url}/models/unload", json={"model": self.config.model})
                if resp.status_code == 200:
                    return True
                logger.info("llama.cpp unload(%s) returned %d: %s",
                            self.config.model, resp.status_code, resp.text[:200])
                return False
        except Exception as e:
            logger.info("llama.cpp unload(%s) failed: %s", self.config.model, e)
            return False
