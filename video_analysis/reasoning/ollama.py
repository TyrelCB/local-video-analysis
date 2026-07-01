"""Ollama client.

Fallback client for text reasoning via Ollama's API.
Ollama's image handling requires base64 embedding.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import httpx

from .server import ChatImage, ChatMessage, CompletionResult, ReasoningClient, VisionClient

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:30b"
    timeout: int = 300
    temperature: float = 0.3
    max_tokens: int = 4096


class OllamaClient(VisionClient):
    """Client for Ollama servers."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self._base_url = self.config.url.rstrip("/")
        self._timeout = self.config.timeout
        self._default_temperature = self.config.temperature
        self._default_max_tokens = self.config.max_tokens

    async def chat(self, messages: list[ChatMessage],
                   temperature: float | None = None,
                   max_tokens: int | None = None,
                   **kwargs) -> CompletionResult:
        """Send a chat completion to Ollama."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self._default_max_tokens,
            },
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            message = data.get("message", {})
            text = message.get("content", "")
            done = data.get("done", False)

            return CompletionResult(
                text=text,
                model=data.get("model", self.config.model),
                finish_reason="stop" if done else "length",
                usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                       "completion_tokens": data.get("eval_count", 0)},
            )

        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            return CompletionResult(text="", error=str(e))
        except Exception as e:
            logger.error("Ollama unexpected error: %s", e)
            return CompletionResult(text="", error=str(e))

    async def chat_with_images(self, messages: list[ChatMessage],
                                images: list[ChatImage],
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                **kwargs) -> CompletionResult:
        """Send a multimodal chat with images to Ollama."""
        url = f"{self._base_url}/api/chat"

        # Ollama expects images as base64 in the messages
        ollama_messages = []
        for msg in messages:
            content = msg.content
            if msg.role == "user" and images:
                # Ollama expects images embedded in user message
                b64_images = [base64.b64encode(img.data).decode("ascii") for img in images]
                content_with_images = {
                    "type": "text",
                    "text": content,
                }
                # Ollama format: images as separate array in message
                ollama_messages.append({
                    "role": "user",
                    "content": content,
                    "images": b64_images,
                })
            else:
                ollama_messages.append({"role": msg.role, "content": content})

        # Append a user message with images
        if images:
            first_user_text = messages[0].content if messages else ""
            b64_images = [base64.b64encode(img.data).decode("ascii") for img in images]
            if not any(m["role"] == "user" for m in ollama_messages):
                ollama_messages.insert(0, {
                    "role": "user",
                    "content": first_user_text,
                    "images": b64_images,
                })
            else:
                # Add images to existing user message
                for m in ollama_messages:
                    if m["role"] == "user":
                        existing_images = m.get("images", [])
                        m["images"] = existing_images + b64_images
                        break

        payload = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self._default_max_tokens,
            },
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            message = data.get("message", {})
            text = message.get("content", "")
            done = data.get("done", False)

            return CompletionResult(
                text=text,
                model=data.get("model", self.config.model),
                finish_reason="stop" if done else "length",
                usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                       "completion_tokens": data.get("eval_count", 0)},
            )

        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            return CompletionResult(text="", error=str(e))
        except Exception as e:
            logger.error("Ollama unexpected error: %s", e)
            return CompletionResult(text="", error=str(e))

    async def health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
