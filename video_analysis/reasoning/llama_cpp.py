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

    async def unload(self, wait_seconds: float = 15.0) -> bool:
        """Ask a llama.cpp router server to unload this client's model, and wait
        until the model actually reports unloaded before returning.

        The ``POST /models/unload`` request returns 200 almost immediately (~5ms),
        and even the ``/v1/models`` status flips to ``unloaded`` before the CUDA
        memory is actually released. On this box's unified memory, another process
        that grabs the GPU in that gap (the AST audio classifier) hits a
        driver-level ``CUDA out of memory`` at context creation while the server's
        pool is still mapped. So we wait for the status flip AND then for GPU
        memory to actually drop and settle (via ``nvidia-smi``) before returning.

        Best-effort: only the router build (``--models-preset``) exposes these
        endpoints; a plain single-model server returns 404. Callers should treat
        a False return as advisory, not an error.
        """
        import asyncio

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base_url}/models/unload", json={"model": self.config.model})
                if resp.status_code != 200:
                    logger.info("llama.cpp unload(%s) returned %d: %s",
                                self.config.model, resp.status_code, resp.text[:200])
                    return False

                # 1) Wait for the router to report the model unloaded.
                deadline = asyncio.get_event_loop().time() + wait_seconds
                status_ok = False
                while asyncio.get_event_loop().time() < deadline:
                    if await self._model_status() == "unloaded":
                        status_ok = True
                        break
                    await asyncio.sleep(0.25)
                if not status_ok:
                    logger.info("llama.cpp unload(%s): not 'unloaded' after %.0fs",
                                self.config.model, wait_seconds)
                    return False

                # 2) The status flips before CUDA memory is actually freed. Wait
                #    until GPU used-memory has dropped and stayed low for two
                #    consecutive reads, so the pool is genuinely released before a
                #    GPU-hungry successor (AST) starts.
                await self._wait_for_gpu_release(deadline)
                return True
        except Exception as e:
            logger.info("llama.cpp unload(%s) failed: %s", self.config.model, e)
            return False

    @staticmethod
    async def _wait_for_gpu_release(deadline: float, settle_reads: int = 2) -> None:
        """Block until nvidia-smi total used memory is low on two consecutive
        reads (the freed pool has settled), or ``deadline`` passes. No-op if
        nvidia-smi isn't available."""
        import asyncio
        low_streak = 0
        while asyncio.get_event_loop().time() < deadline:
            used = await LlamaCppClient._gpu_used_mib()
            if used is None:
                return  # can't measure; don't block
            # < ~2 GiB means the multi-GB model's pool is gone (idle contexts are
            # only a few hundred MiB).
            if used < 2048:
                low_streak += 1
                if low_streak >= settle_reads:
                    return
            else:
                low_streak = 0
            await asyncio.sleep(0.25)

    @staticmethod
    async def _gpu_used_mib() -> int | None:
        """Total GPU memory in use across compute apps (MiB), or None."""
        import asyncio
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await proc.communicate()
            return sum(int(x) for x in out.decode().split() if x.strip().isdigit())
        except Exception:
            return None

    async def _model_status(self) -> str | None:
        """Return this model's router status ('loaded'/'unloaded') or None."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                if resp.status_code != 200:
                    return None
                for m in resp.json().get("data", []):
                    if m.get("id") == self.config.model:
                        return (m.get("status") or {}).get("value")
        except Exception:
            return None
        return None
