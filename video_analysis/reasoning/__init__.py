"""Reasoning layer: model inference clients.

Provides a unified interface for both text reasoning (Pass 1/2) and
vision captioning (Stage 0) via llama.cpp and Ollama servers.
"""

from .server import ChatImage, ChatMessage, CompletionResult, ReasoningClient, VisionClient
from .llama_cpp import LlamaCppClient, LlamaCppConfig
from .ollama import OllamaClient, OllamaConfig

__all__ = [
    "ChatImage",
    "ChatMessage",
    "CompletionResult",
    "ReasoningClient",
    "VisionClient",
    "LlamaCppClient",
    "LlamaCppConfig",
    "OllamaClient",
    "OllamaConfig",
]
