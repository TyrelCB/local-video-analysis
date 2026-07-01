"""Configuration loader and manager.

Reads config.yaml and applies mode-specific overrides.
Provides a flat interface for all pipeline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class ServerConfig:
    """Model/reasoning server configuration."""
    type: str = "llama_cpp"       # llama_cpp | ollama
    url: str = "http://localhost:9090"
    model: str = "Qwen3.6-35B-A3B-UD-Q4_K_XL"
    timeout: int = 300
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class AnalysisConfig:
    """Analysis pipeline configuration."""
    default_mode: str = "deep"
    architecture: str = "stage_0_plus_two_pass"
    chunk_seconds: int = 90
    overlap_seconds: int = 10
    max_chunks_per_batch: int = 10
    enable_diarization: bool = False
    enable_ocr: bool = False
    enable_scene_detection: bool = True
    enable_audio_events: bool = True
    enable_embeddings: bool = True


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    transcription_backend: str = "faster_whisper"
    whisper_model: str = "medium"
    language: str = "auto"
    word_timestamps: bool = True
    diarization_backend: str = "pyannote"
    # Optional interpreter holding the GPU ASR stack (torch / NeMo). When set,
    # parakeet/torch_whisper run there via subprocess. Empty = run in-process.
    asr_python: str = ""


@dataclass
class VideoConfig:
    """Video processing configuration."""
    frame_sampling_fps: float = 0.5
    scene_threshold: float = 0.3
    max_frames_per_chunk: int = 12
    ocr_backend: str = "paddleocr"


@dataclass
class StorageConfig:
    """Storage configuration."""
    database: str = "sqlite"
    vector_store: str = "sqlite_fts5"
    output_dir: str = "./outputs"
    retain_intermediate_files: bool = True


@dataclass
class PrivacyConfig:
    """Privacy policy configuration."""
    local_only: bool = True
    allow_api_processing: bool = False
    allow_cloud_uploads: bool = False


@dataclass
class AppConfig:
    """Root application configuration."""
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    reasoning_server: ServerConfig = field(default_factory=ServerConfig)
    vision_server: ServerConfig = field(default_factory=ServerConfig)
    embedding_server: ServerConfig = field(default_factory=lambda: ServerConfig(type="none"))
    storage: StorageConfig = field(default_factory=StorageConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        # Analysis
        if "analysis" in data:
            a = data["analysis"]
            cfg.analysis = AnalysisConfig(
                default_mode=a.get("default_mode", cfg.analysis.default_mode),
                chunk_seconds=a.get("chunk_seconds", cfg.analysis.chunk_seconds),
                overlap_seconds=a.get("overlap_seconds", cfg.analysis.overlap_seconds),
                max_chunks_per_batch=a.get("max_chunks_per_batch", cfg.analysis.max_chunks_per_batch),
                enable_diarization=a.get("enable_diarization", cfg.analysis.enable_diarization),
                enable_ocr=a.get("enable_ocr", cfg.analysis.enable_ocr),
                enable_scene_detection=a.get("enable_scene_detection", cfg.analysis.enable_scene_detection),
                enable_audio_events=a.get("enable_audio_events", cfg.analysis.enable_audio_events),
            )

        # Audio
        if "audio" in data:
            au = data["audio"]
            cfg.audio = AudioConfig(
                transcription_backend=au.get("transcription_backend", cfg.audio.transcription_backend),
                whisper_model=au.get("whisper_model", cfg.audio.whisper_model),
                language=au.get("language", cfg.audio.language),
                word_timestamps=au.get("word_timestamps", cfg.audio.word_timestamps),
                asr_python=au.get("asr_python", cfg.audio.asr_python),
            )

        # Video
        if "video" in data:
            v = data["video"]
            cfg.video = VideoConfig(
                frame_sampling_fps=v.get("frame_sampling_fps", cfg.video.frame_sampling_fps),
                scene_threshold=v.get("scene_threshold", cfg.video.scene_threshold),
                max_frames_per_chunk=v.get("max_frames_per_chunk", cfg.video.max_frames_per_chunk),
            )

        # Models — reasoning server
        if "models" in data:
            m = data["models"]
            if "reasoning_server" in m:
                rs = m["reasoning_server"]
                cfg.reasoning_server = ServerConfig(
                    type=rs.get("type", cfg.reasoning_server.type),
                    url=rs.get("url", cfg.reasoning_server.url),
                    model=rs.get("model", cfg.reasoning_server.model),
                    timeout=rs.get("timeout", cfg.reasoning_server.timeout),
                    temperature=rs.get("temperature", cfg.reasoning_server.temperature),
                    max_tokens=rs.get("max_tokens", cfg.reasoning_server.max_tokens),
                )
            if "vision_server" in m:
                vs = m["vision_server"]
                cfg.vision_server = ServerConfig(
                    type=vs.get("type", cfg.vision_server.type),
                    url=vs.get("url", cfg.vision_server.url),
                    model=vs.get("model", cfg.vision_server.model),
                    timeout=vs.get("timeout", cfg.vision_server.timeout),
                    temperature=vs.get("temperature", cfg.vision_server.temperature),
                    max_tokens=vs.get("max_tokens", cfg.vision_server.max_tokens),
                )
            if "embedding_server" in m:
                es = m["embedding_server"]
                cfg.embedding_server = ServerConfig(
                    type=es.get("type", "none"),
                )

        # Storage
        if "storage" in data:
            s = data["storage"]
            cfg.storage = StorageConfig(
                output_dir=s.get("output_dir", cfg.storage.output_dir),
                retain_intermediate_files=s.get("retain_intermediate_files", cfg.storage.retain_intermediate_files),
            )

        # Modes
        cfg.modes = data.get("modes", {})

        return cfg

    def get_mode_config(self, mode: str) -> dict[str, Any]:
        """Get mode-specific overrides."""
        return self.modes.get(mode, {})

    def resolve_analysis_config(self, mode: str) -> AnalysisConfig:
        """Resolve analysis config with mode-specific overrides applied."""
        base = self.analysis
        overrides = self.get_mode_config(mode)
        return AnalysisConfig(
            chunk_seconds=overrides.get("chunk_seconds", base.chunk_seconds),
            overlap_seconds=overrides.get("overlap_seconds", base.overlap_seconds),
            enable_diarization=overrides.get("enable_diarization", base.enable_diarization),
            enable_ocr=overrides.get("enable_ocr", base.enable_ocr),
            enable_scene_detection=overrides.get("enable_scene_detection", base.enable_scene_detection),
            enable_audio_events=overrides.get("enable_audio_events", base.enable_audio_events),
        )

    def resolve_video_config(self, mode: str) -> VideoConfig:
        """Resolve video config with mode-specific overrides applied."""
        base = self.video
        overrides = self.get_mode_config(mode)
        return VideoConfig(
            frame_sampling_fps=overrides.get("frame_sampling_fps", base.frame_sampling_fps),
        )
