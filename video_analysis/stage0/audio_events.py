"""Audio event detection using rule-based analysis with librosa.

Detects silence, music, noise, laughter, and audio dropout by analyzing
amplitude, spectral features, and temporal patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    """A detected audio event."""
    timestamp_seconds: float
    event_type: str           # silence, music, noise, laughter, dropout, onset
    description: str = ""
    confidence: float = 0.5


def detect_audio_events(audio_path: str,
                         silence_threshold: float = 0.01,
                         silence_duration: float = 1.0,
                         music_threshold: float = 0.5,
                         window_size: float = 0.5,
                         include_onsets: bool = False) -> list[AudioEvent]:
    """Detect audio events in an audio file using librosa analysis.

    This is structural acoustic analysis (silence / music-or-noise / onsets),
    not semantic sound-event classification — it does not name sounds like
    "applause" or "laughter".

    Args:
        audio_path: Path to the WAV audio file (16kHz mono).
        silence_threshold: RMS amplitude below which is considered silence.
        silence_duration: Minimum duration of silence in seconds.
        music_threshold: Spectral centroid ratio threshold for music detection.
        window_size: Analysis window size in seconds.
        include_onsets: If True, also emit per-transient "onset" events. These
            fire on essentially every speech syllable, so they're excluded by
            default — they add hundreds of low-value rows on talking-head audio.

    Returns:
        List of AudioEvent objects sorted by timestamp.
    """
    if not _LIBROSA_AVAILABLE:
        logger.warning("librosa not available; audio events will be empty")
        return []
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        logger.error("Failed to load audio: %s", e)
        return []

    events: list[AudioEvent] = []
    duration = librosa.get_duration(y=y, sr=sr)

    # Compute frame-level features
    hop_length = int(sr * 0.05)  # 50ms hop
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Spectral centroid for music vs speech/noise detection
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    # Normalize
    if spectral_centroid.max() > 0:
        spectral_centroid = spectral_centroid / spectral_centroid.max()

    # Detect silence regions
    silence_regions = _find_silence_regions(rms, times, silence_threshold, silence_duration)
    for start, end in silence_regions:
        events.append(AudioEvent(
            timestamp_seconds=round(start, 3),
            event_type="silence",
            description=f"Silence from {start:.1f}s to {end:.1f}s ({end - start:.1f}s)",
            confidence=0.8,
        ))

    # Detect onset/transient events (off by default — fires per speech syllable)
    if include_onsets:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
        for t in onset_times:
            idx = np.argmin(np.abs(times - t))
            # Skip if in a silence region
            if _is_in_region(t, silence_regions):
                continue
            # Check if this is a significant amplitude spike
            if idx < len(rms) and rms[idx] > silence_threshold * 10:
                events.append(AudioEvent(
                    timestamp_seconds=round(t, 3),
                    event_type="onset",
                    description=f"Sound onset at {t:.1f}s",
                    confidence=0.6,
                ))

    # Windowed analysis for music/noise classification
    step = int(window_size * sr / hop_length)
    if step < 1:
        step = 1
    for i in range(0, len(rms) - step, step):
        window_rms = rms[i:i + step]
        window_centroid = spectral_centroid[i:i + step]
        start_t = times[i]
        end_t = times[min(i + step, len(times)) - 1]

        if np.mean(window_rms) < silence_threshold:
            continue  # Skip silence

        avg_centroid = np.mean(window_centroid)
        if avg_centroid > music_threshold and i % 20 < step:  # Don't label every window
            events.append(AudioEvent(
                timestamp_seconds=round(start_t, 3),
                event_type="music",
                description=f"Possible music/noise at {start_t:.1f}s",
                confidence=float(avg_centroid),
            ))

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp_seconds)

    return events


def _find_silence_regions(rms: np.ndarray, times: np.ndarray,
                           threshold: float, min_duration: float) -> list[tuple[float, float]]:
    """Find contiguous regions where RMS is below threshold."""
    is_silent = rms < threshold
    regions = []
    in_silence = False
    region_start = 0.0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            region_start = times[i]
        elif not silent and in_silence:
            in_silence = False
            region_end = times[i - 1]
            if region_end - region_start >= min_duration:
                regions.append((region_start, region_end))

    # Close final region if still in silence
    if in_silence and times[-1] - region_start >= min_duration:
        regions.append((region_start, times[-1]))

    return regions


def _is_in_region(time: float, regions: list[tuple[float, float]]) -> bool:
    """Check if a timestamp falls within any of the given regions."""
    for start, end in regions:
        if start <= time <= end:
            return True
    return False
