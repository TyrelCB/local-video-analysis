"""SQL schema for the analysis database.

Tables:
  - videos: video metadata
  - jobs: pipeline execution tracking
  - stage0_artifacts: Stage 0 extraction status
  - chunks: Pass 1 chunk data
  - key_moments: timestamped key moments
  - transcript_segments: word/segment-level transcription
  - speakers: diarization results (optional)
  - scenes: scene boundaries
  - frames: sampled frame metadata
  - ocr_results: on-screen text (Phase 2)
  - visual_captions: vision model descriptions
  - audio_events: non-speech audio detection
  - exports: exported file tracking

FTS5 virtual tables:
  - search_content: full-text search over transcripts, captions, OCR, summaries
"""

SCHEMA_SQL = """
-- Video metadata
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    duration_seconds REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    analysis_mode TEXT DEFAULT 'deep',
    status TEXT DEFAULT 'pending'  -- pending, running, completed, failed
);

-- Pipeline job tracking
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    current_stage TEXT,              -- stage0, pass1, pass2
    current_step TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    progress_percent REAL DEFAULT 0,
    error_message TEXT,
    config_json TEXT,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- Stage 0 artifact tracking
CREATE TABLE IF NOT EXISTS stage0_artifacts (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,     -- audio, transcript, scenes, frames, vision_captions, audio_events
    status TEXT DEFAULT 'pending',   -- pending, processing, completed, failed
    output_path TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- Pass 1 chunks
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    transcript TEXT,
    visual_summary TEXT,
    ocr_text TEXT,
    audio_events TEXT,               -- JSON array
    summary TEXT,
    tags TEXT,                       -- JSON array
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- Key moments
CREATE TABLE IF NOT EXISTS key_moments (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_id TEXT,
    timestamp_seconds REAL NOT NULL,
    title TEXT,
    description TEXT,
    importance INTEGER DEFAULT 1,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

-- Transcript segments (from faster-whisper)
CREATE TABLE IF NOT EXISTS transcript_segments (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_id TEXT,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    text TEXT,
    speaker TEXT,                    -- SPEAKER_01, etc. (empty if no diarization)
    words TEXT,                      -- JSON array of {word, start, end, confidence}
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

-- Speaker diarization (Phase 2)
CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    speaker_label TEXT NOT NULL,     -- SPEAKER_01, etc.
    start_seconds REAL,
    end_seconds REAL,
    segment_count INTEGER DEFAULT 0,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- Scene boundaries
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    scene_index INTEGER NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    frame_path TEXT,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- Sampled frames
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_id TEXT,
    timestamp_seconds REAL NOT NULL,
    frame_path TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

-- OCR results (Phase 2)
CREATE TABLE IF NOT EXISTS ocr_results (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id TEXT,
    timestamp_seconds REAL NOT NULL,
    text TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(frame_id) REFERENCES frames(id)
);

-- Visual captions from vision model
CREATE TABLE IF NOT EXISTS visual_captions (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id TEXT,
    timestamp_seconds REAL NOT NULL,
    caption TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(frame_id) REFERENCES frames(id)
);

-- Audio events
CREATE TABLE IF NOT EXISTS audio_events (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_id TEXT,
    timestamp_seconds REAL NOT NULL,
    event_type TEXT NOT NULL,        -- silence, music, applause, noise, laughter
    description TEXT,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

-- Exported files
CREATE TABLE IF NOT EXISTS exports (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    format TEXT NOT NULL,            -- json, markdown, srt, csv
    file_path TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

-- FTS5 full-text search table
CREATE VIRTUAL TABLE IF NOT EXISTS search_content USING fts5(
    content,                        -- combined searchable text
    video_id,
    chunk_id,
    timestamp_seconds,
    content_type                    -- transcript, visual_caption, ocr, summary
);

-- Trigger: auto-populate FTS on chunk insert
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO search_content (content, video_id, chunk_id, timestamp_seconds, content_type)
    VALUES (
        COALESCE(NEW.transcript, '') || ' ' || COALESCE(NEW.visual_summary, '') || ' ' || COALESCE(NEW.tags, ''),
        NEW.video_id,
        NEW.id,
        NEW.start_seconds,
        'chunk_summary'
    );
END;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_start ON chunks(video_id, start_seconds);
CREATE INDEX IF NOT EXISTS idx_key_moments_video ON key_moments(video_id);
CREATE INDEX IF NOT EXISTS idx_transcript_video ON transcript_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_scenes_video ON scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_visual_captions_video ON visual_captions(video_id);
CREATE INDEX IF NOT EXISTS idx_audio_events_video ON audio_events(video_id);
-- search_content is an FTS5 table — use MATCH queries instead of indexes
"""
