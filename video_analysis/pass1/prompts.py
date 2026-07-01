"""Pass 1 prompt templates.

Templates for segment-level analysis sent to the reasoning model.
"""

PASS1_SYSTEM_PROMPT = (
    "You are an AI assistant analyzing segments of a video. "
    "For each segment, you will receive the transcript, visual descriptions, "
    "OCR text (if any), audio events, and scene data. "
    "Return your analysis as structured JSON with the fields described below. "
    "Be precise with timestamps. Do not make up timestamps or content that "
    "is not present in the input data."
)

PASS1_USER_PROMPT_TEMPLATE = (
    "SEGMENT ANALYSIS\n"
    "Video duration: {video_duration} | Segment: {segment_time}\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "VISUAL DESCRIPTIONS:\n{visual_captions}\n\n"
    "OCR TEXT:\n{ocr_text}\n\n"
    "AUDIO EVENTS:\n{audio_events}\n\n"
    "SCENES:\n{scenes}\n\n"
    "---\n"
    "Return a JSON object with these fields:\n"
    "- summary: 2-3 sentence summary of what happens\n"
    "- key_moments: array of {{time, description}}\n"
    "- tags: array of topic/action tags\n"
    "- quotes: array of {{time, speaker, text}}\n"
    "- issues: array of error/warning/problem strings\n"
    "- detected_actions: array of action strings\n"
    "- speaker_labels: array of speaker labels found\n\n"
    "Use timestamps in seconds relative to the video start."
)
