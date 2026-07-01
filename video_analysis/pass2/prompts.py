"""Pass 2 prompt templates.

Templates for global synthesis sent to the reasoning model.
"""

PASS2_SYSTEM_PROMPT = (
    "You are an AI assistant synthesizing analysis results from multiple "
    "video segments into a coherent full-video report. "
    "You will receive segment summaries, key moments, and tags from all chunks. "
    "Return your synthesis as structured JSON with the fields described below. "
    "Be thorough in connecting themes across segments. "
    "Create meaningful chapter titles that reflect the actual content."
)

PASS2_USER_PROMPT_TEMPLATE = (
    "FULL VIDEO SYNTHESIS\n"
    "Total segments analyzed: {num_chunks}\n"
    "Video duration: {video_duration}s\n\n"
    "--- SEGMENT SUMMARIES ---\n"
    "{chunk_summaries}\n\n"
    "--- ALL KEY MOMENTS ---\n"
    "{all_key_moments}\n\n"
    "--- ALL TAGS ---\n"
    "{all_tags}\n\n"
    "--- TASK ---\n"
    "Return a JSON object with these fields:\n"
    "- executive_summary: 1 paragraph overview\n"
    "- detailed_summary: 3-5 paragraphs by theme\n"
    "- chapters: array of {{start_seconds, end_seconds, title, summary}}\n"
    "    (start_seconds/end_seconds are integer SECONDS from the video start, in\n"
    "    the range 0..{video_duration}; do NOT use minutes or chunk indices)\n"
    "- key_moments: array of {{time, title, description, importance(1-5)}}\n"
    "    (time is SECONDS from the video start, 0..{video_duration})\n"
    "- speaker_summary: object mapping speaker to topics discussed\n"
    "- visual_summary: how visuals progress through the video\n"
    "- audio_summary: audio events and patterns\n"
    "- action_items: array of things viewer should do/learn\n"
    "- tags: comprehensive keyword list\n\n"
    "All timestamps are in SECONDS (0..{video_duration}), never minutes. "
    "Chapters should cover the full video with no gaps; the last chapter's "
    "end_seconds should be close to {video_duration}."
)
