"""Gradio web UI for video analysis.

Provides a web interface with:
- Video upload / path input
- Analysis mode selector
- User prompt input
- Progress display
- Output tabs (summary, transcript, timeline, search, exports)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import gradio as gr

from ..config import AppConfig

logger = logging.getLogger(__name__)

# Global state for tracking active analyses
_analysis_state: dict = {"running": False, "progress": "", "result": None}


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    cfg = AppConfig.load()

    with gr.Blocks(title="Video Analysis Engine", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Local Video Analysis Engine")
        gr.Markdown("Upload or provide a path to a video file. Run local AI analysis to generate summaries, chapters, and key moments.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## Input")
                video_input = gr.File(label="Upload Video (optional)", file_types=[".mp4", ".mkv", ".avi", ".mov"])
                video_path_input = gr.Textbox(
                    label="Or provide local video path",
                    placeholder="/path/to/video.mp4",
                )

                mode_select = gr.Radio(
                    choices=["quick", "deep", "forensic"],
                    value=cfg.analysis.default_mode,
                    label="Analysis Mode",
                )
                user_prompt = gr.Textbox(
                    label="Analysis Prompt (optional)",
                    placeholder="Analyze this as a tutorial and identify important setup steps.",
                    lines=2,
                )
                start_btn = gr.Button("Start Analysis", variant="primary")

            with gr.Column(scale=1):
                # Progress section
                gr.Markdown("## Progress")
                progress_output = gr.Textbox(label="Status", lines=3)
                status_badge = gr.Label(label="Status", value="idle")

        # Output tabs
        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_output = gr.Markdown()

            with gr.Tab("Chapters"):
                chapters_output = gr.Markdown()

            with gr.Tab("Key Moments"):
                moments_output = gr.Markdown()

            with gr.Tab("Transcript"):
                transcript_output = gr.Textbox(lines=15, max_lines=50)

            with gr.Tab("Search"):
                search_input = gr.Textbox(label="Search query", placeholder="Where does the speaker explain installation?")
                search_btn = gr.Button("Search")
                search_output = gr.JSON()

            with gr.Tab("Exports"):
                gr.Markdown("Download your analysis results:")
                json_download = gr.File(label="JSON Report")
                md_download = gr.File(label="Markdown Report")
                srt_download = gr.File(label="SRT Subtitles")

        # Event handlers
        start_btn.click(
            fn=_start_analysis,
            inputs=[video_path_input, mode_select, user_prompt],
            outputs=[progress_output, status_badge],
        ).then(
            fn=_show_results,
            outputs=[summary_output, chapters_output, moments_output,
                     transcript_output, json_download, md_download, srt_download],
        )

        search_btn.click(
            fn=_search,
            inputs=[search_input],
            outputs=[search_output],
        )

    return app


def _start_analysis(video_path: str, mode: str, user_prompt: str):
    """Handle analysis start."""
    if not video_path:
        yield "❌ No video provided. Please upload a file or enter a path.", "error"
        return

    _analysis_state["running"] = True
    yield f"▶ Starting analysis (mode: {mode})...", "running"

    # Run pipeline in a thread
    import threading
    from ..pipeline import run_pipeline

    def run():
        cfg = AppConfig.load()
        _analysis_state["result"] = run_pipeline(
            video_path=video_path,
            config=cfg,
            user_prompt=user_prompt,
            mode=mode,
        )
        _analysis_state["running"] = False

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Poll for progress
    while _analysis_state["running"]:
        time.sleep(1)
        yield "⏳ Analyzing...", "running"

    result = _analysis_state.get("result", {})
    if result.get("status") == "complete":
        yield f"✅ Complete! Duration: {result.get('duration', 'N/A')}", "complete"
    else:
        yield f"❌ Failed: {result.get('error', 'Unknown')}", "error"


def _show_results():
    """Show analysis results in output tabs."""
    result = _analysis_state.get("result", {})

    if not result or result.get("status") != "complete":
        return ("", "", "", "", None, None, None)

    # Summary
    summary_text = result.get("summary", {}).get("executive", "")
    summary_md = f"## Executive Summary\n\n{summary_text}"

    # Chapters
    chapters = result.get("chapters", [])
    chapters_md = "## Chapters\n\n" + "\n\n".join(
        f"### {ch.get('title', f'Chapter {i+1}')} (`{_ts(ch.get('start_seconds', 0))}`)\n{ch.get('summary', '')}"
        for i, ch in enumerate(chapters)
    )

    # Key moments
    moments = result.get("key_moments", [])
    moments_md = "## Key Moments\n\n" + "\n\n".join(
        f"**{_ts(m.get('time', 0))}** — {m.get('title', m.get('description', ''))}"
        for m in moments[:15]
    )

    # Transcript placeholder (would come from DB)
    transcript_text = "(Run from CLI to get full transcript)"

    # File downloads
    json_path = result.get("json_path")
    md_path = result.get("report_path")

    return (summary_md, chapters_md, moments_md, transcript_text,
            json_path, md_path, None)


def _search(query: str):
    """Search the analysis database."""
    # Placeholder — would connect to VideoSearch
    return {"query": query, "results": [], "message": "Search requires a completed analysis stored in the database."}


def _ts(seconds: float) -> str:
    """Convert seconds to timestamp string."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
