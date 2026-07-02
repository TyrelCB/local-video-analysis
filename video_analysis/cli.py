"""CLI entry point for the video analysis engine.

Commands:
    analyze   — Run analysis on a video file
    ui        — Start the Gradio web UI
    mcp       — Start the MCP server
    info      — Show system info and available models
"""

from __future__ import annotations

import logging
import sys

import click

from .config import AppConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-c", "--config", type=click.Path(exists=True),
              help="Path to config.yaml.")
@click.pass_context
def main(ctx, verbose: bool, config: str | None):
    """Local Long-Form Video Analysis Engine."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    setup_logging(verbose)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--mode", "-m", type=click.Choice(["quick", "deep", "forensic"]),
              default=None, help="Analysis mode.")
@click.option("--prompt", "-p", type=str, default=None,
              help="User prompt for focused analysis.")
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Output directory.")
@click.pass_context
def analyze(ctx, video_path: str, mode: str, prompt: str, output_dir: str):
    """Analyze a video file.

    VIDEO_PATH is the path to the video file to analyze.
    """
    cfg = AppConfig.load(ctx.obj.get("config_path"))
    from .pipeline import run_pipeline

    progress_count = [0]

    def on_progress(stage: str, pct: float, msg: str):
        progress_count[0] += 1
        if progress_count[0] % 5 == 0 or stage in ("stage0", "pass2"):
            click.echo(f"[{stage.upper()}] {pct:.0f}% — {msg}")

    click.echo(f"Analyzing: {video_path}")
    click.echo(f"Mode: {mode or cfg.analysis.default_mode}")
    click.echo("")

    import asyncio

    result = asyncio.run(run_pipeline(
        video_path=video_path,
        config=cfg,
        user_prompt=prompt,
        mode=mode,
        output_dir=output_dir,
        on_progress=on_progress,
    ))

    click.echo("")
    if result.get("status") == "complete":
        click.echo(f"Video ID: {result['video_id']}")
        click.echo(f"Duration: {result['duration']}")
        click.echo(f"Executive summary: {result['summary']['executive'][:200]}...")
        click.echo(f"Chapters: {len(result['chapters'])}")
        click.echo(f"Key moments: {len(result['key_moments'])}")
        click.echo(f"Tags: {', '.join(result.get('tags', []))}")
        click.echo(f"Report: {result.get('report_path', 'N/A')}")
        click.echo(f"JSON: {result.get('json_path', 'N/A')}")
    else:
        click.echo(f"FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)


@main.command()
@click.option("--host", "-h", default="127.0.0.1", help="Server host.")
@click.option("--port", "-p", default=7860, help="Server port.")
def ui(host: str, port: int):
    """Start the Gradio web UI."""
    import os
    from .ui.app import create_app
    app = create_app()
    click.echo(f"Starting Gradio UI on http://{host}:{port}")
    # Videos are supplied by absolute path (the path textbox) and played back in
    # the gr.Video component, so Gradio must be allowed to serve files from
    # outside its cache. Without this, path inputs under $HOME raise
    # InvalidPathError ("not created by the application ... not in cwd/temp").
    app.launch(
        server_name=host, server_port=port,
        allowed_paths=[os.path.expanduser("~"), "/tmp", os.getcwd()],
    )


@main.command()
@click.option("--port", "-p", default=8000, help="MCP server port.")
def mcp(port: int):
    """Start the MCP server."""
    from .mcp.tools import create_server
    server = create_server()
    server.run(port=port)


@main.command()
@click.pass_context
def info(ctx):
    """Show system info and available models."""
    import platform
    import subprocess

    cfg = AppConfig.load(ctx.obj.get("config_path") if ctx.obj else None)

    click.echo("=== System Info ===")
    click.echo(f"Platform: {platform.system()} {platform.release()}")
    click.echo(f"Python: {platform.python_version()}")

    # GPU info
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines:
                if "Name" in line and "Memory" in line:
                    click.echo(f"GPU: {line.strip()}")
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        click.echo("GPU: Not detected (nvidia-smi not available)")

    # Model availability
    click.echo("")
    click.echo("=== Model Availability ===")

    # Check faster-whisper
    try:
        import faster_whisper
        click.echo("faster-whisper: installed (v" + faster_whisper.__version__ + ")")
    except ImportError:
        click.echo("faster-whisper: NOT installed")

    # Check Ollama
    import httpx
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                click.echo(f"Ollama: connected ({len(models)} models loaded)")
                for m in models:
                    click.echo(f"  - {m['name']} ({m['size'] / 1e9:.0f} GB)")
    except Exception:
        click.echo("Ollama: NOT connected")

    # Check llama.cpp — use the configured server URL(s), not a hardcoded port.
    # reasoning_server and vision_server may point at the same manager.
    server_urls = []
    for url in (cfg.reasoning_server.url, cfg.vision_server.url):
        if url and url not in server_urls:
            server_urls.append(url)
    for url in server_urls:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{url.rstrip('/')}/v1/models")
                if resp.status_code == 200:
                    # llama.cpp/OpenAI format returns models under "data".
                    models = resp.json().get("data") or resp.json().get("models", [])
                    ids = [m.get("id", m.get("model", "?")) for m in models]
                    click.echo(f"llama.cpp ({url}): connected, {len(ids)} models")
                    for mid in ids:
                        click.echo(f"  - {mid}")
                else:
                    click.echo(f"llama.cpp ({url}): HTTP {resp.status_code}")
        except Exception:
            click.echo(f"llama.cpp ({url}): NOT connected")

    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            click.echo(f"FFmpeg: {first_line.split(' version ')[1] if ' version ' in first_line else 'installed'}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        click.echo("FFmpeg: NOT installed")


if __name__ == "__main__":
    main()
