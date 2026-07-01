"""JSON export for analysis results.

Exports the complete analysis result as a structured JSON file
with metadata, summary, chapters, key moments, and chunk data.
"""

from __future__ import annotations

import json
from pathlib import Path


def export_json(result: dict, output_path: str | Path) -> str:
    """Export analysis result as JSON.

    Args:
        result: Analysis result dictionary (from AnalysisResult.to_dict()).
        output_path: Path for the output JSON file.

    Returns:
        Path to the exported file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return str(output_path)


def format_json_export(result: dict) -> str:
    """Format analysis result as a JSON string."""
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)
