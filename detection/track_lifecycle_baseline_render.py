"""Rendering and output helpers for the track lifecycle baseline.

Provides render_table, render_markdown, write_json, and write_outputs used by
track_lifecycle_baseline.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for _, key in columns:
            value = row.get(key, "")
            rendered.append(str(value) if key == "phase_id" else fmt(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def render_markdown(data_root: Path, phase_metrics: pd.DataFrame, quality: dict[str, Any]) -> str:
    rows = phase_metrics.to_dict(orient="records")
    lines = [
        "# Track Lifecycle Baseline current",
        "",
        f"Data root: `{data_root}`",
        "",
        "## Status",
        "",
        f"- Artifact status: `{quality['status']}`",
        f"- Detector family ID: `{quality['detector_family_id']}`",
        f"- Lifecycle rows: `{quality['lifecycle_row_count']}`",
        f"- Model-facing default: `{str(quality['model_facing_default']).lower()}`",
        "",
        "## Claim Boundary",
        "",
        str(quality["claim_boundary"]),
        "",
        "## Phase Metrics",
        "",
        render_table(
            rows,
            [
                ("Phase", "phase_id"),
                ("Pd", "pd_any_cfar"),
                ("Pfa", "pfa_any_cfar"),
                ("Confirm", "track_confirmation_rate"),
                ("False Track", "false_track_rate"),
                ("Missed", "missed_track_rate"),
                ("Init s", "track_initiation_latency_s"),
                ("Frag", "track_fragmentation_rate"),
                ("Delete", "deletion_rate"),
                ("Stable", "stable_track_rate"),
            ],
        ),
        "",
        "## Audit Checks",
        "",
        "| Check | Value |",
        "|---|---|",
        f"| Denied lifecycle columns | {', '.join(quality['denied_lifecycle_columns_present']) or 'none'} |",
        f"| Missing lifecycle columns | {', '.join(quality['missing_lifecycle_columns']) or 'none'} |",
        f"| Missing phase metric columns | {', '.join(quality['missing_phase_metric_columns']) or 'none'} |",
        f"| Missing source IDs | {quality['source_id_missing_count']} |",
        f"| Detector IDs OK | {quality['detector_ids_ok']} |",
        "",
        "## Interpretation",
        "",
        "- This baseline gives later detector products a reproducible lifecycle reference for initiation, deletion, fragmentation, false tracks, and misses.",
        "- Current false-track, missed-track, and fragmentation values remain blocker evidence for regeneration-ready claims.",
        "- `initial_take_up` may legitimately remain weak for radar because line of sight is often horizon-masked; the row is reported rather than hidden.",
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(
    data_root: Path,
    out_dir: Path,
    report_path: Path,
    lifecycle: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    quality: dict[str, Any],
    denied_lifecycle_columns: set[str],
    track_detector_family_id: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.to_csv(out_dir / "track_lifecycle_rows.csv", index=False, float_format="%.6f")
    phase_metrics.to_csv(out_dir / "track_lifecycle_phase_metrics.csv", index=False, float_format="%.6f")
    schema = {
        "artifact": track_detector_family_id,
        "files": {
            "lifecycle_rows": "track_lifecycle_rows.csv",
            "phase_metrics": "track_lifecycle_phase_metrics.csv",
            "quality": "track_lifecycle_quality.json",
        },
        "lifecycle_columns": list(lifecycle.columns),
        "phase_metric_columns": list(phase_metrics.columns),
        "denied_lifecycle_columns": sorted(denied_lifecycle_columns),
        "model_facing_default": False,
    }
    write_json(out_dir / "track_lifecycle_schema.json", schema)
    write_json(out_dir / "track_lifecycle_quality.json", quality)
    markdown = render_markdown(data_root, phase_metrics, quality)
    report_path.write_text(markdown, encoding="utf-8")
    (out_dir / "track_lifecycle_baseline.md").write_text(markdown, encoding="utf-8")
