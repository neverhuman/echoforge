"""Rendering helpers for the benchmark regeneration report.

Provides metric_table, render_markdown, and write_outputs used by
benchmark_regeneration_report.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def metric_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = ""
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if key != "phase_id":
                value = fmt(value)
            values.append(str(value))
        body += "| " + " | ".join(values) + " |\n"
    return header + divider + body


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark current Regeneration Report",
        "",
        f"Data root: `{payload['data_root']}`",
        "",
        "## Decision",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Regeneration ready: `{str(payload['regeneration_ready']).lower()}`",
        f"- Reason: {payload['reason']}",
        "",
        "## Claim Boundary",
        "",
        str(payload["claim_boundary"]),
        "",
        "## Resolved Integrity Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for key, value in payload.get("resolved_integrity_gates", {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Readiness Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
    )
    for key, value in payload.get("readiness_metrics", {}).items():
        if isinstance(value, float):
            rendered = fmt(value)
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value) if value else "none"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.extend(
        [
            "",
            "## Radar Phase Metrics",
            "",
            metric_table(
                payload.get("phase_operational_metrics", []),
                [
                    ("Phase", "phase_id"),
                    ("Pd", "pd_any_cfar"),
                    ("Pfa", "pfa_any_cfar"),
                    ("First Hit s", "first_hit_latency_s"),
                    ("Track Init s", "track_initiation_latency_s"),
                    ("Frag", "track_fragmentation_rate"),
                    ("False Track", "false_track_rate"),
                    ("Missed", "missed_track_rate"),
                    ("Horizon Masked", "horizon_masked_fraction"),
                ],
            ).rstrip(),
            "",
            "## Acoustic Phase Metrics",
            "",
            metric_table(
                payload.get("acoustic_phase_metrics", []),
                [
                    ("Phase", "phase_id"),
                    ("Pd", "pd_any_acoustic_cue"),
                    ("Pfa", "pfa_any_acoustic_cue"),
                    ("First Cue s", "first_cue_latency_s"),
                    ("Frag", "track_fragmentation_rate"),
                    ("False Track", "false_track_rate"),
                    ("Missed", "missed_track_rate"),
                    ("Pre-Radar LOS Cues", "pre_radar_los_cue_count"),
                ],
            ).rstrip(),
            "",
            "## Blockers",
            "",
        ]
    )
    for blocker in payload.get("blockers", []):
        lines.append(f"- `{blocker['blocker_id']}` (`{blocker['status']}`): {blocker['evidence']}")
        if blocker.get("required_next_evidence"):
            lines.append(f"  Required next evidence: {blocker['required_next_evidence']}")
    lines.extend(
        [
            "",
            "## Next Safe Packets",
            "",
        ]
    )
    for packet in payload.get("next_safe_packets", []):
        lines.append(f"- `{packet}`")
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], out_dir: Path, report_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(payload)
    report_path.write_text(markdown, encoding="utf-8")
    (out_dir / "benchmark_regeneration_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "benchmark_regeneration_report.md").write_text(markdown, encoding="utf-8")
