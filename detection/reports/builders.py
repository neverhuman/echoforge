"""Markdown and JSON report assembly helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from detection.pipeline_contracts.core import write_json


def whitepaper_trace(title: str, spec_id: str, reference_lines: Sequence[str], artifacts: Sequence[str]) -> str:
    lines = [f"# {title}", "", f"- pipeline: `{spec_id}`", "- evidence type: synthetic public-proxy benchmark", ""]
    lines.extend(f"- reference: {item}" for item in reference_lines)
    lines.append("")
    lines.extend(f"- artifact: {artifact}" for artifact in artifacts)
    lines.append("")
    lines.append("Limitation: public-proxy synthetic evidence only; not measured truth; not proprietary-equivalent.")
    return "\n".join(lines) + "\n"
