#!/usr/bin/env python3
"""Validate the EchoForge IEEE paper package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_FIGURES = (
    "architecture_stack.png",
    "monte_carlo_split_flow.png",
    "kpi_ranking.png",
    "phase_kpi.png",
    "iq_drone_samples.png",
    "iq_negative_samples.png",
    "detector_ml_pipeline.png",
    "locked_algorithm.png",
)

FORBIDDEN_FIGURE_METADATA = (b"fallback=", b"fallback source:")
FORBIDDEN_PAPER_PATTERNS = (
    r"250 positive groups",
    r"38 positive groups",
    r"114 are positive",
    r"114 positive phase rows",
    r"positive phase records, 250 in each phase",
)
PRIMARY_KPI_PATTERN = (
    r"LCB95 Recall@1\\%FPR|lower 95\\% group-block bootstrap bound of recall at FPR <= 1\\%"
)

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear|nocite)"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}",
    re.MULTILINE,
)
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]*)\}",
    re.MULTILINE,
)


def fail(message: str) -> None:
    print(f"paper validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def pdf_page_count(pdf_path: Path) -> int:
    if not pdf_path.exists():
        fail(f"missing PDF: {pdf_path}")
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        fail("missing pdfinfo; install poppler-utils")
    except subprocess.CalledProcessError as exc:
        fail(f"pdfinfo failed: {exc.stderr.strip() or exc.stdout.strip()}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        fail("could not read page count from pdfinfo output")
    return int(match.group(1))


def bib_keys(bib_path: Path) -> set[str]:
    if not bib_path.exists():
        fail(f"missing bibliography: {bib_path}")
    text = bib_path.read_text(encoding="utf-8")
    keys = set(BIB_ENTRY_RE.findall(text))
    if not 30 <= len(keys) <= 40:
        fail(f"bibliography entry count {len(keys)} outside 30-40")
    return keys


def citation_keys(tex_path: Path) -> set[str]:
    if not tex_path.exists():
        fail(f"missing TeX source: {tex_path}")
    text = tex_path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for group in CITE_RE.findall(text):
        for key in group.split(","):
            clean = key.strip()
            if clean and clean != "*":
                keys.add(clean)
    if not keys:
        fail("TeX source has no citations")
    return keys


def validate_figures(figures_dir: Path) -> None:
    missing = [name for name in REQUIRED_FIGURES if not (figures_dir / name).is_file()]
    if missing:
        fail("missing required figure(s): " + ", ".join(missing))
    flagged: list[str] = []
    for name in REQUIRED_FIGURES:
        payload = (figures_dir / name).read_bytes()
        if any(marker in payload for marker in FORBIDDEN_FIGURE_METADATA):
            flagged.append(name)
    if flagged:
        fail("figure metadata contains fallback markers: " + ", ".join(flagged))


def includegraphics_files(tex_path: Path) -> list[str]:
    if not tex_path.exists():
        fail(f"missing TeX source: {tex_path}")
    text = tex_path.read_text(encoding="utf-8")
    names = [name.strip() for name in INCLUDEGRAPHICS_RE.findall(text)]
    if not names:
        fail("TeX source has no includegraphics references")
    return names


def validate_includegraphics(tex_path: Path, figures_dir: Path) -> int:
    missing: list[str] = []
    for raw_name in includegraphics_files(tex_path):
        graphic_name = Path(raw_name).name
        candidates = [figures_dir / graphic_name]
        if not Path(graphic_name).suffix:
            candidates.extend(
                figures_dir / f"{graphic_name}{suffix}"
                for suffix in (".png", ".pdf", ".jpg", ".jpeg")
            )
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(raw_name)
    if missing:
        fail("missing includegraphics file(s): " + ", ".join(missing))
    return len(includegraphics_files(tex_path))


def validate_paper_text(tex_path: Path) -> None:
    text = tex_path.read_text(encoding="utf-8")
    missing_patterns = [pattern for pattern in FORBIDDEN_PAPER_PATTERNS if re.search(pattern, text)]
    if missing_patterns:
        fail("paper text still contains stale v1 evidence markers: " + ", ".join(missing_patterns))
    if not re.search(PRIMARY_KPI_PATTERN, text):
        fail("paper text is missing the primary KPI statement")


def validate_paper_evidence(evidence_root: Path) -> None:
    manifest_path = evidence_root / "paper_evidence_manifest.json"
    if not manifest_path.exists():
        fail(f"missing paper evidence manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"paper evidence manifest is invalid JSON: {exc}")
    if not isinstance(manifest, dict):
        fail("paper evidence manifest must be a JSON object")
    generative = manifest.get("generative_origin_audit", {})
    if not isinstance(generative, dict):
        fail("paper evidence manifest is missing generative-origin audit data")
    percentage = generative.get("percentage_generative_loc")
    if percentage is None:
        fail("paper evidence manifest is missing generative-origin percentage")
    try:
        percentage_value = float(percentage)
    except (TypeError, ValueError):
        fail("paper evidence manifest has an invalid generative-origin percentage")
    if not 0.0 <= percentage_value <= 100.0:
        fail("paper evidence manifest has an out-of-range generative-origin percentage")
    feedback = manifest.get("feedback_coverage", {})
    if not isinstance(feedback, dict) or feedback.get("status") != "pass":
        fail("paper evidence manifest is missing feedback coverage status")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-evidence-root",
        type=Path,
        default=Path("outputs/paper-evidence/major-upgrade-v1"),
    )
    args = parser.parse_args()

    validate_figures(args.figures_dir)
    figure_refs = validate_includegraphics(args.tex, args.figures_dir)
    validate_paper_text(args.tex)
    validate_paper_evidence(args.paper_evidence_root)
    bib = bib_keys(args.bib)
    cites = citation_keys(args.tex)
    missing_cites = sorted(cites - bib)
    if missing_cites:
        fail("missing BibTeX entries for citation(s): " + ", ".join(missing_cites))

    pages = pdf_page_count(args.pdf)
    if not 8 <= pages <= 14:
        fail(f"page count {pages} outside 8-14")

    print(
        "paper validation passed: "
        f"pages={pages} bib_entries={len(bib)} citations={len(cites)} "
        f"figures={len(REQUIRED_FIGURES)} includegraphics={figure_refs}"
    )


if __name__ == "__main__":
    main()
