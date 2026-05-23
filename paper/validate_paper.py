#!/usr/bin/env python3
"""Validate the EchoForge IEEE paper package."""

from __future__ import annotations

import argparse
import csv
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
VECTOR_FIGURES = (
    "architecture_stack.pdf",
    "monte_carlo_split_flow.pdf",
    "kpi_ranking.pdf",
    "phase_kpi.pdf",
    "iq_drone_samples.pdf",
    "detector_ml_pipeline.pdf",
    "locked_algorithm.pdf",
)

FORBIDDEN_FIGURE_METADATA = (b"fallback=", b"fallback source:")
FORBIDDEN_PAPER_PATTERNS = (
    r"250 positive groups",
    r"38 positive groups",
    r"114 are positive",
    r"114 positive phase rows",
    r"positive phase records, 250 in each phase",
    r"canary\s+fail",
    r"Recall@1\\%FPR",
    r"sensor\s+parity",
    r"operational\s+parity",
    r"matches\s+classified\s+fidelity",
    r"matches\s+proprietary-equivalent\s+behavior",
)
PRIMARY_KPI_PATTERN = r"LCB95 Recall@\$\\leq\$1\\%FPR|lower 95\\% group-block bootstrap bound of recall at FPR <= 1\\%"

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
    if not 30 <= len(keys) <= 50:
        fail(f"bibliography entry count {len(keys)} outside 30-50")
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
    missing_vectors = [name for name in VECTOR_FIGURES if not (figures_dir / name).is_file()]
    if missing_vectors:
        fail("missing vector figure(s): " + ", ".join(missing_vectors))
    if (figures_dir / "iq_negative_samples.pdf").exists():
        fail("iq_negative_samples must remain raster-only")
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
    if "Engineered Intelligence" not in text:
        fail("paper text is missing the Engineered Intelligence section")
    if "Fixed-Wing Pusher-Prop Public-Proxy Appendix" not in text:
        fail("paper text is missing the fixed-wing public-proxy appendix")
    if "Regional Bird and RC Hard-Negative Appendix" not in text:
        fail("paper text is missing the regional bird appendix")
    if "LCB95 Recall@$\\leq$1\\%FPR" not in text:
        fail("paper text is missing the exact LCB95 Recall@<=1%FPR label")
    if "n/a" in text:
        fail("paper text still contains unresolved n/a values")


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
    for key in (
        "sensor_archetype_cards",
        "regional_hard_negative_taxonomy",
        "modality_transparency",
        "normalized_anchor_comparison",
        "primary_kpi_rows",
    ):
        value = manifest.get(key)
        if not value:
            fail(f"paper evidence manifest is missing {key}")
    for key in (
        "selected_threshold_confusion_matrix",
        "false_alarm_by_method_family",
        "top_method_false_positive_frequency",
        "regional_bird_library",
        "source_pack_bird_coverage_summary",
        "engineered_intelligence",
        "public_proxy_positive_class_card",
    ):
        value = manifest.get(key)
        if not value:
            fail(f"paper evidence manifest is missing {key}")
    modality = manifest.get("modality_transparency", {})
    modality_rows = modality.get("modality_rows", []) if isinstance(modality, dict) else []
    required_views = {
        "radar_only",
        "acoustic_only",
        "passive_rf_only",
        "radar_acoustic",
        "radar_rf",
        "full_fusion",
    }
    present_views = {row.get("view") for row in modality_rows if isinstance(row, dict)}
    missing_views = sorted(required_views - present_views)
    if missing_views:
        fail("paper evidence modality transparency missing view(s): " + ", ".join(missing_views))
    normalized = manifest.get("normalized_anchor_comparison", [])
    if not isinstance(normalized, list) or not normalized:
        fail("paper evidence KTH anchor comparison is not normalized")
    if any(
        "unitless" not in str(row.get("unitless_basis", ""))
        for row in normalized
        if isinstance(row, dict)
    ):
        fail("paper evidence KTH anchor comparison lacks unitless basis markers")
    false_alarm_path = evidence_root / "false_alarm_family_breakdown.csv"
    if not false_alarm_path.exists():
        fail(f"missing false-alarm family table: {false_alarm_path}")
    with false_alarm_path.open(encoding="utf-8", newline="") as handle:
        false_alarm_rows = list(csv.DictReader(handle))
    if any(row.get("family") == "positive" for row in false_alarm_rows):
        fail("false-alarm family table includes positive rows")
    for column in ("near_threshold_count", "near_threshold_rate", "p95_score"):
        if false_alarm_rows and column not in false_alarm_rows[0]:
            fail(f"false-alarm family table is missing {column}")
    for required_csv in (
        "selected_threshold_confusion_matrix.csv",
        "false_alarm_by_method_family.csv",
        "top_method_false_positive_frequency.csv",
        "regional_bird_library.csv",
        "source_pack_bird_coverage_summary.csv",
        "engineered_intelligence_transparency.csv",
    ):
        if not (evidence_root / required_csv).exists():
            fail(f"missing paper evidence csv: {required_csv}")
    if not (evidence_root / "public_proxy_positive_class_card.json").exists():
        fail("missing public proxy positive-class card JSON")


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
