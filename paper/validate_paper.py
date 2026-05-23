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
    "architecture_stack.pdf",
    "monte_carlo_split_flow.pdf",
    "kpi_ranking.pdf",
    "phase_kpi.pdf",
    "iq_drone_samples.pdf",
    "iq_negative_samples.png",
    "detector_ml_pipeline.pdf",
    "anchor_overlay.pdf",
    "ei_workflow.pdf",
    "appendix_modeling_map.pdf",
)
PREVIEW_PNGS = (
    "architecture_stack.png",
    "monte_carlo_split_flow.png",
    "kpi_ranking.png",
    "phase_kpi.png",
    "iq_drone_samples.png",
    "detector_ml_pipeline.png",
    "anchor_overlay.png",
    "ei_workflow.png",
    "appendix_modeling_map.png",
)
LEGACY_FIGURES = ("locked_algorithm.pdf", "locked_algorithm.png")

FORBIDDEN_FIGURE_METADATA = (b"fallback=", b"fallback source:")
FORBIDDEN_PAPER_PATTERNS = (
    r"NeverHumqn",
    r"Table-IV-compatible",
    r"Engineered Intelligence Transparency Summary",
    r"Main Experiment Lanes",
    r"Primary KPI Selection for the EI Holdout Evaluation",
    r"\+743\\%",
    r"\+186\\%",
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
FORBIDDEN_READER_TERMS = (
    r"\blocked candidate\b",
    r"\bselected candidate\b",
    r"\bselected AP\b",
)
PRIMARY_KPI_PATTERN = r"LCB95 Recall@\$\\leq\$1\\%FPR|lower 95\\% group-block bootstrap bound of recall at FPR <= 1\\%"
REQUIRED_GAIN_PATTERNS = (
    r"NeverHuman Research Group",
    r"\\kpigain\{",
    r"\+61\.6",
    r"\+54\.1",
    r"8\.42",
    r"2\.85",
    r"\+742\\%",
    r"\+185\\%",
    r"49 to 1",
    r"98\.0",
    r"EI gain vs prior",
)
REQUIRED_AP_GAIN_PATTERN = r"\+545\\%|\+544\\%"
REQUIRED_COMPARATOR_PATTERN = (
    r"human-engineered prior fusion|accepted human-engineered prior fusion|best-practice comparator"
)
REQUIRED_FIGURE_PHRASES = (
    r"1\\% FPR operating cap",
    r"false-alarm family legend",
    r"near-threshold",
)
REQUIRED_WORLDCLASS_PATTERNS = (
    r"Core Experiment Roadmap",
    r"Main KPI Gain Ledger Versus Accepted Prior Fusion",
    r"Rich Public-Proxy Modeling Appendix",
    r"Fixed-Wing Pusher-Prop / Iranian-Drone Public-Proxy Modeling Card",
    r"Noise, Clutter, RFI, and Receiver Modeling Details",
    r"Detector-View Modeling Contract",
    r"main(?:\\_|\_)kpi(?:\\_|\_)gain(?:\\_|\_)table\.csv",
    r"public(?:\\_|\_)proxy(?:\\_|\_)model(?:\\_|\_)detail(?:\\_|\_)rows\.csv",
    r"environment(?:\\_|\_)impairment(?:\\_|\_)model(?:\\_|\_)rows\.csv",
    r"no measured Iranian-drone radar signature",
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
INPUT_RE = re.compile(r"\\input\s*\{([^}]*)\}", re.MULTILINE)


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
    missing_previews = [name for name in PREVIEW_PNGS if not (figures_dir / name).is_file()]
    if missing_previews:
        fail("missing vector preview PNG(s): " + ", ".join(missing_previews))
    if (figures_dir / "iq_negative_samples.pdf").exists():
        fail("iq_negative_samples must remain raster-only")
    legacy = [name for name in LEGACY_FIGURES if (figures_dir / name).exists()]
    if legacy:
        fail("legacy locked_algorithm figure(s) must be renamed: " + ", ".join(legacy))
    flagged: list[str] = []
    for name in REQUIRED_FIGURES + PREVIEW_PNGS:
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


def expanded_tex_text(tex_path: Path) -> str:
    text = tex_path.read_text(encoding="utf-8")
    chunks = [text]
    repo_root = tex_path.resolve().parents[1]
    for raw_name in INPUT_RE.findall(text):
        input_path = Path(raw_name)
        candidates = [
            input_path,
            repo_root / input_path,
            tex_path.parent / input_path,
            tex_path.parent / f"{raw_name}.tex",
            repo_root / f"{raw_name}.tex",
        ]
        for candidate in candidates:
            if candidate.is_file():
                chunks.append(candidate.read_text(encoding="utf-8"))
                break
    return "\n".join(chunks)


def validate_includegraphics(tex_path: Path, figures_dir: Path) -> int:
    missing: list[str] = []
    for raw_name in includegraphics_files(tex_path):
        graphic_name = Path(raw_name).name
        suffix = Path(graphic_name).suffix.lower()
        if suffix not in {".pdf", ".png"}:
            fail(f"includegraphics must explicitly reference .pdf or .png: {raw_name}")
        if graphic_name == "iq_negative_samples.pdf":
            fail("TeX must not include iq_negative_samples.pdf")
        if graphic_name != "iq_negative_samples.png" and suffix != ".pdf":
            fail(f"non-heatmap figure must be included as vector PDF: {raw_name}")
        candidates = [figures_dir / graphic_name]
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(raw_name)
    if missing:
        fail("missing includegraphics file(s): " + ", ".join(missing))
    return len(includegraphics_files(tex_path))


def validate_paper_text(tex_path: Path) -> None:
    text = expanded_tex_text(tex_path)
    missing_patterns = [pattern for pattern in FORBIDDEN_PAPER_PATTERNS if re.search(pattern, text)]
    if missing_patterns:
        fail("paper text still contains stale v1 evidence markers: " + ", ".join(missing_patterns))
    reader_terms = [
        pattern
        for pattern in FORBIDDEN_READER_TERMS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if reader_terms:
        fail("paper text contains banned reader-facing EI terminology: " + ", ".join(reader_terms))
    if not re.search(PRIMARY_KPI_PATTERN, text):
        fail("paper text is missing the primary KPI statement")
    missing_gain_patterns = [
        pattern for pattern in REQUIRED_GAIN_PATTERNS if not re.search(pattern, text)
    ]
    if missing_gain_patterns:
        fail("paper text is missing gain/result phrasing: " + ", ".join(missing_gain_patterns))
    if not re.search(REQUIRED_AP_GAIN_PATTERN, text):
        fail("paper text is missing the AP gain phrasing")
    if not re.search(REQUIRED_COMPARATOR_PATTERN, text, flags=re.IGNORECASE):
        fail("paper text is missing the human-engineered prior fusion comparator language")
    missing_figure_phrases = [
        pattern
        for pattern in REQUIRED_FIGURE_PHRASES
        if not re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if missing_figure_phrases:
        fail("paper text is missing figure phrasing: " + ", ".join(missing_figure_phrases))
    if "Engineered Intelligence" not in text:
        fail("paper text is missing the Engineered Intelligence section")
    if "Fixed-Wing Pusher-Prop Public-Proxy Appendix" not in text:
        fail("paper text is missing the fixed-wing public-proxy appendix")
    if "Regional Bird and RC Hard-Negative Appendix" not in text:
        fail("paper text is missing the regional bird appendix")
    if "Source, Noise, Detector, and Assumption Appendix" not in text:
        fail("paper text is missing the source/noise/detector appendix")
    if "LCB95 Recall@$\\leq$1\\%FPR" not in text:
        fail("paper text is missing the exact LCB95 Recall@<=1%FPR label")
    if "n/a" in text:
        fail("paper text still contains unresolved n/a values")
    missing_worldclass = [
        pattern
        for pattern in REQUIRED_WORLDCLASS_PATTERNS
        if not re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if missing_worldclass:
        fail(
            "paper text is missing world-class appendix/roadmap markers: "
            + ", ".join(missing_worldclass)
        )


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
    if manifest.get("version") != "major-upgrade-v2":
        fail("paper evidence manifest version must be major-upgrade-v2")
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
        "main_kpi_gain_rows",
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
        "public_proxy_model_detail_rows",
        "environment_impairment_model_rows",
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
        "main_kpi_gain_table.csv",
        "public_proxy_model_detail_rows.csv",
        "environment_impairment_model_rows.csv",
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
    if not 8 <= pages <= 18:
        fail(f"page count {pages} outside 8-18")

    print(
        "paper validation passed: "
        f"pages={pages} bib_entries={len(bib)} citations={len(cites)} "
        f"figures={len(REQUIRED_FIGURES)} includegraphics={figure_refs}"
    )


if __name__ == "__main__":
    main()
