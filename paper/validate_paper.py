#!/usr/bin/env python3
"""Validate the final EchoForge IEEE paper package."""

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
    "phase_method_ladder.pdf",
    "ei_evolution_money_plot.pdf",
    "false_alarm_breakdown.pdf",
    "radar_positive_vs_false_positive.pdf",
    "appendix_radar_samples.png",
)
PREVIEW_PNGS = (
    "architecture_stack.png",
    "phase_method_ladder.png",
    "ei_evolution_money_plot.png",
    "false_alarm_breakdown.png",
    "radar_positive_vs_false_positive.png",
    "appendix_radar_samples.png",
)
LEGACY_MAIN_FIGURES = (
    "monte_carlo_split_flow",
    "kpi_ranking",
    "phase_kpi",
    "detector_ml_pipeline",
    "anchor_overlay",
    "ei_workflow",
    "data_processing_flow",
    "appendix_modeling_map",
    "iq_drone_samples",
    "iq_negative_samples",
)
REQUIRED_SECTIONS = (
    r"\\section\{Introduction\}",
    r"\\section\{Radar Processing Background\}",
    r"\\section\{EchoForge Simulator\}",
    r"\\section\{Benchmark Design\}",
    r"\\section\{Human Detection Ladder\}",
    r"\\section\{Engineered Intelligence\}",
    r"\\section\{Results\}",
    r"\\section\{False-Positive Burden\}",
    r"\\section\{Why It Matters\}",
    r"\\section\{Limitations\}",
    r"\\section\{Conclusion\}",
)
REQUIRED_METRIC_MACROS = (
    "MetricPriorLcb",
    "MetricEiLcb",
    "MetricLcbRelativePercent",
    "MetricPriorPointRecall",
    "MetricEiPointRecall",
    "MetricPointRecallRelativePercent",
    "MetricPriorAP",
    "MetricEiAP",
    "MetricAPRelativePercent",
    "MetricPriorSelectedFP",
    "MetricEiSelectedFP",
    "MetricSelectedFPReductionPercent",
    "MetricPriorRocAuc",
    "MetricEiRocAuc",
    "MetricPriorTPFPFN",
    "MetricEiTPFPFN",
)
FORBIDDEN_PAPER_PATTERNS = (
    r"NeverHumqn",
    r"world'?s most advanced",
    r"measured Iranian-platform radar signatures are claimed",
    r"claims? proprietary-equivalent",
    r"claims? classified fidelity",
    r"claims? operational performance",
    r"fielded-system equivalence is claimed",
    r"Recall@1\\%FPR",
    r"\+743\\%",
    r"\+186\\%",
    r"49 to 1",
    r"0\.699",
)
FORBIDDEN_MAIN_ARTIFACT_PATTERNS = (
    r"\.csv\b",
    r"\.json\b",
    r"outputs/",
    r"target/paper",
    r"selection_lock\.json",
    r"source_appendix_hashes",
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
MACRO_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\{\\(?P<name>[^}]+)\}\{(?P<value>[^}]*)\}"
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
    if not 35 <= len(keys) <= 80:
        fail(f"bibliography entry count {len(keys)} outside 35-80")
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


def includegraphics_files(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    names = [name.strip() for name in INCLUDEGRAPHICS_RE.findall(text)]
    if not names:
        fail("TeX source has no includegraphics references")
    return names


def validate_figures(figures_dir: Path) -> None:
    missing = [name for name in REQUIRED_FIGURES if not (figures_dir / name).is_file()]
    if missing:
        fail("missing required figure(s): " + ", ".join(missing))
    missing_previews = [name for name in PREVIEW_PNGS if not (figures_dir / name).is_file()]
    if missing_previews:
        fail("missing PNG preview(s): " + ", ".join(missing_previews))
    if (figures_dir / "appendix_radar_samples.pdf").exists():
        fail("appendix_radar_samples must remain raster-only")


def validate_includegraphics(tex_path: Path, figures_dir: Path) -> int:
    refs = includegraphics_files(tex_path)
    missing: list[str] = []
    names = [Path(ref).name for ref in refs]
    for required in REQUIRED_FIGURES:
        if required not in names:
            fail(f"TeX does not include required figure: {required}")
    for raw_name in refs:
        graphic_name = Path(raw_name).name
        suffix = Path(graphic_name).suffix.lower()
        if suffix not in {".pdf", ".png"}:
            fail(f"includegraphics must explicitly reference .pdf or .png: {raw_name}")
        if graphic_name != "appendix_radar_samples.png" and suffix != ".pdf":
            fail(f"main figure must be included as vector PDF: {raw_name}")
        if not (figures_dir / graphic_name).is_file():
            missing.append(raw_name)
    if missing:
        fail("missing includegraphics file(s): " + ", ".join(missing))
    for legacy in LEGACY_MAIN_FIGURES:
        if any(legacy in name for name in names):
            fail(f"TeX still includes legacy clutter figure: {legacy}")
    return len(refs)


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


def main_body_text(tex_path: Path) -> str:
    text = tex_path.read_text(encoding="utf-8")
    if "\\appendices" in text:
        text = text.split("\\appendices", 1)[0]
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    return text


def validate_generated_metric_macros(evidence_root: Path) -> None:
    macro_path = Path("target/paper/generated_metrics.tex")
    if not macro_path.exists():
        fail("missing generated metric macros: target/paper/generated_metrics.tex")
    text = macro_path.read_text(encoding="utf-8")
    macros = {match.group("name"): match.group("value") for match in MACRO_RE.finditer(text)}
    missing = [name for name in REQUIRED_METRIC_MACROS if name not in macros]
    if missing:
        fail("generated metric macros missing: " + ", ".join(missing))
    gain_rows = {}
    with (evidence_root / "main_kpi_gain_table.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gain_rows[row.get("metric", "")] = row
    expected_pairs = {
        "MetricPriorLcb": float(gain_rows["lcb95_recall_at_leq_1pct_fpr"]["prior_fusion_baseline"]),
        "MetricEiLcb": float(gain_rows["lcb95_recall_at_leq_1pct_fpr"]["ei_candidate"]),
        "MetricPriorPointRecall": float(
            gain_rows["point_recall_at_leq_1pct_fpr"]["prior_fusion_baseline"]
        ),
        "MetricEiPointRecall": float(gain_rows["point_recall_at_leq_1pct_fpr"]["ei_candidate"]),
        "MetricPriorAP": float(gain_rows["average_precision"]["prior_fusion_baseline"]),
        "MetricEiAP": float(gain_rows["average_precision"]["ei_candidate"]),
    }
    for name, expected in expected_pairs.items():
        try:
            actual = float(macros[name])
        except ValueError:
            fail(f"generated macro {name} is not numeric: {macros[name]}")
        if abs(actual - expected) > 0.001:
            fail(f"generated macro {name}={actual} does not match evidence {expected}")


def validate_paper_text(tex_path: Path) -> None:
    text = expanded_tex_text(tex_path)
    body = main_body_text(tex_path)
    missing_sections = [pattern for pattern in REQUIRED_SECTIONS if not re.search(pattern, text)]
    if missing_sections:
        fail("paper is missing final narrative section(s): " + ", ".join(missing_sections))
    section_positions = []
    for pattern in REQUIRED_SECTIONS:
        match = re.search(pattern, text)
        section_positions.append(match.start() if match else -1)
    if section_positions != sorted(section_positions):
        fail("final narrative sections are not in the required order")
    forbidden = [
        pattern for pattern in FORBIDDEN_PAPER_PATTERNS if re.search(pattern, text, re.IGNORECASE)
    ]
    if forbidden:
        fail(
            "paper text contains forbidden stale or overclaiming language: " + ", ".join(forbidden)
        )
    main_artifacts = [
        pattern
        for pattern in FORBIDDEN_MAIN_ARTIFACT_PATTERNS
        if re.search(pattern, body, re.IGNORECASE)
    ]
    if main_artifacts:
        fail("main body contains reader-facing artifact/file clutter: " + ", ".join(main_artifacts))
    required_phrases = (
        "strict-open",
        "fixed-wing pusher-prop public proxy",
        "Human Detection Ladder",
        "s_{ij}=g_j(x_i)",
        "w_j\\geq0",
        "0.25",
        "no holdout optimization curve",
        "False-Positive Burden",
        "Runtime Code Appendix",
        "not measured imagery",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    if missing_phrases:
        fail("paper text is missing required final-paper phrase(s): " + ", ".join(missing_phrases))
    for legacy in LEGACY_MAIN_FIGURES:
        if legacy in text:
            fail(f"paper text still references legacy clutter figure: {legacy}")
    if "LCB95 Recall@$\\leq$1\\%FPR" not in text:
        fail("paper text is missing the exact LCB95 Recall@<=1%FPR label")
    if re.search(r"(?<![A-Za-z0-9_/])n/a(?![A-Za-z0-9_/])", text, re.IGNORECASE):
        fail("paper text still contains unresolved n/a values")


def validate_source_appendix(evidence_root: Path) -> None:
    source_tex = Path("target/paper/source_appendix/source_code_appendix.tex")
    source_meta = Path("target/paper/source_appendix/source_appendix_metadata.json")
    if not source_tex.exists() or not source_meta.exists():
        fail("missing generated source appendix output under target/paper/source_appendix")
    source_text = source_tex.read_text(encoding="utf-8")
    for banned in (
        "Source:",
        "detection/",
        "_write_json",
        "_write_csv",
        "report builder",
        "run_advanced_main_run_detectors",
    ):
        if banned in source_text:
            fail(f"source appendix contains banned path/plumbing marker: {banned}")
    if "Runtime Code Appendix" not in source_text:
        fail("generated appendix is missing Runtime Code Appendix heading")
    try:
        metadata = json.loads(source_meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"source appendix metadata is invalid JSON: {exc}")
    listings = metadata.get("listings", [])
    if not isinstance(listings, list) or not listings:
        fail("source appendix metadata is missing listings")
    with (evidence_root / "source_appendix_hashes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        hash_rows = list(csv.DictReader(handle))
    metadata_index = {
        (
            str(item.get("block_id", "")),
            str(item.get("path", "")),
            str(item.get("origin", "")),
            str(item.get("symbol", "")),
            str(item.get("line_start", "")),
            str(item.get("line_end", "")),
            str(item.get("sha256", "")),
        )
        for item in listings
        if isinstance(item, dict)
    }
    csv_index = {
        (
            str(row.get("block_id", "")),
            str(row.get("path", "")),
            str(row.get("origin", "")),
            str(row.get("symbols", "")),
            str(row.get("line_start", "")),
            str(row.get("line_end", "")),
            str(row.get("sha256", "")),
        )
        for row in hash_rows
        if row.get("status") == "included"
    }
    if metadata_index != csv_index:
        fail("source appendix metadata does not match source appendix hash rows")


def validate_paper_evidence(evidence_root: Path) -> None:
    manifest_path = evidence_root / "paper_evidence_manifest.json"
    if not manifest_path.exists():
        fail(f"missing paper evidence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        fail("paper evidence manifest must be a JSON object")
    if manifest.get("version") != "tier1-final":
        fail("paper evidence manifest version must be tier1-final")
    required_manifest_keys = (
        "primary_kpi_rows",
        "main_kpi_gain_rows",
        "phase_method_ladder_rows",
        "ei_evolution_summary",
        "source_appendix_hashes",
        "false_alarm_by_method_family",
        "selected_threshold_confusion_matrix",
        "public_proxy_positive_class_card",
        "environment_impairment_model_rows",
        "detector_processing_baseline_rows",
        "fusion_baseline_rows",
        "ei_objective_rows",
    )
    missing = [key for key in required_manifest_keys if not manifest.get(key)]
    if missing:
        fail("paper evidence manifest missing key(s): " + ", ".join(missing))
    required_csv = (
        "main_kpi_gain_table.csv",
        "phase_method_ladder_rows.csv",
        "ei_evolution_trace.csv",
        "false_alarm_by_method_family.csv",
        "selected_threshold_confusion_matrix.csv",
        "source_appendix_hashes.csv",
    )
    missing_csv = [name for name in required_csv if not (evidence_root / name).exists()]
    if missing_csv:
        fail("missing paper evidence csv(s): " + ", ".join(missing_csv))
    summary = json.loads((evidence_root / "ei_evolution_summary.json").read_text(encoding="utf-8"))
    if not summary.get("true_evaluation_order_available"):
        fail("EI evolution summary must use true evaluation order")
    if summary.get("trace_basis") != "true_evaluation_order":
        fail("EI evolution trace basis must be true_evaluation_order")
    with (evidence_root / "ei_evolution_trace.csv").open(encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    selected_rows = [row for row in trace_rows if row.get("selected_by_cv") == "True"]
    if len(selected_rows) != 1:
        fail(f"EI evolution trace must mark exactly one selected row, got {len(selected_rows)}")
    if any(row.get("selection_split") != "train_cv" for row in trace_rows):
        fail("EI evolution trace contains non-train/CV selection rows")
    if any(row.get("holdout_rows_used_for_selection") not in {"0", 0} for row in trace_rows):
        fail("EI evolution trace uses holdout rows for selection")
    with (evidence_root / "false_alarm_by_method_family.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        false_alarm_rows = list(csv.DictReader(handle))
    families = {row.get("family") for row in false_alarm_rows}
    required_families = {
        "single_bird",
        "bird_flock",
        "rc_fixed_wing",
        "weather_cell",
        "clutter_only_counterfactual",
        "rfi_burst",
        "multipath_ghost",
        "terrain_glint",
        "ground_vehicle",
    }
    missing_families = sorted(required_families - families)
    if missing_families:
        fail("false-alarm family table missing family/families: " + ", ".join(missing_families))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-evidence-root",
        type=Path,
        default=Path("outputs/paper-evidence/tier1-final"),
    )
    args = parser.parse_args()

    validate_figures(args.figures_dir)
    figure_refs = validate_includegraphics(args.tex, args.figures_dir)
    validate_generated_metric_macros(args.paper_evidence_root)
    validate_paper_text(args.tex)
    validate_paper_evidence(args.paper_evidence_root)
    validate_source_appendix(args.paper_evidence_root)
    bib = bib_keys(args.bib)
    cites = citation_keys(args.tex)
    missing_cites = sorted(cites - bib)
    if missing_cites:
        fail("missing BibTeX entries for citation(s): " + ", ".join(missing_cites))

    pages = pdf_page_count(args.pdf)
    if not 8 <= pages <= 40:
        fail(f"page count {pages} outside 8-40")

    print(
        "paper validation passed: "
        f"pages={pages} bib_entries={len(bib)} citations={len(cites)} "
        f"figures={len(REQUIRED_FIGURES)} includegraphics={figure_refs}"
    )


if __name__ == "__main__":
    main()
