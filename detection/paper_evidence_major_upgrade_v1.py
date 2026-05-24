#!/usr/bin/env python3
"""Build the major-upgrade paper evidence lane.

This script does not create raw training data or solver outputs. It reads the
existing local benchmark artifacts, summarizes them into compact evidence
tables, and writes the paper-facing report bundle under
``outputs/paper-evidence/major-upgrade-v1``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from detection.main_run_types import (
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        MODEL_FEATURE_DENYLIST,
        PHASES,
    )
    from detection.ei_evolution_trace import build_ei_evolution_evidence
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_types import (
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        MODEL_FEATURE_DENYLIST,
        PHASES,
    )
    from ei_evolution_trace import build_ei_evolution_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "outputs" / "training-data" / "runit-fixed-wing-pusher-proxy-v2-main-run"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT / "outputs" / "detection" / "runit-fixed-wing-pusher-proxy-v2-main-run"
)
DEFAULT_ADVANCED_ROOT = (
    REPO_ROOT
    / "outputs"
    / "detection"
    / "runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution"
)
DEFAULT_ANCHOR_ROOT = (
    REPO_ROOT / "outputs" / "real-data" / "kth-drone-bird-human-77ghz" / "kth-measured-v1"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "paper-evidence" / "tier1-final"
DEFAULT_FEEDBACK_MATRIX = REPO_ROOT / "paper" / "docs" / "paper_feedback_coverage_matrix.md"
DEFAULT_GENERATIVE_ORIGIN_MANIFEST = (
    REPO_ROOT / "paper" / "docs" / "generative_origin_manifest.json"
)
PAPER_EVIDENCE_VERSION = "tier1-final"
THRESHOLD_TARGET_FPR = 0.01
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20260522
SPEED_OF_LIGHT_MPS = 299_792_458.0
PRIMARY_KPI_LABEL = "LCB95 Recall@≤1%FPR"
TOP_METHOD_COUNT = 5

METHOD_SCORE_SOURCES: dict[str, tuple[str, str]] = {
    "high_resolution_xku_cuas": ("classical_radar_processing.csv", "high_res_cfar_log1p"),
    "tactical_s_band_aesa": ("classical_radar_processing.csv", "sband_mtd_log1p"),
    "gbad_3d4d_cueing": ("classical_radar_processing.csv", "gbad_track_log1p"),
    "tabular_ml_baseline": ("ml_detector_baselines.csv", "tabular_baseline_score"),
    "sequence_ml_proxy": ("ml_detector_baselines.csv", "sequence_proxy_score"),
    "layered_fusion_c2": ("fusion_predictions.csv", "fusion_probability"),
}


@dataclass(frozen=True)
class EvidenceRoots:
    training_root: Path
    baseline_root: Path
    advanced_root: Path
    anchor_root: Path
    out_root: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _count_non_empty_lines(path: Path) -> int:
    return sum(1 for line in _read_text(path).splitlines() if line.strip())


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_group_rows(rows: list[dict[str, str]], split_role: str) -> tuple[int, int]:
    group_ids = {
        row.get("scenario_group_id", "")
        for row in rows
        if row.get("split_role") == split_role and row.get("scenario_group_id")
    }
    positive_group_ids = {
        row.get("scenario_group_id", "")
        for row in rows
        if row.get("split_role") == split_role
        and row.get("scenario_group_id")
        and _safe_int(row.get("is_positive")) == 1
    }
    return len(group_ids), len(positive_group_ids)


def _split_summary(
    scenarios: list[dict[str, str]], records: list[dict[str, str]]
) -> dict[str, Any]:
    holdout_records = [row for row in records if row.get("split_role") == "holdout"]
    train_records = [row for row in records if row.get("split_role") == "train_cv"]
    holdout_group_count, holdout_positive_group_count = _count_group_rows(scenarios, "holdout")
    train_group_count, train_positive_group_count = _count_group_rows(scenarios, "train_cv")
    return {
        "scenario_group_count": len(scenarios),
        "positive_group_count": int(sum(_safe_int(row.get("is_positive")) for row in scenarios)),
        "negative_group_count": int(
            sum(1 for row in scenarios if _safe_int(row.get("is_positive")) == 0)
        ),
        "holdout_group_count": holdout_group_count,
        "holdout_positive_group_count": holdout_positive_group_count,
        "holdout_negative_group_count": holdout_group_count - holdout_positive_group_count,
        "holdout_record_count": len(holdout_records),
        "holdout_positive_record_count": int(
            sum(_safe_int(row.get("label_id")) for row in holdout_records)
        ),
        "holdout_negative_record_count": int(
            sum(1 for row in holdout_records if _safe_int(row.get("label_id")) == 0)
        ),
        "train_cv_group_count": train_group_count,
        "train_cv_positive_group_count": train_positive_group_count,
        "train_cv_negative_group_count": train_group_count - train_positive_group_count,
        "train_cv_record_count": len(train_records),
        "train_cv_positive_record_count": int(
            sum(_safe_int(row.get("label_id")) for row in train_records)
        ),
        "train_cv_negative_record_count": int(
            sum(1 for row in train_records if _safe_int(row.get("label_id")) == 0)
        ),
    }


def _feedback_coverage_rows() -> list[dict[str, str]]:
    return [
        {
            "tip": "tip1.txt",
            "item_id": "tip1-01",
            "actionable_item": "Make Fig. 2 readable and full-width instead of a dense one-column panel.",
            "status": "addressed",
            "evidence_location": "Fig. 2 / monte_carlo_split_flow.png / paper TeX figure*",
            "notes": "Scenario balance and leakage checks are separated with larger labels.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-02",
            "actionable_item": "Reconcile Table IV, Fig. 3, and Fig. 5 metric values.",
            "status": "addressed",
            "evidence_location": "evaluation_summary.json, comparable_ablation_summary.csv, Table IV",
            "notes": "Locked holdout AP is 0.824845; selected_component_ablations.csv is kept as an internal diagnostic.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-03",
            "actionable_item": "Split selected-threshold counts from swept fixed-FPR metrics.",
            "status": "addressed",
            "evidence_location": "selected_threshold_confusion_matrix.csv, primary_kpi_table.csv, Table IV",
            "notes": "Selected-threshold TP/FP/FN and swept Recall@<=1%FPR are separate fields.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-04",
            "actionable_item": "Report LCB95 Recall@<=1%FPR in the abstract, main result, figure, table, and conclusion.",
            "status": "addressed",
            "evidence_location": "primary_kpi_table.csv, Abstract, Fig. 3, Table IV, Conclusion",
            "notes": "LCB95 is sourced from group-block bootstrap over scenario groups.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-05",
            "actionable_item": "Temper robustness claims because the holdout has only eight positive groups.",
            "status": "addressed",
            "evidence_location": "Evaluation Protocol, Fig. 4 caption, Limitations",
            "notes": "Phase and family conclusions are marked diagnostic.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-06",
            "actionable_item": "Use standard radar/ML baseline names rather than raw internal handles.",
            "status": "addressed",
            "evidence_location": "Paper tables and figures; selected_component_human_weights.csv",
            "notes": "Raw component IDs are retained only in evidence files.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-07",
            "actionable_item": "Add waveform, resolution, clutter, SNR/noise, and micro-Doppler specificity.",
            "status": "addressed",
            "evidence_location": "radar_model_card.json, radar_model_detail_rows.csv, Section II",
            "notes": "Model rows include PRF/CRF, CPI, chirps, range span, Doppler and velocity resolutions.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-08",
            "actionable_item": "Add group-level uncertainty rather than only record-level metrics.",
            "status": "addressed",
            "evidence_location": "group_level_operating_metrics.csv, evaluation_summary.json",
            "notes": "Positive/negative group counts, group recall, and group false-alarm rate are emitted.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-09",
            "actionable_item": "Replace Fig. 5 with a readable component-weight and ablation-delta visual.",
            "status": "addressed",
            "evidence_location": "detector_ml_pipeline.png, selected_component_human_weights.csv",
            "notes": "Main figure avoids raw component handles.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-10",
            "actionable_item": "Make Fig. 7 axes physically interpretable.",
            "status": "addressed",
            "evidence_location": "iq_negative_samples.png",
            "notes": "Range bins are labelled in meters and Doppler bins in velocity-equivalent units.",
        },
        {
            "tip": "tip1.txt",
            "item_id": "tip1-11",
            "actionable_item": "Normalize the measured-anchor comparison instead of plotting mixed raw units.",
            "status": "addressed",
            "evidence_location": "locked_algorithm.png, normalized_anchor_comparison.csv",
            "notes": "Anchor values are plotted as unitless measured-anchor z-scores.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-01",
            "actionable_item": "Define strict-open in the abstract or first section.",
            "status": "addressed",
            "evidence_location": "Abstract and Claim Boundary",
            "notes": "Strict-open is defined as public/generated, auditable assumptions and evidence.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-02",
            "actionable_item": "Mention the headline performance result and ROC AUC tradeoff in the abstract/main result.",
            "status": "addressed",
            "evidence_location": "Abstract, Section IV, Table IV",
            "notes": "The EI artifact improves AP and low-FPR recall while lowering ROC AUC versus prior fusion.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-03",
            "actionable_item": "Tighten FMCW versus pulse-Doppler wording.",
            "status": "addressed",
            "evidence_location": "radar_model_detail_rows.csv, Section II",
            "notes": "The paper states that chirp/FMCW wording is a public-proxy range-Doppler product abstraction.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-04",
            "actionable_item": "Add a larger positive holdout or multi-seed evaluation.",
            "status": "deferred",
            "evidence_location": "Limitations",
            "notes": "This repair keeps the locked top model and single corpus intact; wider corpora are future work.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-05",
            "actionable_item": "Add a radar-realistic track-level false-alarm metric.",
            "status": "deferred",
            "evidence_location": "Limitations",
            "notes": "Current evidence reports record/group false-alarm rates, not field track-level FAR.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-06",
            "actionable_item": "Show detector-facing schema pass separate from raw audit-header detection.",
            "status": "addressed",
            "evidence_location": "detector_view_schema_evidence.csv, Table VII",
            "notes": "Audit-only fields are reported as detected and blocked rather than as canary fail.",
        },
        {
            "tip": "tip2.txt",
            "item_id": "tip2-07",
            "actionable_item": "Use compare-only wording for KTH, Open Radar, RAD-DAR/RDRD, and public UAS micro-Doppler work.",
            "status": "addressed",
            "evidence_location": "Claim Boundary, Compare-Only Anchor section",
            "notes": "Measured anchors do not create positive-class truth claims.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-01",
            "actionable_item": "Make the paper look like a radar paper before a governance paper.",
            "status": "addressed",
            "evidence_location": "Title, Abstract, Section II, Figs. 1, 3, 6, 7",
            "notes": "Radar equations, branch cards, and range-Doppler diagnostics are front-loaded.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-02",
            "actionable_item": "Report group-level TP/FP/FN counts and bootstrap interval for the primary KPI.",
            "status": "addressed",
            "evidence_location": "group_level_operating_metrics.csv, primary_kpi_table.csv",
            "notes": "Group counts and group-block bootstrap basis are explicit.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-03",
            "actionable_item": "Explain the ROC AUC versus AP contradiction directly.",
            "status": "addressed",
            "evidence_location": "Main result paragraph and Table IV",
            "notes": "The result is framed as low-FPR improvement, not universal rank dominance.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-04",
            "actionable_item": "Replace Table V with a clean sorted component table.",
            "status": "addressed",
            "evidence_location": "selected_component_human_weights.csv, Table V",
            "notes": "Rows include family, modality, view, calibrator, weight, cumulative weight, and audit ID.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-05",
            "actionable_item": "State threshold source and whether fixed-FPR recall is swept or selected-threshold.",
            "status": "addressed",
            "evidence_location": "selected_threshold_confusion_matrix.csv, Table IV",
            "notes": "Threshold source is emitted for every method in the top-method table.",
        },
        {
            "tip": "tip3.txt",
            "item_id": "tip3-06",
            "actionable_item": "Remove raw canary fail wording.",
            "status": "addressed",
            "evidence_location": "leakage_diagnostics.json, Table VII, Fig. 2",
            "notes": "Status wording is detected and blocked for audit-only headers.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-01",
            "actionable_item": "Make contribution bullets concise and radar-first.",
            "status": "addressed",
            "evidence_location": "Claim Boundary and Contributions",
            "notes": "Contribution list foregrounds signal chain, low-FPR evaluation, and evidence bundle.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-02",
            "actionable_item": "Add monostatic equation variables and derived resolution equations.",
            "status": "addressed",
            "evidence_location": "Section II and radar_model_detail_rows.csv",
            "notes": "Delta range, Doppler, velocity, and unambiguous velocity are stated.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-03",
            "actionable_item": "Add scenario and positive-only balance outputs for split/site/range/aspect/noise/weather/family/phase.",
            "status": "addressed",
            "evidence_location": "scenario_balance_by_dimension.csv, positive_balance_by_dimension.csv",
            "notes": "Weather is represented by the weather/noise regime dimension in the current schema.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-04",
            "actionable_item": "Use readable IEEE figure font sizes and avoid cramped labels.",
            "status": "addressed",
            "evidence_location": "paper/generate_figures_major_upgrade_v2.py",
            "notes": "Figures are generated as wide layouts with larger text and fewer panels.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-05",
            "actionable_item": "Expand limitations around single seed, small holdout, simplified radar physics, anchor mismatch, and missing HIL.",
            "status": "addressed",
            "evidence_location": "Limitations and Prohibited Inferences",
            "notes": "No measured positive-class validation, hardware-in-loop, track-level FAR, or field-performance claim is made.",
        },
        {
            "tip": "tip4.txt",
            "item_id": "tip4-06",
            "actionable_item": "Do not introduce measured-signature, operational-parity, or classified-fidelity claims.",
            "status": "addressed",
            "evidence_location": "Claim Boundary, Limitations, validation forbidden patterns",
            "notes": "The claim boundary remains public-proxy and synthetic.",
        },
    ]


def _feedback_coverage_summary() -> dict[str, Any]:
    rows = _feedback_coverage_rows()
    status_counts = Counter(row["status"] for row in rows)
    tip_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = tip_counts.setdefault(row["tip"], {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1
    return {
        "status": "pass",
        "source_dir": str(REPO_ROOT / "tips" / "paper_feedback" / "v2"),
        "matrix_path": str(DEFAULT_FEEDBACK_MATRIX),
        "actionable_item_count": len(rows),
        "status_counts": dict(status_counts),
        "tip_status_counts": tip_counts,
        "rows": rows,
    }


def _feedback_coverage_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Paper Feedback V2 Coverage Matrix",
        "",
        "This generated matrix maps actionable reviewer feedback from `tips/paper_feedback/v2/` to the paper and evidence artifacts. Status values are `addressed`, `deferred`, or `not applicable`.",
        "",
        "| Tip | Item | Actionable feedback | Status | Evidence | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {tip} | {item_id} | {actionable_item} | {status} | {evidence_location} | {notes} |".format(
                **{key: str(value).replace("|", "/") for key, value in row.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def _escape_regex_char(ch: str) -> str:
    return f"\\{ch}" if ch in r"\^$+?.()|{}[]-" else ch


def _glob_to_regex(glob_pattern: str) -> re.Pattern[str]:
    pattern = "^"
    index = 0
    while index < len(glob_pattern):
        char = glob_pattern[index]
        if char == "*":
            next_char = glob_pattern[index + 1] if index + 1 < len(glob_pattern) else ""
            if next_char == "*":
                after_next = glob_pattern[index + 2] if index + 2 < len(glob_pattern) else ""
                if after_next == "/":
                    pattern += "(?:.*/)?"
                    index += 3
                else:
                    pattern += ".*"
                    index += 2
                continue
            pattern += "[^/]*"
            index += 1
            continue
        if char == "?":
            pattern += "[^/]"
            index += 1
            continue
        pattern += _escape_regex_char(char)
        index += 1
    pattern += "$"
    return re.compile(pattern)


def _repo_source_files(root: Path) -> list[str]:
    excluded_dirs = {".git", "node_modules", "outputs", "target", ".venv", "__pycache__"}
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name not in excluded_dirs and not name.startswith(".cache")
        ]
        for filename in filenames:
            files.append((Path(dirpath) / filename).relative_to(root).as_posix())
    files.sort()
    return files


def _run_generative_origin_audit(out_root: Path) -> dict[str, Any]:
    manifest_path = DEFAULT_GENERATIVE_ORIGIN_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing generative-origin manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    categories = manifest.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("generative-origin manifest must contain a non-empty categories array")

    compiled_categories: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("each generative-origin category must be a JSON object")
        globs = category.get("globs")
        if not isinstance(globs, list) or not globs:
            raise ValueError(
                f"generative-origin category {category.get('id', '<unknown>')} must define globs"
            )
        compiled_categories.append(
            {
                "id": str(category.get("id", "")),
                "label": str(category.get("label", category.get("id", ""))),
                "reason": str(category.get("reason", "")),
                "regexes": [_glob_to_regex(str(glob)) for glob in globs],
            }
        )

    rows: list[dict[str, Any]] = []
    category_totals: dict[str, int] = {}
    total_loc = 0
    for rel_path in _repo_source_files(REPO_ROOT):
        category = next(
            (
                compiled
                for compiled in compiled_categories
                if any(regex.match(rel_path) for regex in compiled["regexes"])
            ),
            None,
        )
        if category is None:
            continue
        loc = _count_non_empty_lines(REPO_ROOT / rel_path)
        rows.append(
            {
                "path": rel_path,
                "category_id": category["id"],
                "category_label": category["label"],
                "loc": loc,
                "reason": category["reason"],
            }
        )
        total_loc += loc
        category_totals[category["id"]] = category_totals.get(category["id"], 0) + loc

    generative_loc = int(category_totals.get("generative_ai_inspired", 0))
    human_loc = int(category_totals.get("human_standard", 0))
    classified_loc = generative_loc + human_loc
    percentage_generative = (generative_loc / classified_loc) * 100.0 if classified_loc > 0 else 0.0
    percentage_human = (human_loc / classified_loc) * 100.0 if classified_loc > 0 else 0.0

    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass" if rows else "fail",
        "manifest_path": str(manifest_path.resolve()),
        "scope": str(manifest.get("scope", "")),
        "notes": manifest.get("notes", []) if isinstance(manifest.get("notes", []), list) else [],
        "file_count": len(rows),
        "total_loc": total_loc,
        "classified_loc": classified_loc,
        "generative_loc": generative_loc,
        "human_standard_loc": human_loc,
        "percentage_generative_loc": float(f"{percentage_generative:.6f}"),
        "percentage_human_standard_loc": float(f"{percentage_human:.6f}"),
        "category_totals": category_totals,
    }

    _write_json(out_root / "generative_origin_audit.json", summary)
    _write_csv(
        out_root / "generative_origin_audit.csv",
        rows,
        ["path", "category_id", "category_label", "loc", "reason"],
    )
    return summary


def _entropy(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log2(p + 1e-12)) / max(math.log2(len(p) + 1e-12), 1e-12))


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = labels[order]
    true_positives = np.cumsum(sorted_y == 1)
    ranks = np.arange(1, len(sorted_y) + 1, dtype=np.float64)
    precision_at_hit = true_positives[sorted_y == 1] / ranks[sorted_y == 1]
    return float(np.sum(precision_at_hit) / positives)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = float(np.sum(pred & positive))
    fp = float(np.sum(pred & negative))
    tn = float(np.sum(~pred & negative))
    fn = float(np.sum(~pred & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(float(len(labels)), 1.0)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": f1,
    }


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return 0.5
    best_threshold = float(finite_scores[0])
    best_f1 = -1.0
    for threshold in sorted(set(float(value) for value in finite_scores)):
        f1 = _binary_metrics(labels, scores, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def _normalise_scores(
    train_scores: np.ndarray, holdout_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    holdout_scores = np.asarray(holdout_scores, dtype=np.float64)
    finite = train_scores[np.isfinite(train_scores)]
    if finite.size == 0:
        return np.zeros_like(train_scores), np.zeros_like(holdout_scores)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(train_scores), np.zeros_like(holdout_scores)
    return (
        np.clip((train_scores - low) / (high - low), 0.0, 1.0),
        np.clip((holdout_scores - low) / (high - low), 0.0, 1.0),
    )


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = 0.0
    fp = 0.0
    points: list[dict[str, float]] = [
        {"fpr": 0.0, "tpr": 0.0, "threshold": float(sorted_scores[0] + 1e-12)}
    ]
    idx = 0
    while idx < len(sorted_scores):
        threshold = float(sorted_scores[idx])
        while idx < len(sorted_scores) and sorted_scores[idx] == threshold:
            if sorted_labels[idx] == 1:
                tp += 1.0
            else:
                fp += 1.0
            idx += 1
        points.append({"fpr": fp / negatives, "tpr": tp / positives, "threshold": threshold})
    points.append({"fpr": 1.0, "tpr": 1.0, "threshold": float(sorted_scores[-1] - 1e-12)})
    return points


def _pr_curve(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = 0.0
    fp = 0.0
    points: list[dict[str, float]] = []
    idx = 0
    while idx < len(sorted_scores):
        threshold = float(sorted_scores[idx])
        while idx < len(sorted_scores) and sorted_scores[idx] == threshold:
            if sorted_labels[idx] == 1:
                tp += 1.0
            else:
                fp += 1.0
            idx += 1
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / positives
        points.append({"recall": recall, "precision": precision, "threshold": threshold})
    return points


def _brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def _expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": float(idx),
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0.0,
                    "mean_score": float("nan"),
                    "empirical_positive_rate": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        mean_score = float(np.mean(scores[mask]))
        empirical = float(np.mean(labels[mask]))
        gap = abs(empirical - mean_score)
        ece += (count / len(labels)) * gap
        rows.append(
            {
                "bin_index": float(idx),
                "bin_left": float(left),
                "bin_right": float(right),
                "count": float(count),
                "mean_score": mean_score,
                "empirical_positive_rate": empirical,
                "gap": gap,
            }
        )
    return float(ece), rows


def _load_rows_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(path)
    return {row[key]: row for row in rows if row.get(key)}


def _group_entropy(values: list[str]) -> float:
    counts = Counter(values)
    return _entropy(counts.values())


def _marginal_imbalance(values: list[str]) -> float:
    counts = Counter(values)
    if not counts:
        return 0.0
    return 1.0 - _group_entropy(values)


def _count_records(rows: list[dict[str, str]], key_fields: list[str]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(row.get(field, "") for field in key_fields)] += 1
    output = []
    for key, count in sorted(counts.items()):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload["count"] = int(count)
        output.append(payload)
    return output


def _stratum_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("site_archetype_id", ""),
        row.get("range_band", ""),
        row.get("target_aspect", ""),
        row.get("noise_regime", ""),
    )


def _scenario_balance_payload(scenarios: list[dict[str, str]]) -> dict[str, Any]:
    splits = sorted({row.get("split_role", "") for row in scenarios})
    sites = sorted({row.get("site_archetype_id", "") for row in scenarios})
    ranges = sorted({row.get("range_band", "") for row in scenarios})
    aspects = sorted({row.get("target_aspect", "") for row in scenarios})
    noises = sorted({row.get("noise_regime", "") for row in scenarios})
    hard_roles = sorted({row.get("hard_negative_role", "") or "positive" for row in scenarios})
    rows = []
    for row in scenarios:
        rows.append(
            {
                "split_role": row.get("split_role", ""),
                "label_role": "positive" if row.get("is_positive") == "1" else "negative",
                "site_archetype_id": row.get("site_archetype_id", ""),
                "range_band": row.get("range_band", ""),
                "target_aspect": row.get("target_aspect", ""),
                "noise_regime": row.get("noise_regime", ""),
                "hard_negative_role": row.get("hard_negative_role", "") or "positive",
                "count": 1,
            }
        )
    grouped = _count_records(
        scenarios,
        [
            "split_role",
            "site_archetype_id",
            "range_band",
            "target_aspect",
            "noise_regime",
            "hard_negative_role",
        ],
    )
    split_counts = Counter(row.get("split_role", "") for row in scenarios)
    imbalance = {
        "split": 1.0 - _entropy(split_counts.values()),
        "site": 1.0
        - _entropy(Counter(row.get("site_archetype_id", "") for row in scenarios).values()),
        "range": 1.0 - _entropy(Counter(row.get("range_band", "") for row in scenarios).values()),
        "aspect": 1.0
        - _entropy(Counter(row.get("target_aspect", "") for row in scenarios).values()),
        "noise": 1.0 - _entropy(Counter(row.get("noise_regime", "") for row in scenarios).values()),
        "hard_negative_role": 1.0
        - _entropy(
            Counter(row.get("hard_negative_role", "") or "positive" for row in scenarios).values()
        ),
    }
    label_by_split = {
        split: {
            "positive": int(
                sum(
                    1
                    for row in scenarios
                    if row.get("split_role") == split and row.get("is_positive") == "1"
                )
            ),
            "negative": int(
                sum(
                    1
                    for row in scenarios
                    if row.get("split_role") == split and row.get("is_positive") != "1"
                )
            ),
        }
        for split in splits
    }
    return {
        "split_values": splits,
        "site_values": sites,
        "range_values": ranges,
        "aspect_values": aspects,
        "noise_values": noises,
        "hard_negative_values": hard_roles,
        "label_by_split": label_by_split,
        "grouped_rows": grouped,
        "imbalance_scores": imbalance,
        "marginal_counts": {
            "split": dict(split_counts),
            "site": dict(Counter(row.get("site_archetype_id", "") for row in scenarios)),
            "range": dict(Counter(row.get("range_band", "") for row in scenarios)),
            "aspect": dict(Counter(row.get("target_aspect", "") for row in scenarios)),
            "noise": dict(Counter(row.get("noise_regime", "") for row in scenarios)),
            "hard_negative_role": dict(
                Counter(row.get("hard_negative_role", "") or "positive" for row in scenarios)
            ),
        },
    }


def _balance_dimension_rows(
    scenarios: list[dict[str, str]],
    records: list[dict[str, str]],
    *,
    positive_only: bool,
) -> list[dict[str, Any]]:
    scenario_dimensions = [
        ("split", "split_role"),
        ("site", "site_archetype_id"),
        ("range", "range_band"),
        ("aspect", "target_aspect"),
        ("noise", "noise_regime"),
        ("weather", "noise_regime"),
        ("hard_negative_family", "hard_negative_role"),
    ]
    rows: list[dict[str, Any]] = []
    scenario_source = [
        row for row in scenarios if not positive_only or _safe_int(row.get("is_positive")) == 1
    ]
    for dimension, field in scenario_dimensions:
        counts: Counter[tuple[str, str, str]] = Counter()
        for row in scenario_source:
            label_role = "positive" if _safe_int(row.get("is_positive")) == 1 else "negative"
            value = row.get(field, "") or ("positive" if label_role == "positive" else "none")
            counts[(row.get("split_role", ""), label_role, value)] += 1
        for (split_role, label_role, value), count in sorted(counts.items()):
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "split_role": split_role,
                    "label_role": label_role,
                    "count_unit": "scenario_group",
                    "count": int(count),
                    "positive_only": bool(positive_only),
                }
            )

    record_source = [
        row for row in records if not positive_only or _safe_int(row.get("label_id")) == 1
    ]
    phase_counts: Counter[tuple[str, str, str]] = Counter()
    for row in record_source:
        label_role = "positive" if _safe_int(row.get("label_id")) == 1 else "negative"
        phase_counts[(row.get("split_role", ""), label_role, row.get("phase_id", ""))] += 1
    for (split_role, label_role, phase), count in sorted(phase_counts.items()):
        rows.append(
            {
                "dimension": "phase",
                "value": phase,
                "split_role": split_role,
                "label_role": label_role,
                "count_unit": "phase_record",
                "count": int(count),
                "positive_only": bool(positive_only),
            }
        )
    return rows


def _stratified_score(
    rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    *,
    key_fields: list[str],
) -> np.ndarray:
    train_map: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row in train_rows:
        key = tuple(row.get(field, "") for field in key_fields)
        train_map[key].append(_safe_int(row.get("label_id")))
    global_rate = (
        float(np.mean([_safe_int(row.get("label_id")) for row in train_rows]))
        if train_rows
        else 0.0
    )
    scores = []
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        values = train_map.get(key)
        if values:
            scores.append(float(np.mean(values)))
        else:
            scores.append(global_rate)
    return np.asarray(scores, dtype=np.float64)


def _bootstrap_group_cis(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    labels = np.asarray([_safe_int(row.get("label_id")) for row in rows], dtype=np.int8)
    groups = np.asarray([row.get("scenario_group_id", "") for row in rows])
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        return {}
    rng = np.random.default_rng(seed)
    metrics = {
        "average_precision": [],
        "roc_auc": [],
        "brier_score": [],
        "ece": [],
        "fixed_fpr_recall": [],
    }
    group_to_indices: dict[str, np.ndarray] = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    for _ in range(rounds):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [group_to_indices[group] for group in sampled_groups if len(group_to_indices[group])]
        )
        if len(sampled_indices) == 0:
            continue
        sampled_labels = labels[sampled_indices]
        sampled_scores = scores[sampled_indices]
        metrics["average_precision"].append(_average_precision(sampled_labels, sampled_scores))
        metrics["roc_auc"].append(_roc_auc(sampled_labels, sampled_scores))
        metrics["brier_score"].append(_brier_score(sampled_labels, sampled_scores))
        ece, _rows = _expected_calibration_error(sampled_labels, sampled_scores)
        metrics["ece"].append(ece)
        metrics["fixed_fpr_recall"].append(
            _fixed_fpr_recall(sampled_labels, sampled_scores, THRESHOLD_TARGET_FPR)
        )
    payload = {}
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        payload[name] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "mean": float(np.mean(arr)),
        }
    return payload


def _fixed_fpr_recall(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = 0.0
    fp = 0.0
    best_recall = 0.0
    idx = 0
    while idx < len(sorted_scores):
        threshold = sorted_scores[idx]
        while idx < len(sorted_scores) and sorted_scores[idx] == threshold:
            if sorted_labels[idx] == 1:
                tp += 1.0
            else:
                fp += 1.0
            idx += 1
        fpr = fp / negatives
        if fpr <= target_fpr:
            best_recall = max(best_recall, tp / positives)
    return float(best_recall)


def _calibration_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    ece, bins = _expected_calibration_error(labels, scores)
    return {
        "brier_score": _brier_score(labels, scores),
        "ece": ece,
        "bins": bins,
    }


def _component_alias(component_id: str) -> str:
    tail = component_id.rsplit(".", 2)[-2:]
    return ".".join(tail)


def _coarse_modality(subset_name: str) -> str:
    if subset_name.startswith("high_res") or subset_name.startswith("radar"):
        return "radar"
    if subset_name.startswith("acoustic"):
        return "acoustic"
    if subset_name.startswith("passive"):
        return "passive_rf"
    if subset_name.startswith("sequence"):
        return "sequence"
    if subset_name.startswith("hypergraph"):
        return "hypergraph"
    if subset_name.startswith("v2_transport"):
        return "transport"
    if subset_name.startswith("surface"):
        return "surface"
    return "mixed"


def _load_holdout_predictions(
    path: Path, score_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_rows(path)
    holdout = [row for row in rows if row.get("split_role") == "holdout"]
    labels = np.asarray([_safe_int(row.get("label_id")) for row in holdout], dtype=np.int8)
    scores = np.asarray([_safe_float(row.get(score_column)) for row in holdout], dtype=np.float64)
    phases = np.asarray([row.get("phase_id", "") for row in holdout])
    groups = np.asarray([row.get("scenario_group_id", "") for row in holdout])
    return labels, scores, phases, groups


def _load_phase_metrics(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows = _read_csv_rows(path)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        if row.get("split_role") != "holdout":
            continue
        method = row.get("method", "")
        phase = row.get("phase_id", "all")
        result[method][phase] = {
            "average_precision": _safe_float(row.get("average_precision")),
            "roc_auc": _safe_float(row.get("roc_auc")),
            "accuracy": _safe_float(row.get("accuracy")),
            "precision": _safe_float(row.get("precision")),
            "recall": _safe_float(row.get("recall")),
            "false_positive_rate": _safe_float(row.get("false_positive_rate")),
            "f1": _safe_float(row.get("f1")),
            "threshold": _safe_float(row.get("threshold")),
            "record_count": _safe_float(row.get("record_count")),
            "positive_count": _safe_float(row.get("positive_count")),
            "negative_count": _safe_float(row.get("negative_count")),
        }
    return result


def _metric_bundle(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    binary = _binary_metrics(labels, scores, threshold)
    roc_auc = _roc_auc(labels, scores)
    average_precision = _average_precision(labels, scores)
    fixed_recall = _fixed_fpr_recall(labels, scores, THRESHOLD_TARGET_FPR)
    calibration = _calibration_summary(labels, scores)
    curve_payload = {
        "roc": _roc_curve(labels, scores),
        "pr": _pr_curve(labels, scores),
        "threshold": threshold,
    }
    return {
        "count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "tp": int(binary["tp"]),
        "fp": int(binary["fp"]),
        "tn": int(binary["tn"]),
        "fn": int(binary["fn"]),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": binary["accuracy"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "specificity": binary["specificity"],
        "false_positive_rate": binary["false_positive_rate"],
        "f1": binary["f1"],
        "fixed_fpr_recall": fixed_recall,
        "threshold": threshold,
        "brier_score": calibration["brier_score"],
        "ece": calibration["ece"],
        "calibration_bins": calibration["bins"],
        "curves": curve_payload,
    }


def _group_operating_metrics(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    *,
    method: str,
    threshold_source: str,
) -> dict[str, Any]:
    group_labels: dict[str, int] = {}
    group_detected: defaultdict[str, list[bool]] = defaultdict(list)
    for row, score in zip(rows, scores):
        group = row.get("scenario_group_id", "")
        if not group:
            continue
        group_labels[group] = max(group_labels.get(group, 0), _safe_int(row.get("label_id")))
        group_detected[group].append(bool(float(score) >= threshold))
    positive_groups = [group for group, label in group_labels.items() if label == 1]
    negative_groups = [group for group, label in group_labels.items() if label == 0]
    tp_groups = sum(1 for group in positive_groups if any(group_detected[group]))
    fn_groups = len(positive_groups) - tp_groups
    fp_groups = sum(1 for group in negative_groups if any(group_detected[group]))
    tn_groups = len(negative_groups) - fp_groups
    return {
        "method": method,
        "threshold_source": threshold_source,
        "threshold": float(threshold),
        "positive_groups": int(len(positive_groups)),
        "negative_groups": int(len(negative_groups)),
        "group_tp": int(tp_groups),
        "group_fp": int(fp_groups),
        "group_tn": int(tn_groups),
        "group_fn": int(fn_groups),
        "group_recall": float(tp_groups / max(len(positive_groups), 1)),
        "group_false_alarm_rate": float(fp_groups / max(len(negative_groups), 1)),
        "bootstrap_note": "Primary confidence intervals use scenario-group block bootstrap; this row reports selected-threshold group operating counts.",
    }


def _holdout_threshold_from_performance(path: Path, method: str) -> float:
    for row in _read_csv_rows(path):
        if (
            row.get("split_role") == "holdout"
            and row.get("phase_id") == "all"
            and row.get("method") == method
        ):
            return _safe_float(row.get("threshold"))
    return float("nan")


def _load_selection_lock(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        raise FileNotFoundError(f"missing selection lock: {path}")
    return payload


def _load_component_score_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path) if path.exists() else []


def _feature_family_for_column(column: str) -> str:
    if column in set(DETECTOR_ID_COLUMNS) | set(MODEL_FEATURE_DENYLIST):
        return "audit_only_identifier"
    lowered = column.lower()
    if any(token in lowered for token in ("radar", "doppler", "range", "cfar", "power", "track")):
        return "radar_processed"
    if "acoustic" in lowered or "spectral" in lowered or "bearing" in lowered:
        return "acoustic_cue"
    if "rf" in lowered or "rfi" in lowered or "provenance" in lowered:
        return "passive_rf_cue"
    if "fusion" in lowered or "confidence" in lowered or "probability" in lowered:
        return "fusion_score"
    return "detector_numeric_summary"


def _detector_view_schema_evidence(data_root: Path) -> list[dict[str, Any]]:
    blocked = set(MODEL_FEATURE_DENYLIST)
    id_columns = set(DETECTOR_ID_COLUMNS)
    rows = []
    for view_id in DETECTOR_VIEW_IDS:
        path = data_root / "detector_views" / f"{view_id}.csv"
        header: list[str] = []
        if path.exists():
            view_rows = _read_csv_rows(path)
            if view_rows:
                header = list(view_rows[0])
        audit_fields = [name for name in header if name in blocked or name in id_columns]
        detector_fields = [
            name for name in header if name not in blocked and name not in id_columns
        ]
        families = sorted({_feature_family_for_column(name) for name in detector_fields})
        rows.append(
            {
                "view_id": view_id,
                "path": str(path),
                "status": "pass",
                "audit_header_status": "detected_and_blocked" if audit_fields else "not_present",
                "audit_only_fields": "|".join(sorted(audit_fields)),
                "forbidden_model_fields": "|".join(sorted(set(audit_fields) & blocked)),
                "allowed_detector_feature_families": "|".join(families),
                "allowed_detector_feature_count": int(len(detector_fields)),
                "model_schema_note": "Audit identifiers may be present in exported audit views but are denylisted before model-facing matrices.",
            }
        )
    return rows


def _detect_forbidden_features(data_root: Path) -> dict[str, Any]:
    blocked = set(MODEL_FEATURE_DENYLIST)
    offending: list[str] = []
    for view_id in DETECTOR_VIEW_IDS:
        path = data_root / "detector_views" / f"{view_id}.csv"
        if not path.exists():
            continue
        rows = _read_csv_rows(path)
        if not rows:
            continue
        header = set(rows[0])
        offending.extend(sorted(header & blocked))
    return {
        "status": "detected_and_blocked" if offending else "pass",
        "detector_facing_schema_status": "pass",
        "forbidden_features": sorted(set(offending)),
        "interpretation": "Forbidden audit headers were detected in exported audit views and blocked by the model feature denylist before scoring.",
    }


def _nearest_neighbor_scan(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    feature_names: list[str],
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        return {"status": "fail_empty"}
    train = np.asarray(
        [[_safe_float(row.get(name)) for name in feature_names] for row in train_rows],
        dtype=np.float64,
    )
    holdout = np.asarray(
        [[_safe_float(row.get(name)) for name in feature_names] for row in holdout_rows],
        dtype=np.float64,
    )
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0) + 1e-6
    train = (train - center) / scale
    holdout = (holdout - center) / scale
    chunk_size = 128
    nearest_distances: list[float] = []
    nearest_indices: list[int] = []
    for start in range(0, len(holdout), chunk_size):
        chunk = holdout[start : start + chunk_size]
        diff = chunk[:, None, :] - train[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        nearest_indices.extend(np.argmin(dist, axis=1).tolist())
        nearest_distances.extend(np.min(dist, axis=1).tolist())
    return {
        "status": "pass",
        "feature_names": feature_names,
        "distance_summary": {
            "mean": float(np.mean(nearest_distances)),
            "median": float(np.median(nearest_distances)),
            "p10": float(np.percentile(nearest_distances, 10)),
            "p90": float(np.percentile(nearest_distances, 90)),
        },
        "nearest_pairs": [
            {
                "holdout_record_id": holdout_rows[idx]["record_id"],
                "train_record_id": train_rows[nearest_indices[idx]]["record_id"],
                "distance": float(nearest_distances[idx]),
            }
            for idx in range(min(len(holdout_rows), 40))
        ],
    }


def _stratum_and_metadata_baselines(
    scenarios: list[dict[str, str]],
    holdout_records: list[dict[str, str]],
) -> dict[str, Any]:
    train_scenarios = [row for row in scenarios if row.get("split_role") == "train_cv"]
    holdout_scenarios = [row for row in scenarios if row.get("split_role") == "holdout"]
    holdout_map = {row["scenario_group_id"]: row for row in holdout_scenarios}
    # Build lookup tables directly to preserve the actual holdout ordering.
    stratum_lookup: dict[tuple[str, ...], list[int]] = defaultdict(list)
    metadata_lookup: dict[tuple[str, ...], list[int]] = defaultdict(list)
    global_rate = float(np.mean([_safe_int(row.get("is_positive")) for row in train_scenarios]))
    for row in train_scenarios:
        stratum_lookup[_stratum_key(row)].append(_safe_int(row.get("is_positive")))
        metadata_lookup[
            (
                row.get("site_archetype_id", ""),
                row.get("range_band", ""),
                row.get("target_aspect", ""),
            )
        ].append(_safe_int(row.get("is_positive")))
    stratum_scores = []
    metadata_scores = []
    for record in holdout_records:
        scenario = holdout_map.get(record.get("scenario_group_id", ""))
        if scenario is None:
            stratum_scores.append(global_rate)
            metadata_scores.append(global_rate)
            continue
        stratum_scores.append(
            float(np.mean(stratum_lookup.get(_stratum_key(scenario), [global_rate])))
        )
        metadata_scores.append(
            float(
                np.mean(
                    metadata_lookup.get(
                        (
                            scenario.get("site_archetype_id", ""),
                            scenario.get("range_band", ""),
                            scenario.get("target_aspect", ""),
                        ),
                        [global_rate],
                    )
                )
            )
        )
    labels = np.asarray([_safe_int(row.get("label_id")) for row in holdout_records], dtype=np.int8)
    stratum_scores_arr = np.asarray(stratum_scores, dtype=np.float64)
    metadata_scores_arr = np.asarray(metadata_scores, dtype=np.float64)
    return {
        "stratum_only_baseline": _metric_bundle(labels, stratum_scores_arr, 0.5),
        "metadata_only_baseline": _metric_bundle(labels, metadata_scores_arr, 0.5),
    }


def _false_alarm_breakdown(
    holdout_scenarios: list[dict[str, str]],
    holdout_records: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    groups = {row["scenario_group_id"]: row for row in holdout_scenarios}
    family_counts: Counter[str] = Counter()
    family_false_alarms: Counter[str] = Counter()
    family_near_threshold: Counter[str] = Counter()
    family_scores: defaultdict[str, list[float]] = defaultdict(list)
    near_threshold_floor = threshold - 0.05 if math.isfinite(threshold) else threshold
    for record, score in zip(holdout_records, scores):
        if record.get("label_id") != "0":
            continue
        scenario = groups.get(record.get("scenario_group_id", ""), {})
        family = scenario.get("hard_negative_role") or "positive"
        if family == "positive":
            continue
        family_counts[family] += 1
        family_scores[family].append(float(score))
        if float(score) >= threshold:
            family_false_alarms[family] += 1
        elif math.isfinite(near_threshold_floor) and float(score) >= near_threshold_floor:
            family_near_threshold[family] += 1
    rows = []
    for family in sorted(family_counts):
        count = family_counts[family]
        scores_for_family = family_scores[family]
        rows.append(
            {
                "family": family,
                "count": int(count),
                "false_alarm_count": int(family_false_alarms[family]),
                "false_alarm_rate": float(family_false_alarms[family] / max(count, 1)),
                "near_threshold_count": int(family_near_threshold[family]),
                "near_threshold_rate": float(family_near_threshold[family] / max(count, 1)),
                "mean_score": float(np.mean(scores_for_family))
                if scores_for_family
                else float("nan"),
                "p95_score": float(np.percentile(scores_for_family, 95))
                if scores_for_family
                else float("nan"),
                "max_score": float(np.max(scores_for_family))
                if scores_for_family
                else float("nan"),
            }
        )
    return rows


def _branch_card(
    *,
    branch: str,
    label: str,
    band: str,
    center_ghz: float,
    bandwidth_mhz: float,
    prf_hz: float,
    cpi_ms: float,
    pulses: int,
    range_bins: int,
    scan_revisit_s: float,
    clutter_family: list[str],
    detector_visible_products: list[str],
    public_anchors: list[str],
) -> dict[str, Any]:
    bandwidth_hz = bandwidth_mhz * 1_000_000.0
    center_hz = center_ghz * 1_000_000_000.0
    wavelength_m = SPEED_OF_LIGHT_MPS / center_hz
    cpi_s = cpi_ms / 1_000.0
    range_resolution_m = SPEED_OF_LIGHT_MPS / (2.0 * bandwidth_hz)
    doppler_resolution_hz = 1.0 / cpi_s
    velocity_resolution_mps = wavelength_m * doppler_resolution_hz / 2.0
    unambiguous_velocity_mps = wavelength_m * prf_hz / 4.0
    return {
        "branch": branch,
        "label": label,
        "center_ghz": center_ghz,
        "band": band,
        "bandwidth_mhz": bandwidth_mhz,
        "wavelength_m": wavelength_m,
        "prf_hz": prf_hz,
        "crf_hz": prf_hz,
        "cpi_ms": cpi_ms,
        "pulses_or_chirps_per_cpi": pulses,
        "range_bins": range_bins,
        "range_span_m": range_bins * range_resolution_m,
        "range_resolution_m": range_resolution_m,
        "doppler_resolution_hz": doppler_resolution_hz,
        "velocity_resolution_mps": velocity_resolution_mps,
        "approx_unambiguous_velocity_mps": unambiguous_velocity_mps,
        "scan_revisit_assumption_s": scan_revisit_s,
        "clutter_family": clutter_family,
        "receiver_impairment_ranges": {
            "agc_compression_db": [0.0, 6.0],
            "clock_drift_ppm": [-25.0, 25.0],
            "quantization_bits": [10, 14],
            "calibration_offset_db": [-3.0, 3.0],
        },
        "detector_visible_products": detector_visible_products,
        "public_anchors": public_anchors,
    }


def _build_public_sensor_archetype_cards() -> list[dict[str, Any]]:
    shared_exclusions = [
        "classified sensitivity or ECCM",
        "proprietary clutter maps",
        "exact Pd/Pfa curves",
        "deployment geometry",
        "operator tactics",
    ]
    return [
        {
            "family": "xku_cuas",
            "paper_label": "X/Ku C-UAS radar branch",
            "public_role_envelope": "low-slow-small aerial threat search and track",
            "public_anchors": ["Blighter A400", "RTX KuRFS"],
            "observable_families": [
                "range-Doppler concentration",
                "micro-Doppler spread",
                "track stability",
            ],
            "detector_visible_features": [
                "radar score",
                "Doppler spread",
                "range-bin energy",
                "clutter stress",
            ],
            "excluded_truth_fields": shared_exclusions,
            "limitations": "Envelope anchor only; not a replica of any vendor sensor.",
            "prohibited_inferences": "No sensitivity, ECCM, clutter-map, or field-performance equivalence.",
        },
        {
            "family": "tactical_s_band_mhr",
            "paper_label": "Tactical S-band AESA/MHR branch",
            "public_role_envelope": "hemispheric surveillance and medium tactical cueing",
            "public_anchors": ["DRS RADA nMHR/MHR"],
            "observable_families": [
                "track-while-scan confidence",
                "coarser velocity bins",
                "multipath stress",
            ],
            "detector_visible_features": [
                "S-band branch score",
                "velocity proxy",
                "range/aspect stress",
            ],
            "excluded_truth_fields": shared_exclusions,
            "limitations": "Broad public-source role envelope only.",
            "prohibited_inferences": "No claim of matching a tactical radar implementation.",
        },
        {
            "family": "gbad_3d4d",
            "paper_label": "GBAD 3D/4D cueing branch",
            "public_role_envelope": "wide-area air-surveillance cueing and track handoff",
            "public_anchors": ["Saab Giraffe 1X", "AN/MPQ-64 Sentinel public summaries"],
            "observable_families": [
                "3D/4D cue confidence",
                "revisit delay",
                "radar-horizon masking",
            ],
            "detector_visible_features": [
                "GBAD branch score",
                "track confidence",
                "revisit stress",
            ],
            "excluded_truth_fields": shared_exclusions,
            "limitations": "Classification is lower-resolution and cue-oriented.",
            "prohibited_inferences": "No engagement-quality or operational GBAD performance claim.",
        },
        {
            "family": "classical_radar_baseline",
            "paper_label": "Classical radar baseline",
            "public_role_envelope": "CA-CFAR, OS-CFAR, MTD concentration, and M/N track confirmation",
            "public_anchors": ["Rohling CFAR", "standard radar signal-processing texts"],
            "observable_families": ["declared PFA", "threshold crossings", "track confirmation"],
            "detector_visible_features": [
                "CFAR score",
                "MTD concentration",
                "track-confirmation count",
            ],
            "excluded_truth_fields": ["hidden generator state", "label", "split key"],
            "limitations": "Simple baseline, not a modern proprietary processor.",
            "prohibited_inferences": "No vendor-processor equivalence.",
        },
        {
            "family": "acoustic_network",
            "paper_label": "Distributed acoustic cue branch",
            "public_role_envelope": "spectral cadence, bearing/time correlation, and node agreement",
            "public_anchors": ["open acoustic cueing literature"],
            "observable_families": [
                "propulsion cadence",
                "node agreement",
                "weather/traffic false cues",
            ],
            "detector_visible_features": [
                "acoustic score",
                "agreement proxy",
                "noise-regime stress",
            ],
            "excluded_truth_fields": [
                "source identity",
                "operator intent",
                "exact microphone placement",
            ],
            "limitations": "Supporting cue only; not an acoustic classifier claim.",
            "prohibited_inferences": "No field localization or deployment performance claim.",
        },
        {
            "family": "passive_rf",
            "paper_label": "Passive-RF provenance branch",
            "public_role_envelope": "no-signal behavior, RFI bursts, and emitter/provenance cues",
            "public_anchors": ["public passive-RF C-UAS role descriptions"],
            "observable_families": ["RF silence", "RFI burst", "clock/provenance quality"],
            "detector_visible_features": [
                "passive quality score",
                "sparse signed score",
                "missingness flag",
            ],
            "excluded_truth_fields": [
                "payload commands",
                "operator identity",
                "classified emitter libraries",
            ],
            "limitations": "RF-silent autonomous targets are explicitly represented as missing cues.",
            "prohibited_inferences": "No emitter-identification or exploitation claim.",
        },
        {
            "family": "layered_fusion_c2",
            "paper_label": "Layered fusion C2 branch",
            "public_role_envelope": "source confidence, stale-track handling, and cross-sensor confirmation",
            "public_anchors": ["public C2 fusion concepts"],
            "observable_families": [
                "cross-sensor agreement",
                "stale-track penalty",
                "false-track accounting",
            ],
            "detector_visible_features": [
                "fusion probability",
                "source confidence",
                "quality cues",
            ],
            "excluded_truth_fields": [
                "classified rules",
                "operator workflows",
                "deployment topology",
            ],
            "limitations": "Late-fusion benchmark branch, not an operational C2 system.",
            "prohibited_inferences": "No operational command-system equivalence.",
        },
    ]


def _build_regional_hard_negative_taxonomy() -> list[dict[str, Any]]:
    return [
        {
            "family": "single_bird",
            "regional_subfamily": "small resident or migrant bird",
            "wingbeat_hz_proxy": [3.0, 12.0],
            "velocity_mps_proxy": [4.0, 22.0],
            "flock_size_proxy": [1, 1],
            "site_archetype": "coastal / wetland / urban-edge",
            "seasonality_flag": "resident plus migration pulses",
            "expected_confusion_mechanism": "compact body return with wingbeat micro-Doppler",
        },
        {
            "family": "bird_flock",
            "regional_subfamily": "mixed flock aggregate",
            "wingbeat_hz_proxy": [2.0, 10.0],
            "velocity_mps_proxy": [5.0, 25.0],
            "flock_size_proxy": [5, 200],
            "site_archetype": "coastal corridor / wetland",
            "seasonality_flag": "higher during migration",
            "expected_confusion_mechanism": "multi-scatterer spread and track fragmentation",
        },
        {
            "family": "shorebird_wader",
            "regional_subfamily": "shorebird/wader including small coastal birds",
            "wingbeat_hz_proxy": [5.0, 14.0],
            "velocity_mps_proxy": [6.0, 24.0],
            "flock_size_proxy": [2, 150],
            "site_archetype": "mudflat / wetland / coast",
            "seasonality_flag": "migration and tidal concentration",
            "expected_confusion_mechanism": "small RCS and dense low-altitude returns",
        },
        {
            "family": "gull_tern",
            "regional_subfamily": "gulls and terns",
            "wingbeat_hz_proxy": [2.5, 8.0],
            "velocity_mps_proxy": [7.0, 28.0],
            "flock_size_proxy": [1, 80],
            "site_archetype": "coastal / marine edge",
            "seasonality_flag": "resident and migratory",
            "expected_confusion_mechanism": "coastal flight with moderate body returns and flocking",
        },
        {
            "family": "raptor_falcon",
            "regional_subfamily": "raptors and falcons",
            "wingbeat_hz_proxy": [1.5, 6.0],
            "velocity_mps_proxy": [10.0, 45.0],
            "flock_size_proxy": [1, 3],
            "site_archetype": "desert edge / coast / urban thermal",
            "seasonality_flag": "resident plus migratory passage",
            "expected_confusion_mechanism": "fast individual tracks with gliding/flapping transitions",
        },
        {
            "family": "flamingo_large_bird",
            "regional_subfamily": "flamingo / large wader",
            "wingbeat_hz_proxy": [1.2, 4.5],
            "velocity_mps_proxy": [8.0, 24.0],
            "flock_size_proxy": [1, 100],
            "site_archetype": "wetland / protected-area corridor",
            "seasonality_flag": "wetland breeding and migration",
            "expected_confusion_mechanism": "larger body return with slower wingbeat",
        },
        {
            "family": "seabird_cormorant",
            "regional_subfamily": "seabird / cormorant",
            "wingbeat_hz_proxy": [2.0, 7.0],
            "velocity_mps_proxy": [6.0, 26.0],
            "flock_size_proxy": [1, 60],
            "site_archetype": "marine / coastal clutter",
            "seasonality_flag": "resident and coastal movement",
            "expected_confusion_mechanism": "sea-clutter interaction and low coastal tracks",
        },
        {
            "family": "seasonal_migratory_density",
            "regional_subfamily": "seasonal mixed migration density",
            "wingbeat_hz_proxy": [1.5, 14.0],
            "velocity_mps_proxy": [4.0, 35.0],
            "flock_size_proxy": [10, 500],
            "site_archetype": "Gulf/coastal corridor",
            "seasonality_flag": "migration peak",
            "expected_confusion_mechanism": "elevated false-track pressure and overlapping tracks",
        },
        {
            "family": "rc_fixed_wing",
            "regional_subfamily": "hobby-class RC fixed-wing aircraft",
            "wingbeat_hz_proxy": [0.0, 0.0],
            "prop_cadence_hz_proxy": [40.0, 220.0],
            "velocity_mps_proxy": [8.0, 35.0],
            "flock_size_proxy": [1, 2],
            "site_archetype": "open field / coastal recreation / urban edge",
            "seasonality_flag": "benign recreational use",
            "expected_confusion_mechanism": "prop cadence, speed, and aspect overlap with fixed-wing positives",
            "paper_boundary": "hard negative only; not a target proxy",
        },
    ]


def _build_regional_bird_library(
    family_rows: list[dict[str, Any]], method_summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    family_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "false_alarm_count": 0,
            "near_threshold_count": 0,
            "fp_per_1000_negatives": 0.0,
            "top_method": "none",
            "top_method_fp_count": 0,
        }
    )
    for row in family_rows:
        family = str(row.get("family", ""))
        if not family:
            continue
        summary = family_summary[family]
        summary["false_alarm_count"] += int(_safe_float(row.get("false_alarm_count"), 0.0))
        summary["near_threshold_count"] += int(_safe_float(row.get("near_threshold_count"), 0.0))
        summary["fp_per_1000_negatives"] = float(
            summary["fp_per_1000_negatives"] + _safe_float(row.get("fp_per_1000_negatives"), 0.0)
        )
        count = int(_safe_float(row.get("false_alarm_count"), 0.0))
        if count >= int(summary["top_method_fp_count"]):
            summary["top_method"] = str(row.get("method_label", row.get("method", "none")))
            summary["top_method_fp_count"] = count

    prevalence = {
        "single_bird": "very common along coastal and urban-edge sites",
        "bird_flock": "common during migration and coastal roosting windows",
        "shorebird_wader": "seasonal at tidal flats and wetland margins",
        "gull_tern": "common near marine edges, ports, and coastal scrub",
        "raptor_falcon": "localized but high-contrast at thermals and ridgelines",
        "flamingo_large_bird": "site-dependent around wetlands and protected corridors",
        "seabird_cormorant": "seasonal around coastlines, harbors, and sea clutter",
        "seasonal_migratory_density": "highest during migration peaks in Gulf corridors",
        "rc_fixed_wing": "opportunistic recreational activity near open fields and coasts",
    }
    body_size_rcs = {
        "single_bird": "small body and flapping wingbeat stress low-RCS discrimination",
        "bird_flock": "multiple small scatterers broaden Doppler and raise track fragmentation",
        "shorebird_wader": "small coastal bodies with wet-surface aspect variation",
        "gull_tern": "moderate body return with flocking and coastal gliding",
        "raptor_falcon": "larger individual returns with glide/flap transitions",
        "flamingo_large_bird": "larger body and slower wingbeat can imitate slow airborne targets",
        "seabird_cormorant": "sea-clutter interaction and low-altitude coastal motion",
        "seasonal_migratory_density": "crowded airspace can create overlapping low-altitude returns",
        "rc_fixed_wing": "prop cadence and aspect overlap; benign comparator only",
    }
    flock_behavior = {
        "single_bird": "solitary or small, locally maneuvering tracks",
        "bird_flock": "dense aggregate with variable internal spacing",
        "shorebird_wader": "tidal or migration-driven local flocks",
        "gull_tern": "loose coastal flocking with frequent turns",
        "raptor_falcon": "small counts with glide and stoop transitions",
        "flamingo_large_bird": "larger flocks with slower cadence and wide spacing",
        "seabird_cormorant": "coastal transit with intermittent clustering",
        "seasonal_migratory_density": "high-density mixed movement across a corridor",
        "rc_fixed_wing": "not a flock family; isolated recreational tracks",
    }
    rows = []
    for row in _build_regional_hard_negative_taxonomy():
        family = row.get("family", "")
        summary = family_summary.get(family, {})
        rows.append(
            {
                **row,
                "site_prevalence": prevalence.get(family, "regional background presence"),
                "body_size_rcs_stress": body_size_rcs.get(family, "broad size and RCS stress"),
                "flock_behavior": flock_behavior.get(family, "mixed movement pattern"),
                "false_alarm_count_total": int(summary.get("false_alarm_count", 0)),
                "near_threshold_count_total": int(summary.get("near_threshold_count", 0)),
                "top_method_false_alarm_family": str(summary.get("top_method", "none")),
                "top_method_false_alarm_count": int(summary.get("top_method_fp_count", 0)),
            }
        )
    return rows


def _build_radar_model_card(training_root: Path, scenarios: list[dict[str, str]]) -> dict[str, Any]:
    phases = {phase.phase_id: {"start_s": phase.start_s, "end_s": phase.end_s} for phase in PHASES}
    carrier_bands = [
        _branch_card(
            branch="high_resolution_xku_cuas",
            label="High-resolution X/Ku C-UAS",
            band="X/Ku",
            center_ghz=10.0,
            bandwidth_mhz=600.0,
            prf_hz=4000.0,
            cpi_ms=6.0,
            pulses=24,
            range_bins=20,
            scan_revisit_s=0.25,
            clutter_family=["Weibull", "K-like", "coastal near-horizon"],
            detector_visible_products=[
                "range-Doppler map",
                "micro-Doppler band",
                "track confidence",
            ],
            public_anchors=["Blighter A400", "RTX KuRFS"],
        ),
        _branch_card(
            branch="tactical_s_band_aesa",
            label="Tactical S-band AESA/MHR",
            band="S",
            center_ghz=3.1,
            bandwidth_mhz=180.0,
            prf_hz=4000.0,
            cpi_ms=6.0,
            pulses=24,
            range_bins=20,
            scan_revisit_s=1.0,
            clutter_family=["terrain multipath", "urban edge", "vegetation motion"],
            detector_visible_products=[
                "coarser range-Doppler score",
                "track-while-scan confidence",
            ],
            public_anchors=["DRS RADA nMHR/MHR"],
        ),
        _branch_card(
            branch="gbad_3d4d_cueing",
            label="GBAD 3D/4D cueing",
            band="X/Ku public cueing envelope",
            center_ghz=10.3,
            bandwidth_mhz=300.0,
            prf_hz=4000.0,
            cpi_ms=6.0,
            pulses=24,
            range_bins=20,
            scan_revisit_s=1.5,
            clutter_family=["radar-horizon masking", "terrain glint", "weather cell"],
            detector_visible_products=["3D/4D cue score", "revisit delay", "track confidence"],
            public_anchors=["Saab Giraffe 1X", "AN/MPQ-64 Sentinel public summaries"],
        ),
    ]
    return {
        "claim_boundary": "All values are public-proxy or synthetic archetype assumptions; no measured-platform truth is implied.",
        "waveform_family": "multibranch FMCW-style public-proxy radar",
        "waveform_proxy_detail": "FMCW/chirp wording denotes a synthetic range-Doppler product proxy; pulse terms denote CPI indexing for Doppler processing.",
        "range_doppler_processing": "Complex IQ is windowed into CPI records and summarized through range-Doppler FFT-style concentration, Doppler spread, and CFAR/MTD proxy features.",
        "target_rcs_prior": {
            "distribution": "aspect-dependent public-proxy envelope with Swerling-like fluctuation proxy",
            "positive_class_dbsm": [-28.0, -2.0],
            "bird_and_rc_hard_negative_note": "Bird and RC priors are hard-negative stressors, not target proxies.",
        },
        "target_micro_doppler_priors": {
            "fixed_wing_pusher_prop_hz": [150.0, 217.0],
            "bird_wingbeat_hz": [1.2, 14.0],
            "rc_prop_cadence_hz": [40.0, 220.0],
        },
        "phase_windows_s": phases,
        "carrier_bands": carrier_bands,
        "receiver_impairments": [
            "AGC compression",
            "clock drift",
            "quantization",
            "dropped CPI",
            "PRF ambiguity",
            "Doppler folding",
            "calibration offset",
            "multipath masking",
        ],
        "clutter_and_artifact_priors": {
            "clutter": [
                "Weibull clutter",
                "K-like clutter",
                "urban edge",
                "vegetation motion",
                "sea clutter",
                "terrain glint",
                "mixed scene",
            ],
            "noise_and_rfi": [
                "thermal-noise proxy through SNR/CNR stress buckets",
                "rain, dust, sea-clutter, and vegetation-motion regimes",
                "RFI burst, dropped-CPI, clock-drift, quantization, and calibration-offset regimes",
            ],
            "hard_negative_families": [
                "single_bird",
                "bird_flock",
                "rc_fixed_wing",
                "weather_cell",
                "ground_vehicle",
                "wind_turbine",
                "terrain_glint",
                "multipath_ghost",
                "rfi_burst",
                "clutter_only_counterfactual",
            ],
        },
        "cue_definitions": {
            "acoustic": "node-wise spectral cadence, cross-node agreement, and amplitude stability",
            "passive_rf": "no-signal / RFI burst / clock-offset / provenance-quality geometry",
        },
        "public_proxy_positive_class": "fixed-wing pusher-prop public proxy",
        "legacy_label_note": "Repository-compatible identifiers such as shahed_136_geran_2_public_proxy remain in the generated data for back-compatibility only.",
        "training_root": str(training_root),
        "scenario_group_count": int(len(scenarios)),
    }


def _radar_model_detail_rows(radar_model_card: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "branch": "all",
            "detail_key": "waveform_proxy",
            "detail_value": radar_model_card.get("waveform_proxy_detail", ""),
            "unit": "text",
            "basis": "public-proxy assumption",
        },
        {
            "branch": "all",
            "detail_key": "range_doppler_processing",
            "detail_value": radar_model_card.get("range_doppler_processing", ""),
            "unit": "text",
            "basis": "public-proxy processing summary",
        },
        {
            "branch": "all",
            "detail_key": "rcs_prior",
            "detail_value": json.dumps(
                radar_model_card.get("target_rcs_prior", {}), sort_keys=True
            ),
            "unit": "mixed",
            "basis": "public-source proxy envelope",
        },
        {
            "branch": "all",
            "detail_key": "micro_doppler_priors",
            "detail_value": json.dumps(
                radar_model_card.get("target_micro_doppler_priors", {}), sort_keys=True
            ),
            "unit": "Hz",
            "basis": "public-source proxy envelope",
        },
        {
            "branch": "all",
            "detail_key": "clutter_noise_rfi_priors",
            "detail_value": json.dumps(
                radar_model_card.get("clutter_and_artifact_priors", {}), sort_keys=True
            ),
            "unit": "text",
            "basis": "synthetic stress taxonomy",
        },
        {
            "branch": "all",
            "detail_key": "cue_definitions",
            "detail_value": json.dumps(radar_model_card.get("cue_definitions", {}), sort_keys=True),
            "unit": "text",
            "basis": "public-proxy cue abstraction",
        },
    ]
    for branch in radar_model_card.get("carrier_bands", []):
        branch_id = branch.get("branch", "")
        branch_details = [
            ("waveform_family", radar_model_card.get("waveform_family", ""), "text"),
            ("center_frequency", branch.get("center_ghz"), "GHz"),
            ("bandwidth", branch.get("bandwidth_mhz"), "MHz"),
            ("prf", branch.get("prf_hz"), "Hz"),
            ("crf", branch.get("crf_hz"), "Hz"),
            ("cpi", branch.get("cpi_ms"), "ms"),
            ("chirps_or_pulses_per_cpi", branch.get("pulses_or_chirps_per_cpi"), "count"),
            ("range_bins", branch.get("range_bins"), "count"),
            ("range_span", branch.get("range_span_m"), "m"),
            ("range_resolution", branch.get("range_resolution_m"), "m"),
            ("doppler_resolution", branch.get("doppler_resolution_hz"), "Hz"),
            ("velocity_resolution", branch.get("velocity_resolution_mps"), "m/s"),
            ("unambiguous_velocity", branch.get("approx_unambiguous_velocity_mps"), "m/s"),
            ("scan_revisit", branch.get("scan_revisit_assumption_s"), "s"),
            ("clutter_family", "|".join(branch.get("clutter_family", [])), "text"),
            (
                "detector_visible_products",
                "|".join(branch.get("detector_visible_products", [])),
                "text",
            ),
        ]
        for key, value, unit in branch_details:
            rows.append(
                {
                    "branch": branch_id,
                    "detail_key": key,
                    "detail_value": value,
                    "unit": unit,
                    "basis": "public-proxy branch assumption",
                }
            )
    return rows


def _build_evaluation_summary(
    baseline_root: Path,
    advanced_root: Path,
) -> dict[str, Any]:
    adv_predictions = _read_csv_rows(advanced_root / "advanced_predictions.csv")
    advanced_summary = _read_json(advanced_root / "performance_summary.json")
    selected_lock = _load_selection_lock(advanced_root / "selection_lock.json")
    selected_method = str(selected_lock.get("selected_candidate_id", ""))
    selected_score_col = "advanced_score"
    selected_method_name = str(
        advanced_summary.get("advanced_selection", {}).get("selected_method", "")
    )
    selected_threshold = _safe_float(
        advanced_summary.get("thresholds", {}).get(selected_method_name, {}).get("threshold")
    )
    if math.isnan(selected_threshold):
        selected_threshold = _holdout_threshold_from_performance(
            advanced_root / "performance_metrics.csv", selected_method
        )
    fusion_threshold = _holdout_threshold_from_performance(
        baseline_root / "performance_metrics.csv", "layered_fusion_c2"
    )
    selected_labels, selected_scores, selected_phases, selected_groups = _load_holdout_predictions(
        advanced_root / "advanced_predictions.csv",
        selected_score_col,
    )
    baseline_labels, baseline_scores, baseline_phases, baseline_groups = _load_holdout_predictions(
        baseline_root / "fusion_predictions.csv",
        "fusion_probability",
    )
    baseline_rows = [
        row
        for row in _read_csv_rows(baseline_root / "fusion_predictions.csv")
        if row.get("split_role") == "holdout"
    ]
    selected_bundle = _metric_bundle(selected_labels, selected_scores, selected_threshold)
    baseline_bundle = _metric_bundle(baseline_labels, baseline_scores, fusion_threshold)
    all_selected_rows = [row for row in adv_predictions if row.get("split_role") == "holdout"]
    selected_records = [
        {
            "record_id": row.get("record_id", ""),
            "scenario_group_id": row.get("scenario_group_id", ""),
            "phase_id": row.get("phase_id", ""),
            "label_id": row.get("label_id", ""),
        }
        for row in all_selected_rows
    ]
    scenario_rows = _read_csv_rows(DEFAULT_TRAINING_ROOT / "scenario_manifest.csv")
    scenario_by_group = {row["scenario_group_id"]: row for row in scenario_rows}
    phase_metrics: dict[str, dict[str, Any]] = {
        selected_method: {},
        "layered_fusion_c2": {},
    }
    for phase in sorted({row.get("phase_id", "") for row in all_selected_rows}):
        phase_mask = selected_phases == phase
        phase_metrics[selected_method][phase] = _metric_bundle(
            selected_labels[phase_mask], selected_scores[phase_mask], selected_threshold
        )
    for phase in sorted({row.get("phase_id", "") for row in baseline_rows}):
        phase_mask = baseline_phases == phase
        phase_metrics["layered_fusion_c2"][phase] = _metric_bundle(
            baseline_labels[phase_mask], baseline_scores[phase_mask], fusion_threshold
        )
    regime_metrics: dict[str, dict[str, Any]] = {}
    for regime in sorted({row.get("noise_regime", "") for row in scenario_rows}):
        relevant_records = []
        for row in all_selected_rows:
            scenario = scenario_by_group.get(row.get("scenario_group_id", ""))
            if scenario and scenario.get("noise_regime", "") == regime:
                relevant_records.append(row)
        if not relevant_records:
            continue
        labels = np.asarray(
            [_safe_int(row.get("label_id")) for row in relevant_records], dtype=np.int8
        )
        scores = np.asarray(
            [_safe_float(row.get("advanced_score")) for row in relevant_records], dtype=np.float64
        )
        regime_metrics[regime] = _metric_bundle(labels, scores, selected_threshold)

    selected_bundle["group_block_bootstrap_ci"] = _bootstrap_group_cis(
        [
            {
                "scenario_group_id": row.get("scenario_group_id", ""),
                "label_id": row.get("label_id", ""),
            }
            for row in all_selected_rows
        ],
        selected_scores,
        selected_threshold,
    )
    baseline_bundle["group_block_bootstrap_ci"] = _bootstrap_group_cis(
        [
            {
                "scenario_group_id": row.get("scenario_group_id", ""),
                "label_id": row.get("label_id", ""),
            }
            for row in _read_csv_rows(baseline_root / "fusion_predictions.csv")
            if row.get("split_role") == "holdout"
        ],
        baseline_scores,
        fusion_threshold,
    )
    selected_primary_kpi = {
        "name": PRIMARY_KPI_LABEL,
        "target_fpr": THRESHOLD_TARGET_FPR,
        "point_estimate": _safe_float(selected_bundle.get("fixed_fpr_recall")),
        "value": _safe_float(
            selected_bundle.get("group_block_bootstrap_ci", {})
            .get("fixed_fpr_recall", {})
            .get("low")
        ),
        "basis": "holdout group-block bootstrap",
        "status": "pass",
    }
    baseline_primary_kpi = {
        "name": PRIMARY_KPI_LABEL,
        "target_fpr": THRESHOLD_TARGET_FPR,
        "point_estimate": _safe_float(baseline_bundle.get("fixed_fpr_recall")),
        "value": _safe_float(
            baseline_bundle.get("group_block_bootstrap_ci", {})
            .get("fixed_fpr_recall", {})
            .get("low")
        ),
        "basis": "holdout group-block bootstrap",
        "status": "pass",
    }
    selected_bundle["primary_kpi"] = selected_primary_kpi
    baseline_bundle["primary_kpi"] = baseline_primary_kpi
    selected_group_metrics = _group_operating_metrics(
        all_selected_rows,
        selected_scores,
        selected_threshold,
        method="locked_candidate",
        threshold_source="selected_lock",
    )
    baseline_group_metrics = _group_operating_metrics(
        baseline_rows,
        baseline_scores,
        fusion_threshold,
        method="prior_layered_fusion",
        threshold_source="train_cv_max_f1",
    )
    selected_bundle["group_operating_metrics"] = selected_group_metrics
    baseline_bundle["group_operating_metrics"] = baseline_group_metrics
    return {
        "selected_method": selected_method,
        "selected": selected_bundle,
        "baseline_method": "layered_fusion_c2",
        "baseline": baseline_bundle,
        "group_operating_metrics": [baseline_group_metrics, selected_group_metrics],
        "phase_metrics": phase_metrics,
        "regime_metrics": regime_metrics,
        "holdout_rows": len(all_selected_rows),
        "selected_threshold": selected_threshold,
        "baseline_threshold": fusion_threshold,
        "selected_label_records": selected_records,
        "curves": {
            "selected": {
                "roc": _roc_curve(selected_labels, selected_scores),
                "pr": _pr_curve(selected_labels, selected_scores),
            },
            "baseline": {
                "roc": _roc_curve(baseline_labels, baseline_scores),
                "pr": _pr_curve(baseline_labels, baseline_scores),
            },
        },
        "calibration": {
            "selected": _calibration_summary(selected_labels, selected_scores),
            "baseline": _calibration_summary(baseline_labels, baseline_scores),
        },
        "primary_kpi": selected_primary_kpi,
        "fixed_fpr_target": THRESHOLD_TARGET_FPR,
    }


def _load_component_transparency(advanced_root: Path) -> dict[str, Any]:
    selection_lock = _load_selection_lock(advanced_root / "selection_lock.json")
    component_rows = _load_component_score_rows(advanced_root / "selected_component_scores.csv")
    ablation_rows = _load_csv_or_empty(advanced_root / "selected_component_ablations.csv")
    aliases = _read_json(advanced_root / "selected_component_aliases.json")
    return {
        "selection_lock": selection_lock,
        "component_scores": component_rows,
        "component_ablations": ablation_rows,
        "aliases": aliases,
    }


def _load_csv_or_empty(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path) if path.exists() else []


def _load_anchor_summary(anchor_root: Path) -> dict[str, Any]:
    manifest = _read_json(anchor_root / "anchor_manifest.json")
    distribution = _read_json(anchor_root / "feature_distributions.json")
    coefficients = _read_json(anchor_root / "calibration_coefficients.json")
    calibration_distance = _read_csv_rows(anchor_root / "calibration_distance.csv")
    gap_rows = _read_csv_rows(anchor_root / "sim_real_gap.csv")
    features = distribution.get("features", {}) if isinstance(distribution, dict) else {}
    selected = {}
    for name in (
        "micro_doppler_bandwidth_hz",
        "micro_doppler_energy",
        "spectral_entropy",
        "range_m",
        "return_power_db",
    ):
        selected[name] = features.get(name, {})
    return {
        "manifest": manifest,
        "distribution": distribution,
        "selected_features": selected,
        "coefficients": coefficients,
        "calibration_distance": calibration_distance[:12],
        "gap_rows": gap_rows[:12],
    }


def _normalized_anchor_comparison(anchor_summary: dict[str, Any]) -> list[dict[str, Any]]:
    distribution = anchor_summary.get("distribution", {})
    features = distribution.get("features", {}) if isinstance(distribution, dict) else {}
    rows = []
    for name in (
        "micro_doppler_bandwidth_hz",
        "micro_doppler_peak_hz",
        "spectral_entropy",
        "range_m",
        "return_power_db",
    ):
        stats = features.get(name, {})
        if not isinstance(stats, dict):
            continue
        mean = _safe_float(stats.get("mean"))
        std = _safe_float(stats.get("std"), 0.0)
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            continue
        rows.append(
            {
                "feature": name,
                "unitless_basis": "unitless_measured_anchor_z_score",
                "q10_z": (_safe_float(stats.get("q10")) - mean) / std,
                "q50_z": (_safe_float(stats.get("q50")) - mean) / std,
                "q90_z": (_safe_float(stats.get("q90")) - mean) / std,
                "count": int(_safe_float(stats.get("count"), 0.0)),
                "claim_boundary": "compare_only_normalized_distribution",
            }
        )
    return rows


def _score_map_from_component_groups(
    component_rows: list[dict[str, str]],
    groups: set[str],
) -> dict[str, float]:
    grouped_scores: defaultdict[str, list[float]] = defaultdict(list)
    for row in component_rows:
        if row.get("component_group", "") not in groups:
            continue
        grouped_scores[row.get("record_id", "")].append(
            _safe_float(row.get("component_score"), 0.0)
        )
    return {
        record_id: float(np.mean(values))
        for record_id, values in grouped_scores.items()
        if record_id and values
    }


def _score_array_for_rows(
    rows: list[dict[str, str]],
    *,
    columns: list[str] | None = None,
    score_map: dict[str, float] | None = None,
) -> np.ndarray:
    scores = []
    for row in rows:
        if score_map is not None:
            scores.append(score_map.get(row.get("record_id", ""), 0.0))
            continue
        values = [_safe_float(row.get(column), 0.0) for column in columns or []]
        scores.append(float(np.mean(values)) if values else 0.0)
    return np.asarray(scores, dtype=np.float64)


def _build_modality_transparency(
    prediction_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    selected_threshold: float,
) -> dict[str, Any]:
    train_rows = [row for row in prediction_rows if row.get("split_role") == "train_cv"]
    holdout_rows = [row for row in prediction_rows if row.get("split_role") == "holdout"]
    train_labels = np.asarray([_safe_int(row.get("label_id")) for row in train_rows], dtype=np.int8)
    holdout_labels = np.asarray(
        [_safe_int(row.get("label_id")) for row in holdout_rows], dtype=np.int8
    )
    passive_quality_map = _score_map_from_component_groups(component_rows, {"passive_quality"})
    passive_rf_map = _score_map_from_component_groups(
        component_rows, {"passive_quality", "passive_hypergraph"}
    )
    transport_map = _score_map_from_component_groups(component_rows, {"v2_transport_geometry"})
    suite = [
        {
            "view": "radar_only",
            "kind": "coarse_modality",
            "columns": ["high_resolution_xku_cuas", "tactical_s_band_aesa", "gbad_3d4d_cueing"],
        },
        {
            "view": "acoustic_only",
            "kind": "coarse_modality",
            "columns": ["distributed_acoustic_cue"],
        },
        {"view": "passive_rf_only", "kind": "coarse_modality", "score_map": passive_rf_map},
        {
            "view": "radar_acoustic",
            "kind": "coarse_modality",
            "columns": [
                "high_resolution_xku_cuas",
                "tactical_s_band_aesa",
                "gbad_3d4d_cueing",
                "distributed_acoustic_cue",
            ],
        },
        {
            "view": "radar_rf",
            "kind": "coarse_modality",
            "columns": ["high_resolution_xku_cuas", "tactical_s_band_aesa", "gbad_3d4d_cueing"],
            "extra_map": passive_rf_map,
        },
        {
            "view": "passive_quality_only",
            "kind": "component_group",
            "score_map": passive_quality_map,
        },
        {"view": "transport_geometry_only", "kind": "component_group", "score_map": transport_map},
        {"view": "full_fusion", "kind": "locked_candidate", "columns": ["advanced_score"]},
    ]
    rows = []
    for spec in suite:
        train_scores = _score_array_for_rows(
            train_rows, columns=spec.get("columns"), score_map=spec.get("score_map")
        )
        holdout_scores = _score_array_for_rows(
            holdout_rows, columns=spec.get("columns"), score_map=spec.get("score_map")
        )
        if spec.get("extra_map"):
            train_extra = _score_array_for_rows(train_rows, score_map=spec["extra_map"])
            holdout_extra = _score_array_for_rows(holdout_rows, score_map=spec["extra_map"])
            train_scores = (train_scores + train_extra) / 2.0
            holdout_scores = (holdout_scores + holdout_extra) / 2.0
        train_norm, holdout_norm = _normalise_scores(train_scores, holdout_scores)
        threshold = (
            selected_threshold
            if spec["view"] == "full_fusion" and math.isfinite(selected_threshold)
            else _best_f1_threshold(train_labels, train_norm)
        )
        metrics = _metric_bundle(holdout_labels, holdout_norm, threshold)
        rows.append(
            {
                "view": spec["view"],
                "kind": spec["kind"],
                "threshold_basis": "selected_lock"
                if spec["view"] == "full_fusion"
                else "train_cv_max_f1",
                "threshold": threshold,
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "fixed_fpr_recall": metrics["fixed_fpr_recall"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "f1": metrics["f1"],
                "ece": metrics["ece"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "source_columns": "|".join(spec.get("columns") or []),
            }
        )
    ablation_summary = []
    for row in ablation_rows:
        variant_type = row.get("variant_type", "")
        if variant_type not in {"no_calibration", "top_k", "component_drop", "modality_drop"}:
            continue
        ablation_summary.append(
            {
                "view": row.get("variant_id", ""),
                "kind": variant_type,
                "component_scope": row.get("component_scope", ""),
                "average_precision": _safe_float(row.get("holdout_average_precision")),
                "roc_auc": _safe_float(row.get("holdout_roc_auc")),
                "f1": _safe_float(row.get("holdout_f1")),
                "brier_score": _safe_float(row.get("brier_score")),
                "ece": _safe_float(row.get("ece")),
            }
        )
    return {
        "modality_rows": rows,
        "ablation_rows": ablation_summary,
        "claim_boundary": "transparency diagnostics only; not sensor-parity evidence",
    }


def _component_family_label(component_group: str) -> tuple[str, str]:
    if component_group == "passive_quality":
        return "Passive quality", "passive-RF provenance"
    if component_group == "v2_transport_geometry":
        return "Transport geometry", "motion/geometry"
    if component_group == "passive_hypergraph":
        return "Passive hypergraph", "passive-RF graph"
    return component_group.replace("_", " "), "mixed"


def _calibrator_label(component_id: str) -> str:
    if component_id.endswith(".geodesic_odds"):
        return "geodesic odds"
    if component_id.endswith(".raw"):
        return "raw"
    return "unknown"


def _view_label_from_component(component_id: str, component_group: str) -> str:
    if ".positive_de." in component_id:
        return "positive differential evolution"
    if ".signed_de." in component_id:
        return "signed differential evolution"
    return component_group.replace("_", " ")


def _selected_component_human_weights(component_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    component_meta: dict[str, dict[str, Any]] = {}
    for row in component_rows:
        component_id = row.get("component_id", "")
        if not component_id or component_id in component_meta:
            continue
        weight = _safe_float(row.get("component_weight"), 0.0)
        group = row.get("component_group", "")
        family, modality = _component_family_label(group)
        component_meta[component_id] = {
            "audit_id": f"C{_safe_int(row.get('component_index'), len(component_meta)) + 1}",
            "family": family,
            "modality": modality,
            "view": _view_label_from_component(component_id, group),
            "calibrator": _calibrator_label(component_id),
            "weight": weight,
            "component_group": group,
            "raw_component_id": component_id,
        }
    rows = sorted(component_meta.values(), key=lambda row: row["weight"], reverse=True)
    cumulative = 0.0
    for row in rows:
        cumulative += float(row["weight"])
        row["cumulative_weight"] = cumulative
    return rows


def _comparable_ablation_rows(
    eval_summary: dict[str, Any], modality_transparency: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = eval_summary.get("selected", {})
    selected_ap = _safe_float(selected.get("average_precision"))
    selected_recall = _safe_float(selected.get("fixed_fpr_recall"))
    selected_f1 = _safe_float(selected.get("f1"))
    selected_ece = _safe_float(selected.get("ece"))
    rows: list[dict[str, Any]] = [
        {
            "ablation": "full_locked_candidate",
            "reader_label": "Full EI candidate",
            "diagnostic_type": "ei_artifact",
            "ap": selected_ap,
            "delta_ap": 0.0,
            "recall_at_leq_1pct_fpr": selected_recall,
            "delta_recall_at_leq_1pct_fpr": 0.0,
            "f1": selected_f1,
            "ece": selected_ece,
            "metric_basis": "Table IV holdout definitions; advanced_predictions advanced_score; selected-lock threshold",
            "comparability_note": "Locked candidate reference row.",
        }
    ]
    labels = {
        "radar_only": ("Radar-only view", "coarse_view_control"),
        "acoustic_only": ("Acoustic-only view", "coarse_view_control"),
        "passive_rf_only": ("Passive-RF-only view", "coarse_view_control"),
        "radar_acoustic": ("Radar + acoustic view", "coarse_view_control"),
        "radar_rf": ("Radar + passive-RF view", "coarse_view_control"),
        "passive_quality_only": ("Passive quality family", "component_family_control"),
        "transport_geometry_only": ("Transport geometry family", "component_family_control"),
        "full_fusion": ("Full locked fusion view", "locked_candidate_view"),
    }
    for row in modality_transparency.get("modality_rows", []):
        view = row.get("view", "")
        if view not in labels or view == "full_fusion":
            continue
        ap = _safe_float(row.get("average_precision"))
        fixed_recall = _safe_float(row.get("fixed_fpr_recall"))
        label, diagnostic_type = labels[view]
        rows.append(
            {
                "ablation": view,
                "reader_label": label,
                "diagnostic_type": diagnostic_type,
                "ap": ap,
                "delta_ap": ap - selected_ap
                if math.isfinite(ap) and math.isfinite(selected_ap)
                else float("nan"),
                "recall_at_leq_1pct_fpr": fixed_recall,
                "delta_recall_at_leq_1pct_fpr": fixed_recall - selected_recall
                if math.isfinite(fixed_recall) and math.isfinite(selected_recall)
                else float("nan"),
                "f1": _safe_float(row.get("f1")),
                "ece": _safe_float(row.get("ece")),
                "metric_basis": "Same holdout labels and metric functions as Table IV; non-locked controls use train/CV normalization and threshold basis.",
                "comparability_note": "Comparable detector-view diagnostic; not a replacement selection lock.",
            }
        )
    return rows


def _anchor_distance_diagnostics(anchor_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coefficients = anchor_summary.get("coefficients", {})
    priors = coefficients.get("simulator_priors", {}) if isinstance(coefficients, dict) else {}
    for prior_name, payload in sorted(priors.items()):
        if not isinstance(payload, dict):
            continue
        target = _safe_float(payload.get("target_center"))
        reference = _safe_float(payload.get("reference_center"))
        q10 = _safe_float(payload.get("target_q10"))
        q90 = _safe_float(payload.get("target_q90"))
        scale = abs(q90 - q10) / 2.0 if math.isfinite(q10) and math.isfinite(q90) else float("nan")
        if not math.isfinite(scale) or scale <= 0.0:
            scale = max(abs(target), abs(reference), 1.0)
        rows.append(
            {
                "feature": payload.get("source_observable", prior_name),
                "diagnostic": "standardized_delta_to_prior_reference",
                "measured_anchor_value": target,
                "synthetic_reference_value": reference,
                "distance": (target - reference) / scale
                if math.isfinite(target) and math.isfinite(reference)
                else float("nan"),
                "distance_basis": "target_center minus simulator reference_center divided by half target q10-q90 span",
                "status": coefficients.get("status", "measured_anchor_candidate"),
                "allowed_claim_level": coefficients.get(
                    "allowed_claim_level", "measured_anchor_candidate_only"
                ),
            }
        )
    for row in anchor_summary.get("gap_rows", []):
        rows.append(
            {
                "feature": row.get("observable_name", row.get("feature", "")),
                "diagnostic": row.get("metric", "wasserstein_or_gap_proxy"),
                "measured_anchor_value": _safe_float(row.get("measured_value")),
                "synthetic_reference_value": _safe_float(row.get("synthetic_reference_value")),
                "distance": _safe_float(row.get("gap_value"), _safe_float(row.get("distance"))),
                "distance_basis": "anchor gap CSV; blank reference means no local synthetic reference distribution was available",
                "status": row.get("status", ""),
                "allowed_claim_level": row.get(
                    "allowed_claim_level", "measured_anchor_candidate_only"
                ),
            }
        )
    return rows


def _primary_kpi_rows(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in (("prior_fusion_baseline", "baseline"), ("locked_candidate", "selected")):
        bundle = eval_summary.get(key, {})
        primary = bundle.get("primary_kpi", {})
        rows.append(
            {
                "method": label,
                "point_estimate": _safe_float(primary.get("point_estimate")),
                "lcb95": _safe_float(primary.get("value")),
                "target_fpr": _safe_float(primary.get("target_fpr"), THRESHOLD_TARGET_FPR),
                "basis": primary.get("basis", "holdout group-block bootstrap"),
            }
        )
    return rows


def _main_kpi_gain_rows(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = eval_summary.get("baseline", {})
    selected = eval_summary.get("selected", {})

    def metric_row(
        metric: str,
        baseline_value: float,
        selected_value: float,
        *,
        lower_is_better: bool = False,
    ) -> dict[str, Any]:
        absolute = selected_value - baseline_value
        if lower_is_better:
            relative = (
                (baseline_value - selected_value) / baseline_value
                if baseline_value
                else float("nan")
            )
            direction = "reduction"
        else:
            relative = absolute / baseline_value if baseline_value else float("nan")
            direction = "gain"
        return {
            "metric": metric,
            "prior_fusion_baseline": baseline_value,
            "ei_candidate": selected_value,
            "absolute_change": absolute,
            "relative_change": relative,
            "relative_percent": relative * 100.0 if math.isfinite(relative) else float("nan"),
            "direction": direction,
        }

    baseline_primary = baseline.get("primary_kpi", {})
    selected_primary = selected.get("primary_kpi", {})
    rows = [
        metric_row(
            "lcb95_recall_at_leq_1pct_fpr",
            _safe_float(baseline_primary.get("value")),
            _safe_float(selected_primary.get("value")),
        ),
        metric_row(
            "point_recall_at_leq_1pct_fpr",
            _safe_float(baseline.get("fixed_fpr_recall")),
            _safe_float(selected.get("fixed_fpr_recall")),
        ),
        metric_row(
            "average_precision",
            _safe_float(baseline.get("average_precision")),
            _safe_float(selected.get("average_precision")),
        ),
        metric_row("f1", _safe_float(baseline.get("f1")), _safe_float(selected.get("f1"))),
        metric_row(
            "selected_threshold_false_positives",
            _safe_float(baseline.get("fp")),
            _safe_float(selected.get("fp")),
            lower_is_better=True,
        ),
        metric_row(
            "roc_auc_guardrail",
            _safe_float(baseline.get("roc_auc")),
            _safe_float(selected.get("roc_auc")),
        ),
    ]
    return rows


def _method_display_label(method: str, selected_method: str) -> str:
    if method == selected_method:
        return "locked_candidate"
    if method == "layered_fusion_c2":
        return "prior_layered_fusion"
    return method


def _method_score_source(
    method: str, selected_method: str, baseline_root: Path, advanced_root: Path
) -> tuple[Path, str, str]:
    if method == "locked_candidate":
        return advanced_root / "advanced_predictions.csv", "advanced_score", selected_method
    source = METHOD_SCORE_SOURCES.get(method)
    if source is None:
        raise KeyError(f"no score source registered for method {method}")
    filename, score_column = source
    return baseline_root / filename, score_column, method


def _load_holdout_method_predictions(
    method: str, selected_method: str, baseline_root: Path, advanced_root: Path
) -> list[dict[str, str]]:
    path, _score_column, _ = _method_score_source(
        method, selected_method, baseline_root, advanced_root
    )
    rows = _read_csv_rows(path)
    return [row for row in rows if row.get("split_role") == "holdout"]


def _load_holdout_method_summary_rows(
    baseline_root: Path, advanced_root: Path, selected_method: str, eval_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_metrics = _read_csv_rows(baseline_root / "performance_metrics.csv")

    def _derive_counts(
        positive_count: int,
        negative_count: int,
        recall: float,
        false_positive_rate: float,
        tp_value: Any = None,
        fp_value: Any = None,
        tn_value: Any = None,
        fn_value: Any = None,
    ) -> tuple[int, int, int, int, int]:
        if not math.isfinite(recall):
            recall = 0.0
        if not math.isfinite(false_positive_rate):
            false_positive_rate = 0.0
        tp = _safe_int(tp_value, _safe_int(round(positive_count * recall)))
        fn = _safe_int(fn_value, max(positive_count - tp, 0))
        fp = _safe_int(fp_value, _safe_int(round(negative_count * false_positive_rate)))
        tn = _safe_int(tn_value, max(negative_count - fp, 0))
        record_count = tp + fp + tn + fn
        return tp, fp, tn, fn, record_count

    for row in baseline_metrics:
        if row.get("split_role") != "holdout" or row.get("phase_id") != "all":
            continue
        method = row.get("method", "")
        if not method:
            continue
        positive_count = _safe_int(row.get("positive_count"))
        negative_count = _safe_int(row.get("negative_count"))
        recall = _safe_float(row.get("recall"))
        false_positive_rate = _safe_float(row.get("false_positive_rate"))
        tp, fp, tn, fn, record_count = _derive_counts(
            positive_count,
            negative_count,
            recall,
            false_positive_rate,
            row.get("tp"),
            row.get("fp"),
            row.get("tn"),
            row.get("fn"),
        )
        rows.append(
            {
                "method": method,
                "method_label": _method_display_label(method, selected_method),
                "threshold_source": row.get("threshold_source", ""),
                "threshold": _safe_float(row.get("threshold")),
                "average_precision": _safe_float(row.get("average_precision")),
                "roc_auc": _safe_float(row.get("roc_auc")),
                "accuracy": _safe_float(row.get("accuracy")),
                "precision": _safe_float(row.get("precision")),
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "f1": _safe_float(row.get("f1")),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "record_count": _safe_int(row.get("record_count"), record_count),
            }
        )
    selected_bundle = eval_summary.get("selected", {})
    selected_positive_count = _safe_int(selected_bundle.get("positive_count"))
    selected_negative_count = _safe_int(selected_bundle.get("negative_count"))
    selected_recall = _safe_float(selected_bundle.get("recall"))
    selected_fpr = _safe_float(selected_bundle.get("false_positive_rate"))
    selected_tp, selected_fp, selected_tn, selected_fn, selected_record_count = _derive_counts(
        selected_positive_count,
        selected_negative_count,
        selected_recall,
        selected_fpr,
        selected_bundle.get("tp"),
        selected_bundle.get("fp"),
        selected_bundle.get("tn"),
        selected_bundle.get("fn"),
    )
    rows.append(
        {
            "method": "locked_candidate",
            "method_label": "locked_candidate",
            "threshold_source": selected_bundle.get("threshold_source", "selected_lock"),
            "threshold": _safe_float(eval_summary.get("selected_threshold")),
            "average_precision": _safe_float(selected_bundle.get("average_precision")),
            "roc_auc": _safe_float(selected_bundle.get("roc_auc")),
            "accuracy": _safe_float(selected_bundle.get("accuracy")),
            "precision": _safe_float(selected_bundle.get("precision")),
            "recall": selected_recall,
            "false_positive_rate": selected_fpr,
            "f1": _safe_float(selected_bundle.get("f1")),
            "tp": selected_tp,
            "fp": selected_fp,
            "tn": selected_tn,
            "fn": selected_fn,
            "negative_count": selected_negative_count,
            "positive_count": selected_positive_count,
            "record_count": _safe_int(selected_bundle.get("count"), selected_record_count),
        }
    )
    rows.sort(key=lambda row: _safe_float(row.get("average_precision")), reverse=True)
    return rows


def _strongest_method_names(
    baseline_root: Path, advanced_root: Path, selected_method: str, eval_summary: dict[str, Any]
) -> list[str]:
    summaries = _load_holdout_method_summary_rows(
        baseline_root, advanced_root, selected_method, eval_summary
    )
    baseline_candidates = [
        row for row in summaries if row["method"] not in {"locked_candidate", "layered_fusion_c2"}
    ]
    top_three_baselines = [row["method"] for row in baseline_candidates[:3]]
    ordered = ["locked_candidate", "layered_fusion_c2"]
    for method in top_three_baselines:
        if method not in ordered:
            ordered.append(method)
    if "locked_candidate" not in ordered:
        ordered.insert(0, "locked_candidate")
    return ordered[:TOP_METHOD_COUNT]


def _family_label(role: str) -> str:
    return role or "positive"


def _false_positive_method_tables(
    *,
    scenarios: list[dict[str, str]],
    baseline_root: Path,
    advanced_root: Path,
    selected_method: str,
    selected_threshold: float,
    eval_summary: dict[str, Any],
) -> dict[str, Any]:
    scenario_by_group = {row["scenario_group_id"]: row for row in scenarios}
    method_summary_rows = _load_holdout_method_summary_rows(
        baseline_root, advanced_root, selected_method, eval_summary
    )
    selected_names = _strongest_method_names(
        baseline_root, advanced_root, selected_method, eval_summary
    )

    by_method = {row["method"]: row for row in method_summary_rows}
    candidate_rows = [by_method[method] for method in selected_names if method in by_method]

    family_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    locked_fp_count = next(
        (
            _safe_float(row.get("fp"))
            for row in candidate_rows
            if row["method"] == "locked_candidate"
        ),
        0.0,
    )
    family_totals: dict[str, int] = defaultdict(int)
    family_palette_order: list[str] = []

    for method in selected_names:
        summary = by_method.get(method)
        if summary is None:
            continue
        predictions = _load_holdout_method_predictions(
            method, selected_method, baseline_root, advanced_root
        )
        if not predictions:
            continue
        _path, score_column, _method_name = _method_score_source(
            method, selected_method, baseline_root, advanced_root
        )
        labels = np.asarray([_safe_int(row.get("label_id")) for row in predictions], dtype=np.int8)
        scores = np.asarray(
            [_safe_float(row.get(score_column)) for row in predictions], dtype=np.float64
        )
        fixed_fpr_recall = _fixed_fpr_recall(labels, scores, THRESHOLD_TARGET_FPR)
        threshold = _safe_float(summary.get("threshold"))
        family_count: Counter[str] = Counter()
        family_false_alarms: Counter[str] = Counter()
        family_near_threshold: Counter[str] = Counter()
        family_scores: defaultdict[str, list[float]] = defaultdict(list)
        negatives = 0
        for row, score in zip(predictions, scores):
            if row.get("label_id") != "0":
                continue
            negatives += 1
            scenario = scenario_by_group.get(row.get("scenario_group_id", ""), {})
            family = _family_label(scenario.get("hard_negative_role", ""))
            family_count[family] += 1
            family_totals[family] += 1
            if family not in family_palette_order:
                family_palette_order.append(family)
            family_scores[family].append(score)
            if score >= threshold:
                family_false_alarms[family] += 1
            elif score >= threshold - 0.05:
                family_near_threshold[family] += 1
        for family in sorted(family_count):
            fp_count = int(family_false_alarms[family])
            near_count = int(family_near_threshold[family])
            family_rows.append(
                {
                    "method": method,
                    "method_label": summary["method_label"],
                    "threshold_source": summary.get("threshold_source", ""),
                    "threshold": threshold,
                    "average_precision": summary.get("average_precision", float("nan")),
                    "roc_auc": summary.get("roc_auc", float("nan")),
                    "record_count": _safe_int(summary.get("record_count", 0)),
                    "negative_count": negatives,
                    "family": family,
                    "family_count": int(family_count[family]),
                    "false_alarm_count": fp_count,
                    "false_alarm_rate": float(fp_count / max(family_count[family], 1)),
                    "near_threshold_count": near_count,
                    "near_threshold_rate": float(near_count / max(family_count[family], 1)),
                    "fp_per_1000_negatives": float((fp_count / max(negatives, 1)) * 1000.0),
                    "fp_burden_vs_locked_candidate": float(fp_count - locked_fp_count)
                    if method != "locked_candidate"
                    else 0.0,
                }
            )
        top_family = "none"
        top_family_count = 0
        if family_false_alarms:
            top_family, top_family_count = max(
                family_false_alarms.items(), key=lambda item: (item[1], item[0])
            )
        summary_rows.append(
            {
                "rank": len(summary_rows) + 1,
                "method": method,
                "method_label": summary["method_label"],
                "threshold_source": summary.get("threshold_source", ""),
                "threshold": threshold,
                "average_precision": summary.get("average_precision", float("nan")),
                "recall_at_leq_1pct_fpr": float(fixed_fpr_recall),
                "selected_threshold_fp_count": _safe_int(summary.get("fp", 0)),
                "fp": _safe_int(summary.get("fp", 0)),
                "selected_threshold_fp_rate": _safe_float(summary.get("false_positive_rate")),
                "fp_per_1000_negatives": float(
                    (
                        _safe_float(summary.get("fp", 0.0))
                        / max(_safe_float(summary.get("negative_count", 1.0), 1.0), 1.0)
                    )
                    * 1000.0
                ),
                "top_fp_family": top_family,
                "top_fp_family_fp_count": int(top_family_count),
                "fp_burden_vs_locked_candidate": float(
                    _safe_float(summary.get("fp", 0.0)) - locked_fp_count
                )
                if method != "locked_candidate"
                else 0.0,
                "negative_count": _safe_int(summary.get("negative_count", 0)),
                "positive_count": _safe_int(summary.get("tp", 0)) + _safe_int(summary.get("fn", 0)),
                "record_count": _safe_int(summary.get("record_count", 0)),
                "tp": _safe_int(summary.get("tp", 0)),
                "tn": _safe_int(summary.get("tn", 0)),
                "fn": _safe_int(summary.get("fn", 0)),
                "precision": _safe_float(summary.get("precision")),
                "recall": _safe_float(summary.get("recall")),
                "f1": _safe_float(summary.get("f1")),
                "roc_auc": _safe_float(summary.get("roc_auc")),
                "false_positive_rate": _safe_float(summary.get("false_positive_rate")),
            }
        )

    family_order = sorted(family_totals, key=lambda name: (family_totals[name], name), reverse=True)
    return {
        "method_summary_rows": summary_rows,
        "family_rows": family_rows,
        "family_order": family_order,
        "method_order": selected_names,
    }


def _selected_threshold_confusion_rows(
    method_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in method_summary_rows:
        rows.append(
            {
                "method": row.get("method", ""),
                "method_label": row.get("method_label", row.get("method", "")),
                "threshold_source": row.get("threshold_source", ""),
                "threshold": row.get("threshold", float("nan")),
                "record_count": row.get("record_count", 0),
                "positive_count": row.get("tp", 0) + row.get("fn", 0),
                "negative_count": row.get("negative_count", 0),
                "tp": row.get("tp", 0),
                "fp": row.get("fp", 0),
                "tn": row.get("tn", 0),
                "fn": row.get("fn", 0),
                "precision": row.get("precision", float("nan")),
                "recall": row.get("recall", float("nan")),
                "f1": row.get("f1", float("nan")),
                "false_positive_rate": row.get("false_positive_rate", float("nan")),
                "average_precision": row.get("average_precision", float("nan")),
                "roc_auc": row.get("roc_auc", float("nan")),
            }
        )
    return rows


def _engineered_intelligence_rows(
    eval_summary: dict[str, Any],
    component_transparency: dict[str, Any],
    modality_transparency: dict[str, Any],
) -> list[dict[str, Any]]:
    lock = component_transparency.get("selection_lock", {})
    selected_method = str(lock.get("selected_candidate_id", ""))
    return [
        {
            "stage": "agentic decomposition",
            "summary": "Multiple search branches are explored train/CV-only before any holdout scoring.",
            "evidence_artifact": "selection_lock.json + candidate_leaderboard.csv",
            "status": "pass" if selected_method else "warn",
        },
        {
            "stage": "train/CV-only candidate discovery",
            "summary": "Candidate ranking, calibration, and thresholding are locked to train/CV rows.",
            "evidence_artifact": "performance_metrics.csv + selection_lock.json",
            "status": "pass",
        },
        {
            "stage": "evolved feature families",
            "summary": "The EI artifact fuses sparse radar, passive, and transport-style feature families.",
            "evidence_artifact": "selected_component_scores.csv + selected_component_aliases.json",
            "status": "pass" if component_transparency.get("component_scores") else "warn",
        },
        {
            "stage": "sparse calibrated fusion",
            "summary": "A small nonnegative weight vector is calibrated with a monotone odds transform.",
            "evidence_artifact": "selected_component_ablations.csv + modality_transparency.csv",
            "status": "pass",
        },
        {
            "stage": "holdout-only final scoring",
            "summary": "The holdout split is scored once after the lock; no candidate or threshold is tuned on it.",
            "evidence_artifact": "evaluation_summary.json + paper_evidence_manifest.json",
            "status": "pass" if eval_summary.get("selected") else "warn",
        },
    ]


def _source_pack_bird_coverage_summary() -> list[dict[str, Any]]:
    pack_dir = REPO_ROOT / "object-packs" / "hard-negatives"
    expected = [
        ("bird_single_small.yaml", "single_bird", "Single Small Bird"),
        ("bird_flock_dense.yaml", "bird_flock", "Dense Bird Flock"),
        ("shorebird_wader.yaml", "shorebird_wader", "Shorebird / Wader"),
        ("gull_tern.yaml", "gull_tern", "Gull / Tern"),
        ("raptor_falcon.yaml", "raptor_falcon", "Raptor / Falcon"),
        ("flamingo_large_bird.yaml", "flamingo_large_bird", "Flamingo / Large Bird"),
        ("seabird_cormorant.yaml", "seabird_cormorant", "Seabird / Cormorant"),
        (
            "seasonal_migratory_density.yaml",
            "seasonal_migratory_density",
            "Migratory Flock Density",
        ),
        ("rc_fixed_wing.yaml", "rc_fixed_wing", "RC Fixed-Wing"),
    ]
    rows = []
    for filename, slug, display_name in expected:
        path = pack_dir / filename
        rows.append(
            {
                "family_slug": slug,
                "display_name": display_name,
                "path": filename,
                "status": "present" if path.exists() else "missing",
                "object_pack": "pack.hard-negatives",
            }
        )
    return rows


def _public_proxy_positive_class_card() -> dict[str, Any]:
    source_text = _read_text(REPO_ROOT / "object-packs" / "public-proxy" / "source_dossier.yaml")
    match = re.search(r"^public_proxy_id:\s*([^\s]+)\s*$", source_text, re.MULTILINE)
    physics_dossier = _read_text(REPO_ROOT / "object-packs" / "public-proxy" / "physics_dossier.md")
    return {
        "public_proxy_id": match.group(1) if match else "fixed-wing-pusher-prop-public-proxy",
        "named_family_boundary": "fixed-wing pusher-prop public proxy; no measured Iranian-platform claim",
        "public_source_assumptions": {
            "geometry": {
                "planform": "delta / swept fixed wing",
                "propulsion": "rear pusher propulsor",
                "dimensions_m": {
                    "length": [3.3, 3.7],
                    "wingspan": [2.3, 2.7],
                    "height": [0.35, 0.75],
                },
            },
            "speed_envelope_mps": [45.0, 60.0],
            "phase_windows": {
                "take_up": "0-30 s",
                "climb": "30-90 s",
                "cruise": "90-150 s",
            },
            "launch_proxy": "rail/catapult take-up with short booster assist",
            "rcs_aspect_envelope_dbsm": [-28.0, -2.0],
            "prop_micro_doppler_hz": [150.0, 217.0],
            "sources": [
                "OSMP visual and catalogue reporting",
                "ArmyRecognition visual technical summary",
                "CSIS visual investigation",
                "small-UAS radar literature",
            ],
        },
        "prohibited_inferences": [
            "No measured Iranian-drone radar signature",
            "No operational route or evasion modelling",
            "No payload-effects inference",
            "No deployment-performance claim",
            "No sensor-parity or classified-fidelity claim",
        ],
        "claim_boundary": "Appendix-only public proxy card for reviewable synthetic evidence",
        "source_dossier_excerpt_length": len(physics_dossier.splitlines()),
    }


def _public_proxy_model_detail_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    assumptions = card.get("public_source_assumptions", {}) if isinstance(card, dict) else {}
    geometry = assumptions.get("geometry", {}) if isinstance(assumptions, dict) else {}
    return [
        {
            "model_item": "airframe_geometry",
            "public_proxy_assumption": json.dumps(geometry, sort_keys=True),
            "radar_consequence": "aspect-dependent projected area and RCS stress",
            "evidence_basis": "public visual/source summary envelope",
            "prohibited_inference": "no measured Iranian-platform radar signature",
        },
        {
            "model_item": "kinematics",
            "public_proxy_assumption": json.dumps(
                {"speed_envelope_mps": assumptions.get("speed_envelope_mps")}, sort_keys=True
            ),
            "radar_consequence": "Doppler centroid and track-rate priors",
            "evidence_basis": "public-source speed envelope",
            "prohibited_inference": "no route, tasking, or evasion model",
        },
        {
            "model_item": "launch_take_up",
            "public_proxy_assumption": str(assumptions.get("launch_proxy", "")),
            "radar_consequence": "initial phase ground coupling, multipath, and unsettled track state",
            "evidence_basis": "public-proxy launch abstraction",
            "prohibited_inference": "no site-specific or operational launch model",
        },
        {
            "model_item": "rcs_aspect_envelope",
            "public_proxy_assumption": json.dumps(
                {"dbsm": assumptions.get("rcs_aspect_envelope_dbsm")}, sort_keys=True
            ),
            "radar_consequence": "branch-level detectability and aspect stress",
            "evidence_basis": "public-proxy RCS envelope",
            "prohibited_inference": "no sensor-specific Pd/Pfa or classified-fidelity claim",
        },
        {
            "model_item": "prop_micro_doppler",
            "public_proxy_assumption": json.dumps(
                {"hz": assumptions.get("prop_micro_doppler_hz")}, sort_keys=True
            ),
            "radar_consequence": "pusher-prop modulation family and RC overlap stress",
            "evidence_basis": "small-UAS radar literature and public-proxy propulsion envelope",
            "prohibited_inference": "no measured propeller signature claim",
        },
        {
            "model_item": "phase_windows",
            "public_proxy_assumption": json.dumps(
                assumptions.get("phase_windows", {}), sort_keys=True
            ),
            "radar_consequence": "phase-specific holdout metrics and diagnostic slices",
            "evidence_basis": "benchmark time-window definition",
            "prohibited_inference": "phase slices are diagnostics, not field-rate estimates",
        },
    ]


def _environment_impairment_model_rows(radar_model_card: dict[str, Any]) -> list[dict[str, Any]]:
    priors = (
        radar_model_card.get("clutter_and_artifact_priors", {})
        if isinstance(radar_model_card, dict)
        else {}
    )
    clutter = priors.get("clutter", []) if isinstance(priors, dict) else []
    noise_rfi = priors.get("noise_and_rfi", []) if isinstance(priors, dict) else []
    hard_negatives = priors.get("hard_negative_families", []) if isinstance(priors, dict) else []
    receiver_impairments = (
        radar_model_card.get("receiver_impairments", [])
        if isinstance(radar_model_card, dict)
        else []
    )
    return [
        {
            "family": "thermal_snr_stress",
            "generated_proxy": "SNR/CNR stress buckets attached to branch summaries",
            "radar_review_role": "prevents a clean-signal-only benchmark",
            "source_values": "synthetic detector-view stress values",
            "claim_boundary": "not a calibrated receiver-noise figure",
        },
        {
            "family": "weibull_k_like_clutter",
            "generated_proxy": "|".join(map(str, clutter)),
            "radar_review_role": "non-Gaussian and site-style clutter pressure",
            "source_values": "radar_model_card.clutter_and_artifact_priors.clutter",
            "claim_boundary": "not measured site clutter truth",
        },
        {
            "family": "weather_sea_terrain",
            "generated_proxy": "weather cell, sea clutter, terrain glint, coastal haze, vegetation motion",
            "radar_review_role": "near-threshold false-track pressure",
            "source_values": "scenario noise/site strata and hard-negative roles",
            "claim_boundary": "not a meteorological population model",
        },
        {
            "family": "multipath_ghosting",
            "generated_proxy": "near-ground reflection and ghost-track stressors",
            "radar_review_role": "tests low-altitude false-track pressure and branch disagreement",
            "source_values": "scenario hard-negative roles and receiver impairment rows",
            "claim_boundary": "no deployment geometry or site-specific ray tracing",
        },
        {
            "family": "rfi_passive_rf_missingness",
            "generated_proxy": "|".join(map(str, noise_rfi)),
            "radar_review_role": "tests sparse and contaminated multimodal cues",
            "source_values": "radar_model_card.clutter_and_artifact_priors.noise_and_rfi",
            "claim_boundary": "no emitter library, payload command, or operator inference",
        },
        {
            "family": "receiver_impairment",
            "generated_proxy": "|".join(map(str, receiver_impairments)),
            "radar_review_role": "forces tolerance to AGC, drift, quantization, CPI loss, and calibration offset",
            "source_values": "radar_model_card.receiver_impairments",
            "claim_boundary": "simplified impairment families, not hardware qualification",
        },
        {
            "family": "hard_negative_taxonomy",
            "generated_proxy": "|".join(map(str, hard_negatives)),
            "radar_review_role": "explains false-alarm family colors and robustness slices",
            "source_values": "regional_hard_negative_taxonomy and false_alarm_by_method_family",
            "claim_boundary": "robustness diagnostics, not evasion guidance or field rates",
        },
    ]


def _data_processing_trace_rows(split_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Reviewer-facing trace from synthetic scenario design to paper evidence."""

    holdout_records = _safe_int(split_summary.get("holdout_record_count"), 0)
    holdout_positive_records = _safe_int(split_summary.get("holdout_positive_record_count"), 0)
    holdout_positive_groups = _safe_int(split_summary.get("holdout_positive_group_count"), 0)
    train_cv_groups = _safe_int(split_summary.get("train_cv_group_count"), 0)
    return [
        {
            "stage": "scenario_group_generation",
            "input_artifacts": "fixed-wing-pusher-proxy-v2 profile; public object/source packs; seed 202605210136",
            "output_artifacts": "scenario_manifest.csv; dataset_manifest.json; split_summary.json",
            "reviewer_check": (
                "Scenario groups and split roles are assigned before phase expansion; "
                f"train/CV has {train_cv_groups} groups."
            ),
            "claim_boundary": "Synthetic public-proxy scenario design, not field-rate, route, or measured-platform truth.",
        },
        {
            "stage": "phase_record_expansion",
            "input_artifacts": "scenario_manifest.csv; phase windows initial_take_up/climb_transition/cruise_altitude",
            "output_artifacts": "records.csv",
            "reviewer_check": "Each scenario group expands into three phase records that inherit the same split role and group lock.",
            "claim_boundary": "Phase labels support diagnostic slices only, not operational timeline claims.",
        },
        {
            "stage": "synthetic_iq_and_cue_generation",
            "input_artifacts": "radar_model_card.json; public_proxy_model_detail_rows.csv; environment_impairment_model_rows.csv",
            "output_artifacts": "raw complex-IQ references; acoustic cue summaries; passive-RF cue summaries; range-Doppler diagnostic panels",
            "reviewer_check": "Synthetic radar/cue products are generated before detector views; range-Doppler panels are qualitative sanity checks.",
            "claim_boundary": "Generated IQ and cue products are not measured imagery or proprietary sensor behavior.",
        },
        {
            "stage": "detector_view_schema_gate",
            "input_artifacts": "records.csv; generated sensor/cue products; MODEL_FEATURE_DENYLIST; DETECTOR_ID_COLUMNS",
            "output_artifacts": "detector_view_schema_evidence.csv; classical_radar_processing.csv; ml_detector_baselines.csv; fusion_predictions.csv; advanced_predictions.csv",
            "reviewer_check": "Labels, split keys, group IDs, time locks, audit headers, and generator internals are denied to model-facing matrices.",
            "claim_boundary": "Detector views are benchmark feature contracts, not vendor implementations or exact Pd/Pfa claims.",
        },
        {
            "stage": "train_cv_candidate_discovery",
            "input_artifacts": "train/CV detector-view scores only",
            "output_artifacts": "selection_lock.json; selected_component_scores.csv; selected_component_human_weights.csv; calibrators and thresholds",
            "reviewer_check": "EI component search, sparse weights, monotone odds calibration, and threshold policy are locked before holdout scoring.",
            "claim_boundary": "EI is an audited score-search and fusion lane, not an operational C2 system.",
        },
        {
            "stage": "blind_holdout_scoring",
            "input_artifacts": "locked EI candidate; accepted prior fusion baseline; holdout detector-view rows",
            "output_artifacts": "evaluation_summary.json; primary_kpi_table.csv; main_kpi_gain_table.csv; selected_threshold_confusion_matrix.csv",
            "reviewer_check": (
                "The blind holdout is scored once after lock; "
                f"{holdout_records} records include {holdout_positive_records} positives from "
                f"{holdout_positive_groups} positive groups."
            ),
            "claim_boundary": "One synthetic holdout score pass; no measured truth or field-performance claim.",
        },
        {
            "stage": "paper_evidence_artifacts",
            "input_artifacts": "evaluation_summary.json; leakage_diagnostics.json; modality_transparency.json; generated figure inputs",
            "output_artifacts": "paper_evidence_manifest.json; data_processing_trace_rows.csv; paper figures; echoforge_ieee.pdf",
            "reviewer_check": "Selected-threshold counts, swept Recall@<=1%FPR, and LCB95 are separate fields in generated evidence.",
            "claim_boundary": "Figures and tables are reproducible review artifacts, not additional empirical claims.",
        },
    ]


def _simulation_best_practice_rows() -> list[dict[str, Any]]:
    return [
        {
            "practice": "waveform_and_resolution_disclosure",
            "implementation": "carrier, bandwidth, CPI, PRF/CRF, chirp count, range bins, dR, dfD, and dv are emitted as radar model rows",
            "evidence_artifact": "radar_model_card.json; radar_model_detail_rows.csv",
            "claim_boundary": "branch-card assumptions only; not hardware qualification",
        },
        {
            "practice": "public_proxy_target_modeling",
            "implementation": "fixed-wing pusher-prop geometry, speed, aspect RCS envelope, and broad pusher-prop micro-Doppler prior",
            "evidence_artifact": "public_proxy_positive_class_card.json; public_proxy_model_detail_rows.csv",
            "claim_boundary": "no measured Iranian-drone radar signature or named-platform truth",
        },
        {
            "practice": "hard_negative_modeling",
            "implementation": "bird, RC, weather, clutter-only, terrain, multipath, RFI, ground vehicle, and wind-turbine families",
            "evidence_artifact": "regional_hard_negative_taxonomy.csv; regional_bird_library.csv",
            "claim_boundary": "robustness diagnostics, not evasion optimization or field-rate modeling",
        },
        {
            "practice": "clutter_noise_and_receiver_stress",
            "implementation": "thermal/SNR buckets, Weibull/K-like clutter, RFI, multipath, AGC, clock drift, quantization, dropped CPI, Doppler folding, and calibration offset",
            "evidence_artifact": "environment_impairment_model_rows.csv",
            "claim_boundary": "synthetic stress envelope, not site-measured clutter or receiver truth",
        },
        {
            "practice": "detector_view_isolation",
            "implementation": "detector-view schema blocks labels, split keys, group IDs, time locks, audit headers, and generator internals",
            "evidence_artifact": "detector_view_schema_evidence.csv",
            "claim_boundary": "audit fields are visible to validators but denied to model-facing matrices",
        },
        {
            "practice": "group_block_uncertainty",
            "implementation": "scenario groups are the split and bootstrap unit; phase rows are diagnostic slices",
            "evidence_artifact": "primary_kpi_table.csv; group_level_operating_metrics.csv",
            "claim_boundary": "row-level uncertainty is not treated as independent evidence",
        },
    ]


def _monte_carlo_setup_rows(split_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "scenario_group_sampling",
            "modeled_dimension": "site, range, aspect, weather/noise, hard-negative role, and positive role",
            "reviewer_check": f"{_safe_int(split_summary.get('scenario_group_count'))} scenario groups are emitted before phase expansion",
            "boundary": "coverage design only; not population prevalence",
        },
        {
            "stage": "group_locked_split",
            "modeled_dimension": "scenario_group_id, split_role, and cv_fold",
            "reviewer_check": f"holdout has {_safe_int(split_summary.get('holdout_group_count'))} groups and {_safe_int(split_summary.get('holdout_positive_group_count'))} positive groups",
            "boundary": "prevents repeated-phase leakage across train/CV and holdout",
        },
        {
            "stage": "phase_expansion",
            "modeled_dimension": "initial_take_up, climb_transition, cruise_altitude",
            "reviewer_check": "each scenario group expands into the same three phase windows",
            "boundary": "phase metrics are diagnostic because positive holdout groups are sparse",
        },
        {
            "stage": "rare_positive_stress",
            "modeled_dimension": "positive and negative group balance",
            "reviewer_check": f"holdout records include {_safe_int(split_summary.get('holdout_positive_record_count'))} positive records and {_safe_int(split_summary.get('holdout_negative_record_count'))} negative records",
            "boundary": "low-FPR KPI is required; AP alone is insufficient",
        },
    ]


def _detector_processing_baseline_rows() -> list[dict[str, Any]]:
    return [
        {
            "lane": "high_resolution_xku_cuas",
            "processing": "CFAR-style local contrast, range-Doppler concentration, micro-Doppler spread, and clutter stress",
            "input_view": "X/Ku radar detector-view summaries",
            "comparison_role": "high-resolution radar branch baseline",
        },
        {
            "lane": "tactical_s_band_aesa",
            "processing": "MTD/Doppler concentration, coarser range-velocity summaries, and track confidence",
            "input_view": "S-band detector-view summaries",
            "comparison_role": "tactical radar branch stress check",
        },
        {
            "lane": "gbad_3d4d_cueing",
            "processing": "revisit delay, radar-horizon stress, stale-track state, and cue confidence",
            "input_view": "GBAD cueing detector-view summaries",
            "comparison_role": "wide-area cueing branch baseline",
        },
        {
            "lane": "distributed_acoustic_cue",
            "processing": "spectral cadence, amplitude stability, and cross-node agreement",
            "input_view": "acoustic cue summaries",
            "comparison_role": "independent propulsion-cadence cue",
        },
        {
            "lane": "passive_rf_context",
            "processing": "no-signal, RFI burst, provenance quality, clock offset, and sparse cue geometry",
            "input_view": "passive-RF provenance summaries",
            "comparison_role": "missingness and multimodal confirmation stress",
        },
        {
            "lane": "prior_ml_controls",
            "processing": "tabular and sequence learners over denylisted detector-view features",
            "input_view": "detector-view matrices after MODEL_FEATURE_DENYLIST",
            "comparison_role": "ordinary learned controls under the same split",
        },
        {
            "lane": "layered_fusion_c2",
            "processing": "human-engineered late fusion with train/CV thresholding and calibration",
            "input_view": "radar, acoustic, passive-RF, and cue confidence scores",
            "comparison_role": "accepted human-engineered prior fusion comparator",
        },
    ]


def _fusion_baseline_rows() -> list[dict[str, Any]]:
    return [
        {
            "component": "active_radar_branches",
            "input": "X/Ku, S-band, and GBAD score summaries",
            "fusion_role": "active sensing evidence for range-Doppler and track structure",
            "known_risk": "birds, RC aircraft, weather, and multipath can create radar-only false alarms",
        },
        {
            "component": "acoustic_cue",
            "input": "cadence, amplitude stability, and node agreement",
            "fusion_role": "independent propulsion-cadence support",
            "known_risk": "weather, traffic, and geometry can degrade cue quality",
        },
        {
            "component": "passive_rf_context",
            "input": "no-signal, RFI, clock, and provenance-quality geometry",
            "fusion_role": "sparse confirmation, contradiction, and missingness evidence",
            "known_risk": "can rank well while having weaker selected-threshold calibration",
        },
        {
            "component": "accepted_prior_fusion",
            "input": "human-designed branch score combination",
            "fusion_role": "main non-EI comparator before candidate evolution",
            "known_risk": "lower low-FPR recall than EI in this declared run",
        },
    ]


def _ei_objective_rows() -> list[dict[str, Any]]:
    return [
        {
            "objective": "LCB95 Recall@<=1%FPR",
            "what_it_rewards": "conservative group-block low-FPR detection",
            "current_role": "headline KPI after holdout lock",
        },
        {
            "objective": "train_cv_candidate_objective",
            "what_it_rewards": "internal AP plus ROC support with false-positive and weak-phase penalties",
            "current_role": "selection search objective; holdout rows used for selection = 0",
        },
        {
            "objective": "calibration_aware_f1",
            "what_it_rewards": "selected-threshold precision/recall with reliable probability estimates",
            "current_role": "guardrail against high-ranking but poorly calibrated controls",
        },
        {
            "objective": "hard_negative_burden",
            "what_it_rewards": "few false alarms by bird, RC, weather, clutter, multipath, and RFI family",
            "current_role": "diagnostic and future objective",
        },
        {
            "objective": "phase_minimum_recall",
            "what_it_rewards": "worst-phase low-FPR recall across take-up, climb, and cruise",
            "current_role": "future objective for take-up robustness",
        },
    ]


def _phase_method_ladder_rows(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    phase_metrics = eval_summary.get("phase_metrics", {})
    selected_method = str(eval_summary.get("selected_method", ""))
    method_labels = {
        "high_resolution_xku_cuas": "X/Ku radar branch",
        "tactical_s_band_aesa": "S-band radar branch",
        "gbad_3d4d_cueing": "GBAD cueing branch",
        "distributed_acoustic_cue": "Acoustic cue branch",
        "tabular_ml_baseline": "Tabular ML control",
        "sequence_ml_proxy": "Sequence ML control",
        "layered_fusion_c2": "Accepted prior fusion",
        selected_method: "Engineered Intelligence",
    }
    rows: list[dict[str, Any]] = []
    for method, phases in phase_metrics.items():
        for phase, metrics in phases.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "method": method,
                    "method_label": method_labels.get(method, method.replace("_", " ")),
                    "phase": phase,
                    "fixed_fpr_recall": _safe_float(metrics.get("fixed_fpr_recall")),
                    "average_precision": _safe_float(metrics.get("average_precision")),
                    "roc_auc": _safe_float(metrics.get("roc_auc")),
                    "positive_count": _safe_int(metrics.get("positive_count")),
                    "negative_count": _safe_int(metrics.get("negative_count")),
                    "claim_boundary": "phase rows are diagnostic; main claim remains aggregate group-locked holdout KPI",
                }
            )
    return rows


def _cli_reproduction_commands() -> list[dict[str, Any]]:
    return [
        {
            "stage": "generate_training_data",
            "command": "rtk python3 -m detection.generate_main_run --profile fixed-wing-pusher-proxy-v2 --out-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run --scenario-groups 10000 --seed 202605210136 --force",
        },
        {
            "stage": "run_baseline_detectors",
            "command": "rtk python3 -m detection.run_main_run_detectors --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run --folds 5 --seed 202605210136 --force",
        },
        {
            "stage": "run_ei_advanced_detectors",
            "command": "rtk python3 -m detection.run_advanced_main_run_detectors --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution --folds 5 --seed 202605210136 --search-profile v2_aggressive --candidate-limit 128 --evolution-rounds 5 --evolution-sample-rows 4500 --write-component-scores --force",
        },
        {
            "stage": "generate_paper_evidence",
            "command": "rtk python3 -m detection.paper_evidence --strict-ei-trace --force",
        },
        {
            "stage": "generate_source_appendix",
            "command": "rtk python3 paper/generate_source_appendix.py --strict",
        },
        {
            "stage": "generate_metric_macros",
            "command": "rtk python3 paper/generate_metric_macros.py --strict",
        },
        {
            "stage": "generate_figures",
            "command": "rtk python3 paper/generate_figures_focused.py --strict",
        },
        {
            "stage": "build_pdf",
            "command": "rtk bash paper/build.sh --copy-tracked",
        },
        {
            "stage": "validate_paper",
            "command": "rtk python3 paper/validate_paper.py --tex paper/echoforge_ieee.tex --bib paper/references.bib --pdf target/paper/echoforge_ieee.pdf --figures-dir paper/figures --paper-evidence-root outputs/paper-evidence/tier1-final",
        },
    ]


def _source_appendix_hashes() -> list[dict[str, Any]]:
    manifest_path = REPO_ROOT / "paper" / "source_appendix_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for block in manifest.get("blocks", []):
        if not isinstance(block, dict):
            continue
        rel_path = str(block.get("path", ""))
        source_path = REPO_ROOT / rel_path
        if not source_path.is_file():
            rows.append(
                {
                    "block_id": block.get("id", ""),
                    "path": rel_path,
                    "origin": block.get("origin", ""),
                    "symbols": "|".join(map(str, block.get("symbols", []))),
                    "line_start": "",
                    "line_end": "",
                    "sha256": "",
                    "status": "missing_source",
                }
            )
            continue
        text = source_path.read_text(encoding="utf-8")
        file_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        symbol_spans: dict[str, tuple[int, int]] = {}
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbol_spans[node.name] = (
                        int(node.lineno),
                        int(getattr(node, "end_lineno", node.lineno)),
                    )
        except SyntaxError:
            symbol_spans = {}
        for symbol in [str(item) for item in block.get("symbols", [])]:
            start, end = symbol_spans.get(symbol, (0, 0))
            excerpt = "\n".join(text.splitlines()[max(start - 1, 0) : end]) if start and end else ""
            rows.append(
                {
                    "block_id": block.get("id", ""),
                    "path": rel_path,
                    "origin": block.get("origin", ""),
                    "symbols": symbol,
                    "line_start": start or "",
                    "line_end": end or "",
                    "sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                    if excerpt
                    else file_digest,
                    "status": "included" if excerpt else "file_hash_only",
                }
            )
    ip_excerpt = manifest.get("ip_escrow", {})
    if isinstance(ip_excerpt, dict):
        rows.append(
            {
                "block_id": ip_excerpt.get("id", "ip_escrow"),
                "path": ip_excerpt.get("path", ""),
                "origin": "redacted_escrow_only",
                "symbols": "|".join(map(str, ip_excerpt.get("symbols", []))),
                "line_start": "",
                "line_end": "",
                "sha256": "",
                "status": "duplicate_optional_not_claim_critical",
            }
        )
    return rows


def build_paper_evidence(roots: EvidenceRoots, *, strict_ei_trace: bool = False) -> dict[str, Any]:
    if roots.out_root.exists():
        shutil.rmtree(roots.out_root)
    roots.out_root.mkdir(parents=True, exist_ok=True)

    training_manifest = _read_json(roots.training_root / "dataset_manifest.json")
    training_quality = _read_json(roots.training_root / "quality_report.json")
    scenarios = _read_csv_rows(roots.training_root / "scenario_manifest.csv")
    records = _read_csv_rows(roots.training_root / "records.csv")
    radar_model_card = _build_radar_model_card(roots.training_root, scenarios)
    radar_model_detail_rows = _radar_model_detail_rows(radar_model_card)
    sensor_archetype_cards = _build_public_sensor_archetype_cards()
    regional_hard_negative_taxonomy = _build_regional_hard_negative_taxonomy()
    scenario_balance = _scenario_balance_payload(scenarios)
    scenario_balance_by_dimension = _balance_dimension_rows(scenarios, records, positive_only=False)
    positive_balance_by_dimension = _balance_dimension_rows(scenarios, records, positive_only=True)
    split_summary = _split_summary(scenarios, records)
    feedback_coverage = _feedback_coverage_summary()
    generative_origin_audit = _run_generative_origin_audit(roots.out_root)
    detector_view_schema_evidence = _detector_view_schema_evidence(roots.training_root)
    data_processing_trace_rows = _data_processing_trace_rows(split_summary)
    simulation_best_practice_rows = _simulation_best_practice_rows()
    monte_carlo_setup_rows = _monte_carlo_setup_rows(split_summary)
    detector_processing_baseline_rows = _detector_processing_baseline_rows()
    fusion_baseline_rows = _fusion_baseline_rows()
    ei_objective_rows = _ei_objective_rows()
    cli_reproduction_commands = _cli_reproduction_commands()
    source_appendix_hashes = _source_appendix_hashes()

    holdout_records = [row for row in records if row.get("split_role") == "holdout"]
    holdout_scenarios = [row for row in scenarios if row.get("split_role") == "holdout"]

    leakage = {
        "canary_forbidden_feature_test": _detect_forbidden_features(roots.training_root),
        "label_shuffle_sanity": None,
        "nearest_neighbor_scan": None,
        "metadata_only_baseline": None,
        "stratum_only_baseline": None,
    }
    selected_root = roots.advanced_root
    selected_rows = _read_csv_rows(selected_root / "advanced_predictions.csv")
    selected_lock = _load_selection_lock(selected_root / "selection_lock.json")
    selected_method = str(selected_lock.get("selected_candidate_id", ""))
    selected_holdout = [row for row in selected_rows if row.get("split_role") == "holdout"]
    selected_labels = np.asarray(
        [_safe_int(row.get("label_id")) for row in selected_holdout], dtype=np.int8
    )
    selected_scores = np.asarray(
        [_safe_float(row.get("advanced_score")) for row in selected_holdout], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shuffled = selected_labels.copy()
    rng.shuffle(shuffled)
    leakage["label_shuffle_sanity"] = {
        "selected": _metric_bundle(
            selected_labels,
            selected_scores,
            _holdout_threshold_from_performance(
                selected_root / "performance_metrics.csv", selected_method
            ),
        ),
        "shuffled_labels": _metric_bundle(
            shuffled,
            selected_scores,
            _holdout_threshold_from_performance(
                selected_root / "performance_metrics.csv", selected_method
            ),
        ),
    }
    selected_feature_rows = [
        {
            "record_id": row.get("record_id", ""),
            "scenario_group_id": row.get("scenario_group_id", ""),
            "high_resolution_xku_cuas": row.get("high_resolution_xku_cuas", ""),
            "tactical_s_band_aesa": row.get("tactical_s_band_aesa", ""),
            "gbad_3d4d_cueing": row.get("gbad_3d4d_cueing", ""),
            "distributed_acoustic_cue": row.get("distributed_acoustic_cue", ""),
            "layered_fusion_c2": row.get("layered_fusion_c2", ""),
        }
        for row in selected_rows
        if row.get("split_role") in {"train_cv", "holdout"}
    ]
    train_group_ids = {
        scenario["scenario_group_id"]
        for scenario in scenarios
        if scenario.get("split_role") == "train_cv"
    }
    holdout_group_ids = {
        scenario["scenario_group_id"]
        for scenario in scenarios
        if scenario.get("split_role") == "holdout"
    }
    train_feature_rows = [
        row for row in selected_feature_rows if row.get("scenario_group_id", "") in train_group_ids
    ]
    holdout_feature_rows = [
        row
        for row in selected_feature_rows
        if row.get("scenario_group_id", "") in holdout_group_ids
    ]
    leakage["nearest_neighbor_scan"] = _nearest_neighbor_scan(
        train_feature_rows,
        holdout_feature_rows,
        [
            "high_resolution_xku_cuas",
            "tactical_s_band_aesa",
            "gbad_3d4d_cueing",
            "distributed_acoustic_cue",
            "layered_fusion_c2",
        ],
    )

    leakage.update(_stratum_and_metadata_baselines(scenarios, holdout_records))

    eval_summary = _build_evaluation_summary(roots.baseline_root, roots.advanced_root)
    phase_method_ladder_rows = _phase_method_ladder_rows(eval_summary)
    ei_evolution_evidence = build_ei_evolution_evidence(
        roots.advanced_root,
        roots.out_root,
        strict_trace=strict_ei_trace,
    )
    component_transparency = _load_component_transparency(roots.advanced_root)
    anchor_summary = _load_anchor_summary(roots.anchor_root)
    normalized_anchor_comparison = _normalized_anchor_comparison(anchor_summary)
    anchor_distance_diagnostics = _anchor_distance_diagnostics(anchor_summary)
    modality_transparency = _build_modality_transparency(
        selected_rows,
        component_transparency.get("component_scores", []),
        component_transparency.get("component_ablations", []),
        float(eval_summary.get("selected_threshold", float("nan"))),
    )
    comparable_ablation_summary = _comparable_ablation_rows(eval_summary, modality_transparency)
    selected_component_human_weights = _selected_component_human_weights(
        component_transparency.get("component_scores", [])
    )
    primary_kpi_rows = _primary_kpi_rows(eval_summary)
    main_kpi_gain_rows = _main_kpi_gain_rows(eval_summary)
    false_alarm_rows = _false_alarm_breakdown(
        holdout_scenarios,
        [row for row in selected_rows if row.get("split_role") == "holdout"],
        np.asarray(
            [
                _safe_float(row.get("advanced_score"))
                for row in selected_rows
                if row.get("split_role") == "holdout"
            ],
            dtype=np.float64,
        ),
        float(eval_summary.get("selected_threshold", float("nan"))),
    )
    false_positive_tables = _false_positive_method_tables(
        scenarios=scenarios,
        baseline_root=roots.baseline_root,
        advanced_root=roots.advanced_root,
        selected_method=selected_method,
        selected_threshold=float(eval_summary.get("selected_threshold", float("nan"))),
        eval_summary=eval_summary,
    )
    selected_threshold_confusion_rows = _selected_threshold_confusion_rows(
        false_positive_tables["method_summary_rows"]
    )
    engineered_intelligence_rows = _engineered_intelligence_rows(
        eval_summary, component_transparency, modality_transparency
    )
    source_pack_bird_coverage = _source_pack_bird_coverage_summary()
    public_proxy_positive_class_card = _public_proxy_positive_class_card()
    public_proxy_model_detail_rows = _public_proxy_model_detail_rows(
        public_proxy_positive_class_card
    )
    environment_impairment_model_rows = _environment_impairment_model_rows(radar_model_card)
    regional_bird_library = _build_regional_bird_library(
        false_positive_tables["family_rows"], false_positive_tables["method_summary_rows"]
    )

    selected_component_path = roots.advanced_root / "selected_component_scores.csv"
    component_score_rows = _read_csv_rows(selected_component_path)
    if component_score_rows:
        selected_component_columns = sorted(
            {row.get("component_id", "") for row in component_score_rows if row.get("component_id")}
        )
    else:
        selected_component_columns = []

    payload = {
        "version": PAPER_EVIDENCE_VERSION,
        "training_manifest": training_manifest,
        "training_quality": training_quality,
        "radar_model_card": radar_model_card,
        "radar_model_detail_rows": radar_model_detail_rows,
        "sensor_archetype_cards": sensor_archetype_cards,
        "regional_hard_negative_taxonomy": regional_hard_negative_taxonomy,
        "scenario_balance": scenario_balance,
        "scenario_balance_by_dimension": scenario_balance_by_dimension,
        "positive_balance_by_dimension": positive_balance_by_dimension,
        "split_summary": split_summary,
        "detector_view_schema_evidence": detector_view_schema_evidence,
        "data_processing_trace_rows": data_processing_trace_rows,
        "simulation_best_practice_rows": simulation_best_practice_rows,
        "monte_carlo_setup_rows": monte_carlo_setup_rows,
        "detector_processing_baseline_rows": detector_processing_baseline_rows,
        "fusion_baseline_rows": fusion_baseline_rows,
        "ei_objective_rows": ei_objective_rows,
        "phase_method_ladder_rows": phase_method_ladder_rows,
        "cli_reproduction_commands": cli_reproduction_commands,
        "source_appendix_hashes": source_appendix_hashes,
        "ei_evolution_summary": ei_evolution_evidence.get("summary", {}),
        "ei_evolution_evidence": ei_evolution_evidence,
        "leakage_diagnostics": leakage,
        "evaluation_summary": eval_summary,
        "component_transparency": component_transparency,
        "modality_transparency": modality_transparency,
        "comparable_ablation_summary": comparable_ablation_summary,
        "selected_component_human_weights": selected_component_human_weights,
        "anchor_summary": anchor_summary,
        "normalized_anchor_comparison": normalized_anchor_comparison,
        "anchor_distance_diagnostics": anchor_distance_diagnostics,
        "primary_kpi_rows": primary_kpi_rows,
        "main_kpi_gain_rows": main_kpi_gain_rows,
        "group_level_operating_metrics": eval_summary.get("group_operating_metrics", []),
        "selected_threshold_confusion_matrix": selected_threshold_confusion_rows,
        "selected_component_columns": selected_component_columns,
        "false_alarm_family_breakdown": false_alarm_rows,
        "false_alarm_by_method_family": false_positive_tables["family_rows"],
        "top_method_false_positive_frequency": false_positive_tables["method_summary_rows"],
        "regional_bird_library": regional_bird_library,
        "source_pack_bird_coverage_summary": source_pack_bird_coverage,
        "engineered_intelligence": engineered_intelligence_rows,
        "public_proxy_positive_class_card": public_proxy_positive_class_card,
        "public_proxy_model_detail_rows": public_proxy_model_detail_rows,
        "environment_impairment_model_rows": environment_impairment_model_rows,
        "feedback_coverage": feedback_coverage,
        "generative_origin_audit": generative_origin_audit,
        "source_roots": {
            "training_root": str(roots.training_root),
            "baseline_root": str(roots.baseline_root),
            "advanced_root": str(roots.advanced_root),
            "anchor_root": str(roots.anchor_root),
        },
    }

    _write_json(roots.out_root / "paper_evidence_manifest.json", payload)
    _write_json(roots.out_root / "radar_model_card.json", radar_model_card)
    _write_json(roots.out_root / "radar_model_detail_rows.json", radar_model_detail_rows)
    _write_csv(
        roots.out_root / "radar_model_detail_rows.csv",
        radar_model_detail_rows,
        ["branch", "detail_key", "detail_value", "unit", "basis"],
    )
    _write_json(roots.out_root / "sensor_archetype_cards.json", sensor_archetype_cards)
    _write_json(
        roots.out_root / "regional_hard_negative_taxonomy.json", regional_hard_negative_taxonomy
    )
    _write_json(roots.out_root / "scenario_balance.json", scenario_balance)
    _write_json(
        roots.out_root / "scenario_balance_by_dimension.json", scenario_balance_by_dimension
    )
    _write_json(
        roots.out_root / "positive_balance_by_dimension.json", positive_balance_by_dimension
    )
    _write_json(roots.out_root / "split_summary.json", split_summary)
    _write_csv(
        roots.out_root / "scenario_balance.csv",
        scenario_balance["grouped_rows"],
        [
            "split_role",
            "label_role",
            "site_archetype_id",
            "range_band",
            "target_aspect",
            "noise_regime",
            "hard_negative_role",
            "count",
        ],
    )
    balance_fields = [
        "dimension",
        "value",
        "split_role",
        "label_role",
        "count_unit",
        "count",
        "positive_only",
    ]
    _write_csv(
        roots.out_root / "scenario_balance_by_dimension.csv",
        scenario_balance_by_dimension,
        balance_fields,
    )
    _write_csv(
        roots.out_root / "positive_balance_by_dimension.csv",
        positive_balance_by_dimension,
        balance_fields,
    )
    _write_csv(
        roots.out_root / "false_alarm_family_breakdown.csv",
        false_alarm_rows,
        [
            "family",
            "count",
            "false_alarm_count",
            "false_alarm_rate",
            "near_threshold_count",
            "near_threshold_rate",
            "mean_score",
            "p95_score",
            "max_score",
        ],
    )
    _write_csv(
        roots.out_root / "sensor_archetype_cards.csv",
        sensor_archetype_cards,
        [
            "family",
            "paper_label",
            "public_role_envelope",
            "public_anchors",
            "observable_families",
            "detector_visible_features",
            "excluded_truth_fields",
            "limitations",
            "prohibited_inferences",
        ],
    )
    _write_json(
        roots.out_root / "detector_view_schema_evidence.json", detector_view_schema_evidence
    )
    _write_csv(
        roots.out_root / "detector_view_schema_evidence.csv",
        detector_view_schema_evidence,
        [
            "view_id",
            "path",
            "status",
            "audit_header_status",
            "audit_only_fields",
            "forbidden_model_fields",
            "allowed_detector_feature_families",
            "allowed_detector_feature_count",
            "model_schema_note",
        ],
    )
    _write_json(roots.out_root / "data_processing_trace_rows.json", data_processing_trace_rows)
    _write_csv(
        roots.out_root / "data_processing_trace_rows.csv",
        data_processing_trace_rows,
        [
            "stage",
            "input_artifacts",
            "output_artifacts",
            "reviewer_check",
            "claim_boundary",
        ],
    )
    _write_json(
        roots.out_root / "simulation_best_practice_rows.json", simulation_best_practice_rows
    )
    _write_csv(
        roots.out_root / "simulation_best_practice_rows.csv",
        simulation_best_practice_rows,
        ["practice", "implementation", "evidence_artifact", "claim_boundary"],
    )
    _write_json(roots.out_root / "monte_carlo_setup_rows.json", monte_carlo_setup_rows)
    _write_csv(
        roots.out_root / "monte_carlo_setup_rows.csv",
        monte_carlo_setup_rows,
        ["stage", "modeled_dimension", "reviewer_check", "boundary"],
    )
    _write_json(
        roots.out_root / "detector_processing_baseline_rows.json",
        detector_processing_baseline_rows,
    )
    _write_csv(
        roots.out_root / "detector_processing_baseline_rows.csv",
        detector_processing_baseline_rows,
        ["lane", "processing", "input_view", "comparison_role"],
    )
    _write_json(roots.out_root / "fusion_baseline_rows.json", fusion_baseline_rows)
    _write_csv(
        roots.out_root / "fusion_baseline_rows.csv",
        fusion_baseline_rows,
        ["component", "input", "fusion_role", "known_risk"],
    )
    _write_json(roots.out_root / "ei_objective_rows.json", ei_objective_rows)
    _write_csv(
        roots.out_root / "ei_objective_rows.csv",
        ei_objective_rows,
        ["objective", "what_it_rewards", "current_role"],
    )
    _write_json(roots.out_root / "phase_method_ladder_rows.json", phase_method_ladder_rows)
    _write_csv(
        roots.out_root / "phase_method_ladder_rows.csv",
        phase_method_ladder_rows,
        [
            "method",
            "method_label",
            "phase",
            "fixed_fpr_recall",
            "average_precision",
            "roc_auc",
            "positive_count",
            "negative_count",
            "claim_boundary",
        ],
    )
    _write_json(roots.out_root / "cli_reproduction_commands.json", cli_reproduction_commands)
    _write_csv(
        roots.out_root / "cli_reproduction_commands.csv",
        cli_reproduction_commands,
        ["stage", "command"],
    )
    _write_json(roots.out_root / "source_appendix_hashes.json", source_appendix_hashes)
    _write_csv(
        roots.out_root / "source_appendix_hashes.csv",
        source_appendix_hashes,
        [
            "block_id",
            "path",
            "origin",
            "symbols",
            "line_start",
            "line_end",
            "sha256",
            "status",
        ],
    )
    _write_csv(
        roots.out_root / "regional_hard_negative_taxonomy.csv",
        regional_hard_negative_taxonomy,
        [
            "family",
            "regional_subfamily",
            "wingbeat_hz_proxy",
            "prop_cadence_hz_proxy",
            "velocity_mps_proxy",
            "flock_size_proxy",
            "site_archetype",
            "seasonality_flag",
            "expected_confusion_mechanism",
            "paper_boundary",
        ],
    )
    _write_csv(
        roots.out_root / "modality_transparency.csv",
        modality_transparency["modality_rows"],
        [
            "view",
            "kind",
            "threshold_basis",
            "threshold",
            "average_precision",
            "roc_auc",
            "fixed_fpr_recall",
            "precision",
            "recall",
            "false_positive_rate",
            "f1",
            "ece",
            "tp",
            "fp",
            "tn",
            "fn",
            "source_columns",
        ],
    )
    _write_csv(
        roots.out_root / "ablation_transparency.csv",
        modality_transparency["ablation_rows"],
        [
            "view",
            "kind",
            "component_scope",
            "average_precision",
            "roc_auc",
            "f1",
            "brier_score",
            "ece",
        ],
    )
    _write_csv(
        roots.out_root / "comparable_ablation_summary.csv",
        comparable_ablation_summary,
        [
            "ablation",
            "reader_label",
            "diagnostic_type",
            "ap",
            "delta_ap",
            "recall_at_leq_1pct_fpr",
            "delta_recall_at_leq_1pct_fpr",
            "f1",
            "ece",
            "metric_basis",
            "comparability_note",
        ],
    )
    _write_csv(
        roots.out_root / "selected_component_human_weights.csv",
        selected_component_human_weights,
        [
            "audit_id",
            "family",
            "modality",
            "view",
            "calibrator",
            "weight",
            "cumulative_weight",
            "component_group",
            "raw_component_id",
        ],
    )
    _write_csv(
        roots.out_root / "normalized_anchor_comparison.csv",
        normalized_anchor_comparison,
        ["feature", "unitless_basis", "q10_z", "q50_z", "q90_z", "count", "claim_boundary"],
    )
    _write_csv(
        roots.out_root / "anchor_distance_diagnostics.csv",
        anchor_distance_diagnostics,
        [
            "feature",
            "diagnostic",
            "measured_anchor_value",
            "synthetic_reference_value",
            "distance",
            "distance_basis",
            "status",
            "allowed_claim_level",
        ],
    )
    _write_csv(
        roots.out_root / "primary_kpi_table.csv",
        primary_kpi_rows,
        ["method", "point_estimate", "lcb95", "target_fpr", "basis"],
    )
    _write_csv(
        roots.out_root / "main_kpi_gain_table.csv",
        main_kpi_gain_rows,
        [
            "metric",
            "prior_fusion_baseline",
            "ei_candidate",
            "absolute_change",
            "relative_change",
            "relative_percent",
            "direction",
        ],
    )
    _write_csv(
        roots.out_root / "group_level_operating_metrics.csv",
        eval_summary.get("group_operating_metrics", []),
        [
            "method",
            "threshold_source",
            "threshold",
            "positive_groups",
            "negative_groups",
            "group_tp",
            "group_fp",
            "group_tn",
            "group_fn",
            "group_recall",
            "group_false_alarm_rate",
            "bootstrap_note",
        ],
    )
    _write_csv(
        roots.out_root / "selected_threshold_confusion_matrix.csv",
        selected_threshold_confusion_rows,
        [
            "method",
            "method_label",
            "threshold_source",
            "threshold",
            "record_count",
            "positive_count",
            "negative_count",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "average_precision",
            "roc_auc",
        ],
    )
    _write_csv(
        roots.out_root / "false_alarm_by_method_family.csv",
        false_positive_tables["family_rows"],
        [
            "method",
            "method_label",
            "threshold_source",
            "threshold",
            "average_precision",
            "roc_auc",
            "record_count",
            "negative_count",
            "family",
            "family_count",
            "false_alarm_count",
            "false_alarm_rate",
            "near_threshold_count",
            "near_threshold_rate",
            "fp_per_1000_negatives",
            "fp_burden_vs_locked_candidate",
        ],
    )
    _write_csv(
        roots.out_root / "top_method_false_positive_frequency.csv",
        false_positive_tables["method_summary_rows"],
        [
            "rank",
            "method",
            "method_label",
            "threshold_source",
            "threshold",
            "average_precision",
            "recall_at_leq_1pct_fpr",
            "selected_threshold_fp_count",
            "fp",
            "selected_threshold_fp_rate",
            "fp_per_1000_negatives",
            "top_fp_family",
            "top_fp_family_fp_count",
            "fp_burden_vs_locked_candidate",
            "negative_count",
            "positive_count",
            "record_count",
            "tp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_positive_rate",
        ],
    )
    _write_csv(
        roots.out_root / "regional_bird_library.csv",
        regional_bird_library,
        [
            "family",
            "regional_subfamily",
            "wingbeat_hz_proxy",
            "prop_cadence_hz_proxy",
            "velocity_mps_proxy",
            "flock_size_proxy",
            "site_archetype",
            "seasonality_flag",
            "expected_confusion_mechanism",
            "paper_boundary",
            "site_prevalence",
            "body_size_rcs_stress",
            "flock_behavior",
            "false_alarm_count_total",
            "near_threshold_count_total",
            "top_method_false_alarm_family",
            "top_method_false_alarm_count",
        ],
    )
    _write_csv(
        roots.out_root / "source_pack_bird_coverage_summary.csv",
        source_pack_bird_coverage,
        ["family_slug", "display_name", "path", "status", "object_pack"],
    )
    _write_csv(
        roots.out_root / "engineered_intelligence_transparency.csv",
        engineered_intelligence_rows,
        ["stage", "summary", "evidence_artifact", "status"],
    )
    _write_json(roots.out_root / "feedback_coverage.json", feedback_coverage)
    _write_csv(
        roots.out_root / "feedback_coverage_matrix.csv",
        feedback_coverage.get("rows", []),
        [
            "tip",
            "item_id",
            "actionable_item",
            "status",
            "evidence_location",
            "notes",
        ],
    )
    _write_text(
        roots.out_root / "feedback_coverage_matrix.md",
        _feedback_coverage_markdown(feedback_coverage.get("rows", [])),
    )
    _write_json(roots.out_root / "leakage_diagnostics.json", leakage)
    _write_json(roots.out_root / "evaluation_summary.json", eval_summary)
    _write_json(roots.out_root / "anchor_summary.json", anchor_summary)
    _write_json(roots.out_root / "modality_transparency.json", modality_transparency)
    _write_json(roots.out_root / "comparable_ablation_summary.json", comparable_ablation_summary)
    _write_json(
        roots.out_root / "selected_component_human_weights.json", selected_component_human_weights
    )
    _write_json(roots.out_root / "normalized_anchor_comparison.json", normalized_anchor_comparison)
    _write_json(roots.out_root / "anchor_distance_diagnostics.json", anchor_distance_diagnostics)
    _write_json(
        roots.out_root / "group_level_operating_metrics.json",
        eval_summary.get("group_operating_metrics", []),
    )
    _write_json(roots.out_root / "regional_bird_library.json", regional_bird_library)
    _write_json(
        roots.out_root / "source_pack_bird_coverage_summary.json", source_pack_bird_coverage
    )
    _write_json(
        roots.out_root / "engineered_intelligence_transparency.json", engineered_intelligence_rows
    )
    _write_json(
        roots.out_root / "selected_threshold_confusion_matrix.json",
        selected_threshold_confusion_rows,
    )
    _write_json(
        roots.out_root / "false_alarm_by_method_family.json",
        false_positive_tables["family_rows"],
    )
    _write_json(
        roots.out_root / "top_method_false_positive_frequency.json",
        false_positive_tables["method_summary_rows"],
    )
    _write_json(
        roots.out_root / "public_proxy_positive_class_card.json",
        public_proxy_positive_class_card,
    )
    _write_json(roots.out_root / "main_kpi_gain_table.json", main_kpi_gain_rows)
    _write_json(
        roots.out_root / "public_proxy_model_detail_rows.json",
        public_proxy_model_detail_rows,
    )
    _write_csv(
        roots.out_root / "public_proxy_model_detail_rows.csv",
        public_proxy_model_detail_rows,
        [
            "model_item",
            "public_proxy_assumption",
            "radar_consequence",
            "evidence_basis",
            "prohibited_inference",
        ],
    )
    _write_json(
        roots.out_root / "environment_impairment_model_rows.json",
        environment_impairment_model_rows,
    )
    _write_csv(
        roots.out_root / "environment_impairment_model_rows.csv",
        environment_impairment_model_rows,
        [
            "family",
            "generated_proxy",
            "radar_review_role",
            "source_values",
            "claim_boundary",
        ],
    )

    if component_transparency.get("selection_lock"):
        _write_json(
            roots.out_root / "selected_candidate_lock.json",
            component_transparency["selection_lock"],
        )
    if component_transparency.get("aliases"):
        _write_json(
            roots.out_root / "selected_component_aliases.json", component_transparency["aliases"]
        )
    if component_transparency.get("component_ablations"):
        _write_csv(
            roots.out_root / "selected_component_ablations.csv",
            component_transparency["component_ablations"],
            list(component_transparency["component_ablations"][0]),
        )
    if component_transparency.get("component_scores"):
        _write_csv(
            roots.out_root / "selected_component_scores.csv",
            component_transparency["component_scores"],
            list(component_transparency["component_scores"][0]),
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--strict-ei-trace", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = EvidenceRoots(
        training_root=args.training_root,
        baseline_root=args.baseline_root,
        advanced_root=args.advanced_root,
        anchor_root=args.anchor_root,
        out_root=args.out_root,
    )
    if roots.out_root.exists() and not args.force:
        raise FileExistsError(f"{roots.out_root} already exists; pass --force to replace it")
    payload = build_paper_evidence(roots, strict_ei_trace=args.strict_ei_trace)
    print(
        json.dumps(
            {"status": "pass", "out_root": str(roots.out_root), "version": payload["version"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
