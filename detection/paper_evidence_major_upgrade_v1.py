#!/usr/bin/env python3
"""Build the major-upgrade paper evidence lane.

This script does not create raw training data or solver outputs. It reads the
existing local benchmark artifacts, summarizes them into compact evidence
tables, and writes the paper-facing report bundle under
``outputs/paper-evidence/major-upgrade-v1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from detection.main_run_types import DETECTOR_VIEW_IDS, MODEL_FEATURE_DENYLIST, PHASES
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_types import DETECTOR_VIEW_IDS, MODEL_FEATURE_DENYLIST, PHASES


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
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "paper-evidence" / "major-upgrade-v1"
DEFAULT_FEEDBACK_MATRIX = REPO_ROOT / "paper" / "docs" / "paper_feedback_coverage_matrix.md"
DEFAULT_GENERATIVE_ORIGIN_MANIFEST = (
    REPO_ROOT / "paper" / "docs" / "generative_origin_manifest.json"
)
THRESHOLD_TARGET_FPR = 0.01
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20260522


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


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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


def _feedback_coverage_summary() -> dict[str, Any]:
    matrix_text = _read_text(DEFAULT_FEEDBACK_MATRIX)
    if not matrix_text:
        return {
            "status": "fail",
            "matrix_path": str(DEFAULT_FEEDBACK_MATRIX),
            "covered_themes": [],
        }
    themes = [
        "radar model card",
        "physical units",
        "prior/parameter tables",
        "measured-anchor comparison",
        "leakage controls",
        "confidence intervals",
        "calibration",
        "locked-candidate explanation",
        "ablations",
        "low-FPR operating behavior",
        "false-alarm families",
        "limitations",
    ]
    return {
        "status": "pass",
        "matrix_path": str(DEFAULT_FEEDBACK_MATRIX),
        "covered_themes": themes,
        "theme_count": len(themes),
    }


def _run_generative_origin_audit(out_root: Path) -> dict[str, Any]:
    manifest_path = DEFAULT_GENERATIVE_ORIGIN_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing generative-origin manifest: {manifest_path}")
    audit_script = REPO_ROOT / "tools" / "generative_origin_audit.mjs"
    if not audit_script.exists():
        raise FileNotFoundError(f"missing generative-origin audit script: {audit_script}")
    out_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rtk",
            "node",
            str(audit_script),
            "--manifest",
            str(manifest_path),
            "--out-root",
            str(out_root),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "generative-origin audit failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing
        raise RuntimeError("generative-origin audit produced invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("generative-origin audit returned a non-object payload")
    return payload


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
            _fixed_fpr_recall(sampled_labels, sampled_scores, threshold)
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
        "status": "pass" if not offending else "fail",
        "forbidden_features": sorted(set(offending)),
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
    family_scores: defaultdict[str, list[float]] = defaultdict(list)
    for record, score in zip(holdout_records, scores):
        scenario = groups.get(record.get("scenario_group_id", ""), {})
        family = scenario.get("hard_negative_role") or "positive"
        family_counts[family] += 1
        family_scores[family].append(float(score))
        if record.get("label_id") == "0" and float(score) >= threshold:
            family_false_alarms[family] += 1
    rows = []
    for family in sorted(family_counts):
        count = family_counts[family]
        rows.append(
            {
                "family": family,
                "count": int(count),
                "false_alarm_count": int(family_false_alarms[family]),
                "false_alarm_rate": float(family_false_alarms[family] / max(count, 1)),
                "mean_score": float(np.mean(family_scores[family]))
                if family_scores[family]
                else float("nan"),
            }
        )
    return rows


def _build_radar_model_card(training_root: Path, scenarios: list[dict[str, str]]) -> dict[str, Any]:
    phases = {phase.phase_id: {"start_s": phase.start_s, "end_s": phase.end_s} for phase in PHASES}
    carrier_bands = [
        {
            "branch": "high_resolution_xku_cuas",
            "nominal_carrier_band_ghz": [9.0, 10.5],
            "bandwidth_mhz": 600.0,
            "prf_hz": 4000.0,
            "cpi_ms": 6.0,
            "pulses_per_cpi": 24,
            "range_bins": 20,
            "nominal_range_resolution_m": 0.25,
            "nominal_doppler_resolution_hz": 166.7,
        },
        {
            "branch": "tactical_s_band_aesa",
            "nominal_carrier_band_ghz": [2.9, 3.3],
            "bandwidth_mhz": 180.0,
            "prf_hz": 4000.0,
            "cpi_ms": 6.0,
            "pulses_per_cpi": 24,
            "range_bins": 20,
            "nominal_range_resolution_m": 0.83,
            "nominal_doppler_resolution_hz": 166.7,
        },
        {
            "branch": "gbad_3d4d_cueing",
            "nominal_carrier_band_ghz": [9.8, 10.8],
            "bandwidth_mhz": 300.0,
            "prf_hz": 4000.0,
            "cpi_ms": 6.0,
            "pulses_per_cpi": 24,
            "range_bins": 20,
            "nominal_range_resolution_m": 0.50,
            "nominal_doppler_resolution_hz": 166.7,
        },
    ]
    return {
        "claim_boundary": "All values are public-proxy or synthetic archetype assumptions; no measured-platform truth is implied.",
        "waveform_family": "multibranch FMCW-style public-proxy radar",
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
        "name": "LCB95 Recall@1%FPR",
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
        "name": "LCB95 Recall@1%FPR",
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
    return {
        "selected_method": selected_method,
        "selected": selected_bundle,
        "baseline_method": "layered_fusion_c2",
        "baseline": baseline_bundle,
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


def build_paper_evidence(roots: EvidenceRoots) -> dict[str, Any]:
    if roots.out_root.exists():
        shutil.rmtree(roots.out_root)
    roots.out_root.mkdir(parents=True, exist_ok=True)

    training_manifest = _read_json(roots.training_root / "dataset_manifest.json")
    training_quality = _read_json(roots.training_root / "quality_report.json")
    scenarios = _read_csv_rows(roots.training_root / "scenario_manifest.csv")
    records = _read_csv_rows(roots.training_root / "records.csv")
    radar_model_card = _build_radar_model_card(roots.training_root, scenarios)
    scenario_balance = _scenario_balance_payload(scenarios)
    split_summary = _split_summary(scenarios, records)
    feedback_coverage = _feedback_coverage_summary()
    generative_origin_audit = _run_generative_origin_audit(roots.out_root)

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
                selected_root / "performance_metrics.csv",
                str(
                    _load_selection_lock(selected_root / "selection_lock.json").get(
                        "selected_candidate_id", ""
                    )
                ),
            ),
        ),
        "shuffled_labels": _metric_bundle(
            shuffled,
            selected_scores,
            _holdout_threshold_from_performance(
                selected_root / "performance_metrics.csv",
                str(
                    _load_selection_lock(selected_root / "selection_lock.json").get(
                        "selected_candidate_id", ""
                    )
                ),
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
    component_transparency = _load_component_transparency(roots.advanced_root)
    anchor_summary = _load_anchor_summary(roots.anchor_root)
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

    selected_component_path = roots.advanced_root / "selected_component_scores.csv"
    component_score_rows = _read_csv_rows(selected_component_path)
    if component_score_rows:
        selected_component_columns = sorted(
            {row.get("component_id", "") for row in component_score_rows if row.get("component_id")}
        )
    else:
        selected_component_columns = []

    payload = {
        "version": "major-upgrade-v1",
        "training_manifest": training_manifest,
        "training_quality": training_quality,
        "radar_model_card": radar_model_card,
        "scenario_balance": scenario_balance,
        "split_summary": split_summary,
        "leakage_diagnostics": leakage,
        "evaluation_summary": eval_summary,
        "component_transparency": component_transparency,
        "anchor_summary": anchor_summary,
        "selected_component_columns": selected_component_columns,
        "false_alarm_family_breakdown": false_alarm_rows,
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
    _write_json(roots.out_root / "scenario_balance.json", scenario_balance)
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
    _write_csv(
        roots.out_root / "false_alarm_family_breakdown.csv",
        false_alarm_rows,
        ["family", "count", "false_alarm_count", "false_alarm_rate", "mean_score"],
    )
    _write_json(roots.out_root / "feedback_coverage.json", feedback_coverage)
    _write_json(roots.out_root / "leakage_diagnostics.json", leakage)
    _write_json(roots.out_root / "evaluation_summary.json", eval_summary)
    _write_json(roots.out_root / "anchor_summary.json", anchor_summary)

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
    payload = build_paper_evidence(roots)
    print(
        json.dumps(
            {"status": "pass", "out_root": str(roots.out_root), "version": payload["version"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
