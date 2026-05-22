#!/usr/bin/env python3
"""Generate the radar-first figure set for the major paper upgrade."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - import guard for minimal environments
    np = None  # type: ignore[assignment]

try:
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyArrowPatch, Rectangle
except Exception as exc:  # pragma: no cover - import guard for minimal environments
    raise SystemExit(
        "matplotlib is required to render paper figures; install matplotlib or run this script in the repository dev environment."
    ) from exc


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
DEFAULT_PAPER_EVIDENCE_ROOT = REPO_ROOT / "outputs" / "paper-evidence" / "major-upgrade-v1"
DEFAULT_ANCHOR_ROOT = (
    REPO_ROOT / "outputs" / "real-data" / "kth-drone-bird-human-77ghz" / "kth-measured-v1"
)
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR = DEFAULT_FIGURE_DIR
STRICT_MODE = False
FIGURE_SOURCE_NOTES: dict[str, list[str]] = {}
FIGURE_FALLBACK_NOTES: dict[str, list[str]] = {}

PHASES = ("initial_take_up", "climb_transition", "cruise_altitude")
PHASE_LABELS = {
    "initial_take_up": "Initial take-up",
    "climb_transition": "Climb transition",
    "cruise_altitude": "Cruise altitude",
}
PHASE_WINDOWS = {
    "initial_take_up": "0--30 s",
    "climb_transition": "30--90 s",
    "cruise_altitude": "90--150 s",
}

INK = "#1f2933"
MUTED = "#52606d"
GRID = "#d9e2ec"
BLUE = "#2f6f9f"
TEAL = "#208a7c"
GOLD = "#b7791f"
RED = "#b84a4a"
GREEN = "#3f7f4f"
SLATE = "#eff3f7"
PALE_BLUE = "#e8f2fb"
PALE_TEAL = "#e7f5f1"
PALE_GOLD = "#fff5db"
PALE_RED = "#fae8e8"
PALE_GREEN = "#e9f5ec"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "echoforge_iq",
    ("#10151d", "#24445e", "#2d7c84", "#88b36d", "#f0d56b", "#f7f4e9"),
    N=256,
)


@dataclass(frozen=True)
class FigureRoots:
    training_root: Path
    baseline_root: Path
    advanced_root: Path
    paper_evidence_root: Path
    anchor_root: Path
    output_dir: Path


@dataclass(frozen=True)
class FigureContext:
    roots: FigureRoots
    training_manifest: dict[str, Any]
    training_quality: dict[str, Any]
    scenarios: list[dict[str, str]]
    records: list[dict[str, str]]
    evidence: dict[str, Any]
    anchor_summary: dict[str, Any]
    baseline_metrics: list[dict[str, str]]
    advanced_metrics: list[dict[str, str]]
    advanced_predictions: list[dict[str, str]]
    component_scores: list[dict[str, str]]
    component_ablations: list[dict[str, str]]
    selection_lock: dict[str, Any]


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


def _group_rows(rows: list[dict[str, str]], key_fields: list[str]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        counts[key] = counts.get(key, 0) + 1
    output = []
    for key, count in sorted(counts.items()):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload["count"] = int(count)
        output.append(payload)
    return output


def _entropy(counts: list[int] | list[float]) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log2(p + 1e-12)) / max(math.log2(len(p) + 1e-12), 1e-12))


def _note_source(filename: str, message: str) -> None:
    notes = FIGURE_SOURCE_NOTES.setdefault(filename, [])
    if message not in notes:
        notes.append(message)


def _note_fallback(filename: str, message: str) -> None:
    notes = FIGURE_FALLBACK_NOTES.setdefault(filename, [])
    if message not in notes:
        notes.append(message)


def _require_no_fallback(filename: str) -> None:
    if STRICT_MODE and FIGURE_FALLBACK_NOTES.get(filename):
        raise RuntimeError(f"{filename}: fallback source used in --strict mode")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_note(filename: str) -> str:
    notes = FIGURE_SOURCE_NOTES.get(filename, [])
    return "; ".join(notes) if notes else "tracked evidence"


def _fallback_note(filename: str) -> str:
    notes = FIGURE_FALLBACK_NOTES.get(filename, [])
    return "; ".join(notes) if notes else "none"


def _add_fallback_banner(fig: Any, filename: str) -> None:
    notes = FIGURE_FALLBACK_NOTES.get(filename, [])
    if not notes:
        return
    fig.text(
        0.995,
        0.006,
        "fallback source: " + notes[0],
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=RED,
        bbox={"facecolor": "white", "edgecolor": PALE_RED, "linewidth": 0.6, "pad": 2.0},
    )


def _save_figure(fig: Any, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    metadata = {"Software": "EchoForge paper/generate_figures_major_upgrade_v2.py"}
    description = f"sources={_source_note(filename)}"
    if FIGURE_FALLBACK_NOTES.get(filename):
        description += f"; fallback={_fallback_note(filename)}"
    metadata["Description"] = description
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
    plt.close(fig)
    return path


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
        }
    )


def _add_box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = SLATE,
    edge: str = INK,
    color: str = INK,
    weight: str = "normal",
    size: float = 8.2,
) -> None:
    rect = Rectangle((x, y), w, h, linewidth=1.0, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        weight=weight,
        linespacing=1.15,
    )


def _add_arrow(
    ax: Any, start: tuple[float, float], end: tuple[float, float], color: str = MUTED
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def _setup_diagram(width: float = 7.1, height: float = 4.2) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def _normalize01(values: Any) -> Any:
    if np is None:
        return values
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(arr, (2, 98))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(np.min(arr)), float(np.max(arr))
    if high <= low:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big") & 0xFFFFFFFF


def _load_context(roots: FigureRoots) -> FigureContext:
    training_manifest = _read_json(roots.training_root / "dataset_manifest.json")
    training_quality = _read_json(roots.training_root / "quality_report.json")
    scenarios = _read_csv_rows(roots.training_root / "scenario_manifest.csv")
    records = _read_csv_rows(roots.training_root / "records.csv")
    evidence = _read_json(roots.paper_evidence_root / "paper_evidence_manifest.json")
    if not evidence:
        evidence = {
            "radar_model_card": _read_json(roots.paper_evidence_root / "radar_model_card.json"),
            "scenario_balance": _read_json(roots.paper_evidence_root / "scenario_balance.json"),
            "leakage_diagnostics": _read_json(
                roots.paper_evidence_root / "leakage_diagnostics.json"
            ),
            "evaluation_summary": _read_json(roots.paper_evidence_root / "evaluation_summary.json"),
            "anchor_summary": _read_json(roots.paper_evidence_root / "anchor_summary.json"),
        }
    anchor_summary = evidence.get("anchor_summary", {})
    if not anchor_summary:
        anchor_summary = {
            "manifest": _read_json(roots.anchor_root / "anchor_manifest.json"),
            "distribution": _read_json(roots.anchor_root / "feature_distributions.json"),
            "coefficients": _read_json(roots.anchor_root / "calibration_coefficients.json"),
            "calibration_distance": _read_csv_rows(roots.anchor_root / "calibration_distance.csv"),
            "gap_rows": _read_csv_rows(roots.anchor_root / "sim_real_gap.csv"),
        }
    baseline_metrics = _read_csv_rows(roots.baseline_root / "performance_metrics.csv")
    advanced_metrics = _read_csv_rows(roots.advanced_root / "performance_metrics.csv")
    advanced_predictions = _read_csv_rows(roots.advanced_root / "advanced_predictions.csv")
    component_scores = _read_csv_rows(roots.paper_evidence_root / "selected_component_scores.csv")
    component_ablations = _read_csv_rows(
        roots.paper_evidence_root / "selected_component_ablations.csv"
    )
    selection_lock = _read_json(roots.advanced_root / "selection_lock.json")
    return FigureContext(
        roots=roots,
        training_manifest=training_manifest,
        training_quality=training_quality,
        scenarios=scenarios,
        records=records,
        evidence=evidence,
        anchor_summary=anchor_summary,
        baseline_metrics=baseline_metrics,
        advanced_metrics=advanced_metrics,
        advanced_predictions=advanced_predictions,
        component_scores=component_scores,
        component_ablations=component_ablations,
        selection_lock=selection_lock,
    )


def _load_holdout_rows(
    rows: list[dict[str, str]], *, score_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    holdout = [row for row in rows if row.get("split_role") == "holdout"]
    labels = np.asarray([_safe_int(row.get("label_id")) for row in holdout], dtype=np.int8)
    scores = np.asarray([_safe_float(row.get(score_column)) for row in holdout], dtype=np.float64)
    phases = np.asarray([row.get("phase_id", "") for row in holdout])
    return labels, scores, phases


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
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
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


def _fixed_fpr_recall(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
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
    best = 0.0
    idx = 0
    while idx < len(sorted_scores):
        threshold = sorted_scores[idx]
        while idx < len(sorted_scores) and sorted_scores[idx] == threshold:
            if sorted_labels[idx] == 1:
                tp += 1.0
            else:
                fp += 1.0
            idx += 1
        if fp / negatives <= target_fpr:
            best = max(best, tp / positives)
    return float(best)


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
    points = [{"fpr": 0.0, "tpr": 0.0, "threshold": float(sorted_scores[0] + 1e-12)}]
    tp = 0.0
    fp = 0.0
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
    points = []
    tp = 0.0
    fp = 0.0
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


def _ece(
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


def _metric_bundle(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    binary = _binary_metrics(labels, scores, threshold)
    ece, bins = _ece(labels, scores)
    return {
        "count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "tp": int(binary["tp"]),
        "fp": int(binary["fp"]),
        "tn": int(binary["tn"]),
        "fn": int(binary["fn"]),
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "accuracy": binary["accuracy"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "specificity": binary["specificity"],
        "false_positive_rate": binary["false_positive_rate"],
        "f1": binary["f1"],
        "fixed_fpr_recall": _fixed_fpr_recall(labels, scores),
        "threshold": threshold,
        "brier_score": _brier_score(labels, scores),
        "ece": ece,
        "calibration_bins": bins,
        "curves": {"roc": _roc_curve(labels, scores), "pr": _pr_curve(labels, scores)},
    }


def _bootstrap_group_cis(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    *,
    rounds: int = 200,
    seed: int = 20260522,
) -> dict[str, Any]:
    labels = np.asarray([_safe_int(row.get("label_id")) for row in rows], dtype=np.int8)
    groups = np.asarray([row.get("scenario_group_id", "") for row in rows])
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        return {}
    rng = np.random.default_rng(seed)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    samples = {
        "average_precision": [],
        "roc_auc": [],
        "brier_score": [],
        "ece": [],
        "fixed_fpr_recall": [],
    }
    for _ in range(rounds):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sampled_labels = labels[sampled_indices]
        sampled_scores = scores[sampled_indices]
        samples["average_precision"].append(_average_precision(sampled_labels, sampled_scores))
        samples["roc_auc"].append(_roc_auc(sampled_labels, sampled_scores))
        samples["brier_score"].append(_brier_score(sampled_labels, sampled_scores))
        ece, _ = _ece(sampled_labels, sampled_scores)
        samples["ece"].append(ece)
        samples["fixed_fpr_recall"].append(_fixed_fpr_recall(sampled_labels, sampled_scores))
    payload = {}
    for metric, values in samples.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        payload[metric] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "mean": float(np.mean(arr)),
        }
    return payload


def _top_level_source_statuses(roots: FigureRoots) -> list[tuple[str, Path, bool, str]]:
    statuses = [
        (
            "training manifest",
            roots.training_root / "dataset_manifest.json",
            (roots.training_root / "dataset_manifest.json").exists(),
            "ready" if (roots.training_root / "dataset_manifest.json").exists() else "missing",
        ),
        (
            "training records",
            roots.training_root / "records.csv",
            (roots.training_root / "records.csv").exists(),
            "ready" if (roots.training_root / "records.csv").exists() else "missing",
        ),
        (
            "paper evidence",
            roots.paper_evidence_root / "paper_evidence_manifest.json",
            (roots.paper_evidence_root / "paper_evidence_manifest.json").exists(),
            "ready"
            if (roots.paper_evidence_root / "paper_evidence_manifest.json").exists()
            else "missing",
        ),
        (
            "anchor summary",
            roots.anchor_root / "feature_distributions.json",
            (roots.anchor_root / "feature_distributions.json").exists(),
            "ready" if (roots.anchor_root / "feature_distributions.json").exists() else "missing",
        ),
    ]
    return statuses


def _report_sources(strict: bool, roots: FigureRoots) -> None:
    statuses = _top_level_source_statuses(roots)
    for name, path, available, detail in statuses:
        marker = "available" if available else "missing"
        print(
            f"source[{name}]: {marker} ({_display_path(path)}; {detail})",
            file=__import__("sys").stderr,
        )
    missing = [status for status in statuses if not status[2]]
    if strict and missing:
        raise SystemExit("--strict requires all source artifacts")


def _scenario_balance_counts(context: FigureContext) -> dict[str, Any]:
    balance = context.evidence.get("scenario_balance", {})
    if not balance:
        balance = _read_json(context.roots.paper_evidence_root / "scenario_balance.json")
    return balance


def _evaluation_summary(context: FigureContext) -> dict[str, Any]:
    summary = context.evidence.get("evaluation_summary", {})
    if not summary:
        summary = _read_json(context.roots.paper_evidence_root / "evaluation_summary.json")
    return summary


def figure_architecture_stack(context: FigureContext) -> Path:
    filename = "architecture_stack.png"
    radar_model = context.evidence.get("radar_model_card", {})
    if radar_model:
        _note_source(filename, "paper evidence radar model card")
    else:
        _note_fallback(filename, "radar model card missing; using generic stack")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig, ax = _setup_diagram(7.4, 4.7)
    ax.text(0.15, 5.7, "EchoForge radar signal chain", fontsize=12, weight="bold")
    ax.text(
        0.15,
        5.35,
        "Public-proxy object priors, sensor archetypes, detector views, and evidence outputs stay separated by claim boundary",
        fontsize=8.5,
        color=MUTED,
    )
    layers = [
        ("Public-proxy object priors\nfixed-wing pusher-prop", PALE_BLUE, BLUE),
        ("Radar archetype\ncarrier/bandwidth/PRF/CPI", PALE_TEAL, TEAL),
        ("Propagation and clutter\nmultipath / weather / RFI", PALE_GOLD, GOLD),
        ("Complex IQ and cue streams\nradar + acoustic + passive RF", PALE_GREEN, GREEN),
        ("Detector views and leakage rails\nfeature denylist + split locks", PALE_RED, RED),
        ("Evidence bundle\nmetrics, ablations, anchor compare", SLATE, INK),
    ]
    x, w, h = 0.55, 5.8, 0.58
    y0 = 4.6
    for idx, (text, face, edge) in enumerate(layers):
        y = y0 - idx * 0.72
        _add_box(
            ax, x, y, w, h, text, face=face, edge=edge, weight="bold" if idx == 0 else "normal"
        )
        if idx < len(layers) - 1:
            _add_arrow(ax, (x + w / 2, y - 0.02), (x + w / 2, y - 0.16), color=edge)

    ax.text(7.0, 5.05, "Model card snapshot", fontsize=9.2, weight="bold")
    card_lines = [
        f"waveform: {radar_model.get('waveform_family', 'public-proxy radar')}",
        "pulses: 24 per CPI",
        "range bins: 20",
        "PRF/CPI: 4 kHz / 6 ms",
        "receiver impairments: AGC, drift, quantization",
        "cue family: acoustic + passive RF",
    ]
    for idx, line in enumerate(card_lines):
        _add_box(ax, 6.95, 4.3 - idx * 0.53, 2.45, 0.34, line, face="white", edge=GRID, size=7.6)
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_monte_carlo_split_flow(context: FigureContext) -> Path:
    filename = "monte_carlo_split_flow.png"
    balance = _scenario_balance_counts(context)
    if balance:
        _note_source(filename, "scenario balance and leakage diagnostics")
    else:
        _note_fallback(filename, "scenario balance missing; using summary placeholders")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig = plt.figure(figsize=(10.1, 5.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.45, 1.45, 1.0], hspace=0.32, wspace=0.25)
    fig.suptitle("Scenario balance and leakage diagnostics", fontsize=12.2, weight="bold", y=0.98)
    fig.text(
        0.02,
        0.02,
        "Scenario groups are locked before phase expansion. The paper evidence keeps label, site, range, aspect, noise, and hard-negative family counts together.",
        fontsize=7.2,
        color=MUTED,
    )
    marginal_counts = balance.get("marginal_counts", {})
    ordered_keys = [
        ("split", "Split"),
        ("site", "Site"),
        ("range", "Range"),
        ("aspect", "Aspect"),
        ("noise", "Noise"),
        ("hard_negative_role", "Hard-negative"),
    ]
    for idx, (key, title) in enumerate(ordered_keys):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        counts = marginal_counts.get(key, {})
        if not counts:
            counts = {"missing": 1}
            _note_fallback(filename, f"{key} counts missing")
        names = list(counts)
        values = [counts[name] for name in names]
        palette = [BLUE, TEAL, GOLD, RED, GREEN]
        ax.barh(
            range(len(names)), values, color=[palette[i % len(palette)] for i in range(len(names))]
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.0)
        ax.set_title(title, fontsize=9.4, weight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.01, i, f"{value}", va="center", fontsize=6.8)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    leakage = context.evidence.get("leakage_diagnostics", {})
    canary = leakage.get("canary_forbidden_feature_test", {})
    ax.text(0.02, 0.94, "Leakage checks", fontsize=9.4, weight="bold", transform=ax.transAxes)
    lines = [
        f"canary test: {canary.get('status', 'missing')}",
        f"stratum baseline AP: {leakage.get('stratum_only_baseline', {}).get('average_precision', float('nan')):.3f}",
        f"metadata baseline AP: {leakage.get('metadata_only_baseline', {}).get('average_precision', float('nan')):.3f}",
        f"label shuffle AP: {leakage.get('label_shuffle_sanity', {}).get('shuffled_labels', {}).get('average_precision', float('nan')):.3f}",
    ]
    for idx, line in enumerate(lines):
        _add_box(ax, 0.03, 0.74 - idx * 0.14, 0.92, 0.10, line, face="white", edge=GRID, size=7.1)

    ax = fig.add_subplot(gs[1:, 2])
    ax.axis("off")
    imbalance = balance.get("imbalance_scores", {})
    ax.text(0.02, 0.94, "Marginal imbalance", fontsize=9.4, weight="bold", transform=ax.transAxes)
    for idx, (name, value) in enumerate(sorted(imbalance.items())):
        ax.text(
            0.04, 0.80 - idx * 0.12, f"{name}: {value:.3f}", fontsize=7.3, transform=ax.transAxes
        )
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_kpi_ranking(context: FigureContext) -> Path:
    filename = "kpi_ranking.png"
    evaluation = _evaluation_summary(context)
    if evaluation:
        _note_source(filename, "evaluation summary with curves and bootstrap CIs")
    else:
        _note_fallback(filename, "evaluation summary missing; using deterministic defaults")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig = plt.figure(figsize=(10.4, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    fig.suptitle(
        "Holdout ranking, primary KPI, ROC/PR curves, and calibration",
        fontsize=12.2,
        weight="bold",
        y=0.98,
    )

    ax = fig.add_subplot(gs[0, 0])
    rows = [
        row
        for row in context.baseline_metrics
        if row.get("split_role") == "holdout" and row.get("phase_id") == "all"
    ]
    rows.sort(key=lambda row: _safe_float(row.get("average_precision")), reverse=True)
    selected_method = evaluation.get("selected_method", "")
    selected_row = evaluation.get("selected", {})
    primary_kpi = evaluation.get("primary_kpi", {})
    primary_label = primary_kpi.get("name", "LCB95 Recall@1%FPR")
    primary_value = _safe_float(
        primary_kpi.get("value"),
        _safe_float(selected_row.get("fixed_fpr_recall")),
    )
    labels = [row.get("method", "") for row in rows]
    values = [_safe_float(row.get("average_precision")) for row in rows]
    positions = list(range(len(labels)))
    colors = [GREEN if label == "layered_fusion_c2" else BLUE for label in labels]
    if selected_method and selected_method not in labels:
        labels.append(selected_method)
        values.append(_safe_float(selected_row.get("average_precision")))
        colors.append(INK)
        positions.append(len(positions))
    ax.barh(positions, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([label.replace("_", "\n") for label in labels], fontsize=7.3)
    ax.set_xlabel("Holdout average precision", fontsize=9.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for pos, value, label in zip(positions, values, labels):
        if label in {"layered_fusion_c2", selected_method}:
            ci = evaluation.get("selected" if label == selected_method else "baseline", {}).get(
                "group_block_bootstrap_ci", {}
            )
            ap_ci = ci.get("average_precision", {})
            err = None
            if ap_ci:
                err = [[value - ap_ci.get("low", value)], [ap_ci.get("high", value) - value]]
            if err:
                ax.errorbar(value, pos, xerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
    ax.set_title("Holdout AP ranking", fontsize=9.6, weight="bold")

    right = gs[0, 1].subgridspec(2, 1, hspace=0.24)
    roc_ax = fig.add_subplot(right[0, 0])
    pr_ax = fig.add_subplot(right[1, 0])
    for ax2, metric in ((roc_ax, "roc"), (pr_ax, "pr")):
        sel_curve = evaluation.get("curves", {}).get("selected", {}).get(metric, [])
        base_curve = evaluation.get("curves", {}).get("baseline", {}).get(metric, [])
        if metric == "roc":
            ax2.plot(
                [p["fpr"] for p in base_curve],
                [p["tpr"] for p in base_curve],
                color=GREEN,
                linewidth=1.8,
                label="baseline",
            )
            ax2.plot(
                [p["fpr"] for p in sel_curve],
                [p["tpr"] for p in sel_curve],
                color=INK,
                linewidth=1.8,
                label="selected",
            )
            ax2.set_xlabel("False-positive rate", fontsize=8.0)
            ax2.set_ylabel("True-positive rate", fontsize=8.0)
            ax2.set_title("ROC curve", fontsize=9.4, weight="bold")
            ax2.set_xlim(0.0, 0.08)
            ax2.set_ylim(0.0, 1.02)
        else:
            ax2.plot(
                [p["recall"] for p in base_curve],
                [p["precision"] for p in base_curve],
                color=GREEN,
                linewidth=1.8,
                label="baseline",
            )
            ax2.plot(
                [p["recall"] for p in sel_curve],
                [p["precision"] for p in sel_curve],
                color=INK,
                linewidth=1.8,
                label="selected",
            )
            ax2.set_xlabel("Recall", fontsize=8.0)
            ax2.set_ylabel("Precision", fontsize=8.0)
            ax2.set_title("PR curve", fontsize=9.4, weight="bold")
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.02)
        ax2.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax2.yaxis.grid(True, color=GRID, linewidth=0.7)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
    roc_ax.legend(loc="lower right", fontsize=7.2, frameon=False)
    pr_ax.legend(loc="lower left", fontsize=7.2, frameon=False)
    roc_ax.text(
        0.02,
        0.04,
        f"{primary_label}: {primary_value:.3f}",
        transform=roc_ax.transAxes,
        fontsize=7.0,
        color=MUTED,
    )
    pr_ax.text(
        0.02,
        0.04,
        f"Brier={selected_row.get('brier_score', float('nan')):.3f}  ECE={selected_row.get('ece', float('nan')):.3f}",
        transform=pr_ax.transAxes,
        fontsize=7.0,
        color=MUTED,
    )
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_phase_kpi(context: FigureContext) -> Path:
    filename = "phase_kpi.png"
    evaluation = _evaluation_summary(context)
    if evaluation:
        _note_source(filename, "phase metrics and false-alarm family breakdown")
    else:
        _note_fallback(filename, "phase metrics missing; using deterministic defaults")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig = plt.figure(figsize=(10.0, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)
    fig.suptitle(
        "Phase behavior and false-alarm family breakdown", fontsize=12.1, weight="bold", y=0.98
    )

    phase_metrics = evaluation.get("phase_metrics", {})
    selected = phase_metrics.get(evaluation.get("selected_method", ""), {})
    baseline = phase_metrics.get("layered_fusion_c2", {})
    phase_order = list(PHASES)
    x = np.arange(len(phase_order))
    width = 0.32
    ax = fig.add_subplot(gs[0, 0])
    selected_ap = [selected.get(phase, {}).get("average_precision", 0.0) for phase in phase_order]
    selected_recall = [selected.get(phase, {}).get("recall", 0.0) for phase in phase_order]
    baseline_ap = [baseline.get(phase, {}).get("average_precision", 0.0) for phase in phase_order]
    ax.bar(
        x - width / 2, baseline_ap, width, color=PALE_GREEN, edgecolor=GREEN, label="baseline AP"
    )
    ax.bar(x + width / 2, selected_ap, width, color=INK, edgecolor=GOLD, label="selected AP")
    for idx, value in enumerate(selected_recall):
        ax.plot([idx - 0.12, idx + 0.12], [value, value], color=RED, linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{PHASE_LABELS[p]}\n{PHASE_WINDOWS[p]}" for p in phase_order], fontsize=7.4
    )
    ax.set_ylabel("AP / recall", fontsize=9.0)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    ax.set_title("Per-phase holdout diagnostics", fontsize=9.6, weight="bold")

    fa_ax = fig.add_subplot(gs[0, 1])
    fa_rows = _read_csv_rows(context.roots.paper_evidence_root / "false_alarm_family_breakdown.csv")
    if not fa_rows:
        fa_rows = _fallback_false_alarm_rows()
    fa_rows = sorted(
        fa_rows, key=lambda row: _safe_float(row.get("false_alarm_count")), reverse=True
    )
    families = [row.get("family", "") for row in fa_rows]
    fa_values = [_safe_float(row.get("false_alarm_count")) for row in fa_rows]
    palette = [RED, GOLD, BLUE, TEAL, GREEN]
    fa_ax.barh(
        range(len(families)),
        fa_values,
        color=[palette[i % len(palette)] for i in range(len(families))],
    )
    fa_ax.set_yticks(range(len(families)))
    fa_ax.set_yticklabels(families, fontsize=7.2)
    fa_ax.set_xlabel("False alarms", fontsize=9.0)
    fa_ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    fa_ax.set_axisbelow(True)
    for spine in ("top", "right"):
        fa_ax.spines[spine].set_visible(False)
    fa_ax.set_title("False-alarm family breakdown", fontsize=9.6, weight="bold")
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def _fallback_false_alarm_rows() -> list[dict[str, Any]]:
    return [
        {"family": "single_bird", "false_alarm_count": 1, "count": 1},
        {"family": "bird_flock", "false_alarm_count": 0, "count": 1},
        {"family": "rc_fixed_wing", "false_alarm_count": 0, "count": 1},
    ]


def _iq_ref_parser(ref: str) -> tuple[Path, int] | None:
    if "#row=" not in ref or not ref.endswith(tuple(str(i) for i in range(10))):
        pass
    if "#row=" not in ref:
        return None
    shard, row = ref.split("#row=", 1)
    try:
        row_index = int(row)
    except ValueError:
        return None
    return (
        REPO_ROOT
        / "outputs"
        / "training-data"
        / "runit-fixed-wing-pusher-proxy-v2-main-run"
        / shard,
        row_index,
    )


def _range_doppler(sample: np.ndarray) -> np.ndarray:
    rd = np.fft.fftshift(np.fft.fft2(sample))
    return _normalize01(np.log1p(np.abs(rd)))


def _read_iq_sample(context: FigureContext, record_id: str) -> np.ndarray | None:
    record = next((row for row in context.records if row.get("record_id") == record_id), None)
    if not record:
        return None
    parsed = _iq_ref_parser(record.get("raw_complex_iq_ref", ""))
    if parsed is None:
        return None
    shard_path, row_index = parsed
    if not shard_path.exists() or np is None:
        return None
    with np.load(shard_path, allow_pickle=False) as shard:
        iq = shard["iq"]
        if row_index < 0 or row_index >= iq.shape[0]:
            return None
        sample = np.asarray(iq[row_index])
    if sample.ndim == 3:
        sample = sample.mean(axis=0)
    elif sample.ndim > 3:
        sample = sample.reshape((-1,) + sample.shape[-2:]).mean(axis=0)
    return sample


def figure_iq_drone_samples(context: FigureContext) -> Path:
    filename = "iq_drone_samples.png"
    card = context.evidence.get("radar_model_card", {})
    if card:
        _note_source(filename, "radar model card and public-proxy object priors")
    else:
        _note_fallback(filename, "radar model card missing; using placeholder priors")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig = plt.figure(figsize=(10.0, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Radar model card and signal-chain parameter panel", fontsize=12.1, weight="bold", y=0.98
    )

    model_box = fig.add_subplot(gs[:, 0])
    model_box.axis("off")
    model_box.text(
        0.02,
        0.94,
        "Public-proxy assumptions",
        fontsize=9.4,
        weight="bold",
        transform=model_box.transAxes,
    )
    lines = [
        "positive class: fixed-wing pusher-prop public proxy",
        "carrier bands: X/Ku, S, GBAD 3D/4D (nominal)",
        "bandwidth: 180--600 MHz assumption band",
        "PRF/CPI: 4 kHz / 6 ms",
        "pulses per CPI: 24",
        "range bins: 20",
        "range resolution: 0.25--0.83 m nominal",
    ]
    for idx, line in enumerate(lines):
        _add_box(
            model_box, 0.03, 0.79 - idx * 0.1, 0.94, 0.075, line, face="white", edge=GRID, size=7.1
        )

    for idx, band in enumerate(card.get("carrier_bands", [])[:2]):
        ax = fig.add_subplot(gs[0, idx + 1])
        ax.barh(
            ["bandwidth", "resolution"],
            [band.get("bandwidth_mhz", 0.0), band.get("nominal_range_resolution_m", 0.0)],
            color=[BLUE, GOLD],
        )
        ax.set_title(band.get("branch", "branch"), fontsize=9.2, weight="bold")
        ax.set_xlabel("MHz / m", fontsize=8.0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    impairments = card.get("receiver_impairments", [])[:4]
    ax.barh(range(len(impairments)), [1] * len(impairments), color=PALE_RED, edgecolor=RED)
    ax.set_yticks(range(len(impairments)))
    ax.set_yticklabels([item.replace("_", " ") for item in impairments], fontsize=7.0)
    ax.set_xlim(0, 1.2)
    ax.set_title("Receiver impairments", fontsize=9.2, weight="bold")
    ax.axis("off")
    ax = fig.add_subplot(gs[1, 2])
    cue_defs = card.get("cue_definitions", {})
    cue_text = [
        f"acoustic: {cue_defs.get('acoustic', 'node cadence and agreement')}",
        f"passive RF: {cue_defs.get('passive_rf', 'no-signal / RFI geometry')}",
    ]
    for idx, line in enumerate(cue_text):
        _add_box(ax, 0.02, 0.64 - idx * 0.24, 0.96, 0.16, line, face="white", edge=GRID, size=7.0)
    ax.axis("off")
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_iq_negative_samples(context: FigureContext) -> Path:
    filename = "iq_negative_samples.png"
    if context.records:
        _note_source(filename, "training IQ range-Doppler samples")
    else:
        _note_fallback(filename, "training records missing; using synthetic heatmaps")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig = plt.figure(figsize=(10.1, 5.1))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.0, 1.0, 1.0], hspace=0.18, wspace=0.1)
    fig.suptitle(
        "Range-Doppler proxy diagnostics for positive and hard-negative samples",
        fontsize=12.0,
        weight="bold",
        y=0.98,
    )
    groups = [("sg_00016", True), ("sg_00000", False)]
    image = None
    for row, (group_id, positive) in enumerate(groups):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.02,
            0.95,
            "\n".join(
                [
                    group_id,
                    "public-proxy positive" if positive else "hard negative / artifact",
                    "range-Doppler magnitude",
                ]
            ),
            ha="left",
            va="top",
            fontsize=8.0,
            transform=label_ax.transAxes,
        )
        for phase_idx, phase in enumerate(PHASES, start=1):
            record_id = f"{group_id}_{phase}"
            sample = _read_iq_sample(context, record_id)
            ax = fig.add_subplot(gs[row, phase_idx])
            if sample is None or np is None:
                _note_fallback(filename, f"missing IQ sample {record_id}")
                if STRICT_MODE:
                    _require_no_fallback(filename)
                sample = np.zeros((24, 20), dtype=np.float64) if np is not None else None
            if sample is None:
                continue
            rd = _range_doppler(sample)
            image = ax.imshow(
                rd, aspect="auto", origin="lower", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0
            )
            if row == 0:
                ax.set_title(f"{PHASE_LABELS[phase]}\n{PHASE_WINDOWS[phase]}", fontsize=8.2)
            ax.set_xticks([0, rd.shape[1] // 2, rd.shape[1] - 1])
            ax.set_yticks([0, rd.shape[0] // 2, rd.shape[0] - 1])
            ax.set_xticklabels(["near", "mid", "far"], fontsize=6.5)
            ax.set_yticklabels(["low", "mid", "high"], fontsize=6.5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color(GRID)
    if image is not None:
        cbar = fig.colorbar(image, ax=fig.axes, shrink=0.82, pad=0.01)
        cbar.set_label("normalized magnitude", fontsize=7.4)
        cbar.ax.tick_params(labelsize=7.0)
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_detector_ml_pipeline(context: FigureContext) -> Path:
    filename = "detector_ml_pipeline.png"
    component_rows = context.component_ablations
    if component_rows:
        _note_source(filename, "selected component ablations and transparency outputs")
    else:
        _note_fallback(filename, "component transparency missing; using generic pipeline")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig, ax = _setup_diagram(7.4, 4.7)
    ax.text(
        0.15, 5.7, "Locked-candidate transparency and ablation path", fontsize=12.0, weight="bold"
    )
    ax.text(
        0.15,
        5.35,
        "The selected meta-fusion candidate is shown with no-calibration, top-k, component-drop, and modality-drop ablations.",
        fontsize=8.4,
        color=MUTED,
    )
    stages = [
        ("locked advanced\ncandidate", PALE_BLUE, BLUE),
        ("component scores\nand aliases", PALE_TEAL, TEAL),
        ("ablation rows\nno-cal / top-k / drop", PALE_GOLD, GOLD),
        ("paper evidence bundle\nJSON + CSV", PALE_GREEN, GREEN),
    ]
    x_positions = [0.35, 2.35, 4.35, 6.35]
    for idx, (text, face, edge) in enumerate(stages):
        _add_box(ax, x_positions[idx], 4.25, 1.65, 0.68, text, face=face, edge=edge, size=7.7)
        if idx < len(stages) - 1:
            _add_arrow(
                ax, (x_positions[idx] + 1.65, 4.59), (x_positions[idx + 1], 4.59), color=edge
            )

    ax.text(0.4, 3.35, "Ablation summary", fontsize=9.4, weight="bold")
    sorted_rows = sorted(
        component_rows,
        key=lambda row: (
            _safe_float(row.get("holdout_average_precision")),
            row.get("variant_id", ""),
        ),
        reverse=True,
    )
    summary_lines = []
    for row in sorted_rows[:8]:
        summary_lines.append(
            (
                row.get("variant_id", ""),
                _safe_float(row.get("holdout_average_precision")),
            )
        )
    if not summary_lines:
        summary_lines = [("selected", 0.0)]
    for idx, (label, value) in enumerate(summary_lines[:6]):
        _add_box(
            ax,
            0.35,
            2.95 - idx * 0.46,
            6.35,
            0.32,
            f"{label}: holdout AP {value:.3f}",
            face="white",
            edge=GRID,
            size=7.1,
        )
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_locked_algorithm(context: FigureContext) -> Path:
    filename = "locked_algorithm.png"
    anchor = context.anchor_summary
    if anchor:
        _note_source(filename, "KTH compare-only anchor summary")
    else:
        _note_fallback(filename, "anchor summary missing; using placeholder compare-only panel")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.subplots_adjust(top=0.86)
    fig.suptitle("Compare-only KTH anchor overlay", fontsize=12.0, weight="bold", y=0.98)
    ax.set_title(
        "Measured anchor role: compare-only distribution check", fontsize=9.4, weight="bold"
    )
    selected_features = anchor.get("selected_features", {})
    feature_names = ["micro_doppler_bandwidth_hz", "spectral_entropy", "range_m", "return_power_db"]
    values = []
    labels = []
    for name in feature_names:
        stats = selected_features.get(name, {})
        if isinstance(stats, dict):
            values.append(_safe_float(stats.get("q50")))
        else:
            values.append(float("nan"))
        labels.append(name)
    ax.barh(labels, values, color=[BLUE, TEAL, GOLD, GREEN], edgecolor="white")
    ax.set_xlabel("Median / compare-only observable", fontsize=9.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    note_lines = [
        "KTH drone/bird/human collection is compare-only for hard-negative realism.",
        "No positive fixed-wing proxy truth is inferred from it.",
        "Raw measured traces stay outside Git.",
    ]
    for idx, line in enumerate(note_lines):
        ax.text(0.02, 0.05 - idx * 0.07, line, transform=ax.transAxes, fontsize=7.3, color=MUTED)
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def _fallback_false_alarm_rows() -> list[dict[str, Any]]:
    return [
        {"family": "single_bird", "false_alarm_count": 1},
        {"family": "bird_flock", "false_alarm_count": 1},
        {"family": "rc_fixed_wing", "false_alarm_count": 1},
    ]


def generate_all(context: FigureContext) -> list[Path]:
    _configure_matplotlib()
    return [
        figure_architecture_stack(context),
        figure_monte_carlo_split_flow(context),
        figure_kpi_ranking(context),
        figure_phase_kpi(context),
        figure_iq_drone_samples(context),
        figure_iq_negative_samples(context),
        figure_detector_ml_pipeline(context),
        figure_locked_algorithm(context),
    ]


def main() -> int:
    global FIGURE_DIR
    global STRICT_MODE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--paper-evidence-root", type=Path, default=DEFAULT_PAPER_EVIDENCE_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    roots = FigureRoots(
        training_root=args.training_root,
        baseline_root=args.baseline_root,
        advanced_root=args.advanced_root,
        paper_evidence_root=args.paper_evidence_root,
        anchor_root=args.anchor_root,
        output_dir=args.output_dir,
    )
    FIGURE_DIR = args.output_dir
    STRICT_MODE = args.strict
    _report_sources(args.strict, roots)
    context = _load_context(roots)
    paths = generate_all(context)
    for path in paths:
        print(_display_path(path))
    return 0


def _report_sources(strict: bool, roots: FigureRoots) -> None:
    statuses = _top_level_source_statuses(roots)
    for name, path, available, detail in statuses:
        marker = "available" if available else "missing"
        print(
            f"source[{name}]: {marker} ({_display_path(path)}; {detail})",
            file=__import__("sys").stderr,
        )
    if strict and not all(available for _, _, available, _ in statuses):
        raise SystemExit("--strict requires all source artifacts")


if __name__ == "__main__":
    raise SystemExit(main())
