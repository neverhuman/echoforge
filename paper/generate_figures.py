#!/usr/bin/env python3
"""Generate deterministic PNG figures for the EchoForge paper.

The figures are static paper assets. They prefer the generated full-run output
under outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run when it
is available, and otherwise fall back to deterministic public-proxy heatmaps.
No raw arrays or solver outputs are written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
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
        "matplotlib is required to render paper figures; install matplotlib "
        "or run this script in the repository dev environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "outputs" / "training-data" / "runit-fixed-wing-pusher-proxy-v2-main-run"
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
DRONE_GROUPS = ("sg_00016", "sg_00030")
NEGATIVE_GROUPS = ("sg_00000", "sg_00001")
IQ_REF_RE = re.compile(r"^(?P<path>.+\.npz)#row=(?P<row>\d+)$")
EI_METHOD_ID = "meta_fusion.top8.v2_aggressive.geodesic_odds"
BASELINE_METHOD_ORDER = (
    "high_resolution_xku_cuas",
    "tactical_s_band_aesa",
    "gbad_3d4d_cueing",
    "distributed_acoustic_cue",
    "tabular_ml_baseline",
    "sequence_ml_proxy",
    "layered_fusion_c2",
)
ALL_KPI_METHOD_ORDER = BASELINE_METHOD_ORDER + (EI_METHOD_ID,)
KPI_LABELS = {
    "high_resolution_xku_cuas": "High-resolution X/Ku\nbranch",
    "tactical_s_band_aesa": "Tactical S-band\nbranch",
    "gbad_3d4d_cueing": "GBAD 3D/4D\ncueing",
    "distributed_acoustic_cue": "Distributed acoustic\ncue",
    "tabular_ml_baseline": "Tabular ML\nbaseline",
    "sequence_ml_proxy": "Sequence ML\nproxy",
    "layered_fusion_c2": "Layered fusion C2\nbest prior fusion",
    EI_METHOD_ID: "Locked advanced\ncandidate",
}
KPI_GROUP_LABELS = {
    "high_resolution_xku_cuas": "Sensor branches",
    "tactical_s_band_aesa": "Sensor branches",
    "gbad_3d4d_cueing": "Sensor branches",
    "distributed_acoustic_cue": "Sensor branches",
    "tabular_ml_baseline": "Prior ML",
    "sequence_ml_proxy": "Prior ML",
    "layered_fusion_c2": "Prior fusion",
    EI_METHOD_ID: "Locked candidate",
}
DEFAULT_HOLDOUT_KPI = {
    "high_resolution_xku_cuas": {
        "average_precision": 0.112345,
        "roc_auc": 0.898012,
        "f1": 0.210526,
    },
    "tactical_s_band_aesa": {"average_precision": 0.047577, "roc_auc": 0.897779, "f1": 0.095238},
    "gbad_3d4d_cueing": {"average_precision": 0.040623, "roc_auc": 0.926655, "f1": 0.081818},
    "distributed_acoustic_cue": {
        "average_precision": 0.093170,
        "roc_auc": 0.901838,
        "f1": 0.125000,
    },
    "tabular_ml_baseline": {"average_precision": 0.125996, "roc_auc": 0.927521, "f1": 0.138614},
    "sequence_ml_proxy": {"average_precision": 0.116239, "roc_auc": 0.917421, "f1": 0.062500},
    "layered_fusion_c2": {"average_precision": 0.128188, "roc_auc": 0.937723, "f1": 0.240964},
    EI_METHOD_ID: {"average_precision": 0.824845, "roc_auc": 0.916564, "f1": 0.837209},
}
DEFAULT_PHASE_KPI = {
    "layered_fusion_c2": {
        "initial_take_up": {"average_precision": 0.049621, "roc_auc": 0.809316},
        "climb_transition": {"average_precision": 0.107564, "roc_auc": 0.959534},
        "cruise_altitude": {"average_precision": 0.422702, "roc_auc": 0.991622},
    },
    EI_METHOD_ID: {
        "initial_take_up": {"average_precision": 0.627731, "roc_auc": 0.812500},
        "climb_transition": {"average_precision": 0.844530, "roc_auc": 0.937249},
        "cruise_altitude": {"average_precision": 1.000000, "roc_auc": 1.000000},
    },
}
BASELINE_METRICS_PATH = (
    REPO_ROOT
    / "outputs"
    / "detection"
    / "runit-fixed-wing-pusher-proxy-v2-main-run"
    / "performance_metrics.csv"
)
EI_PREDICTIONS_PATH = (
    REPO_ROOT
    / "outputs"
    / "detection"
    / "runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution"
    / "advanced_predictions.csv"
)
EI_QUALITY_PATH = (
    REPO_ROOT
    / "outputs"
    / "detection"
    / "runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution"
    / "fusion_quality_report.json"
)

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
class RunContext:
    available: bool
    manifest: dict[str, Any]
    quality: dict[str, Any]
    records: dict[str, dict[str, str]]
    scenarios: dict[str, dict[str, str]]


@dataclass(frozen=True)
class SourceStatus:
    name: str
    path: Path
    available: bool
    detail: str


def note_source(filename: str, message: str) -> None:
    notes = FIGURE_SOURCE_NOTES.setdefault(filename, [])
    if message not in notes:
        notes.append(message)


def note_fallback(filename: str, message: str) -> None:
    notes = FIGURE_FALLBACK_NOTES.setdefault(filename, [])
    if message not in notes:
        notes.append(message)


def first_note(notes: list[str], fallback: str) -> str:
    return notes[0] if notes else fallback


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def add_fallback_banner(fig: Any, filename: str) -> None:
    notes = FIGURE_FALLBACK_NOTES.get(filename, [])
    if not notes:
        return
    fig.text(
        0.995,
        0.006,
        "fallback source: " + first_note(notes, "deterministic defaults"),
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=RED,
        bbox={"facecolor": "white", "edgecolor": PALE_RED, "linewidth": 0.6, "pad": 2.0},
    )


def require_no_fallback(filename: str) -> None:
    if STRICT_MODE and FIGURE_FALLBACK_NOTES.get(filename):
        raise RuntimeError(f"{filename}: fallback source used in --strict mode")


def source_statuses() -> list[SourceStatus]:
    training_required = [
        RUN_ROOT / "dataset_manifest.json",
        RUN_ROOT / "quality_report.json",
        RUN_ROOT / "records.csv",
        RUN_ROOT / "scenario_manifest.csv",
        RUN_ROOT / "raw_complex_iq",
    ]
    training_missing = [path for path in training_required if not path.exists()]
    training_detail = (
        "ready"
        if not training_missing and np is not None
        else "missing " + ", ".join(display_path(path) for path in training_missing)
        if training_missing
        else "numpy unavailable for IQ sample reads"
    )
    return [
        SourceStatus(
            "training run", RUN_ROOT, not training_missing and np is not None, training_detail
        ),
        SourceStatus(
            "baseline metrics CSV",
            BASELINE_METRICS_PATH,
            BASELINE_METRICS_PATH.is_file(),
            "ready" if BASELINE_METRICS_PATH.is_file() else "missing",
        ),
        SourceStatus(
            "advanced predictions",
            EI_PREDICTIONS_PATH,
            EI_PREDICTIONS_PATH.is_file(),
            "ready" if EI_PREDICTIONS_PATH.is_file() else "missing",
        ),
        SourceStatus(
            "quality report",
            EI_QUALITY_PATH,
            EI_QUALITY_PATH.is_file(),
            "ready" if EI_QUALITY_PATH.is_file() else "missing",
        ),
    ]


def report_sources(strict: bool) -> None:
    statuses = source_statuses()
    for status in statuses:
        marker = "available" if status.available else "missing"
        print(
            f"source[{status.name}]: {marker} ({display_path(status.path)}; {status.detail})",
            file=sys.stderr,
        )
    missing = [status for status in statuses if not status.available]
    if strict and missing:
        detail = "; ".join(f"{status.name}: {status.detail}" for status in missing)
        raise SystemExit(f"--strict requires all source artifacts; {detail}")


def configure_matplotlib() -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle) if row.get(key)}


def load_context() -> RunContext:
    manifest = read_json(RUN_ROOT / "dataset_manifest.json")
    quality = read_json(RUN_ROOT / "quality_report.json")
    records = read_csv_by_key(RUN_ROOT / "records.csv", "record_id")
    scenarios = read_csv_by_key(RUN_ROOT / "scenario_manifest.csv", "scenario_group_id")
    available = bool(
        RUN_ROOT.exists()
        and (RUN_ROOT / "raw_complex_iq").exists()
        and manifest
        and records
        and np is not None
    )
    return RunContext(available, manifest, quality, records, scenarios)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | float | int | None) -> float:
    try:
        return float(value) if value is not None and value != "" else float("nan")
    except Exception:
        return float("nan")


def roc_auc_from_scores(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return float("nan")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        tied_score = pairs[index][0]
        end = index
        while end < len(pairs) and pairs[end][0] == tied_score:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        for _, label in pairs[index:end]:
            if label:
                positive_rank_sum += avg_rank
        rank += end - index
        index = end
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_from_scores(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return float("nan")
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def classification_metrics(
    labels: list[int], predictions: list[int], scores: list[float]
) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    positives = tp + fn
    negatives = fp + tn
    return {
        "record_count": float(total),
        "positive_count": float(positives),
        "negative_count": float(negatives),
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / positives if positives else 0.0,
        "specificity": tn / negatives if negatives else 0.0,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "roc_auc": roc_auc_from_scores(labels, scores),
        "average_precision": average_precision_from_scores(labels, scores),
    }


def read_metric_table(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv_rows(path)
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        method = row.get("method")
        split_role = row.get("split_role")
        phase_id = row.get("phase_id")
        if method and split_role and phase_id:
            table[(method, split_role, phase_id)] = row
    return table


def lookup_metric(
    table: dict[tuple[str, str, str], dict[str, str]],
    method: str,
    split_role: str = "holdout",
    phase_id: str = "all",
) -> dict[str, float] | None:
    row = table.get((method, split_role, phase_id))
    if not row:
        return None
    return {
        "average_precision": _safe_float(row.get("average_precision")),
        "roc_auc": _safe_float(row.get("roc_auc")),
        "f1": _safe_float(row.get("f1")),
        "accuracy": _safe_float(row.get("accuracy")),
        "precision": _safe_float(row.get("precision")),
        "recall": _safe_float(row.get("recall")),
        "false_positive_rate": _safe_float(row.get("false_positive_rate")),
        "record_count": _safe_float(row.get("record_count")),
        "positive_count": _safe_float(row.get("positive_count")),
        "negative_count": _safe_float(row.get("negative_count")),
    }


def load_ranked_holdout_metrics() -> dict[str, dict[str, float]]:
    filename = "kpi_ranking.png"
    table = read_metric_table(BASELINE_METRICS_PATH)
    metrics = {method: data for method, data in DEFAULT_HOLDOUT_KPI.items()}
    if table:
        note_source(filename, "baseline metrics CSV")
    else:
        note_fallback(filename, "baseline metrics CSV missing; using deterministic KPI defaults")
    for method in BASELINE_METHOD_ORDER:
        row = lookup_metric(table, method, "holdout", "all")
        if row:
            metrics[method] = row
        else:
            note_fallback(filename, f"missing {method} holdout/all row")
    return metrics


def load_phase_holdout_metrics() -> dict[str, dict[str, dict[str, float]]]:
    filename = "phase_kpi.png"
    table = read_metric_table(BASELINE_METRICS_PATH)
    metrics = {
        method: {phase: data for phase, data in phases.items()}
        for method, phases in DEFAULT_PHASE_KPI.items()
    }
    if table:
        note_source(filename, "baseline phase metrics CSV")
    else:
        note_fallback(filename, "baseline metrics CSV missing; using deterministic phase defaults")
    for method in ("layered_fusion_c2",):
        for phase in PHASES:
            row = lookup_metric(table, method, "holdout", phase)
            if row:
                metrics.setdefault(method, {})[phase] = row
            else:
                note_fallback(filename, f"missing {method} holdout/{phase} row")
    return metrics


def load_ei_metrics() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    for filename in ("kpi_ranking.png", "phase_kpi.png"):
        note_source(filename, "advanced predictions or quality report")
    all_phase = dict(DEFAULT_HOLDOUT_KPI[EI_METHOD_ID])
    phase_metrics = {
        phase: dict(values) for phase, values in DEFAULT_PHASE_KPI[EI_METHOD_ID].items()
    }

    predictions = read_csv_rows(EI_PREDICTIONS_PATH)
    if predictions:
        holdout = [row for row in predictions if row.get("split_role") == "holdout"]
        if holdout:
            labels = [int(row.get("label_id", "0")) for row in holdout]
            scores = [_safe_float(row.get("advanced_score")) for row in holdout]
            preds = [int(row.get("binary_prediction", "0")) for row in holdout]
            all_phase.update(classification_metrics(labels, preds, scores))
            for phase in PHASES:
                subset = [row for row in holdout if row.get("phase_id") == phase]
                if subset:
                    labels = [int(row.get("label_id", "0")) for row in subset]
                    scores = [_safe_float(row.get("advanced_score")) for row in subset]
                    preds = [int(row.get("binary_prediction", "0")) for row in subset]
                    phase_metrics[phase] = classification_metrics(labels, preds, scores)
                else:
                    note_fallback(
                        "phase_kpi.png",
                        f"missing advanced-candidate holdout predictions for {phase}",
                    )
    else:
        quality = read_json(EI_QUALITY_PATH)
        note_fallback(
            "phase_kpi.png",
            "advanced predictions missing; using deterministic locked-candidate phase defaults",
        )
        holdout = (
            quality.get("selected_holdout_threshold_metrics", {})
            if isinstance(quality, dict)
            else {}
        )
        if isinstance(holdout, dict):
            for key in ("accuracy", "precision", "recall", "false_positive_rate", "f1", "roc_auc"):
                if key in holdout:
                    all_phase[key] = _safe_float(holdout.get(key))
            all_phase["average_precision"] = (
                _safe_float(
                    quality.get("promotion_gate", {}).get("selected_holdout_average_precision")
                )
                if isinstance(quality.get("promotion_gate"), dict)
                else all_phase["average_precision"]
            )
        if not quality:
            note_fallback(
                "kpi_ranking.png",
                "advanced quality report missing; using deterministic locked-candidate defaults",
            )
    return all_phase, phase_metrics


def save_figure(fig: Any, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    source_note = (
        "; ".join(FIGURE_SOURCE_NOTES.get(filename, [])) or "deterministic tracked paper renderer"
    )
    fallback_note = "; ".join(FIGURE_FALLBACK_NOTES.get(filename, [])) or "none"
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={
            "Software": "EchoForge paper/generate_figures.py",
            "Description": f"sources={source_note}; fallback={fallback_note}",
        },
    )
    plt.close(fig)
    return path


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0xFFFFFFFF


def normalize01(values: Any) -> Any:
    if np is None:
        return values
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(arr, (2, 98))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(np.min(arr)), float(np.max(arr))
    if high <= low:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def synthetic_heatmap(group: str, phase: str, positive: bool) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for deterministic fallback heatmaps")
    phase_index = PHASES.index(phase)
    rng = np.random.default_rng(stable_seed(group, phase, "positive" if positive else "negative"))
    pulses, bins = 24, 20
    y = np.linspace(0.0, 1.0, pulses)[:, None]
    x = np.linspace(0.0, 1.0, bins)[None, :]
    texture = 0.22 + 0.05 * np.sin(2.0 * math.pi * (x * 2.1 + y * 0.7))
    texture += rng.normal(0.0, 0.025, size=(pulses, bins))

    if positive:
        center = 0.30 + 0.18 * phase_index + 0.04 * ((stable_seed(group) % 5) - 2)
        center = float(np.clip(center, 0.18, 0.82))
        walk = center + 0.055 * np.sin(2.0 * math.pi * (y * 1.1 + 0.13 * phase_index))
        track = np.exp(-((x - walk) ** 2) / (2.0 * 0.055**2))
        modulation = 0.60 + 0.35 * np.sin(2.0 * math.pi * (y * 3.0 + 0.17 * phase_index)) ** 2
        glint = np.exp(-((x - (center + 0.13)) ** 2) / 0.010) * np.exp(-((y - 0.62) ** 2) / 0.055)
        arr = texture + 0.82 * track * modulation + 0.28 * glint
    else:
        broad_center = 0.26 + 0.22 * ((stable_seed(group, "family") % 4) / 3.0)
        broad = np.exp(-((x - broad_center) ** 2) / (2.0 * 0.16**2))
        vertical = 0.18 * np.sin(2.0 * math.pi * (y * (phase_index + 1.5))) ** 2
        smear = 0.28 * broad * (0.55 + vertical)
        speckle = 0.10 * rng.random((pulses, bins))
        blocker = 0.16 * np.exp(-((y - 0.26 - 0.18 * phase_index) ** 2) / 0.018)
        arr = texture + smear + speckle + blocker

    return normalize01(arr)


def parse_iq_ref(ref: str) -> tuple[Path, int] | None:
    match = IQ_REF_RE.match(ref)
    if not match:
        return None
    return RUN_ROOT / match.group("path"), int(match.group("row"))


def real_heatmap(context: RunContext, group: str, phase: str) -> tuple[Any | None, str]:
    if np is None:
        return None, "numpy unavailable"
    if not context.available:
        return None, "training run context unavailable"
    record = context.records.get(f"{group}_{phase}")
    if not record:
        return None, "record missing"
    parsed = parse_iq_ref(record.get("raw_complex_iq_ref", ""))
    if parsed is None:
        return None, "raw_complex_iq_ref missing or malformed"
    shard_path, row_index = parsed
    if not shard_path.exists():
        return None, f"missing shard {shard_path.relative_to(REPO_ROOT)}"
    try:
        with np.load(shard_path, allow_pickle=False) as shard:
            iq = shard["iq"]
            if row_index < 0 or row_index >= iq.shape[0]:
                return None, "raw IQ row out of bounds"
            sample = np.asarray(iq[row_index])
    except Exception as exc:
        return None, f"failed to read IQ shard: {exc}"

    magnitude = np.abs(sample)
    if magnitude.ndim == 3:
        magnitude = magnitude.mean(axis=0)
    elif magnitude.ndim > 3:
        magnitude = magnitude.reshape((-1,) + magnitude.shape[-2:]).mean(axis=0)
    return normalize01(np.log1p(magnitude)), "full-run output"


def sample_heatmap(
    context: RunContext, group: str, phase: str, positive: bool, filename: str
) -> tuple[Any, str]:
    real, reason = real_heatmap(context, group, phase)
    if real is not None:
        note_source(filename, "full-run IQ samples")
        return real, "full-run output"
    note_fallback(filename, f"{group}/{phase}: {reason}")
    require_no_fallback(filename)
    return synthetic_heatmap(group, phase, positive), "deterministic fallback"


def scenario_label(context: RunContext, group: str) -> str:
    row = context.scenarios.get(group, {})
    if not row:
        return group
    if row.get("is_positive") == "1":
        return "public-proxy positive"
    role = (
        row.get("hard_negative_role")
        or row.get("confuser_family")
        or row.get("target_role")
        or "negative"
    )
    return f"hard negative: {role.replace('_', ' ')}"


def scenario_metadata_lines(context: RunContext, group: str) -> list[str]:
    row = context.scenarios.get(group, {})
    if not row:
        return [group]
    lines = [group, scenario_label(context, group)]
    site = str(row.get("site_archetype_id") or "site unknown").replace("_", " ")
    noise = str(row.get("noise_regime") or "noise unknown").replace("_", " ")
    range_band = str(row.get("range_band") or "range unknown").replace("_", " ")
    aspect = str(row.get("target_aspect") or "aspect unknown").replace("_", " ")
    lines.append(f"site: {site}")
    lines.append(f"noise: {noise}")
    lines.append(f"range/aspect: {range_band} / {aspect}")
    if row.get("is_positive") == "1":
        lines.append("take-up stress: ground coupling")
    else:
        role = row.get("hard_negative_role") or row.get("confuser_family") or "confuser"
        lines.append(f"stressor: {role.replace('_', ' ')}")
    return lines


def add_box(
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
    size: float = 8.5,
) -> None:
    rect = Rectangle((x, y), w, h, linewidth=1.1, edgecolor=edge, facecolor=face)
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


def add_arrow(
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


def setup_diagram(width: float = 7.1, height: float = 4.2) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def figure_architecture_stack(_context: RunContext) -> Path:
    filename = "architecture_stack.png"
    note_source(filename, "tracked paper renderer and repository policy")
    fig, ax = setup_diagram(7.2, 4.5)
    ax.text(0.15, 5.7, "EchoForge paper figure architecture", fontsize=12, weight="bold")
    ax.text(
        0.15,
        5.35,
        "Strict-open, generated, public-proxy artifacts for reproducible sensing evidence",
        fontsize=8.5,
        color=MUTED,
    )

    layers = [
        ("Public-proxy object packs\nand scenario strata", PALE_BLUE, BLUE),
        ("Monte Carlo scenario\nand phase scheduler", PALE_TEAL, TEAL),
        ("GPU-native radar/IQ\nartifact synthesis", PALE_GOLD, GOLD),
        ("Detector views and\nfeature contracts", PALE_GREEN, GREEN),
        ("ML split, calibration,\nand uncertainty scoring", PALE_RED, RED),
        ("Evidence bundle:\nmanifests, reports, PNG figures", SLATE, INK),
    ]
    x, w, h = 0.55, 5.8, 0.58
    y0 = 4.55
    for idx, (text, face, edge) in enumerate(layers):
        y = y0 - idx * 0.72
        add_box(ax, x, y, w, h, text, face=face, edge=edge, weight="bold" if idx == 0 else "normal")
        if idx < len(layers) - 1:
            add_arrow(ax, (x + w / 2, y - 0.02), (x + w / 2, y - 0.16), color=edge)

    rails = [
        ("Claim boundary\npublic-proxy only", PALE_RED, RED),
        ("Deterministic seeds\nand sample IDs", PALE_BLUE, BLUE),
        ("Group split locks\nprevent leakage", PALE_TEAL, TEAL),
        ("Figure export only\nno raw arrays", PALE_GOLD, GOLD),
    ]
    ax.text(7.0, 4.98, "Governance rails", fontsize=9.5, weight="bold")
    for idx, (text, face, edge) in enumerate(rails):
        add_box(ax, 6.95, 4.25 - idx * 0.92, 2.55, 0.62, text, face=face, edge=edge, size=8.0)
        add_arrow(ax, (6.35, 4.84 - idx * 0.72), (6.95, 4.56 - idx * 0.92), color=MUTED)

    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def split_stats(context: RunContext) -> tuple[int, int, dict[str, int]]:
    scenarios = int(context.manifest.get("scenario_groups") or len(context.scenarios) or 0)
    records = int(context.manifest.get("record_count") or len(context.records) or 0)
    split_counts: dict[str, int] = {}
    for row in context.scenarios.values():
        role = row.get("split_role", "unknown")
        split_counts[role] = split_counts.get(role, 0) + 1
    if not split_counts:
        raw_counts = context.quality.get("split_counts_by_group", {})
        if isinstance(raw_counts, dict):
            split_counts = {str(k): int(v) for k, v in raw_counts.items()}
    return scenarios, records, split_counts


def figure_monte_carlo_split_flow(context: RunContext) -> Path:
    filename = "monte_carlo_split_flow.png"
    fig, ax = setup_diagram(7.2, 4.2)
    scenarios, records, split_counts = split_stats(context)
    phase_count = len(context.manifest.get("phase_ids", PHASES))
    positives = int(context.manifest.get("positive_groups") or 0)
    if context.available:
        note_source(filename, "training run manifest and quality report")
    else:
        note_fallback(filename, "training run unavailable; using layout defaults")
    require_no_fallback(filename)

    ax.text(0.1, 5.65, "Monte Carlo group split flow", fontsize=12, weight="bold")
    subtitle = "Scenario groups stay phase-locked before train/CV/test assignment"
    ax.text(0.1, 5.32, subtitle, fontsize=8.5, color=MUTED)

    boxes = [
        ("Strata grid\nsite x range x aspect x noise", PALE_BLUE, BLUE),
        (f"Scenario groups\nN={scenarios:,} groups", PALE_TEAL, TEAL),
        (f"Phase expansion\n{phase_count} phases per group", PALE_GOLD, GOLD),
        (f"Record table\nN={records:,} rows", PALE_GREEN, GREEN),
        ("Locked split keys\nfolds and holdout", PALE_RED, RED),
    ]
    x0, y, w, h, gap = 0.35, 3.75, 1.55, 0.82, 0.36
    for idx, (text, face, edge) in enumerate(boxes):
        x = x0 + idx * (w + gap)
        add_box(ax, x, y, w, h, text, face=face, edge=edge, size=7.6)
        if idx < len(boxes) - 1:
            add_arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2), color=edge)

    ax.text(0.45, 2.72, "Group-level split counts", fontsize=9.5, weight="bold")
    if split_counts:
        total = max(1, sum(split_counts.values()))
        colors = [BLUE, TEAL, GOLD, RED, GREEN]
        x = 0.45
        bar_y = 2.23
        bar_w = 4.25
        start = x
        for idx, (name, count) in enumerate(sorted(split_counts.items())):
            width = bar_w * count / total
            rect = Rectangle(
                (start, bar_y), width, 0.35, facecolor=colors[idx % len(colors)], edgecolor="white"
            )
            ax.add_patch(rect)
            ax.text(
                start + width / 2,
                bar_y - 0.18,
                f"{name}\n{count:,}",
                ha="center",
                va="top",
                fontsize=7.0,
            )
            start += width
        ax.add_patch(
            Rectangle((x, bar_y), bar_w, 0.35, facecolor="none", edgecolor=INK, linewidth=0.8)
        )
    else:
        add_box(
            ax,
            0.45,
            2.05,
            4.25,
            0.65,
            "Split counts read from run output when available",
            face=SLATE,
        )

    checks = [
        f"Positive public-proxy groups: {positives:,}"
        if positives
        else "Positive groups read from manifest",
        "Phase rows inherit the same split key",
        "Restricted truth metadata is denylisted",
        "Quality report records leakage guard status",
    ]
    ax.text(5.45, 2.72, "Leakage guards", fontsize=9.5, weight="bold")
    for idx, text in enumerate(checks):
        add_box(ax, 5.45, 2.35 - idx * 0.48, 3.95, 0.34, text, face="white", edge=GRID, size=7.4)

    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def figure_kpi_ranking(_context: RunContext) -> Path:
    filename = "kpi_ranking.png"
    metrics = load_ranked_holdout_metrics()
    ei_all, _ei_phases = load_ei_metrics()
    metrics[EI_METHOD_ID] = ei_all
    require_no_fallback(filename)
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        "Holdout KPI ranking: baselines and locked candidate", fontsize=12.5, weight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.90,
        "Average precision is the primary bar; the tag shows ROC AUC and F1 on the blind holdout.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=MUTED,
    )

    colors = {
        "high_resolution_xku_cuas": BLUE,
        "tactical_s_band_aesa": TEAL,
        "gbad_3d4d_cueing": GOLD,
        "distributed_acoustic_cue": RED,
        "tabular_ml_baseline": "#62758a",
        "sequence_ml_proxy": "#8899ab",
        "layered_fusion_c2": GREEN,
        EI_METHOD_ID: INK,
    }
    edgecolors = {
        EI_METHOD_ID: GOLD,
        "layered_fusion_c2": GREEN,
    }

    y_positions = list(range(len(ALL_KPI_METHOD_ORDER)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([KPI_LABELS[method] for method in ALL_KPI_METHOD_ORDER], fontsize=8.4)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Holdout average precision", fontsize=9.0)
    ax.set_ylim(-0.8, len(y_positions) - 0.2)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    for idx, method in enumerate(ALL_KPI_METHOD_ORDER):
        data = metrics.get(method, DEFAULT_HOLDOUT_KPI[method])
        ap = float(data["average_precision"])
        auc = float(data["roc_auc"])
        f1 = float(data["f1"])
        color = colors[method]
        edge = edgecolors.get(method, "white")
        bar = ax.barh(idx, ap, height=0.62, color=color, edgecolor=edge, linewidth=1.0)
        for patch in bar:
            if method == EI_METHOD_ID:
                patch.set_hatch("///")
        tag = f"AP {ap:.3f}   AUC {auc:.3f}   F1 {f1:.3f}"
        ax.text(
            min(ap + 0.015, 0.995),
            idx,
            tag,
            ha="left",
            va="center",
            fontsize=7.6,
            color=INK,
            weight="bold" if method == EI_METHOD_ID else "normal",
        )

    ax.axhline(3.5, color=GRID, linewidth=0.9)
    ax.axhline(5.5, color=GRID, linewidth=0.9)
    ax.axhline(6.5, color=GRID, linewidth=1.0)
    ax.text(0.003, -0.47, "Sensor branches", fontsize=8.1, color=MUTED, weight="bold")
    ax.text(0.003, 3.53, "Prior ML", fontsize=8.1, color=MUTED, weight="bold")
    ax.text(0.003, 5.53, "Prior fusion", fontsize=8.1, color=MUTED, weight="bold")
    ax.text(0.003, 6.53, "Locked candidate", fontsize=8.1, color=MUTED, weight="bold")
    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def figure_phase_kpi(_context: RunContext) -> Path:
    filename = "phase_kpi.png"
    baseline = load_phase_holdout_metrics().get(
        "layered_fusion_c2", DEFAULT_PHASE_KPI["layered_fusion_c2"]
    )
    _ei_all, ei_phases = load_ei_metrics()
    require_no_fallback(filename)
    phase_labels = [f"{PHASE_LABELS[phase]}\n{PHASE_WINDOWS[phase]}" for phase in PHASES]
    x = np.arange(len(PHASES))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True)
    fig.suptitle(
        "Phase KPI comparison: holdout behavior from take-up to cruise",
        fontsize=12,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.02,
        "Holdout rows only; train/CV locks are fixed before locked-candidate selection and scoring.",
        fontsize=7.2,
        color=MUTED,
    )

    panel_specs = [
        ("average_precision", "Average precision", "AP"),
        ("roc_auc", "ROC AUC", "AUC"),
    ]
    for ax, (metric_key, ylabel, _short_label) in zip(axes, panel_specs):
        baseline_values = [
            float(baseline.get(phase, DEFAULT_PHASE_KPI["layered_fusion_c2"][phase])[metric_key])
            for phase in PHASES
        ]
        ei_values = [
            float(ei_phases.get(phase, DEFAULT_PHASE_KPI[EI_METHOD_ID][phase])[metric_key])
            for phase in PHASES
        ]
        baseline_bars = ax.bar(
            x - width / 2,
            baseline_values,
            width,
            label="layered_fusion_c2",
            color=PALE_GREEN,
            edgecolor=GREEN,
            linewidth=1.0,
        )
        ei_bars = ax.bar(
            x + width / 2,
            ei_values,
            width,
            label="locked candidate",
            color=INK,
            edgecolor=GOLD,
            linewidth=1.0,
            hatch="///",
        )
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=7.6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        for bar in list(baseline_bars) + list(ei_bars):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.018, 1.0),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=INK,
            )
        ax.legend(loc="upper left", fontsize=7.6, frameon=False)

    axes[0].set_title("AP comparison by phase", fontsize=9.6, weight="bold")
    axes[1].set_title("ROC AUC comparison by phase", fontsize=9.6, weight="bold")
    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def plot_iq_grid(
    context: RunContext, groups: tuple[str, str], positive: bool, filename: str, title: str
) -> Path:
    fig = plt.figure(figsize=(10.3, 5.1))
    grid = fig.add_gridspec(
        len(groups),
        len(PHASES) + 1,
        width_ratios=[1.45, 1.0, 1.0, 1.0],
        hspace=0.28,
        wspace=0.08,
    )
    source_labels: set[str] = set()
    heatmap_axes: list[Any] = []
    image = None

    for row_index, group in enumerate(groups):
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.set_facecolor(PALE_BLUE if positive else PALE_RED)
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_color(BLUE if positive else RED)
            spine.set_linewidth(1.1)
        label_ax.text(
            0.06,
            0.94,
            "\n".join(scenario_metadata_lines(context, group)),
            ha="left",
            va="top",
            fontsize=7.8,
            color=INK,
            linespacing=1.18,
            transform=label_ax.transAxes,
        )
        if positive:
            label_ax.text(
                0.06,
                0.12,
                "Take-up: closest-to-ground coupling",
                ha="left",
                va="bottom",
                fontsize=7.1,
                color=MUTED,
                transform=label_ax.transAxes,
                wrap=True,
            )
        else:
            label_ax.text(
                0.06,
                0.12,
                "Hard-negative cueing and artifact stressor",
                ha="left",
                va="bottom",
                fontsize=7.1,
                color=MUTED,
                transform=label_ax.transAxes,
                wrap=True,
            )

        for col_index, phase in enumerate(PHASES, start=1):
            ax = fig.add_subplot(grid[row_index, col_index])
            heatmap_axes.append(ax)
            heatmap, source = sample_heatmap(context, group, phase, positive, filename)
            source_labels.add(source)
            image = ax.imshow(
                heatmap, aspect="auto", origin="lower", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0
            )
            if row_index == 0:
                ax.set_title(f"{PHASE_LABELS[phase]}\n{PHASE_WINDOWS[phase]}", fontsize=8.5, pad=6)
            ax.set_xticks([0, 9, 19])
            ax.set_xticklabels(["near", "mid", "far"], fontsize=6.9)
            ax.set_yticks([0, 11, 23])
            ax.set_yticklabels(["0", "12", "24"], fontsize=6.9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color(GRID)
            ax.grid(color="white", linewidth=0.25, alpha=0.32)

    fig.suptitle(title, fontsize=12.2, weight="bold", y=0.98)
    source_note = " + ".join(sorted(source_labels))
    fig.text(
        0.02,
        0.015,
        f"Normalized IQ magnitude on a shared 0-1 scale; source: {source_note}. Public-proxy generated artifacts, not measured truth.",
        fontsize=7.0,
        color=MUTED,
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=heatmap_axes, shrink=0.82, pad=0.016)
        cbar.set_label("normalized magnitude", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7.0)

    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def figure_iq_drone_samples(context: RunContext) -> Path:
    return plot_iq_grid(
        context,
        DRONE_GROUPS,
        True,
        "iq_drone_samples.png",
        "Deterministic IQ samples: public-proxy positive groups",
    )


def figure_iq_negative_samples(context: RunContext) -> Path:
    return plot_iq_grid(
        context,
        NEGATIVE_GROUPS,
        False,
        "iq_negative_samples.png",
        "Deterministic IQ samples: confusers and sensor artifacts",
    )


def figure_detector_ml_pipeline(context: RunContext) -> Path:
    filename = "detector_ml_pipeline.png"
    fig, ax = setup_diagram(7.2, 4.45)
    if context.available:
        note_source(filename, "training run detector-view manifest")
    else:
        note_fallback(filename, "training run unavailable; using canonical detector-view labels")
    require_no_fallback(filename)
    ax.text(0.15, 5.7, "Detector and ML evidence pipeline", fontsize=12, weight="bold")
    ax.text(
        0.15,
        5.38,
        "Detector-facing artifacts are split-aware and uncertainty-scored before paper evidence is emitted",
        fontsize=8.5,
        color=MUTED,
    )

    view_names = context.manifest.get("detector_views", [])
    if not isinstance(view_names, list) or not view_names:
        view_names = [
            "high_resolution_xku_cuas",
            "tactical_s_band_aesa",
            "gbad_3d4d_cueing",
            "distributed_acoustic_cue",
            "layered_fusion_c2",
        ]

    add_box(ax, 0.35, 4.25, 1.65, 0.68, "Raw generated\nsensor streams", face=PALE_BLUE, edge=BLUE)
    add_box(ax, 2.45, 4.25, 1.65, 0.68, "Detector view\nmaterialization", face=PALE_TEAL, edge=TEAL)
    add_box(ax, 4.55, 4.25, 1.65, 0.68, "Feature tables\nand denylist", face=PALE_GOLD, edge=GOLD)
    add_box(ax, 6.65, 4.25, 1.65, 0.68, "Split-aware ML\ntraining", face=PALE_GREEN, edge=GREEN)
    add_arrow(ax, (2.0, 4.59), (2.45, 4.59), BLUE)
    add_arrow(ax, (4.1, 4.59), (4.55, 4.59), TEAL)
    add_arrow(ax, (6.2, 4.59), (6.65, 4.59), GOLD)

    ax.text(0.4, 3.35, "Detector views", fontsize=9.5, weight="bold")
    for idx, name in enumerate(view_names[:5]):
        y = 2.88 - idx * 0.42
        clean = str(name).replace("_", " ")
        add_box(ax, 0.35, y, 2.5, 0.28, clean, face="white", edge=GRID, size=7.0)
        add_arrow(ax, (2.85, y + 0.14), (3.35, 3.0), color=GRID)

    add_box(ax, 3.35, 2.55, 2.1, 0.72, "Calibration and\nthreshold selection", face=SLATE, edge=INK)
    add_box(ax, 6.05, 2.85, 2.1, 0.72, "Uncertainty bands\nand abstention", face=PALE_RED, edge=RED)
    add_box(
        ax, 6.05, 1.82, 2.1, 0.72, "Failure slices\nand hard negatives", face=PALE_GOLD, edge=GOLD
    )
    add_arrow(ax, (5.45, 2.91), (6.05, 3.21), color=INK)
    add_arrow(ax, (5.45, 2.91), (6.05, 2.18), color=INK)

    add_box(
        ax,
        3.35,
        1.35,
        4.8,
        0.55,
        "Evidence outputs: quality report, split audit, paper PNGs",
        face=PALE_GREEN,
        edge=GREEN,
    )
    add_arrow(ax, (7.1, 2.85), (7.1, 1.9), color=RED)
    add_arrow(ax, (7.1, 1.82), (7.1, 1.9), color=GOLD)

    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def figure_locked_algorithm(context: RunContext) -> Path:
    filename = "locked_algorithm.png"
    fig, ax = setup_diagram(7.1, 4.2)
    if context.available:
        note_source(filename, "training run manifest and leakage status")
    else:
        note_fallback(filename, "training run unavailable; using fixed lock labels")
    require_no_fallback(filename)
    ax.text(0.15, 5.65, "Locked candidate evidence path", fontsize=12, weight="bold")
    ax.text(
        0.15,
        5.32,
        "Eight components, nonnegative weights, geodesic-odds calibration, and one-shot holdout scoring",
        fontsize=8.5,
        color=MUTED,
    )

    seed = context.manifest.get("seed", "fixed")
    status = context.quality.get("leakage_guard_status", "checked when run output exists")
    outputs_generated = context.manifest.get("outputs_are_generated", "true")

    center_x, center_y = 5.0, 3.25
    add_box(
        ax,
        center_x - 1.1,
        center_y - 0.42,
        2.2,
        0.84,
        "LOCKED\nCANDIDATE",
        face=INK,
        edge=INK,
        color="white",
        weight="bold",
    )

    inputs = [
        (0.45, 4.25, "Manifest seed\nand config lock", f"seed={seed}"),
        (0.45, 3.05, "Sample IDs\nand phase order", "sg_00016, sg_00030,\nsg_00000, sg_00001"),
        (0.45, 1.85, "Strict-open\nclaim boundary", f"generated={outputs_generated}"),
    ]
    colors = [(PALE_BLUE, BLUE), (PALE_TEAL, TEAL), (PALE_RED, RED)]
    for idx, (x, y, label, detail) in enumerate(inputs):
        face, edge = colors[idx]
        add_box(ax, x, y, 2.45, 0.58, label, face=face, edge=edge, weight="bold", size=7.8)
        ax.text(x + 1.225, y - 0.16, detail, ha="center", va="top", fontsize=6.8, color=MUTED)
        add_arrow(ax, (x + 2.45, y + 0.29), (center_x - 1.1, center_y), color=edge)

    outputs = [
        (7.05, 4.25, "Feature denylist\nblocks truth leaks", f"leakage={status}"),
        (7.05, 3.05, "Deterministic\nPNG renderer", "matplotlib Agg\nfixed sample set"),
        (7.05, 1.85, "Reviewable paper\nevidence only", "no raw arrays\nwritten"),
    ]
    for idx, (x, y, label, detail) in enumerate(outputs):
        face, edge = colors[idx]
        add_box(ax, x, y, 2.45, 0.58, label, face=face, edge=edge, weight="bold", size=7.8)
        ax.text(x + 1.225, y - 0.16, detail, ha="center", va="top", fontsize=6.8, color=MUTED)
        add_arrow(ax, (center_x + 1.1, center_y), (x, y + 0.29), color=edge)

    add_box(
        ax,
        3.35,
        1.1,
        3.3,
        0.55,
        "Same inputs produce the same figure set;\nmissing run output uses deterministic fallback heatmaps",
        face=SLATE,
        edge=INK,
        size=7.5,
    )
    add_arrow(ax, (center_x, center_y - 0.42), (center_x, 1.65), color=INK)

    add_fallback_banner(fig, filename)
    return save_figure(fig, filename)


def generate_all() -> list[Path]:
    configure_matplotlib()
    context = load_context()
    return [
        figure_kpi_ranking(context),
        figure_phase_kpi(context),
        figure_architecture_stack(context),
        figure_monte_carlo_split_flow(context),
        figure_iq_drone_samples(context),
        figure_iq_negative_samples(context),
        figure_detector_ml_pipeline(context),
        figure_locked_algorithm(context),
    ]


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper.generate_figures_major_upgrade_v2 import main as major_upgrade_main

    return major_upgrade_main()


if __name__ == "__main__":
    raise SystemExit(main())
