#!/usr/bin/env python3
"""Generate the focused final EchoForge paper figure set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to generate paper figures") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run"
)
DEFAULT_BASELINE_ROOT = REPO_ROOT / "outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run"
DEFAULT_ADVANCED_ROOT = (
    REPO_ROOT / "outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution"
)
DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "outputs/paper-evidence/tier1-final"
DEFAULT_FIGURE_DIR = REPO_ROOT / "paper/figures"
PDF_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)

PHASES = (
    ("initial_take_up", "Take-off", "0--30 s"),
    ("climb_transition", "Climb", "30--90 s"),
    ("cruise_altitude", "Cruise", "90--150 s"),
)

METHOD_ORDER = (
    "high_resolution_xku_cuas",
    "tactical_s_band_aesa",
    "gbad_3d4d_cueing",
    "distributed_acoustic_cue",
    "tabular_ml_baseline",
    "sequence_ml_proxy",
    "layered_fusion_c2",
    "locked_candidate",
)
METHOD_LABELS = {
    "high_resolution_xku_cuas": "X/Ku",
    "tactical_s_band_aesa": "S-band",
    "gbad_3d4d_cueing": "GBAD",
    "distributed_acoustic_cue": "Acoustic",
    "tabular_ml_baseline": "Tabular ML",
    "sequence_ml_proxy": "Seq. ML",
    "layered_fusion_c2": "Accepted fusion",
    "locked_candidate": "EI",
}

FAMILY_ORDER = (
    "single_bird",
    "bird_flock",
    "rc_fixed_wing",
    "weather_cell",
    "clutter_only_counterfactual",
    "rfi_burst",
    "multipath_ghost",
    "terrain_glint",
    "ground_vehicle",
    "wind_turbine",
)
FAMILY_LABELS = {
    "single_bird": "single bird",
    "bird_flock": "bird flock",
    "rc_fixed_wing": "RC fixed-wing",
    "weather_cell": "weather",
    "clutter_only_counterfactual": "clutter-only",
    "rfi_burst": "RFI",
    "multipath_ghost": "multipath",
    "terrain_glint": "terrain/glint",
    "ground_vehicle": "ground vehicle",
    "wind_turbine": "wind turbine",
}

INK = "#1e293b"
MUTED = "#475569"
GRID = "#cbd5e1"
BLUE = "#2563eb"
TEAL = "#0f766e"
GOLD = "#a16207"
GREEN = "#166534"
RED = "#b91c1c"
PURPLE = "#6d28d9"
SLATE = "#64748b"
PALE_BLUE = "#eff6ff"
PALE_GREEN = "#f0fdf4"
PALE_GOLD = "#fefce8"
PALE_RED = "#fef2f2"
PALE_SLATE = "#f8fafc"

METHOD_COLORS = {
    "high_resolution_xku_cuas": BLUE,
    "tactical_s_band_aesa": TEAL,
    "gbad_3d4d_cueing": GOLD,
    "distributed_acoustic_cue": PURPLE,
    "tabular_ml_baseline": SLATE,
    "sequence_ml_proxy": "#94a3b8",
    "layered_fusion_c2": GREEN,
    "locked_candidate": INK,
}
FAMILY_COLORS = {
    "single_bird": "#2563eb",
    "bird_flock": "#60a5fa",
    "rc_fixed_wing": "#0f766e",
    "weather_cell": "#a16207",
    "clutter_only_counterfactual": "#64748b",
    "rfi_burst": "#b91c1c",
    "multipath_ghost": "#6d28d9",
    "terrain_glint": "#c2410c",
    "ground_vehicle": "#166534",
    "wind_turbine": "#334155",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "echoforge_rd",
    ("#111827", "#1d4ed8", "#0f766e", "#84cc16", "#fde047", "#f8fafc"),
    N=256,
)


@dataclass(frozen=True)
class Roots:
    training: Path
    baseline: Path
    advanced: Path
    evidence: Path
    figures: Path


@dataclass(frozen=True)
class Context:
    records: list[dict[str, str]]
    scenarios: dict[str, dict[str, str]]
    scores: dict[str, dict[str, float]]
    advanced_rows: dict[str, dict[str, str]]
    evidence: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean(ax: Any, *, xgrid: bool = False, ygrid: bool = False) -> None:
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.55)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _wrap(text: Any, width: int = 24) -> str:
    return "\n".join(
        textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False)
    )


def _box(
    ax: Any,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str,
    edge: str,
    size: float = 7.0,
    weight: str = "normal",
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(
        x + w / 2,
        y + h / 2,
        _wrap(text, 25),
        ha="center",
        va="center",
        fontsize=size,
        weight=weight,
        color=INK,
    )


def _arrow(
    ax: Any, start: tuple[float, float], end: tuple[float, float], *, color: str = MUTED
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
        )
    )


def _save(fig: Any, roots: Roots, filename: str, *, vector: bool = True) -> Path:
    roots.figures.mkdir(parents=True, exist_ok=True)
    path = roots.figures / filename
    metadata = {"Software": "EchoForge paper/generate_figures_focused.py"}
    if vector:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.06,
            metadata={
                "Creator": metadata["Software"],
                "CreationDate": PDF_TIMESTAMP,
                "ModDate": PDF_TIMESTAMP,
            },
        )
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.06, metadata=metadata)
    plt.close(fig)
    return path


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("record_id", ""): row for row in rows}


def _scenario_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("scenario_group_id", ""): row for row in rows}


def _load_context(roots: Roots, *, strict: bool) -> Context:
    records = _read_csv(roots.training / "records.csv")
    scenarios = _scenario_index(_read_csv(roots.training / "scenario_manifest.csv"))
    cfar = _index(_read_csv(roots.baseline / "classical_radar_processing.csv"))
    ml = _index(_read_csv(roots.baseline / "ml_detector_baselines.csv"))
    fusion = _index(_read_csv(roots.baseline / "fusion_predictions.csv"))
    advanced = _index(_read_csv(roots.advanced / "advanced_predictions.csv"))
    evidence = {
        "manifest": _read_json(roots.evidence / "paper_evidence_manifest.json"),
        "split_summary": _read_json(roots.evidence / "split_summary.json"),
        "radar_model_card": _read_json(roots.evidence / "radar_model_card.json"),
        "ei_evolution_summary": _read_json(roots.evidence / "ei_evolution_summary.json"),
    }
    missing: list[str] = []
    for label, rows in (
        ("records", records),
        ("scenario manifest", scenarios),
        ("classical radar processing", cfar),
        ("ML detector baselines", ml),
        ("fusion predictions", fusion),
        ("advanced predictions", advanced),
    ):
        if not rows:
            missing.append(label)
    if strict and missing:
        raise SystemExit("--strict requires source data: " + ", ".join(missing))
    if strict and evidence["manifest"].get("version") != "tier1-final":
        raise SystemExit("--strict requires tier1-final paper evidence")

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        record_id = record.get("record_id", "")
        cfar_row = cfar.get(record_id, {})
        ml_row = ml.get(record_id, {})
        fusion_row = fusion.get(record_id, {})
        advanced_row = advanced.get(record_id, {})
        scores["high_resolution_xku_cuas"][record_id] = _safe_float(
            cfar_row.get("high_res_cfar_log1p"), 0.0
        )
        scores["tactical_s_band_aesa"][record_id] = _safe_float(
            cfar_row.get("sband_mtd_log1p"), 0.0
        )
        scores["gbad_3d4d_cueing"][record_id] = _safe_float(cfar_row.get("gbad_track_log1p"), 0.0)
        scores["distributed_acoustic_cue"][record_id] = _safe_float(
            advanced_row.get("distributed_acoustic_cue"), 0.0
        )
        scores["tabular_ml_baseline"][record_id] = _safe_float(
            ml_row.get("tabular_baseline_score"), 0.0
        )
        scores["sequence_ml_proxy"][record_id] = _safe_float(
            ml_row.get("sequence_proxy_score"), 0.0
        )
        scores["layered_fusion_c2"][record_id] = _safe_float(
            fusion_row.get("fusion_probability"), 0.0
        )
        scores["locked_candidate"][record_id] = _safe_float(advanced_row.get("advanced_score"), 0.0)

    return Context(
        records=records,
        scenarios=scenarios,
        scores=dict(scores),
        advanced_rows=advanced,
        evidence=evidence,
    )


def _fixed_fpr_recall(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    tp = 0
    fp = 0
    best = 0.0
    idx = 0
    while idx < len(scores):
        threshold = scores[idx]
        while idx < len(scores) and scores[idx] == threshold:
            if labels[idx] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        if fp / negatives <= target_fpr:
            best = max(best, tp / positives)
    return best


def _phase_metric_grid(context: Context) -> dict[tuple[str, str], float]:
    grid: dict[tuple[str, str], float] = {}
    holdout = [row for row in context.records if row.get("split_role") == "holdout"]
    for phase, _label, _window in PHASES:
        phase_rows = [row for row in holdout if row.get("phase_id") == phase]
        labels = np.asarray([_safe_int(row.get("label_id")) for row in phase_rows], dtype=np.int8)
        for method in METHOD_ORDER:
            values = np.asarray(
                [
                    context.scores.get(method, {}).get(row.get("record_id", ""), 0.0)
                    for row in phase_rows
                ],
                dtype=np.float64,
            )
            grid[(method, phase)] = _fixed_fpr_recall(labels, values)
    return grid


def figure_architecture_stack(roots: Roots, context: Context) -> Path:
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.18, 5.72, "EchoForge simulator and evidence stack", fontsize=10, weight="bold")
    ax.text(
        0.18,
        5.35,
        "Strict-open public-proxy assumptions stay separated from synthetic sensing, detector views, and claims.",
        fontsize=7.2,
        color=MUTED,
    )
    layers = [
        ("Public-proxy object priors", PALE_BLUE, BLUE),
        ("Scenario groups and phase windows", PALE_SLATE, SLATE),
        ("Radar branch cards and cue streams", PALE_GOLD, GOLD),
        ("Synthetic range-Doppler products", PALE_GREEN, TEAL),
        ("Detector-view contract and leakage rails", PALE_RED, RED),
        ("Train/CV lock and blind holdout evidence", "white", GREEN),
    ]
    x, w, h = 0.32, 4.85, 0.50
    y0 = 4.75
    for idx, (label, face, edge) in enumerate(layers):
        y = y0 - idx * 0.67
        _box(
            ax,
            (x, y),
            (w, h),
            label,
            face=face,
            edge=edge,
            size=7.0,
            weight="bold" if idx == 0 else "normal",
        )
        if idx < len(layers) - 1:
            _arrow(ax, (x + w / 2, y - 0.01), (x + w / 2, y - 0.16), color=edge)

    card = context.evidence.get("radar_model_card", {})
    branches = card.get("carrier_bands", []) if isinstance(card, dict) else []
    branch_text = []
    for branch in branches[:3]:
        branch_text.append(
            f"{branch.get('branch', 'branch').replace('_', ' ')}: "
            f"{_safe_float(branch.get('carrier_ghz'), 0.0):.1f} GHz, "
            f"{_safe_float(branch.get('bandwidth_mhz'), 0.0):.0f} MHz"
        )
    if not branch_text:
        branch_text = [
            "X/Ku: 10.0 GHz, 600 MHz",
            "S-band: 3.1 GHz, 180 MHz",
            "GBAD: 10.3 GHz, 300 MHz",
        ]

    _box(
        ax,
        (5.85, 4.54),
        (3.55, 0.58),
        "Model card: readable radar assumptions",
        face=PALE_BLUE,
        edge=BLUE,
        size=7.2,
        weight="bold",
    )
    card_lines = [
        *branch_text,
        "CPI 6 ms, 24 chirps/pulses",
        "impairments: clutter, RFI, AGC, drift",
        "claim boundary: synthetic public-proxy evidence",
    ]
    for idx, line in enumerate(card_lines):
        _box(ax, (5.85, 3.84 - idx * 0.51), (3.55, 0.36), line, face="white", edge=GRID, size=6.3)
    return _save(fig, roots, "architecture_stack.png")


def figure_phase_method_ladder(roots: Roots, context: Context) -> Path:
    grid = _phase_metric_grid(context)
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.24)
    fig.text(
        0.08,
        0.94,
        "Phase method ladder: human detector branches -> accepted fusion -> EI",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.08,
        0.895,
        "Each phase uses the same method order and the same fixed-FPR recall operating cap.",
        fontsize=7.2,
        color=MUTED,
    )
    x = np.arange(len(PHASES))
    width = 0.085
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width
    for idx, method in enumerate(METHOD_ORDER):
        values = [grid.get((method, phase), 0.0) for phase, _label, _window in PHASES]
        bars = ax.bar(
            x + offsets[idx],
            values,
            width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            label=METHOD_LABELS[method],
        )
        if method in {"layered_fusion_c2", "locked_candidate"}:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.02, value + 0.03),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color=METHOD_COLORS[method],
                    rotation=90,
                )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Recall at <=1% FPR", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{window}" for _phase, label, window in PHASES], fontsize=7.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=6.2)
    _clean(ax, ygrid=True)
    ax.text(
        0.0,
        -0.30,
        "Branch values are diagnostic fixed-FPR holdout scores; the aggregate claim remains the locked all-phase KPI.",
        transform=ax.transAxes,
        fontsize=6.5,
        color=MUTED,
    )
    return _save(fig, roots, "phase_method_ladder.png")


def figure_ei_evolution_money_plot(roots: Roots, context: Context) -> Path:
    rows = _read_csv(roots.evidence / "ei_evolution_trace.csv")
    summary = context.evidence.get("ei_evolution_summary", {})
    if not rows:
        raise SystemExit("missing EI evolution trace")
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("candidate_index"), 0.0))
    x = np.asarray(
        [_safe_float(row.get("candidate_index"), idx + 1) for idx, row in enumerate(ordered)]
    )
    y = np.asarray([_safe_float(row.get("train_cv_objective"), 0.0) for row in ordered])
    best = np.asarray([_safe_float(row.get("running_best_objective"), 0.0) for row in ordered])
    stages = [row.get("stage", "unknown") for row in ordered]
    stage_colors = {
        "base_candidate_search": BLUE,
        "surface_control": GOLD,
        "meta_fusion_search": GREEN,
    }
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.18)
    fig.text(0.08, 0.94, "EI train/CV evolution money plot", fontsize=10, weight="bold")
    fig.text(
        0.08,
        0.895,
        "Candidate order and running best are train/CV evidence only; no holdout optimization curve is drawn.",
        fontsize=7.2,
        color=GREEN,
    )
    for stage in sorted(set(stages)):
        mask = np.asarray([item == stage for item in stages])
        ax.scatter(
            x[mask],
            y[mask],
            s=15,
            alpha=0.62,
            color=stage_colors.get(stage, SLATE),
            label=stage.replace("_", " "),
        )
    ax.plot(x, best, color=INK, linewidth=1.8, label="running best train/CV objective")
    selected_id = str(summary.get("selected_candidate_id", ""))
    selected_rows = [
        row
        for row in ordered
        if str(row.get("candidate_id")) == selected_id
        or str(row.get("selected_by_cv", "")).lower() == "true"
    ]
    if selected_rows:
        selected = selected_rows[-1]
        sx = _safe_float(selected.get("candidate_index"))
        sy = _safe_float(selected.get("train_cv_objective"))
        ax.scatter([sx], [sy], color=RED, marker="*", s=115, zorder=5, label="selected lock")
        ax.annotate(
            "selected lock",
            xy=(sx, sy),
            xytext=(0.70, 0.24),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": RED},
            fontsize=7,
            color=RED,
        )
    ax.set_xlabel("Candidate evaluation order", fontsize=7.5)
    ax.set_ylabel("Train/CV objective", fontsize=7.5)
    _clean(ax, xgrid=True, ygrid=True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.4)
    ax.text(
        0.01,
        0.03,
        "Guardrail: holdout is scored once after lock.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=RED,
        bbox={"facecolor": "white", "edgecolor": PALE_RED, "linewidth": 0.6, "pad": 2.0},
    )
    return _save(fig, roots, "ei_evolution_money_plot.png")


def figure_false_alarm_breakdown(roots: Roots, _context: Context) -> Path:
    rows = _read_csv(roots.evidence / "false_alarm_by_method_family.csv")
    if not rows:
        raise SystemExit("missing false_alarm_by_method_family.csv")
    totals: dict[str, dict[str, float]] = {method: defaultdict(float) for method in METHOD_ORDER}
    near_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        method = row.get("method", "")
        family = row.get("family", "")
        if method in totals and family in FAMILY_LABELS:
            totals[method][family] += _safe_float(row.get("false_alarm_count"), 0.0)
            near_totals[method] += _safe_float(row.get("near_threshold_count"), 0.0)
    methods = [
        method
        for method in METHOD_ORDER
        if sum(totals[method].values()) > 0 or method == "locked_candidate"
    ]
    fig, ax = plt.subplots(figsize=(7.16, 3.45))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.82, bottom=0.24)
    fig.text(
        0.09,
        0.94,
        "False-positive burden by method and hard-negative family",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.09,
        0.895,
        "Stacked selected-threshold false positives show where each method spends the false-alarm budget.",
        fontsize=7.2,
        color=MUTED,
    )
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for family in FAMILY_ORDER:
        values = np.asarray([totals[method].get(family, 0.0) for method in methods])
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.35,
            label=FAMILY_LABELS[family],
        )
        bottom += values
    for idx, (method, total) in enumerate(zip(methods, bottom)):
        ax.text(
            idx, total + 1.0, f"{int(total)} FP", ha="center", va="bottom", fontsize=6.4, color=INK
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in methods], rotation=25, ha="right", fontsize=6.8
    )
    ax.set_ylabel("Selected-threshold false positives", fontsize=7.5)
    ax.set_ylim(0.0, max(bottom) * 1.20 if len(bottom) else 1.0)
    _clean(ax, ygrid=True)
    handles = [
        Patch(facecolor=FAMILY_COLORS[fam], label=FAMILY_LABELS[fam]) for fam in FAMILY_ORDER
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.50), frameon=False, fontsize=6.1
    )
    ax.text(
        0.0,
        -0.35,
        "Near-threshold negative counts are retained as evidence guardrails; the stacked bars show actual selected-threshold FP.",
        transform=ax.transAxes,
        fontsize=6.4,
        color=MUTED,
    )
    return _save(fig, roots, "false_alarm_breakdown.png")


def _parse_iq_ref(roots: Roots, ref: str) -> tuple[Path, int] | None:
    if "#row=" not in ref:
        return None
    shard, row = ref.split("#row=", 1)
    try:
        row_index = int(row)
    except ValueError:
        return None
    return roots.training / shard, row_index


def _read_iq(roots: Roots, record: dict[str, str]) -> np.ndarray:
    parsed = _parse_iq_ref(roots, record.get("raw_complex_iq_ref", ""))
    if parsed is None:
        raise FileNotFoundError(f"missing raw_complex_iq_ref for {record.get('record_id', '')}")
    shard_path, row_index = parsed
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    with np.load(shard_path, allow_pickle=False) as shard:
        iq = np.asarray(shard["iq"][row_index])
    if iq.ndim == 3:
        iq = iq.mean(axis=0)
    elif iq.ndim > 3:
        iq = iq.reshape((-1,) + iq.shape[-2:]).mean(axis=0)
    return iq


def _range_doppler(iq: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(iq))))


def _normalize_images(images: list[np.ndarray]) -> list[np.ndarray]:
    if not images:
        return []
    stacked = np.concatenate([image.ravel() for image in images])
    lo = float(np.percentile(stacked, 2))
    hi = float(np.percentile(stacked, 99))
    if hi <= lo:
        return [np.zeros_like(image) for image in images]
    return [np.clip((image - lo) / (hi - lo), 0.0, 1.0) for image in images]


def _records_by_id(context: Context) -> dict[str, dict[str, str]]:
    return {row.get("record_id", ""): row for row in context.records}


def _positive_phase_records(context: Context) -> list[dict[str, str]]:
    holdout_positive = [
        row
        for row in context.records
        if row.get("split_role") == "holdout" and _safe_int(row.get("label_id")) == 1
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in holdout_positive:
        groups[row.get("scenario_group_id", "")].append(row)
    for group_id in sorted(groups):
        phase_map = {row.get("phase_id"): row for row in groups[group_id]}
        if all(phase in phase_map for phase, _label, _window in PHASES):
            return [phase_map[phase] for phase, _label, _window in PHASES]
    return holdout_positive[:3]


def _ei_false_positive_records(context: Context) -> list[dict[str, str]]:
    by_id = _records_by_id(context)
    fps = []
    for row in context.advanced_rows.values():
        if row.get("split_role") != "holdout":
            continue
        if _safe_int(row.get("label_id")) != 0 or _safe_int(row.get("binary_prediction")) != 1:
            continue
        record = by_id.get(row.get("record_id", ""))
        if record:
            fps.append(record)
    return fps[:3]


def _accepted_fusion_false_positive_records(roots: Roots, context: Context) -> list[dict[str, str]]:
    confusion_rows = _read_csv(roots.evidence / "selected_threshold_confusion_matrix.csv")
    threshold = None
    for row in confusion_rows:
        if row.get("method") == "layered_fusion_c2":
            threshold = _safe_float(row.get("threshold"))
            break
    if threshold is None or not math.isfinite(threshold):
        return []
    fusion = _index(_read_csv(roots.baseline / "fusion_predictions.csv"))
    fps: list[dict[str, str]] = []
    for record in context.records:
        if record.get("split_role") != "holdout" or _safe_int(record.get("label_id")) != 0:
            continue
        score = _safe_float(
            fusion.get(record.get("record_id", ""), {}).get("fusion_probability"), 0.0
        )
        if score >= threshold:
            fps.append(record)
    return fps[:3]


def _family_for_record(context: Context, record: dict[str, str]) -> str:
    scenario = context.scenarios.get(record.get("scenario_group_id", ""), {})
    family = scenario.get("hard_negative_role") or scenario.get("confuser_family") or "positive"
    return FAMILY_LABELS.get(family, family.replace("_", " "))


def _render_rd_grid(
    roots: Roots,
    context: Context,
    rows: list[list[dict[str, str]]],
    row_labels: list[str],
    titles: list[str],
    filename: str,
    *,
    vector: bool,
) -> Path:
    images = [[_range_doppler(_read_iq(roots, record)) for record in row] for row in rows]
    normalized = _normalize_images([image for row in images for image in row])
    norm_iter = iter(normalized)
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(7.16, 1.55 + 1.32 * len(rows)),
        constrained_layout=False,
    )
    if len(rows) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.11, right=0.91, top=0.82, bottom=0.12, hspace=0.28, wspace=0.12)
    fig.text(0.11, 0.94, titles[0], fontsize=10, weight="bold")
    fig.text(0.11, 0.90, titles[1], fontsize=7.2, color=MUTED)
    image_handle = None
    for row_idx, row in enumerate(rows):
        for col_idx, record in enumerate(row):
            ax = axes[row_idx, col_idx]
            rd_norm = next(norm_iter)
            image_handle = ax.imshow(
                rd_norm, aspect="auto", origin="lower", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0
            )
            phase = record.get("phase_id", "").replace("_", " ")
            if row_idx == 0:
                ax.set_title(_wrap(phase, 16), fontsize=6.8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=7.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color(GRID)
            if row_idx > 0:
                ax.text(
                    0.03,
                    0.05,
                    _wrap(_family_for_record(context, record), 18),
                    transform=ax.transAxes,
                    fontsize=5.8,
                    color="white",
                    bbox={"facecolor": "#111827", "alpha": 0.66, "edgecolor": "none", "pad": 1.5},
                )
    if image_handle is not None:
        cax = fig.add_axes([0.925, 0.18, 0.018, 0.60])
        cbar = fig.colorbar(image_handle, cax=cax)
        cbar.set_label("normalized log magnitude", fontsize=6.2)
        cbar.ax.tick_params(labelsize=5.8)
    return _save(fig, roots, filename, vector=vector)


def figure_radar_positive_vs_false_positive(roots: Roots, context: Context) -> Path:
    positive = _positive_phase_records(context)
    false_positive = _ei_false_positive_records(context)
    if len(positive) < 3 or len(false_positive) < 3:
        raise SystemExit("need three positive and three EI false-positive radar records")
    return _render_rd_grid(
        roots,
        context,
        [positive[:3], false_positive[:3]],
        ["Positive\npublic proxy", "Challenging\nfalse positives"],
        [
            "Synthetic range-Doppler examples: positive public proxy vs challenging false positives",
            "Qualitative normalized diagnostics only; these are not measured imagery and not detector input for the KPI.",
        ],
        "radar_positive_vs_false_positive.png",
        vector=True,
    )


def figure_appendix_radar_samples(roots: Roots, context: Context) -> Path:
    positive = _positive_phase_records(context)
    ei_fp = _ei_false_positive_records(context)
    prior_fp = _accepted_fusion_false_positive_records(roots, context)
    if len(positive) < 3 or len(ei_fp) < 3 or len(prior_fp) < 3:
        raise SystemExit(
            "need positive, EI false-positive, and accepted-fusion false-positive radar records"
        )
    return _render_rd_grid(
        roots,
        context,
        [positive[:3], prior_fp[:3], ei_fp[:3]],
        ["Positive\npublic proxy", "Accepted fusion\nfalse positives", "EI\nfalse positives"],
        [
            "Appendix radar sample gallery",
            "Rows contrast public-proxy positives with selected-threshold false-positive pressure across the same synthetic range-Doppler view.",
        ],
        "appendix_radar_samples.png",
        vector=False,
    )


def generate_all(roots: Roots, context: Context) -> list[Path]:
    _style()
    return [
        figure_architecture_stack(roots, context),
        figure_phase_method_ladder(roots, context),
        figure_ei_evolution_money_plot(roots, context),
        figure_false_alarm_breakdown(roots, context),
        figure_radar_positive_vs_false_positive(roots, context),
        figure_appendix_radar_samples(roots, context),
    ]


def _check_sources(roots: Roots) -> None:
    required = [
        roots.training / "records.csv",
        roots.training / "scenario_manifest.csv",
        roots.baseline / "classical_radar_processing.csv",
        roots.baseline / "fusion_predictions.csv",
        roots.baseline / "ml_detector_baselines.csv",
        roots.advanced / "advanced_predictions.csv",
        roots.evidence / "paper_evidence_manifest.json",
        roots.evidence / "ei_evolution_trace.csv",
        roots.evidence / "false_alarm_by_method_family.csv",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required figure source(s): " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--paper-evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    roots = Roots(
        training=args.training_root,
        baseline=args.baseline_root,
        advanced=args.advanced_root,
        evidence=args.paper_evidence_root,
        figures=args.output_dir,
    )
    if args.strict:
        _check_sources(roots)
    context = _load_context(roots, strict=args.strict)
    for path in generate_all(roots, context):
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
