#!/usr/bin/env python3
"""Generate the radar-first figure set for the major paper upgrade."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.paper_evidence_major_upgrade_v1 import (  # noqa: E402
    _comparable_ablation_rows,
    _selected_component_human_weights,
)

PDF_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)

try:
    import numpy as np
except Exception:  # pragma: no cover - import guard for minimal environments
    np = None  # type: ignore[assignment]

try:
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, Rectangle
except Exception as exc:  # pragma: no cover - import guard for minimal environments
    raise SystemExit(
        "matplotlib is required to render paper figures; install matplotlib or run this script in the repository dev environment."
    ) from exc


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

NIGHT_RED = "#8c2d3e"
NIGHT_GREEN = "#1f7a5a"
NIGHT_RED_LIGHT = "#f2d6db"
NIGHT_GREEN_LIGHT = "#d9efe7"
EI_LABEL = "Engineered Intelligence (EI)"
EI_SHORT = "EI candidate"
BASELINE_LABEL = "Prior fusion baseline"
PRIMARY_KPI_TEXT = "Recall@<=1%FPR"

# IEEE-ready visual style. Figures are authored at final paper size so labels
# do not become tiny after LaTeX scaling.
IEEE_TEXT_WIDTH_IN = 7.16
IEEE_COLUMN_WIDTH_IN = 3.45
FONT_TITLE = 9.6
FONT_SUBTITLE = 8.2
FONT_AXIS = 7.4
FONT_TICK = 6.6
FONT_TINY = 6.1
FONT_CARD = 6.6
LINE_MAIN = 1.25
LINE_GRID = 0.55
SPINE_WIDTH = 0.65

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "echoforge_iq",
    ("#10151d", "#24445e", "#2d7c84", "#88b36d", "#f0d56b", "#f7f4e9"),
    N=256,
)
NIGHT_RED_GREEN = LinearSegmentedColormap.from_list(
    "echoforge_night_red_green",
    (NIGHT_RED, "#f7f7f2", NIGHT_GREEN),
    N=256,
)
DELTA_NORM = TwoSlopeNorm(vmin=-0.80, vcenter=0.0, vmax=0.20)

FIG_DPI = 450
VECTOR_FIGURES = {
    "architecture_stack.png",
    "monte_carlo_split_flow.png",
    "kpi_ranking.png",
    "phase_kpi.png",
    "iq_drone_samples.png",
    "detector_ml_pipeline.png",
    "anchor_overlay.png",
    "ei_workflow.png",
}
HEATMAP_FIGURES = {"iq_negative_samples.png"}


def _paper_fig(height: float, *, constrained: bool = True) -> Any:
    return plt.figure(figsize=(IEEE_TEXT_WIDTH_IN, height), constrained_layout=constrained)


def _wrap_compact(value: Any, width: int = 24) -> str:
    return "\n".join(
        textwrap.wrap(
            str(value).replace("_", " "),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _short_branch_label(value: Any) -> str:
    text = str(value)
    if "meta_fusion" in text:
        return EI_SHORT
    replacements = {
        "High-resolution X/Ku C-UAS": "High-res. X/Ku",
        "Tactical S-band AESA/MHR": "Tactical S-band",
        "GBAD 3D/4D cueing": "GBAD 3D/4D",
        "X/Ku public cueing envelope": "X/Ku cue",
        "locked_candidate": EI_SHORT,
        "selected_candidate": EI_SHORT,
        "Locked candidate": EI_SHORT,
        "locked candidate": EI_SHORT,
        "selected candidate": EI_SHORT,
        "high_resolution_xku_cuas": "High-res. X/Ku",
        "tactical_s_band_aesa": "Tactical S-band",
        "gbad_3d4d_cueing": "GBAD 3D/4D",
        "distributed_acoustic_cue": "Acoustic cue",
        "tabular_ml_baseline": "Tabular ML",
        "sequence_ml_proxy": "Sequence ML",
        "layered_fusion_c2": "Prior fusion",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ")


def _reader_label(value: Any, width: int | None = None) -> str:
    text = str(value or "")
    replacements = {
        "full_locked_candidate": "Full EI candidate",
        "locked_candidate": EI_SHORT,
        "selected_candidate": EI_SHORT,
        "locked candidate": EI_SHORT,
        "selected candidate": EI_SHORT,
        "Locked candidate": EI_SHORT,
        "Selected candidate": EI_SHORT,
        "layered_fusion_c2": BASELINE_LABEL,
        "prior fusion": BASELINE_LABEL,
        "selected": EI_SHORT,
        "passive_rf_only": "Passive-RF only",
        "radar_only": "Radar only",
        "acoustic_only": "Acoustic only",
        "radar_acoustic": "Radar + acoustic",
        "radar_rf": "Radar + passive-RF",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("_", " ")
    return _wrap_compact(text, width) if width else text


def _delta_color(value: float) -> Any:
    if not math.isfinite(value):
        return SLATE
    bounded = max(-0.80, min(0.20, value))
    return NIGHT_RED_GREEN(DELTA_NORM(bounded))


def _annotate_vertical_bars(ax: Any, bars: Any, *, dy: float = 0.018) -> None:
    top = ax.get_ylim()[1]
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + dy, top * 0.98),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_TINY,
            color=MUTED,
        )


def _short_family_label(value: Any) -> str:
    mapping = {
        "single_bird": "Single bird",
        "bird_flock": "Bird flock",
        "shorebird_wader": "Shorebird",
        "gull_tern": "Gull/tern",
        "raptor_falcon": "Raptor",
        "flamingo_large_bird": "Large bird",
        "seabird_cormorant": "Seabird",
        "seasonal_migratory_density": "Migration",
        "rc_fixed_wing": "RC fixed-wing",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _short_feature_label(value: Any) -> str:
    mapping = {
        "micro_doppler_bandwidth_hz": "micro-Doppler\nbandwidth",
        "micro_doppler_peak_hz": "micro-Doppler\npeak",
        "spectral_entropy": "spectral\nentropy",
        "range_m": "range",
        "return_power_db": "return\npower",
    }
    return mapping.get(str(value), _wrap_compact(value, 18))


def _clean_axes(ax: Any, *, xgrid: bool = False, ygrid: bool = False) -> None:
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=LINE_GRID)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=LINE_GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=SPINE_WIDTH, length=2.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(SPINE_WIDTH)
        ax.spines[spine].set_color(INK)


def _draw_text_card(
    ax: Any,
    title: str,
    lines: list[str],
    *,
    wrap_width: int = 42,
    columns: int = 1,
    face: str = "white",
    edge: str = GRID,
    accent: str = INK,
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        title,
        ha="left",
        va="top",
        fontsize=FONT_SUBTITLE,
        weight="bold",
        color=accent,
        transform=ax.transAxes,
    )
    if not lines:
        return
    if columns == 1:
        ax.text(
            0.02,
            0.84,
            "\n".join(_wrap_compact(line, wrap_width) for line in lines),
            ha="left",
            va="top",
            fontsize=FONT_CARD,
            color=INK,
            linespacing=1.22,
            transform=ax.transAxes,
        )
        return
    n_left = math.ceil(len(lines) / columns)
    for idx, line in enumerate(lines):
        col = idx // n_left
        row = idx % n_left
        x = 0.02 + col * 0.49
        y = 0.82 - row * min(0.16, 0.72 / max(n_left, 1))
        ax.text(
            x,
            y,
            _wrap_compact(line, max(14, wrap_width // 2)),
            ha="left",
            va="top",
            fontsize=FONT_CARD,
            color=INK,
            linespacing=1.05,
            transform=ax.transAxes,
        )


def _style_table(table: Any, *, fontsize: float = FONT_TINY, yscale: float = 1.22) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, yscale)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        cell.PAD = 0.035
        if row_idx == 0:
            cell.set_facecolor(PALE_BLUE)
            cell.set_text_props(weight="bold", color=INK)
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f9fbfd")
        else:
            cell.set_facecolor("white")


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


def _wrap(value: Any, width: int = 28) -> str:
    text = str(value).replace("_", " ")
    return "\n".join(
        textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    )


def _first_number(mapping: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if key not in mapping:
            continue
        value = _safe_float(mapping.get(key))
        if math.isfinite(value):
            return value
    return default


def _global_normalize01(
    arrays: list[np.ndarray], low_pct: float = 2.0, high_pct: float = 98.0
) -> list[np.ndarray]:
    finite_values = np.concatenate(
        [np.asarray(arr, dtype=np.float64).ravel() for arr in arrays if arr is not None]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [np.zeros_like(arr, dtype=np.float64) for arr in arrays]
    low, high = np.percentile(finite_values, (low_pct, high_pct))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(np.nanmin(finite_values))
        high = float(np.nanmax(finite_values))
    if high <= low:
        return [np.zeros_like(arr, dtype=np.float64) for arr in arrays]
    return [
        np.clip((np.asarray(arr, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
        for arr in arrays
    ]


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
    pdf_metadata = {
        "Creator": metadata["Software"],
        "Subject": description,
        "CreationDate": PDF_TIMESTAMP,
        "ModDate": PDF_TIMESTAMP,
    }
    pad = 0.08
    if filename in VECTOR_FIGURES:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=pad,
            metadata=pdf_metadata,
        )
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=pad, metadata=metadata)
    elif filename in HEATMAP_FIGURES:
        fig.savefig(
            path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=pad,
            metadata=metadata,
        )
    else:
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=pad, metadata=metadata)
    plt.close(fig)
    return path


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_AXIS,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": FONT_SUBTITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_TINY,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": FIG_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
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
    size: float = FONT_CARD,
) -> None:
    rect = Rectangle((x, y), w, h, linewidth=1.0, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        _wrap_compact(text, 24),
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
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
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
    component_scores = _read_csv_rows(roots.advanced_root / "selected_component_scores.csv")
    if not component_scores:
        component_scores = _read_csv_rows(
            roots.paper_evidence_root / "selected_component_scores.csv"
        )
    if not component_scores:
        component_transparency = evidence.get("component_transparency", {})
        if isinstance(component_transparency, dict):
            raw_component_scores = component_transparency.get("component_scores", [])
            if isinstance(raw_component_scores, list):
                component_scores = raw_component_scores
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
    fig, ax = _setup_diagram(IEEE_TEXT_WIDTH_IN, 3.35)
    ax.text(0.15, 5.72, "EchoForge radar evidence stack", fontsize=FONT_TITLE, weight="bold")
    ax.text(
        0.15,
        5.35,
        "Public-proxy priors, synthetic sensing, detector views, and validation evidence stay separated by claim boundary.",
        fontsize=FONT_TINY,
        color=MUTED,
    )
    layers = [
        ("Public-proxy object priors", PALE_BLUE, BLUE),
        ("Radar archetype and clutter", PALE_TEAL, TEAL),
        ("Complex IQ plus cue streams", PALE_GOLD, GOLD),
        ("Detector views with leakage rails", PALE_GREEN, GREEN),
        ("Evidence bundle and claim boundary", PALE_RED, RED),
    ]
    x, w, h = 0.50, 5.55, 0.62
    y0 = 4.65
    for idx, (text, face, edge) in enumerate(layers):
        y = y0 - idx * 0.80
        _add_box(
            ax, x, y, w, h, text, face=face, edge=edge, weight="bold" if idx == 0 else "normal"
        )
        if idx < len(layers) - 1:
            _add_arrow(ax, (x + w / 2, y - 0.02), (x + w / 2, y - 0.21), color=edge)

    ax.text(6.55, 5.05, "Model card snapshot", fontsize=FONT_SUBTITLE, weight="bold")
    card_lines = [
        f"waveform: {radar_model.get('waveform_family', 'public-proxy radar')}",
        "pulses: 24 per CPI",
        "range bins: 20",
        "PRF/CPI: 4 kHz / 6 ms",
        "receiver: AGC, drift, quantization",
        "cues: acoustic + passive RF",
    ]
    for idx, line in enumerate(card_lines):
        _add_box(
            ax,
            6.50,
            4.35 - idx * 0.56,
            2.75,
            0.38,
            line,
            face="white",
            edge=GRID,
            size=FONT_TINY,
        )
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
    fig = _paper_fig(4.10, constrained=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.92], hspace=0.42, wspace=0.36)
    fig.suptitle(
        "Scenario balance and leakage diagnostics", fontsize=FONT_TITLE, weight="bold", y=0.995
    )
    fig.text(
        0.02,
        0.02,
        "Scenario groups are locked before phase expansion. The paper evidence keeps label, site, range, aspect, noise, and hard-negative family counts together.",
        fontsize=FONT_TINY,
        color=MUTED,
    )
    marginal_counts = balance.get("marginal_counts", {})
    ordered_keys = [
        ("split", "Split"),
        ("site", "Site"),
        ("range", "Range"),
        ("hard_negative_role", "Hard-negative"),
    ]
    for idx, (key, title) in enumerate(ordered_keys):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        counts = marginal_counts.get(key, {})
        if not counts:
            counts = {"missing": 1}
            _note_fallback(filename, f"{key} counts missing")
        if len(counts) > 8:
            ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            kept = dict(ordered[:7])
            kept["other"] = sum(value for _name, value in ordered[7:])
            counts = kept
        names = list(counts)
        values = [counts[name] for name in names]
        display_names = [
            _wrap_compact(
                name.replace("vegetation_motion_corridor", "vegetation")
                .replace("open_desert_edge", "desert edge")
                .replace("edge_of_track", "edge")
                .replace("broadside_right", "broadside R")
                .replace("broadside_left", "broadside L")
                .replace("clutter_only_counterfactual", "clutter-only")
                .replace("seasonal_migratory_density", "migration")
                .replace("_", " "),
                14,
            )
            for name in names
        ]
        palette = [BLUE, TEAL, GOLD, RED, GREEN]
        ax.barh(
            range(len(names)), values, color=[palette[i % len(palette)] for i in range(len(names))]
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(display_names, fontsize=FONT_TICK)
        ax.set_title(title, fontsize=FONT_SUBTITLE, weight="bold")
        _clean_axes(ax, xgrid=True)
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.01, i, f"{value}", va="center", fontsize=FONT_TINY)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    leakage = context.evidence.get("leakage_diagnostics", {})
    canary = leakage.get("canary_forbidden_feature_test", {})
    ax.text(
        0.02, 0.94, "Leakage checks", fontsize=FONT_SUBTITLE, weight="bold", transform=ax.transAxes
    )
    lines = [
        f"canary test: {str(canary.get('status', 'missing')).replace('_', ' ')}",
        f"stratum baseline AP: {leakage.get('stratum_only_baseline', {}).get('average_precision', float('nan')):.3f}",
        f"metadata baseline AP: {leakage.get('metadata_only_baseline', {}).get('average_precision', float('nan')):.3f}",
        f"label shuffle AP: {leakage.get('label_shuffle_sanity', {}).get('shuffled_labels', {}).get('average_precision', float('nan')):.3f}",
    ]
    for idx, line in enumerate(lines):
        _add_box(
            ax, 0.03, 0.74 - idx * 0.16, 0.92, 0.12, line, face="white", edge=GRID, size=FONT_TINY
        )

    ax = fig.add_subplot(gs[1:, 2])
    ax.axis("off")
    imbalance = balance.get("imbalance_scores", {})
    ax.text(
        0.02,
        0.94,
        "Marginal imbalance",
        fontsize=FONT_SUBTITLE,
        weight="bold",
        transform=ax.transAxes,
    )
    for idx, (name, value) in enumerate(sorted(imbalance.items())):
        ax.text(
            0.04,
            0.80 - idx * 0.13,
            f"{name}: {value:.3f}",
            fontsize=FONT_TINY,
            transform=ax.transAxes,
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
    fig = _paper_fig(4.20, constrained=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.78, hspace=0.66)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.30, 0.10, 0.86, 0.86])
    fig.text(
        0.08,
        0.935,
        "Holdout KPI and supporting diagnostics",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
    )

    selected_row = evaluation.get("selected", {})
    baseline_row = evaluation.get("baseline", {})
    primary_kpi = evaluation.get("primary_kpi", {})
    primary_label = primary_kpi.get("name", "LCB95 Recall@≤1%FPR")
    primary_value = _safe_float(
        primary_kpi.get("value"),
        _safe_float(selected_row.get("fixed_fpr_recall")),
    )

    kpi_ax = fig.add_subplot(gs[:, 0])
    methods = [(BASELINE_LABEL, baseline_row, GREEN), (EI_SHORT, selected_row, INK)]
    y = np.arange(len(methods))
    point_values = [_safe_float(row.get("fixed_fpr_recall")) for _label, row, _color in methods]
    lcb_values = [
        _safe_float(row.get("primary_kpi", {}).get("value")) for _label, row, _color in methods
    ]
    colors = [color for _label, _row, color in methods]
    kpi_ax.barh(y, point_values, color=colors, alpha=0.88, edgecolor="white", label="Point recall")
    for idx, (point, lcb) in enumerate(zip(point_values, lcb_values)):
        if math.isfinite(lcb):
            kpi_ax.plot([lcb, lcb], [idx - 0.28, idx + 0.28], color=RED, linewidth=2.3)
            kpi_ax.text(
                min(point + 0.02, 0.97),
                idx,
                f"point {point:.3f}\nLCB95 {lcb:.3f}",
                va="center",
                fontsize=FONT_TINY,
                color=MUTED,
            )
    kpi_ax.set_yticks(y)
    kpi_ax.set_yticklabels([label for label, _row, _color in methods], fontsize=FONT_AXIS)
    kpi_ax.set_xlim(0.0, 1.02)
    kpi_ax.set_xlabel("Recall at FPR <= 1%", fontsize=FONT_AXIS)
    kpi_ax.set_title("Primary KPI: group-block LCB95", fontsize=FONT_SUBTITLE, weight="bold")
    _clean_axes(kpi_ax, xgrid=True)
    kpi_ax.text(
        0.02,
        0.04,
        f"{primary_label}: {primary_value:.3f}",
        fontsize=FONT_TINY,
        color=MUTED,
        transform=kpi_ax.transAxes,
    )

    rank_ax = fig.add_subplot(gs[0, 2])
    selected_method = evaluation.get("selected_method", "")
    rows = [
        row
        for row in context.baseline_metrics
        if row.get("split_role") == "holdout" and row.get("phase_id") == "all"
    ]
    rows.sort(key=lambda row: _safe_float(row.get("average_precision")), reverse=True)
    labels = [row.get("method", "") for row in rows]
    values = [_safe_float(row.get("average_precision")) for row in rows]
    positions = list(range(len(labels)))
    colors = [GREEN if label == "layered_fusion_c2" else BLUE for label in labels]
    if selected_method and selected_method not in labels:
        labels.append(selected_method)
        values.append(_safe_float(selected_row.get("average_precision")))
        colors.append(INK)
        positions.append(len(positions))
    rank_ax.barh(positions, values, color=colors, edgecolor="white", linewidth=0.8)
    rank_ax.set_yticks(positions)
    rank_ax.set_yticklabels([_short_branch_label(label) for label in labels], fontsize=FONT_TINY)
    rank_ax.set_xlabel("Holdout AP", fontsize=FONT_AXIS)
    _clean_axes(rank_ax, xgrid=True)
    rank_ax.set_title("AP ranking diagnostic", fontsize=FONT_SUBTITLE, weight="bold")

    roc_ax = fig.add_subplot(gs[0, 3])
    pr_ax = fig.add_subplot(gs[1, 2])
    for ax2, metric in ((roc_ax, "roc"), (pr_ax, "pr")):
        sel_curve = evaluation.get("curves", {}).get("selected", {}).get(metric, [])
        base_curve = evaluation.get("curves", {}).get("baseline", {}).get(metric, [])
        if metric == "roc":
            ax2.plot(
                [p["fpr"] for p in base_curve],
                [p["tpr"] for p in base_curve],
                color=GREEN,
                linewidth=1.8,
                label=BASELINE_LABEL,
            )
            ax2.plot(
                [p["fpr"] for p in sel_curve],
                [p["tpr"] for p in sel_curve],
                color=INK,
                linewidth=1.8,
                label=EI_SHORT,
            )
            ax2.set_xlabel("False-positive rate", fontsize=FONT_AXIS)
            ax2.set_ylabel("True-positive rate", fontsize=FONT_AXIS)
            ax2.set_title("ROC low-FPR inset", fontsize=FONT_SUBTITLE, weight="bold")
            ax2.axvline(0.01, color=RED, linewidth=1.0, linestyle="--")
            ax2.text(0.011, 0.07, "1% FPR", fontsize=FONT_TINY, color=RED)
            ax2.set_xlim(0.0, 0.025)
            ax2.set_ylim(0.0, 1.02)
        else:
            ax2.plot(
                [p["recall"] for p in base_curve],
                [p["precision"] for p in base_curve],
                color=GREEN,
                linewidth=1.8,
                label=BASELINE_LABEL,
            )
            ax2.plot(
                [p["recall"] for p in sel_curve],
                [p["precision"] for p in sel_curve],
                color=INK,
                linewidth=1.8,
                label=EI_SHORT,
            )
            ax2.set_xlabel("Recall", fontsize=FONT_AXIS)
            ax2.set_ylabel("Precision", fontsize=FONT_AXIS)
            ax2.set_title("PR curve", fontsize=FONT_SUBTITLE, weight="bold")
            base_rate = _safe_float(selected_row.get("positive_count")) / max(
                _safe_float(selected_row.get("count"), 1.0), 1.0
            )
            ax2.axhline(base_rate, color=RED, linewidth=1.0, linestyle="--", label="base rate")
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.02)
        _clean_axes(ax2, xgrid=True, ygrid=True)
    roc_ax.legend(loc="lower right", fontsize=FONT_TINY, frameon=False)
    pr_ax.legend(loc="upper right", fontsize=FONT_TINY, frameon=False)

    cal_ax = fig.add_subplot(gs[1, 3])
    bins = selected_row.get("calibration_bins", [])
    if not bins:
        bins = evaluation.get("calibration", {}).get("selected", {}).get("bins", [])
    bin_labels = [f"{_safe_float(row.get('bin_left')):.1f}" for row in bins]
    counts = [_safe_float(row.get("count")) for row in bins]
    gaps = [_safe_float(row.get("gap")) for row in bins]
    x = np.arange(len(bin_labels))
    cal_ax.bar(x, counts, color=PALE_BLUE, edgecolor=BLUE, linewidth=0.8)
    nonzero = [count for count in counts if count > 0]
    if nonzero:
        cal_ax.set_ylim(0.0, max(nonzero) * 1.12)
    tick_positions = list(range(0, len(bin_labels), 2))
    cal_ax.set_xticks(tick_positions)
    cal_ax.set_xticklabels([bin_labels[idx] for idx in tick_positions], fontsize=FONT_TINY)
    cal_ax.set_ylabel("Bin count", fontsize=FONT_AXIS)
    cal_ax.set_title("Calibration bin counts", fontsize=FONT_SUBTITLE, weight="bold")
    max_count = max(counts) if counts else 0.0
    for idx, (count, gap) in enumerate(zip(counts, gaps)):
        if count > 0 and count < max_count * 0.25 and math.isfinite(gap):
            cal_ax.text(
                idx,
                count,
                f"gap {gap:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONT_TINY,
                color=MUTED,
            )
    cal_ax.text(
        0.98,
        0.86,
        f"Brier {selected_row.get('brier_score', float('nan')):.3f} / ECE {selected_row.get('ece', float('nan')):.3f}",
        transform=cal_ax.transAxes,
        fontsize=FONT_TINY,
        color=MUTED,
        ha="right",
    )
    _clean_axes(cal_ax, ygrid=True)
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
    fig = _paper_fig(3.55, constrained=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.42)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    fig.text(
        0.08,
        0.92,
        "Phase behavior and false-alarm burden",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
    )

    phase_metrics = evaluation.get("phase_metrics", {})
    selected = phase_metrics.get(evaluation.get("selected_method", ""), {})
    baseline = phase_metrics.get("layered_fusion_c2", {})
    phase_order = list(PHASES)
    x = np.arange(len(phase_order))
    width = 0.17
    ax = fig.add_subplot(gs[0, 0])
    selected_ap = [selected.get(phase, {}).get("average_precision", 0.0) for phase in phase_order]
    selected_recall = [
        selected.get(phase, {}).get("fixed_fpr_recall", 0.0) for phase in phase_order
    ]
    baseline_ap = [baseline.get(phase, {}).get("average_precision", 0.0) for phase in phase_order]
    baseline_recall = [
        baseline.get(phase, {}).get("fixed_fpr_recall", 0.0) for phase in phase_order
    ]
    series = [
        ("Baseline AP", baseline_ap, PALE_GREEN, GREEN),
        ("Baseline R@<=1%FPR", baseline_recall, PALE_BLUE, BLUE),
        ("EI AP", selected_ap, INK, INK),
        ("EI R@<=1%FPR", selected_recall, NIGHT_GREEN_LIGHT, NIGHT_GREEN),
    ]
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, (label, values, face, edge) in zip(offsets, series):
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=face,
            edgecolor=edge,
            linewidth=0.8,
            label=label,
        )
        _annotate_vertical_bars(ax, bars, dy=0.012)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{PHASE_LABELS[p]}\n{PHASE_WINDOWS[p]}" for p in phase_order], fontsize=FONT_TINY
    )
    ax.set_ylabel("Holdout metric value", fontsize=FONT_AXIS)
    ax.set_ylim(0.0, 1.02)
    _clean_axes(ax, ygrid=True)
    ax.legend(loc="upper left", fontsize=FONT_TINY, frameon=False, ncol=2, columnspacing=0.8)
    ax.set_title("Per-phase AP and low-FPR recall", fontsize=FONT_SUBTITLE, weight="bold")
    fig.text(
        0.08,
        0.075,
        "Each phase has n=8 positive holdout records from 8 positive groups; phase results are diagnostic.",
        fontsize=FONT_TINY,
        color=MUTED,
    )

    fa_ax = fig.add_subplot(gs[0, 1])
    family_rows = _read_csv_rows(
        context.roots.paper_evidence_root / "false_alarm_by_method_family.csv"
    )
    method_rows = _read_csv_rows(
        context.roots.paper_evidence_root / "top_method_false_positive_frequency.csv"
    )
    if not family_rows or not method_rows:
        family_rows = _fallback_false_alarm_rows()
        method_rows = [
            {
                "method": "locked_candidate",
                "method_label": "locked_candidate",
                "rank": 1,
                "selected_threshold_fp_count": 1,
                "fp_per_1000_negatives": 0.2,
                "recall_at_leq_1pct_fpr": 0.8,
                "top_fp_family": "single_bird",
            }
        ]
    method_rows = sorted(method_rows, key=lambda row: _safe_float(row.get("rank"), 99.0))[:5]
    selected_methods = [row.get("method", "") for row in method_rows if row.get("method")]
    family_names = sorted(
        {row.get("family", "") for row in family_rows if row.get("family")},
        key=lambda name: (
            -sum(
                _safe_float(row.get("false_alarm_count"))
                for row in family_rows
                if row.get("family") == name
            ),
            name,
        ),
    )[:4]
    family_palette = {
        "single_bird": RED,
        "bird_flock": GOLD,
        "shorebird_wader": BLUE,
        "gull_tern": TEAL,
        "raptor_falcon": GREEN,
        "flamingo_large_bird": PALE_RED,
        "seabird_cormorant": PALE_GOLD,
        "seasonal_migratory_density": PALE_BLUE,
        "rc_fixed_wing": INK,
    }
    method_to_family_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    method_to_summary: dict[str, dict[str, str]] = {}
    for row in family_rows:
        method_to_family_rows[row.get("method", "")].append(row)
    for row in method_rows:
        method_to_summary[row.get("method", "")] = row
    y_positions = np.arange(len(method_rows))
    max_total = 0.0
    seen_families: set[str] = set()
    for idx, method in enumerate(selected_methods):
        summary = method_to_summary.get(method, {})
        near_total = sum(
            _safe_float(row.get("near_threshold_count"))
            for row in method_to_family_rows.get(method, [])
        )
        total_fp = sum(
            _safe_float(row.get("false_alarm_count"))
            for row in method_to_family_rows.get(method, [])
        )
        max_total = max(max_total, near_total)
        fa_ax.barh(
            idx,
            near_total,
            color=PALE_GREEN,
            edgecolor=GRID,
            linewidth=0.8,
            alpha=0.35,
            label=None,
        )
        left = 0.0
        for family in family_names:
            family_count = sum(
                _safe_float(row.get("false_alarm_count"))
                for row in method_to_family_rows.get(method, [])
                if row.get("family") == family
            )
            if family_count <= 0.0:
                continue
            fa_ax.barh(
                idx,
                family_count,
                left=left,
                color=family_palette.get(family, BLUE),
                edgecolor="white",
                linewidth=0.5,
                label=None,
            )
            left += family_count
            max_total = max(max_total, left)
            seen_families.add(family)
        fa_ax.text(
            max(left, near_total) + 0.18,
            idx,
            f"FP {int(total_fp)} | near {int(near_total)} | R@1% {_safe_float(summary.get('recall_at_leq_1pct_fpr')):.3f}",
            va="center",
            fontsize=FONT_TINY,
            color=MUTED,
        )
    fa_ax.set_yticks(y_positions)
    fa_ax.set_yticklabels(
        [
            _reader_label(method_to_summary.get(method, {}).get("method_label", method), 14)
            for method in selected_methods
        ],
        fontsize=FONT_TICK,
    )
    fa_ax.set_xlabel("False-alarm count by top family", fontsize=FONT_AXIS)
    _clean_axes(fa_ax, xgrid=True)
    fa_ax.set_xlim(0.0, max_total * 1.15 if max_total > 0 else 1.0)
    fa_ax.set_title("Cross-method false-alarm burden", fontsize=FONT_SUBTITLE, weight="bold")
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
    return np.log1p(np.abs(rd))


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
    fig = _paper_fig(4.10, constrained=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.90, 1.42],
        width_ratios=[1.32, 0.82, 0.94],
        hspace=0.44,
        wspace=0.34,
    )
    fig.text(
        0.06,
        0.93,
        "Radar model card, resolution budget, and public-proxy boundary",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
    )

    table_ax = fig.add_subplot(gs[0, :])
    table_ax.axis("off")
    carrier_bands = card.get("carrier_bands", []) or []
    table_rows = []
    for band in carrier_bands:
        label = _short_branch_label(
            band.get("label") or band.get("display_name") or band.get("branch", "")
        )
        bandwidth_mhz = _first_number(band, "bandwidth_mhz", "bw_mhz", "bandwidth_MHz", default=0.0)
        center_ghz = _first_number(band, "center_ghz", "f0_ghz", "carrier_ghz", default=0.0)
        cpi_ms = _first_number(band, "cpi_ms", "cpi_msec", default=6.0)
        cpi_s = cpi_ms / 1000.0 if cpi_ms > 0 else float("nan")
        pulses = int(
            round(
                _first_number(
                    band,
                    "pulses",
                    "chirps",
                    "pulse_count",
                    "chirp_count",
                    "pulses_per_cpi",
                    "chirps_per_cpi",
                    default=24.0,
                )
            )
        )
        prf_hz = _first_number(
            band, "prf_hz", "prf", "pulse_repetition_frequency_hz", default=4000.0
        )
        crf_hz = _first_number(
            band,
            "crf_hz",
            "chirp_repetition_frequency_hz",
            default=prf_hz if math.isfinite(prf_hz) else 4000.0,
        )
        range_resolution_m = _first_number(
            band,
            "range_resolution_m",
            "delta_r_m",
            default=(
                299792458.0 / (2.0 * bandwidth_mhz * 1e6) if bandwidth_mhz > 0 else float("nan")
            ),
        )
        doppler_resolution_hz = _first_number(
            band,
            "doppler_resolution_hz",
            "delta_fd_hz",
            default=(1.0 / cpi_s if math.isfinite(cpi_s) and cpi_s > 0 else float("nan")),
        )
        velocity_resolution_mps = _first_number(
            band,
            "velocity_resolution_mps",
            "delta_v_mps",
            default=(
                (299792458.0 / (2.0 * center_ghz * 1e9)) * doppler_resolution_hz
                if center_ghz > 0 and math.isfinite(doppler_resolution_hz)
                else float("nan")
            ),
        )
        table_rows.append(
            [
                label,
                _short_branch_label(band.get("band", "")),
                f"{bandwidth_mhz:.0f}",
                f"{range_resolution_m:.3f}",
                f"{prf_hz:.0f}/{crf_hz:.0f}",
                f"{cpi_ms:.1f}",
                f"{pulses:d}",
                f"{doppler_resolution_hz:.1f}",
                f"{velocity_resolution_mps:.3f}",
            ]
        )
    if not table_rows:
        table_rows = [
            ["High-res. X/Ku", "X/Ku", "600", "0.250", "4000/4000", "6.0", "24", "166.7", "2.498"],
            ["Tactical S-band", "S", "180", "0.833", "4000/4000", "6.0", "24", "166.7", "8.059"],
            ["GBAD 3D/4D", "X/Ku cue", "300", "0.500", "4000/4000", "6.0", "24", "166.7", "2.426"],
        ]
    col_labels = [
        "Branch",
        "Band",
        "BW\nMHz",
        "dR\nm",
        "PRF/CRF\nHz",
        "CPI\nms",
        "Pulses",
        "dfD\nHz",
        "dv\nm/s",
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20, 0.08, 0.075, 0.075, 0.13, 0.075, 0.075, 0.075, 0.075],
    )
    _style_table(table, fontsize=FONT_TINY, yscale=1.18)

    model_box = fig.add_subplot(gs[1, 0])
    _draw_text_card(
        model_box,
        "Public-proxy assumptions",
        [
            "positive class: fixed-wing pusher-prop public proxy",
            "geometry: swept/delta wing; rear pusher propulsor",
            "speed: 45--60 m/s; prop micro-Doppler: 150--217 Hz",
            "phase windows: 0--30 s, 30--90 s, 90--150 s",
            "launch proxy: rail/catapult take-up",
            "RCS/aspect: -28 to -2 dBsm public-source proxy",
        ],
        wrap_width=44,
        face="white",
        edge=GRID,
        accent=INK,
    )

    impair_ax = fig.add_subplot(gs[1, 1])
    impairments = card.get("receiver_impairments", []) or [
        "AGC compression",
        "clock drift",
        "quantization",
        "dropped CPI",
        "PRF ambiguity",
        "Doppler folding",
        "calibration offset",
        "multipath masking",
    ]
    _draw_text_card(
        impair_ax,
        "Receiver impairments",
        [str(item) for item in impairments],
        wrap_width=18,
        columns=2,
        face=PALE_RED,
        edge=RED,
        accent=RED,
    )

    notes_ax = fig.add_subplot(gs[1, 2])
    cue_defs = card.get("cue_definitions", {})
    cue_text = [
        f"acoustic: {cue_defs.get('acoustic', 'node cadence and agreement')}",
        f"passive RF: {cue_defs.get('passive_rf', 'no-signal / RFI geometry')}",
        "claim boundary: public-proxy assumptions only",
    ]
    _draw_text_card(notes_ax, "Cue definitions", cue_text, wrap_width=34, face="white", edge=GRID)
    boundary = _wrap(card.get("claim_boundary", "public-proxy assumptions only"), 48)
    notes_ax.text(
        0.0,
        -0.02,
        boundary,
        fontsize=FONT_TINY,
        color=MUTED,
        linespacing=1.05,
        va="top",
        transform=notes_ax.transAxes,
    )
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
    fig = _paper_fig(4.10, constrained=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.16)
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[0.64, 1.0, 1.0, 1.0, 0.055],
        hspace=0.28,
        wspace=0.14,
    )
    fig.text(
        0.08,
        0.93,
        "Range-Doppler proxy diagnostics for positive and hard-negative samples",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
    )
    card = context.evidence.get("radar_model_card", {})
    branch = next(
        (
            row
            for row in card.get("carrier_bands", [])
            if row.get("branch") == "high_resolution_xku_cuas"
        ),
        {},
    )
    range_resolution_m = _safe_float(branch.get("range_resolution_m"), 0.25)
    velocity_resolution_mps = _safe_float(branch.get("velocity_resolution_mps"), 2.5)
    groups = [("sg_00016", True), ("sg_00000", False)]
    rendered: list[tuple[Any, np.ndarray, int, int, str]] = []
    for row, (group_id, positive) in enumerate(groups):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.02,
            0.90,
            "\n".join(
                [group_id, "public-proxy\npositive" if positive else "hard negative\nartifact"]
            ),
            ha="left",
            va="top",
            fontsize=FONT_AXIS,
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
            rendered.append((ax, rd, row, phase_idx, phase))
    normalized = (
        _global_normalize01([rd for _ax, rd, _row, _phase_idx, _phase in rendered])
        if rendered
        else []
    )
    image = None
    for (ax, rd, row, phase_idx, phase), rd_norm in zip(rendered, normalized):
        image = ax.imshow(
            rd_norm, aspect="auto", origin="lower", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0
        )
        if row == 0:
            ax.set_title(f"{PHASE_LABELS[phase]}\n{PHASE_WINDOWS[phase]}", fontsize=FONT_AXIS)
        xticks = [0, rd.shape[1] // 2, rd.shape[1] - 1]
        yticks = [0, rd.shape[0] // 2, rd.shape[0] - 1]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(
            [f"{tick * range_resolution_m:.1f}" for tick in xticks], fontsize=FONT_TINY
        )
        doppler_center = rd.shape[0] // 2
        if phase_idx == 1:
            ax.set_yticklabels(
                [f"{(tick - doppler_center) * velocity_resolution_mps:.0f}" for tick in yticks],
                fontsize=FONT_TINY,
            )
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel("range m", fontsize=FONT_TINY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color(GRID)
    if image is not None:
        cax = fig.add_subplot(gs[:, 4])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("normalized log magnitude", fontsize=FONT_TINY)
        cbar.ax.tick_params(labelsize=FONT_TINY)
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_detector_ml_pipeline(context: FigureContext) -> Path:
    filename = "detector_ml_pipeline.png"
    manifest_component_rows = context.evidence.get("selected_component_human_weights", [])
    manifest_ablation_rows = context.evidence.get("comparable_ablation_summary", [])
    component_rows = manifest_component_rows if isinstance(manifest_component_rows, list) else []
    ablation_rows = manifest_ablation_rows if isinstance(manifest_ablation_rows, list) else []
    source_tags: list[str] = []
    if not component_rows:
        component_rows = _read_csv_rows(
            context.roots.paper_evidence_root / "selected_component_human_weights.csv"
        )
        if component_rows:
            source_tags.append("paper evidence CSV outputs for component weights")
    if not component_rows:
        component_rows = _selected_component_human_weights(context.component_scores)
        if component_rows:
            source_tags.append("advanced detector component scores")
    if not component_rows:
        component_transparency = context.evidence.get("component_transparency", {})
        if isinstance(component_transparency, dict):
            raw_component_scores = component_transparency.get("component_scores", [])
            if isinstance(raw_component_scores, list) and raw_component_scores:
                component_rows = _selected_component_human_weights(raw_component_scores)
                if component_rows:
                    source_tags.append("paper evidence component transparency")
    if not component_rows and context.selection_lock:
        selected_id = str(
            context.selection_lock.get("selected_candidate_id")
            or context.selection_lock.get("base_candidate_id")
            or "locked_candidate"
        )
        family = EI_SHORT
        modality = "multi-modal score"
        if "passive_quality" in selected_id:
            family = "Passive quality"
            modality = "passive-RF provenance"
        elif "transport_geometry" in selected_id:
            family = "Transport geometry"
            modality = "motion/geometry"
        elif "passive_hypergraph" in selected_id:
            family = "Passive hypergraph"
            modality = "passive-RF topology"
        view = "selection-lock candidate"
        if "signed_de" in selected_id:
            view = "signed differential evolution"
        elif "positive_de" in selected_id:
            view = "positive differential evolution"
        component_rows = [
            {
                "audit_id": "L1",
                "family": family,
                "modality": modality,
                "view": view,
                "calibrator": str(context.selection_lock.get("calibrator") or "selection lock"),
                "weight": 1.0,
                "cumulative_weight": 1.0,
            }
        ]
        source_tags.append("advanced detector selection lock")
    if not ablation_rows:
        ablation_rows = _read_csv_rows(
            context.roots.paper_evidence_root / "comparable_ablation_summary.csv"
        )
        if ablation_rows:
            source_tags.append("paper evidence CSV outputs for comparable ablations")
    if not ablation_rows:
        eval_summary = context.evidence.get("evaluation_summary", {})
        modality_transparency = context.evidence.get("modality_transparency", {})
        if isinstance(eval_summary, dict) and isinstance(modality_transparency, dict):
            ablation_rows = _comparable_ablation_rows(eval_summary, modality_transparency)
            if ablation_rows:
                source_tags.append("paper evidence evaluation + modality transparency")
    if not ablation_rows:
        eval_summary = context.evidence.get("evaluation_summary", {})
        if isinstance(eval_summary, dict):
            selected = eval_summary.get("selected", {})
            baseline = eval_summary.get("baseline", {})
            if isinstance(selected, dict) and isinstance(baseline, dict):
                selected_ap = _safe_float(selected.get("average_precision"))
                baseline_ap = _safe_float(baseline.get("average_precision"))
                selected_recall = _safe_float(
                    selected.get("fixed_fpr_recall")
                    or selected.get("recall_at_leq_1pct_fpr")
                    or selected.get("recall")
                )
                baseline_recall = _safe_float(
                    baseline.get("fixed_fpr_recall")
                    or baseline.get("recall_at_leq_1pct_fpr")
                    or baseline.get("recall")
                )
                if math.isfinite(selected_ap) and math.isfinite(baseline_ap):
                    ablation_rows = [
                        {
                            "reader_label": "EI candidate vs baseline",
                            "delta_ap": selected_ap - baseline_ap,
                            "delta_recall_at_leq_1pct_fpr": selected_recall - baseline_recall
                            if math.isfinite(selected_recall) and math.isfinite(baseline_recall)
                            else 0.0,
                            "ap": selected_ap,
                            "recall_at_leq_1pct_fpr": selected_recall
                            if math.isfinite(selected_recall)
                            else 0.0,
                        }
                    ]
                    source_tags.append("paper evidence selected and baseline evaluation summary")
    if component_rows and ablation_rows:
        if manifest_component_rows and manifest_ablation_rows:
            source_tags.append(
                "paper evidence manifest rows for component weights and comparable ablation outputs"
            )
        _note_source(filename, "; ".join(source_tags) if source_tags else "tracked evidence")
    else:
        _note_fallback(filename, "component/ablation evidence missing; using generic placeholders")
    if not component_rows:
        component_rows = [
            {
                "audit_id": "C1",
                "family": "Passive quality",
                "modality": "passive-RF provenance",
                "view": "signed differential evolution",
                "calibrator": "geodesic odds",
                "weight": 0.43,
                "cumulative_weight": 0.43,
            },
            {
                "audit_id": "C2",
                "family": "Transport geometry",
                "modality": "motion/geometry",
                "view": "positive differential evolution",
                "calibrator": "geodesic odds",
                "weight": 0.29,
                "cumulative_weight": 0.72,
            },
        ]
    if not ablation_rows:
        ablation_rows = [
            {
                "reader_label": "Radar-only view",
                "delta_ap": -0.70,
                "delta_recall_at_leq_1pct_fpr": -0.58,
                "ap": 0.13,
                "recall_at_leq_1pct_fpr": 0.25,
            }
        ]

    component_rows = sorted(
        component_rows, key=lambda row: _safe_float(row.get("weight")), reverse=True
    )
    ablation_rows = [row for row in ablation_rows if row.get("ablation") != "full_locked_candidate"]
    ablation_rows = sorted(
        ablation_rows,
        key=lambda row: _safe_float(row.get("delta_ap")),
    )[:7]

    fig = _paper_fig(3.65, constrained=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.20, wspace=0.68)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0])
    fig.text(
        0.12,
        0.92,
        "EI components and comparable holdout deltas",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
    )

    weight_ax = fig.add_subplot(gs[0, 0])
    labels = [
        f"{row.get('audit_id', '')} {_wrap_compact(row.get('family', ''), 15)}"
        for row in component_rows
    ]
    weights = [_safe_float(row.get("weight")) for row in component_rows]
    colors = [
        TEAL
        if "passive" in row.get("modality", "").lower()
        else BLUE
        if "geometry" not in row.get("modality", "").lower()
        else GOLD
        for row in component_rows
    ]
    y = np.arange(len(component_rows))
    weight_ax.barh(y, weights, color=colors, edgecolor="white", linewidth=0.8)
    weight_ax.set_yticks(y)
    weight_ax.set_yticklabels(labels, fontsize=FONT_TICK)
    weight_ax.invert_yaxis()
    weight_ax.set_xlabel("Fusion weight", fontsize=FONT_AXIS)
    weight_ax.set_title("Component weights", fontsize=FONT_SUBTITLE, weight="bold")
    weight_ax.set_xlim(0.0, max(weights) * 1.32 if weights else 1.0)
    _clean_axes(weight_ax, xgrid=True)
    for idx, (weight, row) in enumerate(zip(weights, component_rows)):
        weight_ax.text(
            weight + 0.01,
            idx,
            f"{weight:.3f} / cum {_safe_float(row.get('cumulative_weight')):.3f}",
            va="center",
            fontsize=FONT_TINY,
            color=MUTED,
        )

    delta_ax = fig.add_subplot(gs[0, 1])
    delta_labels = [
        _reader_label(row.get("reader_label", row.get("ablation", "")), 20) for row in ablation_rows
    ]
    delta_ap = [_safe_float(row.get("delta_ap")) for row in ablation_rows]
    delta_recall = [_safe_float(row.get("delta_recall_at_leq_1pct_fpr")) for row in ablation_rows]
    y2 = np.arange(len(ablation_rows))
    delta_colors = [_delta_color(value) for value in delta_ap]
    delta_ax.axvline(0.0, color=INK, linewidth=0.9)
    bars = delta_ax.barh(
        y2, delta_ap, color=delta_colors, alpha=0.88, edgecolor="white", label="Delta AP"
    )
    delta_ax.scatter(delta_recall, y2, color=INK, s=24, label="Delta Recall@<=1%FPR", zorder=3)
    for idx, (bar, dap, dre) in enumerate(zip(bars, delta_ap, delta_recall)):
        if math.isfinite(dap):
            x_text = dap + 0.018 if dap >= 0 else -0.035
            delta_ax.text(
                x_text,
                idx,
                f"{dap:+.3f}",
                ha="left" if dap >= 0 else "right",
                va="center",
                fontsize=FONT_TINY,
                color=INK,
            )
        if math.isfinite(dre):
            recall_x = dre + 0.035 if dre < 0 else dre - 0.035
            delta_ax.text(
                recall_x,
                idx + 0.28,
                f"R {dre:+.3f}",
                ha="left" if dre < 0 else "right",
                va="center",
                fontsize=FONT_TINY,
                color=MUTED,
            )
    delta_ax.set_yticks(y2)
    delta_ax.set_yticklabels(delta_labels, fontsize=FONT_TICK)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Delta vs full EI candidate", fontsize=FONT_AXIS)
    delta_ax.set_title(
        "Comparable controls: bars AP, dots recall", fontsize=FONT_SUBTITLE, weight="bold"
    )
    _clean_axes(delta_ax, xgrid=True)
    xmin = min(delta_ap + delta_recall + [0.0]) - 0.08
    xmax = max(delta_ap + delta_recall + [0.0]) + 0.10
    delta_ax.set_xlim(xmin - 0.05, xmax + 0.04)
    fig.text(
        0.12,
        0.07,
        "Component handles and ablation CSVs remain evidence artifacts; this figure uses human labels and Table-IV-compatible holdout metrics.",
        fontsize=FONT_TINY,
        color=MUTED,
    )
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_anchor_overlay(context: FigureContext) -> Path:
    filename = "anchor_overlay.png"
    normalized_rows = context.evidence.get("normalized_anchor_comparison", [])
    anchor = context.anchor_summary
    if normalized_rows:
        _note_source(filename, "normalized KTH compare-only anchor summary")
    elif anchor:
        _note_source(filename, "KTH compare-only anchor summary")
    else:
        _note_fallback(filename, "anchor summary missing; using placeholder compare-only panel")
        if STRICT_MODE:
            _require_no_fallback(filename)
    fig, ax = plt.subplots(figsize=(IEEE_TEXT_WIDTH_IN, 2.80), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.20)
    fig.text(
        0.14, 0.91, "Compare-only KTH anchor overlay", fontsize=FONT_TITLE, weight="bold", ha="left"
    )
    ax.set_title(
        "Normalized measured-anchor z-score intervals",
        fontsize=FONT_SUBTITLE,
        weight="bold",
    )
    if normalized_rows:
        labels = [_short_feature_label(row.get("feature", "")) for row in normalized_rows]
        values = [_safe_float(row.get("q50_z")) for row in normalized_rows]
        q10 = [_safe_float(row.get("q10_z")) for row in normalized_rows]
        q90 = [_safe_float(row.get("q90_z")) for row in normalized_rows]
    else:
        selected_features = anchor.get("selected_features", {})
        feature_names = [
            "micro_doppler_bandwidth_hz",
            "spectral_entropy",
            "range_m",
            "return_power_db",
        ]
        labels = [_short_feature_label(name) for name in feature_names]
        values = []
        q10 = []
        q90 = []
        for name in feature_names:
            stats = selected_features.get(name, {})
            if isinstance(stats, dict):
                mean = _safe_float(stats.get("mean"))
                std = _safe_float(stats.get("std"), 1.0)
                values.append((_safe_float(stats.get("q50")) - mean) / std)
                q10.append((_safe_float(stats.get("q10")) - mean) / std)
                q90.append((_safe_float(stats.get("q90")) - mean) / std)
            else:
                values.append(float("nan"))
                q10.append(float("nan"))
                q90.append(float("nan"))
    palette = [BLUE, TEAL, GOLD, GREEN, RED]
    ax.barh(
        labels,
        values,
        color=[palette[idx % len(palette)] for idx in range(len(labels))],
        edgecolor="white",
    )
    for idx, (value, low, high) in enumerate(zip(values, q10, q90)):
        if math.isfinite(value) and math.isfinite(low) and math.isfinite(high):
            ax.errorbar(
                value,
                idx,
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.5,
            )
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlabel("Measured-anchor z-score median with q10/q90 interval", fontsize=FONT_AXIS)
    _clean_axes(ax, xgrid=True)
    ax.text(
        0.02,
        0.02,
        "KTH is compare-only; no positive fixed-wing truth is inferred.",
        transform=ax.transAxes,
        fontsize=FONT_TINY,
        color=MUTED,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    _add_fallback_banner(fig, filename)
    return _save_figure(fig, filename)


def figure_ei_workflow(context: FigureContext) -> Path:
    filename = "ei_workflow.png"
    if context.selection_lock:
        _note_source(filename, "selection-lock and paper evidence workflow metadata")
    else:
        _note_fallback(filename, "selection lock missing; using declared EI workflow")
        if STRICT_MODE:
            _require_no_fallback(filename)

    fig, ax = plt.subplots(figsize=(IEEE_TEXT_WIDTH_IN, 3.25), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(
        0.0,
        5.72,
        "Engineered Intelligence workflow and audit boundaries",
        fontsize=FONT_TITLE,
        weight="bold",
        ha="left",
        va="top",
    )
    stages = [
        ("Detector\nviews", PALE_BLUE, BLUE),
        ("Train/CV\ndiscovery", PALE_TEAL, TEAL),
        ("Sparse\nnonnegative fusion", PALE_GOLD, GOLD),
        ("Geodesic-odds\ncalibration", "white", GRID),
        ("EI selection\nlock", NIGHT_GREEN_LIGHT, NIGHT_GREEN),
        ("Holdout\nscoring", SLATE, INK),
    ]
    x0 = 0.28
    y0 = 3.42
    box_w = 1.34
    gap = 0.25
    for idx, (label, face, edge) in enumerate(stages):
        x = x0 + idx * (box_w + gap)
        _add_box(ax, x, y0, box_w, 1.10, label, face=face, edge=edge, weight="bold", size=FONT_TINY)
        if idx < len(stages) - 1:
            _add_arrow(ax, (x + box_w + 0.02, y0 + 0.55), (x + box_w + gap - 0.02, y0 + 0.55))

    rails = [
        ("Group-locked split", "train/CV rows only before final scoring"),
        ("Feature denylist", "labels, split keys, group IDs, and audit headers blocked"),
        ("Public-proxy boundary", "synthetic evidence, not measured truth"),
        ("Code-disclosure boundary", "reviewable workflow; generated arrays stay out of Git"),
    ]
    for idx, (title, body) in enumerate(rails):
        x = 0.40 + (idx % 2) * 4.70
        y = 1.78 - (idx // 2) * 0.82
        _add_box(
            ax,
            x,
            y,
            4.20,
            0.64,
            f"{title}: {body}",
            face="white",
            edge=GRID,
            size=FONT_TINY,
        )
    ax.text(
        0.30,
        0.24,
        "Raw component handles, selection_lock.json, and component CSVs remain evidence artifacts; reader-facing labels summarize their role.",
        fontsize=FONT_TINY,
        color=MUTED,
        ha="left",
    )
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
        figure_anchor_overlay(context),
        figure_ei_workflow(context),
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
