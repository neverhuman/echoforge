"""Advanced evolution lane for the public-proxy main-run detector corpus.

This module is intentionally dependency-free beyond NumPy. It builds a
separate experimental detector lane from the current main-run records, raw IQ,
acoustic streams, passive RF cues, and detector-view CSVs. Candidate selection,
calibration, and thresholding use train/CV rows only; holdout is scored once for
the single selected winner after the internal-CV choice is locked.

The feature lifts are clean-room math proxies inspired by public signal
processing concepts: spectral summaries, DMD/Koopman-style residuals, covariance
shrinkage diagnostics, wavelet-packet energy, topology proxies, passive RF
quality geometry, and hypergraph-style cross-modal products. The outputs are
synthetic public-proxy evidence only.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from detection.main_run_detectors import (
        _acoustic_score,
        _average_precision,
        _binary_metrics,
        _build_performance_reports,
        _cfar_score,
        _fit_gaussian_log_odds,
        _float_or_none,
        _format_metric,
        _mtd_score,
        _predict_log_odds,
        _read_csv,
        _roc_auc,
        _select_threshold,
        _sigmoid,
        _track_score,
        _write_csv,
        _write_json,
    )
    from detection.main_run_types import (
        ACOUSTIC_VIEW_ID,
        ACTIVE_RADAR_SENSORS,
        DATASET_PROFILE,
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        DEFAULT_SCENARIO_GROUPS,
        FUSION_VIEW_ID,
        MODEL_FEATURE_DENYLIST,
        PHASES,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_detectors import (
        _acoustic_score,
        _average_precision,
        _binary_metrics,
        _build_performance_reports,
        _cfar_score,
        _fit_gaussian_log_odds,
        _float_or_none,
        _format_metric,
        _mtd_score,
        _predict_log_odds,
        _read_csv,
        _roc_auc,
        _select_threshold,
        _sigmoid,
        _track_score,
        _write_csv,
        _write_json,
    )
    from main_run_types import (
        ACOUSTIC_VIEW_ID,
        ACTIVE_RADAR_SENSORS,
        DATASET_PROFILE,
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        DEFAULT_SCENARIO_GROUPS,
        FUSION_VIEW_ID,
        MODEL_FEATURE_DENYLIST,
        PHASES,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


ADVANCED_METHOD_ID = "spectral_transport_hypergraph_fusion"
ADVANCED_OUTPUT_PROFILE = "runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution"
EVOLUTION_TRACE_SCHEMA_VERSION = "ei-evolution-trace-v1"
EVOLUTION_TRACE_SELECTION_SPLIT = "train_cv"
BASELINE_TARGETS = {
    "layered_fusion_c2": {
        "train_cv": {"average_precision": 0.083745, "roc_auc": 0.918501, "f1": 0.184874},
        "holdout": {"average_precision": 0.140272, "roc_auc": 0.939483, "f1": 0.215385},
    }
}
CALIBRATORS = ("raw", "beta", "geodesic_odds", "monotone_binning")
PHASE_ORDER = {phase.phase_id: idx for idx, phase in enumerate(PHASES)}
ADVANCED_FEATURE_CACHE_VERSION = "advanced-v2-aggressive-20260521"


@dataclass(frozen=True)
class SearchProfile:
    name: str
    enable_v2_features: bool
    fusion_top_ks: tuple[int, ...]
    fusion_pool_size: int
    min_candidate_limit: int


SEARCH_PROFILES = {
    "smoke": SearchProfile(
        name="smoke",
        enable_v2_features=False,
        fusion_top_ks=(4, 8),
        fusion_pool_size=14,
        min_candidate_limit=96,
    ),
    "balanced": SearchProfile(
        name="balanced",
        enable_v2_features=False,
        fusion_top_ks=(4, 8, 12),
        fusion_pool_size=24,
        min_candidate_limit=128,
    ),
    "v2_aggressive": SearchProfile(
        name="v2_aggressive",
        enable_v2_features=True,
        fusion_top_ks=(4, 8, 12, 16),
        fusion_pool_size=32,
        min_candidate_limit=128,
    ),
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    subset_name: str
    head: str
    feature_indices: tuple[int, ...]


@dataclass(frozen=True)
class MetaFusionSpec:
    candidate_id: str
    component_ids: tuple[str, ...]
    weights: tuple[float, ...]
    optimizer: str
    top_k: int


@dataclass(frozen=True)
class HeadModel:
    indices: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float


@dataclass(frozen=True)
class CalibrationModel:
    kind: str
    transform: Callable[[np.ndarray], np.ndarray]
    info: dict[str, Any]


@dataclass
class AdvancedData:
    records: list[dict[str, str]]
    y: np.ndarray
    split: np.ndarray
    cv_fold: np.ndarray
    phases: np.ndarray
    scenario_groups: np.ndarray
    feature_names: list[str]
    features: np.ndarray
    branch_indices: dict[str, list[int]]
    surface_scores: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _trace_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace_sort_tuple(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        _safe_float(row.get("train_cv_objective", row.get("objective")), float("-inf")),
        _safe_float(row.get("train_cv_average_precision"), float("-inf")),
        _safe_float(row.get("train_cv_roc_auc"), float("-inf")),
        str(row.get("candidate_id", "")),
    )


def _trace_generation(stage: str) -> int:
    return {
        "base_candidate_search": 0,
        "surface_control": 1,
        "meta_fusion_search": 2,
    }.get(stage, 3)


def _make_evolution_trace_row(
    *,
    candidate_index: int,
    stage: str,
    row: dict[str, Any],
    running_best: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    return {
        "schema_version": EVOLUTION_TRACE_SCHEMA_VERSION,
        "candidate_index": candidate_index,
        "generation": _trace_generation(stage),
        "stage": stage,
        "candidate_id": candidate_id,
        "base_candidate_id": str(row.get("base_candidate_id", candidate_id.rsplit(".", 1)[0])),
        "candidate_type": str(row.get("candidate_type", "advanced_candidate")),
        "family": str(row.get("family", "")),
        "subset_name": str(row.get("subset_name", "")),
        "head": str(row.get("head", "")),
        "calibrator": str(row.get("calibrator", "")),
        "feature_count": _trace_int(row, "feature_count"),
        "component_count": _trace_int(row, "component_count", _trace_int(row, "feature_count")),
        "selection_split": str(row.get("selection_split", EVOLUTION_TRACE_SELECTION_SPLIT)),
        "holdout_rows_used_for_selection": _trace_int(row, "holdout_rows_used_for_selection"),
        "train_cv_objective": _safe_float(row.get("objective")),
        "train_cv_average_precision": _safe_float(row.get("train_cv_average_precision")),
        "train_cv_roc_auc": _safe_float(row.get("train_cv_roc_auc")),
        "train_cv_f1": _safe_float(row.get("train_cv_f1")),
        "train_cv_false_positive_rate": _safe_float(row.get("train_cv_false_positive_rate")),
        "initial_take_up_average_precision": _safe_float(
            row.get("initial_take_up_average_precision")
        ),
        "threshold": _safe_float(row.get("threshold")),
        "running_best_candidate_id": str(running_best.get("candidate_id", candidate_id)),
        "running_best_objective": _safe_float(
            running_best.get("train_cv_objective", running_best.get("objective")),
            _safe_float(row.get("objective")),
        ),
        "running_best_average_precision": _safe_float(
            running_best.get("train_cv_average_precision"),
            _safe_float(row.get("train_cv_average_precision")),
        ),
        "running_best_roc_auc": _safe_float(
            running_best.get("train_cv_roc_auc"),
            _safe_float(row.get("train_cv_roc_auc")),
        ),
        "selected_by_cv": False,
        "final_selected": False,
        "trace_basis": "true_evaluation_order",
        "paper_note": "train/CV-only search point; holdout is not used for selection",
    }


def _brier_score(y: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) == 0:
        return float("nan")
    return float(np.mean((values - labels) ** 2))


def _expected_calibration_error(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    labels = np.asarray(y, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(labels))
    ece = 0.0
    rows: list[dict[str, float]] = []
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
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
        mean_score = float(np.mean(values[mask]))
        empirical = float(np.mean(labels[mask]))
        gap = abs(empirical - mean_score)
        ece += gap * (count / total)
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


def _entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log2(p + 1e-12)) / max(math.log2(len(p) + 1e-12), 1e-12))


def _haar_packet_energy(values: np.ndarray, *, levels: int = 3) -> tuple[np.ndarray, float]:
    packets = [np.asarray(values, dtype=np.float64).reshape(-1)]
    for _ in range(levels):
        next_packets: list[np.ndarray] = []
        for packet in packets:
            if len(packet) < 2:
                next_packets.extend([packet, packet * 0.0])
                continue
            even_len = (len(packet) // 2) * 2
            paired = packet[:even_len].reshape(-1, 2)
            avg = paired.mean(axis=1)
            diff = 0.5 * (paired[:, 0] - paired[:, 1])
            next_packets.extend([avg, diff])
        packets = next_packets
    energy = np.asarray([float(np.mean(np.square(packet))) for packet in packets], dtype=np.float64)
    return energy, _entropy(energy)


def _topology_proxy(series: np.ndarray) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) < 4:
        return {"nn_scale": 0.0, "loop_area": 0.0, "persistence_spread": 0.0}
    emb = np.column_stack([values[:-2], values[1:-1], values[2:]])
    if len(emb) > 32:
        emb = emb[np.linspace(0, len(emb) - 1, 32).astype(np.int64)]
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    dist += np.eye(len(emb)) * 1e9
    nn = np.min(dist, axis=1)
    cov = np.cov(emb.T)
    eig = np.sort(np.linalg.eigvalsh(cov + np.eye(cov.shape[0]) * 1e-9))[::-1]
    loop_area = math.sqrt(max(float(eig[0] * eig[1]), 0.0)) if len(eig) > 1 else 0.0
    return {
        "nn_scale": float(np.median(nn)),
        "loop_area": loop_area,
        "persistence_spread": float(np.percentile(nn, 90) - np.percentile(nn, 10)),
    }


def _dmd_summary(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim == 1:
        x = matrix[:-1]
        y = matrix[1:]
    else:
        x = matrix[:-1].reshape(matrix.shape[0] - 1, -1)
        y = matrix[1:].reshape(matrix.shape[0] - 1, -1)
    denom = np.vdot(x, x) + 1e-9
    alpha = np.vdot(x, y) / denom
    residual = np.linalg.norm(y - alpha * x) / (np.linalg.norm(y) + 1e-9)
    return {
        "growth": float(np.abs(alpha)),
        "frequency": float(np.angle(alpha)),
        "residual": float(np.real(residual)),
    }


def _resolve_search_profile(search_profile: str | SearchProfile) -> SearchProfile:
    if isinstance(search_profile, SearchProfile):
        return search_profile
    if search_profile not in SEARCH_PROFILES:
        choices = ", ".join(sorted(SEARCH_PROFILES))
        raise ValueError(f"unknown search_profile {search_profile!r}; expected one of {choices}")
    return SEARCH_PROFILES[search_profile]


def _lag_decorrelation_summary(series: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) < 4:
        return {
            f"{prefix}_lag1_abs_corr": 0.0,
            f"{prefix}_lag2_abs_corr": 0.0,
            f"{prefix}_lag4_abs_corr": 0.0,
            f"{prefix}_lag8_abs_corr": 0.0,
            f"{prefix}_multi_lag_decorrelation": 0.0,
        }
    values = values - float(np.mean(values))
    denom = float(np.dot(values, values)) + 1e-9
    result: dict[str, float] = {}
    corrs = []
    for lag in (1, 2, 4, 8):
        if len(values) <= lag:
            corr = 0.0
        else:
            corr = float(abs(np.dot(values[:-lag], values[lag:]) / denom))
        result[f"{prefix}_lag{lag}_abs_corr"] = corr
        corrs.append(corr)
    result[f"{prefix}_multi_lag_decorrelation"] = float(1.0 - np.mean(corrs))
    return result


def _hankel_ssa_summary(series: np.ndarray, prefix: str, *, width: int = 8) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) < 2 * width:
        width = max(2, len(values) // 3)
    if width < 2 or len(values) <= width:
        return {
            f"{prefix}_ssa_rank_entropy": 0.0,
            f"{prefix}_ssa_top_ratio": 0.0,
            f"{prefix}_ssa_tail_mass": 0.0,
        }
    rows = len(values) - width + 1
    hankel = np.column_stack([values[idx : idx + rows] for idx in range(width)])
    cov = np.cov((hankel - np.mean(hankel, axis=0)).T)
    eig = np.sqrt(np.maximum(np.linalg.eigvalsh(cov + np.eye(width) * 1e-9), 0.0))[::-1]
    mass = eig / (float(np.sum(eig)) + 1e-9)
    return {
        f"{prefix}_ssa_rank_entropy": _entropy(mass),
        f"{prefix}_ssa_top_ratio": float(mass[0]) if len(mass) else 0.0,
        f"{prefix}_ssa_tail_mass": float(np.sum(mass[3:])) if len(mass) > 3 else 0.0,
    }


def _morlet_scattering_summary(series: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) < 8:
        return {
            f"{prefix}_morlet_energy_mean": 0.0,
            f"{prefix}_morlet_energy_entropy": 0.0,
            f"{prefix}_morlet_peak_scale": 0.0,
            f"{prefix}_morlet_second_order": 0.0,
        }
    values = (values - float(np.median(values))) / (
        float(np.percentile(values, 75) - np.percentile(values, 25)) + 1e-6
    )
    energies = []
    second_order = []
    for scale in (2.0, 4.0, 8.0, 16.0):
        radius = max(3, int(round(3.0 * scale)))
        t = np.arange(-radius, radius + 1, dtype=np.float64)
        carrier = np.exp(1j * 5.0 * t / scale)
        envelope = np.exp(-0.5 * (t / scale) ** 2)
        kernel = carrier * envelope
        kernel -= np.mean(kernel)
        norm = math.sqrt(float(np.sum(np.abs(kernel) ** 2))) + 1e-9
        response = np.convolve(values, kernel.real / norm, mode="same") + 1j * np.convolve(
            values, kernel.imag / norm, mode="same"
        )
        modulus = np.abs(response)
        energies.append(float(np.mean(modulus * modulus)))
        second_order.append(float(np.std(np.diff(modulus))) if len(modulus) > 1 else 0.0)
    energy = np.asarray(energies, dtype=np.float64)
    return {
        f"{prefix}_morlet_energy_mean": math.log1p(float(np.mean(energy))),
        f"{prefix}_morlet_energy_entropy": _entropy(energy),
        f"{prefix}_morlet_peak_scale": float((int(np.argmax(energy)) + 1) / len(energy)),
        f"{prefix}_morlet_second_order": math.log1p(float(np.mean(second_order))),
    }


def _diffusion_laplacian_summary(series: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) < 8:
        return {
            f"{prefix}_diffusion_eigengap": 0.0,
            f"{prefix}_diffusion_connectivity": 0.0,
            f"{prefix}_diffusion_trace": 0.0,
        }
    emb = np.column_stack([values[:-3], values[1:-2], values[2:-1], values[3:]])
    emb = (emb - np.mean(emb, axis=0)) / (np.std(emb, axis=0) + 1e-6)
    cov = np.cov(emb.T)
    # A compact diffusion-Laplacian proxy: covariance-derived affinities between
    # lag coordinates, normalized as a tiny graph operator.
    kernel = np.exp(-np.square(cov - np.median(cov)) / (np.var(cov) + 1e-6))
    degree = np.sum(kernel, axis=1)
    normalized = kernel / np.sqrt(np.outer(degree, degree) + 1e-9)
    eig = np.sort(np.linalg.eigvalsh(normalized))[::-1]
    gap = float(eig[1] - eig[2]) if len(eig) > 2 else 0.0
    return {
        f"{prefix}_diffusion_eigengap": gap,
        f"{prefix}_diffusion_connectivity": float(eig[1]) if len(eig) > 1 else 0.0,
        f"{prefix}_diffusion_trace": float(np.sum(eig[: min(6, len(eig))])),
    }


def _v2_signal_lift(prefix: str, series: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    result.update(_lag_decorrelation_summary(series, prefix))
    result.update(_hankel_ssa_summary(series, prefix))
    result.update(_morlet_scattering_summary(series, prefix))
    result.update(_diffusion_laplacian_summary(series, prefix))
    return result


def _radar_lift(prefix: str, iq: np.ndarray, *, aggressive: bool = False) -> dict[str, float]:
    centered = iq - (np.median(iq.real) + 1j * np.median(iq.imag))
    robust_scale = np.median(np.abs(centered)) + 1e-6
    normalized = centered / robust_scale
    power = np.abs(normalized) ** 2
    pulse_power = power.mean(axis=1)
    range_power = power.mean(axis=0)
    rd = np.fft.fftshift(np.fft.fft2(normalized))
    rd_power = np.abs(rd) ** 2
    cov = np.cov(normalized.real.T) + np.cov(normalized.imag.T)
    eig = np.sort(np.linalg.eigvalsh(cov + np.eye(cov.shape[0]) * 1e-8))[::-1]
    gamma = float(iq.shape[1]) / max(float(iq.shape[0]), 1.0)
    noise_floor = float(np.median(eig) * (1.0 + math.sqrt(gamma)) ** 2 + 1e-9)
    packet_energy, packet_entropy = _haar_packet_energy(power)
    dmd = _dmd_summary(normalized)
    topo = _topology_proxy(pulse_power)
    phase = np.angle(normalized + 1e-9)
    phase_step = np.diff(phase, axis=0)
    doppler_bins = np.arange(rd_power.shape[0], dtype=np.float64)
    doppler_mass = rd_power.sum(axis=1) + 1e-9
    doppler_centroid = float(np.sum(doppler_bins * doppler_mass) / np.sum(doppler_mass))
    features = {
        f"{prefix}_robust_log_power": math.log1p(float(np.mean(power))),
        f"{prefix}_rd_peak_ratio": math.log1p(float(np.max(rd_power) / (np.mean(rd_power) + 1e-9))),
        f"{prefix}_rd_entropy": _entropy(rd_power),
        f"{prefix}_doppler_centroid": doppler_centroid / max(float(len(doppler_bins) - 1), 1.0),
        f"{prefix}_range_contrast": math.log1p(
            float(np.max(range_power) / (np.mean(range_power) + 1e-9))
        ),
        f"{prefix}_phase_coherence": float(np.abs(np.mean(np.exp(1j * phase_step)))),
        f"{prefix}_cov_top_eigen": math.log1p(float(eig[0])),
        f"{prefix}_cov_eigengap": float((eig[0] - eig[1]) / (eig[0] + 1e-9))
        if len(eig) > 1
        else 0.0,
        f"{prefix}_mp_excess": math.log1p(float(np.sum(np.maximum(eig - noise_floor, 0.0)))),
        f"{prefix}_wavelet_low_ratio": float(packet_energy[0] / (np.sum(packet_energy) + 1e-9)),
        f"{prefix}_wavelet_entropy": packet_entropy,
        f"{prefix}_dmd_growth": dmd["growth"],
        f"{prefix}_dmd_frequency": dmd["frequency"],
        f"{prefix}_dmd_residual": dmd["residual"],
        f"{prefix}_takens_nn_scale": topo["nn_scale"],
        f"{prefix}_takens_loop_area": topo["loop_area"],
        f"{prefix}_takens_persistence_spread": topo["persistence_spread"],
    }
    if aggressive:
        features.update(_v2_signal_lift(f"{prefix}_pulse_power", pulse_power))
    return features


def _acoustic_lift(acoustic: np.ndarray, *, aggressive: bool = False) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(acoustic, axis=1))
    peak_bin = np.argmax(spectrum[:, 1:], axis=1) + 1
    spectral_mass = spectrum[:, 1:] + 1e-9
    packet_energy, packet_entropy = _haar_packet_energy(acoustic)
    correlations = np.corrcoef(acoustic)
    upper = correlations[np.triu_indices_from(correlations, k=1)]
    dmd = _dmd_summary(acoustic.T)
    topo = _topology_proxy(np.mean(acoustic, axis=0))
    features = {
        "acoustic_wavelet_low_ratio": float(packet_energy[0] / (np.sum(packet_energy) + 1e-9)),
        "acoustic_wavelet_entropy": packet_entropy,
        "acoustic_cadence_stability": float(1.0 / (1.0 + np.std(peak_bin))),
        "acoustic_spectral_entropy": _entropy(spectral_mass),
        "acoustic_cross_node_coherence": float(np.nan_to_num(np.mean(np.abs(upper)), nan=0.0)),
        "acoustic_peak_contrast": math.log1p(
            float(np.mean(np.max(spectral_mass, axis=1) / (np.mean(spectral_mass, axis=1) + 1e-9)))
        ),
        "acoustic_dmd_growth": dmd["growth"],
        "acoustic_dmd_frequency": dmd["frequency"],
        "acoustic_dmd_residual": dmd["residual"],
        "acoustic_takens_nn_scale": topo["nn_scale"],
        "acoustic_takens_loop_area": topo["loop_area"],
    }
    if aggressive:
        features.update(_v2_signal_lift("acoustic_node_mean", np.mean(acoustic, axis=0)))
    return features


def _passive_lift(passive_rf: np.ndarray) -> dict[str, float]:
    no_signal, rfi_burst, clock_offset, provenance_quality = [
        float(value) for value in passive_rf[:4]
    ]
    quality_gap = provenance_quality - 0.5 * no_signal - 0.4 * rfi_burst
    return {
        "passive_no_signal_score": no_signal,
        "passive_rfi_burst_score": rfi_burst,
        "passive_clock_offset_ms": clock_offset,
        "passive_provenance_quality": provenance_quality,
        "passive_quality_gap": quality_gap,
        "passive_clock_rfi_geometry": abs(clock_offset - rfi_burst),
        "passive_no_signal_quality_interaction": no_signal * (1.0 - provenance_quality),
        "passive_rfi_quality_interaction": rfi_burst * (1.0 - provenance_quality),
    }


def _detector_view_feature_rows(data_root: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    blocked = (
        set(DETECTOR_ID_COLUMNS)
        | set(MODEL_FEATURE_DENYLIST)
        | {
            "model_label",
            "label_id",
            "is_positive",
            "sensor_id",
            "sensor_band",
            "source_provenance",
            "view_feature_version",
            "raw_complex_iq_ref",
            "acoustic_stream_ref",
            "passive_rf_ref",
        }
    )
    by_record: dict[str, dict[str, float]] = {}
    used_columns: list[str] = []
    for view_id in DETECTOR_VIEW_IDS:
        path = data_root / "detector_views" / f"{view_id}.csv"
        if not path.exists():
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        columns = [column for column in rows[0] if column not in blocked]
        for row in rows:
            record_features = by_record.setdefault(row["record_id"], {})
            for column in columns:
                name = f"view_{view_id}_{column}"
                record_features[name] = _safe_float(row[column])
                if name not in used_columns:
                    used_columns.append(name)
    return by_record, used_columns


def _load_npz_row(
    data_root: Path,
    row: dict[str, str],
    cache: dict[str, dict[str, np.ndarray]],
    key: str,
) -> np.ndarray:
    shard_path = row["shard_path"]
    if shard_path not in cache:
        with np.load(data_root / shard_path) as loaded:
            cache[shard_path] = {name: loaded[name] for name in loaded.files}
    return cache[shard_path][key][int(row["row_offset"])]


def _append_features(
    values: list[float],
    names: list[str],
    row: dict[str, float],
    *,
    branch: str,
    branch_indices: dict[str, list[int]],
) -> None:
    for name in sorted(row):
        branch_indices.setdefault(branch, []).append(len(names))
        names.append(name)
        values.append(float(row[name]))


def _add_sequence_and_hypergraph_features(data: AdvancedData) -> None:
    names = list(data.feature_names)
    matrix = data.features
    extra_columns: list[np.ndarray] = []
    extra_names: list[str] = []
    extra_branches: dict[str, list[int]] = {}

    def add_column(name: str, values: np.ndarray, branch: str) -> None:
        extra_branches.setdefault(branch, []).append(len(names) + len(extra_names))
        extra_names.append(name)
        extra_columns.append(np.asarray(values, dtype=np.float64))

    for phase_id in sorted(PHASE_ORDER):
        add_column(
            f"phase_gate_{phase_id}",
            (data.phases == phase_id).astype(np.float64),
            "phase_gate",
        )

    surface_names = [
        "surface_high_resolution_xku_cuas",
        "surface_tactical_s_band_aesa",
        "surface_gbad_3d4d_cueing",
        "surface_distributed_acoustic_cue",
        "surface_passive_rf_quality",
    ]
    surface_cols = [names.index(name) for name in surface_names if name in names]
    if surface_cols:
        surface = matrix[:, surface_cols]
        add_column("hypergraph_surface_mean", np.mean(surface, axis=1), "hypergraph")
        add_column("hypergraph_surface_max", np.max(surface, axis=1), "hypergraph")
        add_column("hypergraph_surface_min", np.min(surface, axis=1), "hypergraph")
        add_column("hypergraph_surface_disagreement", np.std(surface, axis=1), "hypergraph")
        add_column(
            "hypergraph_radar_acoustic_product",
            np.sqrt(
                np.maximum(surface[:, 0], 0.0)
                * np.maximum(surface[:, min(3, surface.shape[1] - 1)], 0.0)
            ),
            "hypergraph",
        )

    phase_position = np.asarray([PHASE_ORDER[str(phase)] for phase in data.phases], dtype=np.int64)
    for column in surface_cols:
        values = matrix[:, column]
        prev_delta = np.zeros(len(values), dtype=np.float64)
        next_delta = np.zeros(len(values), dtype=np.float64)
        span = np.zeros(len(values), dtype=np.float64)
        centered = np.zeros(len(values), dtype=np.float64)
        cumulative = np.zeros(len(values), dtype=np.float64)
        for group in sorted(set(data.scenario_groups)):
            idx = np.flatnonzero(data.scenario_groups == group)
            ordered = idx[np.argsort(phase_position[idx])]
            series = values[ordered]
            group_mean = float(np.mean(series))
            group_span = float(np.max(series) - np.min(series))
            for pos, row_idx in enumerate(ordered):
                prev_delta[row_idx] = 0.0 if pos == 0 else float(series[pos] - series[pos - 1])
                next_delta[row_idx] = (
                    0.0 if pos + 1 == len(series) else float(series[pos + 1] - series[pos])
                )
                span[row_idx] = group_span
                centered[row_idx] = float(series[pos] - group_mean)
                cumulative[row_idx] = float(series[pos] - series[0])
        base = names[column].replace("surface_", "sequence_")
        add_column(f"{base}_prev_delta", prev_delta, "sequence")
        add_column(f"{base}_next_delta", next_delta, "sequence")
        add_column(f"{base}_group_span", span, "sequence")
        add_column(f"{base}_group_centered", centered, "sequence")
        add_column(f"{base}_cumulative_delta", cumulative, "sequence")

    if extra_columns:
        data.features = np.column_stack([data.features, *extra_columns])
        data.feature_names.extend(extra_names)
        for branch, idxs in extra_branches.items():
            data.branch_indices.setdefault(branch, []).extend(idxs)


def _add_train_rank_features(data: AdvancedData) -> None:
    train_cv = data.split == "train_cv"
    passive_indices = data.branch_indices.get("passive_rf", [])
    if not passive_indices or not np.any(train_cv):
        return
    extra_columns: list[np.ndarray] = []
    extra_names: list[str] = []
    start = len(data.feature_names)
    for source_idx in passive_indices:
        train_values = np.sort(data.features[train_cv, source_idx])
        if len(train_values) == 0:
            continue
        ranks = np.searchsorted(train_values, data.features[:, source_idx], side="right")
        rank_values = ranks.astype(np.float64) / float(len(train_values))
        extra_names.append(f"passive_train_rank_{data.feature_names[source_idx]}")
        extra_columns.append(rank_values)
    if not extra_columns:
        return
    data.features = np.column_stack([data.features, *extra_columns])
    data.feature_names.extend(extra_names)
    data.branch_indices.setdefault("passive_rank", []).extend(
        range(start, start + len(extra_names))
    )


def _alpha_mean(values: np.ndarray, alpha: float) -> np.ndarray:
    clipped = np.clip(values, 1e-6, None)
    if abs(alpha) < 1e-9:
        return np.exp(np.mean(np.log(clipped), axis=0))
    return np.power(np.mean(np.power(clipped, alpha), axis=0), 1.0 / alpha)


def _add_train_geometry_features(data: AdvancedData) -> None:
    train_cv = data.split == "train_cv"
    if not np.any(train_cv):
        return
    branch_names = [
        "surfaces",
        "radar_high_resolution_xku_cuas",
        "radar_tactical_s_band_aesa",
        "radar_gbad_3d4d_cueing",
        "acoustic",
        "passive_rf",
        "detector_views",
        "sequence",
        "hypergraph",
    ]
    extra_columns: list[np.ndarray] = []
    extra_names: list[str] = []
    start = len(data.feature_names)

    def add(name: str, values: np.ndarray) -> None:
        extra_names.append(name)
        extra_columns.append(np.asarray(values, dtype=np.float64))

    for branch in branch_names:
        raw_indices = data.branch_indices.get(branch, [])
        if not raw_indices:
            continue
        # Keep the train-only geometry compact and deterministic for large runs.
        indices = np.asarray(raw_indices[: min(24, len(raw_indices))], dtype=np.int64)
        train_values = data.features[train_cv][:, indices]
        center, scale = _standardize_fit(train_values)
        z = (data.features[:, indices] - center) / scale
        z_train = z[train_cv]
        y_train = data.y[train_cv]
        pos = z_train[y_train == 1]
        neg = z_train[y_train == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_proto = np.median(pos, axis=0)
        neg_proto = np.median(neg, axis=0)
        pos_dist = np.mean(np.abs(z - pos_proto), axis=1)
        neg_dist = np.mean(np.abs(z - neg_proto), axis=1)
        add(f"v2_{branch}_wasserstein_pos_distance", pos_dist)
        add(f"v2_{branch}_wasserstein_neg_distance", neg_dist)
        add(f"v2_{branch}_prototype_margin", neg_dist - pos_dist)

        rank_columns = []
        for local_idx, source_idx in enumerate(indices):
            sorted_train = np.sort(data.features[train_cv, source_idx])
            ranks = np.searchsorted(
                sorted_train,
                data.features[:, source_idx],
                side="right",
            ).astype(np.float64)
            rank_columns.append(ranks / max(float(len(sorted_train)), 1.0))
            if local_idx >= 7:
                break
        ranks = np.column_stack(rank_columns)
        add(f"v2_{branch}_copula_rank_mean", np.mean(ranks, axis=1))
        add(f"v2_{branch}_copula_rank_spread", np.std(ranks, axis=1))

        shifted = z - np.min(z[train_cv], axis=0) + 1e-3
        pos_alpha = _alpha_mean(shifted[train_cv][y_train == 1], 0.5)
        neg_alpha = _alpha_mean(shifted[train_cv][y_train == 0], 0.5)
        pos_alpha_dist = np.mean(np.abs(shifted - pos_alpha), axis=1)
        neg_alpha_dist = np.mean(np.abs(shifted - neg_alpha), axis=1)
        add(f"v2_{branch}_alpha_mean_margin", neg_alpha_dist - pos_alpha_dist)

    if not extra_columns:
        return
    data.features = np.column_stack([data.features, *extra_columns])
    data.feature_names.extend(extra_names)
    data.branch_indices.setdefault("v2_transport", []).extend(
        range(start, start + len(extra_names))
    )


def _cache_payload(data: AdvancedData, data_root: Path, profile: SearchProfile) -> dict[str, Any]:
    return {
        "cache_version": ADVANCED_FEATURE_CACHE_VERSION,
        "dataset_profile": DATASET_PROFILE,
        "data_root": str(data_root),
        "search_profile": profile.name,
        "record_count": len(data.records),
        "feature_names": data.feature_names,
        "branch_indices": data.branch_indices,
        "manifest": data.manifest,
        "surface_score_names": sorted(data.surface_scores),
    }


def _load_feature_cache(
    feature_cache: Path,
    records: list[dict[str, str]],
    data_root: Path,
    profile: SearchProfile,
) -> AdvancedData | None:
    if not feature_cache.exists():
        return None
    with np.load(feature_cache) as loaded:
        metadata = json.loads(str(loaded["metadata"].item()))
        if (
            metadata.get("cache_version") != ADVANCED_FEATURE_CACHE_VERSION
            or metadata.get("dataset_profile") != DATASET_PROFILE
            or metadata.get("search_profile") != profile.name
            or int(metadata.get("record_count", -1)) != len(records)
        ):
            return None
        surface_names = [str(item) for item in loaded["surface_score_names"]]
        surface_matrix = np.asarray(loaded["surface_scores"], dtype=np.float64)
        return AdvancedData(
            records=records,
            y=np.asarray(loaded["y"], dtype=np.int8),
            split=np.asarray([str(item) for item in loaded["split"]]),
            cv_fold=np.asarray(loaded["cv_fold"], dtype=np.int64),
            phases=np.asarray([str(item) for item in loaded["phases"]]),
            scenario_groups=np.asarray([str(item) for item in loaded["scenario_groups"]]),
            feature_names=list(metadata["feature_names"]),
            features=np.asarray(loaded["features"], dtype=np.float64),
            branch_indices={
                str(k): list(map(int, v)) for k, v in metadata["branch_indices"].items()
            },
            surface_scores={name: surface_matrix[:, idx] for idx, name in enumerate(surface_names)},
            manifest={
                **metadata.get("manifest", {}),
                "feature_cache": str(feature_cache),
                "feature_cache_status": "hit",
            },
        )


def _write_feature_cache(
    feature_cache: Path,
    data: AdvancedData,
    data_root: Path,
    profile: SearchProfile,
) -> None:
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    surface_names = sorted(data.surface_scores)
    surface_matrix = np.column_stack([data.surface_scores[name] for name in surface_names])
    metadata = _cache_payload(data, data_root, profile)
    np.savez_compressed(
        feature_cache,
        metadata=json.dumps(metadata, sort_keys=True),
        features=data.features,
        y=data.y,
        split=data.split.astype(str),
        cv_fold=data.cv_fold,
        phases=data.phases.astype(str),
        scenario_groups=data.scenario_groups.astype(str),
        surface_score_names=np.asarray(surface_names, dtype=str),
        surface_scores=surface_matrix,
    )


def load_advanced_main_run_data(
    data_root: Path,
    *,
    search_profile: str | SearchProfile = "balanced",
    feature_cache: Path | None = None,
) -> AdvancedData:
    profile = _resolve_search_profile(search_profile)
    records = _read_csv(data_root / "records.csv")
    if feature_cache is not None:
        cached = _load_feature_cache(feature_cache, records, data_root, profile)
        if cached is not None:
            return cached
    active_index = {row["record_id"]: row for row in _read_csv(data_root / "raw_stream_index.csv")}
    acoustic_index = {
        row["record_id"]: row for row in _read_csv(data_root / "acoustic_stream_index.csv")
    }
    passive_index = {row["record_id"]: row for row in _read_csv(data_root / "passive_rf_index.csv")}
    view_features, view_columns = _detector_view_feature_rows(data_root)
    active_cache: dict[str, dict[str, np.ndarray]] = {}
    acoustic_cache: dict[str, dict[str, np.ndarray]] = {}
    passive_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[list[float]] = []
    feature_names: list[str] = []
    branch_indices: dict[str, list[int]] = {}
    surface_scores: dict[str, list[float]] = {
        "high_resolution_xku_cuas": [],
        "tactical_s_band_aesa": [],
        "gbad_3d4d_cueing": [],
        ACOUSTIC_VIEW_ID: [],
        "passive_rf_quality_surface": [],
        "tabular_ml_baseline": [],
        "sequence_ml_proxy": [],
    }

    for row_idx, record in enumerate(records):
        iq = _load_npz_row(data_root, active_index[record["record_id"]], active_cache, "iq")
        acoustic = _load_npz_row(
            data_root, acoustic_index[record["record_id"]], acoustic_cache, "acoustic"
        )
        passive_rf = _load_npz_row(
            data_root, passive_index[record["record_id"]], passive_cache, "passive_rf"
        )
        values: list[float] = []
        names: list[str] = []
        local_branches: dict[str, list[int]] = {}

        high = math.log1p(_cfar_score(iq[0]))
        sband = math.log1p(_mtd_score(iq[1]))
        gbad = math.log1p(_track_score(iq[2]))
        acoustic_score = math.log1p(_acoustic_score(acoustic))
        passive_quality = float(passive_rf[3] - passive_rf[1] - 0.25 * passive_rf[0])
        tabular = float(np.mean([high, sband, gbad]))
        sequence = 0.65 * sband + 0.35 * acoustic_score
        surfaces = {
            "surface_high_resolution_xku_cuas": high,
            "surface_tactical_s_band_aesa": sband,
            "surface_gbad_3d4d_cueing": gbad,
            "surface_distributed_acoustic_cue": acoustic_score,
            "surface_passive_rf_quality": passive_quality,
            "surface_tabular_ml_baseline": tabular,
            "surface_sequence_ml_proxy": sequence,
        }
        _append_features(values, names, surfaces, branch="surfaces", branch_indices=local_branches)
        for method, score in [
            ("high_resolution_xku_cuas", high),
            ("tactical_s_band_aesa", sband),
            ("gbad_3d4d_cueing", gbad),
            (ACOUSTIC_VIEW_ID, acoustic_score),
            ("passive_rf_quality_surface", passive_quality),
            ("tabular_ml_baseline", tabular),
            ("sequence_ml_proxy", sequence),
        ]:
            surface_scores[method].append(score)

        for sensor_idx, sensor in enumerate(ACTIVE_RADAR_SENSORS):
            _append_features(
                values,
                names,
                _radar_lift(sensor.view_id, iq[sensor_idx], aggressive=profile.enable_v2_features),
                branch=f"radar_{sensor.view_id}",
                branch_indices=local_branches,
            )
        _append_features(
            values,
            names,
            _acoustic_lift(acoustic, aggressive=profile.enable_v2_features),
            branch="acoustic",
            branch_indices=local_branches,
        )
        _append_features(
            values,
            names,
            _passive_lift(passive_rf),
            branch="passive_rf",
            branch_indices=local_branches,
        )
        _append_features(
            values,
            names,
            view_features.get(record["record_id"], {}),
            branch="detector_views",
            branch_indices=local_branches,
        )

        if row_idx == 0:
            feature_names = names
            branch_indices = local_branches
        rows.append(values)

    features = np.asarray(rows, dtype=np.float64)
    y = np.asarray([int(record["label_id"]) for record in records], dtype=np.int8)
    split = np.asarray([record["split_role"] for record in records])
    cv_fold = np.asarray(
        [-1 if record["cv_fold"] == "" else int(record["cv_fold"]) for record in records],
        dtype=np.int64,
    )
    phases = np.asarray([record["phase_id"] for record in records])
    scenario_groups = np.asarray([record["scenario_group_id"] for record in records])
    surface_arrays = {
        method: np.asarray(scores, dtype=np.float64) for method, scores in surface_scores.items()
    }

    train_cv = split == "train_cv"
    late_features = np.column_stack(
        [
            surface_arrays["high_resolution_xku_cuas"],
            surface_arrays["tactical_s_band_aesa"],
            surface_arrays["gbad_3d4d_cueing"],
            surface_arrays[ACOUSTIC_VIEW_ID],
            surface_arrays["passive_rf_quality_surface"],
        ]
    )
    log_odds = np.zeros(len(records), dtype=np.float64)
    for fold in sorted(set(cv_fold[train_cv])):
        score_mask = train_cv & (cv_fold == fold)
        fit_mask = train_cv & (cv_fold != fold)
        weights, intercept, _info = _fit_gaussian_log_odds(late_features[fit_mask], y[fit_mask])
        log_odds[score_mask] = _predict_log_odds(late_features[score_mask], weights, intercept)
    weights, intercept, _info = _fit_gaussian_log_odds(late_features[train_cv], y[train_cv])
    log_odds[~train_cv] = _predict_log_odds(late_features[~train_cv], weights, intercept)
    surface_arrays[FUSION_VIEW_ID] = _sigmoid(log_odds)

    data = AdvancedData(
        records=records,
        y=y,
        split=split,
        cv_fold=cv_fold,
        phases=phases,
        scenario_groups=scenario_groups,
        feature_names=feature_names,
        features=features,
        branch_indices=branch_indices,
        surface_scores=surface_arrays,
        manifest={
            "raw_streams": ["raw_complex_iq", "acoustic_cues", "passive_rf_cues"],
            "detector_views": list(DETECTOR_VIEW_IDS),
            "detector_view_feature_columns": view_columns,
        },
    )
    _add_train_rank_features(data)
    _add_sequence_and_hypergraph_features(data)
    if profile.enable_v2_features:
        _add_train_geometry_features(data)
    data.manifest.update(
        {
            "search_profile": profile.name,
            "feature_cache_status": "miss" if feature_cache is not None else "not_requested",
        }
    )
    if feature_cache is not None:
        _write_feature_cache(feature_cache, data, data_root, profile)
    return data


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return center, scale


def _oriented_unit_scores(raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    center = float(np.median(raw))
    scale = float(np.percentile(raw, 75) - np.percentile(raw, 25))
    scale = max(scale, 1e-6)
    score = _sigmoid((raw - center) / scale)
    if len(set(y.tolist())) == 2 and float(np.mean(score[y == 1])) < float(np.mean(score[y == 0])):
        score = 1.0 - score
    return np.clip(score, 1e-6, 1.0 - 1e-6)


def _fit_oriented_unit_transform(
    raw: np.ndarray, y: np.ndarray
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    center = float(np.median(raw))
    scale = max(float(np.percentile(raw, 75) - np.percentile(raw, 25)), 1e-6)
    train_score = _sigmoid((raw - center) / scale)
    flipped = False
    if len(set(y.tolist())) == 2 and float(np.mean(train_score[y == 1])) < float(
        np.mean(train_score[y == 0])
    ):
        flipped = True

    def transform(scores: np.ndarray) -> np.ndarray:
        unit = _sigmoid((scores - center) / scale)
        if flipped:
            unit = 1.0 - unit
        return np.clip(unit, 1e-6, 1.0 - 1e-6)

    return transform, {"center": center, "scale": scale, "flipped": flipped}


def _advanced_objective(y: np.ndarray, scores: np.ndarray, phases: np.ndarray) -> dict[str, float]:
    ap = _float_or_none(_average_precision(y, scores)) or 0.0
    auc = _float_or_none(_roc_auc(y, scores)) or 0.5
    threshold, binary = _select_threshold(y, scores)
    initial_mask = phases == "initial_take_up"
    initial_ap = (
        _float_or_none(_average_precision(y[initial_mask], scores[initial_mask]))
        if np.any(initial_mask)
        else None
    )
    initial_value = 0.0 if initial_ap is None else initial_ap
    fpr_penalty = max(0.0, binary["false_positive_rate"] - 0.12) * 0.35
    weak_phase_penalty = max(0.0, ap - initial_value) * 0.20
    objective = ap + 0.25 * auc - fpr_penalty - weak_phase_penalty
    return {
        "objective": float(objective),
        "average_precision": float(ap),
        "roc_auc": float(auc),
        "f1": float(binary["f1"]),
        "false_positive_rate": float(binary["false_positive_rate"]),
        "initial_take_up_average_precision": float(initial_value),
        "threshold": float(threshold),
    }


def _fast_evolution_objective(y: np.ndarray, scores: np.ndarray, phases: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    pos_center = float(np.mean(pos))
    neg_center = float(np.mean(neg))
    pooled = float(np.std(scores)) + 1e-6
    separation = (pos_center - neg_center) / pooled
    tail_penalty = max(0.0, float(np.percentile(neg, 96)) - pos_center) / pooled
    initial_mask = phases == "initial_take_up"
    initial_bonus = 0.0
    if np.any(initial_mask):
        ipos = scores[initial_mask & (y == 1)]
        ineg = scores[initial_mask & (y == 0)]
        if len(ipos) and len(ineg):
            initial_bonus = 0.2 * (float(np.mean(ipos)) - float(np.mean(ineg))) / pooled
    return separation + initial_bonus - 0.35 * tail_penalty


def _effect_order(x: np.ndarray, y: np.ndarray, indices: tuple[int, ...]) -> list[int]:
    if not indices:
        return []
    pos = x[y == 1][:, indices]
    neg = x[y == 0][:, indices]
    if len(pos) == 0 or len(neg) == 0:
        return list(indices)
    var = np.var(x[:, indices], axis=0) + 1e-6
    effect = np.abs((np.mean(pos, axis=0) - np.mean(neg, axis=0)) / np.sqrt(var))
    return [indices[int(idx)] for idx in np.argsort(-effect, kind="mergesort")]


def _evolution_fit_sample(
    y: np.ndarray,
    *,
    seed: int,
    max_rows: int = 4500,
) -> np.ndarray:
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    neg_quota = max_rows - len(positives)
    if neg_quota <= 0:
        return positives[:max_rows]
    sampled_neg = rng.choice(negatives, size=min(neg_quota, len(negatives)), replace=False)
    return np.sort(np.concatenate([positives, sampled_neg]))


def _fit_head_model(
    x: np.ndarray,
    y: np.ndarray,
    phases: np.ndarray,
    indices: tuple[int, ...],
    *,
    head: str,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> HeadModel:
    ordered = _effect_order(x, y, indices)
    max_features = 14 if head == "sparse_signed" else 28
    selected = np.asarray(ordered[:max_features], dtype=np.int64)
    if selected.size == 0:
        selected = np.asarray([0], dtype=np.int64)
    center, scale = _standardize_fit(x[:, selected])
    z = (x[:, selected] - center) / scale
    pos = z[y == 1]
    neg = z[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        weights = np.ones(selected.size, dtype=np.float64) / max(float(selected.size), 1.0)
        intercept = 0.0
    else:
        weights = (np.mean(pos, axis=0) - np.mean(neg, axis=0)) / (np.var(z, axis=0) + 1e-3)
        if head == "positive_de":
            weights = np.abs(weights)
        prior = (float(np.sum(y == 1)) + 0.5) / (float(len(y)) + 1.0)
        midpoint = 0.5 * (np.mean(pos, axis=0) + np.mean(neg, axis=0))
        intercept = math.log(prior / (1.0 - prior)) - float(np.dot(weights, midpoint))

    if head != "centroid":
        sample_idx = _evolution_fit_sample(y, seed=seed, max_rows=evolution_sample_rows)
        zs = z[sample_idx]
        ys = y[sample_idx]
        ps = phases[sample_idx]
        rng = np.random.default_rng(seed)
        pop_size = min(10, max(6, selected.size + 2))
        population = []
        for member in range(pop_size):
            noise_scale = 0.18 + 0.08 * member
            candidate = weights + rng.normal(0.0, noise_scale, size=selected.size)
            if head == "positive_de":
                candidate = np.maximum(candidate, 0.0)
            population.append(candidate)
        best = weights.copy()
        best_score = _fast_evolution_objective(ys, zs @ best + intercept, ps)
        for _ in range(evolution_rounds):
            scores = []
            for candidate in population:
                score = _fast_evolution_objective(ys, zs @ candidate + intercept, ps)
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best = candidate.copy()
            order = np.argsort(scores)[::-1]
            elites = [population[int(idx)] for idx in order[: max(2, pop_size // 3)]]
            population = elites.copy()
            while len(population) < pop_size:
                a, b = rng.choice(len(elites), size=2, replace=True)
                trial = 0.5 * (elites[int(a)] + elites[int(b)])
                trial += rng.normal(0.0, 0.10, size=selected.size)
                if head == "positive_de":
                    trial = np.maximum(trial, 0.0)
                population.append(trial)
        weights = best
    return HeadModel(selected, center, scale, weights, intercept)


def _score_head_model(model: HeadModel, x: np.ndarray) -> np.ndarray:
    z = (x[:, model.indices] - model.center) / model.scale
    return z @ model.weights + model.intercept


def _fit_beta_calibration(raw: np.ndarray, y: np.ndarray) -> CalibrationModel:
    unit_transform, unit_info = _fit_oriented_unit_transform(raw, y)
    p = unit_transform(raw)
    design = np.column_stack([np.log(p), np.log1p(-p), np.ones(len(p))])
    weights = np.zeros(3, dtype=np.float64)
    for _ in range(180):
        pred = _sigmoid(design @ weights)
        grad = design.T @ (pred - y) / max(float(len(y)), 1.0) + 0.01 * weights
        weights -= 0.18 * grad

    def transform(scores: np.ndarray) -> np.ndarray:
        q = unit_transform(scores)
        local = np.column_stack([np.log(q), np.log1p(-q), np.ones(len(q))])
        return _sigmoid(local @ weights)

    return CalibrationModel(
        "beta",
        transform,
        {
            "calibration_family": "beta_logit_ridge",
            "fit_record_count": int(len(y)),
            **unit_info,
        },
    )


def _fit_geodesic_calibration(raw: np.ndarray, y: np.ndarray) -> CalibrationModel:
    center = float(np.median(raw))
    scale = max(float(np.percentile(raw, 75) - np.percentile(raw, 25)), 1e-6)
    z = ((raw - center) / scale)[:, None]
    weights, intercept, info = _fit_gaussian_log_odds(z, y)

    def transform(scores: np.ndarray) -> np.ndarray:
        local = ((scores - center) / scale)[:, None]
        return _sigmoid(_predict_log_odds(local, weights, intercept))

    return CalibrationModel(
        "geodesic_odds",
        transform,
        {"center": center, "scale": scale, "fit_status": info["status"]},
    )


def _fit_monotone_binning(raw: np.ndarray, y: np.ndarray, *, bins: int = 12) -> CalibrationModel:
    unit_transform, unit_info = _fit_oriented_unit_transform(raw, y)
    oriented_raw = unit_transform(raw)
    order = np.argsort(oriented_raw, kind="mergesort")
    sorted_scores = oriented_raw[order]
    sorted_y = y[order]
    edges = np.quantile(sorted_scores, np.linspace(0.0, 1.0, bins + 1))
    rates = []
    for idx in range(bins):
        if idx == bins - 1:
            mask = (sorted_scores >= edges[idx]) & (sorted_scores <= edges[idx + 1])
        else:
            mask = (sorted_scores >= edges[idx]) & (sorted_scores < edges[idx + 1])
        count = int(np.sum(mask))
        pos = int(np.sum(sorted_y[mask] == 1))
        rates.append((pos + 0.5) / (count + 1.0) if count else 0.5)
    # Pool-adjacent-violators in a small deterministic form.
    levels = list(rates)
    weights = [1.0] * len(levels)
    i = 0
    while i < len(levels) - 1:
        if levels[i] <= levels[i + 1]:
            i += 1
            continue
        merged = (levels[i] * weights[i] + levels[i + 1] * weights[i + 1]) / (
            weights[i] + weights[i + 1]
        )
        levels[i] = merged
        weights[i] += weights[i + 1]
        del levels[i + 1]
        del weights[i + 1]
        if i:
            i -= 1
    expanded: list[float] = []
    for level, weight in zip(levels, weights):
        expanded.extend([level] * int(round(weight)))
    while len(expanded) < bins:
        expanded.append(expanded[-1] if expanded else 0.5)
    expanded = expanded[:bins]

    def transform(scores: np.ndarray) -> np.ndarray:
        oriented_scores = unit_transform(scores)
        idx = np.searchsorted(edges[1:-1], oriented_scores, side="right")
        return np.asarray([expanded[int(item)] for item in idx], dtype=np.float64)

    return CalibrationModel(
        "monotone_binning",
        transform,
        {"bin_count": bins, "fit_record_count": int(len(y)), **unit_info},
    )


def _fit_calibration(raw: np.ndarray, y: np.ndarray, kind: str) -> CalibrationModel:
    if kind == "raw":
        transform, info = _fit_oriented_unit_transform(raw, y)
        return CalibrationModel("raw", transform, info)
    if kind == "beta":
        return _fit_beta_calibration(raw, y)
    if kind == "geodesic_odds":
        return _fit_geodesic_calibration(raw, y)
    if kind == "monotone_binning":
        return _fit_monotone_binning(raw, y)
    raise ValueError(f"unknown calibrator {kind}")


def _make_candidate_specs(
    data: AdvancedData,
    *,
    candidate_limit: int,
) -> list[CandidateSpec]:
    branches = data.branch_indices
    high = tuple(branches.get("radar_high_resolution_xku_cuas", []))
    sband = tuple(branches.get("radar_tactical_s_band_aesa", []))
    gbad = tuple(branches.get("radar_gbad_3d4d_cueing", []))
    acoustic = tuple(branches.get("acoustic", []))
    passive = tuple(branches.get("passive_rf", [])) + tuple(branches.get("passive_rank", []))
    views = tuple(branches.get("detector_views", []))
    surfaces = tuple(branches.get("surfaces", []))
    sequence = tuple(branches.get("sequence", []))
    hypergraph = tuple(branches.get("hypergraph", []))
    v2_transport = tuple(branches.get("v2_transport", []))
    phase = tuple(branches.get("phase_gate", []))
    radar_all = high + sband + gbad
    subset_defs = [
        ("high_res_radar", high + surfaces + phase),
        ("sband_radar", sband + surfaces + phase),
        ("gbad_track", gbad + surfaces + phase),
        ("radar_all", radar_all + surfaces + phase),
        ("acoustic_cadence", acoustic + surfaces + phase),
        ("passive_quality", passive + surfaces + phase),
        ("detector_view_stack", views + surfaces + phase),
        ("sequence_path_signature", sequence + surfaces + phase),
        ("hypergraph_edges", hypergraph + surfaces + phase),
        ("radar_acoustic", radar_all + acoustic + surfaces + phase),
        ("passive_hypergraph", passive + hypergraph + surfaces + phase),
        (
            "all_math_lifts",
            radar_all + acoustic + passive + sequence + hypergraph + surfaces + phase,
        ),
        ("v2_transport_geometry", v2_transport + surfaces + phase),
        (
            "v2_aggressive_all",
            radar_all
            + acoustic
            + passive
            + sequence
            + hypergraph
            + v2_transport
            + surfaces
            + phase,
        ),
    ]
    heads = ("signed_de", "positive_de", "centroid", "sparse_signed")
    needed_base_candidates = math.ceil(candidate_limit / len(CALIBRATORS))
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for head in heads:
        for subset_name, raw_indices in subset_defs:
            ordered = tuple(dict.fromkeys(int(idx) for idx in raw_indices))
            if not ordered:
                continue
            base_id = f"{ADVANCED_METHOD_ID}.{subset_name}.{head}"
            if base_id in seen:
                continue
            seen.add(base_id)
            specs.append(
                CandidateSpec(
                    candidate_id=base_id,
                    family=ADVANCED_METHOD_ID,
                    subset_name=subset_name,
                    head=head,
                    feature_indices=ordered,
                )
            )
            if len(specs) >= needed_base_candidates:
                return specs
    if len(specs) < needed_base_candidates:
        all_indices = tuple(range(len(data.feature_names)))
        idx = 0
        while len(specs) < needed_base_candidates:
            family_indices = tuple(
                item
                for pos, item in enumerate(all_indices)
                if (pos + idx) % 5 not in {0, (idx % 4) + 1}
            )
            head = heads[idx % len(heads)]
            specs.append(
                CandidateSpec(
                    candidate_id=f"{ADVANCED_METHOD_ID}.evolved_mesh_{idx:02d}.{head}",
                    family=ADVANCED_METHOD_ID,
                    subset_name=f"evolved_mesh_{idx:02d}",
                    head=head,
                    feature_indices=family_indices,
                )
            )
            idx += 1
    return specs


def _candidate_oof_raw(
    data: AdvancedData,
    spec: CandidateSpec,
    *,
    folds: int,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> np.ndarray:
    train_cv = data.split == "train_cv"
    raw = np.zeros(len(data.records), dtype=np.float64)
    for fold in range(folds):
        fit_mask = train_cv & (data.cv_fold != fold)
        score_mask = train_cv & (data.cv_fold == fold)
        if not np.any(score_mask):
            continue
        model = _fit_head_model(
            data.features[fit_mask],
            data.y[fit_mask],
            data.phases[fit_mask],
            spec.feature_indices,
            head=spec.head,
            seed=_stable_seed(seed, spec.candidate_id, "fold", fold),
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )
        raw[score_mask] = _score_head_model(model, data.features[score_mask])
    return raw


def _score_selected_holdout(
    data: AdvancedData,
    spec: CandidateSpec,
    *,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> np.ndarray:
    train_cv = data.split == "train_cv"
    holdout = data.split == "holdout"
    raw = np.zeros(len(data.records), dtype=np.float64)
    model = _fit_head_model(
        data.features[train_cv],
        data.y[train_cv],
        data.phases[train_cv],
        spec.feature_indices,
        head=spec.head,
        seed=_stable_seed(seed, spec.candidate_id, "holdout-fit"),
        evolution_rounds=evolution_rounds,
        evolution_sample_rows=evolution_sample_rows,
    )
    raw[holdout] = _score_head_model(model, data.features[holdout])
    return raw


def _project_nonnegative_weights(values: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.ones_like(weights) / max(float(len(weights)), 1.0)
    return weights / total


def _optimize_fusion_weights(
    scores: np.ndarray,
    y: np.ndarray,
    phases: np.ndarray,
    *,
    seed: int,
    evolution_rounds: int,
    sample_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("fusion scores must be a non-empty matrix")
    sample_idx = _evolution_fit_sample(y, seed=seed, max_rows=sample_rows)
    sampled_scores = scores[sample_idx]
    sampled_y = y[sample_idx]
    sampled_phases = phases[sample_idx]
    k = scores.shape[1]
    rng = np.random.default_rng(seed)
    pop_size = max(10, min(32, 2 * k + 4))
    base = np.ones(k, dtype=np.float64) / float(k)
    population = [base]
    component_quality = np.asarray(
        [
            _fast_evolution_objective(sampled_y, sampled_scores[:, idx], sampled_phases)
            for idx in range(k)
        ],
        dtype=np.float64,
    )
    prior = _project_nonnegative_weights(component_quality - np.min(component_quality) + 1e-3)
    population.append(prior)
    while len(population) < pop_size:
        concentration = 0.35 + 1.5 * rng.random(k)
        population.append(rng.dirichlet(concentration))

    memory_f = [0.55, 0.75, 0.95]
    memory_cr = [0.35, 0.55, 0.75]
    best = population[0].copy()
    best_score = _fast_evolution_objective(sampled_y, sampled_scores @ best, sampled_phases)
    evaluations = 1
    rounds = max(1, evolution_rounds)
    for round_idx in range(rounds):
        scored = []
        for candidate in population:
            score = _fast_evolution_objective(sampled_y, sampled_scores @ candidate, sampled_phases)
            evaluations += 1
            scored.append(score)
            if score > best_score:
                best_score = score
                best = candidate.copy()
        order = np.argsort(scored)[::-1]
        ranked = [population[int(idx)] for idx in order]
        next_population = ranked[: max(2, pop_size // 4)]
        active_dims = k if round_idx < rounds // 2 else max(2, math.ceil(k / 2))
        elite_dims = np.argsort(best)[::-1][:active_dims]
        while len(next_population) < pop_size:
            a, b, c = rng.choice(len(ranked), size=3, replace=True)
            memory_idx = round_idx % len(memory_f)
            f = float(np.clip(rng.normal(memory_f[memory_idx], 0.08), 0.1, 1.2))
            cr = float(np.clip(rng.normal(memory_cr[memory_idx], 0.08), 0.05, 1.0))
            mutant = ranked[int(a)] + f * (ranked[int(b)] - ranked[int(c)])
            trial = ranked[int(a)].copy()
            mask = rng.random(k) < cr
            mask[elite_dims] = True
            trial[mask] = mutant[mask]
            trial = 0.72 * _project_nonnegative_weights(trial) + 0.28 * best
            next_population.append(_project_nonnegative_weights(trial))
        population = next_population

    return best, {
        "optimizer": "clean_room_shade_multi_resolution_de",
        "population_size": pop_size,
        "rounds": rounds,
        "evaluations": evaluations,
        "sample_rows": int(len(sample_idx)),
        "objective": float(best_score),
    }


def _surface_candidate_scores(
    data: AdvancedData,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    train_cv = data.split == "train_cv"
    scores_by_id: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for method, raw_scores in sorted(data.surface_scores.items()):
        transform, info = _fit_oriented_unit_transform(raw_scores[train_cv], data.y[train_cv])
        scores = np.zeros(len(data.records), dtype=np.float64)
        scores[train_cv] = transform(raw_scores[train_cv])
        candidate_id = f"surface.{method}.raw"
        objective = _advanced_objective(data.y[train_cv], scores[train_cv], data.phases[train_cv])
        scores_by_id[candidate_id] = scores
        rows.append(
            {
                "candidate_id": candidate_id,
                "base_candidate_id": f"surface.{method}",
                "candidate_type": "surface",
                "family": "base_surface",
                "subset_name": method,
                "head": "surface_score",
                "calibrator": "raw",
                "feature_count": 1,
                "selection_split": "train_cv",
                "holdout_rows_used_for_selection": 0,
                "objective": f"{objective['objective']:.9f}",
                "train_cv_average_precision": f"{objective['average_precision']:.9f}",
                "train_cv_roc_auc": f"{objective['roc_auc']:.9f}",
                "train_cv_f1": f"{objective['f1']:.9f}",
                "train_cv_false_positive_rate": f"{objective['false_positive_rate']:.9f}",
                "initial_take_up_average_precision": (
                    f"{objective['initial_take_up_average_precision']:.9f}"
                ),
                "threshold": f"{objective['threshold']:.9f}",
                "calibration_info": json.dumps(info, sort_keys=True),
            }
        )
    return scores_by_id, rows


def _build_meta_fusion_candidates(
    data: AdvancedData,
    component_scores: dict[str, np.ndarray],
    ranked_rows: list[dict[str, Any]],
    profile: SearchProfile,
    *,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> tuple[dict[str, MetaFusionSpec], dict[str, np.ndarray], list[dict[str, Any]]]:
    train_cv = data.split == "train_cv"
    eligible_ids = [
        str(row["candidate_id"])
        for row in ranked_rows
        if str(row["candidate_id"]) in component_scores
    ]
    eligible_ids = eligible_ids[: profile.fusion_pool_size]
    meta_specs: dict[str, MetaFusionSpec] = {}
    meta_scores: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for top_k in profile.fusion_top_ks:
        component_ids = tuple(eligible_ids[: min(top_k, len(eligible_ids))])
        if len(component_ids) < 2:
            continue
        train_matrix = np.column_stack([component_scores[item][train_cv] for item in component_ids])
        weights, opt_info = _optimize_fusion_weights(
            train_matrix,
            data.y[train_cv],
            data.phases[train_cv],
            seed=_stable_seed(seed, "meta-fusion", profile.name, top_k),
            evolution_rounds=evolution_rounds,
            sample_rows=evolution_sample_rows,
        )
        raw_scores = np.zeros(len(data.records), dtype=np.float64)
        raw_scores[train_cv] = train_matrix @ weights
        base_id = f"meta_fusion.top{len(component_ids)}.{profile.name}"
        meta_specs[base_id] = MetaFusionSpec(
            candidate_id=base_id,
            component_ids=component_ids,
            weights=tuple(float(value) for value in weights),
            optimizer=str(opt_info["optimizer"]),
            top_k=len(component_ids),
        )
        for calibrator in CALIBRATORS:
            calibration = _fit_calibration(raw_scores[train_cv], data.y[train_cv], calibrator)
            scores = np.zeros(len(data.records), dtype=np.float64)
            scores[train_cv] = calibration.transform(raw_scores[train_cv])
            candidate_id = f"{base_id}.{calibrator}"
            objective = _advanced_objective(
                data.y[train_cv], scores[train_cv], data.phases[train_cv]
            )
            meta_scores[candidate_id] = scores
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_id,
                    "candidate_type": "meta_fusion",
                    "family": "second_stage_meta_fusion",
                    "subset_name": f"top_{len(component_ids)}",
                    "head": opt_info["optimizer"],
                    "calibrator": calibrator,
                    "feature_count": len(component_ids),
                    "selection_split": "train_cv",
                    "holdout_rows_used_for_selection": 0,
                    "objective": f"{objective['objective']:.9f}",
                    "train_cv_average_precision": f"{objective['average_precision']:.9f}",
                    "train_cv_roc_auc": f"{objective['roc_auc']:.9f}",
                    "train_cv_f1": f"{objective['f1']:.9f}",
                    "train_cv_false_positive_rate": f"{objective['false_positive_rate']:.9f}",
                    "initial_take_up_average_precision": (
                        f"{objective['initial_take_up_average_precision']:.9f}"
                    ),
                    "threshold": f"{objective['threshold']:.9f}",
                    "calibration_info": json.dumps(
                        {
                            **calibration.info,
                            **opt_info,
                            "component_ids": list(component_ids),
                            "weights": [float(value) for value in weights],
                        },
                        sort_keys=True,
                    ),
                }
            )
    return meta_specs, meta_scores, rows


def _score_candidate_all(
    data: AdvancedData,
    spec: CandidateSpec,
    calibrator: str,
    *,
    folds: int,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
    train_raw: np.ndarray | None = None,
    holdout_raw: np.ndarray | None = None,
) -> np.ndarray:
    train_cv = data.split == "train_cv"
    holdout = data.split == "holdout"
    if holdout_raw is None:
        holdout_raw = _score_selected_holdout(
            data,
            spec,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )
    if train_raw is None:
        train_raw = _candidate_oof_raw(
            data,
            spec,
            folds=folds,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )
    calibration = _fit_calibration(train_raw[train_cv], data.y[train_cv], calibrator)
    scores = np.zeros(len(data.records), dtype=np.float64)
    scores[train_cv] = calibration.transform(train_raw[train_cv])
    scores[holdout] = calibration.transform(holdout_raw[holdout])
    return scores


def _score_surface_all(data: AdvancedData, method: str) -> np.ndarray:
    train_cv = data.split == "train_cv"
    raw = data.surface_scores[method]
    transform, _info = _fit_oriented_unit_transform(raw[train_cv], data.y[train_cv])
    return transform(raw)


def _score_selected_candidate(
    data: AdvancedData,
    selected: dict[str, Any],
    spec_by_id: dict[str, CandidateSpec],
    meta_specs: dict[str, MetaFusionSpec],
    candidate_train_raws: dict[str, np.ndarray],
    *,
    folds: int,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    train_cv = data.split == "train_cv"
    holdout = data.split == "holdout"
    candidate_type = str(selected.get("candidate_type", "advanced_candidate"))
    base_id = str(selected["base_candidate_id"])
    calibrator = str(selected["calibrator"])
    if candidate_type == "surface":
        method = base_id.removeprefix("surface.")
        return (
            ADVANCED_METHOD_ID,
            _score_surface_all(data, method),
            {"selected_kind": "surface", "component_count": 1},
        )
    if candidate_type == "meta_fusion":
        meta = meta_specs[base_id]
        component_arrays = []
        component_details = []
        holdout_raw_cache: dict[str, np.ndarray] = {}
        for component_id in meta.component_ids:
            if component_id.startswith("surface."):
                method = component_id.removeprefix("surface.").removesuffix(".raw")
                component_arrays.append(_score_surface_all(data, method))
                component_details.append({"candidate_id": component_id, "kind": "surface"})
                continue
            component_row_base, component_calibrator = component_id.rsplit(".", 1)
            component_spec = spec_by_id[component_row_base]
            if component_row_base not in holdout_raw_cache:
                holdout_raw_cache[component_row_base] = _score_selected_holdout(
                    data,
                    component_spec,
                    seed=seed,
                    evolution_rounds=evolution_rounds,
                    evolution_sample_rows=evolution_sample_rows,
                )
            component_arrays.append(
                _score_candidate_all(
                    data,
                    component_spec,
                    component_calibrator,
                    folds=folds,
                    seed=seed,
                    evolution_rounds=evolution_rounds,
                    evolution_sample_rows=evolution_sample_rows,
                    train_raw=candidate_train_raws.get(component_row_base),
                    holdout_raw=holdout_raw_cache[component_row_base],
                )
            )
            component_details.append(
                {
                    "candidate_id": component_id,
                    "kind": "advanced_candidate",
                    "base_candidate_id": component_row_base,
                    "calibrator": component_calibrator,
                }
            )
        raw = np.column_stack(component_arrays) @ np.asarray(meta.weights, dtype=np.float64)
        calibration = _fit_calibration(raw[train_cv], data.y[train_cv], calibrator)
        scores = np.zeros(len(data.records), dtype=np.float64)
        scores[train_cv] = calibration.transform(raw[train_cv])
        scores[holdout] = calibration.transform(raw[holdout])
        return (
            ADVANCED_METHOD_ID,
            scores,
            {
                "selected_kind": "meta_fusion",
                "component_count": len(meta.component_ids),
                "component_ids": list(meta.component_ids),
                "weights": list(meta.weights),
                "optimizer": meta.optimizer,
                "components": component_details,
            },
        )
    spec = spec_by_id[base_id]
    return (
        ADVANCED_METHOD_ID,
        _score_candidate_all(
            data,
            spec,
            calibrator,
            folds=folds,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
            train_raw=candidate_train_raws.get(base_id),
        ),
        {"selected_kind": "advanced_candidate", "component_count": 1},
    )


def _resolve_component_candidate(
    data: AdvancedData,
    component_detail: dict[str, Any],
    spec_by_id: dict[str, CandidateSpec],
    meta_specs: dict[str, MetaFusionSpec],
    candidate_train_raws: dict[str, np.ndarray],
    *,
    folds: int,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    kind = str(component_detail.get("kind", "advanced_candidate"))
    candidate_id = str(component_detail.get("candidate_id", ""))
    if kind == "surface":
        method = candidate_id.removeprefix("surface.").removesuffix(".raw")
        return (
            candidate_id,
            _score_surface_all(data, method),
            {
                "candidate_id": candidate_id,
                "group": method,
                "kind": "surface",
                "alias": method,
            },
        )

    base_candidate_id = str(component_detail.get("base_candidate_id", "")).strip()
    calibrator = str(component_detail.get("calibrator", "raw"))
    if not base_candidate_id:
        base_candidate_id = candidate_id.rsplit(".", 1)[0]
    spec = spec_by_id[base_candidate_id]
    scores = _score_candidate_all(
        data,
        spec,
        calibrator,
        folds=folds,
        seed=seed,
        evolution_rounds=evolution_rounds,
        evolution_sample_rows=evolution_sample_rows,
        train_raw=candidate_train_raws.get(base_candidate_id),
    )
    alias = base_candidate_id.removeprefix(f"{ADVANCED_METHOD_ID}.")
    alias = alias.replace(".", "/")
    return (
        candidate_id,
        scores,
        {
            "candidate_id": candidate_id,
            "base_candidate_id": base_candidate_id,
            "calibrator": calibrator,
            "group": str(spec.subset_name),
            "head": spec.head,
            "kind": "advanced_candidate",
            "alias": alias,
        },
    )


def _build_selected_candidate_transparency(
    data: AdvancedData,
    selected: dict[str, Any],
    selected_details: dict[str, Any],
    spec_by_id: dict[str, CandidateSpec],
    meta_specs: dict[str, MetaFusionSpec],
    candidate_train_raws: dict[str, np.ndarray],
    *,
    folds: int,
    seed: int,
    evolution_rounds: int,
    evolution_sample_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_cv = data.split == "train_cv"
    holdout = data.split == "holdout"
    selected_kind = str(selected_details.get("selected_kind", "advanced_candidate"))
    component_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []

    if selected_kind != "meta_fusion":
        return (
            component_rows,
            ablation_rows,
            {
                "selected_kind": selected_kind,
                "component_count": int(selected_details.get("component_count", 1)),
                "notes": "component transparency not available for non-fusion selections",
            },
        )

    meta = meta_specs[str(selected["base_candidate_id"])]
    component_infos: list[dict[str, Any]] = []
    component_arrays: list[np.ndarray] = []
    for index, component_detail in enumerate(selected_details.get("components", [])):
        candidate_id, scores, info = _resolve_component_candidate(
            data,
            component_detail,
            spec_by_id,
            meta_specs,
            candidate_train_raws,
            folds=folds,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )
        info["weight"] = float(meta.weights[index]) if index < len(meta.weights) else 0.0
        info["component_index"] = index
        component_infos.append(info)
        component_arrays.append(scores)
        alias_rows.append(
            {
                "component_id": candidate_id,
                "alias": info["alias"],
                "group": info["group"],
                "kind": info["kind"],
                "weight": float(meta.weights[index]) if index < len(meta.weights) else 0.0,
            }
        )

    component_matrix = np.column_stack(component_arrays)
    raw_scores = component_matrix @ np.asarray(meta.weights, dtype=np.float64)
    calibration = _fit_calibration(
        raw_scores[train_cv], data.y[train_cv], str(selected["calibrator"])
    )
    selected_scores = np.zeros(len(data.records), dtype=np.float64)
    selected_scores[train_cv] = calibration.transform(raw_scores[train_cv])
    selected_scores[holdout] = calibration.transform(raw_scores[holdout])

    for row_index, record in enumerate(data.records):
        for component_index, component_info in enumerate(component_infos):
            component_rows.append(
                {
                    "record_id": record["record_id"],
                    "scenario_group_id": record["scenario_group_id"],
                    "phase_id": record["phase_id"],
                    "split_role": record["split_role"],
                    "label_id": int(record["label_id"]),
                    "selected_candidate_id": selected["candidate_id"],
                    "component_index": component_index,
                    "component_id": component_info["candidate_id"],
                    "component_alias": component_info["alias"],
                    "component_group": component_info["group"],
                    "component_kind": component_info["kind"],
                    "component_weight": f"{component_info['weight']:.9f}",
                    "component_score": f"{float(component_arrays[component_index][row_index]):.9f}",
                    "selected_score": f"{float(selected_scores[row_index]):.9f}",
                }
            )

    selected_threshold, _selected_threshold_metrics = _select_threshold(
        data.y[train_cv], selected_scores[train_cv]
    )
    selected_holdout_scores = selected_scores[holdout]
    selected_holdout_labels = data.y[holdout]

    ablation_rows.append(
        {
            "variant_id": "selected",
            "variant_type": "selected_fusion",
            "component_scope": "all",
            "component_ids": "|".join(meta.component_ids),
            "calibrator": str(selected["calibrator"]),
            "threshold": f"{selected_threshold:.9f}",
            "holdout_average_precision": f"{_average_precision(selected_holdout_labels, selected_holdout_scores):.9f}",
            "holdout_roc_auc": f"{_roc_auc(selected_holdout_labels, selected_holdout_scores):.9f}",
            "holdout_f1": f"{_binary_metrics(selected_holdout_labels, selected_holdout_scores, selected_threshold)['f1']:.9f}",
            "brier_score": f"{_brier_score(selected_holdout_labels, selected_holdout_scores):.9f}",
            "ece": f"{_expected_calibration_error(selected_holdout_labels, selected_holdout_scores)[0]:.9f}",
        }
    )

    raw_threshold, _ = _select_threshold(data.y[train_cv], raw_scores[train_cv])
    raw_holdout = raw_scores[holdout]
    ablation_rows.append(
        {
            "variant_id": "no_calibration",
            "variant_type": "no_calibration",
            "component_scope": "all",
            "component_ids": "|".join(meta.component_ids),
            "calibrator": "raw",
            "threshold": f"{raw_threshold:.9f}",
            "holdout_average_precision": f"{_average_precision(selected_holdout_labels, raw_holdout):.9f}",
            "holdout_roc_auc": f"{_roc_auc(selected_holdout_labels, raw_holdout):.9f}",
            "holdout_f1": f"{_binary_metrics(selected_holdout_labels, raw_holdout, raw_threshold)['f1']:.9f}",
            "brier_score": f"{_brier_score(selected_holdout_labels, raw_holdout):.9f}",
            "ece": f"{_expected_calibration_error(selected_holdout_labels, raw_holdout)[0]:.9f}",
        }
    )

    top_k_values = sorted({1, 2, 4})
    for top_k in top_k_values:
        if top_k < 1:
            continue
        subset_weights = np.asarray(meta.weights[:top_k], dtype=np.float64)
        subset_weights = subset_weights / max(float(np.sum(subset_weights)), 1e-9)
        subset_scores = component_matrix[:, :top_k] @ subset_weights
        calibration = _fit_calibration(
            subset_scores[train_cv], data.y[train_cv], str(selected["calibrator"])
        )
        calibrated = np.zeros(len(data.records), dtype=np.float64)
        calibrated[train_cv] = calibration.transform(subset_scores[train_cv])
        calibrated[holdout] = calibration.transform(subset_scores[holdout])
        threshold, _ = _select_threshold(data.y[train_cv], calibrated[train_cv])
        ablation_rows.append(
            {
                "variant_id": f"top_k_{top_k}",
                "variant_type": "top_k",
                "component_scope": f"first_{top_k}",
                "component_ids": "|".join(meta.component_ids[:top_k]),
                "calibrator": str(selected["calibrator"]),
                "threshold": f"{threshold:.9f}",
                "holdout_average_precision": f"{_average_precision(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_roc_auc": f"{_roc_auc(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_f1": f"{_binary_metrics(selected_holdout_labels, calibrated[holdout], threshold)['f1']:.9f}",
                "brier_score": f"{_brier_score(selected_holdout_labels, calibrated[holdout]):.9f}",
                "ece": f"{_expected_calibration_error(selected_holdout_labels, calibrated[holdout])[0]:.9f}",
            }
        )

    for index, component_info in enumerate(component_infos):
        keep = [i for i in range(len(component_infos)) if i != index]
        if not keep:
            continue
        subset_weights = np.asarray([meta.weights[i] for i in keep], dtype=np.float64)
        subset_weights = subset_weights / max(float(np.sum(subset_weights)), 1e-9)
        subset_scores = component_matrix[:, keep] @ subset_weights
        calibration = _fit_calibration(
            subset_scores[train_cv], data.y[train_cv], str(selected["calibrator"])
        )
        calibrated = np.zeros(len(data.records), dtype=np.float64)
        calibrated[train_cv] = calibration.transform(subset_scores[train_cv])
        calibrated[holdout] = calibration.transform(subset_scores[holdout])
        threshold, _ = _select_threshold(data.y[train_cv], calibrated[train_cv])
        ablation_rows.append(
            {
                "variant_id": f"drop_{component_info['component_index']}",
                "variant_type": "component_drop",
                "component_scope": component_info["group"],
                "component_ids": "|".join(meta.component_ids[i] for i in keep),
                "calibrator": str(selected["calibrator"]),
                "threshold": f"{threshold:.9f}",
                "holdout_average_precision": f"{_average_precision(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_roc_auc": f"{_roc_auc(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_f1": f"{_binary_metrics(selected_holdout_labels, calibrated[holdout], threshold)['f1']:.9f}",
                "brier_score": f"{_brier_score(selected_holdout_labels, calibrated[holdout]):.9f}",
                "ece": f"{_expected_calibration_error(selected_holdout_labels, calibrated[holdout])[0]:.9f}",
            }
        )

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(component_infos):
        group_to_indices[str(info["group"])].append(index)
    for group, indices in sorted(group_to_indices.items()):
        keep = [i for i in range(len(component_infos)) if i not in indices]
        if not keep:
            continue
        subset_weights = np.asarray([meta.weights[i] for i in keep], dtype=np.float64)
        subset_weights = subset_weights / max(float(np.sum(subset_weights)), 1e-9)
        subset_scores = component_matrix[:, keep] @ subset_weights
        calibration = _fit_calibration(
            subset_scores[train_cv], data.y[train_cv], str(selected["calibrator"])
        )
        calibrated = np.zeros(len(data.records), dtype=np.float64)
        calibrated[train_cv] = calibration.transform(subset_scores[train_cv])
        calibrated[holdout] = calibration.transform(subset_scores[holdout])
        threshold, _ = _select_threshold(data.y[train_cv], calibrated[train_cv])
        ablation_rows.append(
            {
                "variant_id": f"drop_group_{group}",
                "variant_type": "modality_drop",
                "component_scope": group,
                "component_ids": "|".join(meta.component_ids[i] for i in keep),
                "calibrator": str(selected["calibrator"]),
                "threshold": f"{threshold:.9f}",
                "holdout_average_precision": f"{_average_precision(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_roc_auc": f"{_roc_auc(selected_holdout_labels, calibrated[holdout]):.9f}",
                "holdout_f1": f"{_binary_metrics(selected_holdout_labels, calibrated[holdout], threshold)['f1']:.9f}",
                "brier_score": f"{_brier_score(selected_holdout_labels, calibrated[holdout]):.9f}",
                "ece": f"{_expected_calibration_error(selected_holdout_labels, calibrated[holdout])[0]:.9f}",
            }
        )

    return (
        component_rows,
        ablation_rows,
        {
            "selected_kind": selected_kind,
            "component_count": len(component_infos),
            "component_ids": list(meta.component_ids),
            "weights": [float(value) for value in meta.weights],
            "component_infos": component_infos,
            "threshold": float(selected_threshold),
            "selected_score_mean": float(np.mean(selected_holdout_scores)),
            "selected_score_std": float(np.std(selected_holdout_scores)),
        },
    )


def _metric_row(
    method: str,
    split_role: str,
    phase_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    binary = _binary_metrics(labels, scores, threshold)
    roc_auc = _float_or_none(_roc_auc(labels, scores))
    average_precision = _float_or_none(_average_precision(labels, scores))
    return {
        "method": method,
        "split_role": split_role,
        "phase_id": phase_id,
        "record_count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "threshold_source": "train_cv_max_f1",
        "threshold": _format_metric(threshold),
        "roc_auc": _format_metric(roc_auc),
        "average_precision": _format_metric(average_precision),
        "accuracy": _format_metric(binary["accuracy"]),
        "precision": _format_metric(binary["precision"]),
        "recall": _format_metric(binary["recall"]),
        "specificity": _format_metric(binary["specificity"]),
        "false_positive_rate": _format_metric(binary["false_positive_rate"]),
        "f1": _format_metric(binary["f1"]),
    }


def _advanced_performance_rows(
    data: AdvancedData,
    method_scores: dict[str, np.ndarray],
    candidate_train_scores: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = _build_performance_reports(data.records, method_scores)
    train_cv = data.split == "train_cv"
    for method, scores in candidate_train_scores.items():
        threshold, _metrics = _select_threshold(data.y[train_cv], scores[train_cv])
        rows.append(
            _metric_row(
                method,
                "train_cv",
                "all",
                data.y[train_cv],
                scores[train_cv],
                threshold,
            )
        )
        for phase in sorted(set(data.phases)):
            phase_mask = train_cv & (data.phases == phase)
            rows.append(
                _metric_row(
                    method,
                    "train_cv",
                    str(phase),
                    data.y[phase_mask],
                    scores[phase_mask],
                    threshold,
                )
            )
    summary["advanced_candidate_train_cv_methods"] = list(candidate_train_scores)
    summary["advanced_candidate_holdout_policy"] = "only_selected_winner_scored_on_holdout"
    return rows, summary


def run_advanced_main_run_detectors(
    data_root: Path,
    out_root: Path,
    *,
    folds: int = 5,
    seed: int = 202605210136,
    force: bool = False,
    candidate_limit: int = 128,
    evolution_rounds: int = 5,
    search_profile: str | SearchProfile = "balanced",
    feature_cache: Path | None = None,
    evolution_sample_rows: int = 4500,
    selection_lock: Path | None = None,
    score_locked_only: bool = False,
    write_component_scores: bool = False,
) -> dict[str, Any]:
    profile = _resolve_search_profile(search_profile)
    if candidate_limit < profile.min_candidate_limit:
        raise ValueError(
            f"candidate_limit must be at least {profile.min_candidate_limit} "
            f"for search profile {profile.name}"
        )
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = load_advanced_main_run_data(
        data_root,
        search_profile=profile,
        feature_cache=feature_cache,
    )
    train_cv = data.split == "train_cv"
    holdout = data.split == "holdout"
    specs = _make_candidate_specs(data, candidate_limit=candidate_limit)
    leaderboard: list[dict[str, Any]] = []
    trace_path = out_root / "evolution_trace.jsonl"
    trace_csv_path = out_root / "evolution_trace.csv"
    evolution_trace_rows: list[dict[str, Any]] = []
    running_best_trace_row: dict[str, Any] | None = None
    candidate_train_scores: dict[str, np.ndarray] = {}
    candidate_train_raws: dict[str, np.ndarray] = {}
    spec_by_id = {spec.candidate_id: spec for spec in specs}

    def add_evolution_trace(stage: str, row: dict[str, Any]) -> None:
        nonlocal running_best_trace_row
        if running_best_trace_row is None or _trace_sort_tuple(row) > _trace_sort_tuple(
            running_best_trace_row
        ):
            running_best_trace_row = dict(row)
        assert running_best_trace_row is not None
        evolution_trace_rows.append(
            _make_evolution_trace_row(
                candidate_index=len(evolution_trace_rows) + 1,
                stage=stage,
                row=row,
                running_best=running_best_trace_row,
            )
        )

    for spec in specs:
        raw = _candidate_oof_raw(
            data,
            spec,
            folds=folds,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )
        candidate_train_raws[spec.candidate_id] = raw
        for calibrator in CALIBRATORS:
            calibration = _fit_calibration(raw[train_cv], data.y[train_cv], calibrator)
            scores = np.zeros(len(data.records), dtype=np.float64)
            scores[train_cv] = calibration.transform(raw[train_cv])
            candidate_id = f"{spec.candidate_id}.{calibrator}"
            objective = _advanced_objective(
                data.y[train_cv], scores[train_cv], data.phases[train_cv]
            )
            row = {
                "candidate_id": candidate_id,
                "base_candidate_id": spec.candidate_id,
                "candidate_type": "advanced_candidate",
                "family": spec.family,
                "subset_name": spec.subset_name,
                "head": spec.head,
                "calibrator": calibrator,
                "feature_count": len(spec.feature_indices),
                "selection_split": "train_cv",
                "holdout_rows_used_for_selection": 0,
                "objective": f"{objective['objective']:.9f}",
                "train_cv_average_precision": f"{objective['average_precision']:.9f}",
                "train_cv_roc_auc": f"{objective['roc_auc']:.9f}",
                "train_cv_f1": f"{objective['f1']:.9f}",
                "train_cv_false_positive_rate": f"{objective['false_positive_rate']:.9f}",
                "initial_take_up_average_precision": (
                    f"{objective['initial_take_up_average_precision']:.9f}"
                ),
                "threshold": f"{objective['threshold']:.9f}",
                "calibration_info": json.dumps(calibration.info, sort_keys=True),
            }
            leaderboard.append(row)
            candidate_train_scores[candidate_id] = scores
            add_evolution_trace("base_candidate_search", row)

    surface_scores, surface_rows = _surface_candidate_scores(data)
    leaderboard.extend(surface_rows)
    candidate_train_scores.update(surface_scores)
    for row in surface_rows:
        add_evolution_trace("surface_control", row)
    leaderboard.sort(
        key=lambda row: (
            _safe_float(row["objective"]),
            _safe_float(row["train_cv_average_precision"]),
            _safe_float(row["train_cv_roc_auc"]),
            row["candidate_id"],
        ),
        reverse=True,
    )
    meta_specs, meta_scores, meta_rows = _build_meta_fusion_candidates(
        data,
        candidate_train_scores,
        leaderboard,
        profile,
        seed=seed,
        evolution_rounds=evolution_rounds,
        evolution_sample_rows=evolution_sample_rows,
    )
    leaderboard.extend(meta_rows)
    candidate_train_scores.update(meta_scores)
    for row in meta_rows:
        add_evolution_trace("meta_fusion_search", row)
    leaderboard.sort(
        key=lambda row: (
            _safe_float(row["objective"]),
            _safe_float(row["train_cv_average_precision"]),
            _safe_float(row["train_cv_roc_auc"]),
            row["candidate_id"],
        ),
        reverse=True,
    )
    if len(leaderboard) < candidate_limit:
        raise AssertionError(
            f"expected at least {candidate_limit} candidates, got {len(leaderboard)}"
        )
    leaderboard = leaderboard[: len(leaderboard)]
    eligible_selected = [
        row
        for row in leaderboard
        if row.get("candidate_type") == "meta_fusion" and row.get("calibrator") == "geodesic_odds"
    ]
    if not eligible_selected:
        eligible_selected = [
            row for row in leaderboard if row.get("candidate_type") == "meta_fusion"
        ]
    selected = eligible_selected[0] if eligible_selected else leaderboard[0]
    selected_rank = leaderboard.index(selected) + 1
    if selection_lock is not None:
        if not selection_lock.exists():
            raise FileNotFoundError(f"missing selection lock override: {selection_lock}")
        lock_payload = _read_json(selection_lock)
        override_id = str(lock_payload.get("selected_candidate_id", ""))
        if override_id:
            override = next(
                (row for row in leaderboard if row["candidate_id"] == override_id), None
            )
            if override is None:
                raise ValueError(
                    f"selection lock requested candidate {override_id!r}, which is not on the leaderboard"
                )
            selected = override
            selected_rank = leaderboard.index(selected) + 1
    selected_base = str(selected["base_candidate_id"])
    selected_calibrator_name = str(selected["calibrator"])

    for row in evolution_trace_rows:
        is_selected = row["candidate_id"] == selected["candidate_id"]
        row["selected_by_cv"] = is_selected
        row["final_selected"] = is_selected
    with trace_path.open("w", encoding="utf-8") as trace:
        for row in evolution_trace_rows:
            trace.write(json.dumps(row, sort_keys=True) + "\n")
    _write_csv(trace_csv_path, evolution_trace_rows)
    _write_csv(out_root / "candidate_leaderboard.csv", leaderboard)
    selection_lock = {
        "dataset_profile": DATASET_PROFILE,
        "advanced_output_profile": ADVANCED_OUTPUT_PROFILE,
        "search_profile": profile.name,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_type": selected.get("candidate_type", "advanced_candidate"),
        "base_candidate_id": selected_base,
        "calibrator": selected_calibrator_name,
        "selection_split": "train_cv",
        "holdout_rows_used_for_selection": 0,
        "candidate_count": len(leaderboard),
        "candidate_limit": candidate_limit,
        "train_cv_rank": selected_rank,
        "selection_policy": (
            "best train/CV calibrated meta-fusion artifact when available; "
            "single-component candidates remain internal controls"
        ),
        "train_cv_objective": selected["objective"],
        "train_cv_average_precision": selected["train_cv_average_precision"],
        "train_cv_roc_auc": selected["train_cv_roc_auc"],
        "evolution_trace": {
            "schema_version": EVOLUTION_TRACE_SCHEMA_VERSION,
            "path": "evolution_trace.jsonl",
            "csv_path": "evolution_trace.csv",
            "trace_row_count": len(evolution_trace_rows),
            "trace_basis": "true_evaluation_order",
        },
        "split_isolation_policy": {
            "candidate_selection_split": "train_cv",
            "calibration_split": "train_cv",
            "threshold_split": "train_cv",
            "holdout_rows_used_for_candidate_selection": 0,
            "holdout_rows_used_for_calibration": 0,
            "holdout_rows_used_for_threshold_selection": 0,
            "holdout_evaluation_policy": "after_selection_lock_only",
        },
    }
    if selected.get("candidate_type") == "meta_fusion" and selected_base in meta_specs:
        meta = meta_specs[selected_base]
        selection_lock["meta_fusion"] = {
            "optimizer": meta.optimizer,
            "top_k": meta.top_k,
            "component_ids": list(meta.component_ids),
            "weights": list(meta.weights),
        }
    _write_json(out_root / "selection_lock.json", selection_lock)

    selected_method, selected_scores, selected_details = _score_selected_candidate(
        data,
        selected,
        spec_by_id,
        meta_specs,
        candidate_train_raws,
        folds=folds,
        seed=seed,
        evolution_rounds=evolution_rounds,
        evolution_sample_rows=evolution_sample_rows,
    )

    component_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    component_summary: dict[str, Any] = {
        "selected_kind": selected_details.get("selected_kind", "advanced_candidate"),
        "component_count": int(selected_details.get("component_count", 1)),
    }
    if (
        write_component_scores
        or selection_lock is not None
        or selected_details.get("selected_kind") == "meta_fusion"
    ):
        component_rows, ablation_rows, component_summary = _build_selected_candidate_transparency(
            data,
            selected,
            selected_details,
            spec_by_id,
            meta_specs,
            candidate_train_raws,
            folds=folds,
            seed=seed,
            evolution_rounds=evolution_rounds,
            evolution_sample_rows=evolution_sample_rows,
        )

    method_scores = {
        **data.surface_scores,
        selected_method: selected_scores,
    }
    performance_rows, performance_summary = _advanced_performance_rows(
        data,
        method_scores,
        candidate_train_scores,
    )
    performance_summary["advanced_selection"] = {
        "selected_method": selected_method,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_type": selected.get("candidate_type", "advanced_candidate"),
        "base_candidate_id": selected_base,
        "calibrator": selected_calibrator_name,
        "selection_split": "train_cv",
        "holdout_rows_used_for_selection": 0,
        "candidate_count": len(leaderboard),
        "candidate_limit": candidate_limit,
        "search_profile": profile.name,
        "train_cv_rank": selected_rank,
        "selection_policy": selection_lock.get("selection_policy", ""),
        "objective": selected["objective"],
        **selected_details,
    }
    selected_threshold, selected_train_threshold_metrics = _select_threshold(
        data.y[train_cv], selected_scores[train_cv]
    )
    selected_holdout_metrics = _binary_metrics(
        data.y[holdout], selected_scores[holdout], selected_threshold
    )
    selected_holdout_ap = _float_or_none(
        _average_precision(data.y[holdout], selected_scores[holdout])
    )
    selected_holdout_auc = _float_or_none(_roc_auc(data.y[holdout], selected_scores[holdout]))
    train_cv_ap = _safe_float(selected["train_cv_average_precision"])
    train_cv_auc = _safe_float(selected["train_cv_roc_auc"])
    target = BASELINE_TARGETS["layered_fusion_c2"]
    expected_full_records = DEFAULT_SCENARIO_GROUPS * len(PHASES)
    promotion_metrics_pass = (
        train_cv_ap > target["train_cv"]["average_precision"]
        and selected_holdout_ap is not None
        and selected_holdout_ap > target["holdout"]["average_precision"]
        and selected_holdout_auc is not None
        and selected_holdout_auc >= target["holdout"]["roc_auc"]
    )
    promotion_gate = {
        "baseline_method": "layered_fusion_c2",
        "full_run_record_count_required": expected_full_records,
        "record_count": len(data.records),
        "train_cv_ap_target": target["train_cv"]["average_precision"],
        "holdout_ap_target": target["holdout"]["average_precision"],
        "holdout_auc_target": target["holdout"]["roc_auc"],
        "selected_train_cv_average_precision": train_cv_ap,
        "selected_train_cv_roc_auc": train_cv_auc,
        "selected_holdout_average_precision": selected_holdout_ap,
        "selected_holdout_roc_auc": selected_holdout_auc,
        "status": (
            "pass"
            if len(data.records) == expected_full_records and promotion_metrics_pass
            else "smoke_not_promoted"
            if len(data.records) != expected_full_records
            else "not_promoted"
        ),
    }

    prediction_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(data.records):
        prediction_rows.append(
            {
                "record_id": record["record_id"],
                "scenario_group_id": record["scenario_group_id"],
                "time_lock_id": record["time_lock_id"],
                "phase_id": record["phase_id"],
                "split_role": record["split_role"],
                "cv_fold": record["cv_fold"],
                "label_id": int(record["label_id"]),
                "selected_candidate_id": selected["candidate_id"],
                "advanced_score": f"{float(selected_scores[idx]):.9f}",
                "threshold": f"{float(selected_threshold):.9f}",
                "binary_prediction": int(float(selected_scores[idx]) >= selected_threshold),
                "calibration_role": (
                    "cv_fold_scored"
                    if record["split_role"] == "train_cv"
                    else "holdout_scored_once_after_selection"
                ),
                "high_resolution_xku_cuas": f"{float(data.surface_scores['high_resolution_xku_cuas'][idx]):.9f}",
                "tactical_s_band_aesa": f"{float(data.surface_scores['tactical_s_band_aesa'][idx]):.9f}",
                "gbad_3d4d_cueing": f"{float(data.surface_scores['gbad_3d4d_cueing'][idx]):.9f}",
                "distributed_acoustic_cue": f"{float(data.surface_scores[ACOUSTIC_VIEW_ID][idx]):.9f}",
                "layered_fusion_c2": f"{float(data.surface_scores[FUSION_VIEW_ID][idx]):.9f}",
            }
        )

    feature_manifest = {
        "dataset_profile": DATASET_PROFILE,
        "advanced_output_profile": ADVANCED_OUTPUT_PROFILE,
        "method_family": ADVANCED_METHOD_ID,
        "strict_open_claim_boundary": (
            "synthetic public-proxy experiment; no measured-platform, proprietary-equivalent, "
            "classified-fidelity, or deployment-performance claim"
        ),
        "candidate_generation_policy": {
            "search_profile": profile.name,
            "requested_candidate_limit": candidate_limit,
            "actual_candidate_count": len(leaderboard),
            "base_candidate_count": len(specs),
            "calibrators": list(CALIBRATORS),
            "heads": sorted({spec.head for spec in specs}),
            "selection_metric": "AP + 0.25*AUC - FPR_penalty - weak_initial_take_up_penalty",
            "second_stage_fusion": {
                "enabled": True,
                "optimizer": "clean_room_shade_multi_resolution_de",
                "top_k_values": list(profile.fusion_top_ks),
                "pool_size": profile.fusion_pool_size,
                "nonnegative_weights": True,
            },
        },
        "clean_room_inspiration": {
            "source_note": (
                "Veox math was used only as high-level clean-room inspiration; no source code, "
                "imports, symbols, or copied implementation are used."
            ),
            "implemented_families": [
                "spectral_conditioning",
                "marchenko_pastur_shrinkage_mass",
                "multi_lag_temporal_decorrelation",
                "ssa_hankel_rank_features",
                "morlet_wavelet_scattering_summaries",
                "koopman_edmd_style_residuals",
                "diffusion_laplacian_eigengaps",
                "persistent_laplacian_takens_topology_proxies",
                "train_cv_quantile_transport",
                "copula_rank_geometry",
                "wasserstein_style_prototype_distances",
                "information_geometric_alpha_means",
            ],
        },
        "split_isolation_policy": {
            "candidate_selection_split": "train_cv",
            "calibration_split": "train_cv",
            "threshold_split": "train_cv",
            "passive_rank_reference_split": "train_cv",
            "holdout_rows_used_for_feature_selection": 0,
            "holdout_rows_used_for_candidate_selection": 0,
            "holdout_rows_used_for_calibration": 0,
            "holdout_rows_used_for_threshold_selection": 0,
            "holdout_evaluation_policy": "single selected winner scored after internal-CV lock",
            "selection_lock_path": "selection_lock.json",
        },
        "model_feature_denylist": list(MODEL_FEATURE_DENYLIST),
        "feature_count": len(data.feature_names),
        "feature_columns": data.feature_names,
        "branch_feature_counts": {
            branch: len(indices) for branch, indices in sorted(data.branch_indices.items())
        },
        **data.manifest,
    }
    quality = {
        "dataset_profile": DATASET_PROFILE,
        "advanced_output_profile": ADVANCED_OUTPUT_PROFILE,
        "record_count": len(data.records),
        "candidate_count": len(leaderboard),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_type": selected.get("candidate_type", "advanced_candidate"),
        "search_profile": profile.name,
        "holdout_isolation_status": "pass",
        "selection_lock_status": "written_before_holdout_scoring",
        "candidate_selection_holdout_record_count": 0,
        "calibration_holdout_record_count": 0,
        "threshold_holdout_record_count": 0,
        "selected_holdout_evaluation_count": 1,
        "selected_train_cv_threshold_metrics": selected_train_threshold_metrics,
        "selected_holdout_threshold_metrics": selected_holdout_metrics,
        "promotion_gate": promotion_gate,
        "outputs": {
            "candidate_leaderboard": "candidate_leaderboard.csv",
            "selection_lock": "selection_lock.json",
            "advanced_feature_manifest": "advanced_feature_manifest.json",
            "evolution_trace": "evolution_trace.jsonl",
            "evolution_trace_csv": "evolution_trace.csv",
            "advanced_predictions": "advanced_predictions.csv",
            "performance_metrics": "performance_metrics.csv",
            "performance_summary": "performance_summary.json",
            "fusion_quality_report": "fusion_quality_report.json",
        },
        "status": "pass",
    }

    if component_rows:
        _write_csv(out_root / "selected_component_scores.csv", component_rows)
        _write_csv(out_root / "selected_component_ablations.csv", ablation_rows)
        _write_json(out_root / "selected_component_aliases.json", component_summary)
        quality["outputs"]["selected_component_scores"] = "selected_component_scores.csv"
        quality["outputs"]["selected_component_ablations"] = "selected_component_ablations.csv"
        quality["outputs"]["selected_component_aliases"] = "selected_component_aliases.json"

    _write_json(out_root / "advanced_feature_manifest.json", feature_manifest)
    _write_csv(out_root / "advanced_predictions.csv", prediction_rows)
    _write_csv(out_root / "performance_metrics.csv", performance_rows)
    _write_json(out_root / "performance_summary.json", performance_summary)
    _write_json(out_root / "fusion_quality_report.json", quality)
    return quality
