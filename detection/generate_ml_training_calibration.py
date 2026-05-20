"""Calibration anchor artifacts for the ML training generator.

Builds reference-distribution calibration reports using Wasserstein-1
and Kolmogorov-Smirnov distances.  Imports from generate_ml_training_types
only — no circular imports.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from generate_ml_training_types import (
    CALIBRATION_SAMPLE_COUNT,
    FRAME_INDEX,
)


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def quantile_samples(values: np.ndarray, sample_count: int = CALIBRATION_SAMPLE_COUNT) -> list[float]:
    arr = np.sort(finite_values(values))
    if arr.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    samples = np.quantile(arr, quantiles)
    return [float(value) for value in samples if math.isfinite(float(value))]


def target_distribution(bounds: tuple[float, float], sample_count: int = CALIBRATION_SAMPLE_COUNT) -> list[float]:
    low, high = bounds
    return [float(value) for value in np.linspace(float(low), float(high), sample_count, dtype=np.float64)]


def target_beta_distribution(
    bounds: tuple[float, float],
    alpha: float,
    beta: float,
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
) -> list[float]:
    # Deterministic inverse-free approximation: sort fixed beta draws so the
    # report carries a bounded target distribution without vendored traces.
    rng = np.random.default_rng(20260518)
    low, high = bounds
    draws = np.sort(rng.beta(alpha, beta, sample_count))
    scaled = float(low) + draws * (float(high) - float(low))
    return [float(value) for value in scaled]


def wasserstein_1d_sorted(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, max(a.size, b.size), dtype=np.float64)
    qa = np.quantile(np.sort(a), quantiles)
    qb = np.quantile(np.sort(b), quantiles)
    return float(np.mean(np.abs(qa - qb)))


def ks_distance_1d_sorted(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    values = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, values, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, values, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def calibration_distance(metric: str, target_samples: list[float], observed_samples: list[float]) -> float:
    target = finite_values(np.asarray(target_samples, dtype=np.float64))
    observed = finite_values(np.asarray(observed_samples, dtype=np.float64))
    if metric == "wasserstein_1":
        return wasserstein_1d_sorted(target, observed)
    if metric == "kolmogorov_smirnov":
        return ks_distance_1d_sorted(target, observed)
    raise ValueError(f"unsupported calibration metric: {metric}")


def frame_observation_samples(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    selector: pd.Series,
    column_name: str,
    transform: str | None = None,
) -> list[float]:
    indices = records.index[selector.to_numpy(dtype=bool)].to_numpy(dtype=np.int64)
    if indices.size == 0:
        return []
    values = frames[indices, :, FRAME_INDEX[column_name]]
    mask = valid_mask[indices]
    observed = finite_values(values[mask])
    if transform == "abs":
        observed = np.abs(observed)
    return quantile_samples(observed)


def record_observation_samples(records: pd.DataFrame, selector: pd.Series, column_name: str) -> list[float]:
    values = records.loc[selector, column_name].astype(float).to_numpy(dtype=np.float64)
    return quantile_samples(values)


def calibration_anchor_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_name": "snr_db_detector_public_proxy",
            "feature_family": "range_doppler_observable",
            "unit": "dB",
            "metric": "wasserstein_1",
            "tolerance": 22.0,
            "target_samples": target_distribution((-18.0, 24.0)),
            "citation_id": "public_proxy_link_budget_reference_envelope",
            "citation": "Strict-open current public-proxy link-budget envelope; target samples are broad reference priors, not measured radar traces.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "abs_radial_velocity_mps_detector_public_proxy",
            "feature_family": "doppler_observable",
            "unit": "m/s",
            "metric": "wasserstein_1",
            "tolerance": 28.0,
            "target_samples": target_distribution((0.0, 60.0)),
            "citation_id": "public_proxy_radial_velocity_reference_envelope",
            "citation": "Strict-open current radial-velocity reference envelope derived from broad public speed priors and random line-of-sight geometry; not true-speed truth.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "doppler_scr_detector_public_proxy",
            "feature_family": "doppler_clutter_observable",
            "unit": "dB",
            "metric": "wasserstein_1",
            "tolerance": 20.0,
            "target_samples": target_distribution((-14.0, 24.0)),
            "citation_id": "public_proxy_doppler_scr_reference_envelope",
            "citation": "Strict-open current Doppler signal-to-clutter reference envelope for simulator consistency checks; no measured target samples are vendored.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "rfi_pressure_all_scenes",
            "feature_family": "interference_observable",
            "unit": "unitless",
            "metric": "kolmogorov_smirnov",
            "tolerance": 0.75,
            "target_samples": target_beta_distribution((0.0, 1.0), alpha=1.3, beta=5.5),
            "citation_id": "public_proxy_interference_pressure_reference",
            "citation": "Strict-open current interference-pressure reference distribution for smoke artifact integrity; not deployment metadata or measured spectrum capture.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "horizon_masked_fraction_initial_take_up",
            "feature_family": "operational_los_metric",
            "unit": "fraction",
            "metric": "kolmogorov_smirnov",
            "tolerance": 0.80,
            "target_samples": target_distribution((0.35, 1.0)),
            "citation_id": "public_proxy_low_grazing_los_reference",
            "citation": "Strict-open current low-grazing line-of-sight reference envelope for initial-take-up reporting; not measured radar performance.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
    ]


def build_calibration_artifacts(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    positive = records["scene_role"].astype(str) == "positive_public_proxy"
    initial = records["phase_id"].astype(str) == "initial_take_up"
    all_records = pd.Series(True, index=records.index)
    observation_by_feature = {
        "snr_db_detector_public_proxy": frame_observation_samples(records, frames, valid_mask, positive, "snr_db"),
        "abs_radial_velocity_mps_detector_public_proxy": frame_observation_samples(
            records, frames, valid_mask, positive, "radial_velocity_mps", transform="abs",
        ),
        "doppler_scr_detector_public_proxy": frame_observation_samples(records, frames, valid_mask, positive, "doppler_scr"),
        "rfi_pressure_all_scenes": frame_observation_samples(records, frames, valid_mask, all_records, "rfi_pressure"),
        "horizon_masked_fraction_initial_take_up": record_observation_samples(records, initial, "horizon_masked_fraction"),
    }

    anchors = []
    observations = []
    distance_rows = []
    manifest_anchors = []
    for spec in calibration_anchor_specs():
        feature_name = str(spec["feature_name"])
        target_samples = [float(value) for value in spec["target_samples"]]
        observed_samples = [float(value) for value in observation_by_feature.get(feature_name, [])]
        distance = calibration_distance(str(spec["metric"]), target_samples, observed_samples)
        sample_counts_ok = len(target_samples) >= 32 and len(observed_samples) >= 32
        distance_ok = math.isfinite(distance) and distance <= float(spec["tolerance"])
        status = "pass" if sample_counts_ok and distance_ok else "fail"
        anchors.append({
            "feature_name": feature_name,
            "citation": spec["citation"],
            "url": spec["url"],
            "target_samples": target_samples,
            "metric": spec["metric"],
            "tolerance": float(spec["tolerance"]),
        })
        observations.append({"feature_name": feature_name, "observed_samples": observed_samples})
        distance_rows.append({
            "feature_name": feature_name,
            "feature_family": spec["feature_family"],
            "unit": spec["unit"],
            "citation_id": spec["citation_id"],
            "citation": spec["citation"],
            "url": spec["url"] or "",
            "metric": spec["metric"],
            "tolerance": float(spec["tolerance"]),
            "distance": distance,
            "target_sample_count": len(target_samples),
            "observed_sample_count": len(observed_samples),
            "status": status,
            "claim_boundary": "reference-only synthetic public-proxy distribution audit; not measured-anchor promotion",
        })
        manifest_anchors.append({
            "feature_name": feature_name,
            "feature_family": spec["feature_family"],
            "unit": spec["unit"],
            "citation_id": spec["citation_id"],
            "metric": spec["metric"],
            "tolerance": float(spec["tolerance"]),
            "target_sample_count": len(target_samples),
            "observed_sample_count": len(observed_samples),
            "target_min": float(min(target_samples)),
            "target_max": float(max(target_samples)),
            "observed_min": float(min(observed_samples)) if observed_samples else None,
            "observed_max": float(max(observed_samples)) if observed_samples else None,
            "source_boundary": spec["source_boundary"],
            "status": status,
        })

    artifact_status = "pass" if len(anchors) >= 3 and all(row["status"] == "pass" for row in distance_rows) else "fail"
    summary = {
        "status": artifact_status,
        "calibration_anchor_status": "reference_only" if artifact_status == "pass" else "reference_only_artifact_fail",
        "validation_tier": "unvalidated/basic synthetic public-proxy benchmark",
        "current_promotion_status": "not_promoted_public_proxy_reference_only",
        "anchor_count": len(anchors),
        "min_samples_per_distribution": 32,
        "distance_failures": [row["feature_name"] for row in distance_rows if row["status"] != "pass"],
        "claim_boundary": "Calibration artifacts use strict-open reference target distributions and synthetic observations only; they do not promote the smoke benchmark to measured-anchored current.",
    }
    report = {"anchors": anchors, "observations": observations}
    manifest = {
        "artifact": "calibration_anchor_manifest",
        "benchmark_profile": "ml-training-three-tier",
        "validation_tier": summary["validation_tier"],
        "calibration_anchor_status": summary["calibration_anchor_status"],
        "artifact_integrity_status": artifact_status,
        "current_promotion_status": summary["current_promotion_status"],
        "claim_boundary": summary["claim_boundary"],
        "rust_schema_compatibility": {
            "calibration_report": "calibration_report.json",
            "shape": {"anchors": "Vec<CalibrationDistributionAnchor>", "observations": "Vec<CalibrationDistributionObservation>"},
            "metrics": ["wasserstein_1", "kolmogorov_smirnov"],
            "min_anchors": 3,
            "min_samples_per_distribution": 32,
        },
        "source_policy": [
            "Target samples are broad public-proxy reference distributions, not measured traces.",
            "Observed samples are synthetic detector-facing or operational smoke outputs.",
            "No raw captures, proprietary sensor data, deployment metadata, or exact platform signatures are included.",
        ],
        "anchors": manifest_anchors,
    }
    return report, manifest, pd.DataFrame(distance_rows), summary
