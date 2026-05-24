"""Paper-facing core runtime math for the EchoForge appendix.

Only compact detector, calibration, and fusion routines live here. File I/O,
report builders, JSON/CSV writers, CLI wrappers, and orchestration code are
intentionally excluded from the paper appendix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibratedFusion:
    """Train/CV-fitted linear log-odds fusion model."""

    weights: np.ndarray
    intercept: float


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Scale a detector score with median/MAD robustness."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) + 1.0e-6)
    return np.clip((values - median) / mad, -12.0, 12.0)


def cfar_range_doppler_score(iq: np.ndarray) -> float:
    """Human radar branch: CFAR-like contrast in range-Doppler power."""
    power = np.abs(np.fft.fftshift(np.fft.fft2(iq))) ** 2
    clutter_median = float(np.median(power))
    clutter_mad = float(np.median(np.abs(power - clutter_median)) + 1.0e-6)
    return float((np.max(power) - clutter_median) / clutter_mad)


def mtd_doppler_score(iq: np.ndarray) -> float:
    """Human radar branch: moving-target Doppler concentration."""
    doppler_power = np.abs(np.fft.fft(iq, axis=0)) ** 2
    return float(np.max(doppler_power) / (np.mean(doppler_power) + 1.0e-6))


def track_continuity_score(iq: np.ndarray) -> float:
    """Human cueing branch: range contrast discounted by pulse instability."""
    power = np.abs(iq) ** 2
    range_power = power.mean(axis=0)
    pulse_power = power.mean(axis=1)
    range_contrast = float(np.max(range_power) / (np.mean(range_power) + 1.0e-6))
    pulse_continuity = float(1.0 / (1.0 + np.std(np.diff(pulse_power))))
    return range_contrast * pulse_continuity


def distributed_acoustic_score(acoustic: np.ndarray) -> float:
    """Human cueing branch: propulsion cadence and node agreement."""
    spectrum = np.abs(np.fft.rfft(acoustic, axis=1))
    peak = np.max(spectrum[:, 1:], axis=1)
    mean = np.mean(spectrum[:, 1:], axis=1) + 1.0e-6
    peak_ratio = peak / mean
    node_peaks = np.argmax(spectrum[:, 1:], axis=1).astype(np.float64)
    agreement = 1.0 / (1.0 + np.std(node_peaks))
    return float(np.mean(peak_ratio) * agreement)


def passive_rf_provenance_score(cue_quality: float, rfi_burst: float, no_signal: float) -> float:
    """Human cueing branch: sparse passive-RF provenance, not emitter identity."""
    clean_quality = np.clip(cue_quality, 0.0, 1.0)
    contamination = np.clip(0.65 * rfi_burst + 0.35 * no_signal, 0.0, 1.0)
    return float(np.clip(clean_quality * (1.0 - contamination), 0.0, 1.0))


def accepted_prior_fusion(features: np.ndarray, labels: np.ndarray) -> CalibratedFusion:
    """Human accepted fusion: train/CV-only Gaussian log-odds weights."""
    labels = labels.astype(np.int8)
    positive = features[labels == 1]
    negative = features[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return CalibratedFusion(np.ones(features.shape[1], dtype=np.float64) * 0.1, 0.0)

    positive_mean = np.mean(positive, axis=0)
    negative_mean = np.mean(negative, axis=0)
    pooled_var = np.var(features, axis=0) + 1.0e-3
    weights = (positive_mean - negative_mean) / pooled_var

    prior = (float(len(positive)) + 0.5) / (float(len(labels)) + 1.0)
    midpoint = 0.5 * (positive_mean + negative_mean)
    intercept = math.log(prior / (1.0 - prior)) - float(np.dot(weights, midpoint))
    return CalibratedFusion(weights=weights, intercept=intercept)


def geodesic_odds_calibration(score: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mixed-origin EI calibration: monotone logit-on-logit fit."""
    eps = 1.0e-6
    bounded = np.clip(score, eps, 1.0 - eps)
    geodesic_odds = np.log(bounded / (1.0 - bounded))
    y = labels.astype(np.float64)
    x_mean = float(np.mean(geodesic_odds))
    y_mean = float(np.mean(y))
    slope = float(
        np.dot(geodesic_odds - x_mean, y - y_mean)
        / (np.dot(geodesic_odds - x_mean, geodesic_odds - x_mean) + eps)
    )
    intercept = y_mean - slope * x_mean
    return slope, intercept


def ei_sparse_geodesic_fusion(
    component_scores: Iterable[np.ndarray],
    nonnegative_weights: np.ndarray,
    calibrator: tuple[float, float],
) -> np.ndarray:
    """Generated/evolved EI: sparse nonnegative fusion plus geodesic odds."""
    weights = np.asarray(nonnegative_weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / (float(np.sum(weights)) + 1.0e-12)

    stacked = np.vstack([np.asarray(score, dtype=np.float64) for score in component_scores])
    fused = np.sum(weights[:, None] * stacked, axis=0)
    fused = np.clip(fused, 1.0e-6, 1.0 - 1.0e-6)

    slope, intercept = calibrator
    logit = np.log(fused / (1.0 - fused))
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(slope * logit + intercept, -40.0, 40.0)))
    return calibrated
