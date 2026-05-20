"""Tensor and micro-Doppler feature helpers."""

from __future__ import annotations

from typing import Mapping

from .physics import feature_value, sigmoid


TENSOR_WEIGHTS = {
    "mean_micro_doppler_energy": 1.8,
    "stft_energy_mean": 1.1,
    "weighted_spectrum_peak_mean": 0.8,
    "cepstrum_peak_mean": 0.6,
    "cadence_velocity_peak_mean": 0.35,
    "range_doppler_time_energy_mean": 0.55,
    "cfar_detection_fraction": 1.1,
    "mean_track_score": 1.4,
    "mean_snr_db": 0.020,
    "mean_rfi_pressure": -1.0,
    "dropout_fraction": -1.1,
}


def tensor_score(row: Mapping[str, str]) -> float:
    value = -1.75
    for key, weight in TENSOR_WEIGHTS.items():
        value += feature_value(row, key) * weight
    return sigmoid(value)
