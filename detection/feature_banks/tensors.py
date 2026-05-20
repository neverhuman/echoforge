"""Tensor and micro-Doppler feature helpers."""

from __future__ import annotations

from typing import Mapping

from .physics import sigmoid


TENSOR_WEIGHTS = {
    "mean_micro_doppler_energy": 1.8,
    "micro_doppler_peak_hz_proxy": 0.004,
    "micro_doppler_bandwidth_hz_proxy": 0.002,
    "cfar_detection_fraction": 1.1,
    "mean_track_score": 1.4,
    "mean_snr_db": 0.020,
    "mean_rfi_pressure": -1.0,
    "dropout_fraction": -1.1,
}


def tensor_score(row: Mapping[str, str]) -> float:
    value = -1.75
    for key, weight in TENSOR_WEIGHTS.items():
        value += float(row.get(key, 0.0)) * weight
    return sigmoid(value)
