"""Physics-tabular feature helpers."""

from __future__ import annotations

from math import exp
from typing import Mapping


PHYSICS_WEIGHTS = {
    "mean_snr_db": 0.045,
    "max_snr_db": 0.055,
    "mean_doppler_scr": 0.030,
    "mean_rfi_pressure": -0.90,
    "dropout_fraction": -1.40,
    "mean_micro_doppler_energy": 1.60,
    "micro_doppler_peak_hz_proxy": 0.0025,
    "micro_doppler_bandwidth_hz_proxy": 0.0015,
    "mean_track_score": 1.75,
    "cfar_detection_fraction": 0.95,
    "first_detectable_frame": -0.010,
}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def physics_score(row: Mapping[str, str]) -> float:
    value = -2.0
    for key, weight in PHYSICS_WEIGHTS.items():
        value += float(row.get(key, 0.0)) * weight
    return sigmoid(value)


def feature_vector(row: Mapping[str, str]) -> dict[str, float]:
    return {key: float(row.get(key, 0.0)) for key in PHYSICS_WEIGHTS}
