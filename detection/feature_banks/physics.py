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
    "stft_energy_mean": 0.90,
    "weighted_spectrum_peak_mean": 0.70,
    "cepstrum_peak_mean": 0.45,
    "cadence_velocity_peak_mean": 0.30,
    "range_doppler_time_energy_mean": 0.35,
    "mean_track_score": 1.75,
    "cfar_detection_fraction": 0.95,
    "first_detectable_frame": -0.010,
}

FEATURE_ALIASES = {
    "mean_snr_db": ("mean_snr_db", "snr_db_mean"),
    "max_snr_db": ("max_snr_db", "snr_db_max"),
    "mean_doppler_scr": ("mean_doppler_scr", "doppler_scr_mean"),
    "mean_rfi_pressure": ("mean_rfi_pressure", "rfi_pressure_mean"),
    "dropout_fraction": ("dropout_fraction", "dropout_fraction_mean"),
    "mean_micro_doppler_energy": ("mean_micro_doppler_energy", "micro_doppler_energy_mean"),
    "mean_track_score": ("mean_track_score", "tbd_track_score_mean"),
    "cfar_detection_fraction": ("cfar_detection_fraction", "cfar_detected_mean"),
}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def feature_value(row: Mapping[str, str], key: str) -> float:
    for candidate in FEATURE_ALIASES.get(key, (key,)):
        if candidate in row and row[candidate] not in {"", None}:
            return float(row[candidate])
    return 0.0


def physics_score(row: Mapping[str, str]) -> float:
    value = -2.0
    for key, weight in PHYSICS_WEIGHTS.items():
        value += feature_value(row, key) * weight
    return sigmoid(value)


def feature_vector(row: Mapping[str, str]) -> dict[str, float]:
    return {key: feature_value(row, key) for key in PHYSICS_WEIGHTS}
