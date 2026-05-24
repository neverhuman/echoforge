"""Shared type definitions for detection utilities."""

from __future__ import annotations

DEFAULT_DATA_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
DEFAULT_OUT_ROOT = "outputs/detection"
FRAME_PERIOD_S = 0.5
SINGLE_FEATURE_AUC_GATE = 0.85
TRUTH_LIKE_FRAME_COLUMNS = {"altitude_m"}
MODEL_DENYLIST_FRAME_COLUMNS = {
    "altitude_m",
    "micro_doppler_peak_hz_proxy",
    "micro_doppler_bandwidth_hz_proxy",
    "normalized_snr",
}


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUT_ROOT",
    "FRAME_PERIOD_S",
    "MODEL_DENYLIST_FRAME_COLUMNS",
    "SINGLE_FEATURE_AUC_GATE",
    "TRUTH_LIKE_FRAME_COLUMNS",
]
