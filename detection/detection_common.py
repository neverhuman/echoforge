"""Shared v2 benchmark loading, metrics, and report helpers."""

from __future__ import annotations

from .detection_common_types import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_ROOT,
    FRAME_PERIOD_S,
    SINGLE_FEATURE_AUC_GATE,
    TRUTH_LIKE_FRAME_COLUMNS,
)
from .detection_common_helpers import (
    horizon_label,
    parse_horizons,
    load_records,
    load_frame_store,
    select_frames,
    horizon_name,
    ensure_output_root,
)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUT_ROOT",
    "FRAME_PERIOD_S",
    "SINGLE_FEATURE_AUC_GATE",
    "TRUTH_LIKE_FRAME_COLUMNS",
    "horizon_label",
    "parse_horizons",
    "load_records",
    "load_frame_store",
    "select_frames",
    "horizon_name",
    "ensure_output_root",
]