"""Shared v2 benchmark loading, metrics, feature builders, and report helpers."""

from __future__ import annotations

try:  # direct script entrypoints import this module as a top-level file
    from detection.detection_common_types import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUT_ROOT,
        FRAME_PERIOD_S,
        SINGLE_FEATURE_AUC_GATE,
        TRUTH_LIKE_FRAME_COLUMNS,
    )
    from detection.detection_common_helpers import (
        baseline_scores,
        counts,
        evaluate_scores,
        horizon_slice,
        horizon_label,
        operating_metrics,
        parse_horizons,
        load_records,
        load_frame_store,
        sequence_features,
        select_frames,
        slice_metrics,
        horizon_name,
        tabular_features,
        ensure_output_root,
        track_metrics,
        transformer_features,
        write_auxiliary_reports,
        write_reports,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from detection/
    from detection_common_types import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUT_ROOT,
        FRAME_PERIOD_S,
        SINGLE_FEATURE_AUC_GATE,
        TRUTH_LIKE_FRAME_COLUMNS,
    )
    from detection_common_helpers import (
        baseline_scores,
        counts,
        evaluate_scores,
        horizon_slice,
        horizon_label,
        operating_metrics,
        parse_horizons,
        load_records,
        load_frame_store,
        sequence_features,
        select_frames,
        slice_metrics,
        horizon_name,
        tabular_features,
        ensure_output_root,
        track_metrics,
        transformer_features,
        write_auxiliary_reports,
        write_reports,
    )


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUT_ROOT",
    "FRAME_PERIOD_S",
    "SINGLE_FEATURE_AUC_GATE",
    "TRUTH_LIKE_FRAME_COLUMNS",
    "baseline_scores",
    "counts",
    "evaluate_scores",
    "horizon_slice",
    "horizon_label",
    "operating_metrics",
    "parse_horizons",
    "load_records",
    "load_frame_store",
    "sequence_features",
    "select_frames",
    "slice_metrics",
    "horizon_name",
    "tabular_features",
    "ensure_output_root",
    "track_metrics",
    "transformer_features",
    "write_auxiliary_reports",
    "write_reports",
]
