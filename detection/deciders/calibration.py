"""Thresholding, calibration, and abstention helpers."""

from __future__ import annotations

from typing import Sequence

from detection.pipeline_contracts.metrics import calibration_bins, threshold_at_precision


def calibration_report(labels: Sequence[float], scores: Sequence[float], *, bins: int = 10) -> dict[str, object]:
    bins_payload = calibration_bins(labels, scores, bins=bins)
    return {
        "bins": bins_payload,
        "threshold_at_90p_precision": threshold_at_precision(labels, scores, 0.90),
    }


def abstention_policy(score: float, lower: float = 0.40, upper: float = 0.60) -> str:
    if lower <= score <= upper:
        return "abstain"
    return "accept" if score > upper else "reject"
