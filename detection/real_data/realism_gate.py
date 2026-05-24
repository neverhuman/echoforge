"""Radar-operating realism gate.

AUC is retained as a leakage diagnostic. Promotion depends on calibration,
false-alarm behavior, lifecycle metrics, and domain holdouts.
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_THRESHOLDS = {
    "high_auc": 0.95,
    "max_pfa": 0.01,
    "max_false_track_rate": 0.05,
    "max_missed_track_rate": 0.20,
    "max_track_fragmentation_rate": 2.0,
    "max_calibration_distance": 0.35,
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def evaluate_realism_gate(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    auc = _number(metrics.get("holdout_auc")) or _number(_nested(metrics, "overall", "roc_auc"))
    pfa = _number(metrics.get("pfa_at_1pct_budget")) or _number(metrics.get("pfa"))
    false_track = _number(metrics.get("false_track_rate"))
    missed_track = _number(metrics.get("missed_track_rate"))
    fragmentation = _number(metrics.get("track_fragmentation_rate"))
    calibration_status = str(metrics.get("calibration_anchor_status", "reference_only"))
    calibration_distance = _number(metrics.get("max_calibration_distance"))
    domain_holdout_status = str(metrics.get("domain_holdout_status", "unknown"))

    failures: list[dict[str, Any]] = []
    if calibration_status not in {"pass", "measured_anchor_candidate"}:
        failures.append(
            {
                "gate": "calibration_anchor",
                "value": calibration_status,
                "limit": "pass_or_candidate",
            }
        )
    if (
        calibration_distance is not None
        and calibration_distance > limits["max_calibration_distance"]
    ):
        failures.append(
            {
                "gate": "calibration_distance",
                "value": calibration_distance,
                "limit": limits["max_calibration_distance"],
            }
        )
    if pfa is None or pfa > limits["max_pfa"]:
        failures.append({"gate": "pfa", "value": pfa, "limit": limits["max_pfa"]})
    if false_track is not None and false_track > limits["max_false_track_rate"]:
        failures.append(
            {
                "gate": "false_track_rate",
                "value": false_track,
                "limit": limits["max_false_track_rate"],
            }
        )
    if missed_track is not None and missed_track > limits["max_missed_track_rate"]:
        failures.append(
            {
                "gate": "missed_track_rate",
                "value": missed_track,
                "limit": limits["max_missed_track_rate"],
            }
        )
    if fragmentation is not None and fragmentation > limits["max_track_fragmentation_rate"]:
        failures.append(
            {
                "gate": "track_fragmentation_rate",
                "value": fragmentation,
                "limit": limits["max_track_fragmentation_rate"],
            }
        )
    if domain_holdout_status not in {"pass", "not_applicable"}:
        failures.append({"gate": "domain_holdout", "value": domain_holdout_status, "limit": "pass"})

    high_auc = auc is not None and auc >= limits["high_auc"]
    return {
        "status": "no_go" if failures else "go",
        "auc": auc,
        "auc_role": "leakage_diagnostic_not_headline",
        "high_auc_with_failed_realism": bool(high_auc and failures),
        "failures": failures,
        "thresholds": limits,
        "policy": "High AUC cannot promote a run when calibration, false-alarm, lifecycle, or domain-holdout gates fail.",
    }
