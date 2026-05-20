"""Real-anchor prior loading and conservative simulator adjustment helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _clamp(value: float, low: float | None = None, high: float | None = None) -> float:
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def load_real_anchor_priors(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    prior_path = Path(path)
    raw = prior_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if "simulator_priors" in payload:
        priors = payload.get("simulator_priors")
    else:
        priors = payload.get("simulator_knobs")
    if not isinstance(priors, dict):
        raise ValueError("real-anchor prior file must contain simulator_priors")
    family_priors = payload.get("simulator_priors_by_family", {})
    if family_priors is None:
        family_priors = {}
    if not isinstance(family_priors, dict):
        raise ValueError("simulator_priors_by_family must be an object when present")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "source_path": prior_path.as_posix(),
        "source_sha256": digest,
        "format": payload.get("format", "unknown"),
        "dataset_id": payload.get("dataset_id", "unknown"),
        "status": payload.get("status", "unknown"),
        "allowed_claim_level": payload.get("allowed_claim_level", "unknown"),
        "claim_boundary": payload.get(
            "claim_boundary",
            "Measured anchors may adjust public-proxy priors only.",
        ),
        "simulator_priors": priors,
        "simulator_priors_by_family": family_priors,
        "applicability_notes": payload.get("applicability_notes", {}),
    }


def prior_manifest(priors: dict[str, Any] | None) -> dict[str, Any]:
    if not priors:
        return {
            "status": "reference_only",
            "source_path": "",
            "source_sha256": "",
            "simulator_knob_count": 0,
            "policy": "Default synthetic priors were used; no measured-anchor prior file was supplied.",
        }
    return {
        "status": priors.get("status", "unknown"),
        "dataset_id": priors.get("dataset_id", "unknown"),
        "source_path": priors.get("source_path", ""),
        "source_sha256": priors.get("source_sha256", ""),
        "format": priors.get("format", "unknown"),
        "allowed_claim_level": priors.get("allowed_claim_level", "unknown"),
        "simulator_knob_count": len(priors.get("simulator_priors", {})),
        "simulator_family_count": len(priors.get("simulator_priors_by_family", {})),
        "claim_boundary": priors.get("claim_boundary", ""),
        "applicability_notes": priors.get("applicability_notes", {}),
        "policy": (
            "Real-anchor priors affect public-proxy simulator distributions and run "
            "manifests only. They are not detector features and do not validate exact "
            "platform truth, proprietary-equivalent behavior, or classified fidelity."
        ),
    }


def knob_prior(
    priors: dict[str, Any] | None, knob: str, family: str | None = None
) -> dict[str, Any] | None:
    if not priors:
        return None
    if family:
        by_family = priors.get("simulator_priors_by_family", {})
        family_map = by_family.get(family) if isinstance(by_family, dict) else None
        if isinstance(family_map, dict):
            candidate = family_map.get(knob)
            if isinstance(candidate, dict):
                return candidate
    candidate = priors.get("simulator_priors", {}).get(knob)
    return candidate if isinstance(candidate, dict) else None


def prior_offset(priors: dict[str, Any] | None, knob: str, family: str | None = None) -> float:
    prior = knob_prior(priors, knob, family)
    if not prior:
        return 0.0
    try:
        return float(prior.get("offset", 0.0))
    except (TypeError, ValueError):
        return 0.0


def adjust_scalar(
    value: float,
    priors: dict[str, Any] | None,
    knob: str,
    *,
    low: float | None = None,
    high: float | None = None,
    family: str | None = None,
) -> float:
    prior = knob_prior(priors, knob, family)
    if not prior:
        return value
    try:
        shrinkage = _clamp(float(prior.get("shrinkage", 1.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        shrinkage = 1.0
    if "target_center" in prior:
        try:
            target = float(prior["target_center"])
            value = value * (1.0 - shrinkage) + target * shrinkage
        except (TypeError, ValueError):
            pass
    value += prior_offset(priors, knob, family)
    return _clamp(value, low, high)


def adjust_range(
    values: tuple[float, float],
    priors: dict[str, Any] | None,
    knob: str,
    *,
    low: float | None = None,
    high: float | None = None,
    family: str | None = None,
) -> tuple[float, float]:
    prior = knob_prior(priors, knob, family)
    if not prior:
        return values
    lo, hi = float(values[0]), float(values[1])
    try:
        shrinkage = _clamp(float(prior.get("shrinkage", 1.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        shrinkage = 1.0
    if "target_q10" in prior and "target_q90" in prior:
        try:
            target_lo = float(prior["target_q10"])
            target_hi = float(prior["target_q90"])
            lo = lo * (1.0 - shrinkage) + target_lo * shrinkage
            hi = hi * (1.0 - shrinkage) + target_hi * shrinkage
        except (TypeError, ValueError):
            pass
    offset = prior_offset(priors, knob, family)
    lo = _clamp(lo + offset, low, high)
    hi = _clamp(hi + offset, low, high)
    if hi <= lo:
        hi = lo + max(1e-6, abs(lo) * 0.05)
        hi = _clamp(hi, low, high)
    return lo, hi
