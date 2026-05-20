"""Model-facing feature policy for observable-only radar products."""

from __future__ import annotations

from typing import Iterable


MODEL_FEATURE_DENYLIST = {
    "altitude_m",
    "calibration_anchor_ids",
    "class_id",
    "confuser_family",
    "ground_speed_mps",
    "hard_negative_family",
    "holdout_role",
    "link_budget_snr_db",
    "micro_doppler_bandwidth_hz_proxy",
    "micro_doppler_peak_hz_proxy",
    "nominal_snr_db",
    "normalized_snr",
    "object_seed",
    "phase_id",
    "raw_rcs_dbsm",
    "rcs_dbsm",
    "scenario_seed",
    "scene_role",
    "source_metadata",
    "target_family",
    "true_speed_mps",
    "validation_tier",
}


def _matches_stem(name: str, stem: str) -> bool:
    lowered = name.lower()
    stem = stem.lower()
    return (
        lowered == stem
        or lowered.startswith(f"{stem}_")
        or lowered.endswith(f"_{stem}")
        or f"_{stem}_" in lowered
    )


def observable_only_violations(feature_names: Iterable[str]) -> list[str]:
    violations = []
    for name in feature_names:
        if any(_matches_stem(str(name), restricted) for restricted in MODEL_FEATURE_DENYLIST):
            violations.append(str(name))
    return sorted(set(violations))


def require_observable_only(feature_names: Iterable[str]) -> None:
    violations = observable_only_violations(feature_names)
    if violations:
        raise ValueError(f"non-observable or generator-only model features: {violations}")
