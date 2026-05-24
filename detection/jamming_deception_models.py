"""Strict-open defensive jamming/deception stress models.

The routines here add nondimensional receiver-view artifacts for detector
robustness work. They do not encode real emitter power, timing recipes,
platform tactics, ECCM behavior, or field-performance claims.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


JAMMING_DECEPTION_PROFILES: tuple[str, ...] = (
    "noise_spot",
    "noise_sweep",
    "noise_barrage",
    "pulse_cover",
    "mechanical_chaff_cloud",
    "corner_reflector_decoy",
    "repeater_like_false_target",
    "signature_augmentation",
    "inadvertent_interference",
)

JAMMING_DECEPTION_FAMILIES: dict[str, str] = {
    "noise_spot": "noise_like_electronic",
    "noise_sweep": "noise_like_electronic",
    "noise_barrage": "noise_like_electronic",
    "pulse_cover": "noise_like_electronic",
    "mechanical_chaff_cloud": "mechanical_like",
    "corner_reflector_decoy": "mechanical_like",
    "repeater_like_false_target": "deception_like",
    "signature_augmentation": "deception_like",
    "inadvertent_interference": "inadvertent_interference",
}

SAFE_PUBLIC_PROXY_JD_BOUNDARY = (
    "Synthetic nondimensional detector stress only; no real-power, timing, "
    "antenna, ECCM, route, or deployment guidance."
)


@dataclass(frozen=True)
class JammingDeceptionProfile:
    profile: str
    family: str
    severity: float
    range_extent_bins: int
    pulse_extent: int
    ghost_offset_bins: int
    doppler_shift_norm: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "family": self.family,
            "severity": round(float(self.severity), 6),
            "range_extent_bins": int(self.range_extent_bins),
            "pulse_extent": int(self.pulse_extent),
            "ghost_offset_bins": int(self.ghost_offset_bins),
            "doppler_shift_norm": round(float(self.doppler_shift_norm), 6),
            "boundary": SAFE_PUBLIC_PROXY_JD_BOUNDARY,
        }


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


def validate_jamming_deception_rate(rate: float) -> float:
    parsed = float(rate)
    if parsed == 0.0 or 0.10 <= parsed <= 0.20:
        return parsed
    raise ValueError("jamming-deception-rate must be 0 for ablation or between 0.10 and 0.20")


def jamming_deception_rate_policy(rate: float) -> str:
    parsed = validate_jamming_deception_rate(rate)
    if parsed == 0.0:
        return "ablation_zero"
    return f"official_{parsed:.2f}_scenario_group_rate"


def scenario_jamming_deception_fields(
    *,
    group_index: int,
    seed: int,
    rate: float,
    active_override: bool | None = None,
) -> dict[str, Any]:
    """Return group-level audit fields for a synthetic detector stressor."""

    parsed_rate = validate_jamming_deception_rate(rate)
    policy = jamming_deception_rate_policy(parsed_rate)
    if parsed_rate == 0.0:
        return {
            "jamming_deception_active": 0,
            "jamming_deception_profile": "none",
            "jamming_deception_family": "none",
            "jamming_deception_rate_policy": policy,
        }
    if active_override is None:
        unit = stable_seed(seed, group_index, "jd-active") / float((1 << 63) - 1)
        active = int(unit < parsed_rate)
    else:
        active = int(active_override)
    if not active:
        profile = "none"
        family = "none"
    else:
        profile = JAMMING_DECEPTION_PROFILES[
            stable_seed(seed, group_index, "jd-profile") % len(JAMMING_DECEPTION_PROFILES)
        ]
        family = JAMMING_DECEPTION_FAMILIES[profile]
    return {
        "jamming_deception_active": active,
        "jamming_deception_profile": profile,
        "jamming_deception_family": family,
        "jamming_deception_rate_policy": policy,
    }


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def _profile_from_scenario(
    scenario: dict[str, Any],
    *,
    record_id: str,
    sensor_id: str,
    seed: int,
) -> JammingDeceptionProfile:
    profile = str(scenario.get("jamming_deception_profile", "none") or "none")
    if profile not in JAMMING_DECEPTION_FAMILIES:
        profile = "none"
    family = JAMMING_DECEPTION_FAMILIES.get(profile, "none")
    rng = _rng(seed, scenario.get("scenario_group_id", ""), record_id, sensor_id, "jd-profile")
    severity_by_family = {
        "noise_like_electronic": (0.16, 0.38),
        "mechanical_like": (0.18, 0.44),
        "deception_like": (0.14, 0.34),
        "inadvertent_interference": (0.05, 0.16),
    }
    lo, hi = severity_by_family.get(family, (0.0, 0.0))
    return JammingDeceptionProfile(
        profile=profile,
        family=family,
        severity=float(rng.uniform(lo, hi)),
        range_extent_bins=int(rng.integers(2, 11)),
        pulse_extent=int(rng.integers(3, 14)),
        ghost_offset_bins=int(rng.integers(3, 16)),
        doppler_shift_norm=float(rng.uniform(-0.16, 0.16)),
    )


def _complex_noise(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)


def _gaussian(length: int, center: float, width: float) -> np.ndarray:
    axis = np.arange(length, dtype=np.float64)
    return np.exp(-0.5 * np.square((axis - center) / max(width, 0.8)))


def apply_jamming_deception(
    iq: np.ndarray,
    *,
    record: dict[str, Any],
    scenario: dict[str, Any],
    sensor_id: str,
    seed: int,
) -> tuple[np.ndarray, JammingDeceptionProfile]:
    """Apply an abstract detector stressor to a complex IQ matrix."""

    profile = _profile_from_scenario(
        scenario,
        record_id=str(record.get("record_id", "")),
        sensor_id=sensor_id,
        seed=seed,
    )
    matrix = np.asarray(iq, dtype=np.complex128).copy()
    if int(scenario.get("jamming_deception_active", 0) or 0) == 0 or profile.profile == "none":
        return matrix.astype(np.complex64), profile

    rng = _rng(seed, record.get("record_id", ""), sensor_id, "jd-apply")
    pulses, range_bins = matrix.shape
    pulse_axis = np.arange(pulses, dtype=np.float64)
    range_axis = np.arange(range_bins, dtype=np.float64)
    amp = profile.severity * (0.85 + 0.25 * rng.random())

    if profile.profile == "noise_spot":
        center = float(rng.uniform(0.1, 0.9) * range_bins)
        envelope = _gaussian(range_bins, center, max(1.0, profile.range_extent_bins / 2.5))
        matrix += _complex_noise(rng, matrix.shape, amp) * envelope[None, :]
    elif profile.profile == "noise_sweep":
        start = float(rng.uniform(0.0, range_bins - 1))
        slope = float(rng.choice([-1.0, 1.0]) * range_bins / max(pulses, 1))
        for pulse_idx in range(pulses):
            center = (start + slope * pulse_idx) % range_bins
            envelope = _gaussian(range_bins, center, max(1.0, profile.range_extent_bins / 4.0))
            matrix[pulse_idx, :] += _complex_noise(rng, (range_bins,), amp) * envelope
    elif profile.profile == "noise_barrage":
        range_taper = 0.55 + 0.45 * rng.random((1, range_bins))
        matrix += _complex_noise(rng, matrix.shape, amp * 0.70) * range_taper
    elif profile.profile == "pulse_cover":
        start = int(rng.integers(0, max(1, pulses - profile.pulse_extent + 1)))
        end = min(pulses, start + profile.pulse_extent)
        matrix[start:end, :] += _complex_noise(rng, (end - start, range_bins), amp * 1.25)
    elif profile.profile == "mechanical_chaff_cloud":
        center = float(rng.uniform(0.22, 0.78) * range_bins)
        envelope = _gaussian(range_bins, center, max(4.0, profile.range_extent_bins * 1.8))
        slow_phase = np.exp(1j * 2.0 * np.pi * rng.uniform(-0.018, 0.018) * pulse_axis)
        speckle = 0.55 + 0.45 * rng.random((pulses, range_bins))
        matrix += amp * slow_phase[:, None] * envelope[None, :] * speckle
    elif profile.profile == "corner_reflector_decoy":
        for _ in range(2):
            center = float(rng.uniform(0.12, 0.88) * range_bins)
            envelope = _gaussian(range_bins, center, rng.uniform(1.0, 2.2))
            doppler = rng.uniform(-0.035, 0.035)
            phase = np.exp(1j * 2.0 * np.pi * doppler * pulse_axis)
            matrix += amp * rng.uniform(0.8, 1.4) * phase[:, None] * envelope[None, :]
    elif profile.profile == "repeater_like_false_target":
        source_bin = int(np.argmax(np.mean(np.abs(matrix) ** 2, axis=0)))
        for offset_weight in (1.0, 1.45, 2.1):
            center = float((source_bin + int(profile.ghost_offset_bins * offset_weight)) % range_bins)
            envelope = _gaussian(range_bins, center, rng.uniform(1.1, 2.8))
            phase = np.exp(1j * 2.0 * np.pi * profile.doppler_shift_norm * pulse_axis)
            matrix += amp * rng.uniform(0.35, 0.70) * phase[:, None] * envelope[None, :]
    elif profile.profile == "signature_augmentation":
        source_bin = int(np.argmax(np.mean(np.abs(matrix) ** 2, axis=0)))
        phase = np.exp(1j * 2.0 * np.pi * profile.doppler_shift_norm * pulse_axis)
        for offset, weight in ((-3, 0.55), (3, 0.55), (6, 0.32)):
            envelope = _gaussian(range_bins, (source_bin + offset) % range_bins, 1.5)
            matrix += amp * weight * phase[:, None] * envelope[None, :]
    elif profile.profile == "inadvertent_interference":
        count = max(1, int(0.018 * pulses * range_bins))
        pulse_idx = rng.integers(0, pulses, size=count)
        range_idx = rng.integers(0, range_bins, size=count)
        matrix[pulse_idx, range_idx] += _complex_noise(rng, (count,), amp * 2.0)

    return matrix.astype(np.complex64), profile


def _local_peak_count(image: np.ndarray, threshold: float) -> int:
    if image.shape[0] < 3 or image.shape[1] < 3:
        return int(np.sum(image > threshold))
    center = image[1:-1, 1:-1]
    neighbours = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, :-2],
        image[1:-1, 2:],
        image[2:, :-2],
        image[2:, 1:-1],
        image[2:, 2:],
    ]
    return int(np.sum((center > threshold) & (center >= np.maximum.reduce(neighbours))))


def jamming_deception_features(iq: np.ndarray) -> dict[str, float]:
    """Return detector-facing diagnostics computed only from IQ."""

    matrix = np.asarray(iq, dtype=np.complex128)
    power = np.abs(matrix) ** 2
    rd_power = np.abs(np.fft.fftshift(np.fft.fft2(matrix))) ** 2
    flatness = float(np.exp(np.mean(np.log(rd_power + 1e-9))) / (np.mean(rd_power) + 1e-9))
    range_mass = rd_power.sum(axis=0)
    line_occupancy = float(np.mean(range_mass > np.percentile(range_mass, 84.0)))
    pulse_power = power.mean(axis=1)
    burstiness = float(np.max(pulse_power) / (np.mean(pulse_power) + 1e-9))
    threshold = float(np.percentile(rd_power, 97.5))
    low_half_width = max(1, rd_power.shape[0] // 10)
    center = rd_power.shape[0] // 2
    low_doppler = rd_power[max(0, center - low_half_width) : center + low_half_width + 1, :]
    return {
        "jd_spectral_flatness": flatness,
        "jd_range_line_occupancy": line_occupancy,
        "jd_pulse_burstiness": math.log1p(burstiness),
        "jd_ghost_peak_count": float(_local_peak_count(rd_power, threshold)),
        "jd_low_doppler_cloud_mass": float(np.sum(low_doppler) / (np.sum(rd_power) + 1e-9)),
    }


def jamming_deception_model_card() -> dict[str, Any]:
    return {
        "model_card": "strict_open_jamming_deception_stress",
        "boundary": SAFE_PUBLIC_PROXY_JD_BOUNDARY,
        "activation": "scenario-group-level, independent of label and split",
        "official_rate_policy": "0.10-0.20 active scenario groups; 0.15 default",
        "ablation_rate_policy": "0 disables the lane for ablation only",
        "profiles": list(JAMMING_DECEPTION_PROFILES),
        "families": JAMMING_DECEPTION_FAMILIES,
        "detector_visible_features": list(jamming_deception_features(np.ones((8, 8))).keys()),
        "not_modeled": [
            "real emitter power",
            "J/S computation",
            "burn-through range",
            "pulse-timing prescription",
            "exact ECCM response",
            "evasion optimization",
            "field performance",
        ],
    }
