"""Low-level signal-synthesis helpers for the v1 benchmark generators.

Extracted from ml_training_generators to keep each module ≤350 LOC.
Contains: numeric utilities, kinematics, radar-equation, and two-ray loss.
"""

from __future__ import annotations

import math

import numpy as np

from ml_training_config import (
    FAMILY_TRAITS,
    FRAME_PERIOD_S,
    RCS_TABLE_DB,
    FamilyTraits,
    SensorArchetype,
    SiteArchetype,
)


def stable_seed(seed: int, *parts: int) -> int:
    value = int(seed) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        mix = int(part) + 0x9E3779B97F4A7C15 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2)
        value ^= mix & 0xFFFFFFFFFFFFFFFF
        value &= 0xFFFFFFFFFFFFFFFF
    return int(value % (2**63 - 1))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(values)))


def uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def correlated_noise(rng: np.random.Generator, n: int, sigma: float, alpha: float = 0.82) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    innovation = rng.normal(0.0, sigma, n).astype(np.float32)
    for idx in range(1, n):
        out[idx] = alpha * out[idx - 1] + innovation[idx]
    return out


def rcs_lookup_dbsm(family: str, aspect: str, sensor: SensorArchetype) -> float:
    table = RCS_TABLE_DB[family][aspect]
    base = float(table.get(sensor.band, table["X"]))
    if sensor.polarization == "HV":
        base -= 1.5
    return base


def phase_profile(phase_id: str) -> dict[str, float]:
    if phase_id == "initial_take_up":
        return {"occlusion": 1.0, "clutter": 1.35, "multipath": 1.40, "threshold": 1.35}
    if phase_id == "climb_transition":
        return {"occlusion": 0.52, "clutter": 1.00, "multipath": 1.08, "threshold": 1.05}
    return {"occlusion": 0.20, "clutter": 0.78, "multipath": 0.82, "threshold": 0.86}


def speed_profile_for_baseline(rng: np.random.Generator, t_abs: np.ndarray) -> np.ndarray:
    cruise_speed = float(rng.triangular(45.0, 52.5, 60.0))
    launch_speed = uniform(rng, (8.0, 18.0))
    transition_speed = uniform(rng, (28.0, 42.0))
    return np.interp(
        t_abs,
        [0.0, 30.0, 90.0, max(float(t_abs[-1]), 91.0)],
        [launch_speed, transition_speed, cruise_speed, cruise_speed],
    ).astype(np.float32)


def kinematics(
    rng: np.random.Generator,
    family: str,
    traits: FamilyTraits,
    t_abs: np.ndarray,
    base_range_m: float,
    radial_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if family == "public_proxy_pusher_prop_baseline":
        true_speed = speed_profile_for_baseline(rng, t_abs)
        altitude = np.interp(
            t_abs,
            [0.0, 30.0, 90.0, max(float(t_abs[-1]), 91.0)],
            [
                uniform(rng, (2.0, 10.0)),
                uniform(rng, (18.0, 65.0)),
                uniform(rng, (140.0, 520.0)),
                uniform(rng, (220.0, 1_400.0)),
            ],
        ).astype(np.float32)
    else:
        true_speed = np.full(t_abs.size, uniform(rng, traits.speed_mps), dtype=np.float32)
        altitude = np.full(t_abs.size, uniform(rng, traits.altitude_m), dtype=np.float32)
        if traits.stationary:
            radial_fraction *= 0.05
    if family == "fast_jet_owa_public_proxy":
        altitude += np.linspace(120.0, 420.0, t_abs.size, dtype=np.float32)
    if family in {"ground_vehicle", "multipath_ghost", "clutter_only", "rfi_burst"}:
        altitude += 8.0 * np.sin(t_abs / 9.0 + rng.uniform(0.0, 6.28))
    else:
        altitude += 22.0 * np.sin(t_abs / rng.uniform(15.0, 38.0) + rng.uniform(0.0, 6.28))
    altitude += correlated_noise(rng, t_abs.size, sigma=5.5, alpha=0.91)
    radial_velocity = true_speed * radial_fraction
    radial_velocity += correlated_noise(rng, t_abs.size, sigma=0.46, alpha=0.82)
    range_m = base_range_m + np.cumsum(radial_velocity * FRAME_PERIOD_S).astype(np.float32)
    return (
        np.clip(range_m, 100.0, 18_000.0),
        radial_velocity.astype(np.float32),
        np.clip(altitude, 0.0, 3_500.0),
        true_speed.astype(np.float32),
    )


def radar_equation_snr_db(
    sensor: SensorArchetype,
    range_m: np.ndarray,
    rcs_dbsm: np.ndarray,
    cpi_pulses: int,
    propagation_loss_db: np.ndarray,
    clutter_loss_db: np.ndarray,
) -> np.ndarray:
    wavelength_m = 299_792_458.0 / sensor.frequency_hz
    boltzmann = 1.380649e-23
    noise_power_dbw = 10.0 * math.log10(boltzmann * 290.0 * sensor.bandwidth_hz) + sensor.noise_figure_db
    processing_gain_db = 10.0 * math.log10(max(1, cpi_pulses))
    geometric_loss_db = 30.0 * math.log10(4.0 * math.pi) + 40.0 * np.log10(np.maximum(range_m, 100.0))
    received_power_dbw = (
        sensor.peak_power_dbw
        + sensor.tx_gain_dbi
        + sensor.rx_gain_dbi
        + 20.0 * math.log10(wavelength_m)
        + rcs_dbsm
        - geometric_loss_db
        - propagation_loss_db
        - clutter_loss_db
        - sensor.system_loss_db
    )
    return (received_power_dbw - noise_power_dbw + processing_gain_db).astype(np.float32)


def two_ray_loss_db(
    sensor: SensorArchetype,
    site: SiteArchetype,
    range_m: np.ndarray,
    altitude_m: np.ndarray,
) -> np.ndarray:
    wavelength_m = 299_792_458.0 / sensor.frequency_hz
    phase = 4.0 * math.pi * site.radar_height_m * np.maximum(1.0, altitude_m) / (
        wavelength_m * np.maximum(range_m, 100.0)
    )
    fading = np.abs(np.sin(phase))
    loss = -20.0 * np.log10(np.clip(0.18 + (1.0 - site.two_ray_weight) * 0.30 + fading, 0.18, 1.0))
    return np.clip(loss, 0.0, 16.0).astype(np.float32)
