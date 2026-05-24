"""Low-level signal-synthesis helpers for the v2 benchmark generators.

Extracted from ml_training_v2_generators to keep each module ≤350 LOC.
Contains: numeric utilities, strata builder, radar-equation, and impairment
masks.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from detection.ml_training_v2_config import (
        BAND_CENTER_HZ,
        CLUTTER,
        DIFFICULTY,
        FAMILY_TRAITS,
        FRAME_COUNT,
        HELDOUT_CONFUSER_FAMILIES,
        HELDOUT_STRATA,
        INTERFERENCE,
        ScenarioStratum,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from ml_training_v2_config import (
        BAND_CENTER_HZ,
        CLUTTER,
        DIFFICULTY,
        FAMILY_TRAITS,
        FRAME_COUNT,
        HELDOUT_CONFUSER_FAMILIES,
        HELDOUT_STRATA,
        INTERFERENCE,
        ScenarioStratum,
    )

CONFUSER_FAMILIES = [name for name in FAMILY_TRAITS if name != "public_proxy_fixed_wing"]


def stable_seed(seed: int, *parts: int) -> int:
    value = int(seed) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        mix = int(part) + 0x9E3779B97F4A7C15 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2)
        value ^= mix & 0xFFFFFFFFFFFFFFFF
        value &= 0xFFFFFFFFFFFFFFFF
    return int(value % (2**63 - 1))


def uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def correlated_noise(
    rng: np.random.Generator, n: int, sigma: float, alpha: float = 0.82
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    innovation = rng.normal(0.0, sigma, n).astype(np.float32)
    for idx in range(1, n):
        out[idx] = alpha * out[idx - 1] + innovation[idx]
    return out


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(values)))


def build_strata(count: int = 50) -> list[ScenarioStratum]:
    if count != 50:
        raise ValueError("ml-training-v2 currently requires exactly 50 strata")
    bands = ["L", "S", "C", "X", "Ku", "Ka"]
    clutter = list(CLUTTER)
    aspects = ["nose", "tail", "broadside", "oblique", "rolling_scintillation"]
    motions = [
        "straight",
        "gentle_turn",
        "descent",
        "terrain_following",
        "crossing",
        "intermittent",
    ]
    interference = list(INTERFERENCE)
    confusers = CONFUSER_FAMILIES
    difficulties = ["easy", "medium", "hard", "barely_visible"]
    strata = []
    for idx in range(count):
        bucket = difficulties[idx % len(difficulties)]
        rmin, rmax = DIFFICULTY[bucket]["range"]
        holdout = "unseen_stratum_confuser" if idx in HELDOUT_STRATA else "seen"
        if idx in HELDOUT_STRATA:
            confuser = sorted(HELDOUT_CONFUSER_FAMILIES)[idx - min(HELDOUT_STRATA)]
        else:
            visible_confusers = [
                family for family in confusers if family not in HELDOUT_CONFUSER_FAMILIES
            ]
            confuser = visible_confusers[(idx * 7 + idx // 3) % len(visible_confusers)]
        strata.append(
            ScenarioStratum(
                stratum_id=f"wave_{idx:02d}",
                wave_index=idx,
                difficulty_bucket=bucket,
                sensor_band=bands[(idx * 5 + 1) % len(bands)],
                range_bin=f"{bucket}_range",
                range_m_min=rmin,
                range_m_max=rmax,
                grazing_angle_deg=float(0.6 + (idx % 10) * 1.8 + (idx // 10) * 0.35),
                clutter_regime=clutter[(idx * 3 + idx // 4) % len(clutter)],
                target_aspect=aspects[(idx * 2 + idx // 5) % len(aspects)],
                motion_pattern=motions[(idx * 4 + 2) % len(motions)],
                interference=interference[(idx * 5 + idx // 2) % len(interference)],
                confuser_family=confuser,
                holdout_role=holdout,
            )
        )
    return strata


def radar_equation_snr_db(
    band: str,
    range_m: np.ndarray,
    rcs_dbsm: float,
    cpi_pulses: int,
    propagation_loss_db: float,
    clutter_loss_db: float,
) -> np.ndarray:
    """Return a first-order monostatic radar-equation SNR diagnostic."""
    frequency_hz = BAND_CENTER_HZ[band]
    wavelength_m = 299_792_458.0 / frequency_hz
    peak_power_dbw = 50.0
    tx_gain_dbi = 28.0
    rx_gain_dbi = 28.0
    bandwidth_hz = 2.0e6
    noise_figure_db = 5.5
    system_temperature_k = 290.0
    boltzmann = 1.380649e-23
    noise_power_dbw = (
        10.0 * math.log10(boltzmann * system_temperature_k * bandwidth_hz) + noise_figure_db
    )
    processing_gain_db = 10.0 * math.log10(max(1, cpi_pulses))
    unmodeled_system_loss_db = 45.0
    geometric_loss_db = 30.0 * math.log10(4.0 * math.pi) + 40.0 * np.log10(
        np.maximum(range_m, 100.0)
    )
    received_power_dbw = (
        peak_power_dbw
        + tx_gain_dbi
        + rx_gain_dbi
        + 20.0 * math.log10(wavelength_m)
        + rcs_dbsm
        - geometric_loss_db
        - propagation_loss_db
        - unmodeled_system_loss_db
    )
    return received_power_dbw - noise_power_dbw + processing_gain_db - clutter_loss_db


def choose_label(rng: np.random.Generator, stratum: ScenarioStratum) -> bool:
    base = 0.36
    if stratum.difficulty_bucket == "barely_visible":
        base = 0.34
    return bool(rng.random() < base)


def choose_family(rng: np.random.Generator, stratum: ScenarioStratum, positive: bool) -> str:
    if positive:
        return "public_proxy_fixed_wing"
    if stratum.wave_index in HELDOUT_STRATA:
        return stratum.confuser_family
    pool = [stratum.confuser_family]
    if rng.random() < 0.35:
        pool.extend(["rc_fixed_wing", "hobby_glider", "bird_flock", "multipath_ghost"])
    if rng.random() < 0.20:
        pool.extend(["ground_vehicle", "rain_cell", "rfi_burst"])
    pool = [family for family in pool if family not in HELDOUT_CONFUSER_FAMILIES]
    return str(rng.choice(pool))


def aspect_gain_db(aspect: str, rng: np.random.Generator) -> float:
    gains = {
        "nose": (-2.8, 1.0),
        "tail": (-3.0, 0.8),
        "broadside": (-0.6, 3.5),
        "oblique": (-1.6, 2.0),
        "rolling_scintillation": (-4.0, 4.0),
    }
    return uniform(rng, gains[aspect])


def impairment_masks(
    rng: np.random.Generator, interference: str, base_dropout: float
) -> tuple[np.ndarray, np.ndarray]:
    dropout = np.clip(rng.beta(1.4, 12.0, FRAME_COUNT) * 0.55 + base_dropout, 0.0, 0.92).astype(
        np.float32
    )
    burst = np.zeros(FRAME_COUNT, dtype=np.float32)
    if interference in {"dropped_cpi", "rfi_burst", "multipath_masking", "agc_compression"}:
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(0, FRAME_COUNT - 3))
            width = int(rng.integers(2, 12))
            burst[start : min(FRAME_COUNT, start + width)] = 1.0
    dropout = np.clip(dropout + burst * rng.uniform(0.22, 0.55), 0.0, 0.96)
    occlusion = np.zeros(FRAME_COUNT, dtype=np.float32)
    if rng.random() < 0.42:
        start = int(rng.integers(4, FRAME_COUNT - 10))
        width = int(rng.integers(5, 20))
        occlusion[start : min(FRAME_COUNT, start + width)] = rng.uniform(0.25, 0.85)
    return dropout, occlusion
