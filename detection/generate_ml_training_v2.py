#!/usr/bin/env python3
"""Generate the v2 difficulty-controlled radar detection benchmark.

The benchmark is a strict-open synthetic public-proxy corpus. Public radar
literature is used only as distribution-metadata context; no measured traces
or proprietary-equivalent claims are emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FRAME_PERIOD_S = 0.5
MAX_TIME_S = 45.0
FRAME_COUNT = int(MAX_TIME_S / FRAME_PERIOD_S) + 1
DEFAULT_OUT_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
DEFAULT_RECORDS = 50_000
SPLIT_TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SINGLE_FEATURE_AUC_GATE = 0.85
NEGATIVE_CONTROL_AUC_GATE = 0.60
HELDOUT_STRATA = set(range(45, 50))
HELDOUT_CONFUSER_FAMILIES = {"kite", "balloon", "wind_turbine", "multipath_ghost", "terrain_glint"}

FRAME_COLUMNS = [
    "time_s",
    "cpi_pulses",
    "range_m",
    "radial_velocity_mps",
    "altitude_m",
    "snr_db",
    "cfar_statistic",
    "cfar_threshold",
    "cfar_detected",
    "tbd_track_score",
    "local_noise_floor_db",
    "doppler_scr",
    "rfi_pressure",
    "dropout_fraction",
    "phase_impairment_rad",
    "amplitude_impairment",
    "micro_doppler_energy",
    "micro_doppler_peak_hz_proxy",
    "micro_doppler_bandwidth_hz_proxy",
    "stft_energy",
    "weighted_spectrum_peak",
    "cepstrum_peak",
    "cadence_velocity_peak",
    "range_time_energy",
    "doppler_time_energy",
    "range_doppler_time_energy",
    "normalized_snr",
]
FRAME_INDEX = {name: idx for idx, name in enumerate(FRAME_COLUMNS)}


@dataclass(frozen=True)
class ScenarioStratum:
    stratum_id: str
    wave_index: int
    difficulty_bucket: str
    sensor_band: str
    range_bin: str
    range_m_min: float
    range_m_max: float
    grazing_angle_deg: float
    clutter_regime: str
    target_aspect: str
    motion_pattern: str
    interference: str
    confuser_family: str
    holdout_role: str


DIFFICULTY = {
    "easy": {"snr_center": 8.0, "dropout": 0.04, "range": (800.0, 2_400.0)},
    "medium": {"snr_center": 3.0, "dropout": 0.09, "range": (1_800.0, 5_500.0)},
    "hard": {"snr_center": -1.8, "dropout": 0.16, "range": (3_800.0, 9_500.0)},
    "barely_visible": {"snr_center": -5.2, "dropout": 0.26, "range": (7_000.0, 15_000.0)},
}

BAND_ADJUST_DB = {
    "L": -1.0,
    "S": -0.4,
    "C": 0.0,
    "X": 0.4,
    "Ku": 0.2,
    "Ka": -0.8,
}

BAND_CENTER_HZ = {
    "L": 1.3e9,
    "S": 3.0e9,
    "C": 5.6e9,
    "X": 9.6e9,
    "Ku": 15.0e9,
    "Ka": 34.0e9,
}

CLUTTER = {
    "low_ground_weibull": {"loss": 0.8, "shape": 1.4, "threshold": 0.6, "glint": 0.10},
    "urban_edge_k": {"loss": 1.4, "shape": 0.9, "threshold": 1.0, "glint": 0.22},
    "vegetation_motion": {"loss": 1.1, "shape": 1.1, "threshold": 0.8, "glint": 0.18},
    "sea_clutter": {"loss": 1.7, "shape": 0.8, "threshold": 1.2, "glint": 0.25},
    "terrain_glints": {"loss": 1.3, "shape": 0.7, "threshold": 0.9, "glint": 0.35},
    "rain_cell": {"loss": 1.9, "shape": 1.0, "threshold": 1.4, "glint": 0.20},
    "dust_weather": {"loss": 1.6, "shape": 1.2, "threshold": 1.0, "glint": 0.16},
    "open_sky": {"loss": 0.1, "shape": 1.8, "threshold": 0.1, "glint": 0.04},
}

INTERFERENCE = {
    "none": {"rfi": 0.03, "dropout": 0.00, "phase": 0.01, "agc": 0.02},
    "rfi_burst": {"rfi": 0.36, "dropout": 0.05, "phase": 0.04, "agc": 0.05},
    "agc_compression": {"rfi": 0.10, "dropout": 0.02, "phase": 0.02, "agc": 0.15},
    "dropped_cpi": {"rfi": 0.08, "dropout": 0.16, "phase": 0.02, "agc": 0.04},
    "prf_ambiguity": {"rfi": 0.12, "dropout": 0.04, "phase": 0.03, "agc": 0.05},
    "doppler_folding": {"rfi": 0.11, "dropout": 0.03, "phase": 0.03, "agc": 0.04},
    "clock_drift": {"rfi": 0.06, "dropout": 0.02, "phase": 0.09, "agc": 0.03},
    "calibration_offset": {"rfi": 0.04, "dropout": 0.01, "phase": 0.04, "agc": 0.08},
    "multipath_masking": {"rfi": 0.08, "dropout": 0.08, "phase": 0.06, "agc": 0.04},
}

FAMILY_TRAITS = {
    "public_proxy_fixed_wing": {
        "class_id": "shahed136-public-proxy-fixed-wing-v1",
        "speed": (24.0, 58.0),
        "altitude": (60.0, 1_600.0),
        "rcs": (-10.0, 2.0),
        "micro_peak": (45.0, 170.0),
        "micro_bw": (30.0, 130.0),
        "micro_amp": (0.34, 0.90),
        "coherence": (0.48, 0.92),
        "stationary": False,
    },
    "rc_fixed_wing": {
        "class_id": "hard-negative-rc-fixed-wing-v1",
        "speed": (14.0, 42.0),
        "altitude": (20.0, 550.0),
        "rcs": (-16.0, -2.0),
        "micro_peak": (55.0, 190.0),
        "micro_bw": (35.0, 145.0),
        "micro_amp": (0.30, 0.85),
        "coherence": (0.38, 0.88),
        "stationary": False,
    },
    "hobby_glider": {
        "class_id": "hard-negative-hobby-glider-v1",
        "speed": (8.0, 28.0),
        "altitude": (35.0, 900.0),
        "rcs": (-18.0, -3.0),
        "micro_peak": (4.0, 40.0),
        "micro_bw": (18.0, 95.0),
        "micro_amp": (0.06, 0.34),
        "coherence": (0.42, 0.90),
        "stationary": False,
    },
    "bird_flock": {
        "class_id": "hard-negative-bird-flock-dense-v1",
        "speed": (6.0, 25.0),
        "altitude": (10.0, 650.0),
        "rcs": (-17.0, 1.0),
        "micro_peak": (4.0, 22.0),
        "micro_bw": (25.0, 130.0),
        "micro_amp": (0.18, 0.68),
        "coherence": (0.20, 0.72),
        "stationary": False,
    },
    "single_bird": {
        "class_id": "hard-negative-bird-single-small-v1",
        "speed": (5.0, 22.0),
        "altitude": (8.0, 450.0),
        "rcs": (-23.0, -7.0),
        "micro_peak": (5.0, 28.0),
        "micro_bw": (20.0, 110.0),
        "micro_amp": (0.12, 0.55),
        "coherence": (0.20, 0.70),
        "stationary": False,
    },
    "bat_insect_cloud": {
        "class_id": "hard-negative-bat-insect-cloud-v1",
        "speed": (1.0, 13.0),
        "altitude": (1.0, 140.0),
        "rcs": (-25.0, -8.0),
        "micro_peak": (15.0, 90.0),
        "micro_bw": (40.0, 180.0),
        "micro_amp": (0.12, 0.70),
        "coherence": (0.08, 0.45),
        "stationary": False,
    },
    "kite": {
        "class_id": "hard-negative-kite-v1",
        "speed": (-2.0, 8.0),
        "altitude": (15.0, 280.0),
        "rcs": (-16.0, -1.0),
        "micro_peak": (0.5, 12.0),
        "micro_bw": (8.0, 70.0),
        "micro_amp": (0.05, 0.32),
        "coherence": (0.12, 0.55),
        "stationary": False,
    },
    "balloon": {
        "class_id": "hard-negative-balloon-weather-v1",
        "speed": (-4.0, 12.0),
        "altitude": (60.0, 2_400.0),
        "rcs": (-12.0, 5.0),
        "micro_peak": (0.0, 8.0),
        "micro_bw": (5.0, 50.0),
        "micro_amp": (0.02, 0.20),
        "coherence": (0.18, 0.62),
        "stationary": False,
    },
    "windborne_debris": {
        "class_id": "hard-negative-plastic-bag-debris-v1",
        "speed": (0.0, 22.0),
        "altitude": (0.0, 180.0),
        "rcs": (-24.0, -5.0),
        "micro_peak": (2.0, 45.0),
        "micro_bw": (20.0, 160.0),
        "micro_amp": (0.08, 0.52),
        "coherence": (0.05, 0.45),
        "stationary": False,
    },
    "ground_vehicle": {
        "class_id": "hard-negative-ground-vehicle-v1",
        "speed": (0.0, 32.0),
        "altitude": (0.0, 4.0),
        "rcs": (-8.0, 8.0),
        "micro_peak": (4.0, 80.0),
        "micro_bw": (20.0, 120.0),
        "micro_amp": (0.10, 0.48),
        "coherence": (0.35, 0.92),
        "stationary": False,
    },
    "tower": {
        "class_id": "hard-negative-cell-tower-v1",
        "speed": (-0.5, 0.5),
        "altitude": (15.0, 120.0),
        "rcs": (-5.0, 9.0),
        "micro_peak": (0.0, 5.0),
        "micro_bw": (3.0, 45.0),
        "micro_amp": (0.02, 0.22),
        "coherence": (0.60, 0.98),
        "stationary": True,
    },
    "wind_turbine": {
        "class_id": "hard-negative-wind-turbine-large-v1",
        "speed": (-1.5, 1.5),
        "altitude": (35.0, 160.0),
        "rcs": (-3.0, 10.0),
        "micro_peak": (25.0, 95.0),
        "micro_bw": (60.0, 220.0),
        "micro_amp": (0.40, 0.92),
        "coherence": (0.40, 0.92),
        "stationary": True,
    },
    "rain_cell": {
        "class_id": "hard-negative-rain-cell-v1",
        "speed": (-6.0, 18.0),
        "altitude": (50.0, 1_800.0),
        "rcs": (-7.0, 8.0),
        "micro_peak": (0.0, 25.0),
        "micro_bw": (80.0, 260.0),
        "micro_amp": (0.15, 0.75),
        "coherence": (0.05, 0.45),
        "stationary": False,
    },
    "dust_weather": {
        "class_id": "hard-negative-dust-storm-haze-v1",
        "speed": (-3.0, 20.0),
        "altitude": (0.0, 800.0),
        "rcs": (-12.0, 5.0),
        "micro_peak": (0.0, 18.0),
        "micro_bw": (60.0, 230.0),
        "micro_amp": (0.10, 0.62),
        "coherence": (0.03, 0.40),
        "stationary": False,
    },
    "rfi_burst": {
        "class_id": "hard-negative-rfi-burst-v1",
        "speed": (-45.0, 45.0),
        "altitude": (0.0, 2_000.0),
        "rcs": (-5.0, 8.0),
        "micro_peak": (20.0, 220.0),
        "micro_bw": (120.0, 320.0),
        "micro_amp": (0.30, 0.95),
        "coherence": (0.02, 0.38),
        "stationary": False,
    },
    "terrain_glint": {
        "class_id": "hard-negative-terrain-glint-v1",
        "speed": (-2.0, 2.0),
        "altitude": (0.0, 30.0),
        "rcs": (-4.0, 11.0),
        "micro_peak": (0.0, 20.0),
        "micro_bw": (10.0, 95.0),
        "micro_amp": (0.04, 0.35),
        "coherence": (0.30, 0.85),
        "stationary": True,
    },
    "multipath_ghost": {
        "class_id": "hard-negative-multipath-ghost-v1",
        "speed": (8.0, 58.0),
        "altitude": (0.0, 1_400.0),
        "rcs": (-12.0, 4.0),
        "micro_peak": (20.0, 150.0),
        "micro_bw": (30.0, 160.0),
        "micro_amp": (0.15, 0.70),
        "coherence": (0.12, 0.55),
        "stationary": False,
    },
}

CONFUSER_FAMILIES = [name for name in FAMILY_TRAITS if name != "public_proxy_fixed_wing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--scale-name", default="standard")
    parser.add_argument("--strata", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, *parts: int) -> int:
    value = int(seed) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        mix = (int(part) + 0x9E3779B97F4A7C15 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2))
        value ^= mix & 0xFFFFFFFFFFFFFFFF
        value &= 0xFFFFFFFFFFFFFFFF
    return int(value % (2**63 - 1))


def build_strata(count: int = 50) -> list[ScenarioStratum]:
    if count != 50:
        raise ValueError("ml-training-v2 currently requires exactly 50 strata")
    bands = ["L", "S", "C", "X", "Ku", "Ka"]
    clutter = list(CLUTTER)
    aspects = ["nose", "tail", "broadside", "oblique", "rolling_scintillation"]
    motions = ["straight", "gentle_turn", "descent", "terrain_following", "crossing", "intermittent"]
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
            visible_confusers = [family for family in confusers if family not in HELDOUT_CONFUSER_FAMILIES]
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


def uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def radar_equation_snr_db(
    band: str,
    range_m: np.ndarray,
    rcs_dbsm: float,
    cpi_pulses: int,
    propagation_loss_db: float,
    clutter_loss_db: float,
) -> np.ndarray:
    """Return a first-order monostatic radar-equation SNR diagnostic.

    This is intentionally a transparent public-proxy link budget, not a
    measured-sensor calibration. It makes SNR an auditable derived quantity
    instead of a label-only amplitude knob.
    """
    frequency_hz = BAND_CENTER_HZ[band]
    wavelength_m = 299_792_458.0 / frequency_hz
    peak_power_dbw = 50.0
    tx_gain_dbi = 28.0
    rx_gain_dbi = 28.0
    bandwidth_hz = 2.0e6
    noise_figure_db = 5.5
    system_temperature_k = 290.0
    boltzmann = 1.380649e-23
    noise_power_dbw = 10.0 * math.log10(boltzmann * system_temperature_k * bandwidth_hz) + noise_figure_db
    processing_gain_db = 10.0 * math.log10(max(1, cpi_pulses))
    unmodeled_system_loss_db = 45.0
    geometric_loss_db = 30.0 * math.log10(4.0 * math.pi) + 40.0 * np.log10(np.maximum(range_m, 100.0))
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


def correlated_noise(rng: np.random.Generator, n: int, sigma: float, alpha: float = 0.82) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    innovation = rng.normal(0.0, sigma, n).astype(np.float32)
    for idx in range(1, n):
        out[idx] = alpha * out[idx - 1] + innovation[idx]
    return out


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(values)))


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


def impairment_masks(rng: np.random.Generator, interference: str, base_dropout: float) -> tuple[np.ndarray, np.ndarray]:
    dropout = np.clip(rng.beta(1.4, 12.0, FRAME_COUNT) * 0.55 + base_dropout, 0.0, 0.92).astype(np.float32)
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


def simulate_record(
    rng: np.random.Generator,
    record_id: str,
    record_index: int,
    stratum: ScenarioStratum,
    positive: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    family = choose_family(rng, stratum, positive)
    traits = FAMILY_TRAITS[family]
    clutter = CLUTTER[stratum.clutter_regime]
    interference = INTERFERENCE[stratum.interference]
    difficulty = DIFFICULTY[stratum.difficulty_bucket]
    t = np.arange(FRAME_COUNT, dtype=np.float32) * FRAME_PERIOD_S
    cpi_pulses = int(rng.choice([16, 24, 32, 40, 48, 64]))

    range_start = rng.uniform(stratum.range_m_min, stratum.range_m_max)
    speed = uniform(rng, traits["speed"])
    radial_fraction = rng.uniform(-0.88, 0.88)
    if stratum.motion_pattern == "crossing":
        radial_fraction *= 0.35
    if traits["stationary"]:
        radial_fraction *= 0.08
    radial_velocity = speed * radial_fraction
    if stratum.interference in {"prf_ambiguity", "doppler_folding"} and abs(radial_velocity) > 18.0:
        radial_velocity = ((radial_velocity + 18.0) % 36.0) - 18.0
    turn = correlated_noise(rng, FRAME_COUNT, sigma=0.20, alpha=0.92)
    range_m = range_start + radial_velocity * t + np.cumsum(turn).astype(np.float32)
    range_m = np.clip(range_m, 100.0, 18_000.0)

    altitude = np.full(FRAME_COUNT, uniform(rng, traits["altitude"]), dtype=np.float32)
    if stratum.motion_pattern == "descent":
        altitude += np.linspace(120.0, -180.0, FRAME_COUNT, dtype=np.float32)
    elif stratum.motion_pattern == "terrain_following":
        altitude += 35.0 * np.sin(t / 6.0 + rng.uniform(0.0, 2.0 * math.pi))
    altitude += correlated_noise(rng, FRAME_COUNT, sigma=8.0, alpha=0.94)
    altitude = np.clip(altitude, 0.0, 3_500.0)

    rcs_db = uniform(rng, traits["rcs"]) + aspect_gain_db(stratum.target_aspect, rng)
    propagation_loss = clutter["loss"] + max(0.0, stratum.grazing_angle_deg - 5.0) * 0.08
    clutter_loss = clutter["threshold"] * 2.6 + interference["rfi"] * 5.0
    link_budget_snr = radar_equation_snr_db(
        stratum.sensor_band,
        range_m,
        rcs_db,
        cpi_pulses,
        propagation_loss,
        clutter_loss,
    )
    scene_visibility_margin = difficulty["snr_center"] + BAND_ADJUST_DB[stratum.sensor_band]
    nominal_snr = link_budget_snr + scene_visibility_margin + rng.normal(0.0, 2.6)
    scintillation_sigma = 1.1 + (1.9 if stratum.target_aspect == "rolling_scintillation" else 0.0)
    scintillation = correlated_noise(rng, FRAME_COUNT, sigma=scintillation_sigma, alpha=0.78)
    clutter_glints = rng.weibull(max(0.45, clutter["shape"]), FRAME_COUNT).astype(np.float32) * clutter["glint"] * 4.5
    dropout, occlusion = impairment_masks(rng, stratum.interference, difficulty["dropout"] + interference["dropout"])
    signal_mask = 1.0 - np.clip(dropout * 0.65 + occlusion * 0.55, 0.0, 0.96)
    snr_db = nominal_snr + scintillation + clutter_glints - dropout * rng.uniform(3.0, 7.0) - occlusion * rng.uniform(2.0, 8.0)
    if stratum.interference == "agc_compression":
        snr_db = np.tanh(snr_db / 12.0) * 12.0
    if stratum.interference == "calibration_offset":
        snr_db += rng.normal(-0.5, 1.5)
    if stratum.sensor_band in {"Ku", "Ka"} and stratum.clutter_regime in {"rain_cell", "dust_weather"}:
        snr_db -= rng.uniform(0.6, 2.2)
    if rng.random() < 0.28:
        quant = rng.choice([0.1, 0.2, 0.5])
        snr_db = np.round(snr_db / quant) * quant

    local_noise_floor_db = (
        -42.0
        + clutter["loss"] * 3.2
        + interference["rfi"] * 8.0
        + correlated_noise(rng, FRAME_COUNT, sigma=0.65, alpha=0.86)
    )
    rfi_base = interference["rfi"] + (0.30 if family == "rfi_burst" else 0.0)
    rfi_pressure = np.clip(rfi_base + rng.beta(1.2, 6.0, FRAME_COUNT) * 0.55 + dropout * 0.18, 0.0, 1.0)
    if stratum.interference == "rfi_burst" or family == "rfi_burst":
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(0, FRAME_COUNT - 4))
            width = int(rng.integers(3, 18))
            rfi_pressure[start : min(FRAME_COUNT, start + width)] = np.clip(
                rfi_pressure[start : min(FRAME_COUNT, start + width)] + rng.uniform(0.25, 0.65),
                0.0,
                1.0,
            )

    cfar_threshold = (
        9.2
        + clutter["threshold"] * 1.8
        + rfi_pressure * 2.4
        + dropout * 1.3
        + correlated_noise(rng, FRAME_COUNT, sigma=0.35, alpha=0.84)
    )
    doppler_scr = np.clip(
        snr_db
        - clutter["threshold"] * 1.7
        + np.abs(radial_velocity) * 0.06
        + rng.normal(0.0, 1.2, FRAME_COUNT)
        - rfi_pressure * 1.8,
        -12.0,
        22.0,
    )
    cfar_statistic = snr_db + 0.32 * doppler_scr + clutter_glints * 0.35 + rng.normal(0.0, 1.1, FRAME_COUNT)
    cfar_detected = (cfar_statistic - cfar_threshold + rng.normal(0.0, 0.65, FRAME_COUNT)) > 0.0

    margin = cfar_statistic - cfar_threshold
    coherence = uniform(rng, traits["coherence"])
    tbd_track_score = np.zeros(FRAME_COUNT, dtype=np.float32)
    state = rng.uniform(0.0, 0.05)
    for idx, value in enumerate(margin):
        evidence = max(0.0, float(value)) * (0.030 + coherence * 0.020)
        evidence += max(0.0, 1.0 - dropout[idx]) * coherence * 0.010
        evidence += max(0.0, doppler_scr[idx]) * 0.002
        state = state * (0.84 + 0.08 * coherence) + evidence - dropout[idx] * 0.025 - rfi_pressure[idx] * 0.010
        tbd_track_score[idx] = max(0.0, state)

    micro_peak = uniform(rng, traits["micro_peak"])
    micro_bw = uniform(rng, traits["micro_bw"])
    micro_amp = uniform(rng, traits["micro_amp"])
    if stratum.interference in {"rfi_burst", "doppler_folding"}:
        micro_bw += rng.uniform(20.0, 85.0)
    micro_variation = 1.0 + 0.22 * np.sin(t / rng.uniform(3.5, 9.0) + rng.uniform(0.0, 6.28))
    micro_doppler_energy = np.clip(
        micro_amp * micro_variation * (0.42 + 0.58 * signal_mask) + rfi_pressure * 0.16 + rng.normal(0.0, 0.07, FRAME_COUNT),
        0.0,
        1.6,
    )
    if family in {"rain_cell", "dust_weather", "terrain_glint"}:
        micro_doppler_energy *= rng.uniform(0.65, 1.12)
    micro_peak_series = np.clip(micro_peak + correlated_noise(rng, FRAME_COUNT, sigma=max(1.0, micro_peak * 0.035)), 0.0, 260.0)
    micro_bw_series = np.clip(micro_bw + correlated_noise(rng, FRAME_COUNT, sigma=max(2.0, micro_bw * 0.045)), 1.0, 380.0)

    normalized_snr = np.clip(sigmoid((snr_db - 0.5) / 5.5) + rng.normal(0.0, 0.035, FRAME_COUNT), 0.0, 1.0)
    range_time_energy = np.clip(0.18 + normalized_snr * 0.55 + clutter_glints * 0.045 + rfi_pressure * 0.06, 0.0, 1.6)
    doppler_time_energy = np.clip(0.16 + sigmoid(doppler_scr / 6.5) * 0.52 + micro_doppler_energy * 0.16, 0.0, 1.7)
    range_doppler_time_energy = np.clip(
        0.12 + range_time_energy * 0.46 + doppler_time_energy * 0.38 + cfar_detected.astype(np.float32) * 0.08,
        0.0,
        1.8,
    )
    stft_energy = np.clip(micro_doppler_energy * 0.58 + doppler_time_energy * 0.24 + rng.normal(0.0, 0.035, FRAME_COUNT), 0.0, 1.6)
    weighted_spectrum_peak = np.clip(stft_energy * 0.62 + normalized_snr * 0.24 + rng.normal(0.0, 0.025, FRAME_COUNT), 0.0, 1.6)
    cepstrum_peak = np.clip(micro_doppler_energy * 0.42 + rng.normal(0.0, 0.035, FRAME_COUNT), 0.0, 1.4)
    cadence_velocity_peak = np.clip(np.abs(radial_velocity) * 0.0009 + micro_peak_series / 3_000.0 + rng.normal(0.0, 0.004, FRAME_COUNT), 0.0, 0.24)
    phase_impairment_rad = (
        interference["phase"]
        + rng.normal(0.0, 0.012, FRAME_COUNT)
        + (0.0008 * t if stratum.interference == "clock_drift" else 0.0)
    )
    amplitude_impairment = np.clip(
        1.0 + interference["agc"] * rng.normal(0.0, 0.9, FRAME_COUNT) + rng.normal(0.0, 0.025, FRAME_COUNT),
        0.70,
        1.35,
    )

    frame = np.zeros((FRAME_COUNT, len(FRAME_COLUMNS)), dtype=np.float32)
    frame[:, FRAME_INDEX["time_s"]] = t
    frame[:, FRAME_INDEX["cpi_pulses"]] = float(cpi_pulses)
    frame[:, FRAME_INDEX["range_m"]] = range_m
    frame[:, FRAME_INDEX["radial_velocity_mps"]] = radial_velocity + correlated_noise(rng, FRAME_COUNT, sigma=0.65, alpha=0.70)
    frame[:, FRAME_INDEX["altitude_m"]] = altitude
    frame[:, FRAME_INDEX["snr_db"]] = snr_db
    frame[:, FRAME_INDEX["cfar_statistic"]] = cfar_statistic
    frame[:, FRAME_INDEX["cfar_threshold"]] = cfar_threshold
    frame[:, FRAME_INDEX["cfar_detected"]] = cfar_detected.astype(np.float32)
    frame[:, FRAME_INDEX["tbd_track_score"]] = tbd_track_score
    frame[:, FRAME_INDEX["local_noise_floor_db"]] = local_noise_floor_db
    frame[:, FRAME_INDEX["doppler_scr"]] = doppler_scr
    frame[:, FRAME_INDEX["rfi_pressure"]] = rfi_pressure
    frame[:, FRAME_INDEX["dropout_fraction"]] = dropout
    frame[:, FRAME_INDEX["phase_impairment_rad"]] = phase_impairment_rad
    frame[:, FRAME_INDEX["amplitude_impairment"]] = amplitude_impairment
    frame[:, FRAME_INDEX["micro_doppler_energy"]] = micro_doppler_energy
    frame[:, FRAME_INDEX["micro_doppler_peak_hz_proxy"]] = micro_peak_series
    frame[:, FRAME_INDEX["micro_doppler_bandwidth_hz_proxy"]] = micro_bw_series
    frame[:, FRAME_INDEX["stft_energy"]] = stft_energy
    frame[:, FRAME_INDEX["weighted_spectrum_peak"]] = weighted_spectrum_peak
    frame[:, FRAME_INDEX["cepstrum_peak"]] = cepstrum_peak
    frame[:, FRAME_INDEX["cadence_velocity_peak"]] = cadence_velocity_peak
    frame[:, FRAME_INDEX["range_time_energy"]] = range_time_energy
    frame[:, FRAME_INDEX["doppler_time_energy"]] = doppler_time_energy
    frame[:, FRAME_INDEX["range_doppler_time_energy"]] = range_doppler_time_energy
    frame[:, FRAME_INDEX["normalized_snr"]] = normalized_snr

    mixed_scene = bool(rng.random() < 0.28)
    scene_object_count = int(rng.integers(2, 5)) if mixed_scene else 1
    if mixed_scene:
        frame[:, FRAME_INDEX["rfi_pressure"]] = np.clip(
            frame[:, FRAME_INDEX["rfi_pressure"]] + rng.uniform(0.02, 0.20), 0.0, 1.0
        )
        frame[:, FRAME_INDEX["dropout_fraction"]] = np.clip(
            frame[:, FRAME_INDEX["dropout_fraction"]] + rng.uniform(0.00, 0.12), 0.0, 0.97
        )

    object_seed = int(rng.integers(0, 2**63 - 1))
    scenario_seed = stable_seed(record_index, stratum.wave_index, int(range_start))
    record = {
        "record_id": record_id,
        "record_index": record_index,
        "split": "",
        "class_id": str(traits["class_id"]),
        "target_family": family,
        "is_public_proxy_positive": bool(positive),
        "is_hard_negative": not positive,
        "hard_negative_family": "" if positive else family,
        "scenario_seed": scenario_seed,
        "object_seed": object_seed,
        "frame_count": FRAME_COUNT,
        "cpi_pulses": cpi_pulses,
        "tensor_dir": "",
        "streaming_features_path": "",
        "frame_labels_path": "",
        "truth_metadata_path": "",
        "detector_events_path": "",
        "feature_family_availability_path": "",
        "micro_doppler_dir": "",
        "multi_view_dir": "",
        "learned_windows_dir": "",
        "stratum_id": stratum.stratum_id,
        "wave_index": stratum.wave_index,
        "difficulty_bucket": stratum.difficulty_bucket,
        "sensor_band": stratum.sensor_band,
        "range_bin": stratum.range_bin,
        "grazing_angle_deg": stratum.grazing_angle_deg,
        "clutter_regime": stratum.clutter_regime,
        "target_aspect": stratum.target_aspect,
        "motion_pattern": stratum.motion_pattern,
        "interference": stratum.interference,
        "confuser_family": "" if positive else family,
        "scene_object_count": scene_object_count,
        "mixed_scene": mixed_scene,
        "holdout_role": stratum.holdout_role,
        "nominal_snr_db": float(np.mean(nominal_snr)),
        "link_budget_snr_db": float(np.mean(link_budget_snr)),
        "propagation_loss_db": float(propagation_loss),
        "clutter_loss_db": float(clutter_loss),
        "mean_range_m": float(np.mean(range_m)),
        "mean_abs_radial_velocity_mps": float(np.mean(np.abs(frame[:, FRAME_INDEX["radial_velocity_mps"]]))),
        "calibration_anchor_ids": "scientific-data-2026-drone-radar-rf;rahman-robertson-drone-bird-micro-doppler;low-grazing-uav-detection-cfar-micro-doppler",
    }
    return record, frame


def assign_splits(records: list[dict[str, Any]], seed: int) -> None:
    n = len(records)
    target_counts = {
        "train": int(round(n * SPLIT_TARGETS["train"])),
        "validation": int(round(n * SPLIT_TARGETS["validation"])),
    }
    target_counts["test"] = n - target_counts["train"] - target_counts["validation"]
    for record in records:
        if int(record["wave_index"]) in HELDOUT_STRATA:
            record["split"] = "test"
    counts = {"train": 0, "validation": 0, "test": sum(1 for r in records if r["split"] == "test")}
    rng = np.random.default_rng(seed + 17)
    remaining = [idx for idx, record in enumerate(records) if not record["split"]]
    remaining = [remaining[idx] for idx in rng.permutation(len(remaining))]
    for split in ["validation", "test", "train"]:
        need = max(0, target_counts[split] - counts[split])
        for idx in remaining[:need]:
            records[idx]["split"] = split
        remaining = remaining[need:]
        counts[split] = target_counts[split]
    for idx in remaining:
        records[idx]["split"] = "train"


def aggregate_frame_features(frames: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    for col_name in FRAME_COLUMNS:
        if col_name == "time_s":
            continue
        values = frames[:, :, FRAME_INDEX[col_name]]
        for suffix, arr in [
            ("mean", values.mean(axis=1)),
            ("std", values.std(axis=1)),
            ("min", values.min(axis=1)),
            ("max", values.max(axis=1)),
            ("last", values[:, -1]),
        ]:
            if len(rows) < len(names) + 1:
                pass
        if not names:
            names = []
    feature_values = []
    feature_names = []
    for col_name in FRAME_COLUMNS:
        if col_name == "time_s":
            continue
        values = frames[:, :, FRAME_INDEX[col_name]]
        stats = {
            f"{col_name}_mean": values.mean(axis=1),
            f"{col_name}_std": values.std(axis=1),
            f"{col_name}_min": values.min(axis=1),
            f"{col_name}_max": values.max(axis=1),
            f"{col_name}_last": values[:, -1],
            f"{col_name}_q25": np.quantile(values, 0.25, axis=1),
            f"{col_name}_q75": np.quantile(values, 0.75, axis=1),
        }
        for name, value in stats.items():
            feature_names.append(name)
            feature_values.append(value.astype(np.float32))
    matrix = np.stack(feature_values, axis=1)
    return pd.DataFrame(matrix, columns=feature_names), feature_names


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    auc = float(roc_auc_score(labels, score))
    return max(auc, 1.0 - auc)


def overlap_coefficient(pos: np.ndarray, neg: np.ndarray, bins: int = 40) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    lo = float(min(np.min(pos), np.min(neg)))
    hi = float(max(np.max(pos), np.max(neg)))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return 1.0
    pos_hist, edges = np.histogram(pos, bins=bins, range=(lo, hi), density=True)
    neg_hist, _ = np.histogram(neg, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(pos_hist, neg_hist)) * width)


def probe_auc(X: np.ndarray, y: np.ndarray, split: np.ndarray) -> float:
    train = split == "train"
    test = split == "test"
    if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
        return float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=136),
    )
    model.fit(X[train], y[train])
    return float(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))


def one_hot_frame(records: pd.DataFrame, categorical: list[str], numeric: list[str]) -> np.ndarray:
    parts = []
    if categorical:
        parts.append(pd.get_dummies(records[categorical].astype(str), dtype=np.float32).to_numpy(np.float32))
    if numeric:
        parts.append(records[numeric].astype(np.float32).to_numpy(np.float32))
    if not parts:
        return np.zeros((len(records), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def negative_control_audit(records: pd.DataFrame, feature_df: pd.DataFrame, labels: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(20260518)
    shuffled = labels.copy()
    for split_name in ["train", "validation", "test"]:
        mask = split == split_name
        shuffled[mask] = rng.permutation(shuffled[mask])
    allowed_metadata = one_hot_frame(
        records,
        categorical=[
            "stratum_id",
            "difficulty_bucket",
            "sensor_band",
            "range_bin",
            "clutter_regime",
            "target_aspect",
            "motion_pattern",
            "interference",
            "mixed_scene",
        ],
        numeric=["wave_index", "grazing_angle_deg", "scene_object_count"],
    )
    seed_hash = np.stack(
        [
            records["scenario_seed"].astype(np.uint64).to_numpy() % 997,
            records["object_seed"].astype(np.uint64).to_numpy() % 991,
            records["record_index"].astype(np.uint64).to_numpy() % 983,
        ],
        axis=1,
    ).astype(np.float32)
    row_index = records[["record_index"]].astype(np.float32).to_numpy()
    sensor_features = feature_df.to_numpy(np.float32)
    probes = {
        "allowed_metadata_only_auc": probe_auc(allowed_metadata, labels, split),
        "seed_hash_only_auc": probe_auc(seed_hash, labels, split),
        "row_index_only_auc": probe_auc(row_index, labels, split),
        "label_shuffled_sensor_feature_auc": probe_auc(sensor_features, shuffled, split),
    }
    probes["gate"] = NEGATIVE_CONTROL_AUC_GATE
    probes["status"] = "pass" if all(
        (not math.isfinite(value)) or value <= NEGATIVE_CONTROL_AUC_GATE
        for key, value in probes.items()
        if key.endswith("_auc")
    ) else "fail"
    probes["policy"] = (
        "Negative-control probes must stay near chance; high AUC indicates metadata leakage, "
        "seed/split artifacts, row-order artifacts, or preprocessing leakage."
    )
    return probes


def write_diagnostics(out_root: Path, records: pd.DataFrame, frames: np.ndarray, feature_df: pd.DataFrame) -> dict[str, Any]:
    labels = records["is_public_proxy_positive"].astype(bool).to_numpy(dtype=np.int64)
    split = records["split"].astype(str).to_numpy()
    controls = negative_control_audit(records, feature_df, labels, split)
    (out_root / "negative_control_audit.json").write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n")
    feature_rows = []
    max_auc = 0.0
    for name in feature_df.columns:
        values = feature_df[name].to_numpy(np.float32)
        row = {
            "feature": name,
            "auc_abs_all": safe_auc(labels, values),
            "auc_abs_train": safe_auc(labels[split == "train"], values[split == "train"]),
            "auc_abs_validation": safe_auc(labels[split == "validation"], values[split == "validation"]),
            "auc_abs_holdout": safe_auc(labels[split == "test"], values[split == "test"]),
            "positive_q10": float(np.quantile(values[labels == 1], 0.10)),
            "positive_q50": float(np.quantile(values[labels == 1], 0.50)),
            "positive_q90": float(np.quantile(values[labels == 1], 0.90)),
            "confuser_q10": float(np.quantile(values[labels == 0], 0.10)),
            "confuser_q50": float(np.quantile(values[labels == 0], 0.50)),
            "confuser_q90": float(np.quantile(values[labels == 0], 0.90)),
            "overlap_coefficient": overlap_coefficient(values[labels == 1], values[labels == 0]),
        }
        max_auc = max(max_auc, row["auc_abs_all"])
        feature_rows.append(row)
    auc_df = pd.DataFrame(feature_rows).sort_values("auc_abs_all", ascending=False)
    auc_df.to_csv(out_root / "single_feature_auc_audit.csv", index=False, float_format="%.6f")

    balance = (
        records.groupby(["stratum_id", "difficulty_bucket", "split", "is_public_proxy_positive"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["stratum_id", "split", "is_public_proxy_positive"])
    )
    balance.to_csv(out_root / "per_stratum_split_balance.csv", index=False)

    overlap = auc_df[
        [
            "feature",
            "positive_q10",
            "positive_q50",
            "positive_q90",
            "confuser_q10",
            "confuser_q50",
            "confuser_q90",
            "overlap_coefficient",
        ]
    ]
    overlap.to_csv(out_root / "distribution_overlap.csv", index=False, float_format="%.6f")

    metadata_leakage = []
    for col in [
        "class_id",
        "target_family",
        "is_public_proxy_positive",
        "is_hard_negative",
        "hard_negative_family",
        "confuser_family",
        "holdout_role",
    ]:
        unique_by_label = records.groupby("is_public_proxy_positive")[col].nunique(dropna=False).to_dict()
        metadata_leakage.append(
            {
                "column": col,
                "status": "restricted_metadata_not_available_to_consumers",
                "unique_values_by_label": {str(k): int(v) for k, v in unique_by_label.items()},
            }
        )
    frame_product_audit = {
        "max_abs_single_feature_auc": float(max_auc),
        "gate": SINGLE_FEATURE_AUC_GATE,
        "status": "pass" if max_auc <= SINGLE_FEATURE_AUC_GATE else "fail",
        "highest_features": auc_df.head(20).to_dict(orient="records"),
    }
    leakage_report = {
        "metadata_columns": metadata_leakage,
        "frame_products": frame_product_audit,
        "consumer_feature_policy": "Detection scripts consume frame tensor products and numeric aggregates only; label, class, family, split key, and holdout-role metadata are excluded from model features.",
    }
    (out_root / "label_leakage_audit.json").write_text(json.dumps(leakage_report, indent=2, sort_keys=True) + "\n")

    cfar_margin = frames[:, :, FRAME_INDEX["cfar_statistic"]] - frames[:, :, FRAME_INDEX["cfar_threshold"]]
    baseline_score = (
        frames[:, :, FRAME_INDEX["snr_db"]].max(axis=1) * 0.30
        + frames[:, :, FRAME_INDEX["snr_db"]].mean(axis=1) * 0.15
        + cfar_margin.max(axis=1) * 0.25
        + frames[:, :, FRAME_INDEX["tbd_track_score"]].max(axis=1) * 0.30
    )
    baseline_auc = float(roc_auc_score(labels, baseline_score)) if np.unique(labels).size == 2 else float("nan")
    quality = {
        "benchmark_version": "ml-training-v2",
        "record_count": int(len(records)),
        "stratum_count": int(records["stratum_id"].nunique()),
        "frame_count_per_record": FRAME_COUNT,
        "single_feature_auc_gate": SINGLE_FEATURE_AUC_GATE,
        "single_feature_auc_max": float(max_auc),
        "single_feature_auc_gate_status": "pass" if max_auc <= SINGLE_FEATURE_AUC_GATE else "fail",
        "negative_control_auc_gate": NEGATIVE_CONTROL_AUC_GATE,
        "negative_control_status": controls["status"],
        "negative_control_summary": {
            key: value for key, value in controls.items() if key.endswith("_auc")
        },
        "baseline_cfar_tbd_auc_all_records": baseline_auc,
        "split_counts": {str(k): int(v) for k, v in records["split"].value_counts().sort_index().items()},
        "positive_rate_by_split": {
            str(k): float(v) for k, v in records.groupby("split")["is_public_proxy_positive"].mean().sort_index().items()
        },
        "holdout_unseen_strata": sorted(records.loc[records["holdout_role"] != "seen", "stratum_id"].unique().tolist()),
        "holdout_unseen_confuser_families": sorted(HELDOUT_CONFUSER_FAMILIES),
        "status": "pass" if max_auc <= SINGLE_FEATURE_AUC_GATE and controls["status"] == "pass" else "fail",
    }
    (out_root / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    if max_auc > SINGLE_FEATURE_AUC_GATE:
        raise AssertionError(f"single-feature AUC gate failed: {max_auc:.4f} > {SINGLE_FEATURE_AUC_GATE:.2f}")
    if controls["status"] != "pass":
        raise AssertionError(f"negative-control audit failed: {controls}")
    return quality


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(
    out_root: Path,
    records: pd.DataFrame,
    strata: list[ScenarioStratum],
    feature_names: list[str],
    quality: dict[str, Any],
    scale_name: str,
    seed: int,
) -> None:
    strata_df = pd.DataFrame([asdict(stratum) for stratum in strata])
    strata_df.to_csv(out_root / "scenario_strata.csv", index=False)
    split_manifest = records[
        [
            "record_id",
            "split",
            "scenario_seed",
            "object_seed",
            "class_id",
            "target_family",
            "hard_negative_family",
            "stratum_id",
            "difficulty_bucket",
            "confuser_family",
            "holdout_role",
        ]
    ].copy()
    split_manifest.insert(2, "split_key_kind", "stratum_confuser_holdout")
    split_manifest.insert(
        3,
        "split_key",
        records["stratum_id"].astype(str) + ":" + records["target_family"].astype(str) + ":" + records["holdout_role"].astype(str),
    )
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)

    feature_schema = {
        "benchmark_version": "ml-training-v2",
        "frame_period_s": FRAME_PERIOD_S,
        "frame_count": FRAME_COUNT,
        "frame_columns": FRAME_COLUMNS,
        "aggregate_feature_columns": feature_names,
        "sensor_public_contract": {
            "allowed_to_models": [
                "frame_features.npz frame tensor columns",
                "features.csv aggregate frame observables",
            ],
            "not_allowed_to_models": [
                "records.csv generator metadata",
                "split_manifest.csv split keys",
                "nominal_snr_db",
                "link_budget_snr_db",
                "propagation_loss_db",
                "clutter_loss_db",
                "altitude_m",
                "class_id",
                "target_family",
                "confuser_family",
                "scenario_seed",
                "object_seed",
                "holdout_role",
            ],
        },
        "restricted_metadata_columns": [
            "class_id",
            "target_family",
            "is_public_proxy_positive",
            "is_hard_negative",
            "hard_negative_family",
            "confuser_family",
            "split",
            "holdout_role",
            "nominal_snr_db",
            "link_budget_snr_db",
            "propagation_loss_db",
            "clutter_loss_db",
            "altitude_m",
        ],
        "notes": "Frame products are synthetic public-proxy observables with overlapping target/confuser envelopes; metadata columns are for audit and slicing only.",
    }
    (out_root / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True) + "\n")
    label_schema = {
        "positive_label": "public_proxy_fixed_wing",
        "negative_label": "scenario_confuser_or_sensor_artifact",
        "claim_boundary": "Labels describe synthetic public-proxy benchmark roles, not measured-object truth or proprietary-equivalent sensor behavior.",
    }
    (out_root / "label_schema.json").write_text(json.dumps(label_schema, indent=2, sort_keys=True) + "\n")
    calibration_sources = [
        {
            "id": "scientific-data-2026-drone-radar-rf",
            "title": "Time-synchronized multi-sensor drone radar/RF dataset",
            "url": "https://www.nature.com/articles/s41597-026-06802-6",
            "role": "distribution_anchor_metadata_only",
            "local_data_default": "not_vendored",
            "license_notes": "No source traces copied; local operators must review terms before using external data.",
        },
        {
            "id": "rahman-robertson-drone-bird-micro-doppler",
            "title": "Radar micro-Doppler signatures of drones and birds",
            "url": "https://research-repository.st-andrews.ac.uk/handle/10023/16577",
            "role": "micro_doppler_range_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
        {
            "id": "eusipco-2020-micro-doppler-representations",
            "title": "Comparison of micro-Doppler signal representations",
            "url": "https://eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001561.pdf",
            "role": "representation_family_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
        {
            "id": "low-grazing-uav-detection-cfar-micro-doppler",
            "title": "Low-grazing UAV detection literature on CFAR, clutter, and trajectory extraction",
            "url": "https://arxiv.org/abs/1902.05483",
            "role": "cfar_tbd_failure_mode_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
    ]
    (out_root / "external_calibration_sources.json").write_text(
        json.dumps(calibration_sources, indent=2, sort_keys=True) + "\n"
    )
    calibration_targets = {
        "validation_tier": {
            "target": "V0/V1 public-proxy simulation with explicit assumptions; not measured-truth validation.",
            "check_file": "science_assumptions.json",
        },
        "link_budget": {
            "target": "SNR diagnostics derive from a transparent monostatic radar-equation budget plus public-proxy scene losses; SNR diagnostics are not model features.",
            "check_file": "records.csv",
        },
        "snr_db": {
            "target": "Overlapping easy/medium/hard/barely-visible target and confuser ranges; not fitted to measured truth.",
            "check_file": "distribution_overlap.csv",
        },
        "micro_doppler_energy": {
            "target": "Fixed-wing, RC fixed-wing, bird/flock, turbine, weather, and RFI envelopes overlap in aggregate energy and bandwidth.",
            "check_file": "single_feature_auc_audit.csv",
        },
        "clutter_and_impairments": {
            "target": "Includes Weibull/K-like clutter pressure, glints, vegetation/weather motion, RFI, AGC, dropped CPIs, ambiguity, folding, quantization, drift, and calibration offsets.",
            "check_file": "scenario_strata.csv",
        },
    }
    (out_root / "calibration_targets.json").write_text(json.dumps(calibration_targets, indent=2, sort_keys=True) + "\n")
    science_assumptions = {
        "validation_tier": "V0/V1 synthetic public-proxy benchmark",
        "claim_boundary": "No measured-target truth, classified fidelity, or proprietary-equivalent sensor behavior is claimed.",
        "radar_equation_budget": {
            "frequency_source": "sensor_band public-proxy center frequency",
            "mode": "monostatic first-order received-power diagnostic",
            "assumed_peak_power_dbw": 50.0,
            "assumed_tx_gain_dbi": 28.0,
            "assumed_rx_gain_dbi": 28.0,
            "assumed_bandwidth_hz": 2.0e6,
            "assumed_noise_figure_db": 5.5,
            "assumed_system_temperature_k": 290.0,
            "assumed_unmodeled_system_loss_db": 45.0,
            "limitations": [
                "No measured antenna pattern",
                "No complex-IQ coherent propagation",
                "No measured clutter map",
                "No validated target RCS table",
            ],
        },
        "horizon_semantics": "Features are causal prefixes with frame time_s <= horizon_s. The benchmark is an accumulated-evidence detector, not a claim of future-event prediction before a measured launch event.",
        "leakage_controls": [
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "per_stratum_split_balance.csv",
        ],
        "deferred_physics": [
            "Complex IQ radar cube",
            "Aspect/frequency/polarization RCS tables",
            "Terrain mesh and two-ray/multipath propagation",
            "Measured-data calibration and V5 holdout",
            "Multi-scan tracker with association and false-track lifecycle",
        ],
    }
    (out_root / "science_assumptions.json").write_text(json.dumps(science_assumptions, indent=2, sort_keys=True) + "\n")
    dataset_card = {
        "dataset_id": f"shahed136-public-proxy-ml-training-v2-{scale_name}",
        "benchmark_version": "ml-training-v2",
        "record_count": int(len(records)),
        "scenario_strata": len(strata),
        "seed": seed,
        "validation_tier": "V0/V1 synthetic public-proxy",
        "strict_open_boundary": "Synthetic public-proxy benchmark; no measured traces, classified fidelity, or proprietary-equivalent behavior claims.",
        "difficulty_policy": "Fifty scenario strata cover sensor band, range, grazing angle, clutter, aspect, motion, interference, confuser family, and visibility bucket combinations.",
        "quality_report": "quality_report.json",
        "diagnostics": [
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "per_stratum_split_balance.csv",
            "distribution_overlap.csv",
        ],
    }
    (out_root / "dataset_card.json").write_text(json.dumps(dataset_card, indent=2, sort_keys=True) + "\n")
    manifest = {
        "dataset_id": dataset_card["dataset_id"],
        "benchmark_version": "ml-training-v2",
        "files": [
            "records.csv",
            "split_manifest.csv",
            "frame_features.npz",
            "features.csv",
            "scenario_strata.csv",
            "quality_report.json",
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "per_stratum_split_balance.csv",
            "distribution_overlap.csv",
            "feature_schema.json",
            "label_schema.json",
            "dataset_card.json",
            "external_calibration_sources.json",
            "calibration_targets.json",
            "science_assumptions.json",
            "runtime_report.json",
        ],
    }
    (out_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    runtime_report = {
        "generator": "detection/generate_ml_training_v2.py",
        "seed": seed,
        "scale_name": scale_name,
        "quality_status": quality["status"],
        "generated_artifact_policy": "outputs/ is gitignored; do not stage generated records, tensors, or model outputs.",
    }
    (out_root / "runtime_report.json").write_text(json.dumps(runtime_report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace generated artifacts")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    strata = build_strata(args.strata)
    records: list[dict[str, Any]] = []
    frames = np.zeros((args.records, FRAME_COUNT, len(FRAME_COLUMNS)), dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    for idx in range(args.records):
        stratum = strata[idx % len(strata)]
        record_rng = np.random.default_rng(stable_seed(args.seed, idx, stratum.wave_index))
        positive = choose_label(record_rng, stratum)
        record_id = f"record_{idx:06d}"
        record, frame = simulate_record(record_rng, record_id, idx, stratum, positive)
        records.append(record)
        frames[idx] = frame
        if (idx + 1) % 10_000 == 0:
            print(f"generated {idx + 1}/{args.records} records", flush=True)
    assign_splits(records, args.seed)
    records_df = pd.DataFrame(records)
    records_df.to_csv(out_root / "records.csv", index=False)
    np.savez_compressed(
        out_root / "frame_features.npz",
        frames=frames,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U32"),
        frame_columns=np.asarray(FRAME_COLUMNS, dtype="<U64"),
        frame_period_s=np.array(FRAME_PERIOD_S, dtype=np.float32),
        benchmark_version=np.asarray(["ml-training-v2"], dtype="<U32"),
    )
    feature_df, feature_names = aggregate_frame_features(frames)
    feature_df.insert(0, "record_id", records_df["record_id"].to_numpy())
    feature_df.to_csv(out_root / "features.csv", index=False, float_format="%.6f")
    quality = write_diagnostics(out_root, records_df, frames, feature_df.drop(columns=["record_id"]))
    write_metadata(out_root, records_df, strata, feature_names, quality, args.scale_name, args.seed)
    print(
        f"wrote {out_root} records={len(records_df)} strata={len(strata)} "
        f"single_feature_auc_max={quality['single_feature_auc_max']:.4f} "
        f"baseline_auc={quality['baseline_cfar_tbd_auc_all_records']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
