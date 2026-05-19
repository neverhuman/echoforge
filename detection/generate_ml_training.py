#!/usr/bin/env python3
"""Generate the current three-tier radar realism smoke benchmark.

This is a physics-first benchmark gate, not a measured-radar validation claim.
It publishes detector-facing frame products, operational phase metrics, and
counterfactual audits while keeping generator truth columns out of model
features.
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
DEFAULT_OUT_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-smoke"
DEFAULT_SCENARIO_GROUPS = 64
DEFAULT_MAX_TIME_S = 150.0
NEGATIVE_CONTROL_AUC_GATE = 0.65
TARGET_MASKED_AUC_GATE = 0.65
CALIBRATION_SAMPLE_COUNT = 64
ACOUSTIC_DETECTOR_FAMILY_ID = "acoustic-cueing-network-products"
ACOUSTIC_NODE_IDS = (
    "acoustic_node_north_01",
    "acoustic_node_east_02",
    "acoustic_node_south_03",
    "acoustic_node_west_04",
    "acoustic_node_mast_05",
    "acoustic_node_rooftop_06",
)
ACOUSTIC_FALSE_CUE_SOURCES = (
    "wind_gust",
    "road_traffic",
    "industrial_machinery",
    "generator_hum",
    "thunder_rumble",
)

FRAME_COLUMNS = [
    "time_s",
    "cpi_pulses",
    "range_m",
    "radial_velocity_mps",
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


def stable_seed(seed: int, *parts: int) -> int:
    value = int(seed) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        mix = int(part) + 0x9E3779B97F4A7C15 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2)
        value ^= mix & 0xFFFFFFFFFFFFFFFF
        value &= 0xFFFFFFFFFFFFFFFF
    return int(value % (2**63 - 1))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(values)))


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

RESTRICTED_FEATURE_NAMES = {
    "scenario_seed",
    "object_seed",
    "class_id",
    "target_family",
    "scene_role",
    "phase_id",
    "altitude_m",
    "nominal_snr_db",
    "link_budget_snr_db",
    "raw_rcs_dbsm",
    "rcs_dbsm",
    "validation_tier",
    "calibration_anchor_ids",
    "source_metadata",
    "true_speed_mps",
    "ground_speed_mps",
    "estimated_ground_speed_mps",
}

ROLES = [
    "positive_public_proxy",
    "matched_confuser",
    "target_masked_counterfactual",
    "no_target_counterfactual",
]

CONFUSER_FAMILIES = [
    "bird_flapping",
    "rc_fixed_wing",
    "ground_vehicle",
    "wind_turbine",
    "multipath_ghost",
    "rfi_burst",
]


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    start_s: float
    end_s: float
    radar_meaning: str


@dataclass(frozen=True)
class SiteArchetype:
    site_archetype_id: str
    radar_height_m: float
    terrain_horizon_deg: float
    land_clutter_loss_db: float
    two_ray_weight: float
    multipath_probability: float
    weather_loss_db: float
    scan_gap_probability: float


@dataclass(frozen=True)
class SensorArchetype:
    sensor_archetype_id: str
    band: str
    polarization: str
    frequency_hz: float
    peak_power_dbw: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    bandwidth_hz: float
    noise_figure_db: float
    system_loss_db: float
    scan_revisit_s: float
    dwell_s: float
    cpi_choices: tuple[int, ...]


@dataclass(frozen=True)
class FamilyTraits:
    class_id: str
    speed_mps: tuple[float, float]
    altitude_m: tuple[float, float]
    micro_peak_hz: tuple[float, float]
    micro_bandwidth_hz: tuple[float, float]
    micro_amplitude: tuple[float, float]
    coherence: tuple[float, float]
    stationary: bool = False


@dataclass(frozen=True)
class SpeedPrior:
    prior_id: str
    propulsion_class: str
    role: str
    phase_speed_mps: dict[str, tuple[float, float]]
    cruise_main_estimate_mps: tuple[float, float] | None
    stress_class: bool
    baseline_positive: bool
    policy: str


PHASE_SPECS = [
    PhaseSpec(
        phase_id="initial_take_up",
        start_s=0.0,
        end_s=30.0,
        radar_meaning="geometry and line-of-sight limited initial acquisition",
    ),
    PhaseSpec(
        phase_id="climb_transition",
        start_s=30.0,
        end_s=90.0,
        radar_meaning="track initiation and confirmation under changing aspect",
    ),
    PhaseSpec(
        phase_id="cruise_altitude",
        start_s=90.0,
        end_s=150.0,
        radar_meaning="coherent Doppler and micro-Doppler classification interval",
    ),
]

SITE_ARCHETYPES = [
    SiteArchetype(
        site_archetype_id="gulf_coastal_desert",
        radar_height_m=18.0,
        terrain_horizon_deg=0.7,
        land_clutter_loss_db=2.2,
        two_ray_weight=0.70,
        multipath_probability=0.42,
        weather_loss_db=0.7,
        scan_gap_probability=0.10,
    ),
    SiteArchetype(
        site_archetype_id="gulf_urban_edge",
        radar_height_m=24.0,
        terrain_horizon_deg=1.1,
        land_clutter_loss_db=3.0,
        two_ray_weight=0.58,
        multipath_probability=0.55,
        weather_loss_db=0.5,
        scan_gap_probability=0.13,
    ),
]

SENSOR_ARCHETYPES = [
    SensorArchetype(
        sensor_archetype_id="x_band_medium_revisit",
        band="X",
        polarization="HH",
        frequency_hz=9.6e9,
        peak_power_dbw=62.0,
        tx_gain_dbi=34.0,
        rx_gain_dbi=34.0,
        bandwidth_hz=1.5e6,
        noise_figure_db=4.2,
        system_loss_db=23.0,
        scan_revisit_s=2.0,
        dwell_s=0.75,
        cpi_choices=(24, 32, 40, 48, 64),
    ),
    SensorArchetype(
        sensor_archetype_id="c_band_fast_revisit",
        band="C",
        polarization="HV",
        frequency_hz=5.6e9,
        peak_power_dbw=61.0,
        tx_gain_dbi=32.0,
        rx_gain_dbi=32.0,
        bandwidth_hz=1.8e6,
        noise_figure_db=4.5,
        system_loss_db=24.0,
        scan_revisit_s=1.5,
        dwell_s=0.65,
        cpi_choices=(32, 40, 48, 64),
    ),
]

FAMILY_TRAITS = {
    "public_proxy_pusher_prop_baseline": FamilyTraits(
        class_id="public-proxy-pusher-prop-baseline",
        speed_mps=(45.0, 60.0),
        altitude_m=(2.0, 1_600.0),
        micro_peak_hz=(45.0, 170.0),
        micro_bandwidth_hz=(35.0, 135.0),
        micro_amplitude=(0.32, 0.88),
        coherence=(0.36, 0.82),
    ),
    "fast_prop_owa_public_proxy": FamilyTraits(
        class_id="fast-prop-owa-public-proxy-stress",
        speed_mps=(56.0, 78.0),
        altitude_m=(80.0, 1_800.0),
        micro_peak_hz=(70.0, 210.0),
        micro_bandwidth_hz=(45.0, 160.0),
        micro_amplitude=(0.28, 0.78),
        coherence=(0.30, 0.74),
    ),
    "fast_jet_owa_public_proxy": FamilyTraits(
        class_id="fast-jet-owa-public-proxy-stress",
        speed_mps=(110.0, 160.0),
        altitude_m=(120.0, 2_200.0),
        micro_peak_hz=(0.0, 55.0),
        micro_bandwidth_hz=(30.0, 180.0),
        micro_amplitude=(0.08, 0.38),
        coherence=(0.22, 0.62),
    ),
    "bird_flapping": FamilyTraits(
        class_id="bird-flapping-hard-negative",
        speed_mps=(4.0, 30.0),
        altitude_m=(5.0, 900.0),
        micro_peak_hz=(3.0, 130.0),
        micro_bandwidth_hz=(10.0, 180.0),
        micro_amplitude=(0.18, 0.72),
        coherence=(0.18, 0.58),
    ),
    "rc_fixed_wing": FamilyTraits(
        class_id="rc-fixed-wing-hard-negative",
        speed_mps=(12.0, 58.0),
        altitude_m=(8.0, 850.0),
        micro_peak_hz=(45.0, 190.0),
        micro_bandwidth_hz=(24.0, 150.0),
        micro_amplitude=(0.22, 0.80),
        coherence=(0.22, 0.68),
    ),
    "ground_vehicle": FamilyTraits(
        class_id="ground-vehicle-hard-negative",
        speed_mps=(0.0, 34.0),
        altitude_m=(0.0, 25.0),
        micro_peak_hz=(0.0, 115.0),
        micro_bandwidth_hz=(12.0, 130.0),
        micro_amplitude=(0.08, 0.46),
        coherence=(0.10, 0.44),
    ),
    "wind_turbine": FamilyTraits(
        class_id="wind-turbine-hard-negative",
        speed_mps=(-1.0, 1.0),
        altitude_m=(25.0, 180.0),
        micro_peak_hz=(15.0, 120.0),
        micro_bandwidth_hz=(20.0, 190.0),
        micro_amplitude=(0.24, 0.86),
        coherence=(0.24, 0.70),
        stationary=True,
    ),
    "multipath_ghost": FamilyTraits(
        class_id="multipath-ghost-hard-negative",
        speed_mps=(-8.0, 18.0),
        altitude_m=(0.0, 120.0),
        micro_peak_hz=(0.0, 90.0),
        micro_bandwidth_hz=(20.0, 210.0),
        micro_amplitude=(0.12, 0.58),
        coherence=(0.06, 0.36),
    ),
    "rfi_burst": FamilyTraits(
        class_id="rfi-burst-hard-negative",
        speed_mps=(-2.0, 2.0),
        altitude_m=(0.0, 60.0),
        micro_peak_hz=(10.0, 240.0),
        micro_bandwidth_hz=(50.0, 320.0),
        micro_amplitude=(0.18, 0.96),
        coherence=(0.02, 0.20),
        stationary=True,
    ),
    "clutter_only": FamilyTraits(
        class_id="no-target-counterfactual",
        speed_mps=(-2.0, 2.0),
        altitude_m=(0.0, 35.0),
        micro_peak_hz=(0.0, 80.0),
        micro_bandwidth_hz=(15.0, 220.0),
        micro_amplitude=(0.03, 0.34),
        coherence=(0.02, 0.22),
        stationary=True,
    ),
}

SPEED_PRIORS = {
    "baseline_pusher_prop_public_proxy": SpeedPrior(
        prior_id="baseline_pusher_prop_public_proxy",
        propulsion_class="piston_pusher_prop",
        role="baseline_positive",
        phase_speed_mps={
            "initial_take_up": (8.0, 18.0),
            "climb_transition": (28.0, 42.0),
            "cruise_altitude": (45.0, 60.0),
        },
        cruise_main_estimate_mps=(50.0, 55.0),
        stress_class=False,
        baseline_positive=True,
        policy="Baseline public-proxy positives use a broad 45-60 m/s cruise working band; true speed is restricted truth.",
    ),
    "fast_prop_public_proxy_stress": SpeedPrior(
        prior_id="fast_prop_public_proxy_stress",
        propulsion_class="fast_prop",
        role="stress_negative",
        phase_speed_mps={
            "initial_take_up": (28.0, 48.0),
            "climb_transition": (45.0, 66.0),
            "cruise_altitude": (56.0, 78.0),
        },
        cruise_main_estimate_mps=None,
        stress_class=True,
        baseline_positive=False,
        policy="Fast prop or modified variants are stress classes and are not blended into baseline positives.",
    ),
    "fast_jet_owa_public_proxy": SpeedPrior(
        prior_id="fast_jet_owa_public_proxy",
        propulsion_class="jet",
        role="stress_negative",
        phase_speed_mps={
            "initial_take_up": (80.0, 120.0),
            "climb_transition": (100.0, 145.0),
            "cruise_altitude": (110.0, 160.0),
        },
        cruise_main_estimate_mps=None,
        stress_class=True,
        baseline_positive=False,
        policy="Jet-powered public-proxy variants are separate fast_jet_owa_public_proxy stress data, not normal baseline data.",
    ),
    "airborne_confuser_overlap": SpeedPrior(
        prior_id="airborne_confuser_overlap",
        propulsion_class="airborne_confuser",
        role="hard_negative",
        phase_speed_mps={
            "initial_take_up": (4.0, 58.0),
            "climb_transition": (4.0, 58.0),
            "cruise_altitude": (4.0, 58.0),
        },
        cruise_main_estimate_mps=None,
        stress_class=False,
        baseline_positive=False,
        policy="Bird and small fixed-wing confusers deliberately overlap parts of the observable speed/radial-velocity space.",
    ),
    "stationary_or_ground_artifact": SpeedPrior(
        prior_id="stationary_or_ground_artifact",
        propulsion_class="stationary_or_ground",
        role="hard_negative_or_counterfactual",
        phase_speed_mps={
            "initial_take_up": (-8.0, 34.0),
            "climb_transition": (-8.0, 34.0),
            "cruise_altitude": (-8.0, 34.0),
        },
        cruise_main_estimate_mps=None,
        stress_class=False,
        baseline_positive=False,
        policy="Ground, stationary, RFI, clutter, and multipath artifacts are handled as confusers or counterfactuals.",
    ),
}

FAMILY_SPEED_PRIOR = {
    "public_proxy_pusher_prop_baseline": "baseline_pusher_prop_public_proxy",
    "fast_prop_owa_public_proxy": "fast_prop_public_proxy_stress",
    "fast_jet_owa_public_proxy": "fast_jet_owa_public_proxy",
    "bird_flapping": "airborne_confuser_overlap",
    "rc_fixed_wing": "airborne_confuser_overlap",
    "ground_vehicle": "stationary_or_ground_artifact",
    "wind_turbine": "stationary_or_ground_artifact",
    "multipath_ghost": "stationary_or_ground_artifact",
    "rfi_burst": "stationary_or_ground_artifact",
    "clutter_only": "stationary_or_ground_artifact",
}

RCS_TABLE_DB = {
    "public_proxy_pusher_prop_baseline": {
        "nose": {"X": -18.0, "C": -19.5},
        "tail": {"X": -16.5, "C": -18.0},
        "oblique": {"X": -10.0, "C": -11.5},
        "broadside": {"X": -3.5, "C": -5.0},
        "rolling_scintillation": {"X": -6.0, "C": -7.5},
    },
    "fast_prop_owa_public_proxy": {
        "nose": {"X": -17.0, "C": -18.0},
        "tail": {"X": -15.0, "C": -16.0},
        "oblique": {"X": -8.5, "C": -10.0},
        "broadside": {"X": -2.5, "C": -4.0},
        "rolling_scintillation": {"X": -5.5, "C": -7.0},
    },
    "fast_jet_owa_public_proxy": {
        "nose": {"X": -12.0, "C": -13.5},
        "tail": {"X": -10.0, "C": -11.5},
        "oblique": {"X": -6.0, "C": -8.0},
        "broadside": {"X": 0.5, "C": -1.5},
        "rolling_scintillation": {"X": -4.0, "C": -6.0},
    },
    "bird_flapping": {
        "nose": {"X": -28.0, "C": -30.0},
        "tail": {"X": -29.0, "C": -31.0},
        "oblique": {"X": -23.0, "C": -25.0},
        "broadside": {"X": -18.0, "C": -20.0},
        "rolling_scintillation": {"X": -21.0, "C": -23.0},
    },
    "rc_fixed_wing": {
        "nose": {"X": -20.0, "C": -21.0},
        "tail": {"X": -19.0, "C": -20.0},
        "oblique": {"X": -13.0, "C": -15.0},
        "broadside": {"X": -7.0, "C": -9.0},
        "rolling_scintillation": {"X": -10.0, "C": -12.0},
    },
    "ground_vehicle": {
        "nose": {"X": -6.0, "C": -7.0},
        "tail": {"X": -5.0, "C": -6.0},
        "oblique": {"X": 0.0, "C": -1.0},
        "broadside": {"X": 6.0, "C": 5.0},
        "rolling_scintillation": {"X": 2.5, "C": 1.5},
    },
    "wind_turbine": {
        "nose": {"X": -8.0, "C": -9.0},
        "tail": {"X": -8.0, "C": -9.0},
        "oblique": {"X": 4.0, "C": 2.0},
        "broadside": {"X": 12.0, "C": 10.0},
        "rolling_scintillation": {"X": 8.0, "C": 6.0},
    },
    "multipath_ghost": {
        "nose": {"X": -26.0, "C": -28.0},
        "tail": {"X": -26.0, "C": -28.0},
        "oblique": {"X": -20.0, "C": -22.0},
        "broadside": {"X": -14.0, "C": -16.0},
        "rolling_scintillation": {"X": -17.0, "C": -19.0},
    },
    "rfi_burst": {
        "nose": {"X": -32.0, "C": -32.0},
        "tail": {"X": -32.0, "C": -32.0},
        "oblique": {"X": -32.0, "C": -32.0},
        "broadside": {"X": -32.0, "C": -32.0},
        "rolling_scintillation": {"X": -32.0, "C": -32.0},
    },
    "clutter_only": {
        "nose": {"X": -30.0, "C": -31.0},
        "tail": {"X": -30.0, "C": -31.0},
        "oblique": {"X": -24.0, "C": -26.0},
        "broadside": {"X": -18.0, "C": -20.0},
        "rolling_scintillation": {"X": -22.0, "C": -24.0},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--scenario-groups", "--records", type=int, default=DEFAULT_SCENARIO_GROUPS)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--scale-name", default="smoke")
    parser.add_argument("--max-time-s", type=float, default=DEFAULT_MAX_TIME_S)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def correlated_noise(rng: np.random.Generator, n: int, sigma: float, alpha: float = 0.82) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    innovation = rng.normal(0.0, sigma, n).astype(np.float32)
    for idx in range(1, n):
        out[idx] = alpha * out[idx - 1] + innovation[idx]
    return out


def phase_specs(max_time_s: float) -> list[PhaseSpec]:
    if max_time_s < 120.0:
        raise ValueError("current max-time-s must be at least 120 seconds so cruise_altitude exists")
    return [
        PHASE_SPECS[0],
        PHASE_SPECS[1],
        PhaseSpec(PHASE_SPECS[2].phase_id, 90.0, max_time_s, PHASE_SPECS[2].radar_meaning),
    ]


def split_for_group(group_index: int) -> str:
    mod = group_index % 20
    if mod < 3:
        return "test"
    if mod < 6:
        return "validation"
    return "train"


def group_conditions(seed: int, group_index: int) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(seed, group_index, 4_211))
    site = SITE_ARCHETYPES[int(rng.integers(0, len(SITE_ARCHETYPES)))]
    sensor = SENSOR_ARCHETYPES[int(rng.integers(0, len(SENSOR_ARCHETYPES)))]
    aspect = str(rng.choice(["nose", "tail", "oblique", "broadside", "rolling_scintillation"]))
    confuser = str(rng.choice(CONFUSER_FAMILIES))
    if group_index % 17 == 0:
        confuser = "fast_jet_owa_public_proxy"
    elif group_index % 11 == 0:
        confuser = "fast_prop_owa_public_proxy"
    return {
        "counterfactual_group_id": f"current_cf_group_{group_index:06d}",
        "group_index": group_index,
        "split": split_for_group(group_index),
        "site": site,
        "sensor": sensor,
        "target_aspect": aspect,
        "matched_confuser_family": confuser,
        "clutter_regime": str(rng.choice(["desert_ground", "urban_edge", "sea_glint", "rain_cell", "dust_weather"])),
        "interference": str(rng.choice(["none", "rfi_burst", "dropped_cpi", "agc_compression", "multipath_masking"])),
        "base_range_m": float(rng.uniform(900.0, 12_500.0)),
        "radial_fraction": float(rng.uniform(-0.88, 0.88)),
        "weather_loss_db": float(rng.gamma(1.4, 0.45)),
        "scan_phase_s": float(rng.uniform(0.0, sensor.scan_revisit_s)),
        "multipath_enabled": bool(rng.random() < site.multipath_probability),
        "scenario_seed": stable_seed(seed, group_index, 7_407),
    }


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


def two_ray_loss_db(sensor: SensorArchetype, site: SiteArchetype, range_m: np.ndarray, altitude_m: np.ndarray) -> np.ndarray:
    wavelength_m = 299_792_458.0 / sensor.frequency_hz
    phase = 4.0 * math.pi * site.radar_height_m * np.maximum(1.0, altitude_m) / (
        wavelength_m * np.maximum(range_m, 100.0)
    )
    fading = np.abs(np.sin(phase))
    loss = -20.0 * np.log10(np.clip(0.18 + (1.0 - site.two_ray_weight) * 0.30 + fading, 0.18, 1.0))
    return np.clip(loss, 0.0, 16.0).astype(np.float32)


def build_frame_products(
    rng: np.random.Generator,
    group: dict[str, Any],
    phase: PhaseSpec,
    scene_role: str,
) -> tuple[dict[str, Any], np.ndarray]:
    site: SiteArchetype = group["site"]
    sensor: SensorArchetype = group["sensor"]
    frame_count = int(round((phase.end_s - phase.start_s) / FRAME_PERIOD_S)) + 1
    t_phase = np.arange(frame_count, dtype=np.float32) * FRAME_PERIOD_S
    t_abs = t_phase + phase.start_s
    profile = phase_profile(phase.phase_id)

    if scene_role in {"positive_public_proxy", "target_masked_counterfactual"}:
        family = "public_proxy_pusher_prop_baseline"
        path_seed = stable_seed(int(group["scenario_seed"]), int(phase.start_s), 101)
    elif scene_role == "matched_confuser":
        family = str(group["matched_confuser_family"])
        path_seed = stable_seed(int(group["scenario_seed"]), int(phase.start_s), 202)
    else:
        family = "clutter_only"
        path_seed = stable_seed(int(group["scenario_seed"]), int(phase.start_s), 303)
    product_family = "clutter_only" if scene_role == "target_masked_counterfactual" else family
    if scene_role in {"target_masked_counterfactual", "no_target_counterfactual"}:
        path_seed = stable_seed(int(group["scenario_seed"]), int(phase.start_s), 303)
        rng = np.random.default_rng(stable_seed(int(group["scenario_seed"]), int(phase.start_s), 404))
    cpi_pulses = int(rng.choice(sensor.cpi_choices))
    traits = FAMILY_TRAITS[product_family]
    path_rng = np.random.default_rng(path_seed)
    range_m, radial_velocity, altitude_m, true_speed_mps = kinematics(
        path_rng,
        product_family,
        traits,
        t_abs,
        float(group["base_range_m"]) + (0.0 if scene_role != "matched_confuser" else rng.normal(0.0, 100.0)),
        float(group["radial_fraction"]) + (0.0 if scene_role != "matched_confuser" else rng.normal(0.0, 0.10)),
    )
    truth_altitude_m = altitude_m
    truth_speed_mps = true_speed_mps
    truth_radial_velocity = radial_velocity
    if scene_role == "target_masked_counterfactual":
        target_rng = np.random.default_rng(stable_seed(int(group["scenario_seed"]), int(phase.start_s), 101))
        _, truth_radial_velocity, truth_altitude_m, truth_speed_mps = kinematics(
            target_rng,
            family,
            FAMILY_TRAITS[family],
            t_abs,
            float(group["base_range_m"]),
            float(group["radial_fraction"]),
        )

    aspect = str(group["target_aspect"])
    rcs_center = rcs_lookup_dbsm(product_family, aspect, sensor)
    scintillation_sigma = 3.6 if aspect == "rolling_scintillation" else 1.7
    rcs_dbsm = rcs_center + correlated_noise(rng, frame_count, sigma=scintillation_sigma, alpha=0.72)
    elevation_deg = np.degrees(np.arctan2(np.maximum(0.0, altitude_m - site.radar_height_m), np.maximum(range_m, 100.0)))
    horizon_mask = elevation_deg < site.terrain_horizon_deg
    low_grazing_loss = np.clip((3.0 - elevation_deg) * 1.7, 0.0, 13.0)
    occlusion_loss = horizon_mask.astype(np.float32) * rng.uniform(10.0, 24.0) * float(profile["occlusion"])
    two_ray = (
        two_ray_loss_db(sensor, site, range_m, altitude_m) * float(profile["multipath"])
        if bool(group["multipath_enabled"])
        else np.zeros(frame_count, dtype=np.float32)
    )
    weather_loss = site.weather_loss_db + float(group["weather_loss_db"])
    clutter_base = {
        "desert_ground": 2.1,
        "urban_edge": 3.4,
        "sea_glint": 3.8,
        "rain_cell": 3.2,
        "dust_weather": 2.8,
    }[str(group["clutter_regime"])]
    rfi_base = {
        "none": 0.04,
        "rfi_burst": 0.38,
        "dropped_cpi": 0.10,
        "agc_compression": 0.12,
        "multipath_masking": 0.14,
    }[str(group["interference"])]
    clutter_loss = site.land_clutter_loss_db + clutter_base * float(profile["clutter"]) + rfi_base * 3.8
    propagation_loss = low_grazing_loss + occlusion_loss + two_ray + weather_loss
    link_budget_snr = radar_equation_snr_db(sensor, range_m, rcs_dbsm, cpi_pulses, propagation_loss, clutter_loss)

    scan_phase = np.mod(t_abs + float(group["scan_phase_s"]), sensor.scan_revisit_s)
    scan_gap = scan_phase > sensor.dwell_s
    dropout = np.clip(
        rng.beta(1.35, 12.0, frame_count).astype(np.float32) * 0.40
        + scan_gap.astype(np.float32) * site.scan_gap_probability
        + horizon_mask.astype(np.float32) * 0.34
        + (0.12 if str(group["interference"]) == "dropped_cpi" else 0.0),
        0.0,
        0.98,
    )
    clutter_glints = rng.weibull(1.0 if str(group["clutter_regime"]) != "sea_glint" else 0.62, frame_count).astype(np.float32)
    clutter_glints *= (1.7 + clutter_base * 0.30)
    signal_enabled = scene_role in {"positive_public_proxy", "matched_confuser"}
    suppressed_snr = link_budget_snr - horizon_mask.astype(np.float32) * 24.0 - dropout * rng.uniform(4.0, 9.0)
    snr_db = suppressed_snr + correlated_noise(rng, frame_count, sigma=1.8, alpha=0.75) + clutter_glints * 0.25
    if not signal_enabled:
        snr_db = (
            -10.5
            + clutter_glints * 1.1
            + correlated_noise(rng, frame_count, sigma=2.0, alpha=0.80)
            - dropout * rng.uniform(1.2, 4.2)
        ).astype(np.float32)
    if str(group["interference"]) == "agc_compression":
        snr_db = np.tanh(snr_db / 14.0) * 14.0
    if str(group["interference"]) == "rfi_burst" or product_family == "rfi_burst":
        rfi_base += 0.22
    rfi_pressure = np.clip(rfi_base + rng.beta(1.2, 6.5, frame_count) * 0.55 + dropout * 0.16, 0.0, 1.0)
    local_noise_floor_db = (
        -43.0
        + site.land_clutter_loss_db
        + clutter_base * 2.4
        + rfi_pressure * 7.4
        + correlated_noise(rng, frame_count, sigma=0.60, alpha=0.86)
    )
    cfar_threshold = (
        8.9
        + clutter_base * 0.55
        + rfi_pressure * 2.4
        + dropout * 1.6
        + horizon_mask.astype(np.float32) * 1.4
        + float(profile["threshold"])
        + correlated_noise(rng, frame_count, sigma=0.34, alpha=0.84)
    )
    doppler_scr = np.clip(
        snr_db
        - clutter_base * 0.6
        + np.abs(radial_velocity) * 0.052
        - rfi_pressure * 1.8
        + rng.normal(0.0, 1.15, frame_count),
        -18.0,
        26.0,
    )
    cfar_statistic = snr_db + 0.30 * doppler_scr + clutter_glints * 0.28 + rng.normal(0.0, 1.0, frame_count)
    cfar_detected = (cfar_statistic - cfar_threshold + rng.normal(0.0, 0.65, frame_count)) > 0.0

    coherence = uniform(rng, traits.coherence)
    tbd_track_score = np.zeros(frame_count, dtype=np.float32)
    state = rng.uniform(0.0, 0.05)
    margin = cfar_statistic - cfar_threshold
    for idx, value in enumerate(margin):
        evidence = max(0.0, float(value)) * (0.022 + coherence * 0.020)
        evidence += max(0.0, 1.0 - float(dropout[idx])) * coherence * 0.008
        evidence += max(0.0, float(doppler_scr[idx])) * 0.0016
        state = state * (0.84 + 0.08 * coherence) + evidence - float(dropout[idx]) * 0.024 - float(rfi_pressure[idx]) * 0.010
        tbd_track_score[idx] = max(0.0, state)

    micro_peak = uniform(rng, traits.micro_peak_hz)
    micro_bw = uniform(rng, traits.micro_bandwidth_hz)
    micro_amp = uniform(rng, traits.micro_amplitude)
    if not signal_enabled:
        micro_amp *= rng.uniform(0.20, 0.48)
    micro_variation = 1.0 + 0.23 * np.sin(t_abs / rng.uniform(3.5, 9.0) + rng.uniform(0.0, 6.28))
    signal_mask = 1.0 - np.clip(dropout * 0.70 + horizon_mask.astype(np.float32) * 0.25, 0.0, 0.98)
    micro_doppler_energy = np.clip(
        micro_amp * micro_variation * (0.34 + 0.66 * signal_mask)
        + rfi_pressure * 0.14
        + clutter_glints * 0.009
        + rng.normal(0.0, 0.065, frame_count),
        0.0,
        1.6,
    )
    micro_peak_series = np.clip(micro_peak + correlated_noise(rng, frame_count, sigma=max(1.0, micro_peak * 0.040)), 0.0, 280.0)
    micro_bw_series = np.clip(micro_bw + correlated_noise(rng, frame_count, sigma=max(2.0, micro_bw * 0.050)), 1.0, 400.0)
    normalized_snr = np.clip(sigmoid((snr_db - 0.5) / 5.8) + rng.normal(0.0, 0.035, frame_count), 0.0, 1.0)
    range_time_energy = np.clip(0.18 + normalized_snr * 0.55 + clutter_glints * 0.045 + rfi_pressure * 0.06, 0.0, 1.6)
    doppler_time_energy = np.clip(0.16 + sigmoid(doppler_scr / 6.5) * 0.52 + micro_doppler_energy * 0.16, 0.0, 1.7)
    range_doppler_time_energy = np.clip(
        0.12 + range_time_energy * 0.46 + doppler_time_energy * 0.38 + cfar_detected.astype(np.float32) * 0.08,
        0.0,
        1.8,
    )
    stft_energy = np.clip(micro_doppler_energy * 0.58 + doppler_time_energy * 0.24 + rng.normal(0.0, 0.035, frame_count), 0.0, 1.6)
    weighted_spectrum_peak = np.clip(stft_energy * 0.62 + normalized_snr * 0.24 + rng.normal(0.0, 0.025, frame_count), 0.0, 1.6)
    cepstrum_peak = np.clip(micro_doppler_energy * 0.42 + rng.normal(0.0, 0.035, frame_count), 0.0, 1.4)
    cadence_velocity_peak = np.clip(np.abs(radial_velocity) * 0.0009 + micro_peak_series / 3_000.0 + rng.normal(0.0, 0.004, frame_count), 0.0, 0.24)
    phase_impairment_rad = {
        "none": 0.01,
        "rfi_burst": 0.04,
        "dropped_cpi": 0.02,
        "agc_compression": 0.02,
        "multipath_masking": 0.06,
    }[str(group["interference"])] + rng.normal(0.0, 0.012, frame_count)
    amplitude_impairment = np.clip(1.0 + rfi_pressure * rng.normal(0.0, 0.12, frame_count), 0.70, 1.35)

    frame = np.zeros((frame_count, len(FRAME_COLUMNS)), dtype=np.float32)
    frame[:, FRAME_INDEX["time_s"]] = t_phase
    frame[:, FRAME_INDEX["cpi_pulses"]] = float(cpi_pulses)
    frame[:, FRAME_INDEX["range_m"]] = range_m
    frame[:, FRAME_INDEX["radial_velocity_mps"]] = radial_velocity + correlated_noise(rng, frame_count, sigma=0.40, alpha=0.70)
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

    truth = {
        "scene_role": scene_role,
        "target_family": family,
        "class_id": FAMILY_TRAITS[family].class_id,
        "target_presence": scene_role != "no_target_counterfactual",
        "target_signal_enabled": bool(signal_enabled),
        "detector_product_family": product_family,
        "phase_id": phase.phase_id,
        "phase_start_s": phase.start_s,
        "phase_end_s": phase.end_s,
        "mean_altitude_m": float(np.mean(truth_altitude_m)),
        "mean_true_speed_mps": float(np.mean(truth_speed_mps)),
        "mean_abs_radial_velocity_mps": float(np.mean(np.abs(truth_radial_velocity))),
        "mean_link_budget_snr_db": float(np.mean(link_budget_snr)),
        "mean_propagation_loss_db": float(np.mean(propagation_loss)),
        "mean_clutter_loss_db": float(np.mean(clutter_loss)),
        "raw_rcs_dbsm_mean": float(np.mean(rcs_dbsm)),
        "raw_rcs_dbsm_min": float(np.min(rcs_dbsm)),
        "raw_rcs_dbsm_max": float(np.max(rcs_dbsm)),
        "horizon_masked_fraction": float(np.mean(horizon_mask)),
        "los_eligible_fraction": float(1.0 - np.mean(horizon_mask)),
        "below_horizon_frame_count": int(horizon_mask.sum()),
        "radar_equation_inputs": {
            "peak_power_dbw": sensor.peak_power_dbw,
            "tx_gain_dbi": sensor.tx_gain_dbi,
            "rx_gain_dbi": sensor.rx_gain_dbi,
            "frequency_hz": sensor.frequency_hz,
            "wavelength_m": 299_792_458.0 / sensor.frequency_hz,
            "bandwidth_hz": sensor.bandwidth_hz,
            "noise_figure_db": sensor.noise_figure_db,
            "system_loss_db": sensor.system_loss_db,
            "cpi_pulses": cpi_pulses,
        },
    }
    return truth, frame


def pad_frames(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(frame.shape[0] for frame in frames)
    out = np.zeros((len(frames), max_len, len(FRAME_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(frames), max_len), dtype=bool)
    for idx, frame in enumerate(frames):
        out[idx, : frame.shape[0], :] = frame
        out[idx, frame.shape[0] :, FRAME_INDEX["time_s"]] = np.arange(frame.shape[0], max_len, dtype=np.float32) * FRAME_PERIOD_S
        mask[idx, : frame.shape[0]] = True
    return out, mask


def aggregate_features(frames: np.ndarray, valid_mask: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    feature_values = []
    feature_names = []
    for col_name in FRAME_COLUMNS:
        if col_name == "time_s":
            continue
        idx = FRAME_INDEX[col_name]
        values = np.where(valid_mask, frames[:, :, idx], np.nan)
        stats = {
            f"{col_name}_mean": np.nanmean(values, axis=1),
            f"{col_name}_std": np.nanstd(values, axis=1),
            f"{col_name}_min": np.nanmin(values, axis=1),
            f"{col_name}_max": np.nanmax(values, axis=1),
            f"{col_name}_last": np.asarray(
                [values[row, np.where(valid_mask[row])[0][-1]] for row in range(values.shape[0])],
                dtype=np.float32,
            ),
            f"{col_name}_q25": np.nanquantile(values, 0.25, axis=1),
            f"{col_name}_q75": np.nanquantile(values, 0.75, axis=1),
        }
        for name, value in stats.items():
            feature_names.append(name)
            feature_values.append(np.nan_to_num(value.astype(np.float32), nan=0.0))
    return pd.DataFrame(np.stack(feature_values, axis=1), columns=feature_names), feature_names


def denylist_violations(frame_columns: list[str], feature_names: list[str]) -> list[str]:
    published = set(frame_columns) | set(feature_names)
    violations = sorted(name for name in RESTRICTED_FEATURE_NAMES if name in published)

    def stems(restricted: str) -> set[str]:
        out = {restricted}
        for suffix in ["_mps", "_dbsm", "_db", "_m", "_hz"]:
            if restricted.endswith(suffix):
                out.add(restricted[: -len(suffix)])
        return out

    def leaks(feature_name: str, restricted: str) -> bool:
        lowered = feature_name.lower()
        for stem in stems(restricted.lower()):
            if lowered == stem:
                return True
            if lowered.startswith(f"{stem}_") or lowered.endswith(f"_{stem}"):
                return True
            if f"_{stem}_" in lowered:
                return True
        return False

    derived_violations = sorted(
        feature_name
        for feature_name in feature_names
        if any(leaks(feature_name, restricted) for restricted in RESTRICTED_FEATURE_NAMES)
    )
    return sorted(set(violations + derived_violations))


def first_true_time(mask: np.ndarray, time_values: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(time_values[int(np.argmax(mask))])


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def operational_metrics(records: pd.DataFrame, frames: np.ndarray, valid_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    split_values = records["split"].astype(str).to_numpy()
    phase_values = records["phase_id"].astype(str).to_numpy()
    for phase_id, phase_records in records.groupby("phase_id"):
        indices = phase_records.index.to_numpy(dtype=np.int64)
        labels = phase_records["is_public_proxy_positive"].astype(bool).to_numpy()
        valid = valid_mask[indices]
        cfar = frames[indices, :, FRAME_INDEX["cfar_detected"]] > 0.5
        tbd = frames[indices, :, FRAME_INDEX["tbd_track_score"]]
        times = frames[indices, :, FRAME_INDEX["time_s"]]
        train_phase = (split_values == "train") & (phase_values == str(phase_id))
        train_tbd = frames[train_phase, :, FRAME_INDEX["tbd_track_score"]]
        train_valid = valid_mask[train_phase]
        threshold_values = train_tbd[train_valid] if train_tbd.size else np.asarray([], dtype=np.float32)
        threshold = float(np.quantile(threshold_values, 0.64)) if threshold_values.size else 0.0
        confirmed = (tbd > threshold) & cfar & valid
        positive_any = (cfar & valid)[labels].any(axis=1) if labels.any() else np.asarray([], dtype=bool)
        negative_any = (cfar & valid)[~labels].any(axis=1) if (~labels).any() else np.asarray([], dtype=bool)
        first_hits = []
        track_inits = []
        fragments = []
        missed_tracks = 0
        for row_cfar, row_confirmed, row_valid, row_times in zip(cfar[labels], confirmed[labels], valid[labels], times[labels]):
            hit_mask = row_cfar & row_valid
            first = first_true_time(hit_mask, row_times)
            if first is not None:
                first_hits.append(first)
            track_first = first_true_time(row_confirmed & row_valid, row_times)
            if track_first is None:
                missed_tracks += 1
            else:
                track_inits.append(track_first)
            transitions = np.diff(np.r_[False, row_confirmed & row_valid, False].astype(np.int8))
            fragments.append(float((transitions == 1).sum()))
        rows.append(
            {
                "phase_id": phase_id,
                "record_count": int(len(indices)),
                "positive_count": int(labels.sum()),
                "negative_count": int((~labels).sum()),
                "pd_any_cfar": float(positive_any.mean()) if positive_any.size else float("nan"),
                "pfa_any_cfar": float(negative_any.mean()) if negative_any.size else float("nan"),
                "first_hit_latency_s": float(np.mean(first_hits)) if first_hits else float("nan"),
                "track_initiation_latency_s": float(np.mean(track_inits)) if track_inits else float("nan"),
                "track_fragmentation_rate": float(np.mean(fragments)) if fragments else float("nan"),
                "missed_track_rate": float(missed_tracks / max(1, int(labels.sum()))),
                "false_track_rate": float((confirmed[~labels] & valid[~labels]).any(axis=1).mean()) if (~labels).any() else float("nan"),
                "horizon_masked_fraction": float(phase_records["horizon_masked_fraction"].mean()),
                "los_eligible_fraction": float(phase_records["los_eligible_fraction"].mean()),
                "micro_doppler_confidence": float(
                    np.nanmean(np.where(valid, frames[indices, :, FRAME_INDEX["micro_doppler_energy"]], np.nan))
                ),
                "initial_low_pd_allowed": bool(phase_id == "initial_take_up"),
            }
        )
    return pd.DataFrame(rows).sort_values("phase_id")


def negative_control_audit(records: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, Any]:
    labels = records["is_public_proxy_positive"].astype(bool).to_numpy(dtype=np.int64)
    split = records["split"].astype(str).to_numpy()
    rng = np.random.default_rng(20260518)
    shuffled = labels.copy()
    for split_name in ["train", "validation", "test"]:
        mask = split == split_name
        shuffled[mask] = rng.permutation(shuffled[mask])
    seed_hash = np.stack(
        [
            records["scenario_seed"].astype(np.uint64).to_numpy() % 997,
            records["object_seed"].astype(np.uint64).to_numpy() % 991,
        ],
        axis=1,
    ).astype(np.float32)
    row_index = records[["record_index"]].astype(np.float32).to_numpy()
    audit_metadata_probe = one_hot_frame(
        records,
        categorical=[
            "phase_id",
            "site_archetype_id",
            "sensor_archetype_id",
            "clutter_regime",
            "target_aspect",
            "interference",
            "range_bin",
        ],
        numeric=["phase_start_s", "phase_end_s", "available_history_s"],
    )
    restricted_family = one_hot_frame(records, categorical=["target_family", "scene_role", "class_id"], numeric=[])
    sensor_features = feature_df.drop(columns=["record_id"]).to_numpy(np.float32)
    target_masked = records["scene_role"].astype(str).to_numpy() == "target_masked_counterfactual"
    no_target = records["scene_role"].astype(str).to_numpy() == "no_target_counterfactual"
    masked_subset = target_masked | no_target
    masked_labels = target_masked[masked_subset].astype(np.int64)
    masked_split = split[masked_subset]
    target_masked_auc = probe_auc(sensor_features[masked_subset], masked_labels, masked_split)
    probes = {
        "seed_only_auc": probe_auc(seed_hash, labels, split),
        "row_index_only_auc": probe_auc(row_index, labels, split),
        "audit_metadata_probe_auc": probe_auc(audit_metadata_probe, labels, split),
        "shuffled_label_sensor_feature_auc": probe_auc(sensor_features, shuffled, split),
        "target_masked_counterfactual_auc": target_masked_auc,
        "generator_family_probe_auc_restricted": probe_auc(restricted_family, labels, split),
        "gate": NEGATIVE_CONTROL_AUC_GATE,
        "target_masked_gate": TARGET_MASKED_AUC_GATE,
    }
    pass_keys = [
        "seed_only_auc",
        "row_index_only_auc",
        "audit_metadata_probe_auc",
        "shuffled_label_sensor_feature_auc",
        "target_masked_counterfactual_auc",
    ]
    probes["status"] = "pass" if all(
        math.isfinite(float(probes[key])) and float(probes[key]) <= NEGATIVE_CONTROL_AUC_GATE
        for key in pass_keys
    ) else "fail"
    probes["policy"] = (
        "Generator-family, class, scene-role, seed, phase, and truth metadata are audit-only. "
        "The restricted generator-family probe is reported to prove why those columns stay out of model features."
    )
    return probes


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def range_bin(mean_range_m: float) -> str:
    if mean_range_m < 2_500.0:
        return "near_range"
    if mean_range_m < 6_000.0:
        return "mid_range"
    if mean_range_m < 11_000.0:
        return "far_range"
    return "edge_range"


def speed_prior_for_family(family: str) -> SpeedPrior:
    return SPEED_PRIORS[FAMILY_SPEED_PRIOR.get(family, "stationary_or_ground_artifact")]


def speed_prior_manifest_payload() -> dict[str, Any]:
    return {
        "benchmark_profile": "ml-training-three-tier",
        "claim_boundary": "Public-source speed priors and stress-class boundaries only; no measured target truth or proprietary-equivalent behavior is claimed.",
        "policy": [
            "Baseline piston pusher-prop public-proxy positives use the 45-60 m/s cruise working band, with 50-55 m/s as the main estimate.",
            "Fast prop or modified variants remain separate stress classes.",
            "Jet-powered variants remain fast_jet_owa_public_proxy stress data around 110+ m/s and are not blended into baseline positives.",
            "radial_velocity_mps is radar-observable line-of-sight velocity, not true speed.",
            "estimated_ground_speed_mps remains denylisted until a track-history or multi-view estimator is implemented.",
        ],
        "family_to_prior": FAMILY_SPEED_PRIOR,
        "priors": {prior_id: asdict(prior) for prior_id, prior in SPEED_PRIORS.items()},
        "source_context": [
            "OSMP visual guide: delta/pusher fixed-wing public proxy family context.",
            "Army Recognition public technical profile: broad piston pusher-prop speed prior context.",
            "Reuters-syndicated reporting: faster prop and jet variants treated as stress classes, not baseline positives.",
        ],
    }


def build_kinematics_audit(records: pd.DataFrame, kinematic_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = pd.DataFrame(kinematic_rows).sort_values(["record_index", "record_id"]).reset_index(drop=True)
    positive_families = sorted(records.loc[records["is_public_proxy_positive"], "target_family"].astype(str).unique())
    stress_families = {"fast_prop_owa_public_proxy", "fast_jet_owa_public_proxy"}
    baseline_cruise = audit[
        (audit["scene_role"] == "positive_public_proxy")
        & (audit["target_family"] == "public_proxy_pusher_prop_baseline")
        & (audit["phase_id"] == "cruise_altitude")
    ]
    if baseline_cruise.empty:
        baseline_cruise_status = "fail"
    else:
        baseline_cruise_status = "pass" if (
            (baseline_cruise["mean_true_speed_mps"] >= 45.0)
            & (baseline_cruise["mean_true_speed_mps"] <= 60.0)
        ).all() else "fail"
    stress_separation_status = "pass" if not any(family in stress_families for family in positive_families) else "fail"
    crossing_examples = audit[
        (audit["target_family"] == "public_proxy_pusher_prop_baseline")
        & (audit["phase_id"] == "cruise_altitude")
        & (audit["mean_true_speed_mps"] >= 45.0)
        & (audit["radial_to_true_speed_ratio"] <= 0.25)
    ]
    summary = {
        "baseline_positive_families": positive_families,
        "baseline_cruise_speed_band_status": baseline_cruise_status,
        "stress_class_separation_status": stress_separation_status,
        "crossing_target_low_radial_count": int(len(crossing_examples)),
        "estimated_ground_speed_exposed": False,
        "status": "pass" if baseline_cruise_status == "pass" and stress_separation_status == "pass" else "fail",
    }
    return audit, summary


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def quantile_samples(values: np.ndarray, sample_count: int = CALIBRATION_SAMPLE_COUNT) -> list[float]:
    arr = np.sort(finite_values(values))
    if arr.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    samples = np.quantile(arr, quantiles)
    return [float(value) for value in samples if math.isfinite(float(value))]


def target_distribution(bounds: tuple[float, float], sample_count: int = CALIBRATION_SAMPLE_COUNT) -> list[float]:
    low, high = bounds
    return [float(value) for value in np.linspace(float(low), float(high), sample_count, dtype=np.float64)]


def target_beta_distribution(
    bounds: tuple[float, float],
    alpha: float,
    beta: float,
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
) -> list[float]:
    # Deterministic inverse-free approximation: sort fixed beta draws so the
    # report carries a bounded target distribution without vendored traces.
    rng = np.random.default_rng(20260518)
    low, high = bounds
    draws = np.sort(rng.beta(alpha, beta, sample_count))
    scaled = float(low) + draws * (float(high) - float(low))
    return [float(value) for value in scaled]


def wasserstein_1d_sorted(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, max(a.size, b.size), dtype=np.float64)
    qa = np.quantile(np.sort(a), quantiles)
    qb = np.quantile(np.sort(b), quantiles)
    return float(np.mean(np.abs(qa - qb)))


def ks_distance_1d_sorted(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    values = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, values, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, values, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def calibration_distance(metric: str, target_samples: list[float], observed_samples: list[float]) -> float:
    target = finite_values(np.asarray(target_samples, dtype=np.float64))
    observed = finite_values(np.asarray(observed_samples, dtype=np.float64))
    if metric == "wasserstein_1":
        return wasserstein_1d_sorted(target, observed)
    if metric == "kolmogorov_smirnov":
        return ks_distance_1d_sorted(target, observed)
    raise ValueError(f"unsupported calibration metric: {metric}")


def frame_observation_samples(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    selector: pd.Series,
    column_name: str,
    transform: str | None = None,
) -> list[float]:
    indices = records.index[selector.to_numpy(dtype=bool)].to_numpy(dtype=np.int64)
    if indices.size == 0:
        return []
    values = frames[indices, :, FRAME_INDEX[column_name]]
    mask = valid_mask[indices]
    observed = finite_values(values[mask])
    if transform == "abs":
        observed = np.abs(observed)
    return quantile_samples(observed)


def record_observation_samples(records: pd.DataFrame, selector: pd.Series, column_name: str) -> list[float]:
    values = records.loc[selector, column_name].astype(float).to_numpy(dtype=np.float64)
    return quantile_samples(values)


def calibration_anchor_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_name": "snr_db_detector_public_proxy",
            "feature_family": "range_doppler_observable",
            "unit": "dB",
            "metric": "wasserstein_1",
            "tolerance": 22.0,
            "target_samples": target_distribution((-18.0, 24.0)),
            "citation_id": "public_proxy_link_budget_reference_envelope",
            "citation": "Strict-open current public-proxy link-budget envelope; target samples are broad reference priors, not measured radar traces.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "abs_radial_velocity_mps_detector_public_proxy",
            "feature_family": "doppler_observable",
            "unit": "m/s",
            "metric": "wasserstein_1",
            "tolerance": 28.0,
            "target_samples": target_distribution((0.0, 60.0)),
            "citation_id": "public_proxy_radial_velocity_reference_envelope",
            "citation": "Strict-open current radial-velocity reference envelope derived from broad public speed priors and random line-of-sight geometry; not true-speed truth.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "doppler_scr_detector_public_proxy",
            "feature_family": "doppler_clutter_observable",
            "unit": "dB",
            "metric": "wasserstein_1",
            "tolerance": 20.0,
            "target_samples": target_distribution((-14.0, 24.0)),
            "citation_id": "public_proxy_doppler_scr_reference_envelope",
            "citation": "Strict-open current Doppler signal-to-clutter reference envelope for simulator consistency checks; no measured target samples are vendored.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "rfi_pressure_all_scenes",
            "feature_family": "interference_observable",
            "unit": "unitless",
            "metric": "kolmogorov_smirnov",
            "tolerance": 0.75,
            "target_samples": target_beta_distribution((0.0, 1.0), alpha=1.3, beta=5.5),
            "citation_id": "public_proxy_interference_pressure_reference",
            "citation": "Strict-open current interference-pressure reference distribution for smoke artifact integrity; not deployment metadata or measured spectrum capture.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
        {
            "feature_name": "horizon_masked_fraction_initial_take_up",
            "feature_family": "operational_los_metric",
            "unit": "fraction",
            "metric": "kolmogorov_smirnov",
            "tolerance": 0.80,
            "target_samples": target_distribution((0.35, 1.0)),
            "citation_id": "public_proxy_low_grazing_los_reference",
            "citation": "Strict-open current low-grazing line-of-sight reference envelope for initial-take-up reporting; not measured radar performance.",
            "url": None,
            "source_boundary": "synthetic_public_proxy_reference_distribution",
        },
    ]


def build_calibration_artifacts(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    positive = records["scene_role"].astype(str) == "positive_public_proxy"
    initial = records["phase_id"].astype(str) == "initial_take_up"
    all_records = pd.Series(True, index=records.index)
    observation_by_feature = {
        "snr_db_detector_public_proxy": frame_observation_samples(records, frames, valid_mask, positive, "snr_db"),
        "abs_radial_velocity_mps_detector_public_proxy": frame_observation_samples(
            records,
            frames,
            valid_mask,
            positive,
            "radial_velocity_mps",
            transform="abs",
        ),
        "doppler_scr_detector_public_proxy": frame_observation_samples(records, frames, valid_mask, positive, "doppler_scr"),
        "rfi_pressure_all_scenes": frame_observation_samples(records, frames, valid_mask, all_records, "rfi_pressure"),
        "horizon_masked_fraction_initial_take_up": record_observation_samples(
            records,
            initial,
            "horizon_masked_fraction",
        ),
    }

    anchors = []
    observations = []
    distance_rows = []
    manifest_anchors = []
    for spec in calibration_anchor_specs():
        feature_name = str(spec["feature_name"])
        target_samples = [float(value) for value in spec["target_samples"]]
        observed_samples = [float(value) for value in observation_by_feature.get(feature_name, [])]
        distance = calibration_distance(str(spec["metric"]), target_samples, observed_samples)
        sample_counts_ok = len(target_samples) >= 32 and len(observed_samples) >= 32
        distance_ok = math.isfinite(distance) and distance <= float(spec["tolerance"])
        status = "pass" if sample_counts_ok and distance_ok else "fail"
        anchors.append(
            {
                "feature_name": feature_name,
                "citation": spec["citation"],
                "url": spec["url"],
                "target_samples": target_samples,
                "metric": spec["metric"],
                "tolerance": float(spec["tolerance"]),
            }
        )
        observations.append(
            {
                "feature_name": feature_name,
                "observed_samples": observed_samples,
            }
        )
        distance_rows.append(
            {
                "feature_name": feature_name,
                "feature_family": spec["feature_family"],
                "unit": spec["unit"],
                "citation_id": spec["citation_id"],
                "citation": spec["citation"],
                "url": spec["url"] or "",
                "metric": spec["metric"],
                "tolerance": float(spec["tolerance"]),
                "distance": distance,
                "target_sample_count": len(target_samples),
                "observed_sample_count": len(observed_samples),
                "status": status,
                "claim_boundary": "reference-only synthetic public-proxy distribution audit; not measured-anchor promotion",
            }
        )
        manifest_anchors.append(
            {
                "feature_name": feature_name,
                "feature_family": spec["feature_family"],
                "unit": spec["unit"],
                "citation_id": spec["citation_id"],
                "metric": spec["metric"],
                "tolerance": float(spec["tolerance"]),
                "target_sample_count": len(target_samples),
                "observed_sample_count": len(observed_samples),
                "target_min": float(min(target_samples)),
                "target_max": float(max(target_samples)),
                "observed_min": float(min(observed_samples)) if observed_samples else None,
                "observed_max": float(max(observed_samples)) if observed_samples else None,
                "source_boundary": spec["source_boundary"],
                "status": status,
            }
        )

    artifact_status = "pass" if len(anchors) >= 3 and all(row["status"] == "pass" for row in distance_rows) else "fail"
    summary = {
        "status": artifact_status,
        "calibration_anchor_status": "reference_only" if artifact_status == "pass" else "reference_only_artifact_fail",
        "validation_tier": "unvalidated/basic synthetic public-proxy benchmark",
        "current_promotion_status": "not_promoted_public_proxy_reference_only",
        "anchor_count": len(anchors),
        "min_samples_per_distribution": 32,
        "distance_failures": [row["feature_name"] for row in distance_rows if row["status"] != "pass"],
        "claim_boundary": "Calibration artifacts use strict-open reference target distributions and synthetic observations only; they do not promote the smoke benchmark to measured-anchored current.",
    }
    report = {
        "anchors": anchors,
        "observations": observations,
    }
    manifest = {
        "artifact": "calibration_anchor_manifest",
        "benchmark_profile": "ml-training-three-tier",
        "validation_tier": summary["validation_tier"],
        "calibration_anchor_status": summary["calibration_anchor_status"],
        "artifact_integrity_status": artifact_status,
        "current_promotion_status": summary["current_promotion_status"],
        "claim_boundary": summary["claim_boundary"],
        "rust_schema_compatibility": {
            "calibration_report": "calibration_report.json",
            "shape": {"anchors": "Vec<CalibrationDistributionAnchor>", "observations": "Vec<CalibrationDistributionObservation>"},
            "metrics": ["wasserstein_1", "kolmogorov_smirnov"],
            "min_anchors": 3,
            "min_samples_per_distribution": 32,
        },
        "source_policy": [
            "Target samples are broad public-proxy reference distributions, not measured traces.",
            "Observed samples are synthetic detector-facing or operational smoke outputs.",
            "No raw captures, proprietary sensor data, deployment metadata, or exact platform signatures are included.",
        ],
        "anchors": manifest_anchors,
    }
    return report, manifest, pd.DataFrame(distance_rows), summary


def circular_mean_deg(values: list[float]) -> float:
    if not values:
        return float("nan")
    radians = np.radians(np.asarray(values, dtype=np.float64))
    angle = math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
    return float((angle + 360.0) % 360.0)


def acoustic_false_source(rng: np.random.Generator, record: pd.Series) -> str:
    if str(record.get("interference", "")) == "rfi_burst":
        return "generator_hum"
    if str(record.get("clutter_regime", "")) in {"rain_cell", "dust_weather", "sea_glint"}:
        return str(rng.choice(["wind_gust", "thunder_rumble"]))
    if str(record.get("site_archetype_id", "")).endswith("urban_edge"):
        return str(rng.choice(["road_traffic", "industrial_machinery", "generator_hum"]))
    return str(rng.choice(ACOUSTIC_FALSE_CUE_SOURCES))


def acoustic_detection_probability(record: pd.Series, mean_range_m: float, horizon_masked_fraction: float) -> float:
    phase_id = str(record["phase_id"])
    scene_role = str(record["scene_role"])
    is_positive = bool(record["is_public_proxy_positive"])
    if is_positive or scene_role == "target_masked_counterfactual":
        base = {
            "initial_take_up": 0.70,
            "climb_transition": 0.78,
            "cruise_altitude": 0.62,
        }.get(phase_id, 0.62)
        if phase_id == "initial_take_up" and horizon_masked_fraction > 0.55:
            base += 0.10
    elif scene_role == "matched_confuser":
        base = 0.26
    else:
        base = 0.16
    range_penalty = max(0.0, (mean_range_m - 5_000.0) / 11_000.0) * 0.28
    urban_bonus = 0.05 if str(record.get("site_archetype_id", "")).endswith("urban_edge") else 0.0
    return float(np.clip(base - range_penalty + urban_bonus, 0.03, 0.92))


def acoustic_spectral_features(
    rng: np.random.Generator,
    signal_like: bool,
    false_source: str,
    micro_peak_hz: float,
    micro_bandwidth_hz: float,
    micro_energy: float,
) -> dict[str, float]:
    if signal_like:
        engine_peak = float(np.clip(85.0 + micro_peak_hz * 0.42 + rng.normal(0.0, 18.0), 45.0, 260.0))
        prop_harmonic = float(np.clip(engine_peak * rng.uniform(1.8, 2.4), 90.0, 620.0))
        entropy = float(np.clip(0.32 + rng.normal(0.0, 0.08) + micro_bandwidth_hz / 900.0, 0.12, 0.92))
        modulation = float(np.clip(0.38 + micro_energy * 0.40 + rng.normal(0.0, 0.08), 0.05, 0.98))
        acoustic_snr = float(np.clip(7.5 + micro_energy * 13.0 + rng.normal(0.0, 3.0), -4.0, 28.0))
    else:
        source_base = {
            "wind_gust": (55.0, 0.82),
            "road_traffic": (115.0, 0.72),
            "industrial_machinery": (180.0, 0.58),
            "generator_hum": (120.0, 0.48),
            "thunder_rumble": (38.0, 0.88),
            "none": (95.0, 0.70),
        }.get(false_source, (100.0, 0.70))
        engine_peak = float(np.clip(source_base[0] + rng.normal(0.0, 32.0), 18.0, 320.0))
        prop_harmonic = float(np.clip(engine_peak * rng.uniform(1.2, 3.5), 35.0, 720.0))
        entropy = float(np.clip(source_base[1] + rng.normal(0.0, 0.10), 0.18, 0.98))
        modulation = float(np.clip(0.08 + rng.beta(1.2, 5.5) * 0.42, 0.02, 0.62))
        acoustic_snr = float(np.clip(rng.normal(2.0, 4.0), -9.0, 18.0))
    return {
        "engine_band_peak_hz": engine_peak,
        "prop_harmonic_hz": prop_harmonic,
        "spectral_bandwidth_hz": float(np.clip(micro_bandwidth_hz + rng.normal(0.0, 22.0), 18.0, 420.0)),
        "spectral_entropy": entropy,
        "modulation_confidence": modulation,
        "acoustic_snr_db": acoustic_snr,
    }


def build_acoustic_cue_products(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    node_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    phase_metric_rows: list[dict[str, Any]] = []
    track_count_by_record: dict[str, int] = {}
    first_latency_by_record: dict[str, float] = {}
    pre_los_cue_count = 0
    false_source_counts: dict[str, int] = {source: 0 for source in ACOUSTIC_FALSE_CUE_SOURCES}

    for record_idx, record in records.iterrows():
        record_id = str(record["record_id"])
        rng = np.random.default_rng(
            stable_seed(int(record["scenario_seed"]), int(record["object_seed"]), record_idx, 66_211)
        )
        valid = valid_mask[record_idx]
        frame = frames[record_idx, valid]
        if frame.size == 0:
            continue
        mean_range_m = float(np.mean(frame[:, FRAME_INDEX["range_m"]]))
        micro_peak_hz = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_peak_hz_proxy"]]))
        micro_bw_hz = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_bandwidth_hz_proxy"]]))
        micro_energy = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_energy"]]))
        horizon_masked = float(record["horizon_masked_fraction"])
        probability = acoustic_detection_probability(record, mean_range_m, horizon_masked)
        source_count = len(ACOUSTIC_NODE_IDS)
        node_hits = rng.random(source_count) < probability
        if int(node_hits.sum()) < 2:
            continue

        scene_role = str(record["scene_role"])
        signal_like = bool(record["is_public_proxy_positive"]) or scene_role == "target_masked_counterfactual"
        false_source = "none" if signal_like else acoustic_false_source(rng, record)
        if false_source in false_source_counts:
            false_source_counts[false_source] += 1
        spectral = acoustic_spectral_features(rng, signal_like, false_source, micro_peak_hz, micro_bw_hz, micro_energy)
        phase_id = str(record["phase_id"])
        phase_duration_s = float(record["phase_end_s"]) - float(record["phase_start_s"])
        cue_latency_s = float(np.clip(rng.gamma(2.0, 2.2), 0.5, max(1.0, min(phase_duration_s - 0.5, 28.0))))
        bearing_center = float(rng.uniform(0.0, 360.0))
        detected_nodes = [ACOUSTIC_NODE_IDS[idx] for idx, hit in enumerate(node_hits) if bool(hit)]
        if phase_id == "initial_take_up" and horizon_masked > 0.55 and signal_like:
            pre_los_cue_count += 1
        track_fragments = 1 + int(signal_like and rng.random() < (0.08 + 0.12 * horizon_masked))

        for fragment_idx in range(track_fragments):
            fragment_latency = cue_latency_s + fragment_idx * float(rng.uniform(4.0, 11.0))
            fragment_bearings: list[float] = []
            track_id = f"acu_track_{record_id}_{fragment_idx + 1:02d}"
            event_id = f"acu_event_{record_id}_{fragment_idx + 1:02d}"
            for node_idx, source_id in enumerate(detected_nodes):
                doa_uncertainty = float(np.clip(rng.normal(8.0, 2.5) + mean_range_m / 7_500.0, 2.0, 24.0))
                doa = float((bearing_center + rng.normal(0.0, doa_uncertainty)) % 360.0)
                fragment_bearings.append(doa)
                node_rows.append(
                    {
                        "record_id": record_id,
                        "phase_id": phase_id,
                        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
                        "acoustic_event_id": event_id,
                        "acoustic_track_id": track_id,
                        "source_id": source_id,
                        "cue_time_s": fragment_latency,
                        "cue_latency_s": fragment_latency,
                        "doa_bearing_deg": doa,
                        "doa_uncertainty_deg": doa_uncertainty,
                        "tdoa_pair_id": f"{source_id}:{detected_nodes[(node_idx + 1) % len(detected_nodes)]}",
                        "tdoa_delta_ms": float(rng.normal(0.0, 18.0)),
                        "tdoa_residual_ms": float(abs(rng.normal(3.0, 2.4))),
                        "engine_band_peak_hz": spectral["engine_band_peak_hz"],
                        "prop_harmonic_hz": spectral["prop_harmonic_hz"],
                        "spectral_bandwidth_hz": spectral["spectral_bandwidth_hz"],
                        "spectral_entropy": spectral["spectral_entropy"],
                        "modulation_confidence": spectral["modulation_confidence"],
                        "acoustic_snr_db": spectral["acoustic_snr_db"],
                        "false_cue_source": false_source,
                    }
                )

            source_weight = max(1.0, math.sqrt(len(detected_nodes)))
            uncertainty_major = float(np.clip(mean_range_m * rng.uniform(0.10, 0.24) / source_weight, 45.0, 2_800.0))
            uncertainty_minor = float(np.clip(uncertainty_major * rng.uniform(0.38, 0.72), 20.0, uncertainty_major))
            confidence = float(
                np.clip(
                    0.18
                    + 0.12 * len(detected_nodes)
                    + spectral["modulation_confidence"] * 0.34
                    + max(0.0, spectral["acoustic_snr_db"]) * 0.012
                    - (0.18 if false_source != "none" else 0.0),
                    0.02,
                    0.99,
                )
            )
            triangulated_range = float(np.clip(mean_range_m + rng.normal(0.0, uncertainty_major), 100.0, 22_000.0))
            track_rows.append(
                {
                    "record_id": record_id,
                    "phase_id": phase_id,
                    "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
                    "acoustic_track_id": track_id,
                    "source_count": len(detected_nodes),
                    "source_ids": ";".join(detected_nodes),
                    "cue_time_s": fragment_latency,
                    "first_cue_latency_s": fragment_latency,
                    "triangulated_bearing_deg": circular_mean_deg(fragment_bearings),
                    "triangulated_range_m": triangulated_range,
                    "uncertainty_major_m": uncertainty_major,
                    "uncertainty_minor_m": uncertainty_minor,
                    "uncertainty_orientation_deg": float(rng.uniform(0.0, 180.0)),
                    "cue_confidence": confidence,
                    "engine_band_peak_hz": spectral["engine_band_peak_hz"],
                    "prop_harmonic_hz": spectral["prop_harmonic_hz"],
                    "spectral_entropy": spectral["spectral_entropy"],
                    "modulation_confidence": spectral["modulation_confidence"],
                    "acoustic_snr_db": spectral["acoustic_snr_db"],
                    "false_cue_source": false_source,
                }
            )
        track_count_by_record[record_id] = track_count_by_record.get(record_id, 0) + track_fragments
        first_latency_by_record[record_id] = min(first_latency_by_record.get(record_id, float("inf")), cue_latency_s)

    track_df = pd.DataFrame(track_rows)
    node_df = pd.DataFrame(node_rows)
    for phase_id, phase_records in records.groupby("phase_id"):
        ids = phase_records["record_id"].astype(str).to_numpy()
        labels = phase_records["scene_role"].astype(str).isin(
            ["positive_public_proxy", "target_masked_counterfactual"]
        ).to_numpy()
        has_track = np.asarray([track_count_by_record.get(str(record_id), 0) > 0 for record_id in ids], dtype=bool)
        track_counts = np.asarray([track_count_by_record.get(str(record_id), 0) for record_id in ids], dtype=np.float64)
        positive_latency = [
            first_latency_by_record[str(record_id)]
            for record_id, label in zip(ids, labels)
            if label and str(record_id) in first_latency_by_record
        ]
        positive_count = int(labels.sum())
        negative_count = int((~labels).sum())
        phase_metric_rows.append(
            {
                "phase_id": str(phase_id),
                "record_count": int(len(ids)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "pd_any_acoustic_cue": float(has_track[labels].mean()) if positive_count else float("nan"),
                "pfa_any_acoustic_cue": float(has_track[~labels].mean()) if negative_count else float("nan"),
                "first_cue_latency_s": float(np.mean(positive_latency)) if positive_latency else float("nan"),
                "track_initiation_latency_s": float(np.mean(positive_latency)) if positive_latency else float("nan"),
                "track_fragmentation_rate": float(np.mean(np.maximum(track_counts[labels] - 1.0, 0.0))) if positive_count else float("nan"),
                "false_track_rate": float(has_track[~labels].mean()) if negative_count else float("nan"),
                "missed_track_rate": float((~has_track[labels]).mean()) if positive_count else float("nan"),
                "pre_radar_los_cue_count": int(
                    sum(
                        1
                        for _, row in phase_records.iterrows()
                        if str(row["scene_role"]) in {"positive_public_proxy", "target_masked_counterfactual"}
                        and str(row["phase_id"]) == "initial_take_up"
                        and float(row["horizon_masked_fraction"]) > 0.55
                        and track_count_by_record.get(str(row["record_id"]), 0) > 0
                    )
                ),
                "false_cue_sources_reported": ";".join(
                    source for source, count in sorted(false_source_counts.items()) if count > 0
                ),
            }
        )
    phase_df = pd.DataFrame(phase_metric_rows).sort_values("phase_id")
    required_node_cols = {
        "record_id",
        "phase_id",
        "detector_family_id",
        "source_id",
        "doa_bearing_deg",
        "tdoa_residual_ms",
        "engine_band_peak_hz",
        "prop_harmonic_hz",
        "false_cue_source",
    }
    required_track_cols = {
        "record_id",
        "phase_id",
        "detector_family_id",
        "source_count",
        "triangulated_bearing_deg",
        "uncertainty_major_m",
        "uncertainty_minor_m",
        "cue_confidence",
        "false_cue_source",
    }
    node_columns_ok = required_node_cols.issubset(set(node_df.columns))
    track_columns_ok = required_track_cols.issubset(set(track_df.columns))
    phase_ids_ok = set(phase_df["phase_id"].astype(str)) == {phase.phase_id for phase in PHASE_SPECS}
    raw_audio_columns = [
        col
        for col in list(node_df.columns) + list(track_df.columns)
        if "raw_audio" in str(col).lower() or str(col).lower() in {"audio", "audio_path", "waveform_path"}
    ]
    false_sources_present = sum(1 for count in false_source_counts.values() if count > 0) >= 2
    summary_status = (
        "pass"
        if node_columns_ok
        and track_columns_ok
        and phase_ids_ok
        and not raw_audio_columns
        and pre_los_cue_count > 0
        and false_sources_present
        else "fail"
    )
    schema = {
        "artifact": "acoustic_cueing_products",
        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
        "claim_boundary": "Synthetic passive acoustic cue products only; no raw audio, measured acoustic traces, proprietary signatures, or exact platform audio fidelity.",
        "files": {
            "node_detections": "acoustic_node_detections.csv",
            "cue_tracks": "acoustic_cue_tracks.csv",
            "phase_metrics": "acoustic_phase_metrics.csv",
        },
        "node_detection_columns": list(node_df.columns),
        "cue_track_columns": list(track_df.columns),
        "phase_metric_columns": list(phase_df.columns),
        "raw_audio_stored": False,
        "model_facing_default": False,
    }
    summary = {
        "status": summary_status,
        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
        "node_detection_count": int(len(node_df)),
        "cue_track_count": int(len(track_df)),
        "phase_ids": sorted(phase_df["phase_id"].astype(str).unique().tolist()),
        "raw_audio_stored": False,
        "raw_audio_column_violations": raw_audio_columns,
        "pre_radar_los_cue_count": int(pre_los_cue_count),
        "false_cue_source_counts": false_source_counts,
        "required_metrics": [
            "pd_any_acoustic_cue",
            "pfa_any_acoustic_cue",
            "first_cue_latency_s",
            "track_initiation_latency_s",
            "track_fragmentation_rate",
            "false_track_rate",
            "missed_track_rate",
        ],
    }
    return node_df, track_df, phase_df, schema, summary


def write_metadata(
    out_root: Path,
    records: pd.DataFrame,
    feature_names: list[str],
    quality: dict[str, Any],
    phases: list[PhaseSpec],
    args: argparse.Namespace,
) -> None:
    split_manifest = records[
        [
            "record_id",
            "split",
            "scenario_seed",
            "object_seed",
            "class_id",
            "target_family",
            "scene_role",
            "phase_id",
            "counterfactual_group_id",
            "scenario_id",
        ]
    ].copy()
    split_manifest.insert(2, "split_key_kind", "counterfactual_group_phase_site_sensor")
    split_manifest.insert(
        3,
        "split_key",
        records["counterfactual_group_id"].astype(str)
        + ":"
        + records["phase_id"].astype(str)
        + ":"
        + records["site_archetype_id"].astype(str)
        + ":"
        + records["sensor_archetype_id"].astype(str),
    )
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)

    denylist = {
        "policy": "Fail generation if restricted generator truth appears in detector-facing frame or aggregate feature columns.",
        "restricted_feature_names": sorted(RESTRICTED_FEATURE_NAMES),
        "published_frame_columns": FRAME_COLUMNS,
        "published_aggregate_feature_columns": feature_names,
        "violations": denylist_violations(FRAME_COLUMNS, feature_names),
        "status": "pass" if not denylist_violations(FRAME_COLUMNS, feature_names) else "fail",
    }
    write_json(out_root / "generator_truth_denylist.json", denylist)
    if denylist["violations"]:
        raise AssertionError(f"restricted truth leaked into feature columns: {denylist['violations']}")

    write_json(
        out_root / "feature_schema.json",
        {
            "benchmark_profile": "ml-training-three-tier",
            "frame_period_s": FRAME_PERIOD_S,
            "frame_columns": FRAME_COLUMNS,
            "valid_frame_mask": "valid_frame_mask.npz",
            "aggregate_feature_columns": feature_names,
            "phase_model": [asdict(phase) for phase in phases],
            "sensor_public_contract": {
                "allowed_to_models": [
                    "frame_features.npz frame tensor columns",
                    "valid_frame_mask.npz causal valid-frame mask",
                    "features.csv aggregate detector observables",
                ],
                "detector_sidecar_products": [
                    "acoustic_node_detections.csv",
                    "acoustic_cue_tracks.csv",
                    "acoustic_phase_metrics.csv",
                    "acoustic_product_schema.json",
                ],
                "not_allowed_to_models": [
                    "records.csv generator metadata",
                    "split_manifest.csv split keys",
                    "restricted_truth/*.json",
                    "kinematics_audit.csv",
                    "speed_prior_manifest.json",
                    "calibration_report.json",
                    "calibration_anchor_manifest.json",
                    "calibration_distance.csv",
                    "acoustic products unless explicitly wired by a later fusion lane",
                    "phase_id",
                    "scenario_seed",
                    "object_seed",
                    "class_id",
                    "target_family",
                    "scene_role",
                    "altitude_m",
                    "link_budget_snr_db",
                    "raw_rcs_dbsm",
                    "true_speed_mps",
                    "source metadata",
                ],
            },
            "notes": "radial_velocity_mps is a radar observable. True speed and altitude are restricted truth and are not frame features.",
        },
    )
    write_json(out_root / "speed_prior_manifest.json", speed_prior_manifest_payload())
    write_json(
        out_root / "science_assumptions.json",
        {
            "validation_tier": "unvalidated/basic synthetic public-proxy benchmark",
            "claim_boundary": "No measured-target truth, classified fidelity, or proprietary-equivalent sensor behavior is claimed.",
            "three_tier_detection_model": {
                "initial_take_up": "0-30 s, low altitude and geometry limited; low Pd is allowed when LOS is masked.",
                "climb_transition": "30-90 s, track initiation and confirmation under changing aspect.",
                "cruise_altitude": "90+ s, coherent Doppler and micro-Doppler classification interval.",
            },
            "public_speed_priors": speed_prior_manifest_payload()["policy"],
            "speed_prior_manifest": "speed_prior_manifest.json",
            "kinematics_audit": "kinematics_audit.csv",
            "calibration_anchor_manifest": "calibration_anchor_manifest.json",
            "calibration_report": "calibration_report.json",
            "calibration_distance": "calibration_distance.csv",
            "calibration_policy": "Reference-only public-proxy distribution audit. Presence of these files does not promote the smoke benchmark beyond unvalidated/basic.",
            "acoustic_cueing_products": {
                "node_detections": "acoustic_node_detections.csv",
                "cue_tracks": "acoustic_cue_tracks.csv",
                "phase_metrics": "acoustic_phase_metrics.csv",
                "schema": "acoustic_product_schema.json",
                "quality": "acoustic_cue_quality.json",
                "policy": "Synthetic passive acoustic cue summaries only; no raw audio is stored. Fusion with radar belongs to a later fusion lane.",
            },
            "radar_equation_budget": "SNR emerges from power, gains, wavelength, RCS, range^4 loss, propagation loss, clutter loss, receiver noise, bandwidth, processing gain, and system loss. target_snr_db is not a current generation input.",
            "smoke_limitations": [
                "current smoke detector products still include proxy micro-Doppler and spectrum summaries until the complex-IQ end-to-end work slot lands.",
                "Target-masked counterfactuals share background clutter/RFI products with no-target counterfactuals and retain withheld target truth only in restricted_truth.",
                "Calibration-anchor artifacts use reference target distributions and synthetic observations only; lawful measured-anchor distributions are required before any measured-anchored current claim.",
                "Acoustic cue products are generated sidecars and do not claim radar-fused track quality.",
            ],
        },
    )
    write_json(
        out_root / "dataset_card.json",
        {
            "dataset_id": f"shahed136-public-proxy-ml-training-{args.scale_name}",
            "benchmark_profile": "ml-training-three-tier",
            "scenario_groups": int(args.scenario_groups),
            "record_count": int(len(records)),
            "phase_ids": [phase.phase_id for phase in phases],
            "quality_report": "quality_report.json",
            "operational_metrics": "phase_operational_metrics.csv",
            "kinematics_audit": "kinematics_audit.csv",
            "speed_prior_manifest": "speed_prior_manifest.json",
            "calibration_anchor_manifest": "calibration_anchor_manifest.json",
            "calibration_report": "calibration_report.json",
            "calibration_distance": "calibration_distance.csv",
            "acoustic_node_detections": "acoustic_node_detections.csv",
            "acoustic_cue_tracks": "acoustic_cue_tracks.csv",
            "acoustic_phase_metrics": "acoustic_phase_metrics.csv",
            "acoustic_product_schema": "acoustic_product_schema.json",
            "acoustic_cue_quality": "acoustic_cue_quality.json",
            "counterfactual_audit": "negative_control_audit.json",
            "strict_open_boundary": "Synthetic public-proxy benchmark; no measured traces or proprietary-equivalent behavior claims.",
        },
    )
    write_json(
        out_root / "dataset_manifest.json",
        {
            "dataset_id": f"shahed136-public-proxy-ml-training-{args.scale_name}",
            "benchmark_profile": "ml-training-three-tier",
            "files": [
                "records.csv",
                "split_manifest.csv",
                "frame_features.npz",
                "valid_frame_mask.npz",
                "features.csv",
                "kinematics_audit.csv",
                "speed_prior_manifest.json",
                "calibration_anchor_manifest.json",
                "calibration_report.json",
                "calibration_distance.csv",
                "acoustic_node_detections.csv",
                "acoustic_cue_tracks.csv",
                "acoustic_phase_metrics.csv",
                "acoustic_product_schema.json",
                "acoustic_cue_quality.json",
                "phase_operational_metrics.csv",
                "quality_report.json",
                "negative_control_audit.json",
                "generator_truth_denylist.json",
                "feature_schema.json",
                "science_assumptions.json",
                "dataset_card.json",
            ],
        },
    )
    write_json(
        out_root / "runtime_report.json",
        {
            "generator": "detection/generate_ml_training.py",
            "seed": args.seed,
            "scale_name": args.scale_name,
            "quality_status": quality["status"],
            "generated_artifact_policy": "outputs/ is gitignored; do not stage generated records, tensors, or reports.",
        },
    )


def main() -> None:
    args = parse_args()
    if args.scenario_groups < 20:
        raise ValueError("current scenario-groups must be at least 20 for train/validation/test negative-control coverage")
    phases = phase_specs(args.max_time_s)
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace generated artifacts")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "restricted_truth").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    frames_raw: list[np.ndarray] = []
    kinematic_rows: list[dict[str, Any]] = []
    record_index = 0
    for group_index in range(args.scenario_groups):
        group = group_conditions(args.seed, group_index)
        site: SiteArchetype = group["site"]
        sensor: SensorArchetype = group["sensor"]
        for scene_role in ROLES:
            for phase in phases:
                rng = np.random.default_rng(stable_seed(args.seed, group_index, ROLES.index(scene_role), int(phase.start_s), 9_191))
                truth, frame = build_frame_products(rng, group, phase, scene_role)
                frames_raw.append(frame)
                record_id = f"current_record_{record_index:07d}_{phase.phase_id}"
                scenario_id = f"{group['counterfactual_group_id']}:{scene_role}"
                truth_path = f"restricted_truth/{record_id}.json"
                speed_prior = speed_prior_for_family(str(truth["target_family"]))
                write_json(
                    out_root / truth_path,
                    {
                        **truth,
                        "record_id": record_id,
                        "scenario_id": scenario_id,
                        "counterfactual_group_id": group["counterfactual_group_id"],
                        "site_archetype": asdict(site),
                        "sensor_archetype": asdict(sensor),
                    },
                )
                positive = scene_role == "positive_public_proxy"
                mean_range = float(np.mean(frame[:, FRAME_INDEX["range_m"]]))
                record = {
                    "record_id": record_id,
                    "record_index": record_index,
                    "split": group["split"],
                    "class_id": truth["class_id"],
                    "target_family": truth["target_family"],
                    "scene_role": scene_role,
                    "is_public_proxy_positive": positive,
                    "is_hard_negative": not positive,
                    "hard_negative_family": "" if positive else truth["target_family"],
                    "scenario_seed": int(group["scenario_seed"]),
                    "object_seed": int(stable_seed(args.seed, group_index, int(phase.start_s), 12_001)),
                    "frame_count": int(frame.shape[0]),
                    "cpi_pulses": int(frame[0, FRAME_INDEX["cpi_pulses"]]),
                    "streaming_features_path": f"frame_features.npz#record_id={record_id}",
                    "frame_labels_path": "feature_schema.json",
                    "truth_metadata_path": truth_path,
                    "stratum_id": f"current_{site.site_archetype_id}_{sensor.sensor_archetype_id}_{phase.phase_id}_{group['clutter_regime']}",
                    "wave_index": int(group_index),
                    "difficulty_bucket": "initial_los_limited" if phase.phase_id == "initial_take_up" else "phase_realism",
                    "sensor_band": sensor.band,
                    "range_bin": range_bin(mean_range),
                    "clutter_regime": group["clutter_regime"],
                    "target_aspect": group["target_aspect"],
                    "motion_pattern": "three_tier_public_proxy_path" if positive else "matched_confuser_path",
                    "interference": group["interference"],
                    "confuser_family": "" if positive else truth["target_family"],
                    "scene_object_count": 1 if scene_role != "no_target_counterfactual" else 0,
                    "mixed_scene": bool(group["multipath_enabled"]),
                    "holdout_role": "unseen_site_sensor_confuser" if group["split"] == "test" else "seen",
                    "phase_id": phase.phase_id,
                    "phase_start_s": phase.start_s,
                    "phase_end_s": phase.end_s,
                    "available_history_s": phase.end_s - phase.start_s,
                    "site_archetype_id": site.site_archetype_id,
                    "sensor_archetype_id": sensor.sensor_archetype_id,
                    "validation_tier": "unvalidated/basic synthetic public-proxy",
                    "counterfactual_group_id": group["counterfactual_group_id"],
                    "scenario_id": scenario_id,
                    "horizon_masked_fraction": truth["horizon_masked_fraction"],
                    "los_eligible_fraction": truth["los_eligible_fraction"],
                }
                records.append(record)
                true_speed = float(truth["mean_true_speed_mps"])
                radial_speed = float(truth["mean_abs_radial_velocity_mps"])
                kinematic_rows.append(
                    {
                        "record_id": record_id,
                        "record_index": record_index,
                        "split": group["split"],
                        "phase_id": phase.phase_id,
                        "scene_role": scene_role,
                        "target_family": truth["target_family"],
                        "speed_prior_id": speed_prior.prior_id,
                        "propulsion_class": speed_prior.propulsion_class,
                        "stress_class": bool(speed_prior.stress_class),
                        "baseline_positive_prior": bool(speed_prior.baseline_positive),
                        "mean_true_speed_mps": true_speed,
                        "mean_abs_radial_velocity_mps": radial_speed,
                        "radial_to_true_speed_ratio": radial_speed / max(abs(true_speed), 1.0),
                        "radial_velocity_is_true_speed": False,
                        "estimated_ground_speed_exposed": False,
                        "cruise_main_estimate_low_mps": (
                            speed_prior.cruise_main_estimate_mps[0]
                            if speed_prior.cruise_main_estimate_mps is not None
                            else ""
                        ),
                        "cruise_main_estimate_high_mps": (
                            speed_prior.cruise_main_estimate_mps[1]
                            if speed_prior.cruise_main_estimate_mps is not None
                            else ""
                        ),
                    }
                )
                record_index += 1

    order_rng = np.random.default_rng(stable_seed(args.seed, args.scenario_groups, 44_404))
    order = order_rng.permutation(len(records))
    records = [records[int(idx)] for idx in order]
    frames_raw = [frames_raw[int(idx)] for idx in order]
    kinematic_rows = [kinematic_rows[int(idx)] for idx in order]
    for idx, record in enumerate(records):
        record["record_index"] = idx
        kinematic_rows[idx]["record_index"] = idx

    frames, valid_mask = pad_frames(frames_raw)
    records_df = pd.DataFrame(records)
    records_df.to_csv(out_root / "records.csv", index=False)
    kinematics_df, kinematics_summary = build_kinematics_audit(records_df, kinematic_rows)
    kinematics_df.to_csv(out_root / "kinematics_audit.csv", index=False, float_format="%.6f")
    np.savez_compressed(
        out_root / "frame_features.npz",
        frames=frames,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U96"),
        frame_columns=np.asarray(FRAME_COLUMNS, dtype="<U64"),
        frame_period_s=np.array(FRAME_PERIOD_S, dtype=np.float32),
        benchmark_profile=np.asarray(["ml-training-three-tier"], dtype="<U48"),
    )
    np.savez_compressed(
        out_root / "valid_frame_mask.npz",
        valid_frame_mask=valid_mask,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U96"),
        benchmark_profile=np.asarray(["ml-training-valid-frame-mask"], dtype="<U48"),
    )
    feature_df, feature_names = aggregate_features(frames, valid_mask)
    feature_df.insert(0, "record_id", records_df["record_id"].to_numpy())
    feature_df.to_csv(out_root / "features.csv", index=False, float_format="%.6f")
    phase_metrics = operational_metrics(records_df, frames, valid_mask)
    phase_metrics.to_csv(out_root / "phase_operational_metrics.csv", index=False, float_format="%.6f")
    controls = negative_control_audit(records_df, feature_df)
    write_json(out_root / "negative_control_audit.json", controls)
    calibration_report, calibration_manifest, calibration_distance_df, calibration_summary = build_calibration_artifacts(
        records_df,
        frames,
        valid_mask,
    )
    write_json(out_root / "calibration_report.json", calibration_report)
    write_json(out_root / "calibration_anchor_manifest.json", calibration_manifest)
    calibration_distance_df.to_csv(out_root / "calibration_distance.csv", index=False, float_format="%.6f")
    acoustic_node_df, acoustic_track_df, acoustic_phase_df, acoustic_schema, acoustic_summary = build_acoustic_cue_products(
        records_df,
        frames,
        valid_mask,
    )
    acoustic_node_df.to_csv(out_root / "acoustic_node_detections.csv", index=False, float_format="%.6f")
    acoustic_track_df.to_csv(out_root / "acoustic_cue_tracks.csv", index=False, float_format="%.6f")
    acoustic_phase_df.to_csv(out_root / "acoustic_phase_metrics.csv", index=False, float_format="%.6f")
    write_json(out_root / "acoustic_product_schema.json", acoustic_schema)
    write_json(out_root / "acoustic_cue_quality.json", acoustic_summary)

    phase_ids = set(records_df["phase_id"].astype(str))
    expected_phase_ids = {phase.phase_id for phase in phases}
    phase_windows_ok = expected_phase_ids == phase_ids and {
        row.phase_id: (row.start_s, row.end_s) for row in phases
    } == {
        "initial_take_up": (0.0, 30.0),
        "climb_transition": (30.0, 90.0),
        "cruise_altitude": (90.0, float(args.max_time_s)),
    }
    quality = {
        "benchmark_profile": "ml-training-three-tier",
        "record_count": int(len(records_df)),
        "scenario_group_count": int(args.scenario_groups),
        "phase_ids": sorted(phase_ids),
        "phase_windows_status": "pass" if phase_windows_ok else "fail",
        "negative_control_status": controls["status"],
        "negative_control_summary": {key: value for key, value in controls.items() if key.endswith("_auc")},
        "truth_denylist_status": "pass",
        "speed_prior_kinematics_status": kinematics_summary["status"],
        "speed_prior_kinematics_summary": kinematics_summary,
        "calibration_anchor_status": calibration_summary["calibration_anchor_status"],
        "calibration_artifact_status": calibration_summary["status"],
        "calibration_anchor_summary": calibration_summary,
        "acoustic_cueing_status": acoustic_summary["status"],
        "acoustic_cueing_summary": acoustic_summary,
        "initial_take_up_low_pd_allowed": True,
        "operational_metrics": phase_metrics.to_dict(orient="records"),
        "acoustic_operational_metrics": acoustic_phase_df.to_dict(orient="records"),
        "status": (
            "pass"
            if phase_windows_ok
            and controls["status"] == "pass"
            and kinematics_summary["status"] == "pass"
            and calibration_summary["status"] == "pass"
            and acoustic_summary["status"] == "pass"
            else "fail"
        ),
    }
    write_metadata(out_root, records_df, feature_names, quality, phases, args)
    write_json(out_root / "quality_report.json", quality)
    if not phase_windows_ok:
        raise AssertionError("current phase windows do not match initial_take_up/climb_transition/cruise_altitude semantics")
    if controls["status"] != "pass":
        raise AssertionError(f"negative-control audit failed: {controls}")
    if kinematics_summary["status"] != "pass":
        raise AssertionError(f"speed-prior kinematics audit failed: {kinematics_summary}")
    if calibration_summary["status"] != "pass":
        raise AssertionError(f"calibration anchor artifact audit failed: {calibration_summary}")
    if acoustic_summary["status"] != "pass":
        raise AssertionError(f"acoustic cueing artifact audit failed: {acoustic_summary}")
    print(
        f"wrote {out_root} groups={args.scenario_groups} records={len(records_df)} "
        f"phases={','.join(sorted(phase_ids))} controls={controls['status']} "
        f"calibration={calibration_summary['calibration_anchor_status']} "
        f"acoustic={acoustic_summary['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
