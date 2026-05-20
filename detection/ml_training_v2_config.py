"""Configuration: constants, dataclasses, and lookup tables for the v2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass


FRAME_PERIOD_S = 0.5
MAX_TIME_S = 45.0
FRAME_COUNT = int(MAX_TIME_S / FRAME_PERIOD_S) + 1
DEFAULT_OUT_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
DEFAULT_RECORDS = 50_000
SPLIT_TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SINGLE_FEATURE_AUC_GATE = 0.85
NEGATIVE_CONTROL_AUC_GATE = 0.60
MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ = 320.0
MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ = 420.0
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
