"""Shared constants for the Shahed-136/Geran-2 public-proxy main run.

The main-run artifacts are strict-open synthetic proxy data. Labels and
restricted generator metadata are separated from detector-facing features so
the exported views can be audited without presenting measured-platform claims.
"""

from __future__ import annotations

from dataclasses import dataclass


POSITIVE_MODEL_LABEL = "shahed_136_geran_2_public_proxy"
NEGATIVE_MODEL_LABEL = "confuser_or_sensor_artifact"
DATASET_PROFILE = "runit-shahed136-geran2-public-proxy-main-run"
DEFAULT_SCENARIO_GROUPS = 10_000
DEFAULT_POSITIVE_GROUPS = 250
DEFAULT_HOLDOUT_GROUPS = 1_500
DEFAULT_HOLDOUT_POSITIVES = 38
DEFAULT_FOLDS = 5
DEFAULT_SEED = 202605210136
DEFAULT_SHARD_SIZE = 512
TIMESTAMP_EPOCH_NS = 1_790_000_000_000_000_000


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    start_s: float
    end_s: float
    positive_gain: float
    clutter_gain: float


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("initial_take_up", 0.0, 30.0, 0.78, 1.26),
    PhaseSpec("climb_transition", 30.0, 90.0, 1.02, 1.05),
    PhaseSpec("cruise_altitude", 90.0, 150.0, 1.18, 0.94),
)


@dataclass(frozen=True)
class ActiveRadarSensor:
    sensor_id: str
    view_id: str
    display_name: str
    band: str
    pulses: int
    range_bins: int
    channel_gain: float


ACTIVE_RADAR_SENSORS: tuple[ActiveRadarSensor, ...] = (
    ActiveRadarSensor(
        "high_resolution_xku_cuas",
        "high_resolution_xku_cuas",
        "High-resolution X/Ku C-UAS radar",
        "x_ku",
        24,
        20,
        1.10,
    ),
    ActiveRadarSensor(
        "tactical_s_band_aesa",
        "tactical_s_band_aesa",
        "Tactical S-band AESA radar",
        "s_band",
        24,
        20,
        0.92,
    ),
    ActiveRadarSensor(
        "gbad_3d4d_cueing",
        "gbad_3d4d_cueing",
        "Medium-range 3D/4D GBAD cueing radar",
        "gbad_3d4d",
        24,
        20,
        0.98,
    ),
)


ACOUSTIC_VIEW_ID = "distributed_acoustic_cue"
FUSION_VIEW_ID = "layered_fusion_c2"
ACOUSTIC_NODE_COUNT = 4
ACOUSTIC_SAMPLE_COUNT = 96


NEGATIVE_ROLES: tuple[str, ...] = (
    "single_bird",
    "bird_flock",
    "rc_fixed_wing",
    "weather_cell",
    "ground_vehicle",
    "wind_turbine",
    "terrain_glint",
    "multipath_ghost",
    "rfi_burst",
    "clutter_only_counterfactual",
)


SITE_ARCHETYPES: tuple[str, ...] = (
    "open_desert_edge",
    "coastal_haze",
    "urban_edge",
    "vegetation_motion_corridor",
    "rolling_terrain",
)


ASPECT_BUCKETS: tuple[str, ...] = (
    "nose_on",
    "tail_on",
    "broadside_left",
    "broadside_right",
    "quartering",
)


RANGE_BANDS: tuple[str, ...] = (
    "near",
    "mid",
    "far",
    "edge_of_track",
)


NOISE_REGIMES: tuple[str, ...] = (
    "weibull_clutter",
    "k_like_clutter",
    "urban_edge",
    "vegetation_motion",
    "sea_clutter",
    "terrain_glint",
    "rain",
    "dust",
    "open_sky",
    "rfi_burst",
    "agc_compression",
    "dropped_cpi",
    "prf_ambiguity",
    "doppler_folding",
    "clock_drift",
    "calibration_offset",
    "multipath_masking",
    "dropout",
    "occlusion",
    "scintillation",
    "quantization",
    "mixed_scene",
)


DETECTOR_ID_COLUMNS: tuple[str, ...] = (
    "record_id",
    "scenario_group_id",
    "time_lock_id",
    "phase_id",
    "timestamp_start_ns",
    "timestamp_end_ns",
    "split_role",
    "cv_fold",
    "model_label",
    "label_id",
    "raw_complex_iq_ref",
    "acoustic_stream_ref",
    "passive_rf_ref",
)


MODEL_FEATURE_DENYLIST: tuple[str, ...] = (
    "scenario_seed",
    "generator_truth",
    "target_role",
    "target_platform_family",
    "target_public_proxy_family",
    "scenario_group_id",
    "split_key",
    "split_role",
    "cv_fold",
    "restricted_truth_path",
    "time_lock_id",
)


DETECTOR_VIEW_IDS: tuple[str, ...] = (
    "high_resolution_xku_cuas",
    "tactical_s_band_aesa",
    "gbad_3d4d_cueing",
    ACOUSTIC_VIEW_ID,
    FUSION_VIEW_ID,
)
