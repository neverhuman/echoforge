"""Configuration: constants, dataclasses, and static lookup tables for the v1 benchmark.

Dataclasses and the large FAMILY_TRAITS / SPEED_PRIORS / archetype tables live in
ml_training_config_types; all public names are re-exported here so existing
importers need no changes.
"""

from __future__ import annotations

from ml_training_config_types import (
    FAMILY_TRAITS,
    PHASE_SPECS,
    SENSOR_ARCHETYPES,
    SITE_ARCHETYPES,
    SPEED_PRIORS,
    FamilyTraits,
    PhaseSpec,
    SensorArchetype,
    SiteArchetype,
    SpeedPrior,
)

__all__ = [
    "FRAME_PERIOD_S",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_SCENARIO_GROUPS",
    "DEFAULT_MAX_TIME_S",
    "NEGATIVE_CONTROL_AUC_GATE",
    "TARGET_MASKED_AUC_GATE",
    "CALIBRATION_SAMPLE_COUNT",
    "ACOUSTIC_DETECTOR_FAMILY_ID",
    "ACOUSTIC_NODE_IDS",
    "ACOUSTIC_FALSE_CUE_SOURCES",
    "FRAME_COLUMNS",
    "FRAME_INDEX",
    "RESTRICTED_FEATURE_NAMES",
    "ROLES",
    "CONFUSER_FAMILIES",
    "PhaseSpec",
    "SiteArchetype",
    "SensorArchetype",
    "FamilyTraits",
    "SpeedPrior",
    "PHASE_SPECS",
    "SITE_ARCHETYPES",
    "SENSOR_ARCHETYPES",
    "FAMILY_TRAITS",
    "SPEED_PRIORS",
    "FAMILY_SPEED_PRIOR",
    "RCS_TABLE_DB",
]

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
