"""Dataclasses and scalar constants for the ML training generator.

Contains only primitive constants and frozen dataclass definitions.
Instance data tables (PHASE_SPECS, SITE_ARCHETYPES, SENSOR_ARCHETYPES,
FAMILY_TRAITS, SPEED_PRIORS, FAMILY_SPEED_PRIOR, RCS_TABLE_DB) live in
generate_ml_training_tables to keep each file under 350 LOC.
"""

from __future__ import annotations

from dataclasses import dataclass


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
