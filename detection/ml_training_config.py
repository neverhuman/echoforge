"""Configuration: constants, dataclasses, and static lookup tables for the v1 benchmark."""

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
