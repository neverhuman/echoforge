"""Dataclasses and static lookup tables for the v1 benchmark configuration.

Extracted from ml_training_config to keep each module ≤350 LOC.
Import these names from ml_training_config for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass


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

