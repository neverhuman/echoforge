"""Physics-based signal generators for the v1 benchmark.

Covers waveform/signal synthesis: kinematics, radar equation, frame product
assembly, frame padding, and aggregate feature extraction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml_training_config import (
    FAMILY_TRAITS,
    FRAME_COLUMNS,
    FRAME_INDEX,
    FRAME_PERIOD_S,
    FamilyTraits,
    PhaseSpec,
    RCS_TABLE_DB,
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
