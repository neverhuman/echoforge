"""Physics-based signal generators, strata builder, and split assignment for the v2 benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ml_training_v2_config import (
    BAND_ADJUST_DB,
    BAND_CENTER_HZ,
    CLUTTER,
    DIFFICULTY,
    FAMILY_TRAITS,
    FRAME_COLUMNS,
    FRAME_COUNT,
    FRAME_INDEX,
    FRAME_PERIOD_S,
    HELDOUT_CONFUSER_FAMILIES,
    HELDOUT_STRATA,
    INTERFERENCE,
    SPLIT_TARGETS,
    ScenarioStratum,
)
CONFUSER_FAMILIES = [name for name in FAMILY_TRAITS if name != "public_proxy_fixed_wing"]


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
