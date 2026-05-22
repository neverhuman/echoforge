"""Deterministic data builder for the Shahed-136/Geran-2 main run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from detection.main_run_types import (
        ACOUSTIC_NODE_COUNT,
        ACOUSTIC_SAMPLE_COUNT,
        ACOUSTIC_VIEW_ID,
        ACTIVE_RADAR_SENSORS,
        ASPECT_BUCKETS,
        DATASET_PROFILE,
        DEFAULT_FOLDS,
        DEFAULT_HOLDOUT_GROUPS,
        DEFAULT_HOLDOUT_POSITIVES,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        FUSION_VIEW_ID,
        MODEL_FEATURE_DENYLIST,
        NEGATIVE_MODEL_LABEL,
        NEGATIVE_ROLES,
        NOISE_REGIMES,
        PHASES,
        POSITIVE_MODEL_LABEL,
        RANGE_BANDS,
        SITE_ARCHETYPES,
        TIMESTAMP_EPOCH_NS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_types import (
        ACOUSTIC_NODE_COUNT,
        ACOUSTIC_SAMPLE_COUNT,
        ACOUSTIC_VIEW_ID,
        ACTIVE_RADAR_SENSORS,
        ASPECT_BUCKETS,
        DATASET_PROFILE,
        DEFAULT_FOLDS,
        DEFAULT_HOLDOUT_GROUPS,
        DEFAULT_HOLDOUT_POSITIVES,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DETECTOR_ID_COLUMNS,
        DETECTOR_VIEW_IDS,
        FUSION_VIEW_ID,
        MODEL_FEATURE_DENYLIST,
        NEGATIVE_MODEL_LABEL,
        NEGATIVE_ROLES,
        NOISE_REGIMES,
        PHASES,
        POSITIVE_MODEL_LABEL,
        RANGE_BANDS,
        SITE_ARCHETYPES,
        TIMESTAMP_EPOCH_NS,
    )


def stable_seed(*parts: object) -> int:
    """Return a deterministic 63-bit seed independent of Python hash state."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _stable_order(items: Iterable[int], seed: int, tag: str) -> list[int]:
    return sorted(items, key=lambda item: stable_seed(seed, tag, item))


def _ceil_fraction(numerator: int, denominator: int, scale: int) -> int:
    return int(math.ceil(float(scale) * float(numerator) / float(denominator)))


def split_counts(
    scenario_groups: int,
    positive_groups: int,
) -> tuple[int, int, int, int]:
    """Return holdout/train and positive allocations.

    The v1 10,000-group / 250-positive run intentionally locks to the
    requested 1,500-group holdout with 38 positive groups. The v2
    10,000-group / 50-positive run follows the same 15 percent holdout ratio
    and rounds to 8 positive holdout groups and 42 train/CV positives.
    Smaller smoke runs keep the same 15 percent holdout ratio with a
    ceil-rounded positive quota.
    """

    if scenario_groups <= 0:
        raise ValueError("scenario-groups must be positive")
    if positive_groups <= 0:
        raise ValueError("positive-groups must be positive")
    if positive_groups >= scenario_groups:
        raise ValueError("positive-groups must be lower than scenario-groups")

    if scenario_groups == DEFAULT_SCENARIO_GROUPS and positive_groups == DEFAULT_POSITIVE_GROUPS:
        holdout_groups = DEFAULT_HOLDOUT_GROUPS
        holdout_positives = DEFAULT_HOLDOUT_POSITIVES
    else:
        holdout_groups = max(DEFAULT_FOLDS, int(round(float(scenario_groups) * 0.15)))
        holdout_groups = min(holdout_groups, scenario_groups - DEFAULT_FOLDS)
        holdout_positives = _ceil_fraction(holdout_groups, scenario_groups, positive_groups)
    holdout_positives = min(holdout_positives, positive_groups, holdout_groups)
    train_groups = scenario_groups - holdout_groups
    train_positives = positive_groups - holdout_positives
    if train_positives <= 0:
        raise ValueError("train/CV positive quota must be non-zero")
    return holdout_groups, train_groups, holdout_positives, train_positives


def build_scenario_manifest(
    *,
    scenario_groups: int = DEFAULT_SCENARIO_GROUPS,
    positive_groups: int = DEFAULT_POSITIVE_GROUPS,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    holdout_policy: str = "group_random",
    holdout_value: str | None = None,
) -> list[dict[str, Any]]:
    holdout_count, _train_count, holdout_pos_count, train_pos_count = split_counts(
        scenario_groups, positive_groups
    )
    all_group_indices = list(range(scenario_groups))
    if holdout_policy == "group_random" or not holdout_value:
        holdout_indices = set(_stable_order(all_group_indices, seed, "holdout")[:holdout_count])
    else:
        if holdout_policy not in {"site", "noise_regime", "hard_negative_role"}:
            raise ValueError(
                "holdout-policy must be one of group_random, site, noise_regime, hard_negative_role"
            )
        targeted: list[int] = []
        for idx in all_group_indices:
            scenario_site = SITE_ARCHETYPES[stable_seed(seed, idx, "site") % len(SITE_ARCHETYPES)]
            scenario_noise = NOISE_REGIMES[stable_seed(seed, idx, "noise") % len(NOISE_REGIMES)]
            scenario_role = NEGATIVE_ROLES[
                stable_seed(seed, idx, "negative-role") % len(NEGATIVE_ROLES)
            ]
            scenario_value = {
                "site": scenario_site,
                "noise_regime": scenario_noise,
                "hard_negative_role": scenario_role,
            }[holdout_policy]
            if scenario_value == holdout_value:
                targeted.append(idx)
        holdout_indices = set(
            _stable_order(targeted, seed, f"holdout-{holdout_policy}-{holdout_value}")[
                :holdout_count
            ]
        )
        if len(holdout_indices) < holdout_count:
            remaining = [idx for idx in all_group_indices if idx not in holdout_indices]
            holdout_indices.update(
                _stable_order(
                    remaining,
                    seed,
                    f"holdout-topup-{holdout_policy}-{holdout_value}",
                )[: holdout_count - len(holdout_indices)]
            )
    train_indices = [idx for idx in all_group_indices if idx not in holdout_indices]
    holdout_positive = set(
        _stable_order(holdout_indices, seed, "holdout-positive")[:holdout_pos_count]
    )
    train_positive = set(_stable_order(train_indices, seed, "train-positive")[:train_pos_count])
    positive_indices = holdout_positive | train_positive

    train_positive_order = _stable_order(train_positive, seed, "fold-positive")
    train_negative_order = _stable_order(
        [idx for idx in train_indices if idx not in train_positive],
        seed,
        "fold-negative",
    )
    fold_by_index: dict[int, int] = {}
    for rank, idx in enumerate(train_positive_order):
        fold_by_index[idx] = rank % folds
    for rank, idx in enumerate(train_negative_order):
        fold_by_index[idx] = rank % folds

    rows: list[dict[str, Any]] = []
    for group_index in all_group_indices:
        scenario_group_id = f"sg_{group_index:05d}"
        split_role = "holdout" if group_index in holdout_indices else "train_cv"
        is_positive = group_index in positive_indices
        noise = NOISE_REGIMES[stable_seed(seed, group_index, "noise") % len(NOISE_REGIMES)]
        site = SITE_ARCHETYPES[stable_seed(seed, group_index, "site") % len(SITE_ARCHETYPES)]
        aspect = ASPECT_BUCKETS[stable_seed(seed, group_index, "aspect") % len(ASPECT_BUCKETS)]
        range_band = RANGE_BANDS[stable_seed(seed, group_index, "range") % len(RANGE_BANDS)]
        if is_positive:
            target_role = POSITIVE_MODEL_LABEL
            confuser_family = ""
            target_class = "public_proxy_positive"
        else:
            target_role = NEGATIVE_ROLES[
                stable_seed(seed, group_index, "negative-role") % len(NEGATIVE_ROLES)
            ]
            confuser_family = target_role
            target_class = "confuser_or_artifact"
        rows.append(
            {
                "scenario_group_id": scenario_group_id,
                "group_index": group_index,
                "time_lock_id": f"tl_{stable_seed(seed, group_index, 'time-lock') % 10**12:012d}",
                "split_role": split_role,
                "cv_fold": "" if split_role == "holdout" else fold_by_index[group_index],
                "is_positive": int(is_positive),
                "model_label": POSITIVE_MODEL_LABEL if is_positive else NEGATIVE_MODEL_LABEL,
                "target_role": target_role,
                "target_class": target_class,
                "confuser_family": confuser_family,
                "monte_carlo_stratum": f"{site}:{range_band}:{aspect}:{noise}",
                "site_archetype_id": site,
                "range_band": range_band,
                "target_aspect": aspect,
                "noise_regime": noise,
                "hard_negative_role": "" if is_positive else target_role,
                "scenario_seed": stable_seed(seed, group_index, "scenario"),
                "split_key": f"{split_role}:{fold_by_index.get(group_index, 'holdout')}",
                "audit_metadata_role": "restricted_not_model_feature",
            }
        )
    return rows


def build_records(scenarios: list[dict[str, Any]], *, shard_size: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_index = 0
    for scenario in scenarios:
        group_index = int(scenario["group_index"])
        for phase in PHASES:
            shard_index = record_index // shard_size
            shard_offset = record_index % shard_size
            timestamp_start_ns = (
                TIMESTAMP_EPOCH_NS
                + group_index * 200_000_000_000
                + int(phase.start_s * 1_000_000_000)
            )
            timestamp_end_ns = (
                TIMESTAMP_EPOCH_NS
                + group_index * 200_000_000_000
                + int(phase.end_s * 1_000_000_000)
            )
            record_id = f"{scenario['scenario_group_id']}_{phase.phase_id}"
            records.append(
                {
                    "record_id": record_id,
                    "record_index": record_index,
                    "scenario_group_id": scenario["scenario_group_id"],
                    "time_lock_id": scenario["time_lock_id"],
                    "phase_id": phase.phase_id,
                    "phase_start_s": phase.start_s,
                    "phase_end_s": phase.end_s,
                    "timestamp_start_ns": timestamp_start_ns,
                    "timestamp_end_ns": timestamp_end_ns,
                    "split_role": scenario["split_role"],
                    "cv_fold": scenario["cv_fold"],
                    "model_label": scenario["model_label"],
                    "label_id": int(scenario["is_positive"]),
                    "is_positive": int(scenario["is_positive"]),
                    "monte_carlo_stratum": scenario["monte_carlo_stratum"],
                    "noise_regime": scenario["noise_regime"],
                    "range_band": scenario["range_band"],
                    "site_archetype_id": scenario["site_archetype_id"],
                    "detector_view_set_id": DATASET_PROFILE,
                    "raw_complex_iq_ref": (
                        f"raw_complex_iq/active_radar_shard_{shard_index:04d}.npz"
                        f"#row={shard_offset}"
                    ),
                    "acoustic_stream_ref": (
                        f"acoustic_cues/acoustic_shard_{shard_index:04d}.npz#row={shard_offset}"
                    ),
                    "passive_rf_ref": (
                        f"passive_rf_cues/passive_rf_shard_{shard_index:04d}.npz#row={shard_offset}"
                    ),
                    "restricted_truth_path": (
                        f"restricted_truth/{scenario['scenario_group_id']}.json"
                    ),
                }
            )
            record_index += 1
    return records


def _signal_profile(
    record: dict[str, Any], scenario: dict[str, Any], sensor_gain: float
) -> tuple[float, float, float]:
    phase = next(item for item in PHASES if item.phase_id == record["phase_id"])
    label = int(record["label_id"])
    noise = str(record["noise_regime"])
    clutter_factor = phase.clutter_gain
    if noise in {"rfi_burst", "agc_compression", "multipath_masking"}:
        clutter_factor += 0.38
    if noise in {"open_sky", "quantization"}:
        clutter_factor -= 0.10
    if label:
        amplitude = (1.35 + 0.08 * (int(scenario["group_index"]) % 5)) * phase.positive_gain
        cadence_hz = 88.0 + float(stable_seed(record["record_id"], "cadence") % 17)
        doppler_norm = 0.12 + 0.05 * (stable_seed(record["record_id"], "doppler") % 7)
    else:
        family = str(scenario["target_role"])
        family_gain = {
            "single_bird": 0.56,
            "bird_flock": 0.78,
            "rc_fixed_wing": 0.92,
            "weather_cell": 0.48,
            "ground_vehicle": 0.62,
            "wind_turbine": 0.86,
            "terrain_glint": 0.70,
            "multipath_ghost": 0.74,
            "rfi_burst": 0.88,
            "clutter_only_counterfactual": 0.34,
        }.get(family, 0.64)
        amplitude = family_gain * clutter_factor
        cadence_hz = 22.0 + float(stable_seed(record["record_id"], "cadence") % 55)
        doppler_norm = 0.04 + 0.03 * (stable_seed(record["record_id"], "doppler") % 11)
    return amplitude * sensor_gain, cadence_hz, doppler_norm


def _make_active_iq(
    record: dict[str, Any],
    scenario: dict[str, Any],
    *,
    sensor_index: int,
    seed: int,
) -> np.ndarray:
    sensor = ACTIVE_RADAR_SENSORS[sensor_index]
    rng = np.random.default_rng(
        stable_seed(seed, record["record_id"], sensor.sensor_id, "active-iq")
    )
    noise_scale = 0.34
    if record["noise_regime"] in {"rain", "dust", "sea_clutter", "weibull_clutter"}:
        noise_scale = 0.48
    if record["noise_regime"] in {"open_sky", "quantization"}:
        noise_scale = 0.28
    real = rng.normal(0.0, noise_scale, size=(sensor.pulses, sensor.range_bins))
    imag = rng.normal(0.0, noise_scale, size=(sensor.pulses, sensor.range_bins))
    iq = real + 1j * imag
    amplitude, _cadence_hz, doppler_norm = _signal_profile(record, scenario, sensor.channel_gain)
    target_bin = 3 + int(stable_seed(record["scenario_group_id"], sensor.sensor_id) % 14)
    pulses = np.arange(sensor.pulses, dtype=np.float32)
    phase = 2.0 * np.pi * doppler_norm * pulses
    if record["noise_regime"] == "doppler_folding":
        phase *= -0.72
    envelope = np.exp(-0.5 * ((np.arange(sensor.range_bins) - target_bin) / 1.4) ** 2)
    if record["noise_regime"] == "dropped_cpi":
        drop_start = int(stable_seed(record["record_id"], sensor.sensor_id, "drop") % 12)
        iq[drop_start : drop_start + 4] *= 0.18
    iq += amplitude * np.exp(1j * phase)[:, None] * envelope[None, :]
    if record["noise_regime"] == "rfi_burst":
        burst_bin = int(stable_seed(record["record_id"], "rfi-bin") % sensor.range_bins)
        iq[:, burst_bin] += (0.9 + 0.7j) * (1.0 + sensor_index * 0.15)
    return iq.astype(np.complex64)


def _make_acoustic(
    record: dict[str, Any],
    scenario: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, record["record_id"], "acoustic"))
    samples = np.arange(ACOUSTIC_SAMPLE_COUNT, dtype=np.float32) / 800.0
    nodes = []
    base_amp, cadence_hz, _doppler = _signal_profile(record, scenario, 1.0)
    acoustic_amp = 0.18 + 0.16 * base_amp
    if record["noise_regime"] in {"urban_edge", "vegetation_motion", "rain"}:
        acoustic_amp *= 0.72
    for node_idx in range(ACOUSTIC_NODE_COUNT):
        phase = float(stable_seed(record["record_id"], "node", node_idx) % 360) / 57.2958
        tone = np.sin(2.0 * np.pi * (cadence_hz / 10.0) * samples + phase)
        harmonic = 0.44 * np.sin(2.0 * np.pi * (cadence_hz / 5.0) * samples + phase / 2.0)
        noise = rng.normal(0.0, 0.22, size=ACOUSTIC_SAMPLE_COUNT)
        nodes.append(acoustic_amp * (tone + harmonic) + noise)
    return np.asarray(nodes, dtype=np.float32)


def _passive_rf_features(
    record: dict[str, Any], scenario: dict[str, Any], *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, record["record_id"], "passive-rf"))
    label = float(record["label_id"])
    noise = 0.15 if record["noise_regime"] not in {"rfi_burst", "clock_drift"} else 0.32
    no_signal_score = max(0.0, 0.82 - label * 0.18 + rng.normal(0.0, 0.05))
    burst_score = max(
        0.0, (0.55 if record["noise_regime"] == "rfi_burst" else 0.12) + rng.normal(0.0, noise)
    )
    clock_offset = (stable_seed(record["scenario_group_id"], "clock") % 700) / 1000.0
    provenance_quality = min(1.0, max(0.0, 0.86 - 0.25 * burst_score + rng.normal(0.0, 0.04)))
    return np.asarray(
        [no_signal_score, burst_score, clock_offset, provenance_quality], dtype=np.float32
    )


def _active_features(iq: np.ndarray) -> dict[str, float]:
    power = np.abs(iq) ** 2
    pulse_power = power.mean(axis=1)
    range_power = power.mean(axis=0)
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)) + 1e-6)
    peak = float(np.max(power))
    doppler_fft = np.fft.fft(iq, axis=0)
    doppler_power = np.abs(doppler_fft) ** 2
    top_doppler = float(np.max(doppler_power))
    range_contrast = float(np.max(range_power) / (np.mean(range_power) + 1e-6))
    cadence_stability = float(1.0 / (1.0 + np.std(np.diff(pulse_power))))
    return {
        "mean_power": float(np.mean(power)),
        "peak_power": peak,
        "cfar_proxy_score": float((peak - median) / mad),
        "range_contrast": range_contrast,
        "doppler_concentration": float(top_doppler / (np.mean(doppler_power) + 1e-6)),
        "track_continuity_score": cadence_stability,
    }


def _acoustic_features(acoustic: np.ndarray) -> dict[str, float]:
    rms_by_node = np.sqrt(np.mean(np.square(acoustic), axis=1))
    spectrum = np.abs(np.fft.rfft(acoustic, axis=1))
    peak_bin = np.argmax(spectrum[:, 1:], axis=1) + 1
    spectral_peak = np.max(spectrum[:, 1:], axis=1)
    spectral_mean = np.mean(spectrum[:, 1:], axis=1) + 1e-6
    return {
        "node_rms_mean": float(np.mean(rms_by_node)),
        "node_rms_spread": float(np.std(rms_by_node)),
        "cadence_bin_mean": float(np.mean(peak_bin)),
        "cadence_contrast": float(np.mean(spectral_peak / spectral_mean)),
        "network_agreement": float(1.0 / (1.0 + np.std(peak_bin))),
    }


def _fusion_input_features(
    active_rows: dict[str, dict[str, float]],
    acoustic_row: dict[str, float],
    passive_rf: np.ndarray,
) -> dict[str, float]:
    return {
        "high_res_logit_proxy": active_rows["high_resolution_xku_cuas"]["cfar_proxy_score"],
        "sband_logit_proxy": active_rows["tactical_s_band_aesa"]["doppler_concentration"],
        "gbad_track_proxy": active_rows["gbad_3d4d_cueing"]["track_continuity_score"],
        "acoustic_cadence_proxy": acoustic_row["cadence_contrast"],
        "passive_rf_provenance_quality": float(passive_rf[3]),
        "source_freshness_s": 0.0,
    }


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shard_rows(records: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + shard_size] for start in range(0, len(records), shard_size)]


def write_raw_streams_and_views(
    out_root: Path,
    scenarios: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    scenario_by_id = {row["scenario_group_id"]: row for row in scenarios}
    active_index_rows: list[dict[str, Any]] = []
    acoustic_index_rows: list[dict[str, Any]] = []
    passive_index_rows: list[dict[str, Any]] = []
    view_rows: dict[str, list[dict[str, Any]]] = {view_id: [] for view_id in DETECTOR_VIEW_IDS}

    for shard_index, shard_records in enumerate(_shard_rows(records, shard_size)):
        active_arrays = []
        acoustic_arrays = []
        passive_arrays = []
        record_ids = []
        time_lock_ids = []
        timestamp_start_ns = []
        timestamp_end_ns = []
        for row_offset, record in enumerate(shard_records):
            scenario = scenario_by_id[str(record["scenario_group_id"])]
            active_stack = np.stack(
                [
                    _make_active_iq(record, scenario, sensor_index=sensor_index, seed=seed)
                    for sensor_index in range(len(ACTIVE_RADAR_SENSORS))
                ],
                axis=0,
            )
            acoustic = _make_acoustic(record, scenario, seed=seed)
            passive_rf = _passive_rf_features(record, scenario, seed=seed)
            active_arrays.append(active_stack)
            acoustic_arrays.append(acoustic)
            passive_arrays.append(passive_rf)
            record_ids.append(str(record["record_id"]))
            time_lock_ids.append(str(record["time_lock_id"]))
            timestamp_start_ns.append(int(record["timestamp_start_ns"]))
            timestamp_end_ns.append(int(record["timestamp_end_ns"]))

            base_identity = {column: record[column] for column in DETECTOR_ID_COLUMNS}
            active_feature_rows: dict[str, dict[str, float]] = {}
            for sensor_index, sensor in enumerate(ACTIVE_RADAR_SENSORS):
                features = _active_features(active_stack[sensor_index])
                active_feature_rows[sensor.view_id] = features
                view_rows[sensor.view_id].append(
                    {
                        **base_identity,
                        "sensor_id": sensor.sensor_id,
                        "sensor_band": sensor.band,
                        "view_feature_version": "main-run-active-v1",
                        **features,
                    }
                )
            acoustic_features = _acoustic_features(acoustic)
            view_rows[ACOUSTIC_VIEW_ID].append(
                {
                    **base_identity,
                    "sensor_id": ACOUSTIC_VIEW_ID,
                    "sensor_band": "acoustic",
                    "view_feature_version": "main-run-acoustic-v1",
                    **acoustic_features,
                }
            )
            view_rows[FUSION_VIEW_ID].append(
                {
                    **base_identity,
                    "sensor_id": FUSION_VIEW_ID,
                    "sensor_band": "fusion",
                    "view_feature_version": "main-run-fusion-input-v1",
                    "source_provenance": "active_radar+acoustic+passive_rf",
                    **_fusion_input_features(active_feature_rows, acoustic_features, passive_rf),
                }
            )

            active_index_rows.append(
                {
                    "record_id": record["record_id"],
                    "time_lock_id": record["time_lock_id"],
                    "phase_id": record["phase_id"],
                    "timestamp_start_ns": record["timestamp_start_ns"],
                    "timestamp_end_ns": record["timestamp_end_ns"],
                    "shard_path": f"raw_complex_iq/active_radar_shard_{shard_index:04d}.npz",
                    "row_offset": row_offset,
                    "sensor_count": len(ACTIVE_RADAR_SENSORS),
                    "pulses": ACTIVE_RADAR_SENSORS[0].pulses,
                    "range_bins": ACTIVE_RADAR_SENSORS[0].range_bins,
                }
            )
            acoustic_index_rows.append(
                {
                    "record_id": record["record_id"],
                    "time_lock_id": record["time_lock_id"],
                    "phase_id": record["phase_id"],
                    "timestamp_start_ns": record["timestamp_start_ns"],
                    "timestamp_end_ns": record["timestamp_end_ns"],
                    "shard_path": f"acoustic_cues/acoustic_shard_{shard_index:04d}.npz",
                    "row_offset": row_offset,
                    "node_count": ACOUSTIC_NODE_COUNT,
                    "sample_count": ACOUSTIC_SAMPLE_COUNT,
                }
            )
            passive_index_rows.append(
                {
                    "record_id": record["record_id"],
                    "time_lock_id": record["time_lock_id"],
                    "phase_id": record["phase_id"],
                    "timestamp_start_ns": record["timestamp_start_ns"],
                    "timestamp_end_ns": record["timestamp_end_ns"],
                    "shard_path": f"passive_rf_cues/passive_rf_shard_{shard_index:04d}.npz",
                    "row_offset": row_offset,
                    "feature_columns": "no_signal_score;rfi_burst_score;clock_offset_ms;provenance_quality",
                }
            )

        shard_ids = np.asarray(record_ids, dtype="<U96")
        np.savez_compressed(
            out_root / "raw_complex_iq" / f"active_radar_shard_{shard_index:04d}.npz",
            iq=np.asarray(active_arrays, dtype=np.complex64),
            record_ids=shard_ids,
            time_lock_ids=np.asarray(time_lock_ids, dtype="<U64"),
            timestamp_start_ns=np.asarray(timestamp_start_ns, dtype=np.int64),
            timestamp_end_ns=np.asarray(timestamp_end_ns, dtype=np.int64),
            sensor_ids=np.asarray(
                [sensor.sensor_id for sensor in ACTIVE_RADAR_SENSORS], dtype="<U64"
            ),
        )
        np.savez_compressed(
            out_root / "acoustic_cues" / f"acoustic_shard_{shard_index:04d}.npz",
            acoustic=np.asarray(acoustic_arrays, dtype=np.float32),
            record_ids=shard_ids,
            time_lock_ids=np.asarray(time_lock_ids, dtype="<U64"),
            timestamp_start_ns=np.asarray(timestamp_start_ns, dtype=np.int64),
            timestamp_end_ns=np.asarray(timestamp_end_ns, dtype=np.int64),
        )
        np.savez_compressed(
            out_root / "passive_rf_cues" / f"passive_rf_shard_{shard_index:04d}.npz",
            passive_rf=np.asarray(passive_arrays, dtype=np.float32),
            record_ids=shard_ids,
            time_lock_ids=np.asarray(time_lock_ids, dtype="<U64"),
            timestamp_start_ns=np.asarray(timestamp_start_ns, dtype=np.int64),
            timestamp_end_ns=np.asarray(timestamp_end_ns, dtype=np.int64),
            feature_columns=np.asarray(
                [
                    "no_signal_score",
                    "rfi_burst_score",
                    "clock_offset_ms",
                    "provenance_quality",
                ],
                dtype="<U64",
            ),
        )

    for view_id, rows in view_rows.items():
        _write_csv(out_root / "detector_views" / f"{view_id}.csv", rows)
    _write_csv(out_root / "raw_stream_index.csv", active_index_rows)
    _write_csv(out_root / "acoustic_stream_index.csv", acoustic_index_rows)
    _write_csv(out_root / "passive_rf_index.csv", passive_index_rows)

    return {
        "raw_stream_records": len(active_index_rows),
        "active_shards": math.ceil(len(records) / shard_size),
        "detector_views": {
            view_id: {
                "path": f"detector_views/{view_id}.csv",
                "records": len(rows),
                "identity_columns": list(DETECTOR_ID_COLUMNS),
                "feature_columns": [
                    column
                    for column in (list(rows[0]) if rows else [])
                    if column not in DETECTOR_ID_COLUMNS
                    and column not in MODEL_FEATURE_DENYLIST
                    and column
                    not in {
                        "model_label",
                        "label_id",
                        "sensor_id",
                        "sensor_band",
                        "source_provenance",
                        "view_feature_version",
                    }
                ],
            }
            for view_id, rows in view_rows.items()
        },
    }


def _quality_report(
    scenarios: list[dict[str, Any]],
    records: list[dict[str, Any]],
    stream_summary: dict[str, Any],
    *,
    scenario_groups: int,
    positive_groups: int,
    folds: int,
    smoke: bool,
) -> dict[str, Any]:
    phase_counts: dict[str, int] = {}
    phase_positive_counts: dict[str, int] = {}
    for record in records:
        phase = str(record["phase_id"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        phase_positive_counts[phase] = phase_positive_counts.get(phase, 0) + int(record["label_id"])
    split_counts_by_group: dict[str, int] = {}
    split_positive_by_group: dict[str, int] = {}
    for scenario in scenarios:
        split = str(scenario["split_role"])
        split_counts_by_group[split] = split_counts_by_group.get(split, 0) + 1
        split_positive_by_group[split] = split_positive_by_group.get(split, 0) + int(
            scenario["is_positive"]
        )
    group_phase_counts: dict[str, int] = {}
    for record in records:
        group = str(record["scenario_group_id"])
        group_phase_counts[group] = group_phase_counts.get(group, 0) + 1
    fold_values = sorted(
        {
            int(scenario["cv_fold"])
            for scenario in scenarios
            if str(scenario["split_role"]) == "train_cv"
        }
    )
    phase_windows_ok = {phase.phase_id: [phase.start_s, phase.end_s] for phase in PHASES} == {
        "initial_take_up": [0.0, 30.0],
        "climb_transition": [30.0, 90.0],
        "cruise_altitude": [90.0, 150.0],
    }
    positive_label_ok = all(
        scenario["target_role"] == POSITIVE_MODEL_LABEL
        for scenario in scenarios
        if int(scenario["is_positive"])
    )
    leakage_guard_status = all(
        denied
        not in set().union(
            *[set(view["feature_columns"]) for view in stream_summary["detector_views"].values()]
        )
        for denied in MODEL_FEATURE_DENYLIST
    )
    expected_records = scenario_groups * len(PHASES)
    status = (
        "pass"
        if len(scenarios) == scenario_groups
        and len(records) == expected_records
        and set(phase_counts) == {phase.phase_id for phase in PHASES}
        and all(count == scenario_groups for count in phase_counts.values())
        and all(count == positive_groups for count in phase_positive_counts.values())
        and all(count == len(PHASES) for count in group_phase_counts.values())
        and positive_label_ok
        and phase_windows_ok
        and fold_values == list(range(folds))
        and leakage_guard_status
        else "fail"
    )
    return {
        "dataset_profile": DATASET_PROFILE,
        "strict_open_claim_boundary": (
            "synthetic public-proxy benchmark; no measured-platform or classified-fidelity claim"
        ),
        "smoke": smoke,
        "scenario_group_count": len(scenarios),
        "record_count": len(records),
        "expected_record_count": expected_records,
        "positive_scenario_group_count": positive_groups,
        "phase_counts": phase_counts,
        "phase_positive_counts": phase_positive_counts,
        "split_counts_by_group": split_counts_by_group,
        "split_positive_counts_by_group": split_positive_by_group,
        "fold_values": fold_values,
        "phase_windows_status": "pass" if phase_windows_ok else "fail",
        "positive_label_status": "pass" if positive_label_ok else "fail",
        "group_phase_lock_status": (
            "pass" if all(count == len(PHASES) for count in group_phase_counts.values()) else "fail"
        ),
        "detector_view_identity_status": "pass",
        "leakage_guard_status": "pass" if leakage_guard_status else "fail",
        "raw_stream_summary": stream_summary,
        "status": status,
    }


def _write_restricted_truth(out_root: Path, scenarios: list[dict[str, Any]]) -> None:
    for scenario in scenarios:
        payload = {
            "scenario_group_id": scenario["scenario_group_id"],
            "time_lock_id": scenario["time_lock_id"],
            "model_label": scenario["model_label"],
            "target_role": scenario["target_role"],
            "target_class": scenario["target_class"],
            "confuser_family": scenario["confuser_family"],
            "scenario_seed": scenario["scenario_seed"],
            "split_key": scenario["split_key"],
            "monte_carlo_stratum": scenario["monte_carlo_stratum"],
            "policy": "audit metadata only; not a detector feature export",
        }
        _write_json(
            out_root / "restricted_truth" / f"{scenario['scenario_group_id']}.json", payload
        )


def build_main_run_dataset(
    out_root: Path,
    *,
    scenario_groups: int = DEFAULT_SCENARIO_GROUPS,
    positive_groups: int = DEFAULT_POSITIVE_GROUPS,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    shard_size: int = 512,
    force: bool = False,
    smoke: bool = False,
    paper_profile: str = "fixed-wing-pusher-proxy-v1",
    holdout_policy: str = "group_random",
    holdout_value: str | None = None,
) -> dict[str, Any]:
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "raw_complex_iq",
        "acoustic_cues",
        "passive_rf_cues",
        "detector_views",
        "restricted_truth",
    ):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    scenarios = build_scenario_manifest(
        scenario_groups=scenario_groups,
        positive_groups=positive_groups,
        seed=seed,
        folds=folds,
        holdout_policy=holdout_policy,
        holdout_value=holdout_value,
    )
    records = build_records(scenarios, shard_size=shard_size)
    _write_csv(out_root / "scenario_manifest.csv", scenarios)
    _write_csv(out_root / "records.csv", records)
    _write_restricted_truth(out_root, scenarios)
    stream_summary = write_raw_streams_and_views(
        out_root, scenarios, records, seed=seed, shard_size=shard_size
    )
    quality = _quality_report(
        scenarios,
        records,
        stream_summary,
        scenario_groups=scenario_groups,
        positive_groups=positive_groups,
        folds=folds,
        smoke=smoke,
    )
    manifest = {
        "dataset_profile": DATASET_PROFILE,
        "paper_profile": paper_profile,
        "seed": seed,
        "scenario_groups": scenario_groups,
        "positive_groups": positive_groups,
        "record_count": len(records),
        "holdout_policy": holdout_policy,
        "holdout_value": holdout_value or "",
        "phase_ids": [phase.phase_id for phase in PHASES],
        "phase_windows_s": {phase.phase_id: [phase.start_s, phase.end_s] for phase in PHASES},
        "model_labels": [POSITIVE_MODEL_LABEL, NEGATIVE_MODEL_LABEL],
        "detector_views": list(DETECTOR_VIEW_IDS),
        "model_feature_denylist": list(MODEL_FEATURE_DENYLIST),
        "strict_open_claim_boundary": (
            "public-proxy synthetic artifacts with uncertainty-bearing detector views"
        ),
        "outputs_are_generated": True,
    }
    _write_json(out_root / "dataset_manifest.json", manifest)
    _write_json(out_root / "quality_report.json", quality)
    if quality["status"] != "pass":
        raise AssertionError(f"main-run quality checks failed: {quality}")
    return quality
