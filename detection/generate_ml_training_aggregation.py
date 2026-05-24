"""Frame aggregation, denylist validation, operational metrics, and audit helpers.

Imports from generate_ml_training_types, generate_ml_training_frames only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from generate_ml_training_types import (
    FRAME_COLUMNS,
    FRAME_INDEX,
    FRAME_PERIOD_S,
    NEGATIVE_CONTROL_AUC_GATE,
    RESTRICTED_FEATURE_NAMES,
    TARGET_MASKED_AUC_GATE,
)
from generate_ml_training_frames import one_hot_frame, probe_auc


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
        categorical=["phase_id", "site_archetype_id", "sensor_archetype_id", "clutter_regime", "target_aspect", "interference", "range_bin"],
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
