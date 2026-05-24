"""Scenario builders, audit probes, and operational metrics for the v1 benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_training_config import (
    FRAME_INDEX,
    NEGATIVE_CONTROL_AUC_GATE,
    PHASE_SPECS,
    TARGET_MASKED_AUC_GATE,
)
from generate_ml_training_aggregation import (
    denylist_violations,
    operational_metrics,
)


def probe_auc(X: np.ndarray, y: np.ndarray, split: np.ndarray) -> float:
    train = split == "train"
    test = split == "test"
    if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
        return float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=136),
    )
    model.fit(X[train], y[train])
    return float(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))


def one_hot_frame(records: pd.DataFrame, categorical: list[str], numeric: list[str]) -> np.ndarray:
    parts = []
    if categorical:
        parts.append(pd.get_dummies(records[categorical].astype(str), dtype=np.float32).to_numpy(np.float32))
    if numeric:
        parts.append(records[numeric].astype(np.float32).to_numpy(np.float32))
    if not parts:
        return np.zeros((len(records), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def phase_specs(max_time_s: float) -> list[Any]:
    if max_time_s < 120.0:
        raise ValueError("current max-time-s must be at least 120 seconds so cruise_altitude exists")
    from ml_training_config import PhaseSpec
    return [
        PHASE_SPECS[0],
        PHASE_SPECS[1],
        PhaseSpec(PHASE_SPECS[2].phase_id, 90.0, max_time_s, PHASE_SPECS[2].radar_meaning),
    ]


def split_for_group(group_index: int) -> str:
    mod = group_index % 20
    if mod < 3:
        return "test"
    if mod < 6:
        return "validation"
    return "train"


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
        categorical=[
            "phase_id",
            "site_archetype_id",
            "sensor_archetype_id",
            "clutter_regime",
            "target_aspect",
            "interference",
            "range_bin",
        ],
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
