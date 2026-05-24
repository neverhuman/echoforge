"""Feature aggregation, audit probes, and diagnostic reporting for the v2 benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from detection.detection_common_types import MODEL_DENYLIST_FRAME_COLUMNS
    from detection.ml_training_v2_config import (
        FRAME_COLUMNS,
        FRAME_INDEX,
        FRAME_COUNT,
        HELDOUT_CONFUSER_FAMILIES,
        MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ,
        MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ,
        NEGATIVE_CONTROL_AUC_GATE,
        SINGLE_FEATURE_AUC_GATE,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from detection_common_types import MODEL_DENYLIST_FRAME_COLUMNS
    from ml_training_v2_config import (
        FRAME_COLUMNS,
        FRAME_INDEX,
        FRAME_COUNT,
        HELDOUT_CONFUSER_FAMILIES,
        MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ,
        MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ,
        NEGATIVE_CONTROL_AUC_GATE,
        SINGLE_FEATURE_AUC_GATE,
    )


def aggregate_frame_features(frames: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    feature_values = []
    feature_names = []
    for col_name in FRAME_COLUMNS:
        if col_name == "time_s" or col_name in MODEL_DENYLIST_FRAME_COLUMNS:
            continue
        values = frames[:, :, FRAME_INDEX[col_name]]
        stats = {
            f"{col_name}_mean": values.mean(axis=1),
            f"{col_name}_std": values.std(axis=1),
            f"{col_name}_min": values.min(axis=1),
            f"{col_name}_max": values.max(axis=1),
            f"{col_name}_last": values[:, -1],
            f"{col_name}_q25": np.quantile(values, 0.25, axis=1),
            f"{col_name}_q75": np.quantile(values, 0.75, axis=1),
        }
        for name, value in stats.items():
            feature_names.append(name)
            feature_values.append(value.astype(np.float32))
    matrix = np.stack(feature_values, axis=1)
    return pd.DataFrame(matrix, columns=feature_names), feature_names


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    auc = float(roc_auc_score(labels, score))
    return max(auc, 1.0 - auc)


def overlap_coefficient(pos: np.ndarray, neg: np.ndarray, bins: int = 40) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    lo = float(min(np.min(pos), np.min(neg)))
    hi = float(max(np.max(pos), np.max(neg)))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return 1.0
    pos_hist, edges = np.histogram(pos, bins=bins, range=(lo, hi), density=True)
    neg_hist, _ = np.histogram(neg, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(pos_hist, neg_hist)) * width)


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
        parts.append(
            pd.get_dummies(records[categorical].astype(str), dtype=np.float32).to_numpy(np.float32)
        )
    if numeric:
        parts.append(records[numeric].astype(np.float32).to_numpy(np.float32))
    if not parts:
        return np.zeros((len(records), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def negative_control_audit(
    records: pd.DataFrame, feature_df: pd.DataFrame, labels: np.ndarray, split: np.ndarray
) -> dict[str, Any]:
    rng = np.random.default_rng(20260518)
    shuffled = labels.copy()
    for split_name in ["train", "validation", "test"]:
        mask = split == split_name
        shuffled[mask] = rng.permutation(shuffled[mask])
    allowed_metadata = one_hot_frame(
        records,
        categorical=[
            "stratum_id",
            "difficulty_bucket",
            "sensor_band",
            "range_bin",
            "clutter_regime",
            "target_aspect",
            "motion_pattern",
            "interference",
            "mixed_scene",
        ],
        numeric=["wave_index", "grazing_angle_deg", "scene_object_count"],
    )
    seed_hash = np.stack(
        [
            records["scenario_seed"].astype(np.uint64).to_numpy() % 997,
            records["object_seed"].astype(np.uint64).to_numpy() % 991,
            records["record_index"].astype(np.uint64).to_numpy() % 983,
        ],
        axis=1,
    ).astype(np.float32)
    row_index = records[["record_index"]].astype(np.float32).to_numpy()
    sensor_features = feature_df.to_numpy(np.float32)
    probes = {
        "allowed_metadata_only_auc": probe_auc(allowed_metadata, labels, split),
        "seed_hash_only_auc": probe_auc(seed_hash, labels, split),
        "row_index_only_auc": probe_auc(row_index, labels, split),
        "label_shuffled_sensor_feature_auc": probe_auc(sensor_features, shuffled, split),
    }
    probes["gate"] = NEGATIVE_CONTROL_AUC_GATE
    probes["status"] = (
        "pass"
        if all(
            (not math.isfinite(value)) or value <= NEGATIVE_CONTROL_AUC_GATE
            for key, value in probes.items()
            if key.endswith("_auc")
        )
        else "fail"
    )
    probes["policy"] = (
        "Negative-control probes must stay near chance; high AUC indicates metadata leakage, "
        "seed/split artifacts, row-order artifacts, or preprocessing leakage."
    )
    return probes


def micro_doppler_saturation_guard(frames: np.ndarray) -> dict[str, Any]:
    checks = []
    for column, clamp in [
        ("micro_doppler_peak_hz_proxy", MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ),
        ("micro_doppler_bandwidth_hz_proxy", MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ),
    ]:
        values = frames[:, :, FRAME_INDEX[column]].astype(np.float64, copy=False)
        median = float(np.median(values))
        checks.append(
            {
                "column": column,
                "configured_clamp": float(clamp),
                "median": median,
                "at_clamp_fraction": float(np.mean(values >= float(clamp) - 1e-6)),
                "status": "pass" if median < float(clamp) - 1e-6 else "fail",
            }
        )
    return {
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
        "policy": (
            "KTH-hardened public-proxy outputs must not place median "
            "micro-Doppler peak or bandwidth proxy values at the configured clamps."
        ),
    }


def write_diagnostics(
    out_root: Path, records: pd.DataFrame, frames: np.ndarray, feature_df: pd.DataFrame
) -> dict[str, Any]:
    labels = records["is_public_proxy_positive"].astype(bool).to_numpy(dtype=np.int64)
    split = records["split"].astype(str).to_numpy()
    controls = negative_control_audit(records, feature_df, labels, split)
    (out_root / "negative_control_audit.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True) + "\n"
    )
    saturation_guard = micro_doppler_saturation_guard(frames)
    (out_root / "micro_doppler_saturation_guard.json").write_text(
        json.dumps(saturation_guard, indent=2, sort_keys=True) + "\n"
    )
    feature_rows = []
    max_auc = 0.0
    for name in feature_df.columns:
        values = feature_df[name].to_numpy(np.float32)
        row = {
            "feature": name,
            "auc_abs_all": safe_auc(labels, values),
            "auc_abs_train": safe_auc(labels[split == "train"], values[split == "train"]),
            "auc_abs_validation": safe_auc(
                labels[split == "validation"], values[split == "validation"]
            ),
            "auc_abs_holdout": safe_auc(labels[split == "test"], values[split == "test"]),
            "positive_q10": float(np.quantile(values[labels == 1], 0.10)),
            "positive_q50": float(np.quantile(values[labels == 1], 0.50)),
            "positive_q90": float(np.quantile(values[labels == 1], 0.90)),
            "confuser_q10": float(np.quantile(values[labels == 0], 0.10)),
            "confuser_q50": float(np.quantile(values[labels == 0], 0.50)),
            "confuser_q90": float(np.quantile(values[labels == 0], 0.90)),
            "overlap_coefficient": overlap_coefficient(values[labels == 1], values[labels == 0]),
        }
        max_auc = max(max_auc, row["auc_abs_all"])
        feature_rows.append(row)
    auc_df = pd.DataFrame(feature_rows).sort_values("auc_abs_all", ascending=False)
    auc_df.to_csv(out_root / "single_feature_auc_audit.csv", index=False, float_format="%.6f")

    balance = (
        records.groupby(
            ["stratum_id", "difficulty_bucket", "split", "is_public_proxy_positive"], dropna=False
        )
        .size()
        .reset_index(name="count")
        .sort_values(["stratum_id", "split", "is_public_proxy_positive"])
    )
    balance.to_csv(out_root / "per_stratum_split_balance.csv", index=False)

    overlap = auc_df[
        [
            "feature",
            "positive_q10",
            "positive_q50",
            "positive_q90",
            "confuser_q10",
            "confuser_q50",
            "confuser_q90",
            "overlap_coefficient",
        ]
    ]
    overlap.to_csv(out_root / "distribution_overlap.csv", index=False, float_format="%.6f")

    metadata_leakage = []
    for col in [
        "class_id",
        "target_family",
        "is_public_proxy_positive",
        "is_hard_negative",
        "hard_negative_family",
        "confuser_family",
        "holdout_role",
    ]:
        unique_by_label = (
            records.groupby("is_public_proxy_positive")[col].nunique(dropna=False).to_dict()
        )
        metadata_leakage.append(
            {
                "column": col,
                "status": "restricted_metadata_not_available_to_consumers",
                "unique_values_by_label": {str(k): int(v) for k, v in unique_by_label.items()},
            }
        )
    frame_product_audit = {
        "max_abs_single_feature_auc": float(max_auc),
        "gate": SINGLE_FEATURE_AUC_GATE,
        "status": "pass" if max_auc <= SINGLE_FEATURE_AUC_GATE else "fail",
        "highest_features": auc_df.head(20).to_dict(orient="records"),
    }
    leakage_report = {
        "metadata_columns": metadata_leakage,
        "frame_products": frame_product_audit,
        "consumer_feature_policy": "Detection scripts consume frame tensor products and numeric aggregates only; label, class, family, split key, and holdout-role metadata are excluded from model features.",
    }
    (out_root / "label_leakage_audit.json").write_text(
        json.dumps(leakage_report, indent=2, sort_keys=True) + "\n"
    )

    cfar_margin = (
        frames[:, :, FRAME_INDEX["cfar_statistic"]] - frames[:, :, FRAME_INDEX["cfar_threshold"]]
    )
    baseline_score = (
        frames[:, :, FRAME_INDEX["snr_db"]].max(axis=1) * 0.30
        + frames[:, :, FRAME_INDEX["snr_db"]].mean(axis=1) * 0.15
        + cfar_margin.max(axis=1) * 0.25
        + frames[:, :, FRAME_INDEX["tbd_track_score"]].max(axis=1) * 0.30
    )
    baseline_auc = (
        float(roc_auc_score(labels, baseline_score))
        if np.unique(labels).size == 2
        else float("nan")
    )
    quality = {
        "benchmark_version": "ml-training-v2",
        "record_count": int(len(records)),
        "stratum_count": int(records["stratum_id"].nunique()),
        "frame_count_per_record": FRAME_COUNT,
        "single_feature_auc_gate": SINGLE_FEATURE_AUC_GATE,
        "single_feature_auc_max": float(max_auc),
        "single_feature_auc_gate_status": "pass" if max_auc <= SINGLE_FEATURE_AUC_GATE else "fail",
        "negative_control_auc_gate": NEGATIVE_CONTROL_AUC_GATE,
        "negative_control_status": controls["status"],
        "micro_doppler_saturation_status": saturation_guard["status"],
        "negative_control_summary": {
            key: value for key, value in controls.items() if key.endswith("_auc")
        },
        "baseline_cfar_tbd_auc_all_records": baseline_auc,
        "split_counts": {
            str(k): int(v) for k, v in records["split"].value_counts().sort_index().items()
        },
        "positive_rate_by_split": {
            str(k): float(v)
            for k, v in records.groupby("split")["is_public_proxy_positive"]
            .mean()
            .sort_index()
            .items()
        },
        "holdout_unseen_strata": sorted(
            records.loc[records["holdout_role"] != "seen", "stratum_id"].unique().tolist()
        ),
        "holdout_unseen_confuser_families": sorted(HELDOUT_CONFUSER_FAMILIES),
        "status": "pass"
        if max_auc <= SINGLE_FEATURE_AUC_GATE
        and controls["status"] == "pass"
        and saturation_guard["status"] == "pass"
        else "fail",
    }
    (out_root / "quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n"
    )
    if max_auc > SINGLE_FEATURE_AUC_GATE:
        raise AssertionError(
            f"single-feature AUC gate failed: {max_auc:.4f} > {SINGLE_FEATURE_AUC_GATE:.2f}"
        )
    if controls["status"] != "pass":
        raise AssertionError(f"negative-control audit failed: {controls}")
    if saturation_guard["status"] != "pass":
        raise AssertionError(f"micro-Doppler saturation guard failed: {saturation_guard}")
    return quality
