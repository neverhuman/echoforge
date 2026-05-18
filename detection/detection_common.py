"""Shared v2 benchmark loading, metrics, and report helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


DEFAULT_DATA_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
DEFAULT_OUT_ROOT = "outputs/detection"
FRAME_PERIOD_S = 0.5
SINGLE_FEATURE_AUC_GATE = 0.85
TRUTH_LIKE_FRAME_COLUMNS = {"altitude_m"}
REPORT_COLUMNS = [
    "method",
    "horizon_name",
    "horizon_s",
    "train_auc",
    "validation_auc",
    "holdout_auc",
    "baseline_auc",
    "lift_vs_baseline",
    "average_precision_holdout",
    "pd_at_1pct_pfa",
    "pfa_at_1pct_budget",
    "precision_at_1pct_pfa",
    "recall_at_1pct_pfa",
    "pfa_at_80pct_pd",
    "track_initiation_latency_s",
    "track_fragmentation_rate",
    "missed_track_rate",
    "false_track_rate",
    "easy_holdout_auc",
    "medium_holdout_auc",
    "hard_holdout_auc",
    "barely_visible_holdout_auc",
    "unseen_strata_holdout_auc",
    "unseen_confuser_holdout_auc",
    "n_train",
    "n_validation",
    "n_holdout",
    "positive_rate_holdout",
]


def horizon_label(horizon_s: int) -> str:
    labels = {5: "ultra_early_5s", 15: "mid_detection_15s", 45: "typical_detection_45s"}
    return labels.get(horizon_s, f"horizon_{horizon_s}s")


def parse_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def load_records(data_root: Path, seed: int, max_records: int | None) -> pd.DataFrame:
    records = pd.read_csv(data_root / "records.csv")
    splits = pd.read_csv(data_root / "split_manifest.csv", usecols=["record_id", "split"])
    records = records.drop(columns=["split"], errors="ignore").merge(splits, on="record_id", validate="one_to_one")
    records["label"] = records["is_public_proxy_positive"].astype(bool).astype(np.int64)
    if max_records is not None and max_records < len(records):
        rng = np.random.default_rng(seed)
        sampled = []
        remaining = max_records
        for split, fraction in [("train", 0.70), ("validation", 0.15), ("test", 0.15)]:
            group = records[records["split"] == split]
            take = remaining if split == "test" else int(round(max_records * fraction))
            if split != "test":
                remaining -= take
            take = max(1, min(take, len(group)))
            sampled.append(records.loc[np.sort(rng.choice(group.index.to_numpy(), size=take, replace=False))])
        records = pd.concat(sampled, axis=0).sort_values(["split", "record_id"]).reset_index(drop=True)
    return records.reset_index(drop=True)


def load_frame_store(data_root: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    payload = np.load(data_root / "frame_features.npz", allow_pickle=False)
    frames = payload["frames"].astype(np.float32, copy=False)
    record_ids = payload["record_ids"].astype(str).tolist()
    return frames, payload["frame_columns"].astype(str).tolist(), {rid: idx for idx, rid in enumerate(record_ids)}


def select_frames(data_root: Path, records: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frames, columns, index = load_frame_store(data_root)
    take = [index[str(rid)] for rid in records["record_id"].astype(str)]
    return frames[np.asarray(take, dtype=np.int64)], columns


def horizon_slice(frames: np.ndarray, columns: list[str], horizon_s: int) -> tuple[np.ndarray, float]:
    time_idx = columns.index("time_s")
    mask = frames[0, :, time_idx] <= float(horizon_s) + 1e-6
    sliced = frames[:, mask, :]
    max_consumed = float(sliced[:, :, time_idx].max())
    if max_consumed > float(horizon_s) + 1e-6:
        raise AssertionError(f"horizon leakage: consumed {max_consumed}s for {horizon_s}s")
    return sliced, max_consumed


def slope(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float32)
    x = np.arange(values.shape[1], dtype=np.float32)
    x -= x.mean()
    denom = float(np.dot(x, x))
    centered = values - values.mean(axis=1, keepdims=True)
    return (centered @ x / max(denom, 1e-6)).astype(np.float32)


def longest_true_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=np.float32)
    cur = np.zeros(mask.shape[0], dtype=np.float32)
    for idx in range(mask.shape[1]):
        cur = np.where(mask[:, idx], cur + 1.0, 0.0)
        out = np.maximum(out, cur)
    return out


def tabular_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sliced, max_consumed = horizon_slice(frames, columns, horizon_s)
    idx = {name: pos for pos, name in enumerate(columns)}
    feature_parts: list[np.ndarray] = []
    feature_parts.append(np.full((sliced.shape[0], 1), sliced.shape[1], dtype=np.float32))
    feature_parts.append(np.full((sliced.shape[0], 1), max_consumed, dtype=np.float32))

    cfar = sliced[:, :, idx["cfar_detected"]] > 0.5
    margin = sliced[:, :, idx["cfar_statistic"]] - sliced[:, :, idx["cfar_threshold"]]
    positive_margin = np.maximum(margin, 0.0)
    cfar_features = np.stack(
        [
            cfar.mean(axis=1),
            cfar.sum(axis=1),
            longest_true_run(cfar),
            (margin > 0.0).mean(axis=1),
            positive_margin.sum(axis=1),
            positive_margin.max(axis=1),
            margin[:, -1],
            slope(margin),
        ],
        axis=1,
    )
    feature_parts.append(cfar_features.astype(np.float32))

    if profile == "cfar":
        wanted = [
            "snr_db",
            "cfar_statistic",
            "cfar_threshold",
            "tbd_track_score",
            "local_noise_floor_db",
            "doppler_scr",
            "rfi_pressure",
            "dropout_fraction",
            "micro_doppler_energy",
            "range_time_energy",
            "doppler_time_energy",
            "range_doppler_time_energy",
            "normalized_snr",
        ]
        quantiles = []
    elif profile == "catboost":
        wanted = [col for col in columns if col not in {"time_s", "cfar_detected", *TRUTH_LIKE_FRAME_COLUMNS}]
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    else:
        wanted = [col for col in columns if col not in {"time_s", "cfar_detected", *TRUTH_LIKE_FRAME_COLUMNS}]
        quantiles = [0.10, 0.50, 0.90]

    for name in wanted:
        values = sliced[:, :, idx[name]].astype(np.float32)
        stats = [
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
            values[:, -1],
            slope(values),
        ]
        for q in quantiles:
            stats.append(np.quantile(values, q, axis=1).astype(np.float32))
        feature_parts.append(np.stack(stats, axis=1).astype(np.float32))

    X = np.concatenate(feature_parts, axis=1).astype(np.float32)
    B = baseline_matrix(sliced, columns)
    y = records["label"].to_numpy(np.int64)
    splits = records["split"].astype(str).to_numpy(dtype="<U16")
    return X, B, y, splits, max_consumed


def sequence_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    seq_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sliced, max_consumed = horizon_slice(frames, columns, horizon_s)
    idx = {name: pos for pos, name in enumerate(columns)}
    S = np.stack([sliced[:, :, idx[name]] for name in seq_columns], axis=2).astype(np.float32)
    cfar = sliced[:, :, idx["cfar_detected"]] > 0.5
    margin = sliced[:, :, idx["cfar_statistic"]] - sliced[:, :, idx["cfar_threshold"]]
    static = np.stack(
        [
            np.full(sliced.shape[0], sliced.shape[1], dtype=np.float32),
            np.full(sliced.shape[0], max_consumed, dtype=np.float32),
            cfar.mean(axis=1),
            cfar.sum(axis=1),
            longest_true_run(cfar),
            (margin > 0.0).mean(axis=1),
            np.maximum(margin, 0.0).sum(axis=1),
            margin.max(axis=1),
            sliced[:, :, idx["tbd_track_score"]].max(axis=1),
            sliced[:, :, idx["micro_doppler_energy"]].mean(axis=1),
            sliced[:, :, idx["rfi_pressure"]].mean(axis=1),
            sliced[:, :, idx["dropout_fraction"]].mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    B = baseline_matrix(sliced, columns)
    y = records["label"].to_numpy(np.int64)
    splits = records["split"].astype(str).to_numpy(dtype="<U16")
    return S, static, B, y, splits, max_consumed


def transformer_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    token_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sliced, max_consumed = horizon_slice(frames, columns, horizon_s)
    idx = {name: pos for pos, name in enumerate(columns)}
    T = np.stack([sliced[:, :, idx[name]] for name in token_columns], axis=2).astype(np.float32)
    M = np.ones(T.shape[:2], dtype=bool)
    _, F, B, y, splits, _ = sequence_features(frames, columns, records, horizon_s, token_columns)
    return T, M, F, B, y, splits, max_consumed


def baseline_matrix(frames: np.ndarray, columns: list[str]) -> np.ndarray:
    idx = {name: pos for pos, name in enumerate(columns)}
    margin = frames[:, :, idx["cfar_statistic"]] - frames[:, :, idx["cfar_threshold"]]
    values = np.stack(
        [
            frames[:, :, idx["snr_db"]].max(axis=1),
            frames[:, :, idx["snr_db"]].mean(axis=1),
            margin.max(axis=1),
            margin.mean(axis=1),
            frames[:, :, idx["tbd_track_score"]].max(axis=1),
            frames[:, :, idx["tbd_track_score"]].mean(axis=1),
        ],
        axis=1,
    )
    return values.astype(np.float32)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def evaluate_scores(y: np.ndarray, splits: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "train_auc": safe_auc(y[splits == "train"], score[splits == "train"]),
        "validation_auc": safe_auc(y[splits == "validation"], score[splits == "validation"]),
        "holdout_auc": safe_auc(y[splits == "test"], score[splits == "test"]),
    }


def baseline_scores(B: np.ndarray, splits: np.ndarray) -> np.ndarray:
    train = splits == "train"
    mu = B[train].mean(axis=0)
    sigma = B[train].std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    weights = np.array([0.30, 0.15, 0.25, 0.10, 0.15, 0.05], dtype=np.float32)
    return ((B - mu) / sigma) @ weights


def counts(y: np.ndarray, splits: np.ndarray) -> dict[str, Any]:
    holdout = splits == "test"
    return {
        "n_train": int((splits == "train").sum()),
        "n_validation": int((splits == "validation").sum()),
        "n_holdout": int(holdout.sum()),
        "positive_rate_holdout": float(y[holdout].mean()) if holdout.any() else float("nan"),
    }


def operating_metrics(y: np.ndarray, splits: np.ndarray, score: np.ndarray) -> dict[str, float]:
    holdout = splits == "test"
    y_h = y[holdout]
    s_h = score[holdout]
    if np.unique(y_h).size < 2:
        return {
            "average_precision_holdout": float("nan"),
            "pd_at_1pct_pfa": float("nan"),
            "pfa_at_1pct_budget": float("nan"),
            "precision_at_1pct_pfa": float("nan"),
            "recall_at_1pct_pfa": float("nan"),
            "pfa_at_80pct_pd": float("nan"),
        }
    neg = s_h[y_h == 0]
    pos = s_h[y_h == 1]
    threshold = float(np.quantile(neg, 0.99))
    pred = s_h >= threshold
    tp = int(((pred == 1) & (y_h == 1)).sum())
    fp = int(((pred == 1) & (y_h == 0)).sum())
    fn = int(((pred == 0) & (y_h == 1)).sum())
    tn = int(((pred == 0) & (y_h == 0)).sum())
    pd_at_budget = tp / max(1, tp + fn)
    pfa_at_budget = fp / max(1, fp + tn)
    threshold_80pd = float(np.quantile(pos, 0.20))
    pfa_80 = float((neg >= threshold_80pd).mean())
    return {
        "average_precision_holdout": float(average_precision_score(y_h, s_h)),
        "pd_at_1pct_pfa": float(pd_at_budget),
        "pfa_at_1pct_budget": float(pfa_at_budget),
        "precision_at_1pct_pfa": float(tp / max(1, tp + fp)),
        "recall_at_1pct_pfa": float(pd_at_budget),
        "pfa_at_80pct_pd": pfa_80,
    }


def track_metrics(frames: np.ndarray, columns: list[str], y: np.ndarray, splits: np.ndarray) -> dict[str, float]:
    holdout = splits == "test"
    if not holdout.any():
        return {
            "track_initiation_latency_s": float("nan"),
            "track_fragmentation_rate": float("nan"),
            "missed_track_rate": float("nan"),
            "false_track_rate": float("nan"),
        }
    idx = {name: pos for pos, name in enumerate(columns)}
    margin = frames[:, :, idx["cfar_statistic"]] - frames[:, :, idx["cfar_threshold"]]
    tbd = frames[:, :, idx["tbd_track_score"]]
    detected = (margin > 0.0) & (tbd > np.quantile(tbd[splits == "train"], 0.62))
    detected_h = detected[holdout]
    y_h = y[holdout]
    times = frames[holdout, :, idx["time_s"]]
    pos_mask = y_h == 1
    neg_mask = y_h == 0
    first_latency = []
    missed = 0
    fragments = []
    for det, time_row in zip(detected_h[pos_mask], times[pos_mask]):
        if det.any():
            first_latency.append(float(time_row[np.argmax(det)]))
        else:
            missed += 1
        transitions = np.diff(np.r_[False, det, False].astype(np.int8))
        fragments.append(float((transitions == 1).sum()))
    false_track_rate = float(detected_h[neg_mask].any(axis=1).mean()) if neg_mask.any() else float("nan")
    return {
        "track_initiation_latency_s": float(np.mean(first_latency)) if first_latency else float("nan"),
        "track_fragmentation_rate": float(np.mean(fragments)) if fragments else float("nan"),
        "missed_track_rate": float(missed / max(1, int(pos_mask.sum()))),
        "false_track_rate": false_track_rate,
    }


def slice_metrics(records: pd.DataFrame, y: np.ndarray, splits: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {}
    test = splits == "test"
    for bucket, key in [
        ("easy", "easy_holdout_auc"),
        ("medium", "medium_holdout_auc"),
        ("hard", "hard_holdout_auc"),
        ("barely_visible", "barely_visible_holdout_auc"),
    ]:
        mask = test & (records["difficulty_bucket"].astype(str).to_numpy() == bucket)
        out[key] = safe_auc(y[mask], score[mask])
    unseen = test & (records["holdout_role"].astype(str).to_numpy() != "seen")
    out["unseen_strata_holdout_auc"] = safe_auc(y[unseen], score[unseen])
    holdout_role = records["holdout_role"].astype(str).to_numpy()
    confuser = records["confuser_family"].fillna("").astype(str).to_numpy()
    unseen_confuser = test & (
        (holdout_role == "unseen_stratum_confuser")
        | np.isin(confuser, ["kite", "balloon", "wind_turbine", "multipath_ghost", "terrain_glint"])
    )
    out["unseen_confuser_holdout_auc"] = safe_auc(y[unseen_confuser], score[unseen_confuser])
    return out


def write_auxiliary_reports(
    out_root: Path,
    method: str,
    horizon_name: str,
    records: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    score: np.ndarray,
) -> None:
    reports = out_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    test = splits == "test"
    per_stratum_rows = []
    for (stratum, bucket), idxs in records[test].groupby(["stratum_id", "difficulty_bucket"]).groups.items():
        indices = np.asarray(list(idxs), dtype=np.int64)
        per_stratum_rows.append(
            {
                "method": method,
                "horizon_name": horizon_name,
                "stratum_id": stratum,
                "difficulty_bucket": bucket,
                "count": int(indices.size),
                "positive_rate": float(y[indices].mean()) if indices.size else float("nan"),
                "auc": safe_auc(y[indices], score[indices]),
            }
        )
    per_stratum = pd.DataFrame(per_stratum_rows)
    append_csv(reports / "per_stratum_metrics.csv", per_stratum, ["method", "horizon_name", "stratum_id"])

    confusion_rows = []
    y_h = y[test]
    s_h = score[test]
    if np.unique(y_h).size == 2:
        threshold = float(np.quantile(s_h[y_h == 0], 0.99))
        pred = s_h >= threshold
        for family, idxs in records[test].fillna("").groupby("confuser_family").groups.items():
            indices = np.asarray(list(idxs), dtype=np.int64)
            if not indices.size:
                continue
            cm = confusion_matrix(y[indices], score[indices] >= threshold, labels=[0, 1])
            confusion_rows.append(
                {
                    "method": method,
                    "horizon_name": horizon_name,
                    "confuser_family": family or "positive_or_none",
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                    "threshold": threshold,
                }
            )
    append_csv(reports / "confusion_by_confuser.csv", pd.DataFrame(confusion_rows), ["method", "horizon_name", "confuser_family"])

    calibration_rows = []
    if np.unique(y_h).size == 2:
        bins = np.quantile(s_h, np.linspace(0.0, 1.0, 11))
        bins = np.unique(bins)
        if bins.size >= 2:
            ids = np.digitize(s_h, bins[1:-1], right=True)
            for bin_idx in range(bins.size - 1):
                mask = ids == bin_idx
                if mask.any():
                    calibration_rows.append(
                        {
                            "method": method,
                            "horizon_name": horizon_name,
                            "bin": int(bin_idx),
                            "count": int(mask.sum()),
                            "score_mean": float(s_h[mask].mean()),
                            "observed_positive_rate": float(y_h[mask].mean()),
                            "brier_score_holdout": float(brier_score_loss(y_h, np.clip(s_h, 0.0, 1.0))),
                        }
                    )
    append_csv(reports / "calibration_curves.csv", pd.DataFrame(calibration_rows), ["method", "horizon_name", "bin"])


def append_csv(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame()
    if new_rows.empty:
        return
    if not existing.empty:
        merged = existing.copy()
        for _, row in new_rows.iterrows():
            mask = np.ones(len(merged), dtype=bool)
            for key in keys:
                mask &= merged[key].astype(str).to_numpy() == str(row[key])
            merged = merged.loc[~mask]
        out = pd.concat([merged, new_rows], ignore_index=True)
    else:
        out = new_rows
    out = out.sort_values(keys)
    out.to_csv(path, index=False, float_format="%.6f")


def write_reports(row: dict[str, Any], out_root: Path) -> None:
    reports_dir = Path("detection/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_reports = out_root / "reports"
    out_reports.mkdir(parents=True, exist_ok=True)
    raw_path = out_reports / "raw_metrics.json"
    rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
    rows = [
        existing
        for existing in rows
        if not (existing["method"] == row["method"] and existing["horizon_name"] == row["horizon_name"])
    ]
    rows.append(row)
    rows.sort(key=lambda r: (int(r["horizon_s"]), str(r["method"])))
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    df = pd.DataFrame(rows)
    for col in REPORT_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df = df[REPORT_COLUMNS]
    df.to_csv(reports_dir / "auc_table.csv", index=False, float_format="%.6f")
    lines = [
        "# Detection AUC Table",
        "",
        "Synthetic public-proxy benchmark metrics only; not measured-object performance claims.",
        "",
        df.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    if not df.empty:
        best_rows = (
            df.sort_values(["horizon_s", "validation_auc", "holdout_auc"], ascending=[True, False, False])
            .groupby("horizon_s", as_index=False)
            .head(1)
        )
        lines.extend(["## Best Per Horizon", "", best_rows.to_markdown(index=False, floatfmt=".6f"), ""])
    (reports_dir / "auc_table.md").write_text("\n".join(lines))
