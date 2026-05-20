"""Shared helper functions for detection utilities."""

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
    roc_auc_score,
    roc_curve,
)

try:  # direct script entrypoints import from detection/ without a package root on sys.path
    from detection.detection_common_types import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUT_ROOT,
        TRUTH_LIKE_FRAME_COLUMNS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from detection/
    from detection_common_types import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUT_ROOT,
        TRUTH_LIKE_FRAME_COLUMNS,
    )


_TIME_COLUMN = "time_s"


def horizon_label(horizon_s: int) -> str:
    labels = {5: "ultra_early_5s", 15: "mid_detection_15s", 45: "typical_detection_45s"}
    return labels.get(horizon_s, f"horizon_{horizon_s}s")


def parse_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def load_records(
    data_root: Path = Path(DEFAULT_DATA_ROOT), seed: int = 42, max_records: int | None = None
) -> pd.DataFrame:
    records = pd.read_csv(data_root / "records.csv")
    splits = pd.read_csv(data_root / "split_manifest.csv", usecols=["record_id", "split"])
    records = records.drop(columns=["split"], errors="ignore").merge(
        splits, on="record_id", validate="one_to_one"
    )
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
            sampled.append(
                records.loc[np.sort(rng.choice(group.index.to_numpy(), size=take, replace=False))]
            )
        records = (
            pd.concat(sampled, axis=0).sort_values(["split", "record_id"]).reset_index(drop=True)
        )
    return records.reset_index(drop=True)


def load_frame_store(
    data_root: Path = Path(DEFAULT_DATA_ROOT),
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    payload = np.load(data_root / "frame_features.npz", allow_pickle=False)
    frames = payload["frames"].astype(np.float32, copy=False)
    record_ids = payload["record_ids"].astype(str).tolist()
    return (
        frames,
        payload["frame_columns"].astype(str).tolist(),
        {rid: idx for idx, rid in enumerate(record_ids)},
    )


def select_frames(
    data_root: Path = Path(DEFAULT_DATA_ROOT), records: pd.DataFrame | None = None
) -> tuple[np.ndarray, list[str]]:
    frames, columns, index = load_frame_store(data_root)
    if records is None:
        return frames, columns
    take = [index[str(rid)] for rid in records["record_id"].astype(str)]
    return frames[np.asarray(take, dtype=np.int64)], columns


def ensure_output_root(out_root: Path = Path(DEFAULT_OUT_ROOT)) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def horizon_name(horizon_s: int) -> str:
    if horizon_s == 5:
        return "ultra_early_5s"
    if horizon_s == 15:
        return "mid_detection_15s"
    if horizon_s == 45:
        return "typical_detection_45s"
    return f"horizon_{horizon_s}s"


def _time_index(columns: list[str]) -> int:
    try:
        return columns.index(_TIME_COLUMN)
    except ValueError as exc:  # pragma: no cover - guarded by dataset schema
        raise ValueError("frame store is missing time_s") from exc


def _prefix_mask(
    frames: np.ndarray, columns: list[str], horizon_s: int
) -> tuple[np.ndarray, float]:
    time_idx = _time_index(columns)
    times = frames[0, :, time_idx]
    mask = times <= (float(horizon_s) + 1e-6)
    if not mask.any():
        raise ValueError(f"no frames available at horizon {horizon_s}")
    return mask, float(times[mask].max())


def horizon_slice(
    frames: np.ndarray, columns: list[str], horizon_s: int
) -> tuple[np.ndarray, float]:
    mask, max_consumed = _prefix_mask(frames, columns, horizon_s)
    return frames[:, mask, :].copy(), max_consumed


def _series_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected 2D prefix values, got shape {values.shape}")
    return {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "min": values.min(axis=1),
        "max": values.max(axis=1),
        "last": values[:, -1],
        "q25": np.quantile(values, 0.25, axis=1),
        "q75": np.quantile(values, 0.75, axis=1),
        "trend": values[:, -1] - values[:, 0],
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _split_mask(splits: np.ndarray, split_name: str) -> np.ndarray:
    return np.asarray(splits, dtype=str) == split_name


def _holdout_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "average_precision_holdout": 0.0,
            "pd_at_1pct_pfa": float("nan"),
            "pfa_at_1pct_budget": float("nan"),
            "precision_at_1pct_pfa": float("nan"),
            "recall_at_1pct_pfa": float("nan"),
            "pfa_at_80pct_pd": float("nan"),
        }

    ap = float(average_precision_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(fpr <= 0.01)[0]
    idx = int(valid[-1]) if valid.size else 0
    threshold = float(thresholds[idx])
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    pfa = fp / max(1, fp + tn)
    pd = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    recall = pd
    valid_pd = np.where(tpr >= 0.8)[0]
    pfa_at_80 = float(fpr[int(valid_pd[0])]) if valid_pd.size else float("nan")
    return {
        "average_precision_holdout": ap,
        "pd_at_1pct_pfa": float(pd),
        "pfa_at_1pct_budget": float(pfa),
        "precision_at_1pct_pfa": float(precision),
        "recall_at_1pct_pfa": float(recall),
        "pfa_at_80pct_pd": pfa_at_80,
    }


def _profile_columns(profile: str) -> list[str]:
    if profile == "cfar":
        return [
            "snr_db",
            "cfar_statistic",
            "cfar_threshold",
            "cfar_detected",
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
    return [
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


def _augment_feature_bank(
    frames: np.ndarray, columns: list[str], horizon_s: int, profile: str
) -> pd.DataFrame:
    mask, _ = _prefix_mask(frames, columns, horizon_s)
    prefix = frames[:, mask, :].astype(np.float32, copy=False)
    idx = {name: columns.index(name) for name in columns}
    selected_numeric = [
        name
        for name in _profile_columns(profile)
        if name in idx and name not in TRUTH_LIKE_FRAME_COLUMNS
    ]

    derived = {
        "cfar_margin": prefix[:, :, idx["cfar_statistic"]] - prefix[:, :, idx["cfar_threshold"]],
        "snr_minus_noise": prefix[:, :, idx["snr_db"]] - prefix[:, :, idx["local_noise_floor_db"]],
        "track_evidence": np.maximum(
            prefix[:, :, idx["cfar_statistic"]] - prefix[:, :, idx["cfar_threshold"]], 0.0
        )
        + prefix[:, :, idx["tbd_track_score"]],
        "energy_ratio": prefix[:, :, idx["range_doppler_time_energy"]]
        / (1.0 + prefix[:, :, idx["range_time_energy"]] + prefix[:, :, idx["doppler_time_energy"]]),
    }

    values: list[np.ndarray] = []
    names: list[str] = []
    for column in selected_numeric:
        series = prefix[:, :, idx[column]]
        for stat_name, stat_values in _series_stats(series).items():
            values.append(np.nan_to_num(stat_values.astype(np.float32), nan=0.0))
            names.append(f"{column}_{stat_name}")
    for column, series in derived.items():
        for stat_name, stat_values in _series_stats(series).items():
            values.append(np.nan_to_num(stat_values.astype(np.float32), nan=0.0))
            names.append(f"{column}_{stat_name}")
    return pd.DataFrame(np.stack(values, axis=1), columns=names)


def _baseline_from_feature_frame(feature_frame: pd.DataFrame) -> np.ndarray:
    preferred = [
        "cfar_margin_max",
        "cfar_margin_mean",
        "tbd_track_score_max",
        "snr_db_mean",
        "normalized_snr_mean",
        "micro_doppler_energy_mean",
        "range_doppler_time_energy_mean",
        "track_evidence_max",
    ]
    weights = [0.35, 0.22, 0.18, 0.10, 0.08, 0.04, 0.02, 0.01]
    score = np.zeros(len(feature_frame), dtype=np.float32)
    scale = 0.0
    for weight, column in zip(weights, preferred):
        if column in feature_frame:
            score += weight * feature_frame[column].to_numpy(np.float32)
            scale += weight
    if scale > 0:
        score /= scale
    return score


def _render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def format_value(value: Any) -> str:
        if value is None or value == "":
            return ""
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return "nan"
            return f"{float(value):.6f}"
        return str(value)

    table = [[format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for row in table:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header = (
        "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    )
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        for row in table
    ]
    return "\n".join([header, separator, *body]) + "\n"


def evaluate_scores(labels: np.ndarray, splits: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    splits = np.asarray(splits, dtype=str)
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    holdout_mask = splits == "test"
    holdout_labels = labels[holdout_mask]
    holdout_scores = scores[holdout_mask]
    metrics = {
        "train_auc": _safe_auc(labels[train_mask], scores[train_mask]),
        "validation_auc": _safe_auc(labels[validation_mask], scores[validation_mask]),
        "holdout_auc": _safe_auc(holdout_labels, holdout_scores),
    }
    metrics.update(_holdout_metrics(holdout_labels, holdout_scores))
    return metrics


def baseline_scores(baseline: np.ndarray, splits: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(baseline, dtype=np.float32)


def tabular_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    profile: str = "cfar",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    feature_frame = _augment_feature_bank(frames, columns, horizon_s, profile)
    labels = records["is_public_proxy_positive"].astype(np.int64).to_numpy()
    splits = records["split"].astype(str).to_numpy()
    baseline = _baseline_from_feature_frame(feature_frame)
    _, max_consumed = _prefix_mask(frames, columns, horizon_s)
    return (
        feature_frame.to_numpy(np.float32),
        baseline.astype(np.float32),
        labels,
        splits,
        max_consumed,
    )


def sequence_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    selected_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mask, max_consumed = _prefix_mask(frames, columns, horizon_s)
    prefix = frames[:, mask, :]
    selected_idx = [
        columns.index(name)
        for name in selected_columns
        if name in columns and name not in TRUTH_LIKE_FRAME_COLUMNS
    ]
    sequence = prefix[:, :, selected_idx].astype(np.float32, copy=False)
    static_frame = _augment_feature_bank(frames, columns, horizon_s, "lightgbm")
    baseline = _baseline_from_feature_frame(static_frame)
    labels = records["is_public_proxy_positive"].astype(np.int64).to_numpy()
    splits = records["split"].astype(str).to_numpy()
    return (
        sequence,
        static_frame.to_numpy(np.float32),
        baseline.astype(np.float32),
        labels,
        splits,
        max_consumed,
    )


def transformer_features(
    frames: np.ndarray,
    columns: list[str],
    records: pd.DataFrame,
    horizon_s: int,
    selected_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sequence, static, baseline, labels, splits, max_consumed = sequence_features(
        frames, columns, records, horizon_s, selected_columns
    )
    mask = np.ones(sequence.shape[:2], dtype=bool)
    return sequence, mask, static, baseline, labels, splits, max_consumed


def counts(labels: np.ndarray, splits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits, dtype=str)
    holdout = splits == "test"
    return {
        "n_train": int((splits == "train").sum()),
        "n_validation": int((splits == "validation").sum()),
        "n_holdout": int(holdout.sum()),
        "positive_rate_holdout": float(labels[holdout].mean()) if holdout.any() else 0.0,
    }


def operating_metrics(
    labels: np.ndarray, splits: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    # The published AUC table currently does not surface extra operating metrics.
    return {}


def _prefix_operational_metrics(
    frames: np.ndarray, columns: list[str], labels: np.ndarray, splits: np.ndarray
) -> dict[str, float]:
    idx = {name: columns.index(name) for name in columns}
    time_values = frames[0, :, idx[_TIME_COLUMN]]
    train_mask = np.asarray(splits, dtype=str) == "train"
    train_tbd = frames[train_mask, :, idx["tbd_track_score"]]
    threshold = float(np.quantile(train_tbd, 0.64)) if train_tbd.size else 0.0
    cfar = frames[:, :, idx["cfar_detected"]] > 0.5
    tbd = frames[:, :, idx["tbd_track_score"]]
    confirmed = cfar & (tbd > threshold)
    pos_mask = np.asarray(labels, dtype=np.int64) > 0
    neg_mask = ~pos_mask

    def first_true_time(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float(time_values[int(np.argmax(mask))])

    initiations: list[float] = []
    fragments: list[float] = []
    missed = 0
    for row_confirmed in confirmed[pos_mask]:
        init = first_true_time(row_confirmed)
        if init is None:
            missed += 1
        else:
            initiations.append(init)
        transitions = np.diff(np.r_[False, row_confirmed, False].astype(np.int8))
        fragments.append(float((transitions == 1).sum()))
    false_track_rate = (
        float(confirmed[neg_mask].any(axis=1).mean()) if neg_mask.any() else float("nan")
    )
    return {
        "track_initiation_latency_s": float(np.mean(initiations)) if initiations else float("nan"),
        "track_fragmentation_rate": float(np.mean(fragments)) if fragments else float("nan"),
        "missed_track_rate": float(missed / max(1, int(pos_mask.sum()))),
        "false_track_rate": false_track_rate,
    }


def track_metrics(
    frames: np.ndarray, columns: list[str], labels: np.ndarray, splits: np.ndarray
) -> dict[str, float]:
    return _prefix_operational_metrics(frames, columns, labels, splits)


def _subset_auc(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 2:
        return 0.5
    return _safe_auc(np.asarray(labels)[mask], np.asarray(scores)[mask])


def slice_metrics(
    records: pd.DataFrame, labels: np.ndarray, splits: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    split_mask = np.asarray(splits, dtype=str) == "test"
    rows = records.reset_index(drop=True)
    buckets = {}
    for bucket in ["easy", "medium", "hard", "barely_visible"]:
        buckets[f"{bucket}_holdout_auc"] = _subset_auc(
            labels,
            scores,
            split_mask & (rows["difficulty_bucket"].astype(str).to_numpy() == bucket),
        )
    unseen_mask = split_mask & (rows["holdout_role"].astype(str).to_numpy() != "seen")
    if not unseen_mask.any():
        unseen_mask = split_mask
    unseen_strata = _subset_auc(labels, scores, unseen_mask)
    heldout_strata = {f"wave_{idx:02d}" for idx in range(45, 50)}
    confuser_mask = split_mask & rows["stratum_id"].astype(str).isin(heldout_strata).to_numpy()
    if not confuser_mask.any():
        confuser_mask = unseen_mask
    unseen_confuser = _subset_auc(labels, scores, confuser_mask)
    return {
        **buckets,
        "unseen_strata_holdout_auc": unseen_strata,
        "unseen_confuser_holdout_auc": unseen_confuser,
    }


def write_auxiliary_reports(
    out_root: Path,
    method: str,
    horizon_name: str,
    records: pd.DataFrame,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
) -> None:
    report_dir = ensure_output_root(out_root) / "reports" / method / horizon_name
    report_dir.mkdir(parents=True, exist_ok=True)
    holdout_mask = np.asarray(splits, dtype=str) == "test"
    score_rows = pd.DataFrame(
        {
            "record_id": records["record_id"].astype(str).to_numpy(),
            "split": np.asarray(splits, dtype=str),
            "label": np.asarray(labels, dtype=np.int64),
            "score": np.asarray(scores, dtype=np.float32),
        }
    )
    score_rows.to_csv(report_dir / "scores.csv", index=False, float_format="%.6f")
    summary = {
        "method": method,
        "horizon_name": horizon_name,
        "record_count": int(len(records)),
        "holdout_count": int(holdout_mask.sum()),
        "holdout_score_mean": float(np.mean(np.asarray(scores)[holdout_mask]))
        if holdout_mask.any()
        else float("nan"),
        "holdout_score_std": float(np.std(np.asarray(scores)[holdout_mask]))
        if holdout_mask.any()
        else float("nan"),
        "holdout_brier": float(
            brier_score_loss(np.asarray(labels)[holdout_mask], np.asarray(scores)[holdout_mask])
        )
        if holdout_mask.any() and np.unique(np.asarray(labels)[holdout_mask]).size > 1
        else float("nan"),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_reports(row: dict[str, Any], out_root: Path) -> None:
    report_root = ensure_output_root(out_root) / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "auc_table.csv"
    md_path = report_root / "auc_table.md"
    ordered_columns = [
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
    current = {column: row.get(column, "") for column in ordered_columns}
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[
            ~(
                existing["method"].astype(str).eq(str(current["method"]))
                & existing["horizon_name"].astype(str).eq(str(current["horizon_name"]))
            )
        ].copy()
        existing = pd.concat([existing, pd.DataFrame([current])], ignore_index=True)
    else:
        existing = pd.DataFrame([current])
    existing = existing.sort_values(
        ["horizon_s", "holdout_auc", "method"], ascending=[True, False, True]
    ).reset_index(drop=True)
    existing.to_csv(csv_path, index=False, float_format="%.6f")

    best_rows = (
        existing.sort_values(["horizon_s", "holdout_auc", "method"], ascending=[True, False, True])
        .groupby("horizon_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    md = [
        "# Detection AUC Table",
        "",
        "Synthetic public-proxy benchmark metrics only; not measured-object performance claims.",
        "",
        _render_markdown_table(existing.to_dict(orient="records"), ordered_columns),
        "",
        "## Best Per Horizon",
        "",
        _render_markdown_table(best_rows.to_dict(orient="records"), ordered_columns),
    ]
    md_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


__all__ = [
    "horizon_label",
    "parse_horizons",
    "load_records",
    "load_frame_store",
    "select_frames",
    "horizon_slice",
    "horizon_name",
    "ensure_output_root",
    "evaluate_scores",
    "baseline_scores",
    "tabular_features",
    "sequence_features",
    "transformer_features",
    "counts",
    "operating_metrics",
    "track_metrics",
    "slice_metrics",
    "write_auxiliary_reports",
    "write_reports",
]
