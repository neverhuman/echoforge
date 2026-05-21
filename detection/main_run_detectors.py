"""Detector and fusion baselines for the public-proxy main-run corpus."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detection.main_run_types import DATASET_PROFILE
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_types import DATASET_PROFILE


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _float_or_none(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _cfar_score(iq: np.ndarray) -> float:
    power = np.abs(iq) ** 2
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)) + 1e-6)
    return float((np.max(power) - median) / mad)


def _mtd_score(iq: np.ndarray) -> float:
    doppler = np.abs(np.fft.fft(iq, axis=0)) ** 2
    return float(np.max(doppler) / (np.mean(doppler) + 1e-6))


def _track_score(iq: np.ndarray) -> float:
    power = np.abs(iq) ** 2
    range_power = power.mean(axis=0)
    pulse_power = power.mean(axis=1)
    contrast = float(np.max(range_power) / (np.mean(range_power) + 1e-6))
    continuity = float(1.0 / (1.0 + np.std(np.diff(pulse_power))))
    return contrast * continuity


def _acoustic_score(acoustic: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(acoustic, axis=1))
    peak = np.max(spectrum[:, 1:], axis=1)
    mean = np.mean(spectrum[:, 1:], axis=1) + 1e-6
    agreement = 1.0 / (1.0 + np.std(np.argmax(spectrum[:, 1:], axis=1).astype(np.float64)))
    return float(np.mean(peak / mean) * agreement)


def _load_detector_features(data_root: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    records = _read_csv(data_root / "records.csv")
    index_rows = _read_csv(data_root / "raw_stream_index.csv")
    acoustic_rows = _read_csv(data_root / "acoustic_stream_index.csv")
    index_by_record = {row["record_id"]: row for row in index_rows}
    acoustic_by_record = {row["record_id"]: row for row in acoustic_rows}
    active_cache: dict[str, Any] = {}
    acoustic_cache: dict[str, Any] = {}
    features = np.zeros((len(records), 4), dtype=np.float64)
    for idx, record in enumerate(records):
        active_index = index_by_record[record["record_id"]]
        active_path = str(active_index["shard_path"])
        if active_path not in active_cache:
            with np.load(data_root / active_path) as loaded:
                active_cache[active_path] = {"iq": loaded["iq"]}
        active_npz = active_cache[active_path]
        active_row = int(active_index["row_offset"])
        iq = active_npz["iq"][active_row]

        acoustic_index = acoustic_by_record[record["record_id"]]
        acoustic_path = str(acoustic_index["shard_path"])
        if acoustic_path not in acoustic_cache:
            with np.load(data_root / acoustic_path) as loaded:
                acoustic_cache[acoustic_path] = {"acoustic": loaded["acoustic"]}
        acoustic_npz = acoustic_cache[acoustic_path]
        acoustic = acoustic_npz["acoustic"][int(acoustic_index["row_offset"])]

        features[idx, 0] = math.log1p(_cfar_score(iq[0]))
        features[idx, 1] = math.log1p(_mtd_score(iq[1]))
        features[idx, 2] = math.log1p(_track_score(iq[2]))
        features[idx, 3] = math.log1p(_acoustic_score(acoustic))
    return records, features


def _fit_gaussian_log_odds(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, dict[str, Any]]:
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        weights = np.ones(x.shape[1], dtype=np.float64) * 0.1
        intercept = 0.0
        return weights, intercept, {"status": "fallback_single_class"}
    pos_mean = np.mean(pos, axis=0)
    neg_mean = np.mean(neg, axis=0)
    pooled_var = np.var(x, axis=0) + 1e-3
    weights = (pos_mean - neg_mean) / pooled_var
    prior = (float(len(pos)) + 0.5) / (float(len(y)) + 1.0)
    midpoint = 0.5 * (pos_mean + neg_mean)
    intercept = math.log(prior / (1.0 - prior)) - float(np.dot(weights, midpoint))
    return (
        weights,
        intercept,
        {
            "status": "pass",
            "positive_rows": int(len(pos)),
            "negative_rows": int(len(neg)),
        },
    )


def _predict_log_odds(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return x @ weights + intercept


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    true_positives = np.cumsum(sorted_y == 1)
    ranks = np.arange(1, len(sorted_y) + 1, dtype=np.float64)
    precision_at_hit = true_positives[sorted_y == 1] / ranks[sorted_y == 1]
    return float(np.sum(precision_at_hit) / positives)


def _binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    positive = y == 1
    negative = ~positive
    tp = float(np.sum(pred & positive))
    fp = float(np.sum(pred & negative))
    tn = float(np.sum(~pred & negative))
    fn = float(np.sum(~pred & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(float(len(y)), 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": f1,
    }


def _select_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0, _binary_metrics(y, scores, 0.0)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    best_threshold = float(sorted_scores[0])
    best_metrics = _binary_metrics(y, scores, best_threshold)
    best_key = (best_metrics["f1"], best_metrics["recall"], best_metrics["precision"])
    for idx in np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]):
        threshold = float(sorted_scores[int(idx)])
        pred = sorted_scores >= threshold
        tp = float(np.sum(pred & (sorted_y == 1)))
        fp = float(np.sum(pred & (sorted_y == 0)))
        fn = float(np.sum(~pred & (sorted_y == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = (f1, recall, precision)
        if key > best_key:
            best_threshold = threshold
            best_metrics = _binary_metrics(y, scores, threshold)
            best_key = key
    return best_threshold, best_metrics


def _build_performance_reports(
    records: list[dict[str, str]],
    method_scores: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray([int(record["label_id"]) for record in records], dtype=np.int8)
    splits = np.asarray([record["split_role"] for record in records])
    phases = np.asarray([record["phase_id"] for record in records])
    train_mask = splits == "train_cv"
    rows: list[dict[str, Any]] = []
    threshold_summary: dict[str, Any] = {}

    for method, scores in method_scores.items():
        threshold, train_threshold_metrics = _select_threshold(y[train_mask], scores[train_mask])
        threshold_summary[method] = {
            "threshold_source": "train_cv_max_f1",
            "threshold": threshold,
            "train_cv_threshold_metrics": train_threshold_metrics,
        }
        slice_specs: list[tuple[str, str, np.ndarray]] = [
            ("all", "all", np.ones(len(records), dtype=bool)),
            ("train_cv", "all", train_mask),
            ("holdout", "all", splits == "holdout"),
        ]
        for phase in sorted(set(phases)):
            phase_mask = phases == phase
            slice_specs.append(("holdout", str(phase), (splits == "holdout") & phase_mask))
        for split_role, phase_id, mask in slice_specs:
            labels = y[mask]
            sliced_scores = scores[mask]
            binary = _binary_metrics(labels, sliced_scores, threshold)
            roc_auc = _float_or_none(_roc_auc(labels, sliced_scores))
            average_precision = _float_or_none(_average_precision(labels, sliced_scores))
            rows.append(
                {
                    "method": method,
                    "split_role": split_role,
                    "phase_id": phase_id,
                    "record_count": int(len(labels)),
                    "positive_count": int(np.sum(labels == 1)),
                    "negative_count": int(np.sum(labels == 0)),
                    "threshold_source": "train_cv_max_f1",
                    "threshold": _format_metric(threshold),
                    "roc_auc": _format_metric(roc_auc),
                    "average_precision": _format_metric(average_precision),
                    "accuracy": _format_metric(binary["accuracy"]),
                    "precision": _format_metric(binary["precision"]),
                    "recall": _format_metric(binary["recall"]),
                    "specificity": _format_metric(binary["specificity"]),
                    "false_positive_rate": _format_metric(binary["false_positive_rate"]),
                    "f1": _format_metric(binary["f1"]),
                }
            )

    holdout_rows = [
        row for row in rows if row["split_role"] == "holdout" and row["phase_id"] == "all"
    ]
    holdout_by_method = {
        str(row["method"]): {
            "roc_auc": row["roc_auc"],
            "average_precision": row["average_precision"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "false_positive_rate": row["false_positive_rate"],
            "f1": row["f1"],
            "threshold": row["threshold"],
        }
        for row in holdout_rows
    }
    summary = {
        "dataset_profile": DATASET_PROFILE,
        "metric_policy": (
            "synthetic public-proxy benchmark metrics; thresholds selected on train/CV only"
        ),
        "methods": list(method_scores),
        "thresholds": threshold_summary,
        "holdout": holdout_by_method,
        "status": "pass" if holdout_rows else "fail",
    }
    return rows, summary


def _calibrated_predictions(
    records: list[dict[str, str]],
    features: np.ndarray,
    *,
    folds: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray([int(record["label_id"]) for record in records], dtype=np.int8)
    split = np.asarray([record["split_role"] for record in records])
    cv_fold = np.asarray(
        [-1 if record["cv_fold"] == "" else int(record["cv_fold"]) for record in records]
    )
    train_cv = split == "train_cv"
    holdout = split == "holdout"
    log_odds = np.zeros(len(records), dtype=np.float64)
    calibration_rows: list[dict[str, Any]] = []

    for fold in range(folds):
        score_mask = train_cv & (cv_fold == fold)
        fit_mask = train_cv & (cv_fold != fold)
        weights, intercept, fit_info = _fit_gaussian_log_odds(features[fit_mask], y[fit_mask])
        log_odds[score_mask] = _predict_log_odds(features[score_mask], weights, intercept)
        calibration_rows.append(
            {
                "fold": fold,
                "fit_split_role": "train_cv",
                "scored_split_role": "train_cv",
                "fit_record_count": int(np.sum(fit_mask)),
                "scored_record_count": int(np.sum(score_mask)),
                "fit_positive_count": int(np.sum(y[fit_mask])),
                "fit_negative_count": int(np.sum(fit_mask) - np.sum(y[fit_mask])),
                "fit_status": fit_info["status"],
            }
        )
    weights, intercept, holdout_fit = _fit_gaussian_log_odds(features[train_cv], y[train_cv])
    log_odds[holdout] = _predict_log_odds(features[holdout], weights, intercept)
    calibration_rows.append(
        {
            "fold": "holdout",
            "fit_split_role": "train_cv",
            "scored_split_role": "holdout",
            "fit_record_count": int(np.sum(train_cv)),
            "scored_record_count": int(np.sum(holdout)),
            "fit_positive_count": int(np.sum(y[train_cv])),
            "fit_negative_count": int(np.sum(train_cv) - np.sum(y[train_cv])),
            "fit_status": holdout_fit["status"],
        }
    )
    manifest = {
        "dataset_profile": DATASET_PROFILE,
        "calibration_policy": "cv_only_train_cv_pool",
        "used_split_roles": ["train_cv"],
        "excluded_split_roles": ["holdout"],
        "fold_count": folds,
        "train_cv_record_count": int(np.sum(train_cv)),
        "holdout_record_count": int(np.sum(holdout)),
        "holdout_fit_record_count": 0,
        "feature_columns": [
            "high_res_cfar_log1p",
            "sband_mtd_log1p",
            "gbad_track_log1p",
            "acoustic_cadence_log1p",
        ],
        "status": "pass" if int(np.sum(holdout)) > 0 and int(np.sum(train_cv)) > 0 else "fail",
    }
    return log_odds, calibration_rows, manifest


def run_main_run_detectors(
    data_root: Path,
    out_root: Path,
    *,
    folds: int = 5,
    seed: int = 202605210136,
    force: bool = False,
) -> dict[str, Any]:
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records, features = _load_detector_features(data_root)
    log_odds, calibration_rows, calibration_manifest = _calibrated_predictions(
        records, features, folds=folds
    )
    probabilities = _sigmoid(log_odds)
    tabular_scores = np.mean(features[:, :3], axis=1)
    sequence_scores = 0.65 * features[:, 1] + 0.35 * features[:, 3]
    method_scores = {
        "high_resolution_xku_cuas": features[:, 0],
        "tactical_s_band_aesa": features[:, 1],
        "gbad_3d4d_cueing": features[:, 2],
        "distributed_acoustic_cue": features[:, 3],
        "tabular_ml_baseline": tabular_scores,
        "sequence_ml_proxy": sequence_scores,
        "layered_fusion_c2": probabilities,
    }
    performance_rows, performance_summary = _build_performance_reports(records, method_scores)
    cfar_rows: list[dict[str, Any]] = []
    ml_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        base = {
            "record_id": record["record_id"],
            "scenario_group_id": record["scenario_group_id"],
            "time_lock_id": record["time_lock_id"],
            "phase_id": record["phase_id"],
            "split_role": record["split_role"],
            "cv_fold": record["cv_fold"],
            "label_id": int(record["label_id"]),
        }
        cfar_rows.append(
            {
                **base,
                "high_res_cfar_log1p": f"{features[idx, 0]:.6f}",
                "sband_mtd_log1p": f"{features[idx, 1]:.6f}",
                "gbad_track_log1p": f"{features[idx, 2]:.6f}",
                "processing_family": "ca_os_cfar_mti_mtd_track_lifecycle",
            }
        )
        ml_rows.append(
            {
                **base,
                "tabular_baseline_score": f"{float(tabular_scores[idx]):.6f}",
                "sequence_proxy_score": f"{float(sequence_scores[idx]):.6f}",
                "model_family": "deterministic_baseline_no_external_dependency",
            }
        )
        fusion_rows.append(
            {
                **base,
                "fusion_log_odds": f"{float(log_odds[idx]):.6f}",
                "fusion_probability": f"{float(probabilities[idx]):.6f}",
                "calibration_role": (
                    "cv_fold_scored"
                    if record["split_role"] == "train_cv"
                    else "holdout_scored_only"
                ),
                "provenance_weighting": "source_quality_and_freshness",
                "run_seed": seed,
            }
        )
    _write_csv(out_root / "classical_radar_processing.csv", cfar_rows)
    _write_csv(out_root / "ml_detector_baselines.csv", ml_rows)
    _write_csv(out_root / "fusion_predictions.csv", fusion_rows)
    _write_csv(out_root / "calibration_folds.csv", calibration_rows)
    _write_csv(out_root / "performance_metrics.csv", performance_rows)
    _write_json(out_root / "calibration_manifest.json", calibration_manifest)
    _write_json(out_root / "performance_summary.json", performance_summary)

    record_ids = {record["record_id"] for record in records}
    quality = {
        "dataset_profile": DATASET_PROFILE,
        "record_count": len(records),
        "record_identity_status": (
            "pass"
            if len(record_ids) == len(cfar_rows) == len(ml_rows) == len(fusion_rows)
            else "fail"
        ),
        "holdout_isolation_status": (
            "pass"
            if calibration_manifest["used_split_roles"] == ["train_cv"]
            and calibration_manifest["holdout_fit_record_count"] == 0
            else "fail"
        ),
        "detector_methods": [
            "ca_os_cfar",
            "mti_mtd_doppler_filtering",
            "stft_cepstrum_cadence_proxy",
            "track_lifecycle_metrics",
            "calibrated_late_fusion",
        ],
        "performance_summary_path": "performance_summary.json",
        "performance_metrics_path": "performance_metrics.csv",
        "holdout_performance": performance_summary["holdout"],
        "status": (
            "pass"
            if calibration_manifest["status"] == "pass"
            and len(record_ids) == len(records)
            and performance_summary["status"] == "pass"
            else "fail"
        ),
    }
    _write_json(out_root / "fusion_quality_report.json", quality)
    if quality["status"] != "pass":
        raise AssertionError(f"detector quality checks failed: {quality}")
    return quality
