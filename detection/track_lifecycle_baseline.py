#!/usr/bin/env python3
"""Build the current track-lifecycle smoke baseline.

This reads detector-facing current frame tensors and emits per-record lifecycle
rows plus phase rollups. Generator labels are used only for aggregate audit
metrics such as missed-track and false-track rates; they are not written to
the detector-facing lifecycle rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("outputs/training-data/shahed136-public-proxy-ml-training-smoke")
DEFAULT_OUT_DIR = Path("outputs/detector-realism/track-lifecycle-baseline")
DEFAULT_REPORT = Path("detection/reports/track_lifecycle_baseline.md")
TRACK_DETECTOR_FAMILY_ID = "track-lifecycle-baseline"
FRAME_PERIOD_S = 0.5
EXPECTED_PHASES = ("initial_take_up", "climb_transition", "cruise_altitude")
DENIED_LIFECYCLE_COLUMNS = {
    "scenario_seed",
    "object_seed",
    "class_id",
    "target_family",
    "scene_role",
    "is_public_proxy_positive",
    "is_hard_negative",
    "hard_negative_family",
    "confuser_family",
    "truth_metadata_path",
    "mean_true_speed_mps",
    "true_speed_mps",
    "ground_speed_mps",
    "estimated_ground_speed_mps",
    "raw_rcs_dbsm",
    "rcs_dbsm",
    "link_budget_snr_db",
}
REQUIRED_LIFECYCLE_COLUMNS = {
    "record_id",
    "phase_id",
    "detector_family_id",
    "source_id",
    "track_state",
    "fragment_count",
    "first_hit_latency_s",
    "track_initiation_latency_s",
    "track_confidence",
}
REQUIRED_PHASE_METRICS = {
    "phase_id",
    "pd_any_cfar",
    "pfa_any_cfar",
    "track_confirmation_rate",
    "false_track_rate",
    "missed_track_rate",
    "track_initiation_latency_s",
    "track_fragmentation_rate",
    "deletion_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def first_true_time(mask: np.ndarray, times: np.ndarray) -> float | None:
    if mask.size == 0 or not bool(mask.any()):
        return None
    return float(times[int(np.argmax(mask))])


def count_fragments(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    transitions = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    return int((transitions == 1).sum())


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def load_inputs(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    records_path = data_root / "records.csv"
    frames_path = data_root / "frame_features.npz"
    mask_path = data_root / "valid_frame_mask.npz"
    for path in (records_path, frames_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(f"missing required current artifact: {path}")
    records = pd.read_csv(records_path)
    frame_payload = np.load(frames_path)
    mask_payload = np.load(mask_path)
    frames = np.asarray(frame_payload["frames"], dtype=np.float32)
    valid_mask = np.asarray(mask_payload["valid_frame_mask"], dtype=bool)
    frame_columns = [str(value) for value in frame_payload["frame_columns"].tolist()]
    frame_record_ids = [str(value) for value in frame_payload["record_ids"].tolist()]
    record_ids = records["record_id"].astype(str).tolist()
    if frame_record_ids != record_ids:
        raise ValueError("records.csv and frame_features.npz record_id order differ")
    return records, frames, valid_mask, frame_columns


def threshold_by_phase(records: pd.DataFrame, frames: np.ndarray, valid_mask: np.ndarray, col: dict[str, int]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for phase_id in EXPECTED_PHASES:
        phase_mask = (records["phase_id"].astype(str) == phase_id).to_numpy()
        train_mask = (records["split"].astype(str) == "train").to_numpy() & phase_mask
        values = frames[train_mask, :, col["tbd_track_score"]]
        valid = valid_mask[train_mask]
        samples = values[valid] if values.size else np.asarray([], dtype=np.float32)
        thresholds[phase_id] = float(np.quantile(samples, 0.64)) if samples.size else 0.0
    return thresholds


def build_lifecycle(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    frame_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    col = {name: idx for idx, name in enumerate(frame_columns)}
    for required in ("time_s", "cfar_detected", "tbd_track_score", "snr_db", "doppler_scr"):
        if required not in col:
            raise ValueError(f"frame_features.npz missing required frame column: {required}")
    thresholds = threshold_by_phase(records, frames, valid_mask, col)
    rows: list[dict[str, Any]] = []

    for idx, record in records.iterrows():
        phase_id = str(record["phase_id"])
        valid = valid_mask[idx]
        frame = frames[idx, valid]
        if frame.size == 0:
            cfar = np.asarray([], dtype=bool)
            confirmed = np.asarray([], dtype=bool)
            times = np.asarray([], dtype=np.float32)
            tbd = np.asarray([], dtype=np.float32)
            snr = np.asarray([], dtype=np.float32)
            scr = np.asarray([], dtype=np.float32)
        else:
            times = frame[:, col["time_s"]]
            cfar = frame[:, col["cfar_detected"]] > 0.5
            tbd = frame[:, col["tbd_track_score"]]
            threshold = thresholds.get(phase_id, 0.0)
            confirmed = cfar & (tbd > threshold)
            snr = frame[:, col["snr_db"]]
            scr = frame[:, col["doppler_scr"]]
        threshold = thresholds.get(phase_id, 0.0)
        fragment_count = count_fragments(confirmed)
        first_hit = first_true_time(cfar, times)
        track_init = first_true_time(confirmed, times)
        confirmed_count = int(confirmed.sum())
        cfar_count = int(cfar.sum())
        max_score = float(np.max(tbd)) if tbd.size else 0.0
        mean_score = float(np.mean(tbd)) if tbd.size else 0.0
        confidence = sigmoid((max_score - threshold) / max(abs(threshold), 0.05))
        if track_init is None:
            track_state = "unconfirmed"
            deletion_reason = "no_confirmation"
            track_id = ""
        else:
            track_state = "confirmed"
            deletion_reason = "phase_end" if bool(confirmed[-1]) else "below_threshold"
            track_id = f"trk_{record['record_id']}"
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "phase_id": phase_id,
                "detector_family_id": TRACK_DETECTOR_FAMILY_ID,
                "source_id": str(record["sensor_archetype_id"]),
                "track_id": track_id,
                "track_state": track_state,
                "first_hit_latency_s": "" if first_hit is None else first_hit,
                "track_initiation_latency_s": "" if track_init is None else track_init,
                "fragment_count": fragment_count,
                "confirmed_frame_count": confirmed_count,
                "cfar_hit_count": cfar_count,
                "track_duration_s": confirmed_count * FRAME_PERIOD_S,
                "track_confidence": confidence,
                "max_track_score": max_score,
                "mean_track_score": mean_score,
                "tbd_threshold": threshold,
                "max_snr_db": float(np.max(snr)) if snr.size else float("nan"),
                "max_doppler_scr_db": float(np.max(scr)) if scr.size else float("nan"),
                "deletion_reason": deletion_reason,
                "horizon_masked_fraction": float(record["horizon_masked_fraction"]),
                "los_eligible_fraction": float(record["los_eligible_fraction"]),
            }
        )

    lifecycle = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for phase_id in EXPECTED_PHASES:
        phase_records = records[records["phase_id"].astype(str) == phase_id]
        phase_lifecycle = lifecycle.loc[phase_records.index]
        labels = phase_records["is_public_proxy_positive"].astype(bool).to_numpy()
        track_promoted = phase_lifecycle["track_state"].astype(str).to_numpy() == "confirmed"
        cfar_any = phase_lifecycle["cfar_hit_count"].astype(float).to_numpy() > 0
        fragments = phase_lifecycle["fragment_count"].astype(float).to_numpy()
        positive_count = int(labels.sum())
        negative_count = int((~labels).sum())
        positive_inits = [
            float(value)
            for value, label in zip(phase_lifecycle["track_initiation_latency_s"], labels)
            if label and str(value) != "" and math.isfinite(float(value))
        ]
        positive_deletions = [
            reason
            for reason, label, promoted in zip(phase_lifecycle["deletion_reason"], labels, track_promoted)
            if label and promoted
        ]
        metric_rows.append(
            {
                "phase_id": phase_id,
                "record_count": int(len(phase_records)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "pd_any_cfar": float(cfar_any[labels].mean()) if positive_count else float("nan"),
                "pfa_any_cfar": float(cfar_any[~labels].mean()) if negative_count else float("nan"),
                "track_confirmation_rate": float(track_promoted[labels].mean()) if positive_count else float("nan"),
                "false_track_rate": float(track_promoted[~labels].mean()) if negative_count else float("nan"),
                "missed_track_rate": float((~track_promoted[labels]).mean()) if positive_count else float("nan"),
                "track_initiation_latency_s": finite_mean(positive_inits),
                "track_fragmentation_rate": float(fragments[labels].mean()) if positive_count else float("nan"),
                "deletion_rate": (
                    float(sum(1 for reason in positive_deletions if str(reason) == "below_threshold") / len(positive_deletions))
                    if positive_deletions
                    else float("nan")
                ),
                "stable_track_rate": (
                    float(((track_promoted & labels) & (fragments <= 1.0)).sum() / max(1, positive_count))
                    if positive_count
                    else float("nan")
                ),
                "mean_horizon_masked_fraction": float(phase_records["horizon_masked_fraction"].mean()),
                "mean_los_eligible_fraction": float(phase_records["los_eligible_fraction"].mean()),
            }
        )
    phase_metrics = pd.DataFrame(metric_rows)
    lifecycle_columns = set(lifecycle.columns)
    denied_present = sorted(DENIED_LIFECYCLE_COLUMNS & lifecycle_columns)
    missing_lifecycle = sorted(REQUIRED_LIFECYCLE_COLUMNS - lifecycle_columns)
    missing_metrics = sorted(REQUIRED_PHASE_METRICS - set(phase_metrics.columns))
    source_ids_missing = int((lifecycle["source_id"].astype(str) == "").sum())
    detector_ids_ok = bool((lifecycle["detector_family_id"].astype(str) == TRACK_DETECTOR_FAMILY_ID).all())
    phase_ids_ok = sorted(phase_metrics["phase_id"].astype(str).tolist()) == sorted(EXPECTED_PHASES)
    quality = {
        "artifact": TRACK_DETECTOR_FAMILY_ID,
        "status": (
            "pass"
            if not denied_present
            and not missing_lifecycle
            and not missing_metrics
            and source_ids_missing == 0
            and detector_ids_ok
            and phase_ids_ok
            else "fail"
        ),
        "claim_boundary": (
            "Synthetic track-lifecycle smoke baseline over current detector-facing frame products. "
            "Labels are used only for aggregate audit metrics; lifecycle rows are not measured tracking validation."
        ),
        "lifecycle_row_count": int(len(lifecycle)),
        "phase_ids": sorted(phase_metrics["phase_id"].astype(str).tolist()),
        "detector_family_id": TRACK_DETECTOR_FAMILY_ID,
        "denied_lifecycle_columns_present": denied_present,
        "missing_lifecycle_columns": missing_lifecycle,
        "missing_phase_metric_columns": missing_metrics,
        "source_id_missing_count": source_ids_missing,
        "detector_ids_ok": detector_ids_ok,
        "model_facing_default": False,
        "max_false_track_rate": float(phase_metrics["false_track_rate"].max()),
        "max_missed_track_rate": float(phase_metrics["missed_track_rate"].max()),
        "max_track_fragmentation_rate": float(phase_metrics["track_fragmentation_rate"].max()),
    }
    return lifecycle, phase_metrics, quality


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for _, key in columns:
            value = row.get(key, "")
            rendered.append(str(value) if key == "phase_id" else fmt(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def render_markdown(data_root: Path, phase_metrics: pd.DataFrame, quality: dict[str, Any]) -> str:
    rows = phase_metrics.to_dict(orient="records")
    lines = [
        "# Track Lifecycle Baseline current",
        "",
        f"Data root: `{data_root}`",
        "",
        "## Status",
        "",
        f"- Artifact status: `{quality['status']}`",
        f"- Detector family ID: `{quality['detector_family_id']}`",
        f"- Lifecycle rows: `{quality['lifecycle_row_count']}`",
        f"- Model-facing default: `{str(quality['model_facing_default']).lower()}`",
        "",
        "## Claim Boundary",
        "",
        str(quality["claim_boundary"]),
        "",
        "## Phase Metrics",
        "",
        render_table(
            rows,
            [
                ("Phase", "phase_id"),
                ("Pd", "pd_any_cfar"),
                ("Pfa", "pfa_any_cfar"),
                ("Confirm", "track_confirmation_rate"),
                ("False Track", "false_track_rate"),
                ("Missed", "missed_track_rate"),
                ("Init s", "track_initiation_latency_s"),
                ("Frag", "track_fragmentation_rate"),
                ("Delete", "deletion_rate"),
                ("Stable", "stable_track_rate"),
            ],
        ),
        "",
        "## Audit Checks",
        "",
        "| Check | Value |",
        "|---|---|",
        f"| Denied lifecycle columns | {', '.join(quality['denied_lifecycle_columns_present']) or 'none'} |",
        f"| Missing lifecycle columns | {', '.join(quality['missing_lifecycle_columns']) or 'none'} |",
        f"| Missing phase metric columns | {', '.join(quality['missing_phase_metric_columns']) or 'none'} |",
        f"| Missing source IDs | {quality['source_id_missing_count']} |",
        f"| Detector IDs OK | {quality['detector_ids_ok']} |",
        "",
        "## Interpretation",
        "",
        "- This baseline gives later detector products a reproducible lifecycle reference for initiation, deletion, fragmentation, false tracks, and misses.",
        "- Current false-track, missed-track, and fragmentation values remain blocker evidence for regeneration-ready claims.",
        "- `initial_take_up` may legitimately remain weak for radar because line of sight is often horizon-masked; the row is reported rather than hidden.",
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(
    data_root: Path,
    out_dir: Path,
    report_path: Path,
    lifecycle: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    quality: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.to_csv(out_dir / "track_lifecycle_rows.csv", index=False, float_format="%.6f")
    phase_metrics.to_csv(out_dir / "track_lifecycle_phase_metrics.csv", index=False, float_format="%.6f")
    schema = {
        "artifact": TRACK_DETECTOR_FAMILY_ID,
        "files": {
            "lifecycle_rows": "track_lifecycle_rows.csv",
            "phase_metrics": "track_lifecycle_phase_metrics.csv",
            "quality": "track_lifecycle_quality.json",
        },
        "lifecycle_columns": list(lifecycle.columns),
        "phase_metric_columns": list(phase_metrics.columns),
        "denied_lifecycle_columns": sorted(DENIED_LIFECYCLE_COLUMNS),
        "model_facing_default": False,
    }
    write_json(out_dir / "track_lifecycle_schema.json", schema)
    write_json(out_dir / "track_lifecycle_quality.json", quality)
    markdown = render_markdown(data_root, phase_metrics, quality)
    report_path.write_text(markdown, encoding="utf-8")
    (out_dir / "track_lifecycle_baseline.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records, frames, valid_mask, frame_columns = load_inputs(args.data_root)
    lifecycle, phase_metrics, quality = build_lifecycle(records, frames, valid_mask, frame_columns)
    write_outputs(args.data_root, args.out_dir, args.report, lifecycle, phase_metrics, quality)
    print(
        f"wrote {args.report} and {args.out_dir} status={quality['status']} "
        f"rows={quality['lifecycle_row_count']}",
        flush=True,
    )
    if quality["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
