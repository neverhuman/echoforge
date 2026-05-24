#!/usr/bin/env python3
"""Generate the current three-tier radar realism smoke benchmark.

This is a physics-first benchmark gate, not a measured-radar validation claim.
It publishes detector-facing frame products, operational phase metrics, and
counterfactual audits while keeping generator truth columns out of model
features.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_ml_training_types import (
    FRAME_COLUMNS,
    FRAME_INDEX,
    FRAME_PERIOD_S,
    ROLES,
)
from generate_ml_training_physics import (
    group_conditions,
    parse_args,
    phase_specs,
    range_bin,
    speed_prior_for_family,
    stable_seed,
)
from generate_ml_training_frames import build_frame_products
from generate_ml_training_aggregation import (
    aggregate_features,
    negative_control_audit,
    operational_metrics,
    pad_frames,
)
from generate_ml_training_calibration import build_calibration_artifacts
from generate_ml_training_acoustic import build_acoustic_cue_products
from generate_ml_training_report import (
    build_kinematics_audit,
    write_json,
    write_metadata,
)


def main() -> None:
    args = parse_args()
    if args.scenario_groups < 20:
        raise ValueError("current scenario-groups must be at least 20 for train/validation/test negative-control coverage")
    phases = phase_specs(args.max_time_s)
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace generated artifacts")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "restricted_truth").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    frames_raw: list[np.ndarray] = []
    kinematic_rows: list[dict[str, Any]] = []
    record_index = 0
    for group_index in range(args.scenario_groups):
        group = group_conditions(args.seed, group_index)
        site = group["site"]
        sensor = group["sensor"]
        for scene_role in ROLES:
            for phase in phases:
                rng = np.random.default_rng(stable_seed(args.seed, group_index, ROLES.index(scene_role), int(phase.start_s), 9_191))
                truth, frame = build_frame_products(rng, group, phase, scene_role)
                frames_raw.append(frame)
                record_id = f"current_record_{record_index:07d}_{phase.phase_id}"
                scenario_id = f"{group['counterfactual_group_id']}:{scene_role}"
                truth_path = f"restricted_truth/{record_id}.json"
                speed_prior = speed_prior_for_family(str(truth["target_family"]))
                write_json(
                    out_root / truth_path,
                    {
                        **truth,
                        "record_id": record_id,
                        "scenario_id": scenario_id,
                        "counterfactual_group_id": group["counterfactual_group_id"],
                        "site_archetype": asdict(site),
                        "sensor_archetype": asdict(sensor),
                    },
                )
                positive = scene_role == "positive_public_proxy"
                mean_range = float(np.mean(frame[:, FRAME_INDEX["range_m"]]))
                record = {
                    "record_id": record_id,
                    "record_index": record_index,
                    "split": group["split"],
                    "class_id": truth["class_id"],
                    "target_family": truth["target_family"],
                    "scene_role": scene_role,
                    "is_public_proxy_positive": positive,
                    "is_hard_negative": not positive,
                    "hard_negative_family": "" if positive else truth["target_family"],
                    "scenario_seed": int(group["scenario_seed"]),
                    "object_seed": int(stable_seed(args.seed, group_index, int(phase.start_s), 12_001)),
                    "frame_count": int(frame.shape[0]),
                    "cpi_pulses": int(frame[0, FRAME_INDEX["cpi_pulses"]]),
                    "streaming_features_path": f"frame_features.npz#record_id={record_id}",
                    "frame_labels_path": "feature_schema.json",
                    "truth_metadata_path": truth_path,
                    "stratum_id": f"current_{site.site_archetype_id}_{sensor.sensor_archetype_id}_{phase.phase_id}_{group['clutter_regime']}",
                    "wave_index": int(group_index),
                    "difficulty_bucket": "initial_los_limited" if phase.phase_id == "initial_take_up" else "phase_realism",
                    "sensor_band": sensor.band,
                    "range_bin": range_bin(mean_range),
                    "clutter_regime": group["clutter_regime"],
                    "target_aspect": group["target_aspect"],
                    "motion_pattern": "three_tier_public_proxy_path" if positive else "matched_confuser_path",
                    "interference": group["interference"],
                    "confuser_family": "" if positive else truth["target_family"],
                    "scene_object_count": 1 if scene_role != "no_target_counterfactual" else 0,
                    "mixed_scene": bool(group["multipath_enabled"]),
                    "holdout_role": "unseen_site_sensor_confuser" if group["split"] == "test" else "seen",
                    "phase_id": phase.phase_id,
                    "phase_start_s": phase.start_s,
                    "phase_end_s": phase.end_s,
                    "available_history_s": phase.end_s - phase.start_s,
                    "site_archetype_id": site.site_archetype_id,
                    "sensor_archetype_id": sensor.sensor_archetype_id,
                    "validation_tier": "unvalidated/basic synthetic public-proxy",
                    "counterfactual_group_id": group["counterfactual_group_id"],
                    "scenario_id": scenario_id,
                    "horizon_masked_fraction": truth["horizon_masked_fraction"],
                    "los_eligible_fraction": truth["los_eligible_fraction"],
                }
                records.append(record)
                true_speed = float(truth["mean_true_speed_mps"])
                radial_speed = float(truth["mean_abs_radial_velocity_mps"])
                kinematic_rows.append({
                    "record_id": record_id,
                    "record_index": record_index,
                    "split": group["split"],
                    "phase_id": phase.phase_id,
                    "scene_role": scene_role,
                    "target_family": truth["target_family"],
                    "speed_prior_id": speed_prior.prior_id,
                    "propulsion_class": speed_prior.propulsion_class,
                    "stress_class": bool(speed_prior.stress_class),
                    "baseline_positive_prior": bool(speed_prior.baseline_positive),
                    "mean_true_speed_mps": true_speed,
                    "mean_abs_radial_velocity_mps": radial_speed,
                    "radial_to_true_speed_ratio": radial_speed / max(abs(true_speed), 1.0),
                    "radial_velocity_is_true_speed": False,
                    "estimated_ground_speed_exposed": False,
                    "cruise_main_estimate_low_mps": (
                        speed_prior.cruise_main_estimate_mps[0]
                        if speed_prior.cruise_main_estimate_mps is not None
                        else ""
                    ),
                    "cruise_main_estimate_high_mps": (
                        speed_prior.cruise_main_estimate_mps[1]
                        if speed_prior.cruise_main_estimate_mps is not None
                        else ""
                    ),
                })
                record_index += 1

    order_rng = np.random.default_rng(stable_seed(args.seed, args.scenario_groups, 44_404))
    order = order_rng.permutation(len(records))
    records = [records[int(idx)] for idx in order]
    frames_raw = [frames_raw[int(idx)] for idx in order]
    kinematic_rows = [kinematic_rows[int(idx)] for idx in order]
    for idx, record in enumerate(records):
        record["record_index"] = idx
        kinematic_rows[idx]["record_index"] = idx

    frames, valid_mask = pad_frames(frames_raw)
    records_df = pd.DataFrame(records)
    records_df.to_csv(out_root / "records.csv", index=False)
    kinematics_df, kinematics_summary = build_kinematics_audit(records_df, kinematic_rows)
    kinematics_df.to_csv(out_root / "kinematics_audit.csv", index=False, float_format="%.6f")
    np.savez_compressed(
        out_root / "frame_features.npz",
        frames=frames,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U96"),
        frame_columns=np.asarray(FRAME_COLUMNS, dtype="<U64"),
        frame_period_s=np.array(FRAME_PERIOD_S, dtype=np.float32),
        benchmark_profile=np.asarray(["ml-training-three-tier"], dtype="<U48"),
    )
    np.savez_compressed(
        out_root / "valid_frame_mask.npz",
        valid_frame_mask=valid_mask,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U96"),
        benchmark_profile=np.asarray(["ml-training-valid-frame-mask"], dtype="<U48"),
    )
    feature_df, feature_names = aggregate_features(frames, valid_mask)
    feature_df.insert(0, "record_id", records_df["record_id"].to_numpy())
    feature_df.to_csv(out_root / "features.csv", index=False, float_format="%.6f")
    phase_metrics = operational_metrics(records_df, frames, valid_mask)
    phase_metrics.to_csv(out_root / "phase_operational_metrics.csv", index=False, float_format="%.6f")
    controls = negative_control_audit(records_df, feature_df)
    write_json(out_root / "negative_control_audit.json", controls)
    calibration_report, calibration_manifest, calibration_distance_df, calibration_summary = build_calibration_artifacts(
        records_df, frames, valid_mask,
    )
    write_json(out_root / "calibration_report.json", calibration_report)
    write_json(out_root / "calibration_anchor_manifest.json", calibration_manifest)
    calibration_distance_df.to_csv(out_root / "calibration_distance.csv", index=False, float_format="%.6f")
    acoustic_node_df, acoustic_track_df, acoustic_phase_df, acoustic_schema, acoustic_summary = build_acoustic_cue_products(
        records_df, frames, valid_mask,
    )
    acoustic_node_df.to_csv(out_root / "acoustic_node_detections.csv", index=False, float_format="%.6f")
    acoustic_track_df.to_csv(out_root / "acoustic_cue_tracks.csv", index=False, float_format="%.6f")
    acoustic_phase_df.to_csv(out_root / "acoustic_phase_metrics.csv", index=False, float_format="%.6f")
    write_json(out_root / "acoustic_product_schema.json", acoustic_schema)
    write_json(out_root / "acoustic_cue_quality.json", acoustic_summary)

    phase_ids = set(records_df["phase_id"].astype(str))
    expected_phase_ids = {phase.phase_id for phase in phases}
    phase_windows_ok = expected_phase_ids == phase_ids and {
        row.phase_id: (row.start_s, row.end_s) for row in phases
    } == {
        "initial_take_up": (0.0, 30.0),
        "climb_transition": (30.0, 90.0),
        "cruise_altitude": (90.0, float(args.max_time_s)),
    }
    quality = {
        "benchmark_profile": "ml-training-three-tier",
        "record_count": int(len(records_df)),
        "scenario_group_count": int(args.scenario_groups),
        "phase_ids": sorted(phase_ids),
        "phase_windows_status": "pass" if phase_windows_ok else "fail",
        "negative_control_status": controls["status"],
        "negative_control_summary": {key: value for key, value in controls.items() if key.endswith("_auc")},
        "truth_denylist_status": "pass",
        "speed_prior_kinematics_status": kinematics_summary["status"],
        "speed_prior_kinematics_summary": kinematics_summary,
        "calibration_anchor_status": calibration_summary["calibration_anchor_status"],
        "calibration_artifact_status": calibration_summary["status"],
        "calibration_anchor_summary": calibration_summary,
        "acoustic_cueing_status": acoustic_summary["status"],
        "acoustic_cueing_summary": acoustic_summary,
        "initial_take_up_low_pd_allowed": True,
        "operational_metrics": phase_metrics.to_dict(orient="records"),
        "acoustic_operational_metrics": acoustic_phase_df.to_dict(orient="records"),
        "status": (
            "pass"
            if phase_windows_ok
            and controls["status"] == "pass"
            and kinematics_summary["status"] == "pass"
            and calibration_summary["status"] == "pass"
            and acoustic_summary["status"] == "pass"
            else "fail"
        ),
    }
    write_metadata(out_root, records_df, feature_names, quality, phases, args)
    write_json(out_root / "quality_report.json", quality)
    if not phase_windows_ok:
        raise AssertionError("current phase windows do not match initial_take_up/climb_transition/cruise_altitude semantics")
    if controls["status"] != "pass":
        raise AssertionError(f"negative-control audit failed: {controls}")
    if kinematics_summary["status"] != "pass":
        raise AssertionError(f"speed-prior kinematics audit failed: {kinematics_summary}")
    if calibration_summary["status"] != "pass":
        raise AssertionError(f"calibration anchor artifact audit failed: {calibration_summary}")
    if acoustic_summary["status"] != "pass":
        raise AssertionError(f"acoustic cueing artifact audit failed: {acoustic_summary}")
    print(
        f"wrote {out_root} groups={args.scenario_groups} records={len(records_df)} "
        f"phases={','.join(sorted(phase_ids))} controls={controls['status']} "
        f"calibration={calibration_summary['calibration_anchor_status']} "
        f"acoustic={acoustic_summary['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
