"""Report and metadata writers for the ML training generator.

Handles JSON/CSV output, split-manifest generation, speed-prior
manifest, kinematics audit, dataset card, and related artifacts.
Imports from generate_ml_training_types and generate_ml_training_physics
only — no circular imports.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_ml_training_types import (
    FRAME_COLUMNS,
    PhaseSpec,
)
from generate_ml_training_tables import (
    FAMILY_SPEED_PRIOR,
    SPEED_PRIORS,
)
from generate_ml_training_aggregation import denylist_violations


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def speed_prior_manifest_payload() -> dict[str, Any]:
    return {
        "benchmark_profile": "ml-training-three-tier",
        "claim_boundary": "Public-source speed priors and stress-class boundaries only; no measured target truth or proprietary-equivalent behavior is claimed.",
        "policy": [
            "Baseline piston pusher-prop public-proxy positives use the 45-60 m/s cruise working band, with 50-55 m/s as the main estimate.",
            "Fast prop or modified variants remain separate stress classes.",
            "Jet-powered variants remain fast_jet_owa_public_proxy stress data around 110+ m/s and are not blended into baseline positives.",
            "radial_velocity_mps is radar-observable line-of-sight velocity, not true speed.",
            "estimated_ground_speed_mps remains denylisted until a track-history or multi-view estimator is implemented.",
        ],
        "family_to_prior": FAMILY_SPEED_PRIOR,
        "priors": {prior_id: asdict(prior) for prior_id, prior in SPEED_PRIORS.items()},
        "source_context": [
            "OSMP visual guide: delta/pusher fixed-wing public proxy family context.",
            "Army Recognition public technical profile: broad piston pusher-prop speed prior context.",
            "Reuters-syndicated reporting: faster prop and jet variants treated as stress classes, not baseline positives.",
        ],
    }


def build_kinematics_audit(records: pd.DataFrame, kinematic_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = pd.DataFrame(kinematic_rows).sort_values(["record_index", "record_id"]).reset_index(drop=True)
    positive_families = sorted(records.loc[records["is_public_proxy_positive"], "target_family"].astype(str).unique())
    stress_families = {"fast_prop_owa_public_proxy", "fast_jet_owa_public_proxy"}
    baseline_cruise = audit[
        (audit["scene_role"] == "positive_public_proxy")
        & (audit["target_family"] == "public_proxy_pusher_prop_baseline")
        & (audit["phase_id"] == "cruise_altitude")
    ]
    if baseline_cruise.empty:
        baseline_cruise_status = "fail"
    else:
        baseline_cruise_status = "pass" if (
            (baseline_cruise["mean_true_speed_mps"] >= 45.0)
            & (baseline_cruise["mean_true_speed_mps"] <= 60.0)
        ).all() else "fail"
    stress_separation_status = "pass" if not any(family in stress_families for family in positive_families) else "fail"
    crossing_examples = audit[
        (audit["target_family"] == "public_proxy_pusher_prop_baseline")
        & (audit["phase_id"] == "cruise_altitude")
        & (audit["mean_true_speed_mps"] >= 45.0)
        & (audit["radial_to_true_speed_ratio"] <= 0.25)
    ]
    summary = {
        "baseline_positive_families": positive_families,
        "baseline_cruise_speed_band_status": baseline_cruise_status,
        "stress_class_separation_status": stress_separation_status,
        "crossing_target_low_radial_count": int(len(crossing_examples)),
        "estimated_ground_speed_exposed": False,
        "status": "pass" if baseline_cruise_status == "pass" and stress_separation_status == "pass" else "fail",
    }
    return audit, summary


def write_metadata(
    out_root: Path,
    records: pd.DataFrame,
    feature_names: list[str],
    quality: dict[str, Any],
    phases: list[PhaseSpec],
    args: argparse.Namespace,
) -> None:
    split_manifest = records[[
        "record_id", "split", "scenario_seed", "object_seed", "class_id",
        "target_family", "scene_role", "phase_id", "counterfactual_group_id", "scenario_id",
    ]].copy()
    split_manifest.insert(2, "split_key_kind", "counterfactual_group_phase_site_sensor")
    split_manifest.insert(
        3,
        "split_key",
        records["counterfactual_group_id"].astype(str)
        + ":"
        + records["phase_id"].astype(str)
        + ":"
        + records["site_archetype_id"].astype(str)
        + ":"
        + records["sensor_archetype_id"].astype(str),
    )
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)

    denylist = {
        "policy": "Fail generation if restricted generator truth appears in detector-facing frame or aggregate feature columns.",
        "restricted_feature_names": sorted(
            {
                "scenario_seed", "object_seed", "class_id", "target_family", "scene_role",
                "phase_id", "altitude_m", "nominal_snr_db", "link_budget_snr_db", "raw_rcs_dbsm",
                "rcs_dbsm", "validation_tier", "calibration_anchor_ids", "source_metadata",
                "true_speed_mps", "ground_speed_mps", "estimated_ground_speed_mps",
            }
        ),
        "published_frame_columns": FRAME_COLUMNS,
        "published_aggregate_feature_columns": feature_names,
        "violations": denylist_violations(FRAME_COLUMNS, feature_names),
        "status": "pass" if not denylist_violations(FRAME_COLUMNS, feature_names) else "fail",
    }
    write_json(out_root / "generator_truth_denylist.json", denylist)
    if denylist["violations"]:
        raise AssertionError(f"restricted truth leaked into feature columns: {denylist['violations']}")

    write_json(out_root / "feature_schema.json", {
        "benchmark_profile": "ml-training-three-tier",
        "frame_period_s": 0.5,
        "frame_columns": FRAME_COLUMNS,
        "valid_frame_mask": "valid_frame_mask.npz",
        "aggregate_feature_columns": feature_names,
        "phase_model": [asdict(phase) for phase in phases],
        "sensor_public_contract": {
            "allowed_to_models": [
                "frame_features.npz frame tensor columns",
                "valid_frame_mask.npz causal valid-frame mask",
                "features.csv aggregate detector observables",
            ],
            "detector_sidecar_products": [
                "acoustic_node_detections.csv",
                "acoustic_cue_tracks.csv",
                "acoustic_phase_metrics.csv",
                "acoustic_product_schema.json",
            ],
            "not_allowed_to_models": [
                "records.csv generator metadata",
                "split_manifest.csv split keys",
                "restricted_truth/*.json",
                "kinematics_audit.csv",
                "speed_prior_manifest.json",
                "calibration_report.json",
                "calibration_anchor_manifest.json",
                "calibration_distance.csv",
                "acoustic products unless explicitly wired by a later fusion lane",
                "phase_id", "scenario_seed", "object_seed", "class_id", "target_family",
                "scene_role", "altitude_m", "link_budget_snr_db", "raw_rcs_dbsm",
                "true_speed_mps", "source metadata",
            ],
        },
        "notes": "radial_velocity_mps is a radar observable. True speed and altitude are restricted truth and are not frame features.",
    })
    write_json(out_root / "speed_prior_manifest.json", speed_prior_manifest_payload())
    write_json(out_root / "science_assumptions.json", {
        "validation_tier": "unvalidated/basic synthetic public-proxy benchmark",
        "claim_boundary": "No measured-target truth, classified fidelity, or proprietary-equivalent sensor behavior is claimed.",
        "three_tier_detection_model": {
            "initial_take_up": "0-30 s, low altitude and geometry limited; low Pd is allowed when LOS is masked.",
            "climb_transition": "30-90 s, track initiation and confirmation under changing aspect.",
            "cruise_altitude": "90+ s, coherent Doppler and micro-Doppler classification interval.",
        },
        "public_speed_priors": speed_prior_manifest_payload()["policy"],
        "speed_prior_manifest": "speed_prior_manifest.json",
        "kinematics_audit": "kinematics_audit.csv",
        "calibration_anchor_manifest": "calibration_anchor_manifest.json",
        "calibration_report": "calibration_report.json",
        "calibration_distance": "calibration_distance.csv",
        "calibration_policy": "Reference-only public-proxy distribution audit. Presence of these files does not promote the smoke benchmark beyond unvalidated/basic.",
        "acoustic_cueing_products": {
            "node_detections": "acoustic_node_detections.csv",
            "cue_tracks": "acoustic_cue_tracks.csv",
            "phase_metrics": "acoustic_phase_metrics.csv",
            "schema": "acoustic_product_schema.json",
            "quality": "acoustic_cue_quality.json",
            "policy": "Synthetic passive acoustic cue summaries only; no raw audio is stored. Fusion with radar belongs to a later fusion lane.",
        },
        "radar_equation_budget": "SNR emerges from power, gains, wavelength, RCS, range^4 loss, propagation loss, clutter loss, receiver noise, bandwidth, processing gain, and system loss. target_snr_db is not a current generation input.",
        "smoke_limitations": [
            "current smoke detector products still include proxy micro-Doppler and spectrum summaries until the complex-IQ end-to-end work slot lands.",
            "Target-masked counterfactuals share background clutter/RFI products with no-target counterfactuals and retain withheld target truth only in restricted_truth.",
            "Calibration-anchor artifacts use reference target distributions and synthetic observations only; lawful measured-anchor distributions are required before any measured-anchored current claim.",
            "Acoustic cue products are generated sidecars and do not claim radar-fused track quality.",
        ],
    })
    write_json(out_root / "dataset_card.json", {
        "dataset_id": f"shahed136-public-proxy-ml-training-{args.scale_name}",
        "benchmark_profile": "ml-training-three-tier",
        "scenario_groups": int(args.scenario_groups),
        "record_count": int(len(records)),
        "phase_ids": [phase.phase_id for phase in phases],
        "quality_report": "quality_report.json",
        "operational_metrics": "phase_operational_metrics.csv",
        "kinematics_audit": "kinematics_audit.csv",
        "speed_prior_manifest": "speed_prior_manifest.json",
        "calibration_anchor_manifest": "calibration_anchor_manifest.json",
        "calibration_report": "calibration_report.json",
        "calibration_distance": "calibration_distance.csv",
        "acoustic_node_detections": "acoustic_node_detections.csv",
        "acoustic_cue_tracks": "acoustic_cue_tracks.csv",
        "acoustic_phase_metrics": "acoustic_phase_metrics.csv",
        "acoustic_product_schema": "acoustic_product_schema.json",
        "acoustic_cue_quality": "acoustic_cue_quality.json",
        "counterfactual_audit": "negative_control_audit.json",
        "strict_open_boundary": "Synthetic public-proxy benchmark; no measured traces or proprietary-equivalent behavior claims.",
    })
    write_json(out_root / "dataset_manifest.json", {
        "dataset_id": f"shahed136-public-proxy-ml-training-{args.scale_name}",
        "benchmark_profile": "ml-training-three-tier",
        "files": [
            "records.csv", "split_manifest.csv", "frame_features.npz", "valid_frame_mask.npz",
            "features.csv", "kinematics_audit.csv", "speed_prior_manifest.json",
            "calibration_anchor_manifest.json", "calibration_report.json",
            "calibration_distance.csv", "acoustic_node_detections.csv",
            "acoustic_cue_tracks.csv", "acoustic_phase_metrics.csv",
            "acoustic_product_schema.json", "acoustic_cue_quality.json",
            "phase_operational_metrics.csv", "quality_report.json",
            "negative_control_audit.json", "generator_truth_denylist.json",
            "feature_schema.json", "science_assumptions.json", "dataset_card.json",
        ],
    })
    write_json(out_root / "runtime_report.json", {
        "generator": "detection/generate_ml_training.py",
        "seed": args.seed,
        "scale_name": args.scale_name,
        "quality_status": quality["status"],
        "generated_artifact_policy": "outputs/ is gitignored; do not stage generated records, tensors, or reports.",
    })
