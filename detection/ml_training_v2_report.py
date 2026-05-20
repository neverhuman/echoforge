"""Reporting and output utilities for the v2 benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from detection.ml_training_v2_config import (
        FRAME_COUNT,
        FRAME_COLUMNS,
        FRAME_PERIOD_S,
        ScenarioStratum,
    )
    from detection.ml_training_v2_real_priors import prior_manifest
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from ml_training_v2_config import (
        FRAME_COUNT,
        FRAME_COLUMNS,
        FRAME_PERIOD_S,
        ScenarioStratum,
    )
    from ml_training_v2_real_priors import prior_manifest


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(
    out_root: Path,
    records: pd.DataFrame,
    strata: list[ScenarioStratum],
    feature_names: list[str],
    quality: dict[str, Any],
    scale_name: str,
    seed: int,
    real_anchor_priors: dict[str, Any] | None = None,
) -> None:
    strata_df = pd.DataFrame([asdict(stratum) for stratum in strata])
    strata_df.to_csv(out_root / "scenario_strata.csv", index=False)
    split_manifest = records[
        [
            "record_id",
            "split",
            "scenario_seed",
            "object_seed",
            "class_id",
            "target_family",
            "hard_negative_family",
            "stratum_id",
            "difficulty_bucket",
            "confuser_family",
            "holdout_role",
        ]
    ].copy()
    split_manifest.insert(2, "split_key_kind", "stratum_confuser_holdout")
    split_manifest.insert(
        3,
        "split_key",
        records["stratum_id"].astype(str)
        + ":"
        + records["target_family"].astype(str)
        + ":"
        + records["holdout_role"].astype(str),
    )
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)

    feature_schema = {
        "benchmark_version": "ml-training-v2",
        "frame_period_s": FRAME_PERIOD_S,
        "frame_count": FRAME_COUNT,
        "frame_columns": FRAME_COLUMNS,
        "aggregate_feature_columns": feature_names,
        "sensor_public_contract": {
            "allowed_to_models": [
                "frame_features.npz frame tensor columns",
                "features.csv aggregate frame observables",
            ],
            "not_allowed_to_models": [
                "records.csv generator metadata",
                "split_manifest.csv split keys",
                "anchor_observations.csv",
                "calibration_coefficients.json",
                "real_anchor_priors_manifest.json",
                "nominal_snr_db",
                "link_budget_snr_db",
                "propagation_loss_db",
                "clutter_loss_db",
                "altitude_m",
                "micro_doppler_peak_hz_proxy",
                "micro_doppler_bandwidth_hz_proxy",
                "normalized_snr",
                "class_id",
                "target_family",
                "confuser_family",
                "scenario_seed",
                "object_seed",
                "holdout_role",
            ],
        },
        "restricted_metadata_columns": [
            "class_id",
            "target_family",
            "is_public_proxy_positive",
            "is_hard_negative",
            "hard_negative_family",
            "confuser_family",
            "split",
            "holdout_role",
            "nominal_snr_db",
            "link_budget_snr_db",
            "propagation_loss_db",
            "clutter_loss_db",
            "altitude_m",
        ],
        "notes": "Frame products are synthetic public-proxy observables with overlapping target/confuser envelopes; metadata columns are for audit and slicing only.",
    }
    (out_root / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True) + "\n"
    )
    label_schema = {
        "positive_label": "public_proxy_fixed_wing",
        "negative_label": "scenario_confuser_or_sensor_artifact",
        "claim_boundary": "Labels describe synthetic public-proxy benchmark roles, not measured-object truth or proprietary-equivalent sensor behavior.",
    }
    (out_root / "label_schema.json").write_text(
        json.dumps(label_schema, indent=2, sort_keys=True) + "\n"
    )
    calibration_sources = [
        {
            "id": "scientific-data-2026-drone-radar-rf",
            "title": "Time-synchronized multi-sensor drone radar/RF dataset",
            "url": "https://www.nature.com/articles/s41597-026-06802-6",
            "role": "distribution_anchor_metadata_only",
            "local_data_default": "not_vendored",
            "license_notes": "No source traces copied; local operators must review terms before using external data.",
        },
        {
            "id": "rahman-robertson-drone-bird-micro-doppler",
            "title": "Radar micro-Doppler signatures of drones and birds",
            "url": "https://research-repository.st-andrews.ac.uk/handle/10023/16577",
            "role": "micro_doppler_range_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
        {
            "id": "eusipco-2020-micro-doppler-representations",
            "title": "Comparison of micro-Doppler signal representations",
            "url": "https://eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001561.pdf",
            "role": "representation_family_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
        {
            "id": "low-grazing-uav-detection-cfar-micro-doppler",
            "title": "Low-grazing UAV detection literature on CFAR, clutter, and trajectory extraction",
            "url": "https://arxiv.org/abs/1902.05483",
            "role": "cfar_tbd_failure_mode_context",
            "local_data_default": "not_vendored",
            "license_notes": "Publication metadata only.",
        },
    ]
    (out_root / "external_calibration_sources.json").write_text(
        json.dumps(calibration_sources, indent=2, sort_keys=True) + "\n"
    )
    real_anchor_manifest = prior_manifest(real_anchor_priors)
    (out_root / "real_anchor_priors_manifest.json").write_text(
        json.dumps(real_anchor_manifest, indent=2, sort_keys=True) + "\n"
    )
    calibration_targets = {
        "validation_tier": {
            "target": "V0/V1 public-proxy simulation with explicit assumptions; not measured-truth validation.",
            "check_file": "science_assumptions.json",
        },
        "link_budget": {
            "target": "SNR diagnostics derive from a transparent monostatic radar-equation budget plus public-proxy scene losses; SNR diagnostics are not model features.",
            "check_file": "records.csv",
        },
        "snr_db": {
            "target": "Overlapping easy/medium/hard/barely-visible target and confuser ranges; not fitted to measured truth.",
            "check_file": "distribution_overlap.csv",
        },
        "micro_doppler_energy": {
            "target": "Fixed-wing, RC fixed-wing, bird/flock, turbine, weather, and RFI envelopes overlap in aggregate energy and bandwidth.",
            "check_file": "single_feature_auc_audit.csv",
        },
        "clutter_and_impairments": {
            "target": "Includes Weibull/K-like clutter pressure, glints, vegetation/weather motion, RFI, AGC, dropped CPIs, ambiguity, folding, quantization, drift, and calibration offsets.",
            "check_file": "scenario_strata.csv",
        },
    }
    (out_root / "calibration_targets.json").write_text(
        json.dumps(calibration_targets, indent=2, sort_keys=True) + "\n"
    )
    science_assumptions = {
        "validation_tier": "V0/V1 synthetic public-proxy benchmark",
        "claim_boundary": "No measured-target truth, classified fidelity, or proprietary-equivalent sensor behavior is claimed.",
        "radar_equation_budget": {
            "frequency_source": "sensor_band public-proxy center frequency",
            "mode": "monostatic first-order received-power diagnostic",
            "assumed_peak_power_dbw": 50.0,
            "assumed_tx_gain_dbi": 28.0,
            "assumed_rx_gain_dbi": 28.0,
            "assumed_bandwidth_hz": 2.0e6,
            "assumed_noise_figure_db": 5.5,
            "assumed_system_temperature_k": 290.0,
            "assumed_unmodeled_system_loss_db": 45.0,
            "limitations": [
                "No measured antenna pattern",
                "No complex-IQ coherent propagation",
                "No measured clutter map",
                "No validated target RCS table",
            ],
        },
        "horizon_semantics": "Features are causal prefixes with frame time_s <= horizon_s. The benchmark is an accumulated-evidence detector, not a claim of future-event prediction before a measured launch event.",
        "leakage_controls": [
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "per_stratum_split_balance.csv",
        ],
        "deferred_physics": [
            "Complex IQ radar cube",
            "Aspect/frequency/polarization RCS tables",
            "Terrain mesh and two-ray/multipath propagation",
            "Full measured-data V5 holdout",
            "Multi-scan tracker with association and false-track lifecycle",
        ],
        "measured_anchor_hardening": {
            "status": real_anchor_manifest["status"],
            "manifest": "real_anchor_priors_manifest.json",
            "policy": real_anchor_manifest["policy"],
        },
    }
    (out_root / "science_assumptions.json").write_text(
        json.dumps(science_assumptions, indent=2, sort_keys=True) + "\n"
    )
    dataset_card = {
        "dataset_id": f"shahed136-public-proxy-ml-training-v2-{scale_name}",
        "benchmark_version": "ml-training-v2",
        "record_count": int(len(records)),
        "scenario_strata": len(strata),
        "seed": seed,
        "validation_tier": "V0/V1 synthetic public-proxy",
        "strict_open_boundary": "Synthetic public-proxy benchmark; no measured traces, classified fidelity, or proprietary-equivalent behavior claims.",
        "difficulty_policy": "Fifty scenario strata cover sensor band, range, grazing angle, clutter, aspect, motion, interference, confuser family, and visibility bucket combinations.",
        "quality_report": "quality_report.json",
        "real_anchor_priors": "real_anchor_priors_manifest.json",
        "diagnostics": [
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "micro_doppler_saturation_guard.json",
            "per_stratum_split_balance.csv",
            "distribution_overlap.csv",
        ],
    }
    (out_root / "dataset_card.json").write_text(
        json.dumps(dataset_card, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "dataset_id": dataset_card["dataset_id"],
        "benchmark_version": "ml-training-v2",
        "files": [
            "records.csv",
            "split_manifest.csv",
            "frame_features.npz",
            "features.csv",
            "scenario_strata.csv",
            "quality_report.json",
            "single_feature_auc_audit.csv",
            "label_leakage_audit.json",
            "negative_control_audit.json",
            "micro_doppler_saturation_guard.json",
            "per_stratum_split_balance.csv",
            "distribution_overlap.csv",
            "feature_schema.json",
            "label_schema.json",
            "dataset_card.json",
            "external_calibration_sources.json",
            "calibration_targets.json",
            "science_assumptions.json",
            "runtime_report.json",
            "real_anchor_priors_manifest.json",
        ],
    }
    (out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    runtime_report = {
        "generator": "detection/generate_ml_training_v2.py",
        "seed": seed,
        "scale_name": scale_name,
        "quality_status": quality["status"],
        "real_anchor_priors_status": real_anchor_manifest["status"],
        "real_anchor_priors_sha256": real_anchor_manifest["source_sha256"],
        "generated_artifact_policy": "outputs/ is gitignored; do not stage generated records, tensors, or model outputs.",
    }
    (out_root / "runtime_report.json").write_text(
        json.dumps(runtime_report, indent=2, sort_keys=True) + "\n"
    )
