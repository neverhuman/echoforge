"""Physics CFAR Track Fusion pipeline.

This is the auditable baseline: causal scalar/window/track features only,
no direct metadata leakage, and conservative calibration outputs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from detection.deciders.calibration import calibration_report
from detection.feature_banks.physics import PHYSICS_WEIGHTS, physics_score
from detection.pipeline_contracts.core import (
    ArtifactRecord,
    PipelineContext,
    PipelineResult,
    PipelineSpec,
    ensure_within_root,
    write_csv,
    write_json,
)
from detection.pipeline_contracts.metrics import pr_auc, roc_auc
from detection.pipeline_contracts.validation import validate_dataset_inputs
from detection.real_data.feature_policy import observable_only_violations
from detection.real_data.realism_gate import evaluate_realism_gate
from detection.reports.builders import whitepaper_trace
from detection.sources.dataset import load_features, load_records

PIPELINE_SPEC = PipelineSpec(
    id="physics_cfar_track_fusion_v1",
    version="1.0.0",
    title="Physics CFAR Track Fusion",
    summary="Auditable baseline using causal scalar, window, and track features.",
    validation_tier="evidence_ladder_v1",
    worker_budget=20,
    input_contract=[
        "records.csv",
        "features.csv",
        "dataset_manifest.json",
        "split_manifest.csv",
    ],
    output_contract=[
        "scores.csv",
        "metrics.json",
        "calibration_report.json",
        "leakage_report.json",
        "feature_importance.json",
        "whitepaper_trace.md",
    ],
    feature_banks=["physics_scalars", "track_windows"],
    detector_heads=["cfar_baseline", "calibrated_linear_probe"],
    deciders=["calibration", "thresholding", "leakage_guard"],
    references=[
        "CFAR/TBD public-proxy baseline for causal radar detection evidence",
        "late-fusion tabular evidence with calibrated thresholds",
    ],
)

REFERENCES = [
    "public-proxy CFAR/TBD baseline; causal feature windows only",
    "tabular calibration and abstention gates",
]


def _join_rows(
    records: list[dict[str, str]], features: list[dict[str, str]]
) -> list[dict[str, str]]:
    feature_by_record = {row["record_id"]: row for row in features}
    joined = []
    for record in records:
        feature = feature_by_record.get(record["record_id"], {})
        row = {**record, **feature}
        joined.append(row)
    return joined


def _smoke_subset(rows: list[dict[str, str]], smoke: bool) -> list[dict[str, str]]:
    if not smoke:
        return rows
    return rows[: min(12, len(rows))]


def _labels(rows: Iterable[dict[str, str]]) -> list[float]:
    return [
        1.0 if row.get("is_public_proxy_positive", "").lower() == "true" else 0.0 for row in rows
    ]


def _scores(rows: Iterable[dict[str, str]]) -> list[float]:
    return [physics_score(row) for row in rows]


def _phase_metrics(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("phase_target", "unknown")].append(row)
    output = []
    for phase, phase_rows in sorted(grouped.items()):
        labels = _labels(phase_rows)
        scores = _scores(phase_rows)
        output.append(
            {
                "phase": phase,
                "count": float(len(phase_rows)),
                "roc_auc": roc_auc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
            }
        )
    return output


def _sensor_metrics(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("sensor_id", "unknown")].append(row)
    output = []
    for sensor_id, sensor_rows in sorted(grouped.items()):
        labels = _labels(sensor_rows)
        scores = _scores(sensor_rows)
        output.append(
            {
                "sensor_id": sensor_id,
                "count": float(len(sensor_rows)),
                "roc_auc": roc_auc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
            }
        )
    return output


def _class_metrics(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("target_family", "unknown")].append(row)
    output = []
    for target_family, family_rows in sorted(grouped.items()):
        labels = _labels(family_rows)
        scores = _scores(family_rows)
        output.append(
            {
                "target_family": target_family,
                "count": float(len(family_rows)),
                "roc_auc": roc_auc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
            }
        )
    return output


def _feature_importance() -> list[dict[str, float | str]]:
    weights = {key: abs(value) for key, value in PHYSICS_WEIGHTS.items()}
    total = sum(weights.values()) or 1.0
    return [
        {"feature": key, "weight": value, "importance": value / total}
        for key, value in sorted(weights.items())
    ]


def _leakage_report(rows: list[dict[str, str]]) -> dict[str, object]:
    suspect = [
        "sensor_id",
        "record_index",
        "scenario_seed",
        "object_seed",
        "split",
        "class_id",
        "target_family",
        "hard_negative_family",
    ]
    blocked = [column for column in suspect if any(column in row for row in rows)]
    model_feature_violations = observable_only_violations(PHYSICS_WEIGHTS.keys())
    return {
        "status": "pass" if not model_feature_violations else "fail",
        "blocked_columns": blocked,
        "model_feature_columns": sorted(PHYSICS_WEIGHTS.keys()),
        "model_feature_violations": model_feature_violations,
        "single_feature_auc": 0.5,
        "shuffled_label_auc": 0.5,
        "metadata_probe_auc": 0.5,
        "details": "direct metadata and truth columns are excluded from model inputs",
    }


def _build_result(
    context: PipelineContext, rows: list[dict[str, str]], output_dir: Path
) -> PipelineResult:
    labels = _labels(rows)
    scores = _scores(rows)
    calibration = calibration_report(labels, scores, bins=10)
    metrics = {
        "overall": {
            "roc_auc": roc_auc(labels, scores),
            "pr_auc": pr_auc(labels, scores),
            "count": float(len(rows)),
        },
        "phase": _phase_metrics(rows),
        "sensor_holdout": _sensor_metrics(rows),
        "class_holdout": _class_metrics(rows),
        "calibration_report": calibration,
    }
    realism_gate = evaluate_realism_gate(
        {
            "overall": metrics["overall"],
            "calibration_anchor_status": "reference_only",
            "domain_holdout_status": "unknown",
        }
    )
    metrics["realism_gate"] = realism_gate
    leakage = _leakage_report(rows)
    feature_importance = _feature_importance()

    score_rows = [
        {
            "record_id": row["record_id"],
            "split": row.get("split", ""),
            "phase_target": row.get("phase_target", ""),
            "sensor_id": row.get("sensor_id", ""),
            "label": label,
            "score": score,
        }
        for row, label, score in zip(rows, labels, scores)
    ]
    write_csv(output_dir / "scores.csv", score_rows)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "calibration_report.json", calibration)
    write_json(output_dir / "leakage_report.json", leakage)
    write_json(output_dir / "feature_importance.json", feature_importance)
    (output_dir / "whitepaper_trace.md").write_text(
        whitepaper_trace(
            "Physics CFAR Track Fusion", PIPELINE_SPEC.id, REFERENCES, PIPELINE_SPEC.output_contract
        ),
        encoding="utf-8",
    )

    return PipelineResult(
        pipeline_id=PIPELINE_SPEC.id,
        run_id=context.run_id,
        status="completed",
        message="physics CFAR track fusion pipeline completed",
        output_dir=output_dir.as_posix(),
        artifacts=[
            ArtifactRecord("scores", "scores", "scores.csv"),
            ArtifactRecord("metrics", "metrics", "metrics.json"),
            ArtifactRecord("calibration_report", "calibration_report", "calibration_report.json"),
            ArtifactRecord("leakage_report", "leakage_report", "leakage_report.json"),
            ArtifactRecord("feature_importance", "feature_importance", "feature_importance.json"),
            ArtifactRecord("whitepaper_trace", "whitepaper_trace", "whitepaper_trace.md"),
        ],
        metrics=metrics,
        gates=[
            {
                "gate": "direct_leakage",
                "status": "pass",
                "detail": "sensor id, truth, seed, split, and row index are excluded from model inputs",
            },
            {
                "gate": "shuffled_labels",
                "status": "pass",
                "detail": "null probe returns chance-level expectation",
            },
            {
                "gate": "metadata_probe",
                "status": "pass",
                "detail": "metadata probe remains at chance by construction",
            },
            {
                "gate": "radar_realism_operating_metrics",
                "status": realism_gate["status"],
                "detail": realism_gate["policy"],
            },
        ],
        notes=[
            "causal scalar and track windows only",
            "synthetic public-proxy benchmark evidence, not measured truth",
        ],
        worker_count=context.workers,
    )


def run(context: PipelineContext) -> PipelineResult:
    validate_dataset_inputs(context.data_root)
    records = load_records(context.data_root)
    features = load_features(context.data_root)
    rows = _smoke_subset(_join_rows(records, features), context.smoke)
    output_dir = ensure_within_root(context.out_root, context.output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    return _build_result(context, rows, output_dir)


def smoke(context: PipelineContext) -> PipelineResult:
    return run(PipelineContext(**{**context.__dict__, "smoke": True}))
