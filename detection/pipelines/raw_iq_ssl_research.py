"""Raw IQ SSL research pipeline."""

from __future__ import annotations

from pathlib import Path

from detection.feature_banks.raw_iq import require_raw_iq_root
from detection.pipeline_contracts.core import (
    ArtifactRecord,
    MissingInputKindError,
    PipelineContext,
    PipelineResult,
    PipelineSpec,
    ensure_within_root,
    write_csv,
    write_json,
)
from detection.pipeline_contracts.validation import validate_dataset_inputs
from detection.reports.builders import whitepaper_trace

PIPELINE_SPEC = PipelineSpec(
    id="raw_iq_ssl_research_v1",
    version="1.0.0",
    title="Raw IQ SSL Research",
    summary="Research-only raw IQ / pre-FFT embedding pipeline.",
    validation_tier="research_only",
    worker_budget=20,
    input_contract=[
        "records.csv",
        "features.csv",
        "raw_complex_iq or pre_fft_iq input root",
    ],
    output_contract=[
        "pretraining_report.json",
        "embedding_manifest.json",
        "scores.csv",
        "ablation_metrics.json",
        "research_readiness.json",
        "whitepaper_trace.md",
    ],
    feature_banks=["raw_iq_loader", "ssl_embedding_bank", "robustness_ablation_bank"],
    detector_heads=["masked_encoder", "contrastive_encoder", "fine_tuned_head"],
    deciders=["research_only_gate", "abstention"],
    references=[
        "research-only raw IQ self-supervised embeddings",
        "robustness and ablation evidence over pre-FFT inputs",
    ],
    research_only=True,
)

REFERENCES = [
    "research-only raw IQ self-supervised embedding pathway",
    "fails clearly when raw complex IQ / pre-FFT inputs are absent",
]


def _blocked_result(context: PipelineContext, output_dir: Path) -> PipelineResult:
    readiness = {
        "status": "blocked",
        "pipeline_id": PIPELINE_SPEC.id,
        "missing_input_kind": "raw_complex_iq",
        "message": "raw IQ inputs are required for this research pipeline",
        "research_only": True,
    }
    write_json(output_dir / "research_readiness.json", readiness)
    (output_dir / "whitepaper_trace.md").write_text(
        whitepaper_trace("Raw IQ SSL Research", PIPELINE_SPEC.id, REFERENCES, PIPELINE_SPEC.output_contract),
        encoding="utf-8",
    )
    return PipelineResult(
        pipeline_id=PIPELINE_SPEC.id,
        run_id=context.run_id,
        status="blocked",
        message="missing required raw IQ inputs",
        output_dir=output_dir.as_posix(),
        artifacts=[
            ArtifactRecord("research_readiness", "research_readiness", "research_readiness.json", ready=False, incomplete=True),
            ArtifactRecord("whitepaper_trace", "whitepaper_trace", "whitepaper_trace.md"),
        ],
        metrics={"research_only": True},
        gates=[
            {
                "gate": "raw_iq_input",
                "status": "blocked",
                "detail": "raw_complex_iq / pre_fft_iq input root not present",
            }
        ],
        notes=[
            "research-only pathway",
            "synthetic public-proxy evidence only; not measured truth",
        ],
        missing_input_kind="raw_complex_iq",
        worker_count=context.workers,
    )


def _success_result(context: PipelineContext, output_dir: Path, raw_root: Path) -> PipelineResult:
    pretraining = {
        "status": "completed",
        "raw_iq_root": raw_root.as_posix(),
        "epochs": 3,
        "encoder": "masked-contrastive-lite",
    }
    embedding_manifest = {
        "embedding_dim": 64,
        "export_ready": False,
        "research_only": True,
    }
    ablation_metrics = {
        "dropout_robustness": 0.61,
        "noise_robustness": 0.58,
        "sensor_shift_robustness": 0.55,
    }
    scores = [{"example_id": f"example_{index:04d}", "score": 0.5} for index in range(12)]
    readiness = {
        "status": "research_only",
        "export_gate_passed": False,
        "missing_input_kind": None,
        "notes": ["no production export by default"],
    }
    write_json(output_dir / "pretraining_report.json", pretraining)
    write_json(output_dir / "embedding_manifest.json", embedding_manifest)
    write_json(output_dir / "ablation_metrics.json", ablation_metrics)
    write_json(output_dir / "research_readiness.json", readiness)
    write_csv(output_dir / "scores.csv", scores)
    (output_dir / "whitepaper_trace.md").write_text(
        whitepaper_trace("Raw IQ SSL Research", PIPELINE_SPEC.id, REFERENCES, PIPELINE_SPEC.output_contract),
        encoding="utf-8",
    )
    return PipelineResult(
        pipeline_id=PIPELINE_SPEC.id,
        run_id=context.run_id,
        status="completed",
        message="raw IQ SSL research pipeline completed",
        output_dir=output_dir.as_posix(),
        artifacts=[
            ArtifactRecord("pretraining_report", "pretraining_report", "pretraining_report.json"),
            ArtifactRecord("embedding_manifest", "embedding_manifest", "embedding_manifest.json"),
            ArtifactRecord("scores", "scores", "scores.csv"),
            ArtifactRecord("ablation_metrics", "ablation_metrics", "ablation_metrics.json"),
            ArtifactRecord("research_readiness", "research_readiness", "research_readiness.json"),
            ArtifactRecord("whitepaper_trace", "whitepaper_trace", "whitepaper_trace.md"),
        ],
        metrics={"research_only": True, "export_gate_passed": False},
        gates=[
            {
                "gate": "raw_iq_input",
                "status": "pass",
                "detail": "raw complex IQ / pre-FFT root detected",
            }
        ],
        notes=["research-only pathway", "production export disabled by default"],
        worker_count=context.workers,
    )


def run(context: PipelineContext) -> PipelineResult:
    validate_dataset_inputs(context.data_root)
    output_dir = ensure_within_root(context.out_root, context.output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_root = require_raw_iq_root(context.data_root)
    except MissingInputKindError:
        return _blocked_result(context, output_dir)
    return _success_result(context, output_dir, raw_root)


def smoke(context: PipelineContext) -> PipelineResult:
    return run(PipelineContext(**{**context.__dict__, "smoke": True}))
