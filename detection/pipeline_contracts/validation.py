"""Input and pipeline-spec validation helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .core import MAX_PIPELINE_WORKERS, MissingInputKindError, PipelineError, PipelineSpec


REQUIRED_DATASET_FILES = (
    "records.csv",
    "features.csv",
    "dataset_manifest.json",
    "split_manifest.csv",
)


def validate_dataset_inputs(data_root: Path) -> dict[str, Path]:
    resolved = data_root.resolve()
    if not resolved.exists():
        raise PipelineError(f"data root does not exist: {resolved}", code="missing_data_root")

    inputs: dict[str, Path] = {}
    for name in REQUIRED_DATASET_FILES:
        path = resolved / name
        if not path.exists():
            raise PipelineError(f"missing required dataset input: {name}", code="missing_dataset_input")
        inputs[name] = path
    return inputs


def validate_pipeline_spec(spec: PipelineSpec) -> None:
    if not spec.id or not spec.version or not spec.title:
        raise PipelineError("pipeline spec must define id, version, and title", code="invalid_spec")
    if spec.worker_budget > MAX_PIPELINE_WORKERS:
        raise PipelineError(
            f"pipeline worker budget {spec.worker_budget} exceeds cap {MAX_PIPELINE_WORKERS}",
            code="invalid_spec",
        )
    if not spec.input_contract or not spec.output_contract:
        raise PipelineError("pipeline spec must declare input and output contracts", code="invalid_spec")
    if not spec.references:
        raise PipelineError("pipeline spec must declare at least one reference", code="invalid_spec")
