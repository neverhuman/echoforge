"""Shared contracts for the EchoForge ML pipeline system."""

from .core import (
    MAX_PIPELINE_WORKERS,
    MAX_SUITE_CONCURRENCY,
    ArtifactRecord,
    PipelineContext,
    PipelineResult,
    PipelineSpec,
    PipelineError,
    MissingInputKindError,
    default_thread_env,
    ensure_within_root,
    json_dump,
    load_json,
    write_csv,
    write_json,
)
from .metrics import calibration_bins, pr_auc, roc_auc, threshold_at_precision
from .validation import validate_dataset_inputs, validate_pipeline_spec
