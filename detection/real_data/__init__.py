"""Real-data measured-anchor utilities for detection science lanes."""

from __future__ import annotations

from .adapters import build_dataset_report, default_raw_root, dry_run_fetch
from .catalog import load_catalog, validate_catalog
from .feature_policy import observable_only_violations
from .realism_gate import evaluate_realism_gate

__all__ = [
    "build_dataset_report",
    "default_raw_root",
    "dry_run_fetch",
    "evaluate_realism_gate",
    "load_catalog",
    "observable_only_violations",
    "validate_catalog",
]
