"""Catalog loading and validation for public measured-anchor metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_DATASET_FIELDS = (
    "dataset_id",
    "display_name",
    "source_url",
    "license_status",
    "access_status",
    "product_level",
    "radar_band_modality",
    "target_classes",
    "confuser_classes",
    "phase_relevance",
    "retrieval",
    "adapter_name",
    "allowed_claim_level",
    "claim_boundary",
)


class CatalogError(ValueError):
    """Raised when catalog metadata violates the strict-open registry contract."""


def default_catalog_path(repo_root: Path | None = None) -> Path:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return root / "detection" / "real_data" / "catalog.json"


def load_catalog(path: Path | str | None = None) -> dict[str, Any]:
    catalog_path = default_catalog_path() if path is None else Path(path)
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _is_absolute_path_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("/") or value.startswith("~") or ":\\" in value


def validate_dataset_entry(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_DATASET_FIELDS if field not in entry]
    if missing:
        errors.append(f"{entry.get('dataset_id', '<unknown>')}: missing fields {missing}")
    dataset_id = str(entry.get("dataset_id", ""))
    if not dataset_id or any(ch.isspace() for ch in dataset_id):
        errors.append(f"{dataset_id or '<unknown>'}: dataset_id must be a non-empty slug")
    source_url = str(entry.get("source_url", ""))
    if not source_url.startswith(("https://", "http://", "doi:")):
        errors.append(f"{dataset_id}: source_url must be http(s) or doi")
    retrieval = entry.get("retrieval", {})
    if not isinstance(retrieval, dict) or "method" not in retrieval:
        errors.append(f"{dataset_id}: retrieval must include method")
    if _is_absolute_path_like(entry.get("local_path", "")):
        errors.append(f"{dataset_id}: catalog must not contain local absolute paths")
    if _is_absolute_path_like(retrieval.get("local_path", "")):
        errors.append(f"{dataset_id}: retrieval must not contain local absolute paths")
    if entry.get("allowed_claim_level") in {
        "measured_truth",
        "proprietary_equivalent",
        "classified_fidelity",
    }:
        errors.append(f"{dataset_id}: allowed_claim_level violates claim boundary")
    return errors


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append("datasets must be a non-empty list")
        return errors
    seen: set[str] = set()
    for entry in datasets:
        if not isinstance(entry, dict):
            errors.append("dataset entries must be objects")
            continue
        dataset_id = str(entry.get("dataset_id", ""))
        if dataset_id in seen:
            errors.append(f"{dataset_id}: duplicate dataset_id")
        seen.add(dataset_id)
        errors.extend(validate_dataset_entry(entry))
    return errors


def require_valid_catalog(catalog: dict[str, Any]) -> None:
    errors = validate_catalog(catalog)
    if errors:
        raise CatalogError("; ".join(errors))
