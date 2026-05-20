"""Manual real-data adapter scaffolding.

Adapters are intentionally explicit: this lane never downloads or normalizes
measured data as a side effect of CI. Local data is opt-in under the configured
cache root, and missing data emits reference-only reports.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


OBSERVATION_FILENAME = "observations.csv"
CALIBRATION_FIELDS = [
    "dataset_id",
    "feature",
    "metric",
    "measured_value",
    "reference_min",
    "reference_max",
    "distance",
    "status",
    "allowed_claim_level",
]


@dataclass(frozen=True)
class DatasetReport:
    dataset_id: str
    status: str
    output_dir: Path
    message: str

    def to_json(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "output_dir": self.output_dir.as_posix(),
            "message": self.message,
        }


def default_raw_root() -> Path:
    configured = os.environ.get("ECHOFORGE_REAL_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "echoforge" / "real-data"


def dataset_root(raw_root: Path, dataset_id: str) -> Path:
    return raw_root.expanduser() / dataset_id


def derived_output_dir(out_root: Path, dataset_id: str, run_id: str) -> Path:
    return out_root / dataset_id / run_id


def dry_run_fetch(entry: dict[str, Any], raw_root: Path | None = None) -> dict[str, Any]:
    root = default_raw_root() if raw_root is None else Path(raw_root)
    local_root = dataset_root(root, str(entry["dataset_id"]))
    retrieval = dict(entry.get("retrieval", {}))
    return {
        "dataset_id": entry["dataset_id"],
        "status": "available_local" if local_root.exists() else "manual_required",
        "local_root": local_root.as_posix(),
        "retrieval_method": retrieval.get("method", "manual"),
        "source_url": retrieval.get("url", entry.get("source_url", "")),
        "network_side_effects": "none",
        "message": "Dry run only; no files were downloaded or written.",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALIBRATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CALIBRATION_FIELDS})


def _reference_only_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in entry.get("reference_features", ["unspecified"]):
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "feature": feature,
                "metric": "distribution_distance",
                "measured_value": "",
                "reference_min": "",
                "reference_max": "",
                "distance": "",
                "status": "reference_only",
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
    return rows


def _read_observations(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "feature" not in (reader.fieldnames or []) or "value" not in (reader.fieldnames or []):
            raise ValueError("observations.csv must contain feature,value columns")
        grouped: dict[str, list[float]] = {}
        for row in reader:
            feature = str(row.get("feature", "")).strip()
            if not feature:
                continue
            grouped.setdefault(feature, []).append(float(row["value"]))
    return grouped


def _summarize_observations(grouped: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature, values in sorted(grouped.items()):
        if not values:
            continue
        summary[feature] = {
            "count": float(len(values)),
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def _candidate_distance_rows(
    entry: dict[str, Any], summary: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    bounds = entry.get("reference_feature_bounds", {})
    rows: list[dict[str, Any]] = []
    for feature, stats in summary.items():
        feature_bounds = bounds.get(feature, {}) if isinstance(bounds, dict) else {}
        lower = feature_bounds.get("mean_min")
        upper = feature_bounds.get("mean_max")
        measured = stats["mean"]
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            distance = (
                0.0
                if lower <= measured <= upper
                else min(abs(measured - lower), abs(measured - upper))
            )
            status = "candidate_pass" if distance == 0.0 else "candidate_gap"
        else:
            distance = ""
            status = "measured_anchor_candidate_no_reference_bounds"
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "feature": feature,
                "metric": "mean_distribution_distance",
                "measured_value": measured,
                "reference_min": lower if lower is not None else "",
                "reference_max": upper if upper is not None else "",
                "distance": distance,
                "status": status,
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
    return rows


def build_dataset_report(
    entry: dict[str, Any],
    *,
    raw_root: Path | None = None,
    out_root: Path = Path("outputs/real-data"),
    run_id: str = "reference-only",
) -> DatasetReport:
    root = default_raw_root() if raw_root is None else Path(raw_root)
    local_root = dataset_root(root, str(entry["dataset_id"]))
    output_dir = derived_output_dir(Path(out_root), str(entry["dataset_id"]), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = local_root / OBSERVATION_FILENAME

    manifest = {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "source_url": entry["source_url"],
        "adapter_name": entry["adapter_name"],
        "allowed_claim_level": entry["allowed_claim_level"],
        "claim_boundary": entry["claim_boundary"],
        "local_root": local_root.as_posix(),
        "raw_data_tracked_in_git": False,
        "normalized_samples_tracked_in_git": False,
    }

    if not observations_path.exists():
        manifest.update(
            {
                "calibration_anchor_status": "reference_only",
                "local_status": "not_available",
                "message": "No local observations.csv was found; emitted reference-only metadata.",
            }
        )
        _write_json(output_dir / "anchor_manifest.json", manifest)
        _write_json(
            output_dir / "feature_distributions.json",
            {
                "dataset_id": entry["dataset_id"],
                "status": "reference_only",
                "features": {},
            },
        )
        _write_csv(output_dir / "calibration_distance.csv", _reference_only_rows(entry))
        return DatasetReport(
            dataset_id=str(entry["dataset_id"]),
            status="reference_only",
            output_dir=output_dir,
            message="local measured anchor not available",
        )

    grouped = _read_observations(observations_path)
    summary = _summarize_observations(grouped)
    manifest.update(
        {
            "calibration_anchor_status": "measured_anchor_candidate",
            "local_status": "observations_present",
            "message": "Local observations were summarized; no measured traces were copied.",
        }
    )
    _write_json(output_dir / "anchor_manifest.json", manifest)
    _write_json(
        output_dir / "feature_distributions.json",
        {
            "dataset_id": entry["dataset_id"],
            "status": "measured_anchor_candidate",
            "features": summary,
        },
    )
    _write_csv(output_dir / "calibration_distance.csv", _candidate_distance_rows(entry, summary))
    return DatasetReport(
        dataset_id=str(entry["dataset_id"]),
        status="measured_anchor_candidate",
        output_dir=output_dir,
        message="local measured observations summarized",
    )
