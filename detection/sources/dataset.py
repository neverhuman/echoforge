"""CSV and manifest readers for EchoForge generated datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

from detection.pipeline_contracts.core import load_json


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset_manifest(data_root: Path) -> dict[str, Any]:
    return load_json(data_root / "dataset_manifest.json")


def load_records(data_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(data_root / "records.csv")


def load_features(data_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(data_root / "features.csv")


def load_split_manifest(data_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(data_root / "split_manifest.csv")
