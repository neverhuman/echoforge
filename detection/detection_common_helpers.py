"""Shared helper functions for detection utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .detection_common_types import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_ROOT,
    FRAME_PERIOD_S,
    SINGLE_FEATURE_AUC_GATE,
    TRUTH_LIKE_FRAME_COLUMNS,
)


def horizon_label(horizon_s: int) -> str:
    labels = {5: "ultra_early_5s", 15: "mid_detection_15s", 45: "typical_detection_45s"}
    return labels.get(horizon_s, f"horizon_{horizon_s}s")


def parse_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def load_records(data_root: Path = Path(DEFAULT_DATA_ROOT), seed: int = 42, max_records: int | None = None) -> pd.DataFrame:
    records = pd.read_csv(data_root / "records.csv")
    splits = pd.read_csv(data_root / "split_manifest.csv", usecols=["record_id", "split"])
    records = records.drop(columns=["split"], errors="ignore").merge(splits, on="record_id", validate="one_to_one")
    records["label"] = records["is_public_proxy_positive"].astype(bool).astype(np.int64)
    if max_records is not None and max_records < len(records):
        rng = np.random.default_rng(seed)
        sampled = []
        remaining = max_records
        for split, fraction in [("train", 0.70), ("validation", 0.15), ("test", 0.15)]:
            group = records[records["split"] == split]
            take = remaining if split == "test" else int(round(max_records * fraction))
            if split != "test":
                remaining -= take
            take = max(1, min(take, len(group)))
            sampled.append(records.loc[np.sort(rng.choice(group.index.to_numpy(), size=take, replace=False))])
        records = pd.concat(sampled, axis=0).sort_values(["split", "record_id"]).reset_index(drop=True)
    return records.reset_index(drop=True)


def load_frame_store(data_root: Path = Path(DEFAULT_DATA_ROOT)) -> tuple[np.ndarray, list[str], dict[str, int]]:
    payload = np.load(data_root / "frame_features.npz", allow_pickle=False)
    frames = payload["frames"].astype(np.float32, copy=False)
    record_ids = payload["record_ids"].astype(str).tolist()
    return frames, payload["frame_columns"].astype(str).tolist(), {rid: idx for idx, rid in enumerate(record_ids)}


def select_frames(data_root: Path = Path(DEFAULT_DATA_ROOT), records: pd.DataFrame | None = None) -> tuple[np.ndarray, list[str]]:
    frames, columns, index = load_frame_store(data_root)
    if records is None:
        return frames, columns
    take = [index[str(rid)] for rid in records["record_id"].astype(str)]
    return frames[np.asarray(take, dtype=np.int64)], columns


def horizon_name(horizon_s: int) -> str:
    """Convert horizon seconds to a standardized name."""
    if horizon_s == 5:
        return "ultra_early_5s"
    elif horizon_s == 15:
        return "mid_detection_15s"
    elif horizon_s == 45:
        return "typical_detection_45s"
    else:
        return f"horizon_{horizon_s}s"


def ensure_output_root(out_root: Path = Path(DEFAULT_OUT_ROOT)) -> Path:
    """Ensure output root directory exists."""
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


__all__ = [
    "horizon_label",
    "parse_horizons",
    "load_records",
    "load_frame_store",
    "select_frames",
    "horizon_name",
    "ensure_output_root",
]