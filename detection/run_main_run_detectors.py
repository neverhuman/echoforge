#!/usr/bin/env python3
"""Run detector and fusion baselines over a generated main-run corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from detection.main_run_detectors import run_main_run_detectors
    from detection.main_run_types import DEFAULT_FOLDS, DEFAULT_SEED
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_detectors import run_main_run_detectors
    from main_run_types import DEFAULT_FOLDS, DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/training-data/runit-shahed136-main-run-v1"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/detection/runit-shahed136-main-run-v1"),
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = run_main_run_detectors(
        args.data_root,
        args.out_root,
        folds=args.folds,
        seed=args.seed,
        force=args.force,
    )
    print(
        f"wrote {args.out_root} records={quality['record_count']} status={quality['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
