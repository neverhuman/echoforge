#!/usr/bin/env python3
"""Run the separate advanced evolution detector lane for the main-run corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from detection.advanced_main_run_detectors import run_advanced_main_run_detectors
    from detection.main_run_types import DEFAULT_FOLDS, DEFAULT_SEED
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from advanced_main_run_detectors import run_advanced_main_run_detectors
    from main_run_types import DEFAULT_FOLDS, DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/training-data/fixed-wing-pusher-proxy-main-run"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "outputs/detection/fixed-wing-pusher-proxy-main-run-advanced-evolution"
        ),
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--evolution-rounds", type=int, default=5)
    parser.add_argument(
        "--search-profile",
        choices=("smoke", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional generated NumPy feature cache path under outputs/.",
    )
    parser.add_argument(
        "--selection-lock",
        type=Path,
        default=None,
        help="Optional selection-lock JSON path for locked diagnostics.",
    )
    parser.add_argument(
        "--score-locked-only",
        action="store_true",
        help="Skip candidate search and score only the locked candidate.",
    )
    parser.add_argument(
        "--write-component-scores",
        action="store_true",
        help="Emit selected component score and ablation reports.",
    )
    parser.add_argument(
        "--evolution-sample-rows",
        type=int,
        default=4500,
        help="Maximum train/CV rows sampled by evolution optimizers; holdout is never sampled.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = run_advanced_main_run_detectors(
        args.data_root,
        args.out_root,
        folds=args.folds,
        seed=args.seed,
        force=args.force,
        candidate_limit=args.candidate_limit,
        evolution_rounds=args.evolution_rounds,
        search_profile=args.search_profile,
        feature_cache=args.feature_cache,
        selection_lock=args.selection_lock,
        score_locked_only=args.score_locked_only,
        write_component_scores=args.write_component_scores,
        evolution_sample_rows=args.evolution_sample_rows,
    )
    print(
        f"wrote {args.out_root} "
        f"records={quality['record_count']} "
        f"candidates={quality['candidate_count']} "
        f"selected={quality['selected_candidate_id']} "
        f"status={quality['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
