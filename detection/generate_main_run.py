#!/usr/bin/env python3
"""Generate the Shahed-136/Geran-2 public-proxy main-run corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from detection.main_run_generation import build_main_run_dataset
    from detection.main_run_types import (
        DEFAULT_FOLDS,
        DEFAULT_JAMMING_DECEPTION_RATE,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DEFAULT_SHARD_SIZE,
        DEFAULT_LOW_PREVALENCE_POSITIVE_GROUPS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_generation import build_main_run_dataset
    from main_run_types import (
        DEFAULT_FOLDS,
        DEFAULT_JAMMING_DECEPTION_RATE,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DEFAULT_SHARD_SIZE,
        DEFAULT_LOW_PREVALENCE_POSITIVE_GROUPS,
    )

DEFAULT_OUT_ROOT = Path("outputs/training-data/fixed-wing-pusher-proxy-main-run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("fixed-wing-pusher-proxy",),
        default="fixed-wing-pusher-proxy",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
    )
    parser.add_argument("--scenario-groups", type=int, default=DEFAULT_SCENARIO_GROUPS)
    parser.add_argument("--positive-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--jamming-deception-rate",
        type=float,
        default=DEFAULT_JAMMING_DECEPTION_RATE,
    )
    parser.add_argument(
        "--holdout-policy",
        choices=("group_random", "site", "noise_regime", "hard_negative_role"),
        default="group_random",
    )
    parser.add_argument("--holdout-value", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.positive_groups is None:
        args.positive_groups = DEFAULT_LOW_PREVALENCE_POSITIVE_GROUPS
    quality = build_main_run_dataset(
        args.out_root,
        scenario_groups=args.scenario_groups,
        positive_groups=args.positive_groups,
        seed=args.seed,
        folds=args.folds,
        shard_size=args.shard_size,
        force=args.force,
        smoke=args.smoke,
        paper_profile=args.profile,
        holdout_policy=args.holdout_policy,
        holdout_value=args.holdout_value,
        jamming_deception_rate=args.jamming_deception_rate,
    )
    print(
        "wrote "
        f"{args.out_root} "
        f"scenario_groups={quality['scenario_group_count']} "
        f"records={quality['record_count']} "
        f"status={quality['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
