#!/usr/bin/env python3
"""Generate the Shahed-136/Geran-2 public-proxy main-run corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from detection.main_run_generation import build_main_run_dataset
    from detection.main_run_types import (
        DEFAULT_FOLDS,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DEFAULT_SHARD_SIZE,
        DEFAULT_V2_POSITIVE_GROUPS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from main_run_generation import build_main_run_dataset
    from main_run_types import (
        DEFAULT_FOLDS,
        DEFAULT_POSITIVE_GROUPS,
        DEFAULT_SCENARIO_GROUPS,
        DEFAULT_SEED,
        DEFAULT_SHARD_SIZE,
        DEFAULT_V2_POSITIVE_GROUPS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("fixed-wing-pusher-proxy-v1", "fixed-wing-pusher-proxy-v2"),
        default="fixed-wing-pusher-proxy-v1",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/training-data/runit-shahed136-main-run-v1"),
    )
    parser.add_argument("--scenario-groups", type=int, default=DEFAULT_SCENARIO_GROUPS)
    parser.add_argument("--positive-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
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
        args.positive_groups = (
            DEFAULT_V2_POSITIVE_GROUPS
            if args.profile == "fixed-wing-pusher-proxy-v2"
            else DEFAULT_POSITIVE_GROUPS
        )
    if args.profile == "fixed-wing-pusher-proxy-v2" and args.out_root == Path(
        "outputs/training-data/runit-shahed136-main-run-v1"
    ):
        args.out_root = Path("outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run")
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
