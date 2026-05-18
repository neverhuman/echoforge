#!/usr/bin/env python3
"""Generate the v2 benchmark if needed, then run all five detection consumers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_cfar_tbd_fusion.py",
    "02_lightgbm_window_gbdt.py",
    "03_catboost_ordered_boosting.py",
    "04_tcn_inception_time.py",
    "05_multiview_radar_transformer.py",
]
DEFAULT_DATA_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
SMOKE_DATA_ROOT = "outputs/training-data/shahed136-public-proxy-ml-training-v2-smoke"
DEFAULT_OUT_ROOT = "outputs/detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--horizons", default="5,15,45")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--records", type=int, default=None)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def ensure_dataset(detection_dir: Path, data_root: Path, args: argparse.Namespace) -> None:
    if args.skip_generate and not (data_root / "frame_features.npz").exists():
        raise FileNotFoundError(f"{data_root} is missing frame_features.npz")
    if args.skip_generate:
        return
    if (data_root / "frame_features.npz").exists() and not args.force_regenerate:
        return
    records = args.records if args.records is not None else (1000 if args.smoke else 50_000)
    cmd = [
        sys.executable,
        str(detection_dir / "generate_ml_training_v2.py"),
        "--out-root",
        str(data_root),
        "--records",
        str(records),
        "--scale-name",
        "smoke" if args.smoke else "standard",
        "--seed",
        str(args.seed),
        "--force",
    ]
    print(f"generating {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    detection_dir = Path(__file__).resolve().parent
    data_root = Path(SMOKE_DATA_ROOT if args.smoke and args.data_root == DEFAULT_DATA_ROOT else args.data_root)
    ensure_dataset(detection_dir, data_root, args)
    for script in SCRIPTS:
        cmd = [
            sys.executable,
            str(detection_dir / script),
            "--data-root",
            str(data_root),
            "--out-root",
            args.out_root,
            "--seed",
            str(args.seed),
            "--horizons",
            args.horizons,
            "--epochs",
            str(args.epochs),
        ]
        if args.max_records is not None:
            cmd.extend(["--max-records", str(args.max_records)])
        if args.smoke:
            cmd.append("--smoke")
        print(f"running {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    report = Path("detection/reports/auc_table.md")
    if report.exists():
        print(f"wrote {report}")


if __name__ == "__main__":
    main()
