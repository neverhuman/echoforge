#!/usr/bin/env python3
"""Generate the v2 difficulty-controlled radar detection benchmark.

The benchmark is a strict-open synthetic public-proxy corpus. Public radar
literature is used only as distribution-metadata context; no measured traces
or proprietary-equivalent claims are emitted.

Implementation is split across:
  ml_training_v2_config.py     — constants, dataclasses, lookup tables
  ml_training_v2_generators.py — physics generators, strata, split assignment
  ml_training_v2_diagnostics.py — feature aggregation, audit probes, reports
  ml_training_v2_report.py     — metadata / output file writing
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from detection.ml_training_v2_config import (
        DEFAULT_OUT_ROOT,
        DEFAULT_RECORDS,
        FRAME_COLUMNS,
        FRAME_COUNT,
        FRAME_PERIOD_S,
    )
    from detection.ml_training_v2_generators import (
        assign_splits,
        build_strata,
        choose_label,
        simulate_record,
        stable_seed,
    )
    from detection.ml_training_v2_real_priors import load_real_anchor_priors
    from detection.ml_training_v2_diagnostics import (
        aggregate_frame_features,
        write_diagnostics,
    )
    from detection.ml_training_v2_report import write_metadata
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from ml_training_v2_config import (
        DEFAULT_OUT_ROOT,
        DEFAULT_RECORDS,
        FRAME_COLUMNS,
        FRAME_COUNT,
        FRAME_PERIOD_S,
    )
    from ml_training_v2_generators import (
        assign_splits,
        build_strata,
        choose_label,
        simulate_record,
        stable_seed,
    )
    from ml_training_v2_real_priors import load_real_anchor_priors
    from ml_training_v2_diagnostics import (
        aggregate_frame_features,
        write_diagnostics,
    )
    from ml_training_v2_report import write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--scale-name", default="standard")
    parser.add_argument("--strata", type=int, default=50)
    parser.add_argument(
        "--real-anchor-priors",
        type=Path,
        help="Local calibration_coefficients.json from detection.real_data; opt-in only.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{out_root} already exists; pass --force to replace generated artifacts"
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    real_anchor_priors = load_real_anchor_priors(args.real_anchor_priors)
    strata = build_strata(args.strata)
    records: list[dict[str, Any]] = []
    frames = np.zeros((args.records, FRAME_COUNT, len(FRAME_COLUMNS)), dtype=np.float32)
    for idx in range(args.records):
        stratum = strata[idx % len(strata)]
        record_rng = np.random.default_rng(stable_seed(args.seed, idx, stratum.wave_index))
        positive = choose_label(record_rng, stratum)
        record_id = f"record_{idx:06d}"
        record, frame = simulate_record(
            record_rng, record_id, idx, stratum, positive, real_anchor_priors
        )
        records.append(record)
        frames[idx] = frame
        if (idx + 1) % 10_000 == 0:
            print(f"generated {idx + 1}/{args.records} records", flush=True)
    assign_splits(records, args.seed)
    records_df = pd.DataFrame(records)
    records_df.to_csv(out_root / "records.csv", index=False)
    np.savez_compressed(
        out_root / "frame_features.npz",
        frames=frames,
        record_ids=records_df["record_id"].astype(str).to_numpy(dtype="<U32"),
        frame_columns=np.asarray(FRAME_COLUMNS, dtype="<U64"),
        frame_period_s=np.array(FRAME_PERIOD_S, dtype=np.float32),
        benchmark_version=np.asarray(["ml-training-v2"], dtype="<U32"),
    )
    feature_df, feature_names = aggregate_frame_features(frames)
    feature_df.insert(0, "record_id", records_df["record_id"].to_numpy())
    feature_df.to_csv(out_root / "features.csv", index=False, float_format="%.6f")
    quality = write_diagnostics(
        out_root, records_df, frames, feature_df.drop(columns=["record_id"])
    )
    write_metadata(
        out_root,
        records_df,
        strata,
        feature_names,
        quality,
        args.scale_name,
        args.seed,
        real_anchor_priors,
    )
    print(
        f"wrote {out_root} records={len(records_df)} strata={len(strata)} "
        f"single_feature_auc_max={quality['single_feature_auc_max']:.4f} "
        f"baseline_auc={quality['baseline_cfar_tbd_auc_all_records']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
