#!/usr/bin/env python3
"""CatBoost ordered boosting consumer for ml-training-v2 descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier

import detection_common as dc


METHOD = "catboost_ordered_boosting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=dc.DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", default=dc.DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--horizons", default="5,15,45")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def train_and_score(
    X: np.ndarray, y: np.ndarray, splits: np.ndarray, seed: int, epochs: int, model_path: Path
) -> tuple[dict[str, float], np.ndarray]:
    train_idx = np.where(splits == "train")[0]
    order = train_idx[np.random.default_rng(seed).permutation(train_idx.size)]
    val_idx = np.where(splits == "validation")[0]
    pos = max(1, int(y[order].sum()))
    neg = max(1, int(order.size - pos))
    model = CatBoostClassifier(
        iterations=max(50, int(epochs) * 90),
        learning_rate=0.035,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bernoulli",
        subsample=0.82,
        random_seed=seed,
        class_weights=[1.0, float(neg / pos)],
        l2_leaf_reg=4.0,
        allow_writing_files=False,
        thread_count=2,
        verbose=False,
    )
    model.fit(X[order], y[order], eval_set=(X[val_idx], y[val_idx]), use_best_model=False)
    scores = model.predict_proba(X)[:, 1]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return dc.evaluate_scores(y, splits, scores), scores


def run() -> None:
    args = parse_args()
    max_records = 1000 if args.smoke and args.max_records is None else args.max_records
    epochs = min(args.epochs, 1) if args.smoke else args.epochs
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    records = dc.load_records(data_root, args.seed, max_records)
    frames, columns = dc.select_frames(data_root, records)

    for horizon_s in dc.parse_horizons(args.horizons):
        X, B, y, splits, max_consumed = dc.tabular_features(frames, columns, records, horizon_s, profile="catboost")
        if max_consumed > float(horizon_s) + 1e-6:
            raise AssertionError(f"horizon leakage: consumed {max_consumed}s for {horizon_s}s")
        model_path = out_root / "models" / METHOD / f"{dc.horizon_label(horizon_s)}.joblib"
        aucs, scores = train_and_score(X, y, splits, args.seed + horizon_s, epochs, model_path)
        baseline = dc.evaluate_scores(y, splits, dc.baseline_scores(B, splits))
        sliced, _ = dc.horizon_slice(frames, columns, horizon_s)
        row: dict[str, Any] = {
            "method": METHOD,
            "horizon_name": dc.horizon_label(horizon_s),
            "horizon_s": int(horizon_s),
            **aucs,
            "baseline_auc": baseline["holdout_auc"],
            "lift_vs_baseline": aucs["holdout_auc"] - baseline["holdout_auc"],
            **dc.operating_metrics(y, splits, scores),
            **dc.track_metrics(sliced, columns, y, splits),
            **dc.slice_metrics(records, y, splits, scores),
            **dc.counts(y, splits),
        }
        dc.write_auxiliary_reports(out_root, METHOD, dc.horizon_label(horizon_s), records, y, splits, scores)
        dc.write_reports(row, out_root)
        print(
            f"{METHOD} {dc.horizon_label(horizon_s)} "
            f"train_auc={row['train_auc']:.4f} validation_auc={row['validation_auc']:.4f} "
            f"holdout_auc={row['holdout_auc']:.4f} baseline_auc={row['baseline_auc']:.4f}"
        )


if __name__ == "__main__":
    run()
