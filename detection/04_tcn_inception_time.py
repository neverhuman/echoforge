#!/usr/bin/env python3
"""PyTorch temporal CNN consumer for ml-training-v2 frame sequences."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import detection_common as dc


METHOD = "tcn_inception_time"
SEQ_COLUMNS = [
    "snr_db",
    "cfar_statistic",
    "cfar_threshold",
    "tbd_track_score",
    "local_noise_floor_db",
    "doppler_scr",
    "rfi_pressure",
    "dropout_fraction",
    "phase_impairment_rad",
    "amplitude_impairment",
    "micro_doppler_energy",
    "range_time_energy",
    "doppler_time_energy",
    "range_doppler_time_energy",
]


class TemporalConvClassifier(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 24, kernel_size=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(24 + static_dim, 48), nn.ReLU(), nn.Dropout(0.12), nn.Linear(48, 1)
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.conv(seq.transpose(1, 2)).mean(dim=-1)
        return self.head(torch.cat([x, static], dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=dc.DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", default=dc.DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--horizons", default="5,15,45")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def normalize_inputs(
    S: np.ndarray, F: np.ndarray, splits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train = splits == "train"
    seq_mu = S[train].reshape(-1, S.shape[-1]).mean(axis=0)
    seq_std = S[train].reshape(-1, S.shape[-1]).std(axis=0)
    seq_std[seq_std < 1e-6] = 1.0
    feat_mu = F[train].mean(axis=0)
    feat_std = F[train].std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    return (
        ((S - seq_mu) / seq_std).astype(np.float32),
        ((F - feat_mu) / feat_std).astype(np.float32),
        {"seq_mu": seq_mu, "seq_std": seq_std, "feat_mu": feat_mu, "feat_std": feat_std},
    )


def predict_scores(
    model: nn.Module, S: np.ndarray, F: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    model.train(False)
    scores = []
    with torch.no_grad():
        for start in range(0, len(S), batch_size):
            logits = model(
                torch.from_numpy(S[start : start + batch_size]),
                torch.from_numpy(F[start : start + batch_size]),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train_and_score(
    S: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    seed: int,
    epochs: int,
    model_path: Path,
) -> tuple[dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    S_norm, F_norm, norm = normalize_inputs(S, F, splits)
    train_idx = np.where(splits == "train")[0]
    pos = max(1, int(y[train_idx].sum()))
    neg = max(1, int(train_idx.size - pos))
    model = TemporalConvClassifier(seq_dim=S.shape[-1], static_dim=F.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg / pos), dtype=torch.float32))
    rng = np.random.default_rng(seed)
    batch_size = 128
    for _ in range(max(1, int(epochs))):
        order = train_idx[rng.permutation(train_idx.size)]
        model.train()
        for start in range(0, order.size, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(torch.from_numpy(S_norm[idx]), torch.from_numpy(F_norm[idx])),
                torch.from_numpy(y[idx].astype(np.float32)),
            )
            loss.backward()
            optimizer.step()
    scores = predict_scores(model, S_norm, F_norm)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalization": norm,
            "seq_dim": S.shape[-1],
            "static_dim": F.shape[-1],
        },
        model_path,
    )
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
        S, F, B, y, splits, max_consumed = dc.sequence_features(
            frames, columns, records, horizon_s, SEQ_COLUMNS
        )
        if max_consumed > float(horizon_s) + 1e-6:
            raise AssertionError(f"horizon leakage: consumed {max_consumed}s for {horizon_s}s")
        model_path = out_root / "models" / METHOD / f"{dc.horizon_label(horizon_s)}.pt"
        aucs, scores = train_and_score(S, F, y, splits, args.seed + horizon_s, epochs, model_path)
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
        dc.write_auxiliary_reports(
            out_root, METHOD, dc.horizon_label(horizon_s), records, y, splits, scores
        )
        dc.write_reports(row, out_root)
        print(
            f"{METHOD} {dc.horizon_label(horizon_s)} "
            f"train_auc={row['train_auc']:.4f} validation_auc={row['validation_auc']:.4f} "
            f"holdout_auc={row['holdout_auc']:.4f} baseline_auc={row['baseline_auc']:.4f}"
        )


if __name__ == "__main__":
    run()
