#!/usr/bin/env python3
"""PyTorch multi-view transformer consumer for ml-training-v2 frame tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import detection_common as dc


METHOD = "multiview_radar_transformer"
TOKEN_COLUMNS = [
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
    "micro_doppler_peak_hz_proxy",
    "micro_doppler_bandwidth_hz_proxy",
    "stft_energy",
    "weighted_spectrum_peak",
    "cepstrum_peak",
    "cadence_velocity_peak",
    "range_time_energy",
    "doppler_time_energy",
    "range_doppler_time_energy",
    "normalized_snr",
]


class MultiViewTransformer(nn.Module):
    def __init__(self, token_dim: int, static_dim: int, max_frames: int) -> None:
        super().__init__()
        d_model = 32
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_frames, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.12,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, d_model), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_model * 2, 48), nn.ReLU(), nn.Dropout(0.12), nn.Linear(48, 1))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens) + self.pos[:, : tokens.shape[1], :]
        encoded = self.encoder(x, src_key_padding_mask=~mask.bool())
        weights = mask.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([pooled, self.static_proj(static)], dim=1)).squeeze(1)


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
    T: np.ndarray, M: np.ndarray, F: np.ndarray, splits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train = splits == "train"
    token_values = T[train][M[train].astype(bool)]
    token_mu = token_values.mean(axis=0)
    token_std = token_values.std(axis=0)
    token_std[token_std < 1e-6] = 1.0
    feat_mu = F[train].mean(axis=0)
    feat_std = F[train].std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    return (
        ((T - token_mu) / token_std).astype(np.float32),
        ((F - feat_mu) / feat_std).astype(np.float32),
        {"token_mu": token_mu, "token_std": token_std, "feat_mu": feat_mu, "feat_std": feat_std},
    )


def predict_scores(model: nn.Module, T: np.ndarray, M: np.ndarray, F: np.ndarray, batch_size: int = 384) -> np.ndarray:
    model.train(False)
    scores = []
    with torch.no_grad():
        for start in range(0, len(T), batch_size):
            logits = model(
                torch.from_numpy(T[start : start + batch_size]),
                torch.from_numpy(M[start : start + batch_size]),
                torch.from_numpy(F[start : start + batch_size]),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train_and_score(
    T: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    seed: int,
    epochs: int,
    model_path: Path,
) -> tuple[dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    T_norm, F_norm, norm = normalize_inputs(T, M, F, splits)
    train_idx = np.where(splits == "train")[0]
    pos = max(1, int(y[train_idx].sum()))
    neg = max(1, int(train_idx.size - pos))
    model = MultiViewTransformer(token_dim=T.shape[-1], static_dim=F.shape[-1], max_frames=T.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg / pos), dtype=torch.float32))
    rng = np.random.default_rng(seed)
    batch_size = 96
    for _ in range(max(1, int(epochs))):
        order = train_idx[rng.permutation(train_idx.size)]
        model.train()
        for start in range(0, order.size, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(torch.from_numpy(T_norm[idx]), torch.from_numpy(M[idx]), torch.from_numpy(F_norm[idx])),
                torch.from_numpy(y[idx].astype(np.float32)),
            )
            loss.backward()
            optimizer.step()
    scores = predict_scores(model, T_norm, M, F_norm)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalization": norm,
            "token_dim": T.shape[-1],
            "static_dim": F.shape[-1],
            "max_frames": T.shape[1],
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
        T, M, F, B, y, splits, max_consumed = dc.transformer_features(frames, columns, records, horizon_s, TOKEN_COLUMNS)
        if max_consumed > float(horizon_s) + 1e-6:
            raise AssertionError(f"horizon leakage: consumed {max_consumed}s for {horizon_s}s")
        model_path = out_root / "models" / METHOD / f"{dc.horizon_label(horizon_s)}.pt"
        aucs, scores = train_and_score(T, M, F, y, splits, args.seed + horizon_s, epochs, model_path)
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
