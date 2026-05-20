"""Heuristic scoring and pseudo-model helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from detection.pipeline_contracts.metrics import pr_auc, roc_auc


def rank_scores(rows: Sequence[Mapping[str, str]], score_fn) -> list[float]:
    return [float(score_fn(row)) for row in rows]


def metrics_for_rows(labels: Sequence[float], scores: Sequence[float]) -> dict[str, float]:
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
    }
