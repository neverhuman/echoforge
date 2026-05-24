"""Metric helpers for the EchoForge ML pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


def _coerce_pairs(labels: Sequence[float], scores: Sequence[float]) -> list[tuple[float, float]]:
    pairs = [(float(label), float(score)) for label, score in zip(labels, scores)]
    if not pairs:
        return []
    return [(label, score) for label, score in pairs if isfinite(label) and isfinite(score)]


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    pairs = _coerce_pairs(labels, scores)
    positives = sum(1 for label, _ in pairs if label > 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    ranked = sorted(enumerate(pairs), key=lambda item: (item[1][1], item[0]))
    rank_sum = 0.0
    for rank, (_, (label, _)) in enumerate(ranked, start=1):
        if label > 0.5:
            rank_sum += rank

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    pairs = sorted(_coerce_pairs(labels, scores), key=lambda pair: pair[1], reverse=True)
    positives = sum(1 for label, _ in pairs if label > 0.5)
    if positives == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for label, _ in pairs:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return area


def threshold_at_precision(labels: Sequence[float], scores: Sequence[float], target_precision: float) -> float:
    pairs = sorted(_coerce_pairs(labels, scores), key=lambda pair: pair[1], reverse=True)
    positives = sum(1 for label, _ in pairs if label > 0.5)
    tp = 0
    fp = 0
    threshold = 1.0
    for label, score in pairs:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
        precision = tp / max(1, tp + fp)
        if precision >= target_precision:
            threshold = score
    return threshold


def calibration_bins(labels: Sequence[float], scores: Sequence[float], *, bins: int = 10) -> list[dict[str, float]]:
    pairs = _coerce_pairs(labels, scores)
    if not pairs:
        return []
    width = 1.0 / bins
    output = []
    for index in range(bins):
        lower = index * width
        upper = 1.0 if index == bins - 1 else (index + 1) * width
        bucket = [(label, score) for label, score in pairs if lower <= score < upper or (index == bins - 1 and score == 1.0)]
        if bucket:
            mean_score = sum(score for _, score in bucket) / len(bucket)
            positive_rate = sum(1 for label, _ in bucket if label > 0.5) / len(bucket)
        else:
            mean_score = 0.0
            positive_rate = 0.0
        output.append(
            {
                "bin": float(index),
                "lower": lower,
                "upper": upper,
                "count": float(len(bucket)),
                "mean_score": mean_score,
                "positive_rate": positive_rate,
            }
        )
    return output
