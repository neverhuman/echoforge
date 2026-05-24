#!/usr/bin/env python3
"""Normalize the EI candidate search trace for paper evidence.

The paper figure is allowed to show the internal train/CV search trajectory and
one locked holdout endpoint. It must not show holdout performance as an
iterative search curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ADVANCED_ROOT = Path(
    "outputs/detection/fixed-wing-pusher-proxy-main-run-advanced-evolution"
)
DEFAULT_OUT_ROOT = Path("outputs/paper-evidence/current")
SCHEMA_VERSION = "ei-evolution-evidence"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stage_from_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("stage", "")).strip()
    if explicit:
        return explicit
    candidate_type = str(row.get("candidate_type", ""))
    candidate_id = str(row.get("candidate_id", ""))
    if candidate_type == "meta_fusion" or candidate_id.startswith("meta_fusion."):
        return "meta_fusion_search"
    if candidate_type == "surface" or candidate_id.startswith("surface."):
        return "surface_control"
    return "base_candidate_search"


def _generation(stage: str) -> int:
    return {"base_candidate_search": 0, "surface_control": 1, "meta_fusion_search": 2}.get(stage, 3)


def _sort_tuple(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        _safe_float(row.get("train_cv_objective", row.get("objective")), float("-inf")),
        _safe_float(row.get("train_cv_average_precision"), float("-inf")),
        _safe_float(row.get("train_cv_roc_auc"), float("-inf")),
        str(row.get("candidate_id", "")),
    )


def _normalize_trace_rows(rows: list[dict[str, Any]], *, trace_basis: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    running_best: dict[str, Any] | None = None
    for fallback_index, row in enumerate(rows, start=1):
        stage = _stage_from_row(row)
        candidate_index = _safe_int(row.get("candidate_index"), fallback_index)
        local = {
            "schema_version": "ei-evolution-trace",
            "candidate_index": candidate_index,
            "generation": _safe_int(row.get("generation"), _generation(stage)),
            "stage": stage,
            "candidate_id": str(row.get("candidate_id", "")),
            "base_candidate_id": str(row.get("base_candidate_id", "")),
            "candidate_type": str(row.get("candidate_type", "")),
            "family": str(row.get("family", "")),
            "subset_name": str(row.get("subset_name", "")),
            "head": str(row.get("head", "")),
            "calibrator": str(row.get("calibrator", "")),
            "feature_count": _safe_int(row.get("feature_count")),
            "component_count": _safe_int(row.get("component_count", row.get("feature_count"))),
            "selection_split": str(row.get("selection_split", "train_cv")),
            "holdout_rows_used_for_selection": _safe_int(
                row.get("holdout_rows_used_for_selection")
            ),
            "train_cv_objective": _safe_float(row.get("train_cv_objective", row.get("objective"))),
            "train_cv_average_precision": _safe_float(row.get("train_cv_average_precision")),
            "train_cv_roc_auc": _safe_float(row.get("train_cv_roc_auc")),
            "train_cv_f1": _safe_float(row.get("train_cv_f1")),
            "train_cv_false_positive_rate": _safe_float(row.get("train_cv_false_positive_rate")),
            "initial_take_up_average_precision": _safe_float(
                row.get("initial_take_up_average_precision")
            ),
            "threshold": _safe_float(row.get("threshold")),
            "selected_by_cv": _as_bool(row.get("selected_by_cv")),
            "final_selected": _as_bool(row.get("final_selected")),
            "trace_basis": trace_basis,
            "paper_note": "train/CV-only search point; holdout is not used for selection",
        }
        if running_best is None or _sort_tuple(local) > _sort_tuple(running_best):
            running_best = dict(local)
        assert running_best is not None
        local["running_best_candidate_id"] = running_best["candidate_id"]
        local["running_best_objective"] = running_best["train_cv_objective"]
        local["running_best_average_precision"] = running_best["train_cv_average_precision"]
        local["running_best_roc_auc"] = running_best["train_cv_roc_auc"]
        normalized.append(local)
    normalized.sort(key=lambda row: _safe_int(row.get("candidate_index")))
    return normalized


def _load_trace(advanced_root: Path) -> tuple[list[dict[str, Any]], str, bool]:
    csv_rows = _read_csv_rows(advanced_root / "evolution_trace.csv")
    if csv_rows:
        return csv_rows, "true_evaluation_order", True
    jsonl_rows = _read_jsonl_rows(advanced_root / "evolution_trace.jsonl")
    if jsonl_rows and any("candidate_index" in row for row in jsonl_rows):
        return jsonl_rows, "true_evaluation_order", True
    leaderboard = _read_csv_rows(advanced_root / "candidate_leaderboard.csv")
    if leaderboard:
        for index, row in enumerate(leaderboard, start=1):
            row.setdefault("candidate_index", index)
            row.setdefault("paper_note", "leaderboard rank only; not evaluation order")
        return leaderboard, "leaderboard_rank_only", False
    return [], "missing", False


def _holdout_endpoint(advanced_root: Path, selection_lock: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(advanced_root / "performance_summary.json")
    quality = _read_json(advanced_root / "fusion_quality_report.json")
    selected = summary.get("advanced_selection", {}) if isinstance(summary, dict) else {}
    gate = quality.get("promotion_gate", {}) if isinstance(quality, dict) else {}
    metrics = (
        quality.get("selected_holdout_threshold_metrics", {}) if isinstance(quality, dict) else {}
    )
    return {
        "selected_candidate_id": selection_lock.get("selected_candidate_id")
        or selected.get("selected_candidate_id"),
        "selected_method": selected.get("selected_method", "spectral_transport_hypergraph_fusion"),
        "holdout_policy": selected.get(
            "holdout_evaluation_policy", "single selected winner scored after internal-CV lock"
        ),
        "holdout_average_precision": gate.get("selected_holdout_average_precision"),
        "holdout_roc_auc": gate.get("selected_holdout_roc_auc"),
        "holdout_precision": metrics.get("precision"),
        "holdout_recall": metrics.get("recall"),
        "holdout_false_positive_rate": metrics.get("false_positive_rate"),
        "holdout_f1": metrics.get("f1"),
        "gate_status": gate.get("status", quality.get("status")),
    }


def _validate_rows(rows: list[dict[str, Any]], selection_lock: dict[str, Any]) -> None:
    if not rows:
        raise ValueError("EI evolution trace has no rows")
    bad_split = [row for row in rows if row.get("selection_split") != "train_cv"]
    if bad_split:
        raise ValueError("EI evolution trace contains non-train/CV selection rows")
    holdout_used = [
        row for row in rows if _safe_int(row.get("holdout_rows_used_for_selection")) != 0
    ]
    if holdout_used:
        raise ValueError("EI evolution trace uses holdout rows for selection")
    selected_rows = [row for row in rows if _as_bool(row.get("selected_by_cv"))]
    if len(selected_rows) != 1:
        raise ValueError(
            f"EI evolution trace must mark exactly one selected row, got {len(selected_rows)}"
        )
    selected_id = str(selection_lock.get("selected_candidate_id", ""))
    if selected_id and selected_rows[0].get("candidate_id") != selected_id:
        raise ValueError("EI evolution selected row does not match selection_lock.json")


def build_ei_evolution_evidence(
    advanced_root: Path = DEFAULT_ADVANCED_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    *,
    strict_trace: bool = False,
) -> dict[str, Any]:
    raw_rows, trace_basis, true_trace = _load_trace(advanced_root)
    if strict_trace and not true_trace:
        raise FileNotFoundError(
            f"strict trace requested, but no true evolution_trace.csv exists under {advanced_root}"
        )
    rows = _normalize_trace_rows(raw_rows, trace_basis=trace_basis)
    selection_lock = _read_json(advanced_root / "selection_lock.json")
    selected_id = str(selection_lock.get("selected_candidate_id", ""))
    if selected_id:
        for row in rows:
            is_selected = row.get("candidate_id") == selected_id
            row["selected_by_cv"] = is_selected
            row["final_selected"] = is_selected
    if strict_trace:
        _validate_rows(rows, selection_lock)

    best_by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row.get("stage", "unknown"))
        if stage not in best_by_stage or _sort_tuple(row) > _sort_tuple(best_by_stage[stage]):
            best_by_stage[stage] = row
    selected_rows = [row for row in rows if _as_bool(row.get("selected_by_cv"))]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "advanced_root": str(advanced_root),
        "trace_basis": trace_basis,
        "true_evaluation_order_available": true_trace,
        "candidate_evaluation_count": len(rows),
        "selected_candidate_id": selected_id
        or (selected_rows[0]["candidate_id"] if selected_rows else ""),
        "selected_row": selected_rows[0] if selected_rows else {},
        "best_by_stage": best_by_stage,
        "holdout_endpoint": _holdout_endpoint(advanced_root, selection_lock),
        "review_guardrail": (
            "Use train/CV trajectory for the evolution curve; show only one locked holdout endpoint. "
            "Do not plot holdout performance as an iterative search curve."
        ),
    }
    _write_csv(out_root / "ei_evolution_trace.csv", rows)
    _write_json(out_root / "ei_evolution_summary.json", summary)
    return {"summary": summary, "trace_rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--strict-trace", action="store_true")
    args = parser.parse_args()
    payload = build_ei_evolution_evidence(
        args.advanced_root,
        args.out_root,
        strict_trace=args.strict_trace,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "rows": len(payload["trace_rows"]),
                "out_root": str(args.out_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
