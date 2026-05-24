#!/usr/bin/env python3
"""Generate LaTeX metric macros from the locked paper evidence bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE_ROOT = Path("outputs/paper-evidence/current")
DEFAULT_OUT = Path("target/paper/generated_metrics.tex")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite metric value: {value!r}")
    return parsed


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}\\%"


def _fmt_unsigned_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}\\%"


def _fmt_mult(value: float) -> str:
    return f"{value:.2f}x"


def _macro(name: str, value: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _index_gain_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row.get("metric", ""): row for row in rows}
    required = {
        "lcb95_recall_at_leq_1pct_fpr",
        "point_recall_at_leq_1pct_fpr",
        "average_precision",
        "f1",
        "selected_threshold_false_positives",
        "roc_auc_guardrail",
    }
    missing = sorted(required - set(indexed))
    if missing:
        raise KeyError("missing KPI row(s): " + ", ".join(missing))
    return indexed


def _confusion_label(row: dict[str, str]) -> str:
    return f"{int(_float(row['tp']))}/{int(_float(row['fp']))}/{int(_float(row['fn']))}"


def build_macros(evidence_root: Path) -> list[str]:
    manifest = _read_json(evidence_root / "paper_evidence_manifest.json")
    if manifest.get("version") != "current":
        raise ValueError("paper evidence manifest must be current")

    gain_rows = _index_gain_rows(_read_csv(evidence_root / "main_kpi_gain_table.csv"))
    confusion_rows = _read_csv(evidence_root / "selected_threshold_confusion_matrix.csv")
    confusion = {row.get("method", ""): row for row in confusion_rows}
    baseline = confusion.get("layered_fusion_c2")
    ei = confusion.get("locked_candidate")
    if baseline is None or ei is None:
        raise KeyError(
            "selected-threshold confusion matrix must include layered_fusion_c2 and locked_candidate"
        )

    split = _read_json(evidence_root / "split_summary.json")

    lcb = gain_rows["lcb95_recall_at_leq_1pct_fpr"]
    point = gain_rows["point_recall_at_leq_1pct_fpr"]
    ap = gain_rows["average_precision"]
    f1 = gain_rows["f1"]
    fp = gain_rows["selected_threshold_false_positives"]
    roc = gain_rows["roc_auc_guardrail"]

    phase_records = int(split["scenario_group_count"]) * 3

    macros: list[tuple[str, Any]] = [
        ("MetricScenarioGroups", int(split["scenario_group_count"])),
        ("MetricPositiveGroups", int(split["positive_group_count"])),
        ("MetricPhaseRecords", phase_records),
        ("MetricHoldoutRecords", int(split["holdout_record_count"])),
        ("MetricHoldoutPositiveRecords", int(split["holdout_positive_record_count"])),
        ("MetricHoldoutPositiveGroups", int(split["holdout_positive_group_count"])),
        ("MetricPriorLcb", _fmt(_float(lcb["prior_fusion_baseline"]))),
        ("MetricEiLcb", _fmt(_float(lcb["ei_candidate"]))),
        ("MetricLcbAbsChange", _fmt_signed(_float(lcb["absolute_change"]))),
        ("MetricLcbRelativePercent", _fmt_pct(_float(lcb["relative_percent"]))),
        ("MetricLcbMultiplier", _fmt_mult(_float(lcb["relative_change"]))),
        ("MetricPriorPointRecall", _fmt(_float(point["prior_fusion_baseline"]))),
        ("MetricEiPointRecall", _fmt(_float(point["ei_candidate"]))),
        ("MetricPointRecallAbsChange", _fmt_signed(_float(point["absolute_change"]))),
        ("MetricPointRecallRelativePercent", _fmt_pct(_float(point["relative_percent"]))),
        ("MetricPointRecallMultiplier", _fmt_mult(_float(point["relative_change"]))),
        ("MetricPriorAP", _fmt(_float(ap["prior_fusion_baseline"]))),
        ("MetricEiAP", _fmt(_float(ap["ei_candidate"]))),
        ("MetricAPAbsChange", _fmt_signed(_float(ap["absolute_change"]))),
        ("MetricAPRelativePercent", _fmt_pct(_float(ap["relative_percent"]))),
        ("MetricPriorFOne", _fmt(_float(f1["prior_fusion_baseline"]))),
        ("MetricEiFOne", _fmt(_float(f1["ei_candidate"]))),
        ("MetricFOneAbsChange", _fmt_signed(_float(f1["absolute_change"]))),
        ("MetricPriorSelectedFP", int(_float(fp["prior_fusion_baseline"]))),
        ("MetricEiSelectedFP", int(_float(fp["ei_candidate"]))),
        ("MetricSelectedFPReduction", int(abs(_float(fp["absolute_change"])))),
        ("MetricSelectedFPReductionPercent", _fmt_unsigned_pct(_float(fp["relative_percent"]))),
        ("MetricPriorRocAuc", _fmt(_float(roc["prior_fusion_baseline"]))),
        ("MetricEiRocAuc", _fmt(_float(roc["ei_candidate"]))),
        ("MetricRocAucDelta", _fmt_signed(_float(roc["absolute_change"]))),
        ("MetricPriorTPFPFN", _confusion_label(baseline)),
        ("MetricEiTPFPFN", _confusion_label(ei)),
        ("MetricPriorSelectedFPR", _fmt(_float(baseline["false_positive_rate"]), 6)),
        ("MetricEiSelectedFPR", _fmt(_float(ei["false_positive_rate"]), 6)),
    ]

    return [
        "% Generated by paper/generate_metric_macros.py; do not edit by hand.",
        *[_macro(name, value) for name, value in macros],
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    lines = build_macros(args.paper_evidence_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
