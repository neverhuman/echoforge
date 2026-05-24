#!/usr/bin/env python3
"""Produce the current benchmark regeneration readiness report.

The report is intentionally allowed to be a no-go. It summarizes the current
current smoke artifacts and blocks regeneration-ready claims when calibration,
false-alarm, lifecycle, or dependency evidence is still incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_regeneration_report_render import fmt, write_outputs  # noqa: E402


DEFAULT_DATA_ROOT = Path("outputs/training-data/shahed136-public-proxy-ml-training-smoke")
DEFAULT_OUT_DIR = Path("outputs/detector-realism/benchmark-regeneration-report")
DEFAULT_REPORT = Path("detection/reports/benchmark_regeneration_report.md")
EXPECTED_PHASES = ("initial_take_up", "climb_transition", "cruise_altitude")
REQUIRED_FILES = (
    "quality_report.json",
    "phase_operational_metrics.csv",
    "acoustic_phase_metrics.csv",
    "generator_truth_denylist.json",
    "calibration_anchor_manifest.json",
    "calibration_distance.csv",
    "acoustic_cue_quality.json",
)


@dataclass(frozen=True)
class Decision:
    decision: str
    regeneration_ready: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def sort_phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {phase: idx for idx, phase in enumerate(EXPECTED_PHASES)}
    return sorted(rows, key=lambda row: (order.get(str(row.get("phase_id", "")), 99), str(row.get("phase_id", ""))))


def missing_files(data_root: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (data_root / name).exists()]


def max_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def min_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else float("nan")


def build_payload(data_root: Path) -> dict[str, Any]:
    missing = missing_files(data_root)
    if missing:
        return {
            "artifact": "benchmark-regeneration-report",
            "data_root": str(data_root),
            "decision": "no_go_missing_artifacts",
            "regeneration_ready": False,
            "missing_files": missing,
            "blockers": [
                {
                    "blocker_id": "missing_required_smoke_artifacts",
                    "status": "fail",
                    "evidence": ", ".join(missing),
                }
            ],
        }

    quality = read_json(data_root / "quality_report.json")
    truth_denylist = read_json(data_root / "generator_truth_denylist.json")
    calibration_manifest = read_json(data_root / "calibration_anchor_manifest.json")
    acoustic_quality = read_json(data_root / "acoustic_cue_quality.json")
    phase_rows = sort_phase_rows(read_csv(data_root / "phase_operational_metrics.csv"))
    acoustic_rows = sort_phase_rows(read_csv(data_root / "acoustic_phase_metrics.csv"))
    calibration_rows = read_csv(data_root / "calibration_distance.csv")

    phase_ids = sorted(str(row.get("phase_id", "")) for row in phase_rows)
    acoustic_phase_ids = sorted(str(row.get("phase_id", "")) for row in acoustic_rows)
    max_radar_pfa = max_metric(phase_rows, "pfa_any_cfar")
    max_false_track = max_metric(phase_rows, "false_track_rate")
    max_fragmentation = max_metric(phase_rows, "track_fragmentation_rate")
    max_missed_track = max_metric(phase_rows, "missed_track_rate")
    min_pd = min_metric(phase_rows, "pd_any_cfar")
    max_acoustic_pfa = max_metric(acoustic_rows, "pfa_any_acoustic_cue")

    blockers: list[dict[str, Any]] = []
    if quality.get("status") != "pass":
        blockers.append(
            {
                "blocker_id": "smoke_artifact_integrity",
                "status": "fail",
                "evidence": f"quality_report.status={quality.get('status')}",
            }
        )
    if phase_ids != sorted(EXPECTED_PHASES):
        blockers.append(
            {
                "blocker_id": "phase_coverage",
                "status": "fail",
                "evidence": f"phase rows are {phase_ids}",
            }
        )
    if acoustic_phase_ids != sorted(EXPECTED_PHASES):
        blockers.append(
            {
                "blocker_id": "acoustic_phase_coverage",
                "status": "fail",
                "evidence": f"acoustic phase rows are {acoustic_phase_ids}",
            }
        )
    if max_radar_pfa >= 0.40 or max_false_track >= 0.40:
        blockers.append(
            {
                "blocker_id": "radar_false_alarm_calibration",
                "status": "blocked",
                "evidence": f"max radar Pfa={fmt(max_radar_pfa)}, max false-track rate={fmt(max_false_track)}",
                "required_next_evidence": "empirical Pfa and false-track calibration tied to lifecycle metrics",
            }
        )
    if max_fragmentation > 1.0 or max_missed_track > 0.25:
        blockers.append(
            {
                "blocker_id": "track_lifecycle_baseline",
                "status": "blocked",
                "evidence": f"max fragmentation={fmt(max_fragmentation)}, max missed-track rate={fmt(max_missed_track)}",
                "required_next_evidence": "track initiation, deletion, fragmentation, missed-track, and saturation baseline",
            }
        )
    if quality.get("calibration_anchor_status") != "pass":
        blockers.append(
            {
                "blocker_id": "measured_anchor_promotion",
                "status": "blocked",
                "evidence": f"calibration_anchor_status={quality.get('calibration_anchor_status')}",
                "required_next_evidence": "lawful measured-anchor distributions and passing distance checks",
            }
        )
    if quality.get("calibration_artifact_status") != "pass":
        blockers.append(
            {
                "blocker_id": "calibration_artifact_integrity",
                "status": "fail",
                "evidence": f"calibration_artifact_status={quality.get('calibration_artifact_status')}",
            }
        )
    if truth_denylist.get("status") != "pass":
        blockers.append(
            {
                "blocker_id": "truth_denylist",
                "status": "fail",
                "evidence": f"violations={truth_denylist.get('violations')}",
            }
        )
    if acoustic_quality.get("status") != "pass":
        blockers.append(
            {
                "blocker_id": "acoustic_cue_quality",
                "status": "fail",
                "evidence": f"acoustic quality status={acoustic_quality.get('status')}",
            }
        )
    blockers.append(
        {
            "blocker_id": "dependency_alias_reconciliation",
            "status": "blocked",
            "evidence": (
                "active current rows still reference expired aliases while landed Rust receipts use "
                "link-budget-wire-in, complex-iq-end-to-end, k-weibull-clutter-wire-in, "
                "three-tier-phase-aware-detector, and unified-synthesize-scene"
            ),
            "required_next_evidence": "row reconciliation or explicit alias map in the regeneration receipt",
        }
    )
    blockers.append(
        {
            "blocker_id": "detector_stack_completion",
            "status": "blocked",
            "evidence": "Ku/X C-UAS, medium-range 3D surveillance, and detector fusion cue product rows remain open",
            "required_next_evidence": "source-attributed detector products and fused tracks before benchmark-facing detector regeneration",
        }
    )

    hard_blockers = [blocker for blocker in blockers if blocker["status"] in {"fail", "blocked"}]
    decision = Decision(
        decision="no_go",
        regeneration_ready=False,
        reason=(
            "current smoke artifacts are useful integrity evidence, but regeneration-ready claims are blocked by "
            "false-alarm/lifecycle calibration, reference-only calibration anchors, expired dependency aliases, "
            "and incomplete detector/fusion product lanes."
        ),
    )
    if not hard_blockers:
        decision = Decision(
            decision="go",
            regeneration_ready=True,
            reason="All configured readiness checks passed.",
        )

    resolved_integrity = {
        "quality_report_status": quality.get("status"),
        "phase_windows_status": quality.get("phase_windows_status"),
        "negative_control_status": quality.get("negative_control_status"),
        "truth_denylist_status": truth_denylist.get("status"),
        "speed_prior_kinematics_status": quality.get("speed_prior_kinematics_status"),
        "calibration_artifact_status": quality.get("calibration_artifact_status"),
        "acoustic_cueing_status": quality.get("acoustic_cueing_status"),
        "raw_audio_stored": acoustic_quality.get("raw_audio_stored"),
    }
    metrics = {
        "record_count": quality.get("record_count"),
        "scenario_group_count": quality.get("scenario_group_count"),
        "phase_ids": phase_ids,
        "min_radar_pd_any_cfar": min_pd,
        "max_radar_pfa_any_cfar": max_radar_pfa,
        "max_radar_false_track_rate": max_false_track,
        "max_radar_track_fragmentation_rate": max_fragmentation,
        "max_radar_missed_track_rate": max_missed_track,
        "max_acoustic_pfa_any_cue": max_acoustic_pfa,
        "pre_radar_los_acoustic_cue_count": acoustic_quality.get("pre_radar_los_cue_count"),
        "calibration_anchor_status": quality.get("calibration_anchor_status"),
        "calibration_validation_tier": calibration_manifest.get("validation_tier"),
        "calibration_distance_failures": [
            row.get("feature_name") for row in calibration_rows if str(row.get("status")) != "pass"
        ],
    }
    return {
        "artifact": "benchmark-regeneration-report",
        "data_root": str(data_root),
        "decision": decision.decision,
        "regeneration_ready": decision.regeneration_ready,
        "reason": decision.reason,
        "claim_boundary": (
            "This report is a regeneration readiness audit over synthetic public-proxy smoke artifacts. "
            "It is not measured radar validation, deployment guidance, or a claim of real sensor performance."
        ),
        "resolved_integrity_gates": resolved_integrity,
        "readiness_metrics": metrics,
        "phase_operational_metrics": phase_rows,
        "acoustic_phase_metrics": acoustic_rows,
        "blockers": hard_blockers,
        "next_safe_packets": [
            "track-lifecycle-baseline",
            "ku-x-band-cuas-aesa-products",
            "medium-range-3d-surveillance-products",
            "detector-fusion-cue-products",
            "lane-g-c-multidist-cfar",
        ],
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args.data_root)
    write_outputs(payload, args.out_dir, args.report)
    print(
        f"wrote {args.report} and {args.out_dir} decision={payload['decision']} "
        f"regeneration_ready={payload['regeneration_ready']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
