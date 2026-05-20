"""Generate passive RF, EO/IR, and passive-radar backlog packets for current.

This is a strict-open public-proxy planning artifact. It does not implement a
detector, publish measured traces, or claim real sensor performance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from passive_rf_eoir_passive_radar_backlog_data import BacklogPacket, build_packets  # noqa: E402


DEFAULT_OUT_DIR = Path("outputs/detector-realism/passive-backlog")
DEFAULT_REPORT = Path("detection/reports/passive_rf_eoir_passive_radar_backlog.md")

CLAIM_BOUNDARY = (
    "Strict-open public-proxy backlog only. Packets describe modeling work, "
    "data products, and validation gates for passive/supporting sensor layers. "
    "They are not measured Shahed/Geran performance, receiver sensitivity, "
    "classified fidelity, deployment geometry, or proprietary-equivalent "
    "sensor behavior."
)


def payload(generated_at: str) -> dict[str, object]:
    packets = build_packets()
    return {
        "artifact": "passive-rf-eoir-passive-radar-backlog",
        "generated_at_utc": generated_at,
        "benchmark_context": "EchoForge Detector Realism current",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_basis": [
            "detection/RADAR_REALISM.md",
            "detection/reports/detector_osint_family_matrix.md",
            "tips/detectors/tip1.txt",
            "tips/detectors/tip2.txt",
            "tips/detectors/tip3.txt",
            "tips/detectors/tip4.txt",
            "tips/detectors/tip5.txt",
            "tips/detectors/tip6.txt",
            "tips/detectors/tip7.txt",
            "tips/detectors/tip8.txt",
        ],
        "critical_constraints": [
            "Shahed-class one-way UAVs may be RF-silent; passive RF must support explicit missing-signal behavior.",
            "EO/IR needs a separate visual asset/model policy before imagery or thermal crops become detector-facing products.",
            "Passive radar depends on illuminator geometry and must not claim own-transmitter radar performance.",
        ],
        "packets": [asdict(packet) for packet in packets],
    }


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def packet_markdown(packet: BacklogPacket) -> str:
    return f"""### `{packet.packet_id}`: {packet.title}

| Field | Value |
|---|---|
| Priority | `{packet.priority}` |
| Modality | {packet.modality} |
| Role | {packet.role} |

#### Rationale

{bullet_list(packet.rationale)}

#### Implementation Blockers

{bullet_list(packet.implementation_blockers)}

#### Expected Data Products

{bullet_list(packet.expected_data_products)}

#### False-Alarm Controls

{bullet_list(packet.false_alarm_controls)}

#### Claim Boundaries

{bullet_list(packet.claim_boundaries)}

#### Dependencies

{bullet_list(packet.dependencies)}

#### Acceptance Evidence

{bullet_list(packet.acceptance_evidence)}
"""


def build_markdown(data: dict[str, object]) -> str:
    generated_at = data["generated_at_utc"]
    sources = bullet_list([f"`{source}`" for source in data["source_basis"]])
    constraints = bullet_list(data["critical_constraints"])  # type: ignore[arg-type]
    packets = "\n".join(packet_markdown(packet) for packet in build_packets())

    return f"""# Passive RF, EO/IR, and Passive Radar Backlog current

Generated: `{generated_at}`

This report defines backlog packets for the passive and confirmatory sensing
layers behind EchoForge Detector Realism current. These packets are implementation
planning artifacts for public-proxy data products, false-alarm controls, and
claim boundaries. They do not replace the active radar, acoustic cueing, or C2
fusion work.

## Claim Boundary

{data["claim_boundary"]}

## Source Basis

{sources}

## Critical Constraints

{constraints}

## Backlog Packets

{packets}
## Integration Notes

- Keep passive RF, EO/IR, and passive radar as provenance-rich sidecars until
  fusion semantics are defined.
- Publish generated JSON/Markdown copies only under `outputs/`; this committed
  report is the reviewable planning artifact.
- Use the current three-tier phase IDs when later packets add metrics:
  `initial_take_up`, `climb_transition`, and `cruise_altitude`.
- Treat hard negatives as robustness work: RFI, crowded emitters, birds,
  balloons, lights, weather, towers, multipath, and illuminator outages should
  increase false-alarm accounting rather than encode evasion logic.
"""


def write_artifacts(out_dir: Path, report_path: Path | None) -> tuple[Path, Path, Path | None]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = payload(generated_at)
    markdown = build_markdown(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "passive_rf_eoir_passive_radar_backlog.json"
    md_path = out_dir / "passive_rf_eoir_passive_radar_backlog.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    committed_report = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")
        committed_report = report_path

    return json_path, md_path, committed_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Generated artifact directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Committed Markdown report path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only write generated JSON/Markdown under --out-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = None if args.no_report else args.report
    json_path, md_path, committed_report = write_artifacts(args.out_dir, report_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if committed_report is not None:
        print(f"wrote {committed_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
