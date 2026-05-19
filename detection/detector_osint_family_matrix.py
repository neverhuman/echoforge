"""Generate the Detector Realism current public OSINT family matrix.

This artifact is intentionally a public-proxy planning matrix. It summarizes
openly described detector families and does not claim measured sensor truth,
classified fidelity, or proprietary-equivalent behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUT_DIR = Path("outputs/detector-realism/osint-family-matrix")
DEFAULT_REPORT = Path("detection/reports/detector_osint_family_matrix.md")


@dataclass(frozen=True)
class DetectorFamily:
    rank: int
    family_id: str
    family: str
    role: str
    bands_or_modalities: list[str]
    public_range_proxy: str
    update_rate_proxy: str
    track_capacity_proxy: str
    clutter_false_alarm_controls: list[str]
    data_products: list[str]
    representative_public_systems: list[str]
    echoforge_priority: str
    implementation_focus: str
    claim_boundary: str


CLAIM_BOUNDARY = (
    "Strict-open public-proxy summary only. Values are vendor/government/media "
    "range or capacity proxies, not guaranteed Shahed/Geran detection ranges, "
    "receiver sensitivity, Pd/Pfa curves, classified modes, deployment geometry, "
    "or proprietary sensor behavior."
)


def build_matrix() -> list[DetectorFamily]:
    return [
        DetectorFamily(
            rank=1,
            family_id="fusion_c2_layered_architecture",
            family="Layered C2 and sensor-fusion architectures",
            role="Correlate heterogeneous radar, acoustic, EO/IR, RF, and external tracks for alerting, classification, and effector handoff.",
            bands_or_modalities=["multi-sensor", "C2", "track fusion"],
            public_range_proxy="No single range; inherits sensor envelopes. Public examples combine Ku/L-band radars, EO/IR, passive RF, acoustic, and external feeds.",
            update_rate_proxy="System-level real-time correlation; public details usually describe track correlation/cueing rather than fixed Hz.",
            track_capacity_proxy="Architecture dependent; relevant proxy is multi-sensor correlation and weapon-target pairing under saturation.",
            clutter_false_alarm_controls=[
                "cross-sensor confirmation",
                "track correlation and deconfliction",
                "IFF/external feed correlation where available",
                "false-track mitigation and operator confirmation",
            ],
            data_products=[
                "fused track id",
                "source provenance",
                "classification confidence",
                "cue messages",
                "engagement handoff state",
            ],
            representative_public_systems=[
                "U.S. LIDS / FS-LIDS with FAAD C2",
                "Ukraine Sky Map / Sky Fortress-style fusion",
                "EDGE/SIGN4L SKYSHIELD C2",
            ],
            echoforge_priority="P0",
            implementation_focus="Make fusion the top-level benchmark artifact: track provenance, source confidence, expired-track handling, and public-proxy false-track accounting.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=2,
            family_id="high_resolution_cuas_x_ku",
            family="High-resolution X/Ku-band C-UAS radars",
            role="Terminal-area detection, track formation, drone/bird discrimination, and fire-control-quality cueing for low/slow/small targets.",
            bands_or_modalities=["X-band", "Ku-band", "pulse-Doppler", "FMCW", "AESA/e-scan"],
            public_range_proxy="Public proxies include nano/small UAV kilometer-scale examples, Blighter A400 RCS-class ranges, SPEXER small-UAV classification claims, and KuRFS small-object discrimination claims.",
            update_rate_proxy="Fast revisit or persistent sector/hemisphere coverage; some public systems advertise sub-second to 10 Hz-class track products.",
            track_capacity_proxy="Representative public claims include hundreds of tracks or more than 300 tracks per sector for some systems.",
            clutter_false_alarm_controls=[
                "Doppler filtering",
                "micro-Doppler or spectral classification",
                "CFAR/adaptive thresholds",
                "biological/non-biological discrimination",
                "subclutter visibility claims",
            ],
            data_products=[
                "range",
                "azimuth",
                "elevation",
                "radial velocity",
                "RCS/amplitude proxy",
                "micro-Doppler features",
                "track quality",
            ],
            representative_public_systems=[
                "RTX KuRFS / KuMRFS",
                "HENSOLDT SPEXER 2000 3D MkIII",
                "Blighter A400/A800",
                "Echodyne EchoShield",
                "Robin Radar IRIS",
                "Saab Giraffe 1X",
            ],
            echoforge_priority="P0",
            implementation_focus="Model as the primary low-altitude discriminator: range-Doppler cube features, micro-Doppler confidence, clutter class, revisit cadence, and false-alarm budget.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=3,
            family_id="tactical_multimission_s_band_aesa",
            family="Tactical S-band multi-mission AESA radars",
            role="Mobile C-UAS, VSHORAD, C-RAM, and defended-site hemispheric surveillance.",
            bands_or_modalities=["S-band", "4D radar", "AESA", "GaN", "pulse-Doppler"],
            public_range_proxy="RADA MHR public examples include nano UAV 5 km and medium UAV 25 km; Green Rock publishes ultralight-UAV range classes by configuration.",
            update_rate_proxy="Green Rock public refresh proxy: 1 Hz search with priority target updates in the 4-10 Hz class; MHR uses track-while-scan/revisit modes.",
            track_capacity_proxy="MHR-family public material describes hundreds of tracks; configuration dependent.",
            clutter_false_alarm_controls=[
                "advanced clutter and multipath mitigation",
                "Doppler processing",
                "ECCM/IFF support where integrated",
                "multi-panel 360 degree correlation",
            ],
            data_products=[
                "4D tracks",
                "target class proxy",
                "track-while-scan state",
                "C-RAM/C-UAS mission tags",
                "sensor health",
            ],
            representative_public_systems=[
                "Leonardo DRS / RADA MHR, nMHR, exMHR",
                "ELTA Green Rock",
                "ELTA MMR / MS-MMR family",
            ],
            echoforge_priority="P0",
            implementation_focus="Use as the reference medium tactical radar family with target-class range bins, configurable update priority, and clutter/multipath stress cases.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=4,
            family_id="medium_range_gbad_3d_4d",
            family="Medium-range 3D/4D GBAD and SHORAD surveillance radars",
            role="Wide defended-area air picture, low-altitude gap coverage, and cueing for NASAMS/IRIS-T/SHORAD-type networks.",
            bands_or_modalities=["L/S/C/X/G-band public labels", "3D/4D", "rotating or AESA"],
            public_range_proxy="Public examples include Sentinel 75-120 km instrumented range, TRML-4D 250 km instrumented range with fighter and missile proxies, GM200 250-400 km class surveillance proxies, and Giraffe 4A 400 km instrumented range.",
            update_rate_proxy="Public proxies include 1 s 360 degree revisit for Giraffe-family radars and 1.5-3 s mode-dependent GM200 updates.",
            track_capacity_proxy="TRML-4D public proxy: more than 1,500 tracks; Sentinel F1 public proxy: more than 60 tracks.",
            clutter_false_alarm_controls=[
                "track-while-scan",
                "low-altitude target processing",
                "Mode 5/IFF where integrated",
                "sector prioritization",
                "C2 correlation",
            ],
            data_products=[
                "regional 3D/4D track",
                "air-defense cue",
                "classification proxy",
                "track confidence",
                "handoff to fire-control layer",
            ],
            representative_public_systems=[
                "AN/MPQ-64 Sentinel",
                "HENSOLDT TRML-4D",
                "Thales GM200",
                "Saab Giraffe AMB / 4A",
            ],
            echoforge_priority="P1",
            implementation_focus="Represent as cueing and context sensors: radar-horizon masks, revisit delay, lower-resolution classification, and high-capacity saturation queues.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=5,
            family_id="l_band_counterfire_gap_filler",
            family="L-band counterfire and gap-filler multi-mission radars",
            role="Compact 360 degree volume surveillance for C-RAM, C-UAS, counterfire, and low-altitude warning.",
            bands_or_modalities=["L-band", "electronically steered", "coherent pulse-Doppler"],
            public_range_proxy="AN/TPQ-50 public proxy: more than 35 km air-surveillance instrumented range and up to 15 km counterfire/RAM range.",
            update_rate_proxy="Continuous 360 degree surveillance is the public proxy; exact dwell/update behavior is not publicly specified.",
            track_capacity_proxy="Multiple simultaneous incoming weapons and air targets; exact public capacity varies by source and mode.",
            clutter_false_alarm_controls=[
                "coherent pulse-Doppler processing",
                "automatic discrimination/geolocation",
                "mission-mode filtering",
                "C2 correlation in LIDS-like stacks",
            ],
            data_products=[
                "3D target location",
                "air-surveillance track",
                "point-of-origin/point-of-impact for RAM",
                "mission mode",
            ],
            representative_public_systems=["SRC AN/TPQ-50", "LCMR-derived multi-mission radar family"],
            echoforge_priority="P1",
            implementation_focus="Add as a broad 360 degree gap-filler with coarser classification than X/Ku C-UAS but strong cueing value.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=6,
            family_id="distributed_acoustic_networks",
            family="Distributed acoustic detection networks",
            role="Low-cost passive cueing for piston/propeller one-way attack UAVs in radar-horizon or clutter gaps.",
            bands_or_modalities=["acoustic", "microphone arrays", "edge ML", "bearing/time correlation"],
            public_range_proxy="No universal range. Public Ukraine examples emphasize thousands of low-cost sensors and regional cueing, not guaranteed single-node range.",
            update_rate_proxy="Event-driven detections with bearing/time records; fusion cadence depends on network and communications.",
            track_capacity_proxy="Network-level capacity, not radar track count; saturation behavior depends on sensor density and fusion backend.",
            clutter_false_alarm_controls=[
                "engine/propeller acoustic classifiers",
                "bearing/time triangulation",
                "human or C2 confirmation",
                "radar/visual correlation",
                "local-noise rejection",
            ],
            data_products=[
                "bearing estimate",
                "time of arrival",
                "acoustic class confidence",
                "triangulated cue",
                "sensor node provenance",
            ],
            representative_public_systems=["Ukraine Sky Fortress / Zvook-style networks", "Sky Map acoustic fusion"],
            echoforge_priority="P1",
            implementation_focus="Implement as passive cue generation with false positives from traffic, weather, industrial noise, and friendly aircraft.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=7,
            family_id="eo_ir_confirmation_layer",
            family="EO/IR confirmation and precision cueing sensors",
            role="Visual/thermal confirmation, classification, and fire-control support after radar/acoustic/RF cueing.",
            bands_or_modalities=["visible EO", "thermal IR", "PTZ", "image-based classifiers"],
            public_range_proxy="Kilometer-scale public ID/recognition/detection claims vary heavily by optic, weather, aspect, and cue quality.",
            update_rate_proxy="Video frame-rate products; slew-to-cue latency is more important than raw frame rate.",
            track_capacity_proxy="Usually one or few high-quality confirmed tracks per gimbal; wide-area cameras vary by design.",
            clutter_false_alarm_controls=[
                "human-in-the-loop confirmation",
                "thermal/visual cross-check",
                "radar slew-to-cue",
                "background/weather quality gating",
            ],
            data_products=[
                "image crop",
                "line-of-sight angle",
                "visual class confidence",
                "operator confirmation state",
                "track refinement",
            ],
            representative_public_systems=["LIDS EO/IR cameras", "Drone Dome EO/IR", "SKYSHIELD EO/IR", "AUDS EO/IR"],
            echoforge_priority="P1",
            implementation_focus="Treat as a confirmatory sensor with weather/visibility masks and cue latency rather than wide-area primary detection.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=8,
            family_id="passive_rf_esm_direction_finding",
            family="Passive RF / ESM / direction-finding sensors",
            role="Detect, classify, or locate drone RF control/video/telemetry emissions when present.",
            bands_or_modalities=["passive RF", "ESM", "direction finding", "COMINT"],
            public_range_proxy="Emission- and environment-dependent; not reliable as a primary range proxy for pre-programmed RF-silent one-way UAVs.",
            update_rate_proxy="Scan/dwell dependent; public sources usually describe continuous monitoring and direction finding.",
            track_capacity_proxy="Depends on emitter density and channelization; public capacity is rarely comparable across vendors.",
            clutter_false_alarm_controls=[
                "protocol/frequency classification",
                "direction-of-arrival correlation",
                "emitter library matching",
                "radar/EO/acoustic confirmation",
            ],
            data_products=[
                "bearing",
                "emitter class",
                "frequency band",
                "signal strength proxy",
                "operator/controller cue where applicable",
            ],
            representative_public_systems=["SRC EW in LIDS", "Drone Dome RF sensors", "SKYSHIELD direction finders"],
            echoforge_priority="P2",
            implementation_focus="Model as optional supporting evidence with explicit missing-signal behavior for autonomous public-proxy one-way attack UAVs.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=9,
            family_id="passive_radar_multistatic",
            family="Passive radar and multistatic illuminator-of-opportunity sensors",
            role="Silent gap-filler surveillance using broadcast/cellular/other illuminators rather than own transmitter.",
            bands_or_modalities=["VHF/UHF broadcast", "FM/DAB/DVB-T", "cellular", "passive bistatic/multistatic radar"],
            public_range_proxy="Geometry- and illuminator-dependent; public examples include passive 3D tracking systems and micro-UAV claims, but not universal target-class envelopes.",
            update_rate_proxy="Some public systems advertise sub-second or near-real-time tracking; actual cadence depends on illuminators and processing.",
            track_capacity_proxy="Cluster/fusion dependent; public capacity is less standardized than active radar.",
            clutter_false_alarm_controls=[
                "bistatic/multistatic consistency",
                "illuminator quality gating",
                "Doppler processing",
                "active-radar cue correlation",
            ],
            data_products=[
                "passive track",
                "bistatic range/Doppler proxy",
                "illuminator provenance",
                "track confidence",
            ],
            representative_public_systems=["HENSOLDT Twinvis / TwinSens", "Leonardo AULOS", "Chinese JY/YLC passive families"],
            echoforge_priority="P2",
            implementation_focus="Implement after active/acoustic layers as a silent cue source with illuminator geometry and urban multipath assumptions.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=10,
            family_id="organic_shorad_fire_control_radars",
            family="Organic SHORAD search/tracking and fire-control radars",
            role="Vehicle or battery organic search, tracking, and engagement support for guns/missiles in point defense.",
            bands_or_modalities=["S/X/Ku/K public labels", "search radar", "tracking radar", "EO/thermal backup"],
            public_range_proxy="Public examples include Gepard 15 km airspace monitoring and 5 km gun engagement; Pantsir/Tor public sources publish tens-of-kilometers class aircraft/UAV proxies by variant.",
            update_rate_proxy="Engagement-system dependent; public data usually focuses on search/track/engage ranges rather than radar update rates.",
            track_capacity_proxy="Public examples include Tor scanning/tracking subsets and Pantsir tactical-aircraft-sized track counts; exact small-UAV capacity is context dependent.",
            clutter_false_alarm_controls=[
                "separate search and tracking sensors",
                "EO/thermal backup",
                "operator/C2 confirmation",
                "fire-control track gates",
            ],
            data_products=[
                "local search track",
                "fire-control track",
                "engagement state",
                "sensor handoff status",
            ],
            representative_public_systems=["Gepard", "Pantsir family", "Tor family", "SkyKnight / Skynex inner-layer systems"],
            echoforge_priority="P2",
            implementation_focus="Represent as downstream engagement-quality consumers of fused tracks, with local radar recovery and saturation limits.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DetectorFamily(
            rank=11,
            family_id="strategic_long_range_missile_defense_radars",
            family="Strategic and high-end missile-defense radars",
            role="High-value air and missile defense surveillance/discrimination, not cost-matched primary Shahed-class point defense.",
            bands_or_modalities=["X-band", "C/G-band public labels", "large phased array", "BMD/IAMD"],
            public_range_proxy="Public sources publish high-end air/missile-defense ranges and target capacities, but these are not Shahed-specific public-proxy ranges.",
            update_rate_proxy="System-specific and often not public; integration through IAMD/C2 is the relevant proxy.",
            track_capacity_proxy="High-capacity IAMD tracking by design; exact low-altitude small-UAV behavior is not publicly specified.",
            clutter_false_alarm_controls=[
                "large-aperture discrimination",
                "IAMD track correlation",
                "threat classification",
                "high-value asset doctrine",
            ],
            data_products=[
                "IAMD track",
                "threat class",
                "engagement-quality data",
                "discrimination output",
            ],
            representative_public_systems=["Patriot AN/MPQ-53/65", "LTAMDS / GhostEye", "AN/TPY-2"],
            echoforge_priority="P3",
            implementation_focus="Keep as context/outer-layer cueing only unless a scenario explicitly tests high-value IAMD integration.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
    ]


def matrix_payload(generated_at: str) -> dict[str, object]:
    matrix = build_matrix()
    return {
        "artifact": "detector-osint-family-matrix",
        "generated_at_utc": generated_at,
        "source_basis": [
            "tips/detectors/tip1.txt",
            "tips/detectors/tip2.txt",
            "tips/detectors/tip3.txt",
            "tips/detectors/tip4.txt",
            "tips/detectors/tip5.txt",
            "tips/detectors/tip6.txt",
            "tips/detectors/tip7.txt",
            "tips/detectors/tip8.txt",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "priority_meaning": {
            "P0": "Implement first for Detector Realism current benchmarks.",
            "P1": "Implement in current if scope allows; needed for realistic layered operation.",
            "P2": "Supporting or specialized layer; model after primary radar/fusion surfaces.",
            "P3": "Context layer only for current unless explicitly needed.",
        },
        "families": [asdict(row) for row in matrix],
    }


def markdown_table(rows: list[DetectorFamily]) -> str:
    header = (
        "| Rank | Family | Role | Band / modality | Public range proxy | Update | Tracks | "
        "False-alarm controls | Products | EchoForge priority |\n"
        "|---:|---|---|---|---|---|---|---|---|---|"
    )
    body = []
    for row in rows:
        body.append(
            "| {rank} | `{family_id}`<br>{family} | {role} | {bands} | {range_proxy} | "
            "{update} | {tracks} | {controls} | {products} | **{priority}**<br>{focus} |".format(
                rank=row.rank,
                family_id=row.family_id,
                family=row.family,
                role=row.role,
                bands="<br>".join(row.bands_or_modalities),
                range_proxy=row.public_range_proxy,
                update=row.update_rate_proxy,
                tracks=row.track_capacity_proxy,
                controls="<br>".join(row.clutter_false_alarm_controls),
                products="<br>".join(row.data_products),
                priority=row.echoforge_priority,
                focus=row.implementation_focus,
            )
        )
    return "\n".join([header, *body])


def build_markdown(payload: dict[str, object]) -> str:
    rows = build_matrix()
    generated_at = payload["generated_at_utc"]
    sources = "\n".join(f"- `{source}`" for source in payload["source_basis"])
    representatives = []
    for row in rows:
        representatives.append(
            f"- `{row.family_id}`: " + "; ".join(row.representative_public_systems)
        )

    return f"""# Detector OSINT Family Matrix current

Generated: `{generated_at}`

This report ranks public-source detector families for EchoForge Detector Realism
current. It treats detector realism as a layered sensing and fusion problem: wide-area
radars provide warning, high-resolution C-UAS radars form and classify difficult
low-altitude tracks, acoustic/RF/EO layers add supporting evidence, and C2
fusion controls false tracks and handoff.

## Claim Boundary

{payload["claim_boundary"]}

## Source Basis

{sources}

## Priority Meaning

| Priority | Meaning |
|---|---|
| P0 | Implement first for Detector Realism current benchmarks. |
| P1 | Implement in current if scope allows; needed for realistic layered operation. |
| P2 | Supporting or specialized layer; model after primary radar/fusion surfaces. |
| P3 | Context layer only for current unless explicitly needed. |

## Ranked Family Matrix

{markdown_table(rows)}

## Representative Public Systems

{chr(10).join(representatives)}

## EchoForge Implementation Notes

- Start with fused track provenance, because false-track handling and cue
  confidence are the realism surface shared by every sensor family.
- Keep public range values as broad proxies by target class or system role; do
  not encode them as guaranteed Shahed/Geran detection truth.
- Separate `detection`, `tracking`, and `classification` products. Public
  sources often advertise one while omitting the others.
- Model clutter and hard negatives as first-class conditions: birds, weather,
  terrain masking, wind turbines, vehicles, multipath, RFI, and acoustic noise.
- Treat passive RF as optional evidence. Pre-programmed one-way attack UAVs may
  have no useful control-link emission for a passive RF sensor to detect.
- Keep generated JSON/Markdown copies under `outputs/`; only this committed
  report belongs in Git.
"""


def write_artifacts(out_dir: Path, report_path: Path | None) -> tuple[Path, Path, Path | None]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = matrix_payload(generated_at)
    markdown = build_markdown(payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "detector_osint_family_matrix.json"
    md_path = out_dir / "detector_osint_family_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
