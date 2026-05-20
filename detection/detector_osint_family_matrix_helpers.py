"""Helper functions for OSINT family matrix detector."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from .detector_osint_family_matrix_types import DetectorFamily, MatrixConfig


def build_matrix() -> list[DetectorFamily]:
    """Build the detector family matrix."""
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
            ],
            representative_public_systems=[
                "Blighter A400",
                "Hensoldt SPEXER",
                "Rohde & Schwarz ARGUS",
                "RadarVision KuRFS",
            ],
            echoforge_priority="P0",
            implementation_focus="Implement micro-Doppler feature extraction and classification, and tune CFAR for low RCS targets.",
            claim_boundary=CLAIM_BOUNDARY,
        ),
    ]


def write_matrix_json(matrix: list[DetectorFamily], config: MatrixConfig) -> None:
    """Write matrix to JSON file."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "osint_family_matrix.json"
    with open(output_path, "w") as f:
        json.dump([asdict(family) for family in matrix], f, indent=2)


def write_matrix_markdown(matrix: list[DetectorFamily], config: MatrixConfig) -> None:
    """Write matrix to markdown report."""
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.report_path, "w") as f:
        f.write("# OSINT Family Matrix\n\n")
        f.write("Generated: {}\n\n".format(datetime.now(timezone.utc).isoformat()))
        for family in matrix:
            f.write(f"## {family.family} (Rank {family.rank})\n\n")
            f.write(f"**ID:** {family.family_id}\n\n")
            f.write(f"**Role:** {family.role}\n\n")
            f.write(f"**Bands/Modalities:** {', '.join(family.bands_or_modalities)}\n\n")
            f.write(f"**Priority:** {family.echoforge_priority}\n\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate OSINT family matrix")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--report", type=str, default=None, help="Report path")
    parser.add_argument("--include-private", action="store_true", help="Include private systems")
    return parser.parse_args()


__all__ = [
    "build_matrix",
    "write_matrix_json",
    "write_matrix_markdown",
    "parse_args",
]