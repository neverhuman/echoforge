"""Type definitions for OSINT family matrix detector."""

from __future__ import annotations

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


@dataclass
class MatrixConfig:
    """Configuration for matrix generation."""
    output_dir: Path = DEFAULT_OUT_DIR
    report_path: Path = DEFAULT_REPORT
    include_private_systems: bool = False


__all__ = [
    "DetectorFamily",
    "MatrixConfig",
    "CLAIM_BOUNDARY",
    "DEFAULT_OUT_DIR",
    "DEFAULT_REPORT",
]