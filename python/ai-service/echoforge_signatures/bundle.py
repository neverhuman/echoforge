from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List, Optional


@dataclass
class EchoSigManifest:
    artifact_id: str
    version: str = "v0"
    validation_tier: str = "placeholder"
    axes: List[Dict[str, Any]] = field(default_factory=list)
    tensor_files: List[str] = field(default_factory=list)
    dynamic_files: List[str] = field(default_factory=list)
    qa_files: List[str] = field(default_factory=list)


@dataclass
class ProvenanceRecord:
    source: str = "synthetic"
    generated_by: str = "echoforge-sig"
    created_at_utc: str = "1970-01-01T00:00:00Z"
    seed: int = 0
    lineage: List[str] = field(default_factory=list)


@dataclass
class LicenseRecord:
    expression: str = "Apache-2.0"
    spdx_id: Optional[str] = "Apache-2.0"
    notes: Optional[str] = "placeholder license record for synthetic bundle"


@dataclass
class EchoSigArtifactBundle:
    manifest: EchoSigManifest
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    license: LicenseRecord = field(default_factory=LicenseRecord)
    object_card: Optional[str] = None
    material_card: Optional[str] = None
    solver_card: Optional[str] = None


def write_bundle(root: str | Path, bundle: EchoSigArtifactBundle) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "manifest.json").write_text(json.dumps(bundle.manifest.__dict__, indent=2))
    (root_path / "provenance.json").write_text(json.dumps(bundle.provenance.__dict__, indent=2))
    (root_path / "license.json").write_text(json.dumps(bundle.license.__dict__, indent=2))
    if bundle.object_card is not None:
        (root_path / "object_card.yaml").write_text(bundle.object_card)
    if bundle.material_card is not None:
        (root_path / "material_card.yaml").write_text(bundle.material_card)
    if bundle.solver_card is not None:
        (root_path / "solver_card.yaml").write_text(bundle.solver_card)


def read_bundle(root: str | Path) -> EchoSigArtifactBundle:
    root_path = Path(root)
    manifest = EchoSigManifest(**json.loads((root_path / "manifest.json").read_text()))
    provenance = ProvenanceRecord(**json.loads((root_path / "provenance.json").read_text()))
    license_record = LicenseRecord(**json.loads((root_path / "license.json").read_text()))
    object_card = _read_optional_text(root_path / "object_card.yaml")
    material_card = _read_optional_text(root_path / "material_card.yaml")
    solver_card = _read_optional_text(root_path / "solver_card.yaml")
    return EchoSigArtifactBundle(
        manifest=manifest,
        provenance=provenance,
        license=license_record,
        object_card=object_card,
        material_card=material_card,
        solver_card=solver_card,
    )


def placeholder_analytic_report() -> Dict[str, Any]:
    cases = []
    for primitive, frequency_hz in [
        ("pec_sphere", 10.0e9),
        ("flat_plate", 9.6e9),
        ("dihedral", 9.2e9),
        ("trihedral", 8.8e9),
        ("cylinder", 8.4e9),
        ("cone", 8.0e9),
    ]:
        cases.append(
            {
                "primitive": primitive,
                "frequency_hz": frequency_hz,
                "expected_rcs_dbsm": None,
                "measured_rcs_dbsm": None,
                "tolerance_db": 0.0,
                "status": "pending",
                "notes": "placeholder validation record",
            }
        )
    return {
        "validation_tier": "placeholder",
        "cases": cases,
        "passed": False,
        "summary": "analytic validation placeholders are wired but not yet scored",
    }


def _read_optional_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text()

