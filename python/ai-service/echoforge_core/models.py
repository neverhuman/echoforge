from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, ClassVar, Mapping

from .validation import (
    ValidationError,
    canonical_json,
    deterministic_id,
    ensure_non_empty,
    ensure_non_empty_list,
    ensure_probability,
    ensure_slug,
    fingerprint_sha256,
)


def _coerce(field_type: Any, value: Any) -> Any:
    origin = getattr(field_type, "__origin__", None)
    if is_dataclass(field_type) and isinstance(value, Mapping):
        return field_type.from_dict(value)  # type: ignore[attr-defined]
    if origin is list and isinstance(value, list):
        inner = field_type.__args__[0]
        return [_coerce(inner, item) for item in value]
    return value


def _split_document_data(data: Mapping[str, Any], field_names: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    common_keys = {"id", "kind", "schema_version", "public_proxy_id", "provenance", "license", "validation"}
    meta = {key: data[key] for key in common_keys if key in data}
    body = {key: data[key] for key in field_names if key in data}
    return meta, body


@dataclass
class Provenance:
    source_kind: str = "synthetic"
    source_refs: list[str] = None  # type: ignore[assignment]
    generated_by: str = ""
    generated_at: str = ""
    fingerprint_sha256: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            source_kind=data.get("source_kind", "synthetic"),
            source_refs=list(data.get("source_refs", [])),
            generated_by=data.get("generated_by", ""),
            generated_at=data.get("generated_at", ""),
            fingerprint_sha256=data.get("fingerprint_sha256", ""),
        )

    def validate(self) -> None:
        ensure_non_empty(self.generated_by, "generated_by")
        ensure_non_empty(self.generated_at, "generated_at")
        ensure_non_empty_list(self.source_refs or [], "source_refs")
        ensure_slug(self.source_kind, "source_kind")
        if len(self.fingerprint_sha256) not in (0, 64):
            raise ValidationError("fingerprint_sha256 must be 64 hex chars when present")


@dataclass
class LicenseInfo:
    spdx_id: str = ""
    notice: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LicenseInfo":
        return cls(spdx_id=data.get("spdx_id", ""), notice=data.get("notice", ""))

    def validate(self) -> None:
        ensure_non_empty(self.spdx_id, "spdx_id")


@dataclass
class ValidationCheck:
    name: str = ""
    status: str = "warn"
    message: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationCheck":
        return cls(
            name=data.get("name", ""),
            status=data.get("status", "warn"),
            message=data.get("message", ""),
        )

    def validate(self) -> None:
        ensure_non_empty(self.name, "name")
        ensure_non_empty(self.status, "status")


@dataclass
class ValidationInfo:
    tier: str = "unvalidated"
    status: str = "warn"
    uncertainty_score: float = 0.0
    checks: list[ValidationCheck] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationInfo":
        return cls(
            tier=data.get("tier", "unvalidated"),
            status=data.get("status", "warn"),
            uncertainty_score=float(data.get("uncertainty_score", 0.0)),
            checks=[ValidationCheck.from_dict(item) for item in data.get("checks", [])],
        )

    def validate(self) -> None:
        ensure_non_empty(self.tier, "tier")
        ensure_non_empty(self.status, "status")
        ensure_probability(self.uncertainty_score, "uncertainty_score")
        ensure_non_empty_list(self.checks or [], "checks")
        for check in self.checks or []:
            check.validate()


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Vector3":
        return cls(float(data["x"]), float(data["y"]), float(data["z"]))


@dataclass
class NumericRange:
    min: float
    max: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NumericRange":
        return cls(float(data["min"]), float(data["max"]))


@dataclass
class ComplexScalar:
    real: float
    imag: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComplexScalar":
        return cls(float(data["real"]), float(data["imag"]))


class DocumentMixin:
    KIND: ClassVar[str]
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]]

    id: str
    kind: str
    schema_version: str
    public_proxy_id: str
    provenance: Provenance
    license: LicenseInfo
    validation: ValidationInfo

    def validate_common(self) -> None:
        ensure_slug(self.kind, "kind")
        ensure_slug(self.public_proxy_id, "public_proxy_id")
        self.provenance.validate()
        self.license.validate()
        self.validation.validate()

    def identity_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.IDENTITY_FIELDS}

    def finalize(self):
        self.validate_common()
        for name in self.IDENTITY_FIELDS:
            value = getattr(self, name)
            if is_dataclass(value):
                value = value
        payload = {name: getattr(self, name) for name in self.IDENTITY_FIELDS}
        expected_fingerprint = fingerprint_sha256(payload)
        if not self.provenance.fingerprint_sha256:
            self.provenance.fingerprint_sha256 = expected_fingerprint
        elif self.provenance.fingerprint_sha256 != expected_fingerprint:
            raise ValidationError("provenance fingerprint does not match payload")
        expected_id = deterministic_id(self.KIND, self.public_proxy_id, payload)
        if not self.id:
            self.id = expected_id
        elif self.id != expected_id:
            raise ValidationError("deterministic id does not match payload")
        self.kind = self.KIND
        self.schema_version = "1.0.0"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def _from_common(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": data.get("id", ""),
            "kind": data.get("kind", cls.KIND),
            "schema_version": data.get("schema_version", "1.0.0"),
            "public_proxy_id": data.get("public_proxy_id", ""),
            "provenance": Provenance.from_dict(data.get("provenance", {})),
            "license": LicenseInfo.from_dict(data.get("license", {})),
            "validation": ValidationInfo.from_dict(data.get("validation", {})),
        }


@dataclass
class ObjectCard(DocumentMixin):
    KIND: ClassVar[str] = "object_card"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "display_name",
        "object_family",
        "geometry_variant",
        "material_variant",
        "dimensions_m",
        "tags",
    )

    id: str = ""
    kind: str = "object_card"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    display_name: str = ""
    object_family: str = ""
    geometry_variant: str = ""
    material_variant: str = ""
    dimensions_m: Vector3 = None  # type: ignore[assignment]
    tags: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectCard":
        common = cls._from_common(data)
        return cls(
            **common,
            display_name=data.get("display_name", ""),
            object_family=data.get("object_family", ""),
            geometry_variant=data.get("geometry_variant", ""),
            material_variant=data.get("material_variant", ""),
            dimensions_m=Vector3.from_dict(data.get("dimensions_m", {"x": 0, "y": 0, "z": 0})),
            tags=list(data.get("tags", [])),
        ).finalize()


@dataclass
class MaterialCard(DocumentMixin):
    KIND: ClassVar[str] = "material_card"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "material_name",
        "material_family",
        "frequency_range_hz",
        "permittivity",
        "conductivity_s_per_m",
        "roughness_m",
    )

    id: str = ""
    kind: str = "material_card"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    material_name: str = ""
    material_family: str = ""
    frequency_range_hz: NumericRange = None  # type: ignore[assignment]
    permittivity: ComplexScalar = None  # type: ignore[assignment]
    conductivity_s_per_m: float = 0.0
    roughness_m: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterialCard":
        common = cls._from_common(data)
        return cls(
            **common,
            material_name=data.get("material_name", ""),
            material_family=data.get("material_family", ""),
            frequency_range_hz=NumericRange.from_dict(data.get("frequency_range_hz", {"min": 0, "max": 0})),
            permittivity=ComplexScalar.from_dict(data.get("permittivity", {"real": 0, "imag": 0})),
            conductivity_s_per_m=float(data.get("conductivity_s_per_m", 0.0)),
            roughness_m=float(data.get("roughness_m", 0.0)),
        ).finalize()


@dataclass
class MeshManifest(DocumentMixin):
    KIND: ClassVar[str] = "mesh_manifest"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "mesh_name",
        "mesh_format",
        "source_files",
        "units",
        "triangle_count",
        "watertight",
        "mesh_sha256",
    )

    id: str = ""
    kind: str = "mesh_manifest"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    mesh_name: str = ""
    mesh_format: str = ""
    source_files: list[str] = None  # type: ignore[assignment]
    units: str = ""
    triangle_count: int = 0
    watertight: bool = False
    mesh_sha256: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshManifest":
        common = cls._from_common(data)
        return cls(
            **common,
            mesh_name=data.get("mesh_name", ""),
            mesh_format=data.get("mesh_format", ""),
            source_files=list(data.get("source_files", [])),
            units=data.get("units", ""),
            triangle_count=int(data.get("triangle_count", 0)),
            watertight=bool(data.get("watertight", False)),
            mesh_sha256=data.get("mesh_sha256", ""),
        ).finalize()


@dataclass
class SolverCard(DocumentMixin):
    KIND: ClassVar[str] = "solver_card"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "solver_name",
        "solver_family",
        "version",
        "container_image",
        "supported_polarizations",
    )

    id: str = ""
    kind: str = "solver_card"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    solver_name: str = ""
    solver_family: str = ""
    version: str = ""
    container_image: str = ""
    supported_polarizations: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolverCard":
        common = cls._from_common(data)
        return cls(
            **common,
            solver_name=data.get("solver_name", ""),
            solver_family=data.get("solver_family", ""),
            version=data.get("version", ""),
            container_image=data.get("container_image", ""),
            supported_polarizations=list(data.get("supported_polarizations", [])),
        ).finalize()


@dataclass
class RcsCampaign(DocumentMixin):
    KIND: ClassVar[str] = "rcs_campaign"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "campaign_name",
        "object_card_id",
        "solver_card_id",
        "frequency_range_hz",
        "azimuth_deg",
        "tx_polarization",
        "rx_polarization",
        "run_count",
    )

    id: str = ""
    kind: str = "rcs_campaign"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    campaign_name: str = ""
    object_card_id: str = ""
    solver_card_id: str = ""
    frequency_range_hz: NumericRange = None  # type: ignore[assignment]
    azimuth_deg: NumericRange = None  # type: ignore[assignment]
    tx_polarization: str = ""
    rx_polarization: str = ""
    run_count: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RcsCampaign":
        common = cls._from_common(data)
        return cls(
            **common,
            campaign_name=data.get("campaign_name", ""),
            object_card_id=data.get("object_card_id", ""),
            solver_card_id=data.get("solver_card_id", ""),
            frequency_range_hz=NumericRange.from_dict(data.get("frequency_range_hz", {"min": 0, "max": 0})),
            azimuth_deg=NumericRange.from_dict(data.get("azimuth_deg", {"min": 0, "max": 0})),
            tx_polarization=data.get("tx_polarization", ""),
            rx_polarization=data.get("rx_polarization", ""),
            run_count=int(data.get("run_count", 0)),
        ).finalize()


@dataclass
class EchosigManifest(DocumentMixin):
    KIND: ClassVar[str] = "echosig_manifest"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "artifact_name",
        "object_card_id",
        "tensor_axes",
        "tensor_paths",
    )

    id: str = ""
    kind: str = "echosig_manifest"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    artifact_name: str = ""
    object_card_id: str = ""
    tensor_axes: list[str] = None  # type: ignore[assignment]
    tensor_paths: list[str] = None  # type: ignore[assignment]
    qa_paths: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EchosigManifest":
        common = cls._from_common(data)
        return cls(
            **common,
            artifact_name=data.get("artifact_name", ""),
            object_card_id=data.get("object_card_id", ""),
            tensor_axes=list(data.get("tensor_axes", [])),
            tensor_paths=list(data.get("tensor_paths", [])),
            qa_paths=list(data.get("qa_paths", [])),
        ).finalize()


@dataclass
class SensorArchetype(DocumentMixin):
    KIND: ClassVar[str] = "sensor_archetype"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "sensor_name",
        "band_name",
        "waveform_family",
        "center_frequency_hz",
        "sample_rate_hz",
    )

    id: str = ""
    kind: str = "sensor_archetype"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    sensor_name: str = ""
    band_name: str = ""
    waveform_family: str = ""
    center_frequency_hz: float = 0.0
    sample_rate_hz: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensorArchetype":
        common = cls._from_common(data)
        return cls(
            **common,
            sensor_name=data.get("sensor_name", ""),
            band_name=data.get("band_name", ""),
            waveform_family=data.get("waveform_family", ""),
            center_frequency_hz=float(data.get("center_frequency_hz", 0.0)),
            sample_rate_hz=float(data.get("sample_rate_hz", 0.0)),
        ).finalize()


@dataclass
class Scenario(DocumentMixin):
    KIND: ClassVar[str] = "scenario"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "scenario_name",
        "sensor_archetype_id",
        "object_card_ids",
        "environment_label",
        "seed",
    )

    id: str = ""
    kind: str = "scenario"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    scenario_name: str = ""
    sensor_archetype_id: str = ""
    object_card_ids: list[str] = None  # type: ignore[assignment]
    environment_label: str = ""
    seed: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        common = cls._from_common(data)
        return cls(
            **common,
            scenario_name=data.get("scenario_name", ""),
            sensor_archetype_id=data.get("sensor_archetype_id", ""),
            object_card_ids=list(data.get("object_card_ids", [])),
            environment_label=data.get("environment_label", ""),
            seed=int(data.get("seed", 0)),
        ).finalize()


@dataclass
class RadarEpisode(DocumentMixin):
    KIND: ClassVar[str] = "radar_episode"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "episode_name",
        "scenario_id",
        "sample_rate_hz",
        "product_paths",
    )

    id: str = ""
    kind: str = "radar_episode"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    episode_name: str = ""
    scenario_id: str = ""
    sample_rate_hz: float = 0.0
    product_paths: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RadarEpisode":
        common = cls._from_common(data)
        return cls(
            **common,
            episode_name=data.get("episode_name", ""),
            scenario_id=data.get("scenario_id", ""),
            sample_rate_hz=float(data.get("sample_rate_hz", 0.0)),
            product_paths=list(data.get("product_paths", [])),
        ).finalize()


@dataclass
class DetectorGraph(DocumentMixin):
    KIND: ClassVar[str] = "detector_graph"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ("graph_name", "nodes", "edges")

    id: str = ""
    kind: str = "detector_graph"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    graph_name: str = ""
    nodes: list[str] = None  # type: ignore[assignment]
    edges: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectorGraph":
        common = cls._from_common(data)
        return cls(
            **common,
            graph_name=data.get("graph_name", ""),
            nodes=list(data.get("nodes", [])),
            edges=list(data.get("edges", [])),
        ).finalize()


@dataclass
class SplitCounts:
    train: int
    validation: int
    test: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitCounts":
        return cls(int(data["train"]), int(data["validation"]), int(data["test"]))


@dataclass
class DatasetCard(DocumentMixin):
    KIND: ClassVar[str] = "dataset_card"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset_name",
        "source_campaign_ids",
        "splits",
    )

    id: str = ""
    kind: str = "dataset_card"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    dataset_name: str = ""
    source_campaign_ids: list[str] = None  # type: ignore[assignment]
    splits: SplitCounts = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetCard":
        common = cls._from_common(data)
        return cls(
            **common,
            dataset_name=data.get("dataset_name", ""),
            source_campaign_ids=list(data.get("source_campaign_ids", [])),
            splits=SplitCounts.from_dict(data.get("splits", {"train": 0, "validation": 0, "test": 0})),
        ).finalize()


@dataclass
class ValidationReport(DocumentMixin):
    KIND: ClassVar[str] = "validation_report"
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "report_name",
        "subject_kind",
        "subject_id",
        "checks",
        "overall_status",
    )

    id: str = ""
    kind: str = "validation_report"
    schema_version: str = "1.0.0"
    public_proxy_id: str = ""
    provenance: Provenance = None  # type: ignore[assignment]
    license: LicenseInfo = None  # type: ignore[assignment]
    validation: ValidationInfo = None  # type: ignore[assignment]
    report_name: str = ""
    subject_kind: str = ""
    subject_id: str = ""
    checks: list[ValidationCheck] = None  # type: ignore[assignment]
    overall_status: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        common = cls._from_common(data)
        return cls(
            **common,
            report_name=data.get("report_name", ""),
            subject_kind=data.get("subject_kind", ""),
            subject_id=data.get("subject_id", ""),
            checks=[ValidationCheck.from_dict(item) for item in data.get("checks", [])],
            overall_status=data.get("overall_status", ""),
        ).finalize()
