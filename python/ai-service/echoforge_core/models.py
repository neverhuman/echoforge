"""
Core models for echoforge_core package.
Contains dataclasses and validation utilities.
"""
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

    def validate(self) -> None:
        ensure_non_empty(self.tier, "tier")
        ensure_non_empty(self.status, "status")
        ensure_probability(self.uncertainty_score, "uncertainty_score")
        if self.checks is None:
            self.checks = []


__all__ = [
    "Provenance",
    "LicenseInfo",
    "ValidationCheck",
    "ValidationInfo",
    "_coerce",
    "_split_document_data",
]