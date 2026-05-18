from __future__ import annotations

import hashlib
import json
import re
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
ID_VERSION = 1
SLUG_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
ID_RE = re.compile(r"^ef:[a-z0-9]+(?:[._-][a-z0-9]+)*:[a-z0-9]+(?:[._-][a-z0-9]+)*:[0-9a-f]{16}:1$")


class ValidationError(ValueError):
    pass


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_id(kind: str, public_proxy_id: str, payload: Any) -> str:
    digest = hashlib.sha256(
        f"{kind}|{public_proxy_id}|{ID_VERSION}|{fingerprint_sha256(payload)}".encode("utf-8")
    ).hexdigest()
    return f"ef:{kind}:{public_proxy_id}:{digest[:16]}:{ID_VERSION}"


def ensure_slug(value: str, field: str) -> None:
    if not SLUG_RE.match(value):
        raise ValidationError(f"{field} must be a lowercase slug")


def ensure_non_empty(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValidationError(f"{field} must be non-empty")


def ensure_non_empty_list(value: list[Any], field: str) -> None:
    if not value:
        raise ValidationError(f"{field} must not be empty")


def ensure_probability(value: float, field: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValidationError(f"{field} must be between 0 and 1")
