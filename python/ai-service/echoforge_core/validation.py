"""
Validation utilities and error handling for echoforge_core package.
Contains validation functions and error classes.
"""
from __future__ import annotations

import re
from typing import Any


class ValidationError(Exception):
    """Base exception for validation failures."""
    pass


def ensure_non_empty(value: Any, field_name: str) -> None:
    """Ensure a value is not empty or None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{field_name} must not be empty")


def ensure_non_empty_list(value: list[Any], field_name: str) -> None:
    """Ensure a list is not empty."""
    if not value:
        raise ValidationError(f"{field_name} must not be empty")


def ensure_probability(value: float, field_name: str) -> None:
    """Ensure a value is a valid probability between 0 and 1."""
    if not (0.0 <= value <= 1.0):
        raise ValidationError(f"{field_name} must be between 0 and 1")


def ensure_slug(value: str, field_name: str) -> None:
    """Ensure a value is a valid slug (alphanumeric with hyphens and underscores)."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise ValidationError(f"{field_name} must be a valid slug")


def fingerprint_sha256(value: str) -> None:
    """Validate that a string is a valid SHA256 fingerprint."""
    if len(value) not in (0, 64):
        raise ValidationError("fingerprint_sha256 must be 64 hex chars when present")
    if value and not re.match(r'^[a-fA-F0-9]{64}$', value):
        raise ValidationError("fingerprint_sha256 must contain only hex characters")


def canonical_json(data: Any) -> str:
    """Convert data to canonical JSON string."""
    import json
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def deterministic_id(data: Any) -> str:
    """Generate a deterministic ID from data using SHA256."""
    import hashlib
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


__all__ = [
    "ValidationError",
    "ensure_non_empty",
    "ensure_non_empty_list",
    "ensure_probability",
    "ensure_slug",
    "fingerprint_sha256",
    "canonical_json",
    "deterministic_id",
]