"""Raw IQ availability checks."""

from __future__ import annotations

from pathlib import Path

from detection.pipeline_contracts.core import MissingInputKindError


RAW_IQ_HINTS = (
    "raw_complex_iq",
    "pre_fft_iq",
    "raw_iq",
    "iq",
)


def find_raw_iq_root(data_root: Path) -> Path | None:
    for hint in RAW_IQ_HINTS:
        candidate = data_root / hint
        if candidate.exists():
            return candidate
    return None


def require_raw_iq_root(data_root: Path) -> Path:
    root = find_raw_iq_root(data_root)
    if root is None:
        raise MissingInputKindError("raw_complex_iq")
    return root
