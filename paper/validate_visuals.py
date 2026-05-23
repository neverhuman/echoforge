#!/usr/bin/env python3
"""Validate paper figure raster previews and vector-backed chart outputs."""

from __future__ import annotations

import sys
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
REQUIRED_PNGS = (
    "architecture_stack.png",
    "monte_carlo_split_flow.png",
    "kpi_ranking.png",
    "phase_kpi.png",
    "iq_drone_samples.png",
    "iq_negative_samples.png",
    "detector_ml_pipeline.png",
    "locked_algorithm.png",
)
VECTOR_PDFS = (
    "architecture_stack.pdf",
    "monte_carlo_split_flow.pdf",
    "kpi_ranking.pdf",
    "phase_kpi.pdf",
    "iq_drone_samples.pdf",
    "detector_ml_pipeline.pdf",
    "locked_algorithm.pdf",
)
RASTER_ONLY = ("iq_negative_samples.png",)
MIN_PNG_BYTES = 25_000
MIN_PDF_BYTES = 5_000


def fail(message: str) -> None:
    print(f"visual validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    missing = [name for name in REQUIRED_PNGS if not (FIGURES_DIR / name).is_file()]
    if missing:
        fail("missing PNG figure(s): " + ", ".join(missing))
    tiny_pngs = [
        name for name in REQUIRED_PNGS if (FIGURES_DIR / name).stat().st_size < MIN_PNG_BYTES
    ]
    if tiny_pngs:
        fail("tiny PNG figure(s): " + ", ".join(tiny_pngs))

    missing_pdfs = [name for name in VECTOR_PDFS if not (FIGURES_DIR / name).is_file()]
    if missing_pdfs:
        fail("missing vector PDF figure(s): " + ", ".join(missing_pdfs))
    tiny_pdfs = [
        name for name in VECTOR_PDFS if (FIGURES_DIR / name).stat().st_size < MIN_PDF_BYTES
    ]
    if tiny_pdfs:
        fail("tiny vector PDF figure(s): " + ", ".join(tiny_pdfs))

    unexpected = [
        Path(name).with_suffix(".pdf").name
        for name in RASTER_ONLY
        if (FIGURES_DIR / Path(name).with_suffix(".pdf").name).exists()
    ]
    if unexpected:
        fail("raster-only figure has vector PDF: " + ", ".join(unexpected))

    print(
        "visual validation passed: "
        f"pngs={len(REQUIRED_PNGS)} vector_pdfs={len(VECTOR_PDFS)} raster_only={len(RASTER_ONLY)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
