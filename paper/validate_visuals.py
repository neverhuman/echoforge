#!/usr/bin/env python3
"""Validate paper figure raster previews and vector-backed chart outputs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageStat
except Exception as exc:  # pragma: no cover - import guard for minimal environments
    raise SystemExit("visual validation failed: Pillow is required; install pillow") from exc


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
REQUIRED_PNGS = (
    "architecture_stack.png",
    "monte_carlo_split_flow.png",
    "kpi_ranking.png",
    "phase_kpi.png",
    "iq_drone_samples.png",
    "iq_negative_samples.png",
    "detector_ml_pipeline.png",
    "anchor_overlay.png",
    "ei_workflow.png",
    "phase_method_ladder.png",
    "ei_evolution_money_plot.png",
    "data_processing_flow.png",
    "appendix_modeling_map.png",
)
VECTOR_PDFS = (
    "architecture_stack.pdf",
    "monte_carlo_split_flow.pdf",
    "kpi_ranking.pdf",
    "phase_kpi.pdf",
    "iq_drone_samples.pdf",
    "detector_ml_pipeline.pdf",
    "anchor_overlay.pdf",
    "ei_workflow.pdf",
    "phase_method_ladder.pdf",
    "ei_evolution_money_plot.pdf",
    "data_processing_flow.pdf",
    "appendix_modeling_map.pdf",
)
RASTER_ONLY = ("iq_negative_samples.png",)
LEGACY_FIGURES = ("locked_algorithm.png", "locked_algorithm.pdf")
BANNED_VISIBLE_TERMS = (
    re.compile(r"\blocked candidate\b", re.IGNORECASE),
    re.compile(r"\bselected candidate\b", re.IGNORECASE),
    re.compile(r"\bselected AP\b", re.IGNORECASE),
    re.compile(r"NeverHumqn", re.IGNORECASE),
)
MIN_PNG_BYTES = 25_000
MIN_PDF_BYTES = 5_000
MIN_PNG_WIDTH = 2600
MAX_PNG_WIDTH = 3900
MIN_PNG_HEIGHT = 900
MIN_CHANNEL_STDDEV = 3.0


def fail(message: str) -> None:
    print(f"visual validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_png_readability(name: str) -> None:
    path = FIGURES_DIR / name
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if not MIN_PNG_WIDTH <= width <= MAX_PNG_WIDTH:
            fail(f"{name} has unexpected width {width}; expected {MIN_PNG_WIDTH}-{MAX_PNG_WIDTH}")
        if height < MIN_PNG_HEIGHT:
            fail(f"{name} has tiny height {height}; expected at least {MIN_PNG_HEIGHT}")
        stat = ImageStat.Stat(image.convert("RGB"))
        if max(stat.stddev) < MIN_CHANNEL_STDDEV:
            fail(f"{name} appears blank or near-monochrome")


def validate_pdf_terms(
    name: str,
    *,
    required_terms: tuple[str, ...] = (),
    forbidden_terms: tuple[str, ...] = (),
) -> None:
    path = FIGURES_DIR / name
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        fail("missing pdftotext; install poppler-utils")
    except subprocess.CalledProcessError as exc:
        fail(f"pdftotext failed for {name}: {exc.stderr.strip() or exc.stdout.strip()}")
    text = result.stdout
    flagged = [pattern.pattern for pattern in BANNED_VISIBLE_TERMS if pattern.search(text)]
    if flagged:
        fail(f"{name} contains banned visible EI terminology: " + ", ".join(flagged))
    missing = [term for term in required_terms if term not in text]
    if missing:
        fail(f"{name} is missing required visible terms: " + ", ".join(missing))
    present_forbidden = [term for term in forbidden_terms if term in text]
    if present_forbidden:
        fail(f"{name} contains forbidden visible terms: " + ", ".join(present_forbidden))


def main() -> int:
    missing = [name for name in REQUIRED_PNGS if not (FIGURES_DIR / name).is_file()]
    if missing:
        fail("missing PNG figure(s): " + ", ".join(missing))
    legacy = [name for name in LEGACY_FIGURES if (FIGURES_DIR / name).exists()]
    if legacy:
        fail("legacy locked_algorithm figure(s) must be renamed: " + ", ".join(legacy))
    tiny_pngs = [
        name for name in REQUIRED_PNGS if (FIGURES_DIR / name).stat().st_size < MIN_PNG_BYTES
    ]
    if tiny_pngs:
        fail("tiny PNG figure(s): " + ", ".join(tiny_pngs))
    for name in REQUIRED_PNGS:
        validate_png_readability(name)

    missing_pdfs = [name for name in VECTOR_PDFS if not (FIGURES_DIR / name).is_file()]
    if missing_pdfs:
        fail("missing vector PDF figure(s): " + ", ".join(missing_pdfs))
    tiny_pdfs = [
        name for name in VECTOR_PDFS if (FIGURES_DIR / name).stat().st_size < MIN_PDF_BYTES
    ]
    if tiny_pdfs:
        fail("tiny vector PDF figure(s): " + ", ".join(tiny_pdfs))
    for name in VECTOR_PDFS:
        required_terms = ()
        forbidden_terms = ()
        if name == "kpi_ranking.pdf":
            required_terms = (
                "Primary KPI gain",
                "gain vs best practice",
                "red tick = LCB95 lower bound",
                "+742%",
                "+185%",
                "LCB95",
                "1% FPR operating cap",
            )
            forbidden_terms = ("+743%", "+186%")
        elif name == "phase_kpi.pdf":
            required_terms = (
                "False-alarm family legend",
                "Near-threshold total",
                "Fixed-FPR recall",
            )
        elif name == "ei_workflow.pdf":
            required_terms = ("What EI does", "Blind holdout", "Feature denylist")
        elif name == "phase_method_ladder.pdf":
            required_terms = (
                "Phase method ladder",
                "detectors",
                "accepted fusion",
                "EI",
            )
        elif name == "ei_evolution_money_plot.pdf":
            required_terms = (
                "EI evolution money plot",
                "train/CV",
                "Locked holdout endpoint",
                "no holdout optimization curve",
            )
        elif name == "data_processing_flow.pdf":
            required_terms = (
                "Data-processing path",
                "Synthetic sensing",
                "Detector-view schema",
                "Train/CV branch",
                "Blind holdout",
            )
        elif name == "appendix_modeling_map.pdf":
            required_terms = ("positive proxy", "environment", "detector views")
        validate_pdf_terms(name, required_terms=required_terms, forbidden_terms=forbidden_terms)

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
