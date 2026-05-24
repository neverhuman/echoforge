#!/usr/bin/env python3
"""Validate focused final paper figures."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageStat
except Exception as exc:  # pragma: no cover
    raise SystemExit("visual validation failed: Pillow is required; install pillow") from exc


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
REQUIRED_PNGS = (
    "architecture_stack.png",
    "phase_method_ladder.png",
    "ei_evolution_money_plot.png",
    "false_alarm_breakdown.png",
    "radar_positive_vs_false_positive.png",
    "appendix_radar_samples.png",
)
VECTOR_PDFS = (
    "architecture_stack.pdf",
    "phase_method_ladder.pdf",
    "ei_evolution_money_plot.pdf",
    "false_alarm_breakdown.pdf",
    "radar_positive_vs_false_positive.pdf",
)
LEGACY_MAIN_FIGURES = (
    "monte_carlo_split_flow.pdf",
    "kpi_ranking.pdf",
    "phase_kpi.pdf",
    "detector_ml_pipeline.pdf",
    "anchor_overlay.pdf",
    "ei_workflow.pdf",
    "data_processing_flow.pdf",
    "appendix_modeling_map.pdf",
    "iq_drone_samples.pdf",
    "iq_negative_samples.png",
)
BANNED_VISIBLE_TERMS = (
    re.compile(r"\blocked candidate\b", re.IGNORECASE),
    re.compile(r"\bselected candidate\b", re.IGNORECASE),
    re.compile(r"\bselected AP\b", re.IGNORECASE),
    re.compile(r"NeverHumqn", re.IGNORECASE),
    re.compile(r"(?<!\d)0\.0\s*GHz", re.IGNORECASE),
    re.compile(r"burn-through calculation", re.IGNORECASE),
    re.compile(r"jammer power recipe", re.IGNORECASE),
    re.compile(r"target-specific delay program", re.IGNORECASE),
)
MIN_PNG_BYTES = 25_000
MIN_PDF_BYTES = 5_000
MIN_PNG_WIDTH = 2500
MAX_PNG_WIDTH = 5200
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
        fail(f"{name} contains banned visible terminology: " + ", ".join(flagged))
    missing = [term for term in required_terms if term not in text]
    if missing:
        fail(f"{name} is missing required visible terms: " + ", ".join(missing))
    present_forbidden = [term for term in forbidden_terms if term in text]
    if present_forbidden:
        fail(f"{name} contains forbidden visible terms: " + ", ".join(present_forbidden))


def validate_tex_figure_references() -> None:
    tex_path = Path(__file__).resolve().parent / "echoforge_ieee.tex"
    text = tex_path.read_text(encoding="utf-8")
    refs = [name for name in LEGACY_MAIN_FIGURES if name in text]
    if refs:
        fail("paper still references legacy clutter figure(s): " + ", ".join(refs))


def main() -> int:
    missing = [name for name in REQUIRED_PNGS if not (FIGURES_DIR / name).is_file()]
    if missing:
        fail("missing PNG figure(s): " + ", ".join(missing))
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

    required_by_pdf = {
        "architecture_stack.pdf": (
            "EchoForge simulator",
            "Radar Branches",
            "64 x 96",
            "jamming/deception",
            "claim boundary",
        ),
        "phase_method_ladder.pdf": (
            "Phase method ladder",
            "Take-off",
            "Climb",
            "Cruise",
            "EI",
        ),
        "ei_evolution_money_plot.pdf": (
            "EI train/CV evolution",
            "selected lock",
            "no holdout optimization curve",
        ),
        "false_alarm_breakdown.pdf": (
            "False-positive burden",
            "RC fixed-wing",
            "weather",
            "RFI",
        ),
        "radar_positive_vs_false_positive.pdf": (
            "Range-Doppler diagnostic gallery",
            "Positive",
            "EW",
            "not measured imagery",
        ),
    }
    for name in VECTOR_PDFS:
        validate_pdf_terms(name, required_terms=required_by_pdf.get(name, ()))

    if (FIGURES_DIR / "appendix_radar_samples.pdf").exists():
        fail("appendix_radar_samples must remain a raster appendix PNG")
    radar_png = FIGURES_DIR / "radar_positive_vs_false_positive.png"
    with Image.open(radar_png) as image:
        image.load()
        width, height = image.size
        if width < 3000 or height < 1400:
            fail(
                "radar_positive_vs_false_positive.png must be at least "
                f"3000x1400 px, got {width}x{height}"
            )
    validate_tex_figure_references()

    print(f"visual validation passed: pngs={len(REQUIRED_PNGS)} vector_pdfs={len(VECTOR_PDFS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
