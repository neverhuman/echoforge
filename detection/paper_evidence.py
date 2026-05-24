#!/usr/bin/env python3
"""Canonical paper evidence entrypoint for the final EchoForge manuscript."""

from __future__ import annotations

from detection.paper_evidence_builder import (  # noqa: F401
    DEFAULT_ADVANCED_ROOT,
    DEFAULT_ANCHOR_ROOT,
    DEFAULT_BASELINE_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_TRAINING_ROOT,
    EvidenceRoots,
    build_paper_evidence,
    main,
    parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())
