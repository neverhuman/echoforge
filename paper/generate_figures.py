#!/usr/bin/env python3
"""Compatibility entrypoint for the current EchoForge paper figures."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper.generate_figures_focused import main as focused_main

    return focused_main()


if __name__ == "__main__":
    raise SystemExit(main())
