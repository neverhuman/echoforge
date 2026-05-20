"""Generate the Detector Realism current public OSINT family matrix.

This artifact is intentionally a public-proxy planning matrix. It summarizes
openly described detector families and does not claim measured sensor truth,
classified fidelity, or proprietary-equivalent behavior.
"""

from __future__ import annotations

from .detector_osint_family_matrix_types import (
    DetectorFamily,
    MatrixConfig,
    CLAIM_BOUNDARY,
    DEFAULT_OUT_DIR,
    DEFAULT_REPORT,
)
from .detector_osint_family_matrix_helpers import (
    build_matrix,
    write_matrix_json,
    write_matrix_markdown,
    parse_args,
)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = MatrixConfig(
        output_dir=args.output_dir if args.output_dir else DEFAULT_OUT_DIR,
        report_path=args.report if args.report else DEFAULT_REPORT,
        include_private_systems=args.include_private,
    )
    matrix = build_matrix()
    write_matrix_json(matrix, config)
    write_matrix_markdown(matrix, config)


if __name__ == "__main__":
    main()


__all__ = [
    "DetectorFamily",
    "MatrixConfig",
    "CLAIM_BOUNDARY",
    "DEFAULT_OUT_DIR",
    "DEFAULT_REPORT",
    "build_matrix",
    "write_matrix_json",
    "write_matrix_markdown",
    "parse_args",
    "main",
]