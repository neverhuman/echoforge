"""CLI for real-data registry and reference-only measured-anchor reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import build_dataset_report, default_raw_root, dry_run_fetch
from .catalog import load_catalog, require_valid_catalog
from .feature_policy import observable_only_violations
from .git_guard import find_tracked_real_data
from .realism_gate import evaluate_realism_gate


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_valid_catalog(path: Path) -> dict[str, Any]:
    catalog = load_catalog(path)
    require_valid_catalog(catalog)
    return catalog


def _cmd_validate_catalog(args: argparse.Namespace) -> int:
    catalog = _load_valid_catalog(args.catalog)
    _print({"status": "pass", "dataset_count": len(catalog["datasets"])})
    return 0


def _select_entries(catalog: dict[str, Any], dataset_id: str | None) -> list[dict[str, Any]]:
    entries = catalog["datasets"]
    if dataset_id is None:
        return entries
    selected = [entry for entry in entries if entry["dataset_id"] == dataset_id]
    if not selected:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    return selected


def _cmd_dry_run(args: argparse.Namespace) -> int:
    catalog = _load_valid_catalog(args.catalog)
    root = args.raw_root or default_raw_root()
    _print([dry_run_fetch(entry, root) for entry in _select_entries(catalog, args.dataset_id)])
    return 0


def _cmd_build_report(args: argparse.Namespace) -> int:
    catalog = _load_valid_catalog(args.catalog)
    root = args.raw_root or default_raw_root()
    reports = [
        build_dataset_report(
            entry, raw_root=root, out_root=args.out_root, run_id=args.run_id
        ).to_json()
        for entry in _select_entries(catalog, args.dataset_id)
    ]
    _print({"status": "completed", "reports": reports})
    return 0


def _cmd_guard_git(args: argparse.Namespace) -> int:
    violations = find_tracked_real_data(args.repo_root)
    _print({"status": "pass" if not violations else "fail", "violations": violations})
    return 0 if not violations else 2


def _cmd_feature_policy(args: argparse.Namespace) -> int:
    violations = observable_only_violations(args.feature)
    _print({"status": "pass" if not violations else "fail", "violations": violations})
    return 0 if not violations else 2


def _cmd_realism_gate(args: argparse.Namespace) -> int:
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    payload = evaluate_realism_gate(metrics)
    _print(payload)
    return 0 if payload["status"] == "go" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m detection.real_data.cli")
    parser.add_argument("--catalog", type=Path, default=Path("detection/real_data/catalog.json"))
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-catalog")
    validate.set_defaults(func=_cmd_validate_catalog)

    dry = sub.add_parser("dry-run")
    dry.add_argument("--dataset-id")
    dry.add_argument("--raw-root", type=Path)
    dry.set_defaults(func=_cmd_dry_run)

    report = sub.add_parser("build-report")
    report.add_argument("--dataset-id")
    report.add_argument("--raw-root", type=Path)
    report.add_argument("--out-root", type=Path, default=Path("outputs/real-data"))
    report.add_argument("--run-id", default="reference-only")
    report.set_defaults(func=_cmd_build_report)

    guard = sub.add_parser("guard-git")
    guard.add_argument("--repo-root", type=Path, default=Path("."))
    guard.set_defaults(func=_cmd_guard_git)

    features = sub.add_parser("feature-policy")
    features.add_argument("feature", nargs="+")
    features.set_defaults(func=_cmd_feature_policy)

    gate = sub.add_parser("realism-gate")
    gate.add_argument("--metrics", type=Path, required=True)
    gate.set_defaults(func=_cmd_realism_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as exc:
        _print({"status": "fail", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
