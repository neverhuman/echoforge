"""Command line entrypoint for listing and running ML pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .core import (
    MAX_PIPELINE_WORKERS,
    MAX_SUITE_CONCURRENCY,
    PipelineError,
    as_jsonable,
    make_run_id,
)
from .validation import validate_pipeline_spec
from detection.pipelines import physics_cfar_track, raw_iq_ssl_research, tensor_microdoppler_fusion

SUITES = {
    "evidence-ladder-v1": [
        "physics_cfar_track_fusion_v1",
        "tensor_microdoppler_fusion_v1",
        "raw_iq_ssl_research_v1",
    ]
}

PIPELINE_MODULES = (
    physics_cfar_track,
    tensor_microdoppler_fusion,
    raw_iq_ssl_research,
)


def load_pipeline(module):
    spec = getattr(module, "PIPELINE_SPEC")
    validate_pipeline_spec(spec)
    return module, spec


def module_for_pipeline_id(pipeline_id: str):
    for module in PIPELINE_MODULES:
        module, spec = load_pipeline(module)
        if spec.id == pipeline_id:
            return module, spec
    raise PipelineError(f"unknown pipeline id: {pipeline_id}", code="unknown_pipeline")


def list_specs() -> list[dict[str, Any]]:
    return [load_pipeline(module)[1].to_json() for module in PIPELINE_MODULES]


def inspect_spec(pipeline_id: str) -> dict[str, Any]:
    _, spec = module_for_pipeline_id(pipeline_id)
    return spec.to_json()


def execute_pipeline(
    module,
    spec,
    *,
    repo_root: Path,
    data_root: Path,
    out_root: Path,
    workers: int,
    seed: int,
    smoke: bool,
    validation_tier: str,
):
    run_id = make_run_id(seed, spec.id, smoke)
    from .core import PipelineContext

    context = PipelineContext(
        repo_root=repo_root,
        data_root=data_root,
        out_root=out_root,
        pipeline_id=spec.id,
        run_id=run_id,
        workers=workers,
        seed=seed,
        smoke=smoke,
        validation_tier=validation_tier,
    )
    return module.smoke(context) if smoke else module.run(context)


def run_pipeline(
    *,
    pipeline_id: str,
    repo_root: Path,
    data_root: Path,
    out_root: Path,
    workers: int,
    seed: int,
    smoke: bool,
    validation_tier: str,
) -> dict[str, Any]:
    run_id = make_run_id(seed, pipeline_id, smoke)
    try:
        module, spec = module_for_pipeline_id(pipeline_id)
        result = execute_pipeline(
            module,
            spec,
            repo_root=repo_root,
            data_root=data_root,
            out_root=out_root,
            workers=workers,
            seed=seed,
            smoke=smoke,
            validation_tier=validation_tier,
        )
        return as_jsonable(result)
    except PipelineError as exc:
        status = "blocked" if exc.code == "missing_input_kind" else "failed"
        payload = {
            "pipeline_id": pipeline_id,
            "run_id": run_id,
            "status": status,
            "message": str(exc),
            "output_dir": (out_root / run_id / pipeline_id).as_posix(),
            "artifacts": [],
            "metrics": {},
            "gates": [],
            "notes": [],
            "error_code": exc.code,
            "details": exc.details,
            "missing_input_kind": exc.details.get("missing_input_kind"),
            "worker_count": workers,
        }
        return payload


def run_suite(
    *,
    suite: str,
    repo_root: Path,
    data_root: Path,
    out_root: Path,
    workers_per_pipeline: int,
    max_concurrent: int,
    seed: int,
    smoke: bool,
    validation_tier: str,
) -> dict[str, Any]:
    pipeline_ids = SUITES.get(suite)
    if not pipeline_ids:
        raise PipelineError(f"unknown suite: {suite}", code="unknown_suite")
    max_concurrent = max(1, min(max_concurrent, MAX_SUITE_CONCURRENCY))
    workers_per_pipeline = max(1, min(workers_per_pipeline, MAX_PIPELINE_WORKERS))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for index, pipeline_id in enumerate(pipeline_ids):
            futures[
                executor.submit(
                    run_pipeline,
                    pipeline_id=pipeline_id,
                    repo_root=repo_root,
                    data_root=data_root,
                    out_root=out_root,
                    workers=workers_per_pipeline,
                    seed=seed + index,
                    smoke=smoke,
                    validation_tier=validation_tier,
                )
            ] = pipeline_id
        for future in as_completed(futures):
            pipeline_id = futures[future]
            try:
                results.append(future.result())
            except PipelineError as exc:
                results.append(
                    {
                        "pipeline_id": pipeline_id,
                        "status": "failed",
                        "message": str(exc),
                        "error_code": exc.code,
                        "details": exc.details,
                    }
                )
    return {
        "suite": suite,
        "max_concurrent": max_concurrent,
        "workers_per_pipeline": workers_per_pipeline,
        "results": sorted(results, key=lambda item: item.get("pipeline_id", "")),
    }


def _cmd_list(_: argparse.Namespace) -> int:
    print(json.dumps(list_specs(), indent=2, sort_keys=True))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_spec(args.pipeline), indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    payload = run_pipeline(
        pipeline_id=args.pipeline,
        repo_root=args.repo_root,
        data_root=args.data_root,
        out_root=args.out_root,
        workers=args.workers,
        seed=args.seed,
        smoke=args.smoke,
        validation_tier=args.validation_tier,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    payload = run_suite(
        suite=args.suite,
        repo_root=args.repo_root,
        data_root=args.data_root,
        out_root=args.out_root,
        workers_per_pipeline=args.workers_per_pipeline,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        smoke=args.smoke,
        validation_tier=args.validation_tier,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m detection.pipeline_contracts.runner")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=_cmd_list)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("pipeline")
    inspect_parser.set_defaults(func=_cmd_inspect)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--pipeline", required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--out-root", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--validation-tier", default="evidence_ladder_v1")
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.set_defaults(func=_cmd_run)

    suite_parser = sub.add_parser("run-suite")
    suite_parser.add_argument("--suite", required=True)
    suite_parser.add_argument("--repo-root", type=Path, required=True)
    suite_parser.add_argument("--data-root", type=Path, required=True)
    suite_parser.add_argument("--out-root", type=Path, required=True)
    suite_parser.add_argument("--workers-per-pipeline", type=int, required=True)
    suite_parser.add_argument("--max-concurrent", type=int, required=True)
    suite_parser.add_argument("--seed", type=int, required=True)
    suite_parser.add_argument("--validation-tier", default="evidence_ladder_v1")
    suite_parser.add_argument("--smoke", action="store_true")
    suite_parser.set_defaults(func=_cmd_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PipelineError as exc:
        payload = {"error": exc.code, "message": str(exc), "details": exc.details}
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
        payload = {"error": "unexpected_error", "message": str(exc)}
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
