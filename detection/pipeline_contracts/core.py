"""Core dataclasses and I/O helpers for ML pipeline execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import json
import os

MAX_PIPELINE_WORKERS = 20
MAX_SUITE_CONCURRENCY = 3
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


class PipelineError(RuntimeError):
    """Structured pipeline failure."""

    def __init__(self, message: str, *, code: str = "pipeline_error", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class MissingInputKindError(PipelineError):
    """Raised when a requested input modality is absent."""

    def __init__(self, input_kind: str, message: str | None = None):
        super().__init__(
            message or f"missing required input kind: {input_kind}",
            code="missing_input_kind",
            details={"missing_input_kind": input_kind},
        )
        self.input_kind = input_kind


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    kind: str
    path: str
    ready: bool = True
    incomplete: bool = False


@dataclass(frozen=True)
class PipelineSpec:
    id: str
    version: str
    title: str
    summary: str
    validation_tier: str
    worker_budget: int
    input_contract: list[str]
    output_contract: list[str]
    feature_banks: list[str]
    detector_heads: list[str]
    deciders: list[str]
    references: list[str]
    research_only: bool = False

    def to_json(self) -> dict[str, Any]:
        return as_jsonable(self)


@dataclass(frozen=True)
class PipelineContext:
    repo_root: Path
    data_root: Path
    out_root: Path
    pipeline_id: str
    run_id: str
    workers: int
    seed: int
    smoke: bool = False
    validation_tier: str = "evidence_ladder_v1"

    def output_dir(self) -> Path:
        return self.out_root / self.run_id / self.pipeline_id


@dataclass
class PipelineResult:
    pipeline_id: str
    run_id: str
    status: str
    message: str
    output_dir: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    missing_input_kind: str | None = None
    worker_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return as_jsonable(self)


def as_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: as_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, set):
        return [as_jsonable(item) for item in sorted(value, key=repr)]
    return value


def json_dump(value: Any, *, indent: int = 2) -> str:
    return json.dumps(as_jsonable(value), indent=indent, sort_keys=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dump(value) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str]
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        else:
            handle.write("")


def ensure_within_root(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PipelineError(
            f"output path {resolved} escapes root {root}",
            code="unsafe_output_path",
        ) from exc
    return resolved


def default_thread_env(workers: int) -> dict[str, str]:
    capped = max(1, min(workers, MAX_PIPELINE_WORKERS))
    env = {key: str(capped) for key in THREAD_ENV_KEYS}
    env["PYTHONHASHSEED"] = "0"
    return env


def clamp_workers(workers: int) -> int:
    if workers < 1:
        raise PipelineError("workers must be at least 1", code="invalid_workers")
    if workers > MAX_PIPELINE_WORKERS:
        raise PipelineError(
            f"workers must be <= {MAX_PIPELINE_WORKERS}",
            code="invalid_workers",
            details={"max_workers": MAX_PIPELINE_WORKERS},
        )
    return workers


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "pipeline"


def make_run_id(seed: int, pipeline_id: str, smoke: bool) -> str:
    suffix = "smoke" if smoke else "run"
    return f"{suffix}-{slugify(pipeline_id)}-{seed:016x}"
