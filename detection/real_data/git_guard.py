"""Guardrails that keep local measured data and generated outputs out of Git."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable


FORBIDDEN_TRACKED_ROOTS = (
    "real-data/",
    ".cache/echoforge/real-data/",
    "outputs/real-data/",
)
FORBIDDEN_TRACKED_SUFFIXES = (
    ".bag",
    ".bin",
    ".h5",
    ".hdf5",
    ".mat",
    ".pcap",
    ".rar",
    ".rosbag",
    ".tar",
    ".tar.gz",
    ".ulg",
    ".zip",
    ".7z",
)
SKIP_DIRS = {".git", ".jankurai", "node_modules", "target", "__pycache__"}


def _git_dir(repo_root: Path) -> Path | None:
    dotgit = repo_root / ".git"
    if dotgit.is_dir():
        return dotgit
    if dotgit.is_file():
        text = dotgit.read_text(encoding="utf-8", errors="ignore").strip()
        prefix = "gitdir:"
        if text.startswith(prefix):
            return (repo_root / text[len(prefix) :].strip()).resolve()
    return None


def _git_index_paths(repo_root: Path) -> list[str]:
    git_dir = _git_dir(repo_root.resolve())
    if git_dir is None:
        return []
    index_path = git_dir / "index"
    if not index_path.exists():
        return []
    data = index_path.read_bytes()
    if len(data) < 12 or data[:4] != b"DIRC":
        return []
    _version, count = struct.unpack(">II", data[4:12])
    offset = 12
    paths: list[str] = []
    for _ in range(count):
        entry_start = offset
        if offset + 62 > len(data):
            return paths
        flags = struct.unpack(">H", data[offset + 60 : offset + 62])[0]
        offset += 62
        path_len = flags & 0x0FFF
        if path_len == 0x0FFF:
            path_end = data.find(b"\0", offset)
            if path_end < 0:
                return paths
        else:
            path_end = offset + path_len
        raw_path = data[offset:path_end]
        paths.append(raw_path.decode("utf-8", errors="replace"))
        offset = path_end + 1
        padding = (8 - ((offset - entry_start) % 8)) % 8
        offset += padding
    return paths


def _repo_policy_paths(repo_root: Path) -> list[str]:
    paths: list[str] = []
    root = repo_root.resolve()
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
    return paths


def find_tracked_real_data(
    repo_root: Path | str = Path("."),
    tracked_paths: Iterable[str] | None = None,
) -> list[str]:
    if tracked_paths is not None:
        paths = list(tracked_paths)
    else:
        paths = _git_index_paths(Path(repo_root)) or _repo_policy_paths(Path(repo_root))
    violations = []
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        if any(path.startswith(root) for root in FORBIDDEN_TRACKED_ROOTS):
            violations.append(raw_path)
            continue
        if any(path.endswith(suffix) for suffix in FORBIDDEN_TRACKED_SUFFIXES):
            violations.append(raw_path)
    return sorted(set(violations))
