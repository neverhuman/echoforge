"""Public wrapper for the git-crypt protected EI trace evidence builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from detection.crypt_ip_guard import (
        raise_for_private_import_error,
        run_with_developer_key_warning,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from crypt_ip_guard import raise_for_private_import_error, run_with_developer_key_warning


_IMPL_MODULE = "detection.crypt_ip_impl.ei_evolution_trace"
_DIRECT_IMPL_MODULE = "crypt_ip_impl.ei_evolution_trace"

DEFAULT_ADVANCED_ROOT = Path(
    "outputs/detection/fixed-wing-pusher-proxy-main-run-advanced-evolution"
)
DEFAULT_OUT_ROOT = Path("outputs/paper-evidence/current")
SCHEMA_VERSION = "ei-evolution-evidence"


def _impl_module_name() -> str:
    return _DIRECT_IMPL_MODULE if __package__ in {"", None} else _IMPL_MODULE


def _load_impl() -> Any:
    module_name = _impl_module_name()
    try:
        if module_name == _DIRECT_IMPL_MODULE:  # pragma: no cover - direct script path
            from crypt_ip_impl import ei_evolution_trace as impl
        else:
            from detection.crypt_ip_impl import ei_evolution_trace as impl
    except (ModuleNotFoundError, SyntaxError, UnicodeDecodeError) as exc:
        raise_for_private_import_error(module_name, exc)
    return impl


def build_ei_evolution_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_impl().build_ei_evolution_evidence(*args, **kwargs)


def main() -> int:
    return int(_load_impl().main())


def __getattr__(name: str) -> Any:
    return getattr(_load_impl(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_load_impl())))


if __name__ == "__main__":
    raise SystemExit(run_with_developer_key_warning(main))
