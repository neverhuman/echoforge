"""Public wrapper for the git-crypt protected advanced EI detector lane."""

from __future__ import annotations

from typing import Any

try:
    from detection.crypt_ip_guard import raise_for_private_import_error
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from crypt_ip_guard import raise_for_private_import_error


_IMPL_MODULE = "detection.crypt_ip_impl.advanced_main_run_detectors"
_DIRECT_IMPL_MODULE = "crypt_ip_impl.advanced_main_run_detectors"

ADVANCED_METHOD_ID = "spectral_transport_hypergraph_fusion"
ADVANCED_OUTPUT_PROFILE = "fixed-wing-pusher-proxy-main-run-advanced-evolution"
EVOLUTION_TRACE_SCHEMA_VERSION = "ei-evolution-trace"
EVOLUTION_TRACE_SELECTION_SPLIT = "train_cv"
ADVANCED_FEATURE_CACHE_VERSION = "advanced-ew-rd-20260524"


def _impl_module_name() -> str:
    return _DIRECT_IMPL_MODULE if __package__ in {"", None} else _IMPL_MODULE


def _load_impl() -> Any:
    module_name = _impl_module_name()
    try:
        if module_name == _DIRECT_IMPL_MODULE:  # pragma: no cover - direct script path
            from crypt_ip_impl import advanced_main_run_detectors as impl
        else:
            from detection.crypt_ip_impl import advanced_main_run_detectors as impl
    except (ModuleNotFoundError, SyntaxError, UnicodeDecodeError) as exc:
        raise_for_private_import_error(module_name, exc)
    return impl


def run_advanced_main_run_detectors(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_impl().run_advanced_main_run_detectors(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_load_impl(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_load_impl())))
