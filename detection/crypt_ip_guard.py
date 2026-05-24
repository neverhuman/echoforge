"""Developer-key guard for git-crypt protected EI runtime code."""

from __future__ import annotations

import sys
from typing import Callable, TextIO, TypeVar


DEVELOPER_KEY_REQUIRED_MESSAGE = (
    "Engineered Intelligence runtime code is protected by git-crypt; "
    "run `git-crypt unlock` with the developer key, then retry this command."
)
DEVELOPER_KEY_REQUIRED_EXIT_CODE = 78


class DeveloperKeyRequiredError(RuntimeError):
    """Raised when git-crypt protected EI implementation code is unavailable."""


def developer_key_required_message(module_name: str | None = None) -> str:
    if module_name:
        return f"{DEVELOPER_KEY_REQUIRED_MESSAGE} Missing protected module: {module_name}."
    return DEVELOPER_KEY_REQUIRED_MESSAGE


def _is_missing_private_module(module_name: str, exc: ModuleNotFoundError) -> bool:
    missing = exc.name or ""
    return (
        missing == module_name
        or module_name.startswith(f"{missing}.")
        or missing.startswith(f"{module_name}.")
    )


def _is_locked_source_error(exc: BaseException) -> bool:
    if isinstance(exc, UnicodeDecodeError):
        return True
    if isinstance(exc, SyntaxError):
        text = str(exc).lower()
        return (
            "null bytes" in text or "non-utf-8" in text or "invalid non-printable character" in text
        )
    return False


def raise_for_private_import_error(module_name: str, exc: BaseException) -> None:
    """Translate protected EI import failures into a user-facing unlock error."""

    if isinstance(exc, DeveloperKeyRequiredError):
        raise exc
    if isinstance(exc, ModuleNotFoundError):
        if _is_missing_private_module(module_name, exc):
            raise DeveloperKeyRequiredError(developer_key_required_message(module_name)) from exc
        raise
    if isinstance(exc, (SyntaxError, UnicodeDecodeError)):
        if _is_locked_source_error(exc):
            raise DeveloperKeyRequiredError(developer_key_required_message(module_name)) from exc
        raise
    raise exc


def print_developer_key_warning(
    exc: DeveloperKeyRequiredError, *, stream: TextIO | None = None
) -> None:
    target = stream if stream is not None else sys.stderr
    print(f"warning: {exc}", file=target)


T = TypeVar("T")


def run_with_developer_key_warning(
    main_fn: Callable[[], T], *, stream: TextIO | None = None
) -> int:
    try:
        result = main_fn()
    except DeveloperKeyRequiredError as exc:
        print_developer_key_warning(exc, stream=stream)
        return DEVELOPER_KEY_REQUIRED_EXIT_CODE
    if result is None:
        return 0
    return int(result)
