"""Neutral drift-bound Python-runtime closure API for contained children.

The implementation currently lives with the exploratory smoke bundle because
that subsystem owns the persisted schema.  This module is the stable,
product-neutral entry point for other contained workflows; callers should not
depend on training-bundle storage or plan helpers directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from spritelab.training.smoke_bundle import (
    RUNTIME_BOUNDED_RESIDUALS,
    RUNTIME_EXECUTION_BYTE_POLICY,
    SmokeBundleError,
    bound_runtime_import_policy,
    prepare_smoke_runtime_closure,
    smoke_runtime_environment_paths,
    verify_prepared_runtime_closure,
)

RuntimeClosureError = SmokeBundleError


def prepare_exact_python_runtime_closure(
    project_root: str | Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Inventory trusted installed-runtime bytes with explicit residuals."""

    return prepare_smoke_runtime_closure(project_root, operation_check=operation_check)


def verify_exact_python_runtime_closure(
    project_root: str | Path,
    closure: Any,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Rehash every inventoried file/root without claiming load immutability."""

    return verify_prepared_runtime_closure(project_root, closure, operation_check=operation_check)


@contextmanager
def exact_python_runtime_import_policy(
    project_root: str | Path,
    closure: Any,
    *,
    operation_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Load source exactly and pre/post-check native/resource drift."""

    with bound_runtime_import_policy(
        project_root,
        closure,
        operation_check=operation_check,
    ) as verified:
        yield verified


def exact_python_runtime_environment_paths(
    project_root: str | Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return parent-only paths for a `-S` contained interpreter environment."""

    return smoke_runtime_environment_paths(
        project_root,
        operation_check=operation_check,
    )


__all__ = [
    "RUNTIME_BOUNDED_RESIDUALS",
    "RUNTIME_EXECUTION_BYTE_POLICY",
    "RuntimeClosureError",
    "exact_python_runtime_environment_paths",
    "exact_python_runtime_import_policy",
    "prepare_exact_python_runtime_closure",
    "verify_exact_python_runtime_closure",
]
