"""Shared reporting helpers: jsonable, fmt_float, fmt_int — canonical single source."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import numpy as np

    def _is_np_generic(value: object) -> bool:
        return hasattr(np, "generic") and isinstance(value, np.generic)

    def _is_np_ndarray(value: object) -> bool:
        return hasattr(np, "ndarray") and isinstance(value, np.ndarray)
except ImportError:  # pragma: no cover

    def _is_np_generic(value: object) -> bool:
        return False

    def _is_np_ndarray(value: object) -> bool:
        return False


from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form.

    Handles Path, numpy scalars, numpy arrays, Mapping, list, and tuple.
    """
    if isinstance(value, Path):
        return str(value)
    if _is_np_generic(value):
        return value.item()  # type: ignore[union-attr]
    if _is_np_ndarray(value):
        return value.tolist()  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def fmt_float(value: Any) -> str:
    """Format a numeric value to 4 decimal places, or 'n/a' for None/errors."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_int(value: Any) -> str:
    """Format a value as int, or 'n/a' for None/errors."""
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"
