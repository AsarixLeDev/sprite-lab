"""Deterministic sprite-sheet grid proposals for the existing extraction backend."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import numpy as np

SHEET_PLAN_SCHEMA = "spritelab.dataset.sheet_plan.v1"
EXTRACTION_POLICY_VERSION = "spritelab.dataset.sheet_extraction_policy.v1"
_MAX_CELLS = 1024


def propose_sheet_plan(rgba: np.ndarray) -> dict[str, Any]:
    """Propose crops separated by fully transparent bands; ambiguity is explicit."""

    alpha = rgba[:, :, 3]
    column_segments = _content_segments(alpha.max(axis=0) > 0)
    row_segments = _content_segments(alpha.max(axis=1) > 0)
    reasons: list[str] = []
    if len(column_segments) < 2 and len(row_segments) < 2:
        reasons.append("no_fully_transparent_separator_bands")
    if len(column_segments) * len(row_segments) > _MAX_CELLS:
        reasons.append("too_many_candidate_cells")
    columns = len(column_segments)
    rows = len(row_segments)
    if columns > 1 and len({end - start for start, end in column_segments}) != 1:
        reasons.append("nonuniform_column_widths")
    if rows > 1 and len({end - start for start, end in row_segments}) != 1:
        reasons.append("nonuniform_row_heights")
    column_cells, column_cell_reasons = _grid_cells(column_segments, alpha.shape[1], axis="column")
    row_cells, row_cell_reasons = _grid_cells(row_segments, alpha.shape[0], axis="row")
    reasons.extend(column_cell_reasons)
    reasons.extend(row_cell_reasons)
    crops: list[list[int]] = []
    empty_cells = 0
    if column_cells and row_cells and len(column_cells) * len(row_cells) <= _MAX_CELLS:
        for top, bottom in row_cells:
            for left, right in column_cells:
                if not np.any(alpha[top:bottom, left:right]):
                    empty_cells += 1
                    continue
                crops.append([int(left), int(top), int(right), int(bottom)])
    if empty_cells:
        reasons.append("empty_grid_cells")
    if len(crops) < 2:
        reasons.append("fewer_than_two_populated_cells")
    unambiguous = not reasons
    return {
        "schema_version": SHEET_PLAN_SCHEMA,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "grid_columns": columns,
        "grid_rows": rows,
        "crops": crops if unambiguous else [],
        "proposed_crops": crops,
        "empty_cells": empty_cells,
        "unambiguous": unambiguous,
        "confidence": 1.0 if unambiguous else 0.0,
        "identity_contract_passed": unambiguous,
        "ambiguity_reasons": sorted(reasons),
        "separator_policy": "fully_transparent_row_and_column_bands",
    }


def uniform_grid_plan(rgba: np.ndarray, *, columns: int, rows: int) -> dict[str, Any]:
    """Build an explicit user-adjusted uniform grid over the full sheet."""

    height, width = rgba.shape[:2]
    if columns < 1 or rows < 1 or columns * rows > _MAX_CELLS:
        raise ValueError("Grid must have at least one cell and a bounded cell count.")
    if width % columns or height % rows:
        raise ValueError(f"A {columns}x{rows} grid does not divide the {width}x{height} sheet into whole pixels.")
    cell_width = width // columns
    cell_height = height // rows
    alpha = rgba[:, :, 3]
    crops: list[list[int]] = []
    for row in range(rows):
        for column in range(columns):
            left = column * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            if np.any(alpha[top:bottom, left:right]):
                crops.append([left, top, right, bottom])
    if len(crops) < 1:
        raise ValueError("The requested grid produces no non-empty cells.")
    return {
        "schema_version": SHEET_PLAN_SCHEMA,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "grid_columns": columns,
        "grid_rows": rows,
        "crops": crops,
        "proposed_crops": crops,
        "empty_cells": columns * rows - len(crops),
        "unambiguous": True,
        "confidence": 1.0,
        "identity_contract_passed": True,
        "ambiguity_reasons": [],
        "separator_policy": "explicit_user_uniform_grid",
    }


def plan_is_usable(plan: Mapping[str, Any] | None) -> bool:
    return bool(plan) and bool(plan.get("unambiguous")) and len(plan.get("crops") or ()) >= 1


def _content_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, filled in enumerate(mask.tolist()):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _grid_cells(
    segments: list[tuple[int, int]],
    extent: int,
    *,
    axis: str,
) -> tuple[list[tuple[int, int]], list[str]]:
    if not segments:
        return [], []
    if len(segments) == 1:
        return [(0, extent)], []
    starts = [start for start, _end in segments]
    strides = [second - first for first, second in pairwise(starts)]
    reasons: list[str] = []
    if len(set(strides)) == 1:
        stride = strides[0]
        offset = starts[0] % stride
        aligned = all(start == offset + index * stride for index, start in enumerate(starts))
        if aligned and extent % stride == 0 and len(segments) == extent // stride:
            return [(index * stride, (index + 1) * stride) for index in range(len(segments))], []
    else:
        reasons.append(f"nonuniform_{axis}_spacing")
    boundaries = [0]
    boundaries.extend((left[1] + right[0]) // 2 for left, right in pairwise(segments))
    boundaries.append(extent)
    return list(pairwise(boundaries)), reasons
