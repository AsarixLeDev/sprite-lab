"""Validated declarative metadata for source-owned sprite sheets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from spritelab.harvest.config_loader import load_sheet_mappings_config
from spritelab.harvest.label_taxonomy import CATEGORY_VALUES

_CELL_RE = re.compile(r"__r(?P<row>\d+)_c(?P<column>\d+)$")


@dataclass(frozen=True)
class SheetMapping:
    name: str
    source_id: str
    file_glob: str
    tile_width: int
    tile_height: int
    metadata: Mapping[str, str]
    coordinates: Mapping[str, Mapping[str, str]]
    excluded_cells: frozenset[str]


def load_sheet_mappings() -> tuple[SheetMapping, ...]:
    raw = load_sheet_mappings_config({"sheet_mappings": {}})
    if set(raw) != {"schema_version", "sheet_mappings"} or not isinstance(raw.get("sheet_mappings"), Mapping):
        raise ValueError("invalid sheet_mappings config: expected schema_version and sheet_mappings")
    result: list[SheetMapping] = []
    for name, item in dict(raw["sheet_mappings"]).items():
        if not isinstance(name, str) or not isinstance(item, Mapping):
            raise ValueError("invalid sheet mapping entry")
        allowed = {"source_id", "files", "metadata", "coordinates", "excluded_cells"}
        if set(item) - allowed:
            raise ValueError(f"invalid sheet mapping {name!r}: unknown keys {sorted(set(item) - allowed)}")
        if not isinstance(item.get("source_id"), str) or not isinstance(item.get("files"), list):
            raise ValueError(f"invalid sheet mapping {name!r}: source_id and files are required")
        base_metadata = _string_map(item.get("metadata", {}), name, "metadata")
        coordinate_map = _coordinate_map(item.get("coordinates", {}), name)
        excluded = frozenset(str(value) for value in item.get("excluded_cells", ()))
        for file_item in item["files"]:
            if not isinstance(file_item, Mapping) or set(file_item) - {"glob", "tile_width", "tile_height", "metadata"}:
                raise ValueError(f"invalid sheet mapping {name!r}: malformed file rule")
            glob = file_item.get("glob")
            width, height = file_item.get("tile_width"), file_item.get("tile_height")
            if (
                not isinstance(glob, str)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or width < 1
                or height < 1
            ):
                raise ValueError(f"invalid sheet mapping {name!r}: file glob and positive tile dimensions required")
            result.append(
                SheetMapping(
                    name,
                    item["source_id"],
                    glob,
                    width,
                    height,
                    {**base_metadata, **_string_map(file_item.get("metadata", {}), name, "file metadata")},
                    coordinate_map,
                    excluded,
                )
            )
    return tuple(result)


def metadata_for_sheet_cell(source_id: str, source_file: str, tile_path: str | Path) -> dict[str, str]:
    """Return mapping evidence for one sliced tile; unmapped cells are empty."""

    stem = Path(tile_path).stem
    match = _CELL_RE.search(stem)
    if not match:
        return {}
    row, column = int(match["row"]), int(match["column"])
    cell = f"r{row:03d}_c{column:03d}"
    for mapping in load_sheet_mappings():
        if mapping.source_id != source_id or not fnmatchcase(source_file.replace("\\", "/"), mapping.file_glob):
            continue
        if cell in mapping.excluded_cells:
            return {"mapping_name": mapping.name, "mapping_excluded": "true", "sheet_coordinate": cell}
        values = _render(dict(mapping.metadata), row=row, column=column)
        values.update(_render(dict(mapping.coordinates.get(cell, {})), row=row, column=column))
        values.update(
            {
                "mapping_name": mapping.name,
                "sheet_coordinate": cell,
                "native_resolution": f"{mapping.tile_width}x{mapping.tile_height}",
                "source_sheet": source_file,
            }
        )
        if values.get("category", "") and values["category"] not in CATEGORY_VALUES:
            raise ValueError(f"sheet mapping {mapping.name!r} has invalid category {values['category']!r}")
        return values
    return {}


def _string_map(value: object, name: str, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, (str, int)) for key, item in value.items()
    ):
        raise ValueError(f"invalid sheet mapping {name!r}: {field} must be a string map")
    return {str(key): str(item) for key, item in value.items()}


def _coordinate_map(value: object, name: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid sheet mapping {name!r}: coordinates must be a mapping")
    return {str(key): _string_map(item, name, f"coordinates.{key}") for key, item in value.items()}


def _render(values: Mapping[str, str], *, row: int, column: int) -> dict[str, str]:
    try:
        return {key: value.format(row=row, column=column) for key, value in values.items()}
    except KeyError as exc:
        raise ValueError(f"unknown sheet mapping template key {exc.args[0]!r}") from exc
