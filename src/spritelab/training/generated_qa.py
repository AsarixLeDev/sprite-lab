"""QA checks for generated sprite artifact folders."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SPRITE_SIZE = 32


@dataclass
class GeneratedQAResult:
    generated_dir: Path
    sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "generated_dir": str(self.generated_dir),
            "sample_count": int(self.sample_count),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
            "ok": self.ok,
        }

    def to_markdown(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            "# Generated Sprite QA Report",
            "",
            f"Generated: `{self.generated_dir}`",
            f"Status: **{status}**",
            f"Samples: {self.sample_count}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            "",
            "## Checks",
        ]
        for key, value in sorted(self.checks.items()):
            lines.append(f"- {key}: {value}")
        lines.extend(["", "## Warnings"])
        if self.warnings:
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("- (none)")
        lines.extend(["", "## Errors"])
        if self.errors:
            lines.extend(f"- {error}" for error in self.errors)
        else:
            lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)


def qa_generated_sprites(
    generated_dir: str | Path,
    *,
    error_on_fully_transparent: bool = False,
) -> GeneratedQAResult:
    generated_dir = Path(generated_dir)
    result = GeneratedQAResult(generated_dir=generated_dir)
    if not generated_dir.is_dir():
        result.add_error(f"generated directory does not exist: {generated_dir}")
        return result

    manifest_path = generated_dir / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        result.add_error("generated_manifest.jsonl is missing")
        return result

    records = _read_manifest(manifest_path, result)
    result.sample_count = len(records)
    _check_duplicate_ids(records, result)
    _check_reports(generated_dir, records, result)
    for record in records:
        _check_record(generated_dir, record, result, error_on_fully_transparent=error_on_fully_transparent)

    write_generated_qa_reports(result, generated_dir=generated_dir)
    return result


def write_generated_qa_reports(result: GeneratedQAResult, *, generated_dir: Path) -> None:
    generated_dir = Path(generated_dir)
    (generated_dir / "generated_qa_report.json").write_text(
        json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (generated_dir / "generated_qa_report.md").write_text(result.to_markdown(), encoding="utf-8")


def _read_manifest(path: Path, result: GeneratedQAResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            result.add_error(f"generated_manifest.jsonl:{line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            result.add_error(f"generated_manifest.jsonl:{line_no}: record is not a JSON object")
            continue
        records.append(value)
    return records


def _check_duplicate_ids(records: list[Mapping[str, Any]], result: GeneratedQAResult) -> None:
    counts = Counter(str(record.get("sample_id", "")) for record in records)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if sample_id and count > 1)
    empty = [index for index, record in enumerate(records) if not str(record.get("sample_id", "")).strip()]
    result.checks["duplicate_sample_ids"] = duplicates
    if empty:
        result.add_error(f"{len(empty)} manifest records have empty sample_id")
    for sample_id in duplicates:
        result.add_error(f"duplicate sample_id: {sample_id}")


def _check_reports(generated_dir: Path, records: list[Mapping[str, Any]], result: GeneratedQAResult) -> None:
    report_path = generated_dir / "generation_report.json"
    if not report_path.is_file():
        result.add_error("generation_report.json is missing")
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.add_error(f"generation_report.json is invalid JSON: {exc}")
        return
    if not isinstance(report, dict):
        result.add_error("generation_report.json is not a JSON object")
        return
    if int(report.get("sample_count", -1)) != len(records):
        result.add_error("generation_report.json sample_count does not match manifest rows")
    contact_sheet = report.get("contact_sheet")
    if contact_sheet and not (generated_dir / str(contact_sheet)).is_file():
        result.add_error(f"contact sheet referenced by report is missing: {contact_sheet}")
    result.checks["contact_sheet"] = contact_sheet or ""


def _check_record(
    generated_dir: Path,
    record: Mapping[str, Any],
    result: GeneratedQAResult,
    *,
    error_on_fully_transparent: bool,
) -> None:
    sample_id = str(record.get("sample_id", ""))
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    if not paths:
        result.add_error(f"{sample_id}: paths object is missing")
        return
    if not str(record.get("prompt", "")).strip():
        result.add_error(f"{sample_id}: prompt metadata is missing")
    if not str(record.get("prompt_id", "")).strip():
        result.add_error(f"{sample_id}: prompt_id metadata is missing")
    checkpoint = str(record.get("checkpoint", "")).strip()
    if not checkpoint:
        result.add_error(f"{sample_id}: checkpoint provenance is missing")
    elif not Path(checkpoint).is_file():
        result.add_warning(f"{sample_id}: checkpoint path is recorded but not present: {checkpoint}")

    raw_path = _path_from_record(generated_dir, paths, "raw_rgba")
    hard_path = _path_from_record(generated_dir, paths, "hard_rgba")
    indexed_path = _path_from_record(generated_dir, paths, "indexed_png")

    if raw_path is not None:
        _check_png_size(raw_path, result, f"{sample_id}: raw_rgba")
    if hard_path is not None:
        hard = _check_png_size(hard_path, result, f"{sample_id}: hard_rgba")
        if hard is not None:
            _check_hard_alpha(hard, result, sample_id)
            opaque = int(np.count_nonzero(np.asarray(hard.convert("RGBA"))[..., 3] == 255))
            if opaque != int(record.get("alpha_opaque_count", -1)):
                result.add_error(f"{sample_id}: alpha_opaque_count does not match hard_rgba")
            if opaque == 0:
                message = f"{sample_id}: generated sprite is fully transparent"
                result.add_error(message) if error_on_fully_transparent else result.add_warning(message)
    else:
        result.add_error(f"{sample_id}: hard_rgba path is missing")

    if indexed_path is None:
        result.add_error(f"{sample_id}: indexed_png path is missing")
        return
    indexed = _check_png_size(indexed_path, result, f"{sample_id}: indexed_png")
    if indexed is None:
        return
    visible = _visible_color_count(indexed)
    max_colors = int(record.get("max_colors", 32))
    if visible > max_colors:
        result.add_error(f"{sample_id}: indexed_png has {visible} visible colors, above max_colors={max_colors}")
    if visible != int(record.get("visible_color_count", -1)):
        result.add_error(f"{sample_id}: visible_color_count does not match indexed_png")


def _path_from_record(generated_dir: Path, paths: Mapping[str, Any], key: str) -> Path | None:
    value = paths.get(key)
    if not value:
        return None
    path = generated_dir / str(value)
    return path


def _check_png_size(path: Path, result: GeneratedQAResult, label: str) -> Image.Image | None:
    if not path.is_file():
        result.add_error(f"{label}: missing PNG: {path}")
        return None
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        result.add_error(f"{label}: unreadable PNG: {exc}")
        return None
    if image.size != (SPRITE_SIZE, SPRITE_SIZE):
        result.add_error(f"{label}: expected 32x32, got {image.size}")
    return image


def _check_hard_alpha(image: Image.Image, result: GeneratedQAResult, sample_id: str) -> None:
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    values = set(int(value) for value in np.unique(alpha))
    if not values <= {0, 255}:
        result.add_error(f"{sample_id}: hard_rgba alpha contains non-hard values {sorted(values)}")


def _visible_color_count(image: Image.Image) -> int:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    visible = rgba[..., 3] > 0
    if not bool(np.any(visible)):
        return 0
    colors = rgba[..., :3][visible]
    return int(np.unique(colors, axis=0).shape[0])


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QA generated sprite artifact folders.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--error-on-fully-transparent", action="store_true")
    parsed = parser.parse_args(argv)
    result = qa_generated_sprites(
        parsed.generated,
        error_on_fully_transparent=parsed.error_on_fully_transparent,
    )
    print(f"Generated samples: {result.sample_count}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    if not result.ok:
        raise SystemExit(1)
