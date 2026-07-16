from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.codec.roles import ROLE_MIDTONE, ROLE_OUTLINE
from spritelab.training.palette_report import (
    build_palette_semantics_report,
    format_palette_semantics_report_markdown,
    write_palette_semantics_report_json,
)


def test_report_builds_for_one_bundle_and_counts_palette_sizes(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)

    report = build_palette_semantics_report([bundle_dir])

    assert report.bundle_count == 1
    assert report.accepted_count == 1
    assert report.palette_size_counts == {2: 1}


def test_slot_stats_count_sprites_and_pixels(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)

    report = build_palette_semantics_report([bundle_dir])
    slot_one = next(stat for stat in report.slot_stats if stat.slot_id == 1)

    assert slot_one.sprite_count == 1
    assert slot_one.pixel_count == 8


def test_luminance_chroma_and_dominant_role_are_computed(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)

    report = build_palette_semantics_report([bundle_dir])
    slot_one = next(stat for stat in report.slot_stats if stat.slot_id == 1)

    assert slot_one.mean_luminance is not None
    assert slot_one.mean_chroma is not None
    assert slot_one.dominant_role_id == ROLE_OUTLINE
    assert slot_one.dominant_role_name == "outline"


def test_role_entropy_is_zero_for_one_role(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)

    report = build_palette_semantics_report([bundle_dir])
    slot_two = next(stat for stat in report.slot_stats if stat.slot_id == 2)

    assert slot_two.role_entropy == 0.0


def test_missing_role_maps_generate_warning(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=False)

    report = build_palette_semantics_report([bundle_dir])

    assert any("no role_map" in warning for warning in report.warnings)


def test_markdown_report_contains_overview_and_slot_table(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)

    markdown = format_palette_semantics_report_markdown(build_palette_semantics_report([bundle_dir]))

    assert "# Palette Slot Semantics Report" in markdown
    assert "## Overview" in markdown
    assert "| Slot | Sprites | Pixels |" in markdown


def test_json_report_is_serializable(tmp_path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundle", "sprite_a", include_role_map=True)
    report_path = tmp_path / "palette_report.json"

    write_palette_semantics_report_json(build_palette_semantics_report([bundle_dir]), report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["accepted_count"] == 1
    assert data["slot_stats"]


def _write_bundle(directory: Path, sprite_id: str, *, include_role_map: bool) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = None
    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 2
    index_map[10, 10:14] = 1
    index_map[13, 10:14] = 1
    if include_role_map:
        role_map = np.zeros((32, 32), dtype=np.uint8)
        role_map[index_map == 1] = ROLE_OUTLINE
        role_map[index_map == 2] = ROLE_MIDTONE
    palette = np.array([[0, 0, 0], [20, 20, 25], [180, 80, 130]], dtype=np.uint8)
    bundle = SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=role_map,
        metadata=SpriteMetadata(id=sprite_id, palette_size=2),
    )
    save_bundle(bundle, directory)
    return directory
