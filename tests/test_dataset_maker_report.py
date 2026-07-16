from __future__ import annotations

from pathlib import Path

from spritelab.dataset_maker.exporter import DatasetMakerExportResult
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import DatasetMakerItem
from spritelab.dataset_maker.report import build_dataset_maker_report


def test_report_contains_summary_counts() -> None:
    report = build_dataset_maker_report([_sprite("a", "accepted"), _sprite("b", "rejected")])

    assert "Imported: 2" in report
    assert "Accepted: 1" in report
    assert "Rejected: 1" in report


def test_report_contains_category_section() -> None:
    report = build_dataset_maker_report([_sprite("a", "accepted", category="item_icon")])

    assert "## Categories" in report
    assert "- item_icon: 1" in report


def test_report_contains_tags_section() -> None:
    report = build_dataset_maker_report([_sprite("a", "accepted", tags=("copper", "vial"))])

    assert "## Tags" in report
    assert "- copper: 1" in report
    assert "- vial: 1" in report


def test_report_contains_palette_size_section() -> None:
    report = build_dataset_maker_report([_sprite("a", "accepted", palette_size=5)])

    assert "## Palette sizes" in report
    assert "- 5: 1" in report


def test_report_includes_export_counts_when_result_is_provided(tmp_path: Path) -> None:
    result = DatasetMakerExportResult(
        output_dir=tmp_path / "v0",
        train_count=8,
        val_count=1,
        test_count=1,
        accepted_count=10,
        excluded_count=2,
        warnings=(),
    )

    report = build_dataset_maker_report([_sprite("a", "accepted")], result)

    assert "Exported: 10" in report
    assert "Train: 8" in report
    assert "Val: 1" in report
    assert "Test: 1" in report


def _sprite(
    sprite_id: str,
    status: str,
    *,
    category: str = "unknown",
    tags: tuple[str, ...] = (),
    palette_size: int | None = None,
) -> ImportedSprite:
    return ImportedSprite(
        item=DatasetMakerItem(
            sprite_id=sprite_id,
            source_path=Path(f"{sprite_id}.png"),
            status=status,
            category=category,
            tags=tags,
            palette_size=palette_size,
        ),
        bundle=None,
        preview_image=None,
        alpha_preview_image=None,
        role_preview_image=None,
        palette_strip_image=None,
        errors=("bad",) if status == "rejected" else (),
        warnings=(),
    )
