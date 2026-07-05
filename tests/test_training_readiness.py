from __future__ import annotations

from spritelab.training.palette_report import PaletteSemanticsReport, PaletteSlotStats
from spritelab.training.readiness import (
    build_training_readiness_report,
    format_training_readiness_markdown,
)
from spritelab.training.splits import SplitAssignment


def test_readiness_passes_for_small_valid_export_but_warns() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", category="item_icon", has_role_map=True, reports=True)],
        _split(["sprite_a"]),
        _palette_report(),
        accepted_count=1,
        exported_count=1,
    )

    assert report.passed
    assert _has_issue(report, "TINY_DATASET")


def test_no_accepted_sprites_fails() -> None:
    report = build_training_readiness_report([], _split([]), _palette_report(), accepted_count=0, exported_count=0)

    assert not report.passed
    assert _has_issue(report, "NO_ACCEPTED_SPRITES")


def test_zero_exported_sprites_fails() -> None:
    report = build_training_readiness_report([], _split([]), _palette_report(), accepted_count=2, exported_count=0)

    assert not report.passed
    assert _has_issue(report, "ZERO_EXPORTED_SPRITES")


def test_invalid_accepted_bundle_fails() -> None:
    report = build_training_readiness_report(
        [],
        _split([]),
        _palette_report(),
        accepted_count=1,
        exported_count=0,
        invalid_accepted=(("sprite_a", "invalid bundle"),),
    )

    assert not report.passed
    assert _has_issue(report, "INVALID_ACCEPTED_BUNDLE")


def test_duplicate_leakage_fails() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", reports=True)],
        _split(["sprite_a"]),
        _palette_report(),
        accepted_count=1,
        exported_count=1,
        duplicate_leakage=(("sprite_a", "train", "test"),),
    )

    assert not report.passed
    assert _has_issue(report, "DUPLICATE_LEAKAGE")


def test_missing_dedupe_report_warning_appears() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", quality_report_provided=True)],
        _split(["sprite_a"]),
        _palette_report(),
        accepted_count=1,
        exported_count=1,
    )

    assert _has_issue(report, "NO_DEDUPE_REPORT")


def test_missing_quality_report_warning_appears() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", dedupe_report_provided=True)],
        _split(["sprite_a"]),
        _palette_report(),
        accepted_count=1,
        exported_count=1,
    )

    assert _has_issue(report, "NO_QUALITY_REPORT")


def test_many_unknown_categories_warning_appears() -> None:
    records = [_record(f"sprite_{index}", category="unknown", reports=True) for index in range(4)]

    report = build_training_readiness_report(records, _split([r["sprite_id"] for r in records]), _palette_report(), accepted_count=4, exported_count=4)

    assert _has_issue(report, "MANY_UNKNOWN_CATEGORIES")


def test_high_role_entropy_warning_appears() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", reports=True)],
        _split(["sprite_a"]),
        _palette_report(role_entropy=2.0),
        accepted_count=1,
        exported_count=1,
    )

    assert _has_issue(report, "HIGH_SLOT_ROLE_ENTROPY")


def test_markdown_contains_expected_sections() -> None:
    report = build_training_readiness_report(
        [_record("sprite_a", reports=True)],
        _split(["sprite_a"]),
        _palette_report(),
        accepted_count=1,
        exported_count=1,
    )

    markdown = format_training_readiness_markdown(report)

    assert "# Training Readiness Report" in markdown
    assert "## Verdict" in markdown
    assert "## Counts" in markdown
    assert "## Hard errors" in markdown
    assert "## Warnings" in markdown
    assert "## Recommendations" in markdown


def test_readiness_report_is_deterministic() -> None:
    records = [_record("sprite_a", reports=True)]

    left = build_training_readiness_report(records, _split(["sprite_a"]), _palette_report(), accepted_count=1, exported_count=1)
    right = build_training_readiness_report(records, _split(["sprite_a"]), _palette_report(), accepted_count=1, exported_count=1)

    assert left == right


def _record(
    sprite_id: str,
    *,
    category: str = "item_icon",
    has_role_map: bool = True,
    reports: bool = False,
    dedupe_report_provided: bool = False,
    quality_report_provided: bool = False,
) -> dict:
    return {
        "sprite_id": sprite_id,
        "split": "train",
        "category": category,
        "palette_size": 2,
        "has_role_map": has_role_map,
        "dedupe_report_provided": reports or dedupe_report_provided,
        "quality_report_provided": reports or quality_report_provided,
    }


def _split(sprite_ids: list[str]) -> SplitAssignment:
    return SplitAssignment(
        train=tuple(sprite_ids),
        val=(),
        test=(),
        group_by_sprite_id={sprite_id: sprite_id for sprite_id in sprite_ids},
    )


def _palette_report(role_entropy: float = 0.0) -> PaletteSemanticsReport:
    return PaletteSemanticsReport(
        bundle_count=1,
        accepted_count=1,
        palette_size_counts={2: 1},
        slot_stats=[
            PaletteSlotStats(
                slot_id=1,
                sprite_count=1,
                pixel_count=16,
                mean_rgb=(0.1, 0.1, 0.1),
                mean_luminance=0.1,
                mean_chroma=0.0,
                mean_hue_degrees=None,
                dominant_role_id=1,
                dominant_role_name="outline",
                role_counts={"outline": 16},
                role_entropy=role_entropy,
                mean_usage_fraction=1.0,
            )
        ],
        warnings=[],
    )


def _has_issue(report, code: str) -> bool:
    return any(issue.code == code for issue in report.issues)
