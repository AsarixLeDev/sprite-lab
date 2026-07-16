from __future__ import annotations

from spritelab.harvest.sheet_mappings import SheetCoordinate, resolve_sheet_coordinate


def test_ambiguous_coordinates_are_excluded_and_all_candidates_preserved() -> None:
    candidates = [SheetCoordinate(0, 0), SheetCoordinate(0, 1), SheetCoordinate(1, 0)]
    resolution = resolve_sheet_coordinate(candidates, [])
    assert resolution.disposition == "exclude_ambiguous_coordinate"
    assert resolution.selected_coordinate is None
    assert resolution.candidate_coordinates == tuple(candidates)


def test_structured_import_manifest_coordinate_resolves() -> None:
    candidates = [SheetCoordinate(0, 0), SheetCoordinate(0, 1)]
    evidence = [
        {
            "evidence_id": "manifest_sha256:line_7",
            "evidence_type": "import_manifest",
            "auto_metadata": {"sheet_mapping": {"sheet_coordinate": "r000_c001"}},
        }
    ]
    resolution = resolve_sheet_coordinate(candidates, evidence)
    assert resolution.disposition == "resolved"
    assert resolution.selected_coordinate == SheetCoordinate(0, 1)
    assert resolution.candidate_coordinates == tuple(candidates)


def test_filename_pixel_hash_and_semantic_mapping_name_cannot_select_coordinate() -> None:
    candidates = [SheetCoordinate(0, 0), SheetCoordinate(0, 1)]
    non_authoritative = [
        {
            "evidence_id": "not_authoritative",
            "evidence_type": "pixel_identity",
            "final_png_path": "semantic_sword__r000_c001.png",
            "image_sha256": "f" * 64,
            "mapping_name": "sword_at_row_zero_column_one",
            "sheet_coordinate": "r000_c001",
        },
        {
            "evidence_id": "filename_only",
            "evidence_type": "import_manifest",
            "final_png_path": "semantic_sword__r000_c001.png",
            "semantic_filename": "row_zero_column_one_sword.png",
        },
    ]
    resolution = resolve_sheet_coordinate(candidates, non_authoritative)
    assert resolution.disposition == "exclude_ambiguous_coordinate"
    assert resolution.selected_coordinate is None
