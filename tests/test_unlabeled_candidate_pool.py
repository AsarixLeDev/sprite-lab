from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.unlabeled_pool.builder import (
    PoolConfig,
    alpha_mask_sha256,
    build_pool,
    canonical_rgba_sha256,
    choose_representative,
    discover,
    freeze_pool,
    group_variants,
    hash_json,
    validate_provenance,
    verify_pool,
)


def _rgba(color: tuple[int, int, int] = (200, 30, 20), *, offset: int = 0) -> np.ndarray:
    value = np.zeros((32, 32, 4), dtype=np.uint8)
    value[7 + offset : 24 + offset, 12:20, :3] = color
    value[7 + offset : 24 + offset, 12:20, 3] = 255
    return value


def _record(sprite_id: str, color=(200, 30, 20), *, variant="", run="run", suitability="accept") -> dict:
    rgba = _rgba(color)
    return {
        "sprite_id": sprite_id,
        "source_id": "shade_16x16_weapons",
        "pack_id": "shade_16x16_weapons",
        "pack_name": "Shade weapons",
        "author": "Shade",
        "sub_artist": "Shade",
        "license": "cc0",
        "license_url": "",
        "license_confirmed": True,
        "attribution": "Shade",
        "source_url": "https://example.invalid/shade",
        "downloaded_file_hash": "a" * 64,
        "archive_member": "weapons.png",
        "source_image": "weapons.png",
        "source_sheet": "weapons.png",
        "cell_coordinates": "r000_c000",
        "native_dimensions": {"width": 16, "height": 16},
        "resize_policy": "transparent_center_pad_16x16_to_32x32",
        "declared_variant_group": variant,
        "declared_material": "",
        "source_run": run,
        "source_runs": [run],
        "broad_pack_type": "weapon",
        "acquisition_policy_version": "unlabeled_acquisition_policy_v1",
        "suitability_status": suitability,
        "suitability_score": 1.0,
        "suitability_reason_codes": [],
        "suitability_config_hash": "b" * 64,
        "quality_confidence": 1.0,
        "exported_width": 32,
        "exported_height": 32,
        "exported_rgba_hash": canonical_rgba_sha256(rgba),
        "alpha_mask_hash": alpha_mask_sha256(rgba[..., 3]),
        "normalized_alpha_hash": alpha_mask_sha256(rgba[..., 3]),
        "_rgba": rgba,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _make_harvest(root: Path, *, flare: bool = False) -> Path:
    run = root / ("sota_v2_flare_armor_final2" if flare else "sota_v2_shade_weapons_validated")
    run.mkdir(parents=True)
    source_id = "flare_armor" if flare else "shade_16x16_weapons"
    source = {
        "source_id": source_id,
        "source_name": "Pack",
        "author": "Shade",
        "source_url": "https://example.invalid/source",
        "sha256": "a" * 64,
        "license": {"license": "cc0", "user_confirmed": True, "license_url": ""},
    }
    _write_jsonl(run / "sources.jsonl", [source])
    imported = []
    for index, color in enumerate(((180, 20, 20), (20, 180, 20), (20, 20, 180), (180, 180, 20))):
        image = _rgba(color)
        path = run / f"sprite_{index}.png"
        Image.fromarray(image).save(path)
        imported.append(
            {
                "sprite_id": f"{source_id}_r000_c000_{index}",
                "source_id": source_id,
                "source_name": "Pack",
                "author": "Shade",
                "license": "cc0",
                "status": "accepted",
                "relative_path": f"material_{index}.png",
                "final_png_path": str(path),
                "auto_metadata": {
                    "sheet_mapping": {
                        "source_sheet": f"material_{index}.png",
                        "sheet_coordinate": "r000_c000",
                        "native_resolution": "16x16",
                        "variant_group_id": "shade:r000:c000",
                        "material": f"m{index}",
                    }
                },
            }
        )
    _write_jsonl(run / "imported.jsonl", imported)
    _write_jsonl(run / "candidates.jsonl", [])
    return run


def test_provenance_failure_is_blocked():
    row = _record("missing")
    row["source_url"] = ""
    valid, blocked = validate_provenance([row])
    assert not valid
    assert "source_url" in blocked[0]["missing_provenance_fields"]


def test_license_failure_is_fail_closed():
    row = _record("license")
    row["license"] = "unknown"
    valid, blocked = validate_provenance([row])
    assert not valid
    assert blocked[0]["license_failure"]


def test_exact_duplicate_representative_selection():
    rows, groups, _ = group_variants([_record("b"), _record("a")], PoolConfig())
    geometry = next(row for row in groups if row["group_kind"] == "geometry_family")
    assert geometry["representative_sprite_id"] == "a"
    assert {row["acquisition_status"] for row in rows} == {"duplicate_representative", "duplicate_variant"}


def test_recolor_representative_selection():
    rows, groups, stats = group_variants([_record("red"), _record("blue", (20, 20, 180))], PoolConfig())
    assert stats["recolor_families"] == 1
    assert sum(row["annotation_representative"] for row in rows) == 1
    assert any(row["group_kind"] == "alpha_mask_recolor" for row in groups)


def test_annotation_priority_is_deterministic():
    first, _, _ = group_variants([_record("a"), _record("b", (0, 255, 0))], PoolConfig())
    second, _, _ = group_variants([_record("b", (0, 255, 0)), _record("a")], PoolConfig())
    assert {row["sprite_id"]: row["annotation_priority_score"] for row in first} == {
        row["sprite_id"]: row["annotation_priority_score"] for row in second
    }
    assert choose_representative(first, PoolConfig())["sprite_id"] == "a"


def test_shade_like_four_recolor_family():
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    rows, groups, stats = group_variants(
        [_record(f"shade_{index}", color, variant="shade:r0:c0") for index, color in enumerate(colors)], PoolConfig()
    )
    geometry = [row for row in groups if row["group_kind"] == "geometry_family"]
    assert len(rows) == 4
    assert len(geometry) == 1
    assert stats["expected_label_propagation_savings"] == 3
    assert sum(row["annotation_representative"] for row in rows) == 1


def test_immutable_freeze_and_overwrite_refusal(tmp_path: Path):
    output = tmp_path / "pool"
    output.mkdir()
    for name in (
        "candidate_manifest.jsonl",
        "group_manifest.jsonl",
        "annotation_queue.jsonl",
        "quarantine_manifest.jsonl",
        "excluded_manifest.jsonl",
    ):
        (output / name).write_text("", encoding="utf-8")
    for name in ("license_manifest.json", "summary.json"):
        (output / name).write_text("{}\n", encoding="utf-8")
    (output / "README.md").write_text("pool\n", encoding="utf-8")
    (output / "blobs").mkdir()
    freeze_pool(output, {}, hash_json({}))
    with pytest.raises(FileExistsError, match="frozen"):
        freeze_pool(output, {}, hash_json({}))


def test_build_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "pool"
    output.mkdir()
    (output / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="frozen"):
        build_pool(harvest_root=tmp_path, output_dir=output)


def test_flare_exclusion(tmp_path: Path):
    harvest = tmp_path / "harvest_runs"
    _make_harvest(harvest, flare=True)
    result = build_pool(harvest_root=harvest, output_dir=tmp_path / "pool")
    assert result["summary"]["flare_retained_count"] == 0
    assert result["candidate_count"] == 0


def test_no_supervised_label_fields_and_v5_blob_compatibility(tmp_path: Path):
    harvest = tmp_path / "harvest_runs"
    _make_harvest(harvest)
    output = tmp_path / "pool"
    result = build_pool(harvest_root=harvest, output_dir=output)
    assert result["v5_blob_store_compatible"]
    rows = [json.loads(line) for line in (output / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    forbidden = {"object_name", "category", "label", "train", "validation", "test", "split"}
    assert all(not (forbidden & set(row)) for row in rows)


def test_reproducible_hashes(tmp_path: Path):
    harvest = tmp_path / "harvest_runs"
    _make_harvest(harvest)
    first = build_pool(harvest_root=harvest, output_dir=tmp_path / "one")
    second = build_pool(harvest_root=harvest, output_dir=tmp_path / "two")
    assert first["freeze"]["content_manifest_hash"] == second["freeze"]["content_manifest_hash"]
    assert verify_pool(tmp_path / "one")["ok"]


def test_missing_manifest_hash_is_recovered_from_download(tmp_path: Path):
    harvest = tmp_path / "harvest_runs"
    run = _make_harvest(harvest)
    downloads = run / "downloads"
    downloads.mkdir()
    archive = downloads / "shade_16x16_weapons.zip"
    archive.write_bytes(b"downloaded archive bytes")
    source_path = run / "sources.jsonl"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["sha256"] = ""
    _write_jsonl(source_path, [source])
    _, occurrences, source_hashes = discover(harvest)
    expected = __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
    assert {row["downloaded_file_hash"] for row in occurrences} == {expected}
    assert any(path.endswith("shade_16x16_weapons.zip") for path in source_hashes)
