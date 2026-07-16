from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from spritelab.dataset_v5.builder import (
    BalancePolicy,
    BuilderConfig,
    _deduplicate,
    _encode_rgba_adapter,
    _validation_reasons,
    _write_deterministic_npz,
    alpha_mask_sha256,
    assign_splits,
    build_dataset,
    build_groups,
    canonical_rgba_sha256,
    enforce_balance,
    source_file_sha256,
    validate_no_leakage,
    verify_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def _record(
    sprite_id: str,
    *,
    color=(255, 0, 0),
    alpha=None,
    pack="pack_a",
    artist="artist_a",
    obj="sword",
    sheet=None,
    variant="",
):
    mask = np.zeros((32, 32), dtype=np.uint8) if alpha is None else np.asarray(alpha, dtype=np.uint8)
    if alpha is None:
        mask[8:24, 12:20] = 1
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[mask > 0, :3] = color
    rgba[mask > 0, 3] = 255
    return {
        "sprite_id": sprite_id,
        "source_pack": pack,
        "source_id": pack,
        "source_family": pack,
        "sub_artist": artist,
        "author": artist,
        "object_name": obj,
        "category": "weapon",
        "source_sheet": sheet or f"{sprite_id}.png",
        "animation_group": "",
        "known_variant_family": "",
        "declared_variant_ids": [variant] if variant else [],
        "exported_rgba_hash": canonical_rgba_sha256(rgba),
        "alpha_mask_hash": alpha_mask_sha256(mask),
        "_rgba": rgba,
        "_alpha": mask,
    }


@pytest.fixture
def preview_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Create the minimal checked contract without relying on developer datasets."""

    preview = tmp_path / "v5_preview"
    preview.mkdir()
    rgba = _record("fixture_sprite")["_rgba"]
    rgba_hash = canonical_rgba_sha256(rgba)
    blob = preview / "blobs" / f"{rgba_hash}.rgba"
    blob.parent.mkdir()
    blob.write_bytes(rgba.tobytes())
    dataset_row = {
        "sprite_id": "fixture_sprite",
        "split": "train",
        "blob_path": f"blobs/{rgba_hash}.rgba",
        "exported_rgba_hash": rgba_hash,
        "exported_width": 32,
        "exported_height": 32,
        "source_id": "fixture_source",
        "source_pack": "fixture_pack",
        "source_url": "https://example.invalid/fixture",
        "license": "cc0",
        "attribution": "test fixture",
        "downloaded_file_hash": hashlib.sha256(b"fixture").hexdigest(),
        "archive_member": "fixture.png",
        "source_image": "fixture.png",
        "author": "fixture_author",
        "resize_policy": "none",
        "original_width": 32,
        "original_height": 32,
        "label_provenance": {"kind": "test_fixture"},
        "suitability_status": "accept",
    }
    (preview / "dataset_manifest.jsonl").write_text(json.dumps(dataset_row, sort_keys=True) + "\n", encoding="utf-8")
    (preview / "group_manifest.jsonl").write_text("", encoding="utf-8")
    (preview / "split_manifest.json").write_text(
        json.dumps({"source_ood_packs": [], "held_out_artists": []}, sort_keys=True) + "\n", encoding="utf-8"
    )
    arrays = _encode_rgba_adapter(rgba, "fixture_sprite", "weapon")
    _write_deterministic_npz(preview / "train.npz", {key: np.expand_dims(value, 0) for key, value in arrays.items()})
    empty = {
        "alpha": np.zeros((0, 32, 32), dtype=np.uint8),
        "index_map": np.zeros((0, 32, 32), dtype=np.uint8),
        "role_map": np.zeros((0, 32, 32), dtype=np.uint8),
        "palette": np.zeros((0, 32, 3), dtype=np.uint8),
        "palette_mask": np.zeros((0, 32), dtype=bool),
        "category_id": np.zeros((0,), dtype=np.int64),
        "sprite_id": np.zeros((0,), dtype="<U1"),
    }
    _write_deterministic_npz(preview / "val.npz", empty)
    _write_deterministic_npz(preview / "test.npz", empty)
    training_row = {
        "schema_version": "training_manifest_v1.0",
        "sprite_id": "fixture_sprite",
        "split": "train",
        "npz_file": "train.npz",
        "npz_row": 0,
        "caption": "fixture sword",
        "object_name": "sword",
        "category": "weapon",
    }
    (preview / "training_manifest.jsonl").write_text(json.dumps(training_row, sort_keys=True) + "\n", encoding="utf-8")
    v4 = tmp_path / "v4"
    v4.mkdir()
    source = v4 / "immutable_source.txt"
    source.write_text("fixture-v4\n", encoding="utf-8")
    return preview, v4, {source.name: source_file_sha256(source)}


def test_canonical_exported_rgba_hashing_is_versioned_and_dimensioned():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    digest = canonical_rgba_sha256(rgba)
    raw_only = hashlib.sha256(rgba.tobytes()).hexdigest()
    assert digest != raw_only
    assert digest == canonical_rgba_sha256(rgba.copy())
    assert digest != canonical_rgba_sha256(np.zeros((16, 64, 4), dtype=np.uint8))


def test_source_png_hash_can_disagree_for_same_decoded_rgba(tmp_path: Path):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[4:12, 4:12] = (10, 20, 30, 255)
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(rgba).save(first, compress_level=0)
    info = PngImagePlugin.PngInfo()
    info.add_text("encoding", "different bytes")
    Image.fromarray(rgba).save(second, compress_level=9, pnginfo=info)
    assert source_file_sha256(first) != source_file_sha256(second)
    with Image.open(first) as a, Image.open(second) as b:
        assert canonical_rgba_sha256(np.asarray(a.convert("RGBA"))) == canonical_rgba_sha256(
            np.asarray(b.convert("RGBA"))
        )


def test_exact_duplicate_grouping_and_exclusion():
    kept, excluded, report = _deduplicate([_record("b"), _record("a")])
    assert [row["sprite_id"] for row in kept] == ["a"]
    assert excluded[0]["reason_code"] == "exact_exported_rgba_duplicate"
    assert report["exact_duplicate_groups"][0]["representative"] == "a"


def test_alpha_and_recolor_grouping():
    relations, _ = build_groups([_record("red"), _record("blue", color=(0, 0, 255))])
    recolors = [row for row in relations if row["kind"] == "exact_alpha_mask_recolor"]
    assert recolors and recolors[0]["hard_split_constraint"]


def test_translation_padding_grouping():
    one = np.zeros((32, 32), dtype=np.uint8)
    two = np.zeros((32, 32), dtype=np.uint8)
    one[2:5, 2:5] = 1
    two[20:23, 21:24] = 1
    relations, _ = build_groups([_record("one", alpha=one), _record("two", alpha=two)])
    assert any(row["kind"] == "translation_padding_variant" for row in relations)


def test_source_sheet_grouping():
    relations, _ = build_groups([_record("one", sheet="sheet.png"), _record("two", color=(0, 1, 2), sheet="sheet.png")])
    assert any(row["kind"] == "source_sheet_siblings" for row in relations)


def test_declared_variant_family_grouping():
    relations, _ = build_groups(
        [_record("one", variant="family:x"), _record("two", color=(2, 3, 4), variant="family:x")]
    )
    assert any(row["kind"] == "declared_variant_family" for row in relations)


def test_train_validation_test_leakage_fails():
    relation = {
        "relation_id": "r1",
        "kind": "declared_variant_family",
        "members": ["a", "b"],
        "hard_split_constraint": True,
    }
    with pytest.raises(ValueError, match="hard split leakage"):
        validate_no_leakage({"a": "train", "b": "test"}, [relation])


def test_source_ood_isolation():
    other_mask = np.zeros((32, 32), dtype=np.uint8)
    other_mask[4:9, 4:9] = 1
    records = [_record("held", pack="held"), _record("main", pack="main", artist="artist_b", alpha=other_mask)]
    relations, groups = build_groups(records)
    del relations
    group_by_id = {member: row["split_group_id"] for row in groups for member in row["members"]}
    result = assign_splits(records, group_by_id, BuilderConfig(source_ood_packs=("held",)))
    assert result["held"] == "source_ood_test"
    assert result["main"] != "source_ood_test"


def test_missing_license_exclusion():
    row = {"license": "unknown", "exported_width": 32, "exported_height": 32, "is_supervised": True}
    assert "missing_or_unknown_license" in _validation_reasons(row)


def test_missing_provenance_exclusion():
    row = {
        "license": "cc0",
        "license_confirmed": True,
        "exported_width": 32,
        "exported_height": 32,
        "is_supervised": True,
    }
    assert any(reason.startswith("missing_provenance:") for reason in _validation_reasons(row))


def test_pack_share_enforcement_is_auditable():
    records = [_record(f"a{i}", pack="dominant", artist=f"x{i}") for i in range(8)] + [
        _record(f"b{i}", color=(0, i, 0), pack=f"p{i}", artist=f"y{i}") for i in range(4)
    ]
    kept, excluded, _ = enforce_balance(
        records, BalancePolicy(max_pack_share=0.5, max_artist_share=1.0), 7, {"recolor": {}, "near": {}}
    )
    counts = {pack: sum(row["source_pack"] == pack for row in kept) for pack in {row["source_pack"] for row in kept}}
    assert max(counts.values()) / len(kept) <= 0.5
    assert any(row["reason_code"] == "pack_share_exceeded" for row in excluded)


def test_artist_share_enforcement_is_auditable():
    records = [_record(f"a{i}", pack=f"p{i}", artist="dominant") for i in range(8)] + [
        _record(f"b{i}", color=(0, i, 0), pack=f"q{i}", artist=f"y{i}") for i in range(4)
    ]
    kept, excluded, _ = enforce_balance(
        records, BalancePolicy(max_pack_share=1.0, max_artist_share=0.5), 7, {"recolor": {}, "near": {}}
    )
    assert sum(row["sub_artist"] == "dominant" for row in kept) / len(kept) <= 0.5
    assert any(row["reason_code"] == "artist_share_exceeded" for row in excluded)


def test_deterministic_stable_split():
    records = [_record(f"r{i}", color=(i, 0, 0), pack=f"p{i}", artist=f"a{i}", obj="common") for i in range(20)]
    _, groups = build_groups(records)
    group_by_id = {member: row["split_group_id"] for row in groups for member in row["members"]}
    first = assign_splits(records, group_by_id, BuilderConfig(seed=42))
    second = assign_splits(list(reversed(records)), group_by_id, BuilderConfig(seed=42))
    assert first == second


def test_immutable_freeze_and_attempted_overwrite_failure(tmp_path: Path):
    output = tmp_path / "frozen"
    output.mkdir()
    (output / "FREEZE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="frozen"):
        build_dataset(v4_dir=tmp_path / "missing", harvest_root=tmp_path, output_dir=output)


def test_v4_untouched_and_preview_verified(preview_fixture: tuple[Path, Path, dict[str, str]]):
    preview, v4, expected_v4_hashes = preview_fixture
    result = verify_dataset(preview, v4_dir=v4, expected_v4_hashes=expected_v4_hashes)
    assert result["v4_unchanged"]
    assert result["ok"]


def test_training_loader_compatibility(preview_fixture: tuple[Path, Path, dict[str, str]]):
    pytest.importorskip("torch")
    preview, _v4, _expected_v4_hashes = preview_fixture
    result = verify_dataset(preview)
    assert result["training_loader_contract"]["ok"]
    assert result["training_loader_contract"]["row_count"] > 0
    from spritelab.training.data import SpriteTrainingDataset

    dataset = SpriteTrainingDataset(preview, preview / "training_manifest.jsonl", split="train", max_records=1)
    assert len(dataset) == 1
    assert tuple(dataset[0]["rgba"].shape) == (4, 32, 32)


def test_cli_smoke(preview_fixture: tuple[Path, Path, dict[str, str]]):
    preview, _v4, _expected_v4_hashes = preview_fixture
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "spritelab.dataset_v5.cli", "verify", "--dataset", str(preview)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"ok": true' in result.stdout
