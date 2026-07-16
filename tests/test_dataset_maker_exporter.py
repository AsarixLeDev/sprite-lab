from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.roles import ROLE_MIDTONE
from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    export_dataset_from_imported_sprites,
    make_dataset_maker_split,
)
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import DatasetMakerItem


def test_exports_train_val_test_npz_files(tmp_path: Path) -> None:
    result = export_dataset_from_imported_sprites(
        [_imported("sprite_a")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    assert result.train_count == 1
    assert (tmp_path / "v0" / "train.npz").exists()
    assert (tmp_path / "v0" / "val.npz").exists()
    assert (tmp_path / "v0" / "test.npz").exists()


def test_npz_files_contain_required_keys_shapes_and_dtypes(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("sprite_a")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    with np.load(tmp_path / "v0" / "train.npz") as data:
        assert set(data.files) == {
            "alpha",
            "index_map",
            "role_map",
            "palette",
            "palette_mask",
            "category_id",
            "sprite_id",
        }
        assert data["alpha"].shape == (1, 32, 32)
        assert data["index_map"].shape == (1, 32, 32)
        assert data["role_map"].shape == (1, 32, 32)
        assert data["palette"].shape == (1, 33, 3)
        assert data["palette_mask"].shape == (1, 33)
        assert data["category_id"].shape == (1,)
        assert data["sprite_id"].shape == (1,)
        assert data["alpha"].dtype == np.uint8
        assert data["index_map"].dtype == np.int16
        assert data["role_map"].dtype == np.uint8
        assert data["palette"].dtype == np.uint8
        assert data["palette_mask"].dtype == bool
        assert data["category_id"].dtype == np.int64


def test_palette_padding_mask_and_index_contract_are_correct(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("sprite_a")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    with np.load(tmp_path / "v0" / "train.npz") as data:
        np.testing.assert_array_equal(data["palette"][0, 0], [0, 0, 0])
        assert data["palette_mask"][0, 0]
        assert data["palette_mask"][0, 1]
        assert not data["palette_mask"][0, 2]
        np.testing.assert_array_equal(data["palette"][0, 2:], np.zeros((31, 3), dtype=np.uint8))
        alpha = data["alpha"][0]
        index_map = data["index_map"][0]
        assert np.all(index_map[alpha == 0] == 0)
        assert np.all(index_map[alpha == 1] >= 1)


def test_role_map_fallback_is_transparent_and_unknown(tmp_path: Path) -> None:
    bundle = _bundle("sprite_a")
    bundle = SpriteBundle(
        alpha=bundle.alpha,
        palette=bundle.palette,
        index_map=bundle.index_map,
        role_map=None,
        metadata=bundle.metadata,
    )
    export_dataset_from_imported_sprites(
        [_imported("sprite_a", bundle=bundle)],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    with np.load(tmp_path / "v0" / "train.npz") as data:
        role_map = data["role_map"][0]
        assert role_map[0, 0] == 0
        assert role_map[10, 10] == 255


def test_duplicate_sprite_ids_block_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate sprite_id"):
        export_dataset_from_imported_sprites(
            [_imported("same"), _imported("same")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
        )


def test_no_accepted_sprites_blocks_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no accepted sprites"):
        export_dataset_from_imported_sprites(
            [_imported("sprite_a", status="rejected")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
        )


def test_accepted_sprite_with_invalid_bundle_blocks_export(tmp_path: Path) -> None:
    bundle = _bundle("bad")
    bundle.index_map[0, 0] = 1

    with pytest.raises(ValueError, match="invalid SpriteBundle"):
        export_dataset_from_imported_sprites(
            [_imported("bad", bundle=bundle)],
            DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
        )


def test_rejected_items_are_excluded_and_rejected_jsonl_is_written(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("accepted_a"), _imported("rejected_a", status="rejected")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    with np.load(tmp_path / "v0" / "train.npz") as data:
        assert data["sprite_id"].tolist() == ["accepted_a"]
    rejected_lines = (tmp_path / "v0" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rejected_lines) == 1
    assert json.loads(rejected_lines[0])["sprite_id"] == "rejected_a"


def test_manifests_vocab_and_dataset_config_are_written(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("sprite_a", category="item_icon", tags=("copper", "vial"))],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    output = tmp_path / "v0"
    assert (output / "manifest_train.jsonl").exists()
    assert (output / "manifest_val.jsonl").exists()
    assert (output / "manifest_test.jsonl").exists()
    vocab = json.loads((output / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((output / "dataset_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert vocab["category_to_id"]["unknown"] == 0
    assert vocab["category_to_id"]["item_icon"] == 1
    assert vocab["tag_to_id"] == {"copper": 1, "vial": 2}
    assert vocab["role_names"]["255"] == "unknown"
    assert config["dataset_name"] == "v0"
    assert config["max_palette_slots"] == 32
    assert manifest["sprite_id"] == "sprite_a"
    assert manifest["category"] == "item_icon"


def test_split_assignment_is_deterministic_with_seed() -> None:
    items = [_item(f"sprite_{index}") for index in range(12)]

    first = make_dataset_maker_split(items, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    second = make_dataset_maker_split(items, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)

    assert first == second


def test_manual_split_override_is_respected() -> None:
    items = [_item("sprite_a", split="test"), _item("sprite_b"), _item("sprite_c")]

    split = make_dataset_maker_split(items, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)

    assert split["sprite_a"] == "test"


def test_overwrite_false_blocks_existing_output(tmp_path: Path) -> None:
    config = DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path)
    export_dataset_from_imported_sprites([_imported("sprite_a")], config)

    with pytest.raises(FileExistsError):
        export_dataset_from_imported_sprites([_imported("sprite_b")], config)


def test_overwrite_true_allows_replacing_output(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("sprite_a")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )
    result = export_dataset_from_imported_sprites(
        [_imported("sprite_b")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path, overwrite=True),
    )

    assert result.accepted_count == 1
    with np.load(tmp_path / "v0" / "train.npz") as data:
        assert data["sprite_id"].tolist() == ["sprite_b"]


def _item(
    sprite_id: str,
    *,
    status: str = "accepted",
    category: str = "unknown",
    tags: tuple[str, ...] = (),
    split: str | None = None,
) -> DatasetMakerItem:
    return DatasetMakerItem(
        sprite_id=sprite_id,
        source_path=Path(f"{sprite_id}.png"),
        status=status,
        category=category,
        tags=tags,
        split=split,
        palette_size=1,
        has_role_map=True,
    )


def _imported(
    sprite_id: str,
    *,
    status: str = "accepted",
    category: str = "unknown",
    tags: tuple[str, ...] = (),
    split: str | None = None,
    bundle: SpriteBundle | None = None,
) -> ImportedSprite:
    return ImportedSprite(
        item=_item(sprite_id, status=status, category=category, tags=tags, split=split),
        bundle=_bundle(sprite_id) if bundle is None and status == "accepted" else bundle,
        preview_image=None,
        alpha_preview_image=None,
        role_preview_image=None,
        palette_strip_image=None,
        errors=(),
        warnings=(),
    )


def _bundle(sprite_id: str) -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 1
    role_map[10:14, 10:14] = ROLE_MIDTONE
    return SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], [120, 50, 80]], dtype=np.uint8),
        index_map=index_map,
        role_map=role_map,
        metadata=SpriteMetadata(id=sprite_id, palette_size=1),
    )
