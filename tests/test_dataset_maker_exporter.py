from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import spritelab.dataset_maker.exporter as exporter_module
from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.roles import ROLE_MIDTONE
from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    commit_anchored_dataset_maker_export,
    export_dataset_from_imported_sprites,
    export_dataset_from_imported_sprites_anchored,
    make_dataset_maker_split,
    verify_anchored_dataset_maker_export,
)
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import DatasetMakerItem
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation, open_anchored_directory


def test_exports_train_val_test_npz_files(tmp_path: Path) -> None:
    result = export_dataset_from_imported_sprites(
        [_imported("sprite_a")],
        DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path),
    )

    assert result.train_count == 1
    assert (tmp_path / "v0" / "train.npz").exists()
    assert (tmp_path / "v0" / "val.npz").exists()
    assert (tmp_path / "v0" / "test.npz").exists()


def test_anchored_export_publishes_exact_fresh_dataset(tmp_path: Path) -> None:
    output_root = tmp_path / "exports"
    output_root.mkdir()

    with open_anchored_directory(output_root, output_root) as output_parent:
        anchored = export_dataset_from_imported_sprites_anchored(
            [_imported("sprite_a")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=output_root),
            output_parent=output_parent,
        )
        inventory = _flat_inventory(output_root / "v0")
        with pytest.raises(FileNotFoundError):
            verify_anchored_dataset_maker_export(
                output_parent,
                "v0",
                expected_inventory=inventory,
            )
        commit_anchored_dataset_maker_export(
            output_parent,
            "v0",
            expected_parent_identity=anchored.parent_identity,
            expected_directory_identity=anchored.directory_identity,
            expected_inventory=inventory,
        )
        marker = verify_anchored_dataset_maker_export(
            output_parent,
            "v0",
            expected_inventory=inventory,
        )

    result = anchored.result
    assert result.output_dir == output_root / "v0"
    assert marker["inventory"] == inventory
    assert (output_root / "v0.commit.json").is_file()
    assert {path.name for path in result.output_dir.iterdir()} == {
        "dataset_config.json",
        "dataset_report.md",
        "manifest_test.jsonl",
        "manifest_train.jsonl",
        "manifest_val.jsonl",
        "rejected.jsonl",
        "test.npz",
        "train.npz",
        "val.npz",
        "vocab.json",
    }


def test_anchored_export_loader_rejects_payload_changed_after_commit(tmp_path: Path) -> None:
    output_root = tmp_path / "exports"
    output_root.mkdir()
    with open_anchored_directory(output_root, output_root) as output_parent:
        anchored = export_dataset_from_imported_sprites_anchored(
            [_imported("sprite_a")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=output_root),
            output_parent=output_parent,
        )
        inventory = _flat_inventory(output_root / "v0")
        commit_anchored_dataset_maker_export(
            output_parent,
            "v0",
            expected_parent_identity=anchored.parent_identity,
            expected_directory_identity=anchored.directory_identity,
            expected_inventory=inventory,
        )

    (output_root / "v0" / "vocab.json").write_bytes(b"changed-after-commit")
    with open_anchored_directory(output_root, output_root) as output_parent:
        with pytest.raises(UnsafeFilesystemOperation, match="export"):
            verify_anchored_dataset_maker_export(
                output_parent,
                "v0",
                expected_inventory=inventory,
            )


def test_anchored_export_rejects_staging_rename_to_outside_link_without_writing_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel_bytes = b"outside-must-remain-byte-identical"
    sentinel.write_bytes(sentinel_bytes)
    original_open = AnchoredDirectory.open_file
    staging_path: Path | None = None
    parked = output_root / "staging-held-by-test"
    attacked = False

    def swap_then_open(anchor: AnchoredDirectory, name: str, flags: int, mode: int = 0o600) -> int:
        nonlocal attacked, staging_path
        if not attacked and anchor.directory.name == "v0":
            attacked = True
            staging_path = anchor.directory
            os.replace(staging_path, parked)
            try:
                os.symlink(outside, staging_path, target_is_directory=True)
            except OSError as exc:
                os.replace(parked, staging_path)
                pytest.skip(f"directory symbolic links are unavailable in this test session: {exc}")
        return original_open(anchor, name, flags, mode)

    monkeypatch.setattr(AnchoredDirectory, "open_file", swap_then_open)
    try:
        with open_anchored_directory(output_root, output_root) as output_parent:
            with pytest.raises((OSError, UnsafeFilesystemOperation)):
                export_dataset_from_imported_sprites_anchored(
                    [_imported("sprite_a")],
                    DatasetMakerExportConfig(dataset_name="v0", output_root=output_root),
                    output_parent=output_parent,
                )
        assert attacked is True
        assert sentinel.read_bytes() == sentinel_bytes
        assert {path.name for path in outside.iterdir()} == {"sentinel.bin"}
        assert not (output_root / "v0.commit.json").exists()
    finally:
        if staging_path is not None and staging_path.is_symlink():
            staging_path.unlink()
        if staging_path is not None and parked.is_dir() and not staging_path.exists():
            os.replace(parked, staging_path)


def test_anchored_export_detects_hard_link_substitution_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel_bytes = b"outside-must-remain-byte-identical"
    sentinel.write_bytes(sentinel_bytes)
    original_open = AnchoredDirectory.open_file
    real_fsync = exporter_module.os.fsync
    first_output: Path | None = None
    attacked = False

    def remember_output(anchor: AnchoredDirectory, name: str, flags: int, mode: int = 0o600) -> int:
        nonlocal first_output
        if first_output is None and anchor.directory.name == "v0":
            first_output = anchor.directory / name
        return original_open(anchor, name, flags, mode)

    def link_before_fsync(descriptor: int) -> None:
        nonlocal attacked
        if not attacked and first_output is not None:
            try:
                os.link(first_output, outside / "hostile-hard-link.bin")
            except OSError as exc:
                pytest.skip(f"hard links are unavailable in this test session: {exc}")
            attacked = True
        real_fsync(descriptor)

    monkeypatch.setattr(AnchoredDirectory, "open_file", remember_output)
    monkeypatch.setattr(exporter_module.os, "fsync", link_before_fsync)
    with open_anchored_directory(output_root, output_root) as output_parent:
        with pytest.raises(UnsafeFilesystemOperation, match="changed after writing"):
            export_dataset_from_imported_sprites_anchored(
                [_imported("sprite_a")],
                DatasetMakerExportConfig(dataset_name="v0", output_root=output_root),
                output_parent=output_parent,
            )

    assert attacked is True
    assert sentinel.read_bytes() == sentinel_bytes
    assert not (output_root / "v0.commit.json").exists()


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


def test_overwrite_rejects_linked_output_and_preserves_outside_tree(tmp_path: Path) -> None:
    output_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    try:
        os.symlink(outside, output_root / "v0", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this test session")

    with pytest.raises(ValueError, match=r"escapes|link|reparse"):
        export_dataset_from_imported_sprites(
            [_imported("sprite_a")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=output_root, overwrite=True),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_failed_overwrite_keeps_previous_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path)
    export_dataset_from_imported_sprites([_imported("sprite_a")], config)

    def fail_publish(*_args, **_kwargs):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr("spritelab.dataset_maker.exporter._publish_export", fail_publish)
    with pytest.raises(OSError, match="synthetic publish failure"):
        export_dataset_from_imported_sprites(
            [_imported("sprite_b")],
            DatasetMakerExportConfig(dataset_name="v0", output_root=tmp_path, overwrite=True),
        )

    with np.load(tmp_path / "v0" / "train.npz") as data:
        assert data["sprite_id"].tolist() == ["sprite_a"]


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


def _flat_inventory(directory: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: {"sha256": sha256(path.read_bytes()).hexdigest(), "byte_count": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
