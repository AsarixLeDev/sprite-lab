from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.codec.roles import ROLE_MIDTONE, ROLE_TRANSPARENT, ROLE_UNKNOWN
from spritelab.curation.manifest import CurationDecision, append_curation_decision
from spritelab.training.export import TrainingExportConfig, export_training_dataset


def test_export_uses_only_accepted_sprites_and_writes_outputs(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    curation = tmp_path / "curation.jsonl"
    for sprite_id, status in [
        ("accepted_a", "accepted"),
        ("rejected_a", "rejected"),
        ("quarantine_a", "quarantine"),
        ("needs_fix_a", "needs_fix"),
    ]:
        _write_bundle(bundles / sprite_id, sprite_id)
        append_curation_decision(curation, CurationDecision(sprite_id, status))
    _write_bundle(bundles / "uncurated_a", "uncurated_a")

    result = export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    assert result.accepted_count == 1
    assert result.train_count == 1
    assert result.val_count == 0
    assert result.test_count == 0
    assert (tmp_path / "export" / "train.npz").exists()
    assert (tmp_path / "export" / "val.npz").exists()
    assert (tmp_path / "export" / "test.npz").exists()
    assert (tmp_path / "export" / "manifest_train.jsonl").exists()
    assert (tmp_path / "export" / "vocab.json").exists()
    assert (tmp_path / "export" / "export_config.json").exists()


def test_npz_arrays_have_contract_shapes_and_dtypes(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, ["sprite_a"])

    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    with np.load(tmp_path / "export" / "train.npz") as data:
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
        assert data["alpha"].dtype == np.uint8
        assert data["index_map"].shape == (1, 32, 32)
        assert data["index_map"].dtype == np.int16
        assert data["role_map"].dtype == np.uint8
        assert data["palette"].shape == (1, 33, 3)
        assert data["palette"].dtype == np.uint8
        assert data["palette_mask"].shape == (1, 33)
        assert data["palette_mask"].dtype == bool
        assert data["category_id"].dtype == np.int64
        assert data["sprite_id"][0] == "sprite_a"


def test_palette_padding_and_mask_are_correct(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, ["sprite_a"])

    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    with np.load(tmp_path / "export" / "train.npz") as data:
        np.testing.assert_array_equal(data["palette"][0, 0], [0, 0, 0])
        assert data["palette_mask"][0, 0]
        assert data["palette_mask"][0, 1]
        assert not data["palette_mask"][0, 2]
        np.testing.assert_array_equal(data["palette"][0, 2:], np.zeros((31, 3), dtype=np.uint8))


def test_role_map_fallback_works_when_missing(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    curation = tmp_path / "curation.jsonl"
    _write_bundle(bundles / "sprite_a", "sprite_a", include_role_map=False)
    append_curation_decision(curation, CurationDecision("sprite_a", "accepted"))

    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    with np.load(tmp_path / "export" / "train.npz") as data:
        role_map = data["role_map"][0]
        assert role_map[0, 0] == ROLE_TRANSPARENT
        assert role_map[10, 10] == ROLE_UNKNOWN


def test_manifest_vocab_and_config_are_written(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, ["sprite_a"])

    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    manifest_line = json.loads((tmp_path / "export" / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    vocab = json.loads((tmp_path / "export" / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((tmp_path / "export" / "export_config.json").read_text(encoding="utf-8"))

    assert manifest_line["sprite_id"] == "sprite_a"
    assert manifest_line["split"] == "train"
    assert "item_icon" in vocab["category_to_id"]
    assert config["accepted_count"] == 1


def test_same_seed_produces_same_split(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, [f"sprite_{index}" for index in range(12)])

    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "left", seed=42)
    )
    export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "right", seed=42)
    )

    assert _manifest_ids(tmp_path / "left" / "manifest_train.jsonl") == _manifest_ids(tmp_path / "right" / "manifest_train.jsonl")


def test_grouped_duplicates_do_not_cross_splits(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, [f"sprite_{index}" for index in range(8)])
    dedupe_path = tmp_path / "dedupe_report.json"
    dedupe_path.write_text(
        json.dumps(
            {
                "exact_groups": [{"ids": ["sprite_0", "sprite_1"], "kind": "DECODED_RGBA_SHA256"}],
                "near_groups": [],
            }
        ),
        encoding="utf-8",
    )

    export_training_dataset(
        TrainingExportConfig(
            bundle_root=bundles,
            curation_path=curation,
            output_dir=tmp_path / "export",
            dedupe_report_path=dedupe_path,
        )
    )

    split_by_id = {}
    for split in ("train", "val", "test"):
        for record in _manifest_records(tmp_path / "export" / f"manifest_{split}.jsonl"):
            split_by_id[record["sprite_id"]] = split
    assert split_by_id["sprite_0"] == split_by_id["sprite_1"]


def test_invalid_accepted_bundle_fails_export(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, ["sprite_a"])
    metadata_path = bundles / "sprite_a" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["palette_size"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid accepted bundles"):
        export_training_dataset(
            TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
        )


def test_palette_exceeding_max_palette_slots_fails_export(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    curation = tmp_path / "curation.jsonl"
    _write_bundle(bundles / "sprite_a", "sprite_a", palette_rows=4)
    append_curation_decision(curation, CurationDecision("sprite_a", "accepted"))

    with pytest.raises(ValueError, match="max_palette_slots"):
        export_training_dataset(
            TrainingExportConfig(
                bundle_root=bundles,
                curation_path=curation,
                output_dir=tmp_path / "export",
                max_palette_slots=2,
            )
        )


def test_bundle_id_collision_fails_export(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    curation = tmp_path / "curation.jsonl"
    _write_bundle(bundles / "first", "same_id")
    _write_bundle(bundles / "second", "same_id")
    append_curation_decision(curation, CurationDecision("same_id", "accepted"))

    with pytest.raises(ValueError, match="bundle ID collision"):
        export_training_dataset(
            TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
        )


def test_missing_reports_emit_warnings_but_do_not_fail(tmp_path) -> None:
    bundles, curation = _write_accepted_dataset(tmp_path, ["sprite_a"])

    result = export_training_dataset(
        TrainingExportConfig(bundle_root=bundles, curation_path=curation, output_dir=tmp_path / "export")
    )

    assert any("No dedupe" in warning for warning in result.warnings)
    assert any("No quality" in warning for warning in result.warnings)
    assert result.readiness_passed


def _write_accepted_dataset(tmp_path: Path, sprite_ids: list[str]) -> tuple[Path, Path]:
    bundles = tmp_path / "bundles"
    curation = tmp_path / "curation.jsonl"
    for sprite_id in sprite_ids:
        _write_bundle(bundles / sprite_id, sprite_id)
        append_curation_decision(curation, CurationDecision(sprite_id, "accepted", tags=("item_icon",)))
    return bundles, curation


def _write_bundle(
    directory: Path,
    sprite_id: str,
    *,
    include_role_map: bool = True,
    palette_rows: int = 2,
) -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = None
    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 1
    if include_role_map:
        role_map = np.zeros((32, 32), dtype=np.uint8)
        role_map[10:14, 10:14] = ROLE_MIDTONE
    visible = [[100 + index, 40, 80] for index in range(palette_rows - 1)]
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], *visible], dtype=np.uint8),
        index_map=index_map,
        role_map=role_map,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=palette_rows - 1),
    )
    save_bundle(bundle, directory)


def _manifest_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest_ids(path: Path) -> list[str]:
    return [record["sprite_id"] for record in _manifest_records(path)]
