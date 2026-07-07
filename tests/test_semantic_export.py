from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.roles import ROLE_MIDTONE
from spritelab.dataset_maker.exporter import DatasetMakerExportConfig, export_dataset_from_imported_sprites
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import DatasetMakerItem

_SEMANTIC_V3 = {
    "schema_version": "semantic_v3.0",
    "category": "armor",
    "object_name": "golden_chestplate",
    "base_object": "chestplate",
    "open_name": "golden chestplate",
    "attributes": {
        "colors": ["gold", "yellow"],
        "materials": ["metal", "gold"],
        "shapes": ["torso_shaped"],
        "effects": [],
        "state": [],
        "function": ["protection"],
        "mood": ["fantasy"],
        "style": ["32x32", "pixel_art", "rpg_icon"],
        "parts": [],
        "environment": [],
    },
    "aliases": ["chestplate"],
    "captions": ["golden chestplate", "gold chestplate made of metal"],
    "prompt_phrases": ["32x32 pixel art golden chestplate"],
    "negative_tags": ["photorealistic", "large_scene", "text", "watermark"],
    "source_evidence": {},
    "warnings": [],
}


def _imported(sprite_id: str, *, with_semantic: bool) -> ImportedSprite:
    item = DatasetMakerItem(
        sprite_id=sprite_id,
        source_path=Path(f"{sprite_id}.png"),
        status="accepted",
        category="armor",
        tags=("chestplate", "armor", "metal"),
        notes="A metal chestplate icon.",
        source_name="Source Pack",
        license="cc0",
        author="Artist",
        palette_size=1,
        has_role_map=True,
    )
    auto_metadata = {
        "label_v2_applied": True,
        "label_v2_prediction_file": "preds.jsonl",
        "label_v2_bucket": "auto_rpg_496_specialized",
        "label_v2_flags": ["auto_rpg_496_specialized"],
        "label_v2_safe_prefill": {
            "category": "armor",
            "object_name": "golden_chestplate",
            "tags": ["chestplate", "armor", "metal"],
            "short_description": "A metal chestplate icon.",
            "materials": ["metal"],
            "mood": [],
        },
    }
    if with_semantic:
        auto_metadata["semantic_v3"] = dict(_SEMANTIC_V3)
    return ImportedSprite(
        item=item,
        bundle=_bundle(sprite_id),
        preview_image=None,
        alpha_preview_image=None,
        role_preview_image=None,
        palette_strip_image=None,
        errors=(),
        warnings=(),
        auto_metadata=auto_metadata,
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


def _manifest_records(dataset_dir: Path) -> list[dict]:
    records: list[dict] = []
    for split in ("train", "val", "test"):
        path = dataset_dir / f"manifest_{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def test_export_manifest_includes_semantic_v3_when_present(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("armor_semantic", with_semantic=True)],
        DatasetMakerExportConfig(dataset_name="semantic_pack", output_root=tmp_path),
    )

    records = _manifest_records(tmp_path / "semantic_pack")
    assert len(records) == 1
    manifest = records[0]
    semantic = manifest["semantic_v3"]
    assert semantic["schema_version"] == "semantic_v3.0"
    assert semantic["base_object"] == "chestplate"
    assert semantic["open_name"] == "golden chestplate"
    assert semantic["captions"] == ["golden chestplate", "gold chestplate made of metal"]
    assert semantic["attributes"]["colors"] == ["gold", "yellow"]
    # label-v2 audit metadata is preserved alongside semantic metadata
    assert manifest["label_v2"]["applied"] is True
    assert manifest["label_v2"]["bucket"] == "auto_rpg_496_specialized"
    assert manifest["object_name"] == "golden_chestplate"
    assert manifest["category"] == "armor"


def test_export_manifest_omits_semantic_v3_when_absent(tmp_path: Path) -> None:
    export_dataset_from_imported_sprites(
        [_imported("armor_plain", with_semantic=False)],
        DatasetMakerExportConfig(dataset_name="plain_pack", output_root=tmp_path),
    )

    records = _manifest_records(tmp_path / "plain_pack")
    assert len(records) == 1
    assert "semantic_v3" not in records[0]
    assert records[0]["label_v2"]["applied"] is True
