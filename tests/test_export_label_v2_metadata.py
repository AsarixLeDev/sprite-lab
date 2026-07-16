from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.roles import ROLE_MIDTONE
from spritelab.dataset_maker.exporter import DatasetMakerExportConfig, export_dataset_from_imported_sprites
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import DatasetMakerItem


def test_export_manifest_includes_label_v2_object_and_audit_metadata(tmp_path: Path) -> None:
    item = DatasetMakerItem(
        sprite_id="armor_01",
        source_path=Path("armor_01.png"),
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
    imported = ImportedSprite(
        item=item,
        bundle=_bundle("armor_01"),
        preview_image=None,
        alpha_preview_image=None,
        role_preview_image=None,
        palette_strip_image=None,
        errors=(),
        warnings=(),
        auto_metadata={
            "label_v2_applied": True,
            "label_v2_prediction_file": "label_v2_suggestions.jsonl",
            "label_v2_bucket": "auto_rpg_496_specialized",
            "label_v2_label_confidence_tier": "T1",
            "label_v2_flags": ["auto_rpg_496_specialized"],
            "label_v2_candidate_object_names": ["chestplate", "armor"],
            "label_v2_safe_prefill": {
                "category": "armor",
                "object_name": "chestplate",
                "tags": ["chestplate", "armor", "metal"],
                "short_description": "A metal chestplate icon.",
                "materials": ["metal"],
                "mood": ["defensive"],
            },
            "label_v2_vlm_descriptor": {
                "object_name": "armor",
                "alternative_object_names": ["chestplate", "breastplate"],
                "source_consistency": "consistent",
            },
        },
    )

    export_dataset_from_imported_sprites(
        [imported],
        DatasetMakerExportConfig(dataset_name="label_v2_pack", output_root=tmp_path),
    )

    manifest = json.loads(
        (tmp_path / "label_v2_pack" / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert manifest["category"] == "armor"
    assert manifest["object_name"] == "chestplate"
    assert manifest["label_confidence_tier"] == "T1"
    assert manifest["tags"] == ["chestplate", "armor", "metal"]
    assert manifest["short_description"] == "A metal chestplate icon."
    assert manifest["materials"] == ["metal"]
    assert manifest["mood"] == ["defensive"]
    assert manifest["source_name"] == "Source Pack"
    assert manifest["license"] == "cc0"
    assert manifest["author"] == "Artist"
    assert manifest["palette_size"] == 1
    assert manifest["has_role_map"] is True
    assert manifest["label_v2"]["applied"] is True
    assert manifest["label_v2"]["prediction_file"] == "label_v2_suggestions.jsonl"
    assert manifest["label_v2"]["bucket"] == "auto_rpg_496_specialized"
    assert manifest["label_v2"]["label_confidence_tier"] == "T1"
    assert manifest["label_v2"]["safe_prefill"]["object_name"] == "chestplate"
    assert manifest["label_v2"]["vlm_object_name"] == "armor"
    assert manifest["label_v2"]["vlm_alternative_object_names"] == ["chestplate", "breastplate"]
    assert manifest["label_v2"]["vlm_source_consistency"] == "consistent"


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
