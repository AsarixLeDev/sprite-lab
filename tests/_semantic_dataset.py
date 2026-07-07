"""Test helper: build a tiny *exported* semantic-v3 dataset on disk.

Produces the same layout as :mod:`spritelab.dataset_maker.exporter`
(manifest_{split}.jsonl + {split}.npz) so the training-manifest / eval-prompt
layers can be exercised without a full harvest+export run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.harvest.semantic_v3 import build_semantic_v3_record, semantic_v3_to_json


def _prediction(spec: dict[str, Any]) -> dict[str, Any]:
    object_name = spec["object_name"]
    category = spec.get("category", "item_icon")
    return {
        "sprite_id": spec["sprite_id"],
        "candidate_object_names": list(spec.get("candidates", [])),
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": list(spec.get("tags", [object_name, category])),
            "short_description": spec.get("short_description", f"A {object_name.replace('_', ' ')} icon."),
            "materials": list(spec.get("materials", [])),
            "mood": list(spec.get("mood", [])),
        },
        "visual_facts": {
            "dominant_colors": list(spec.get("dominant_colors", [])),
            "shape_hints": list(spec.get("shape_hints", [])),
        },
        "vlm_descriptor": {"object_name": "", "alternative_object_names": list(spec.get("vlm_alternatives", []))},
        "source_profile": {"name": "test_pack", "domain": spec.get("domain", "rpg_icons")},
        "bucket": spec.get("bucket", "auto_filename_trusted"),
        "label_quality": {"bucket": spec.get("bucket", "auto_filename_trusted")},
    }


def _manifest_record(spec: dict[str, Any]) -> dict[str, Any]:
    prediction = _prediction(spec)
    semantic = semantic_v3_to_json(build_semantic_v3_record(prediction))
    safe = prediction["safe_prefill"]
    return {
        "sprite_id": spec["sprite_id"],
        "split": spec["split"],
        "category": safe["category"],
        "category_id": spec.get("category_id", 1),
        "object_name": safe["object_name"],
        "tags": list(safe["tags"]),
        "short_description": safe["short_description"],
        "materials": list(safe["materials"]),
        "mood": list(safe["mood"]),
        "palette_size": spec.get("palette_size", 8),
        "has_role_map": True,
        "source_path": f"{spec['sprite_id']}.png",
        "source_name": "Test Pack",
        "license": "cc0",
        "author": "Tester",
        "notes": "",
        "label_v2": {"applied": True, "bucket": spec.get("bucket", "auto_filename_trusted"), "flags": []},
        "semantic_v3": semantic,
    }


def make_semantic_dataset(
    dataset_dir: Path,
    specs: list[dict[str, Any]],
    *,
    write_semantic: bool = True,
) -> Path:
    """Write manifest_{split}.jsonl and {split}.npz for ``specs``."""

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        record = _manifest_record(spec)
        if not write_semantic:
            record.pop("semantic_v3", None)
        by_split.setdefault(spec["split"], []).append(record)

    for split, records in by_split.items():
        records.sort(key=lambda r: r["sprite_id"])
        manifest_path = dataset_dir / f"manifest_{split}.jsonl"
        manifest_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        _write_npz(dataset_dir / f"{split}.npz", [record["sprite_id"] for record in records])

    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": "test_semantic", "max_palette_slots": 32}) + "\n", encoding="utf-8"
    )
    return dataset_dir


def _write_npz(path: Path, sprite_ids: list[str]) -> None:
    count = len(sprite_ids)
    np.savez_compressed(
        path,
        alpha=np.ones((count, 32, 32), dtype=np.uint8),
        index_map=np.ones((count, 32, 32), dtype=np.int16),
        role_map=np.zeros((count, 32, 32), dtype=np.uint8),
        palette=np.zeros((count, 33, 3), dtype=np.uint8),
        palette_mask=np.ones((count, 33), dtype=bool),
        category_id=np.ones((count,), dtype=np.int64),
        sprite_id=np.array(sprite_ids, dtype=np.str_) if sprite_ids else np.array([], dtype=np.str_),
    )


def default_specs() -> list[dict[str, Any]]:
    """A small spread of sprites across splits and families."""

    return [
        {"split": "train", "sprite_id": "t_gold_sword", "object_name": "golden_sword", "category": "weapon",
         "dominant_colors": ["gold", "yellow", "black"], "materials": ["metal"]},
        {"split": "train", "sprite_id": "t_red_potion", "object_name": "red_potion", "category": "item_icon",
         "dominant_colors": ["red", "black"]},
        {"split": "train", "sprite_id": "t_ruby_gem", "object_name": "ruby_gem", "category": "material",
         "dominant_colors": ["red", "black"], "candidates": ["ruby_gem", "gem"]},
        {"split": "train", "sprite_id": "t_wood_shield", "object_name": "wooden_shield", "category": "armor",
         "dominant_colors": ["brown"]},
        {"split": "val", "sprite_id": "v_blue_vial", "object_name": "blue_vial", "category": "item_icon",
         "dominant_colors": ["blue", "white"]},
        {"split": "test", "sprite_id": "x_green_gem", "object_name": "green_gem", "category": "material",
         "dominant_colors": ["green", "black"]},
    ]
