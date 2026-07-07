"""OOD compositional prompt builders for generator audits."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OOD_OBJECTS: tuple[str, ...] = (
    "sword",
    "axe",
    "bow",
    "hammer",
    "compass",
    "scissors",
    "potion",
    "bottle",
    "book",
    "gem",
    "coin",
    "mushroom",
)
DEFAULT_OOD_COLORS: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "black",
    "white",
    "gold",
)

OOD_OBJECT_CATEGORIES: dict[str, str] = {
    "sword": "weapon",
    "axe": "weapon",
    "bow": "weapon",
    "hammer": "weapon",
    "compass": "tool",
    "scissors": "tool",
    "potion": "item_icon",
    "bottle": "item_icon",
    "book": "item_icon",
    "gem": "material",
    "coin": "material",
    "mushroom": "plant",
}


@dataclass(frozen=True)
class OodCompositionalPromptConfig:
    out: Path
    objects: Sequence[str] = DEFAULT_OOD_OBJECTS
    colors: Sequence[str] = DEFAULT_OOD_COLORS
    max_prompts: int | None = None


def build_ood_compositional_prompts(config: OodCompositionalPromptConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    limit = None if config.max_prompts is None else max(0, int(config.max_prompts))
    for object_name in config.objects:
        object_token = _token(object_name)
        category = OOD_OBJECT_CATEGORIES.get(object_token, "item_icon")
        for color in config.colors:
            color_token = _token(color)
            index = len(rows)
            prompt_id = f"ood_{color_token}_{object_token}"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": f"{color_token} {object_token} 32x32 pixel art icon",
                    "target_sprite_id": prompt_id,
                    "category": category,
                    "object_name": object_token,
                    "base_object": object_token,
                    "colors": [color_token],
                    "eval_prompt_index": index,
                    "conditioning": {
                        "semantic_v3": {
                            "category": category,
                            "object_name": object_token,
                            "open_name": object_token,
                            "base_object": object_token,
                            "attributes": {
                                "colors": [color_token],
                                "materials": [],
                                "shapes": [],
                                "function": [],
                                "effects": [],
                                "state": [],
                                "style": ["pixel_art", "icon"],
                            },
                        }
                    },
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break

    path = Path(config.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    categories = Counter(str(row.get("category", "unknown")) for row in rows)
    return {
        "prompt_file": str(path),
        "prompt_count": len(rows),
        "objects": list(config.objects),
        "colors": list(config.colors),
        "category_counts": dict(sorted(categories.items())),
    }


def _token(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")
