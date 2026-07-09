"""Core data classes for 32x32 palette-index sprite bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

SPRITE_WIDTH = 32
SPRITE_HEIGHT = 32
SPRITE_SIZE = (SPRITE_HEIGHT, SPRITE_WIDTH)
BUNDLE_SCHEMA_VERSION = "1.0"
CODEC_VERSION = "0.1.0"
INDEX_TRANSPARENT = 0
INDEX_MASK = 254
INDEX_PAD = 253
MAX_TRAINING_PALETTE_SLOTS = 252

Array = NDArray[np.generic]


@dataclass(slots=True)
class SpriteMetadata:
    """JSON-serializable metadata attached to a sprite bundle."""

    id: str
    width: int = SPRITE_WIDTH
    height: int = SPRITE_HEIGHT
    category: str | None = None
    subtype: str | None = None
    caption: str | None = None
    prompt: str | None = None
    source: str | None = None
    license: str | None = None
    palette_size: int | None = None
    quality_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION
    codec_version: str = CODEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as a JSON-serializable dictionary."""

        data = asdict(self)
        json.dumps(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpriteMetadata:
        """Build metadata from a dictionary loaded from JSON."""

        copied = dict(data)
        copied.setdefault("bundle_schema_version", BUNDLE_SCHEMA_VERSION)
        copied.setdefault("codec_version", CODEC_VERSION)
        return cls(**copied)


@dataclass(slots=True)
class SpriteBundle:
    """A single 32x32 palette-index sprite.

    Palette convention:
    - ``index_map == 0`` means the pixel is transparent.
    - ``palette[0]`` is a dummy transparent RGB slot, usually ``[0, 0, 0]``.
    - Opaque pixels must use palette slots ``1..palette.shape[0] - 1``.
    - Transparent pixels must use index ``0``.
    """

    alpha: Array
    palette: Array
    index_map: Array
    role_map: Array | None
    metadata: SpriteMetadata
