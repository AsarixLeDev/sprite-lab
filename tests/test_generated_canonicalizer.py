from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritelab.training.generated_canonicalizer import (
    canonicalize_generated_rgba,
    reconstruct_indexed_rgba,
    write_generated_sprite_artifacts,
)


def _rgba_hwc(color: tuple[float, float, float], alpha: float = 1.0) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    rgba[..., :3] = np.array(color, dtype=np.float32)
    rgba[..., 3] = alpha
    return rgba


def test_canonicalizer_accepts_chw_float_rgba() -> None:
    rgba = np.moveaxis(_rgba_hwc((1.0, 0.0, 0.0)), -1, 0)
    sprite = canonicalize_generated_rgba(rgba)
    assert sprite.rgba_raw.shape == (32, 32, 4)
    assert sprite.rgba_hard.shape == (32, 32, 4)
    assert sprite.index_map.shape == (32, 32)
    assert set(np.unique(sprite.rgba_hard[..., 3])) <= {1.0}


def test_canonicalizer_accepts_hwc_float_rgba() -> None:
    sprite = canonicalize_generated_rgba(_rgba_hwc((0.0, 1.0, 0.0)))
    assert sprite.visible_color_count == 1
    assert sprite.palette_mask.sum() == 2


def test_canonicalizer_accepts_u8_rgba_and_normalizes() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 255
    sprite = canonicalize_generated_rgba(rgba)
    assert float(sprite.rgba_raw[..., 0].max()) == 1.0
    assert float(sprite.rgba_raw[..., 3].max()) == 1.0


def test_canonicalizer_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="rgba must have shape"):
        canonicalize_generated_rgba(np.zeros((16, 16, 4), dtype=np.float32))


def test_hard_alpha_contains_only_zero_or_one_and_transparent_index_zero() -> None:
    rgba = _rgba_hwc((0.5, 0.25, 0.0), alpha=1.0)
    rgba[:8, :8, 3] = 0.25
    sprite = canonicalize_generated_rgba(rgba, alpha_threshold=0.5)
    assert set(float(value) for value in np.unique(sprite.rgba_hard[..., 3])) <= {0.0, 1.0}
    assert np.all(sprite.index_map[:8, :8] == 0)
    assert np.allclose(sprite.rgba_hard[:8, :8, :3], 0.0)


def test_unique_colors_under_limit_remain_exact() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rgba[:10, :, :3] = colors[0]
    rgba[10:20, :, :3] = colors[1]
    rgba[20:, :, :3] = colors[2]
    rgba[..., 3] = 1.0
    sprite = canonicalize_generated_rgba(rgba, max_colors=32)
    reconstructed = reconstruct_indexed_rgba(index_map=sprite.index_map, palette=sprite.palette)
    seen = {tuple(np.round(color, 4)) for color in reconstructed[..., :3].reshape(-1, 3)}
    assert seen == {tuple(color) for color in colors}
    assert sprite.visible_color_count == 3


def test_overcolor_image_quantizes_to_limit() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    for y in range(32):
        for x in range(32):
            rgba[y, x, :3] = [x / 31.0, y / 31.0, ((x + y) % 32) / 31.0]
            rgba[y, x, 3] = 1.0
    sprite = canonicalize_generated_rgba(rgba, max_colors=8)
    assert sprite.visible_color_count <= 8
    assert int(np.max(sprite.index_map)) <= 8


def test_fully_transparent_image_warns() -> None:
    sprite = canonicalize_generated_rgba(np.zeros((32, 32, 4), dtype=np.float32))
    assert sprite.alpha_opaque_count == 0
    assert sprite.visible_color_count == 0
    assert any("fully transparent" in warning for warning in sprite.warnings)


def test_canonicalizer_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    rgba = rng.random((32, 32, 4), dtype=np.float32)
    first = canonicalize_generated_rgba(rgba, max_colors=16)
    second = canonicalize_generated_rgba(rgba, max_colors=16)
    assert np.array_equal(first.index_map, second.index_map)
    assert np.array_equal(first.palette, second.palette)


def test_reconstructed_indexed_rgba_is_32x32() -> None:
    sprite = canonicalize_generated_rgba(_rgba_hwc((0.2, 0.3, 0.4)))
    reconstructed = reconstruct_indexed_rgba(index_map=sprite.index_map, palette=sprite.palette)
    assert reconstructed.shape == (32, 32, 4)


def test_write_generated_sprite_artifacts_writes_pngs_and_manifest_paths(tmp_path) -> None:
    sprite = canonicalize_generated_rgba(_rgba_hwc((0.8, 0.1, 0.2)))
    record = write_generated_sprite_artifacts(
        sprite,
        tmp_path,
        "sample_000001",
        {"prompt_id": "p1", "prompt": "red icon", "checkpoint": "ckpt.pt", "seed": 1},
    )
    assert (tmp_path / "raw_rgba" / "sample_000001.png").is_file()
    assert (tmp_path / "hard_rgba" / "sample_000001.png").is_file()
    assert (tmp_path / "indexed_png" / "sample_000001.png").is_file()
    assert record["paths"]["raw_rgba"] == "raw_rgba/sample_000001.png"
    assert record["prompt_id"] == "p1"
    assert Image.open(tmp_path / record["paths"]["indexed_png"]).size == (32, 32)
