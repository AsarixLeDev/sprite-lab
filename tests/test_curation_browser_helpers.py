from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.curation.browser import (
    load_browser_sprites,
    make_alpha_preview_image,
    make_browser_preview_image,
    make_palette_strip_image,
    make_role_preview_image,
)
from spritelab.curation.manifest import CurationDecision, append_curation_decision


def test_load_browser_sprites_discovers_saved_bundles(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "sprite_b", "sprite_b")
    _write_bundle(bundles / "sprite_a", "sprite_a")

    sprites = load_browser_sprites(bundles, tmp_path / "missing_curation.jsonl")

    assert {sprite.sprite_id for sprite in sprites} == {"sprite_a", "sprite_b"}
    assert all(sprite.path.exists() for sprite in sprites)


def test_load_browser_sprites_attaches_latest_curation_status(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "sprite_a", "sprite_a")
    curation_path = tmp_path / "curation.jsonl"
    append_curation_decision(curation_path, CurationDecision("sprite_a", "accepted"))
    append_curation_decision(
        curation_path,
        CurationDecision("sprite_a", "needs_fix", tags=("Needs Cleanup",), reasons=("bad_roles",), notes="Fix roles."),
    )

    sprites = load_browser_sprites(bundles, curation_path)

    assert len(sprites) == 1
    assert sprites[0].status == "needs_fix"
    assert sprites[0].tags == ("needs_cleanup",)
    assert sprites[0].reasons == ("bad_roles",)
    assert sprites[0].notes == "Fix roles."


def test_load_browser_sprites_handles_missing_curation_file(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "sprite_a", "sprite_a")

    sprites = load_browser_sprites(bundles, tmp_path / "missing.jsonl")

    assert sprites[0].status is None
    assert sprites[0].tags == ()


def test_preview_image_helper_returns_pil_image(tmp_path) -> None:
    bundle_dir = tmp_path / "bundles" / "sprite_a"
    _write_bundle(bundle_dir, "sprite_a")

    image = make_browser_preview_image(bundle_dir, scale=2)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGBA"
    assert image.size == (64, 64)


def test_alpha_preview_helper_returns_pil_image(tmp_path) -> None:
    bundle_dir = tmp_path / "bundles" / "sprite_a"
    _write_bundle(bundle_dir, "sprite_a")

    image = make_alpha_preview_image(bundle_dir, scale=3)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGBA"
    assert image.size == (96, 96)


def test_palette_strip_helper_returns_pil_image(tmp_path) -> None:
    bundle_dir = tmp_path / "bundles" / "sprite_a"
    _write_bundle(bundle_dir, "sprite_a")

    image = make_palette_strip_image(bundle_dir, swatch_size=10)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGBA"
    assert image.size == (20, 10)


def test_role_preview_helper_returns_none_for_bundle_without_role_map(tmp_path) -> None:
    bundle_dir = tmp_path / "bundles" / "sprite_a"
    _write_bundle(bundle_dir, "sprite_a", include_role_map=False)

    assert make_role_preview_image(bundle_dir) is None


def test_load_browser_sprites_sorting_is_deterministic(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "c", "c")
    _write_bundle(bundles / "a", "a")
    _write_bundle(bundles / "b", "b")

    first = load_browser_sprites(bundles, tmp_path / "missing.jsonl")
    second = load_browser_sprites(bundles, tmp_path / "missing.jsonl")

    assert [sprite.sprite_id for sprite in first] == [sprite.sprite_id for sprite in second]


def _write_bundle(directory, sprite_id: str, *, include_role_map: bool = False) -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = None
    alpha[8:12, 8:12] = 1
    index_map[8:12, 8:12] = 1
    if include_role_map:
        role_map = np.zeros((32, 32), dtype=np.uint8)
        role_map[8:12, 8:12] = 4

    palette = np.array([[0, 0, 0], [180, 70, 120]], dtype=np.uint8)
    bundle = SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=role_map,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=1),
    )
    save_bundle(bundle, directory)
