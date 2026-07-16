from __future__ import annotations

import json

import numpy as np

from spritelab.codec.bundle import BUNDLE_SCHEMA_VERSION, CODEC_VERSION, SpriteBundle, SpriteMetadata
from spritelab.codec.io import load_bundle, save_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.roles import ColorRole


def make_valid_bundle() -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = np.zeros((32, 32), dtype=np.uint8)

    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 1
    role_map[10:14, 10:14] = int(ColorRole.MIDTONE)

    palette = np.array(
        [
            [0, 0, 0],
            [200, 80, 120],
        ],
        dtype=np.uint8,
    )

    metadata = SpriteMetadata(
        id="roundtrip",
        category="test",
        caption="Roundtrip test sprite.",
        palette_size=1,
        extra={"created_by": "pytest"},
    )

    return SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=role_map,
        metadata=metadata,
    )


def test_bundle_roundtrip(tmp_path) -> None:
    bundle = make_valid_bundle()

    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)

    np.testing.assert_array_equal(loaded.alpha, bundle.alpha)
    np.testing.assert_array_equal(loaded.palette, bundle.palette)
    np.testing.assert_array_equal(loaded.index_map, bundle.index_map)
    assert loaded.role_map is not None
    np.testing.assert_array_equal(loaded.role_map, bundle.role_map)
    assert loaded.metadata.to_dict() == bundle.metadata.to_dict()

    image = reconstruct_rgba(loaded)
    assert image.mode == "RGBA"
    assert image.size == (32, 32)


def test_saved_metadata_json_includes_schema_and_codec_versions(tmp_path) -> None:
    bundle = make_valid_bundle()

    save_bundle(bundle, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert metadata["codec_version"] == CODEC_VERSION


def test_load_save_roundtrip_preserves_schema_and_codec_versions(tmp_path) -> None:
    bundle = make_valid_bundle()
    bundle.metadata.bundle_schema_version = BUNDLE_SCHEMA_VERSION
    bundle.metadata.codec_version = "0.1.0-test"

    save_bundle(bundle, tmp_path / "original")
    loaded = load_bundle(tmp_path / "original")
    save_bundle(loaded, tmp_path / "resaved")

    metadata = json.loads((tmp_path / "resaved" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert metadata["codec_version"] == "0.1.0-test"


def test_old_metadata_without_version_fields_loads_and_resaves_with_defaults(tmp_path) -> None:
    bundle = make_valid_bundle()
    save_bundle(bundle, tmp_path / "old")

    metadata_path = tmp_path / "old" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("bundle_schema_version")
    metadata.pop("codec_version")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    loaded = load_bundle(tmp_path / "old")
    assert loaded.metadata.bundle_schema_version == BUNDLE_SCHEMA_VERSION
    assert loaded.metadata.codec_version == CODEC_VERSION

    save_bundle(loaded, tmp_path / "upgraded")
    upgraded = json.loads((tmp_path / "upgraded" / "metadata.json").read_text(encoding="utf-8"))
    assert upgraded["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert upgraded["codec_version"] == CODEC_VERSION
