"""Read and write sprite bundle directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.preview import save_preview
from spritelab.codec.reconstruct import save_reconstructed_png
from spritelab.codec.validate import assert_valid_bundle


def save_bundle(
    bundle: SpriteBundle,
    directory: str | Path,
    write_reconstructed: bool = True,
    write_preview: bool = True,
) -> None:
    """Save a bundle directory containing arrays, metadata, and optional PNGs."""

    assert_valid_bundle(bundle)

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "alpha": bundle.alpha,
        "palette": bundle.palette,
        "index_map": bundle.index_map,
    }
    if bundle.role_map is not None:
        arrays["role_map"] = bundle.role_map

    np.savez_compressed(output_dir / "bundle.npz", **arrays)

    metadata_json = json.dumps(bundle.metadata.to_dict(), indent=2, sort_keys=True)
    (output_dir / "metadata.json").write_text(metadata_json + "\n", encoding="utf-8")

    if write_reconstructed:
        save_reconstructed_png(bundle, output_dir / "reconstructed.png")

    if write_preview:
        save_preview(bundle, output_dir / "preview_8x.png", scale=8)


def load_bundle(directory: str | Path) -> SpriteBundle:
    """Load and validate a sprite bundle directory."""

    input_dir = Path(directory)
    metadata_data = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = SpriteMetadata.from_dict(metadata_data)

    with np.load(input_dir / "bundle.npz") as data:
        role_map = data["role_map"] if "role_map" in data.files else None
        bundle = SpriteBundle(
            alpha=data["alpha"],
            palette=data["palette"],
            index_map=data["index_map"],
            role_map=role_map,
            metadata=metadata,
        )

    assert_valid_bundle(bundle)
    return bundle
