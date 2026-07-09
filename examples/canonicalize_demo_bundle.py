"""Demonstrate palette canonicalization on a deliberately scrambled bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for path in (SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_demo_bundle import build_demo_bundle

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.canonical_palette import canonicalize_bundle_palette, remap_index_map
from spritelab.codec.io import load_bundle, save_bundle


def load_or_create_demo_bundle() -> SpriteBundle:
    """Load the milestone demo bundle, creating it first when needed."""

    demo_dir = ROOT / "outputs" / "demo_bundle"
    if (demo_dir / "bundle.npz").exists() and (demo_dir / "metadata.json").exists():
        return load_bundle(demo_dir)

    bundle = build_demo_bundle()
    save_bundle(bundle, demo_dir)
    return bundle


def make_scrambled_bundle(bundle: SpriteBundle) -> SpriteBundle:
    """Move visible palette slots into a deliberately non-canonical order."""

    old_order = [0, 4, 1, 3, 5, 2]
    old_to_scrambled = {old_slot: new_slot for new_slot, old_slot in enumerate(old_order)}

    metadata_data = bundle.metadata.to_dict()
    metadata_data["id"] = f"{bundle.metadata.id}_scrambled"
    metadata_data["extra"] = dict(metadata_data.get("extra") or {})
    metadata_data["extra"]["palette_scrambled_for_demo"] = True
    metadata_data["extra"]["palette_scramble_old_to_new"] = {str(old): new for old, new in old_to_scrambled.items()}

    return SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=np.asarray(bundle.palette[old_order]).copy(),
        index_map=remap_index_map(bundle.index_map, old_to_scrambled),
        role_map=None if bundle.role_map is None else np.asarray(bundle.role_map).copy(),
        metadata=SpriteMetadata.from_dict(metadata_data),
    )


def main() -> None:
    demo_bundle = load_or_create_demo_bundle()
    scrambled_bundle = make_scrambled_bundle(demo_bundle)

    output_root = ROOT / "outputs" / "canonicalizer_demo"
    scrambled_dir = output_root / "scrambled"
    canonical_dir = output_root / "canonical"

    save_bundle(scrambled_bundle, scrambled_dir)

    result = canonicalize_bundle_palette(scrambled_bundle)
    save_bundle(result.bundle, canonical_dir)

    report = {
        "old_to_new": {str(old): new for old, new in result.old_to_new.items()},
        "new_to_old": {str(new): old for new, old in result.new_to_old.items()},
        "warnings": result.warnings,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
