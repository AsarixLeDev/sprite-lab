# sprite-lab

`sprite-lab` is the foundation for a native 32x32 palette-index pixel-art generator.
This first milestone does not train a model. It only creates the clean
representation needed for the future generator.

## v1 Sprite Generator: Quickstart

Sprite Lab has a validated v1 release path (Phase 1 EMA checkpoint, CFG 3.0, 30
steps, k16 deterministic palette projection). Full details, including where
the validated metrics came from, live in [`docs/v1_default.md`](docs/v1_default.md).

Build the deterministic v1 demo gallery (never trains a model):

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train build-v1-gallery `
  --out experiments\v1_gallery `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

Or launch the local v1 gallery GUI (requires `pip install gradio`, or the `ui`
extra: `pip install -e ".[ui]"`) to pick an output directory and preview
contact sheets interactively:

```powershell
& $py -m spritelab train v1-gallery-gui --out experiments\v1_gallery_gui
```

See [`docs/v1_default.md`](docs/v1_default.md) for sampling custom prompts
with the same official settings, output layout, and why palette-swap
augmentation is not part of v1.

## SpriteBundle

A `SpriteBundle` stores one sprite as structured data:

- `alpha`: a 32x32 binary transparency mask with values `0` or `1`
- `palette`: a `K x 3` RGB palette array
- `index_map`: a 32x32 integer map into the palette
- `role_map`: an optional 32x32 semantic color-role map
- `metadata`: JSON-serializable descriptive metadata

Palette convention:

- `index_map == 0` means transparent.
- `palette[0]` is a dummy RGB value, usually `[0, 0, 0]`.
- Opaque pixels must use palette slots `1..K-1`.
- Transparent pixels must use index `0`.
- `metadata.palette_size`, when used, should describe the number of visible
  colors and therefore exclude the dummy transparent slot.

This palette-index representation keeps sprites compact and explicit. It makes
future generator work easier because transparency, color choices, and semantic
roles are separate instead of being hidden inside raw RGBA pixels.

## Palette Canonicalization

Stable palette slots matter because future index-map models should not have to
learn that slot 1 sometimes means outline and sometimes means highlight. The
v1 canonicalizer keeps slot 0 fixed as transparency, computes deterministic
per-slot statistics, then sorts visible colors with simple heuristics based on
edge contact, luminance, saturation, frequency, and position.

Canonicalization remaps the `index_map` whenever palette rows move. For valid
bundles, the reconstructed RGBA image should remain pixel-identical before and
after canonicalization.

## Encoding Clean 32x32 PNGs

The strict encoder converts already-clean 32x32 pixel-art PNGs into
`SpriteBundle` objects. It does not resize, quantize, split spritesheets, remove
backgrounds, or repair messy images.

Encoder behavior:

- Input images must be exactly 32x32.
- Images are read as RGBA.
- Alpha is hardened with a threshold: `alpha >= threshold` becomes opaque.
- Transparent pixels do not contribute palette colors.
- Visible colors are extracted exactly.
- Images with too many visible colors are rejected.
- Palette canonicalization can run after encoding.
- Reconstructed transparent pixels become `(0, 0, 0, 0)`.

```python
from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.encode import encode_png_to_bundle
from spritelab.codec.io import save_bundle

bundle = encode_png_to_bundle(
    "my_sprite.png",
    metadata=SpriteMetadata(id="my_sprite", category="item_icon"),
    max_visible_colors=32,
)

save_bundle(bundle, "outputs/my_sprite_bundle")
```

## OKLab Quantization For Over-Color Sprites

Strict encoding remains the default. OKLab quantization is opt-in for 32x32
sprites that have too many near-identical or noisy visible colors. Quantization
only clusters opaque pixels; transparent pixels stay transparent, and palette
slot 0 remains the dummy transparent slot.

OKLab is used because plain RGB distance is a poor fit for perceptual palette
reduction. This helps with anti-aliased edges, small gradients, and export
artifacts, but it is lossy and may merge subtle highlights or damage deliberate
gradients. It does not resize images, split spritesheets, or remove backgrounds.
Run the preview grid and quality report after quantized ingestion.

```python
from pathlib import Path
from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.codec.quantize import QuantizationOptions, encode_png_to_quantized_bundle

bundle = encode_png_to_quantized_bundle(
    "over_color_sprite.png",
    metadata=SpriteMetadata(id="over_color_sprite", category="item_icon"),
    options=QuantizationOptions(target_visible_colors=16),
)

save_bundle(bundle, Path("outputs/over_color_sprite_bundle"))
```

The batch ingester can try strict encoding first, then quantize only images that
fail because they exceed the visible-color limit:

```bash
python -m spritelab.data.ingest_clean_pngs \
  --input data/raw/items \
  --output data/processed/items_quantized_v0 \
  --category item_icon \
  --license CC0 \
  --quantize-over-color \
  --target-visible-colors 16
```

For one-off debugging, the quantizer also has a direct CLI:

```bash
python -m spritelab.codec.quantize \
  --input path/to/sprite.png \
  --output outputs/quantized_sprite_bundle \
  --id sprite_id \
  --target-visible-colors 16
```

## Role-Map Inference

`role_map` is a semantic debug and training aid. It maps each pixel to stable
roles such as outline, shadow, midtone, light, highlight, accent, emissive,
texture detail, transparent, or unknown. Role inference is deterministic
heuristic v2, not ML, and is not guaranteed to match a human artist perfectly.

The heuristic uses palette colors, OKLab luminance/chroma, color frequency,
edge contact, local contrast, same-color neighborhoods, and simple spatial
hints. It is useful for future model training and dataset inspection, especially
when paired with role-map preview images.

```python
from spritelab.codec.io import load_bundle, save_bundle
from spritelab.codec.role_inference import (
    apply_role_inference_to_bundle,
    role_map_to_preview_image,
)

bundle = load_bundle("data/processed/items_v0/bundles/my_sprite")
bundle = apply_role_inference_to_bundle(bundle)

save_bundle(bundle, "outputs/my_sprite_with_roles")
role_map_to_preview_image(bundle.role_map).save("outputs/my_sprite_roles.png")
```

## Batch Ingestion Of Clean PNG Folders

The batch ingester converts a local folder of already-clean exact 32x32 PNG
sprites into a structured dataset. It uses the strict encoder, writes one
`SpriteBundle` directory per accepted image, and writes both `manifest.json` and
`rejected.json`.

The ingester rejects wrong-sized images and over-color images. It does not
quantize, resize, split spritesheets, remove backgrounds, or repair messy input.

```bash
python -m spritelab.data.ingest_clean_pngs \
  --input data/raw/items \
  --output data/processed/items_v0 \
  --category item_icon \
  --license CC0
```

```python
from pathlib import Path
from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder

manifest = ingest_clean_png_folder(
    IngestOptions(
        input_dir=Path("data/raw/items"),
        output_dir=Path("data/processed/items_v0"),
        category="item_icon",
        license="CC0",
    )
)

print(len(manifest.records))
```

## Previewing A Dataset Grid

The preview grid tool creates contact sheets for fast visual auditing before
training. It reads a dataset directory, a direct `manifest.json` path, or a
`bundles/` directory. It uses reconstructed 32x32 sprites, upscales with
nearest-neighbor only, and can filter by category, split, ID substring, and
palette size.

```bash
python -m spritelab.data.preview_grid \
  --dataset data/processed/items_v0 \
  --output outputs/items_grid.png \
  --columns 10 \
  --scale 8 \
  --category item_icon
```

```python
from pathlib import Path
from spritelab.data.preview_grid import PreviewGridOptions, create_preview_grid

create_preview_grid(
    PreviewGridOptions(
        dataset_path=Path("data/processed/items_v0"),
        output_path=Path("outputs/items_grid.png"),
        columns=10,
        scale=8,
    )
)
```

## Dataset Quality Reports

The quality report tool analyzes existing `SpriteBundle` datasets before
training. It computes objective diagnostics, flags suspicious sprites, and
writes `quality_report.json`, `quality_report.md`, and optional `flagged/*.txt`
files. It does not delete, reject, or mutate bundles. Issue codes are heuristic
warnings, not absolute proof that a sprite is bad.

```bash
python -m spritelab.data.quality_report \
  --dataset data/processed/items_v0 \
  --output data/processed/items_v0/quality
```

```python
from pathlib import Path
from spritelab.data.quality_report import QualityReportOptions, create_quality_report

report = create_quality_report(
    QualityReportOptions(
        dataset_path=Path("data/processed/items_v0"),
        output_dir=Path("data/processed/items_v0/quality"),
    )
)

print(report.summary.issue_counts)
```

Issue codes include:

- `EMPTY_SPRITE`: no opaque pixels
- `MOSTLY_EMPTY` / `MOSTLY_FULL`: unusual alpha coverage
- `TINY_BBOX` / `HUGE_BBOX`: suspicious opaque bounding box size
- `OFF_CENTER`: center of mass is far from the sprite center
- `MANY_COMPONENTS`, `FRAGMENTED`, `MANY_SINGLE_PIXELS`: disconnected alpha problems
- `HAS_ALPHA_HOLES`: transparent holes inside opaque regions
- `TOUCHES_EDGE`: opaque pixels touch the canvas edge
- `LOW_CONTRAST`: low used-palette luminance range
- `TINY_PALETTE` / `LARGE_PALETTE`: unusual palette size

## Dataset Dedupe Reports

The dedupe report tool analyzes existing `SpriteBundle` datasets for exact
duplicates, near-duplicates, and train/val/test split leakage. It does not
delete, reject, mutate, or rewrite datasets. Near-duplicate detection is a
heuristic warning and can produce false positives.

```bash
python -m spritelab.data.dedupe_report \
  --dataset data/processed/items_v0 \
  --output data/processed/items_v0/dedupe
```

```python
from pathlib import Path
from spritelab.data.dedupe_report import DedupeReportOptions, create_dedupe_report

report = create_dedupe_report(
    DedupeReportOptions(
        dataset_path=Path("data/processed/items_v0"),
        output_dir=Path("data/processed/items_v0/dedupe"),
        near_duplicate=True,
        near_duplicate_threshold=8,
    )
)

print(report.summary.cross_split_exact_groups)
```

Hash meanings:

- `source_sha256`: SHA256 of the source PNG when available from the manifest
  or local source path.
- `decoded_rgba_sha256`: SHA256 of reconstructed 32x32 RGBA bytes. This is the
  main exact visual duplicate signal and still matches if palette order differs.
- `bundle_content_sha256`: SHA256 of structural bundle arrays, excluding
  metadata. This checks identical internal representation.
- `average_hash` and `difference_hash`: lightweight perceptual fingerprints for
  simple near-duplicate grouping.

## Install

```bash
python -m pip install -e .
```

## Run Tests

```bash
pytest
```

## Run Demo

```bash
python examples/create_demo_bundle.py
```

```bash
python examples/canonicalize_demo_bundle.py
```

```bash
python examples/encode_png_demo.py
```

```bash
python examples/ingest_clean_folder_demo.py
```

```bash
python examples/preview_grid_demo.py
```

```bash
python examples/quality_report_demo.py
```

```bash
python examples/dedupe_report_demo.py
```

```bash
python examples/quantize_png_demo.py
```

```bash
python examples/role_map_demo.py
```

The first demo writes a bundle to `outputs/demo_bundle/` with:

- `bundle.npz`
- `metadata.json`
- `reconstructed.png`
- `preview_8x.png`

The canonicalizer demo writes scrambled and canonicalized bundles under
`outputs/canonicalizer_demo/` plus `report.json`.

The encoder demo writes a source PNG and encoded bundle under
`outputs/encode_png_demo/`.

The ingestion demo writes raw demo PNGs and a processed dataset under
`outputs/ingest_clean_folder_demo/`.

The preview grid demo writes an ingested demo dataset and grid image under
`outputs/preview_grid_demo/`.

The quality report demo writes diagnostics under `outputs/quality_report_demo/`.

The dedupe report demo writes duplicate diagnostics under
`outputs/dedupe_report_demo/`.

The quantization demo writes an over-color source PNG and quantized bundle under
`outputs/quantize_png_demo/`.

The role-map demo writes a role-inferred bundle and debug role preview under
`outputs/role_map_demo/`.
