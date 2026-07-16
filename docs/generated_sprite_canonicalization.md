# Generated Sprite Canonicalization

This note documents the bridge from generator output back into strict
`sprite-lab` sprite artifacts.

## Why RGBA First

Generator v0 predicts universal continuous RGBA because the training export uses
per-sprite palettes. Palette index `3` in one sprite does not mean the same RGB
as palette index `3` in another sprite. Caption-only index prediction would
therefore need palette prediction or palette conditioning first.

RGBA is a model-friendly intermediate. It is not the final artifact contract.

## Why Canonicalize

Strict `sprite-lab` sprites are 32x32 indexed-palette assets:

- hard alpha only;
- transparent pixels use index `0`;
- visible pixels use palette rows `1..K`;
- no more than 32 visible colors;
- deterministic, auditable metadata.

Generated RGBA can have soft alpha and hundreds of continuous RGB values. The
canonicalization pass turns that model output into something QA can validate and
humans can inspect.

## Steps

`canonicalize_generated_rgba` performs:

1. Accept `[4, 32, 32]` or `[32, 32, 4]` RGBA.
2. Normalize `0..255` or `0..1` values into float `0..1`.
3. Clamp channels into range.
4. Threshold alpha with `alpha_threshold`.
5. Zero RGB for transparent pixels.
6. Quantize only visible RGB pixels.
7. Reserve palette row `0` as transparent.
8. Emit `index_map`, RGBA palette, palette mask, counts, and warnings.

The dataclass stores palette rows as `max_colors + 1`: one transparent dummy row
plus up to `max_colors` visible rows. This matches the existing exported dataset
contract.

## Quantization Policy

The implementation reuses the existing deterministic OKLab quantizer in
`spritelab.codec.quantize`.

Policy:

- preserve exact colors when visible unique colors are already within the cap;
- otherwise run deterministic weighted OKLab k-means;
- no dithering by default;
- keep transparent pixels outside quantization;
- warn, rather than fail, for fully transparent generated samples.

`--dither` is accepted for CLI compatibility but v0 still records a warning and
uses no dithering.

## Metadata

`generated_manifest.jsonl` records one sample per line, including:

- `sample_id`;
- `prompt_id`, `prompt`, `category`, and any preserved prompt metadata;
- `checkpoint`;
- `seed` and per-sample `noise_seed`;
- `width`, `height`;
- `alpha_threshold`, `max_colors`;
- `visible_color_count`, `alpha_opaque_count`;
- relative artifact paths;
- canonicalization warnings.

The output directory also contains:

- `generation_report.json`;
- `generation_report.md`;
- `generation_contact_sheet.png`;
- raw/hard/indexed PNG folders.

## QA Checks

`python -m spritelab train generated-qa --generated <dir>` checks:

- manifest and report existence;
- duplicate sample ids;
- prompt and checkpoint provenance;
- referenced PNG existence and readability;
- all PNG dimensions are 32x32;
- hard RGBA alpha is only `0` or `255`;
- indexed output has no more than `max_colors` visible colors;
- manifest color and opaque-pixel counts match artifacts;
- contact sheet exists when the report references it.

Fully transparent generated samples are warnings by default and can be escalated
with `--error-on-fully-transparent`.

## Limitations

This pass validates output form, not semantic quality. It does not infer
role maps, re-score prompts, or check whether the generated shape matches the
caption. Quantized colors can be crude, especially for bad generator outputs.

## Future Work

- palette-aware generation;
- role-map prediction;
- indexed-token generation;
- better no-dither/dither policy experiments;
- RGBA-to-indexed round-trip metrics;
- automatic semantic re-checking of generated samples.
