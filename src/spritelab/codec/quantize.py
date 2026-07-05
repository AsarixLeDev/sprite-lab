"""OKLab palette quantization for over-color 32x32 sprites."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from spritelab.codec.alpha import extract_hard_alpha
from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_SIZE, SPRITE_WIDTH, SpriteBundle, SpriteMetadata
from spritelab.codec.canonical_palette import canonicalize_bundle_palette
from spritelab.codec.io import save_bundle
from spritelab.codec.oklab import oklab_array_to_rgb_u8, rgb_u8_array_to_oklab
from spritelab.codec.palette import DEFAULT_TRANSPARENT_RGB, visible_palette_size
from spritelab.codec.role_inference import apply_role_inference_to_bundle
from spritelab.codec.validate import assert_valid_bundle
from spritelab.utils.image import assert_exact_size, ensure_rgba


@dataclass(frozen=True)
class QuantizationOptions:
    target_visible_colors: int = 16
    max_iterations: int = 32
    seed: int = 12345
    init: Literal["frequency", "kmeans++", "deterministic"] = "frequency"
    preserve_exact_if_under_limit: bool = True
    canonicalize_palette: bool = True
    generate_role_map: bool = True
    alpha_threshold: int = 128


@dataclass(frozen=True)
class QuantizationResult:
    palette: np.ndarray
    index_map: np.ndarray
    alpha: np.ndarray
    original_visible_color_count: int
    quantized_visible_color_count: int
    mean_oklab_error: float
    max_oklab_error: float
    options: dict[str, Any]


def quantize_rgba_image_to_palette_indices(
    image: Image.Image,
    *,
    options: QuantizationOptions,
) -> QuantizationResult:
    """Quantize a 32x32 RGBA image into alpha, palette, and index map."""

    _validate_options(options)
    assert_exact_size(image)
    rgba = ensure_rgba(image)
    rgb_pixels = np.asarray(rgba, dtype=np.uint8)[:, :, :3]
    alpha = extract_hard_alpha(rgba, threshold=options.alpha_threshold)
    opaque_mask = alpha == 1
    if not bool(np.any(opaque_mask)):
        raise ValueError("Cannot quantize an empty sprite with no opaque pixels.")

    opaque_rgb = rgb_pixels[opaque_mask]
    unique_rgb, inverse, counts = _unique_rgb_with_counts(opaque_rgb)
    original_count = int(unique_rgb.shape[0])

    if original_count <= options.target_visible_colors and options.preserve_exact_if_under_limit:
        visible_rgb = unique_rgb.astype(np.uint8)
        palette = _palette_with_dummy(visible_rgb)
        index_map = _index_map_from_exact_inverse(alpha, opaque_mask, inverse)
        return QuantizationResult(
            palette=palette,
            index_map=index_map,
            alpha=alpha,
            original_visible_color_count=original_count,
            quantized_visible_color_count=visible_palette_size(palette),
            mean_oklab_error=0.0,
            max_oklab_error=0.0,
            options=asdict(options),
        )

    k = min(options.target_visible_colors, original_count)
    unique_oklab = rgb_u8_array_to_oklab(unique_rgb)
    centers_oklab, _labels = fit_oklab_kmeans(
        unique_oklab,
        k=k,
        sample_weights=counts.astype(np.float64),
        max_iterations=options.max_iterations,
        seed=options.seed,
        init=options.init,
    )

    center_rgb = oklab_array_to_rgb_u8(centers_oklab)
    visible_rgb = _dedupe_and_sort_rgb(center_rgb)
    visible_oklab = rgb_u8_array_to_oklab(visible_rgb)
    unique_labels = _nearest_labels(unique_oklab, visible_oklab)
    pixel_labels = unique_labels[inverse]

    palette = _palette_with_dummy(visible_rgb)
    index_map = np.zeros(SPRITE_SIZE, dtype=np.uint8)
    index_map[opaque_mask] = (pixel_labels + 1).astype(np.uint8)
    errors = np.linalg.norm(unique_oklab - visible_oklab[unique_labels], axis=1)
    mean_error = float(np.average(errors, weights=counts))
    max_error = float(np.max(errors))

    return QuantizationResult(
        palette=palette,
        index_map=index_map,
        alpha=alpha,
        original_visible_color_count=original_count,
        quantized_visible_color_count=visible_palette_size(palette),
        mean_oklab_error=mean_error,
        max_oklab_error=max_error,
        options=asdict(options),
    )


def encode_rgba_image_to_quantized_bundle(
    image: Image.Image,
    metadata: SpriteMetadata,
    *,
    options: QuantizationOptions,
) -> SpriteBundle:
    """Encode a 32x32 RGBA image into a quantized SpriteBundle."""

    result = quantize_rgba_image_to_palette_indices(image, options=options)
    prepared_metadata = _metadata_with_quantization(metadata, result)
    bundle = SpriteBundle(
        alpha=result.alpha,
        palette=result.palette,
        index_map=result.index_map,
        role_map=None,
        metadata=prepared_metadata,
    )
    assert_valid_bundle(bundle)

    if options.generate_role_map:
        bundle = apply_role_inference_to_bundle(bundle)

    if options.canonicalize_palette:
        bundle = canonicalize_bundle_palette(bundle).bundle
        bundle = _copy_bundle_with_quantization_metadata(bundle, result)
        assert_valid_bundle(bundle)

    return bundle


def encode_png_to_quantized_bundle(
    image_path: str | Path,
    metadata: SpriteMetadata | None = None,
    *,
    options: QuantizationOptions,
) -> SpriteBundle:
    """Load a PNG and encode it into a quantized SpriteBundle."""

    path = Path(image_path)
    if metadata is None:
        metadata = SpriteMetadata(id=path.stem, width=SPRITE_WIDTH, height=SPRITE_HEIGHT, source=str(path))

    with Image.open(path) as image:
        rgba = ensure_rgba(image).copy()

    return encode_rgba_image_to_quantized_bundle(rgba, metadata, options=options)


def fit_oklab_kmeans(
    colors_oklab: np.ndarray,
    *,
    k: int,
    sample_weights: np.ndarray | None = None,
    max_iterations: int = 32,
    seed: int = 12345,
    init: str = "frequency",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a small deterministic weighted k-means model in OKLab space."""

    colors = np.asarray(colors_oklab, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors_oklab must have shape (N, 3).")
    if colors.shape[0] == 0:
        raise ValueError("colors_oklab must contain at least one color.")
    if k < 1 or k > colors.shape[0]:
        raise ValueError("k must be in 1..N.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    weights = _sample_weights(sample_weights, colors.shape[0])
    centers = _initial_centers(colors, weights, k=k, seed=seed, init=init)
    labels = np.full(colors.shape[0], -1, dtype=np.int32)

    for _iteration in range(max_iterations):
        new_labels = _nearest_labels(colors, centers)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        centers = _recompute_centers(colors, weights, labels, centers)

    labels = _nearest_labels(colors, centers)
    return centers, labels


def _unique_rgb_with_counts(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_rgb, inverse, counts = np.unique(rgb, axis=0, return_inverse=True, return_counts=True)
    return unique_rgb.astype(np.uint8), inverse.astype(np.int32), counts.astype(np.int64)


def _palette_with_dummy(visible_rgb: np.ndarray) -> np.ndarray:
    dummy = np.array([DEFAULT_TRANSPARENT_RGB], dtype=np.uint8)
    return np.vstack([dummy, visible_rgb.astype(np.uint8)])


def _index_map_from_exact_inverse(
    alpha: np.ndarray,
    opaque_mask: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    index_map = np.zeros(SPRITE_SIZE, dtype=np.uint8)
    index_map[opaque_mask] = (inverse + 1).astype(np.uint8)
    return index_map


def _dedupe_and_sort_rgb(rgb: np.ndarray) -> np.ndarray:
    seen = {tuple(int(channel) for channel in row) for row in np.asarray(rgb, dtype=np.uint8)}
    rows = sorted(seen)
    return np.array(rows, dtype=np.uint8)


def _sample_weights(sample_weights: np.ndarray | None, count: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(count, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError("sample_weights must have shape (N,).")
    if np.any(weights < 0):
        raise ValueError("sample_weights must be non-negative.")
    if float(np.sum(weights)) <= 0:
        raise ValueError("sample_weights must have positive total weight.")
    return weights


def _initial_centers(
    colors: np.ndarray,
    weights: np.ndarray,
    *,
    k: int,
    seed: int,
    init: str,
) -> np.ndarray:
    if init in {"frequency", "deterministic"}:
        indices = _frequency_init_indices(colors, weights, k)
    elif init == "kmeans++":
        indices = _kmeans_plus_plus_indices(colors, weights, k, seed)
    else:
        raise ValueError("init must be 'frequency', 'deterministic', or 'kmeans++'.")
    return colors[indices].copy()


def _frequency_init_indices(colors: np.ndarray, weights: np.ndarray, k: int) -> list[int]:
    first = min(range(colors.shape[0]), key=lambda index: (-weights[index], index))
    chosen = [first]
    while len(chosen) < k:
        distances = _squared_distances_to_nearest(colors, colors[chosen])
        scores = distances * np.sqrt(weights)
        for index in chosen:
            scores[index] = -1.0
        next_index = int(np.argmax(scores))
        chosen.append(next_index)
    return chosen


def _kmeans_plus_plus_indices(colors: np.ndarray, weights: np.ndarray, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    probabilities = weights / np.sum(weights)
    chosen = [int(rng.choice(colors.shape[0], p=probabilities))]
    while len(chosen) < k:
        distances = _squared_distances_to_nearest(colors, colors[chosen])
        scores = distances * weights
        for index in chosen:
            scores[index] = 0.0
        total = float(np.sum(scores))
        if total <= 0.0:
            remaining = [index for index in range(colors.shape[0]) if index not in chosen]
            chosen.append(remaining[0])
        else:
            chosen.append(int(rng.choice(colors.shape[0], p=scores / total)))
    return chosen


def _nearest_labels(colors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((colors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def _squared_distances_to_nearest(colors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((colors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.min(distances, axis=1)


def _recompute_centers(
    colors: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    previous_centers: np.ndarray,
) -> np.ndarray:
    centers = previous_centers.copy()
    k = previous_centers.shape[0]
    for cluster in range(k):
        mask = labels == cluster
        if bool(np.any(mask)):
            cluster_weights = weights[mask]
            centers[cluster] = np.average(colors[mask], axis=0, weights=cluster_weights)
        else:
            centers[cluster] = colors[_farthest_weighted_index(colors, weights, centers)]
    return centers


def _farthest_weighted_index(colors: np.ndarray, weights: np.ndarray, centers: np.ndarray) -> int:
    distances = _squared_distances_to_nearest(colors, centers)
    scores = distances * np.sqrt(weights)
    return int(np.argmax(scores))

def _metadata_with_quantization(metadata: SpriteMetadata, result: QuantizationResult) -> SpriteMetadata:
    metadata_data = copy.deepcopy(metadata.to_dict())
    metadata_data["width"] = SPRITE_WIDTH
    metadata_data["height"] = SPRITE_HEIGHT
    metadata_data["palette_size"] = result.quantized_visible_color_count
    extra = dict(metadata_data.get("extra") or {})
    extra.update(_quantization_extra(result))
    metadata_data["extra"] = extra
    return SpriteMetadata.from_dict(metadata_data)


def _copy_bundle_with_quantization_metadata(bundle: SpriteBundle, result: QuantizationResult) -> SpriteBundle:
    metadata = _metadata_with_quantization(bundle.metadata, result)
    return SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=np.asarray(bundle.palette).copy(),
        index_map=np.asarray(bundle.index_map).copy(),
        role_map=None if bundle.role_map is None else np.asarray(bundle.role_map).copy(),
        metadata=metadata,
    )


def _quantization_extra(result: QuantizationResult) -> dict[str, Any]:
    quantized = result.original_visible_color_count > result.quantized_visible_color_count
    return {
        "quantized": quantized,
        "original_visible_color_count": result.original_visible_color_count,
        "quantized_visible_color_count": result.quantized_visible_color_count,
        "mean_oklab_error": result.mean_oklab_error,
        "max_oklab_error": result.max_oklab_error,
        "quantization": {
            "target_visible_colors": result.options["target_visible_colors"],
            "max_iterations": result.options["max_iterations"],
            "seed": result.options["seed"],
            "init": result.options["init"],
            "preserve_exact_if_under_limit": result.options["preserve_exact_if_under_limit"],
        },
    }


def _validate_options(options: QuantizationOptions) -> None:
    if options.target_visible_colors < 1:
        raise ValueError("target_visible_colors must be at least 1.")
    if options.target_visible_colors > 255:
        raise ValueError("target_visible_colors must be at most 255 for uint8 index maps.")
    if options.max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if options.alpha_threshold < 0 or options.alpha_threshold > 255:
        raise ValueError("alpha_threshold must be in 0..255.")


def _parse_args() -> tuple[Path, Path, SpriteMetadata, QuantizationOptions]:
    parser = argparse.ArgumentParser(description="Quantize one 32x32 PNG into a SpriteBundle.")
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir")
    parser.add_argument("--id", required=True, dest="sprite_id")
    parser.add_argument("--category")
    parser.add_argument("--license")
    parser.add_argument("--target-visible-colors", type=int, default=16)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--no-canonicalize", action="store_false", dest="canonicalize_palette")
    parser.add_argument("--no-role-map", action="store_false", dest="generate_role_map")
    args = parser.parse_args()
    metadata = SpriteMetadata(
        id=args.sprite_id,
        category=args.category,
        source=str(args.input_path),
        license=args.license,
    )
    options = QuantizationOptions(
        target_visible_colors=args.target_visible_colors,
        max_iterations=args.max_iterations,
        seed=args.seed,
        canonicalize_palette=args.canonicalize_palette,
        generate_role_map=args.generate_role_map,
        alpha_threshold=args.alpha_threshold,
    )
    return args.input_path, args.output_dir, metadata, options


def main() -> None:
    input_path, output_dir, metadata, options = _parse_args()
    bundle = encode_png_to_quantized_bundle(input_path, metadata=metadata, options=options)
    save_bundle(bundle, output_dir)
    extra = bundle.metadata.extra
    print(f"Original visible colors: {extra.get('original_visible_color_count')}")
    print(f"Quantized visible colors: {extra.get('quantized_visible_color_count')}")
    print(f"Mean OKLab error: {extra.get('mean_oklab_error')}")
    print(f"Max OKLab error: {extra.get('max_oklab_error')}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
