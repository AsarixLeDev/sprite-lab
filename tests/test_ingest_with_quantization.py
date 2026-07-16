from __future__ import annotations

from pathlib import Path

from PIL import Image

from spritelab.codec.io import load_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder
from spritelab.data.manifest import load_rejected_report


def test_ingest_without_quantization_rejects_over_color(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_inputs(raw_dir)
    output_dir = tmp_path / "processed"

    manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=output_dir,
            category="item_icon",
            max_visible_colors=16,
        )
    )

    assert [record.id for record in manifest.records] == ["clean_valid"]
    assert manifest.rejected_count == 2
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "rejected.json").exists()
    rejected = load_rejected_report(output_dir / "rejected.json")
    assert any("above max_visible_colors" in record.reason for record in rejected)


def test_ingest_with_quantization_accepts_over_color(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_inputs(raw_dir)
    output_dir = tmp_path / "processed"

    manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=output_dir,
            category="item_icon",
            max_visible_colors=16,
            quantize_over_color=True,
            target_visible_colors=8,
        )
    )

    assert [record.id for record in manifest.records] == ["clean_valid", "over_color_valid"]
    assert manifest.rejected_count == 1
    assert manifest.options["quantized_count"] == 1

    clean = load_bundle(output_dir / "bundles" / "clean_valid")
    quantized = load_bundle(output_dir / "bundles" / "over_color_valid")
    assert clean.metadata.extra["quantized"] is False
    assert quantized.metadata.extra["quantized"] is True
    assert quantized.metadata.extra["quantized_visible_color_count"] <= 8
    assert quantized.palette.shape[0] <= 9


def test_quantized_ingestion_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_inputs(raw_dir)

    first_manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=tmp_path / "first",
            max_visible_colors=16,
            quantize_over_color=True,
            target_visible_colors=8,
            quantization_seed=99,
        )
    )
    second_manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=tmp_path / "second",
            max_visible_colors=16,
            quantize_over_color=True,
            target_visible_colors=8,
            quantization_seed=99,
        )
    )

    first_bundle = load_bundle(Path(first_manifest.records[1].bundle_dir))
    second_bundle = load_bundle(Path(second_manifest.records[1].bundle_dir))

    assert reconstruct_rgba(first_bundle).tobytes() == reconstruct_rgba(second_bundle).tobytes()


def _write_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _clean_image().save(raw_dir / "clean_valid.png")
    _over_color_image().save(raw_dir / "over_color_valid.png")
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(raw_dir / "wrong_size.png")


def _clean_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            image.putpixel((x, y), (30, 30, 40, 255) if x in (10, 21) or y in (10, 21) else (180, 60, 80, 255))
    return image


def _over_color_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 27):
        for x in range(5, 27):
            red = 40 + x * 7
            green = 35 + y * 5
            blue = 70 + ((x + y * 2) % 45)
            image.putpixel((x, y), (red % 256, green % 256, blue % 256, 255))
    return image
