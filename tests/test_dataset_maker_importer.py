from __future__ import annotations

from pathlib import Path

from PIL import Image

from spritelab.dataset_maker.importer import ImportOptions, import_png_as_dataset_item, import_png_directory


def test_imports_valid_32x32_png(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.item.status == "accepted"
    assert imported.bundle is not None
    assert imported.errors == ()


def test_rejects_non_png_files(tmp_path: Path) -> None:
    path = tmp_path / "not_png.txt"
    path.write_text("nope", encoding="utf-8")

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.item.status == "rejected"
    assert any("expected .png" in error for error in imported.errors)


def test_rejects_wrong_size_png_by_default(tmp_path: Path) -> None:
    path = tmp_path / "wrong.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.bundle is None
    assert any("expected image size" in error for error in imported.errors)


def test_optionally_resizes_wrong_size_png_with_nearest_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "wrong.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions(allow_nearest_resize=True))

    assert imported.bundle is not None
    assert imported.bundle.alpha.shape == (32, 32)
    assert any("resized to 32x32" in warning for warning in imported.warnings)


def test_rejects_soft_alpha(tmp_path: Path) -> None:
    path = tmp_path / "soft.png"
    image = _valid_image()
    image.putpixel((5, 5), (255, 0, 0, 127))
    image.save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.bundle is None
    assert any("soft alpha" in error for error in imported.errors)


def test_accepts_hard_alpha(tmp_path: Path) -> None:
    path = tmp_path / "hard.png"
    image = _valid_image()
    image.putpixel((0, 0), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 0, 0, 255))
    image.save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.bundle is not None
    assert imported.errors == ()


def test_over_color_png_is_quantized_when_enabled(tmp_path: Path) -> None:
    path = tmp_path / "over.png"
    _over_color_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions(max_palette_slots=8, quantize_overcolor=True))

    assert imported.bundle is not None
    assert imported.item.palette_size is not None
    assert imported.item.palette_size <= 8
    assert any("quantized over-color" in warning for warning in imported.warnings)


def test_over_color_png_is_rejected_when_quantization_disabled(tmp_path: Path) -> None:
    path = tmp_path / "over.png"
    _over_color_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions(max_palette_slots=8, quantize_overcolor=False))

    assert imported.bundle is None
    assert imported.item.status == "rejected"
    assert any("above max_visible_colors" in error for error in imported.errors)


def test_default_category_and_tags_are_applied(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(
        path,
        options=ImportOptions(),
        default_category="Item Icon",
        default_tags=("Copper", "Vial"),
    )

    assert imported.item.category == "item_icon"
    assert imported.item.tags == ("copper", "vial")


def test_generated_sprite_id_comes_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "Copper Vial 001.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.item.sprite_id == "copper_vial_001"


def test_preview_images_are_created(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions())

    assert imported.preview_image is not None
    assert imported.alpha_preview_image is not None
    assert imported.palette_strip_image is not None


def test_role_map_is_created_when_inference_enabled(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions(infer_role_map=True))

    assert imported.bundle is not None
    assert imported.bundle.role_map is not None
    assert imported.role_preview_image is not None


def test_missing_role_map_is_allowed_when_inference_disabled(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _valid_image().save(path)

    imported = import_png_as_dataset_item(path, options=ImportOptions(infer_role_map=False))

    assert imported.bundle is not None
    assert imported.bundle.role_map is None
    assert imported.role_preview_image is None


def test_directory_import_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _valid_image((0, 255, 0, 255)).save(_prepare(root / "b.png"))
    _valid_image((255, 0, 0, 255)).save(_prepare(root / "a.png"))
    _valid_image((0, 0, 255, 255)).save(_prepare(root / ".hidden.png"))

    imported = import_png_directory(root, options=ImportOptions())

    assert [sprite.item.sprite_id for sprite in imported] == ["a", "b"]


def _prepare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _valid_image(color: tuple[int, int, int, int] = (220, 70, 80, 255)) -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            image.putpixel((x, y), (20, 20, 30, 255) if x in (10, 21) or y in (10, 21) else color)
    return image


def _over_color_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 27):
        for x in range(5, 27):
            red = 32 + x * 6
            green = 24 + y * 5
            blue = 60 + ((x * 3 + y * 2) % 40)
            image.putpixel((x, y), (red % 256, green % 256, blue % 256, 255))
    return image
