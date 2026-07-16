from __future__ import annotations

from pathlib import Path

from PIL import Image

from spritelab.data.ids import make_sprite_id, sha256_file
from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder


def test_simple_filename_becomes_safe_id() -> None:
    assert make_sprite_id("Red Mushroom.png") == "red_mushroom"


def test_unsafe_characters_are_replaced_and_lowercase() -> None:
    assert make_sprite_id("Weird@Name!.png") == "weird_name"


def test_nested_path_with_root_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "items" / "Red Mushroom.png"

    first = make_sprite_id(path, root=root)
    second = make_sprite_id(path, root=root)

    assert first == "items_red_mushroom"
    assert first == second


def test_ingestion_disambiguates_same_stem_in_different_folders(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "b").mkdir(parents=True)
    _write_sprite(input_dir / "a" / "same.png", (255, 0, 0, 255))
    _write_sprite(input_dir / "b" / "same.png", (0, 255, 0, 255))

    manifest = ingest_clean_png_folder(IngestOptions(input_dir=input_dir, output_dir=tmp_path / "processed"))

    ids = [record.id for record in manifest.records]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert ids == sorted(ids)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "file.png"
    path.write_bytes(b"sprite")

    assert sha256_file(path) == sha256_file(path)


def _write_sprite(path: Path, color: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
