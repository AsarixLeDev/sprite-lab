from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from spritelab.data.ingest_clean_pngs import IngestOptions, find_png_files, ingest_clean_png_folder


def test_find_png_files_is_deterministically_sorted(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    _write_valid_sprite(input_dir / "b.png", (255, 0, 0, 255))
    _write_valid_sprite(input_dir / "nested" / "a.png", (0, 255, 0, 255))

    files = find_png_files(input_dir)

    assert [path.relative_to(input_dir).as_posix() for path in files] == ["b.png", "nested/a.png"]


def test_ingest_encodes_valid_files_and_rejects_invalid_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _write_valid_sprite(input_dir / "valid_1.png", (255, 0, 0, 255))
    _write_valid_sprite(input_dir / "valid_2.png", (0, 255, 0, 255))
    _write_wrong_size(input_dir / "wrong_size.png")
    _write_too_many_colors(input_dir / "too_many_colors.png")

    manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            category="item_icon",
            license="CC0",
        )
    )

    assert manifest.total_seen == 4
    assert len(manifest.records) == 2
    assert manifest.rejected_count == 2
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "rejected.json").exists()

    rejected = json.loads((output_dir / "rejected.json").read_text(encoding="utf-8"))
    reasons = " ".join(record["reason"] for record in rejected["rejected"])
    assert "expected image size" in reasons
    assert "above max_visible_colors=32" in reasons

    for record in manifest.records:
        bundle_dir = Path(record.bundle_dir)
        assert (bundle_dir / "bundle.npz").exists()
        assert (bundle_dir / "metadata.json").exists()
        assert (bundle_dir / "reconstructed.png").exists()
        assert (bundle_dir / "preview_8x.png").exists()
        assert (output_dir / "previews" / f"{record.id}_preview_8x.png").exists()
        assert record.palette_size > 0
        assert record.source_path.endswith(".png")
        assert len(record.sha256) == 64
        assert record.category == "item_icon"
        assert record.license == "CC0"


def test_ingest_create_split_is_deterministic(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    for index in range(6):
        _write_valid_sprite(input_dir / f"sprite_{index}.png", (index + 1, 0, 0, 255))

    options = IngestOptions(
        input_dir=input_dir,
        output_dir=tmp_path / "processed_a",
        create_split=True,
        split_seed=99,
    )
    first = ingest_clean_png_folder(options)
    second = ingest_clean_png_folder(
        IngestOptions(
            input_dir=input_dir,
            output_dir=tmp_path / "processed_b",
            create_split=True,
            split_seed=99,
        )
    )

    assert [record.split for record in first.records] == [record.split for record in second.records]
    assert all(record.split in {"train", "val", "test"} for record in first.records)


def test_skip_existing_does_not_rewrite_existing_bundle(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _write_valid_sprite(input_dir / "sprite.png", (255, 0, 0, 255))

    first = ingest_clean_png_folder(IngestOptions(input_dir=input_dir, output_dir=output_dir))
    bundle_path = Path(first.records[0].bundle_dir) / "bundle.npz"
    initial_mtime = bundle_path.stat().st_mtime_ns

    second = ingest_clean_png_folder(IngestOptions(input_dir=input_dir, output_dir=output_dir, skip_existing=True))

    assert len(second.records) == 1
    assert bundle_path.stat().st_mtime_ns == initial_mtime


def _write_valid_sprite(path: Path, color: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_wrong_size(path: Path) -> None:
    image = Image.new("RGBA", (31, 32), (0, 0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_too_many_colors(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for index in range(33):
        image.putpixel((index % 32, index // 32), (index, 255 - index, (index * 5) % 256, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
