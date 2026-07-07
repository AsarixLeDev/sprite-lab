from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.training.generated_canonicalizer import (
    build_generation_contact_sheet,
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.generated_qa import qa_generated_sprites


def _opaque_rgba(color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    rgba[..., :3] = np.array(color, dtype=np.float32)
    rgba[..., 3] = 1.0
    return rgba


def _generated_dir(tmp_path: Path, *, transparent: bool = False) -> tuple[Path, dict]:
    out = tmp_path / "generated"
    out.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake")
    rgba = np.zeros((32, 32, 4), dtype=np.float32) if transparent else _opaque_rgba()
    sprite = canonicalize_generated_rgba(rgba, max_colors=32)
    record = write_generated_sprite_artifacts(
        sprite,
        out,
        "sample_000001",
        {
            "prompt_id": "p1",
            "prompt": "red icon",
            "category": "seen_object",
            "checkpoint": str(checkpoint),
            "seed": 1,
            "noise_seed": 2,
            "max_colors": 32,
            "alpha_threshold": 0.5,
        },
    )
    contact = build_generation_contact_sheet(out, [record], out / "generation_contact_sheet.png")
    write_generation_reports(
        out_dir=out,
        records=[record],
        config={"checkpoint": str(checkpoint)},
        contact_sheet=None if contact is None else contact.name,
    )
    return out, record


def test_valid_generated_folder_passes(tmp_path: Path) -> None:
    out, _record = _generated_dir(tmp_path)
    result = qa_generated_sprites(out)
    assert result.ok
    assert result.sample_count == 1
    assert (out / "generated_qa_report.json").is_file()
    assert (out / "generated_qa_report.md").is_file()


def test_generated_qa_missing_png_is_error(tmp_path: Path) -> None:
    out, record = _generated_dir(tmp_path)
    (out / record["paths"]["indexed_png"]).unlink()
    result = qa_generated_sprites(out)
    assert not result.ok
    assert any("missing PNG" in error for error in result.errors)


def test_generated_qa_bad_dimensions_are_error(tmp_path: Path) -> None:
    out, record = _generated_dir(tmp_path)
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(out / record["paths"]["hard_rgba"])
    result = qa_generated_sprites(out)
    assert not result.ok
    assert any("expected 32x32" in error for error in result.errors)


def test_generated_qa_soft_alpha_in_hard_image_is_error(tmp_path: Path) -> None:
    out, record = _generated_dir(tmp_path)
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[..., :3] = [255, 0, 0]
    arr[..., 3] = 128
    Image.fromarray(arr, mode="RGBA").save(out / record["paths"]["hard_rgba"])
    result = qa_generated_sprites(out)
    assert not result.ok
    assert any("non-hard values" in error for error in result.errors)


def test_generated_qa_too_many_visible_colors_is_error(tmp_path: Path) -> None:
    out, record = _generated_dir(tmp_path)
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    for index in range(33):
        y = index // 8
        x = (index % 8) * 4
        arr[y, x : x + 4, :3] = [index, 255 - index, (index * 7) % 256]
        arr[y, x : x + 4, 3] = 255
    Image.fromarray(arr, mode="RGBA").save(out / record["paths"]["indexed_png"])
    result = qa_generated_sprites(out)
    assert not result.ok
    assert any("above max_colors" in error for error in result.errors)


def test_generated_qa_fully_transparent_warns_by_default_and_can_error(tmp_path: Path) -> None:
    out, _record = _generated_dir(tmp_path, transparent=True)
    warning_result = qa_generated_sprites(out)
    assert warning_result.ok
    assert any("fully transparent" in warning for warning in warning_result.warnings)
    error_result = qa_generated_sprites(out, error_on_fully_transparent=True)
    assert not error_result.ok
    assert any("fully transparent" in error for error in error_result.errors)


def test_generated_qa_duplicate_sample_ids_are_error(tmp_path: Path) -> None:
    out, _record = _generated_dir(tmp_path)
    manifest = out / "generated_manifest.jsonl"
    line = manifest.read_text(encoding="utf-8").splitlines()[0]
    manifest.write_text(line + "\n" + line + "\n", encoding="utf-8")
    report = json.loads((out / "generation_report.json").read_text(encoding="utf-8"))
    report["sample_count"] = 2
    (out / "generation_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    result = qa_generated_sprites(out)
    assert not result.ok
    assert any("duplicate sample_id" in error for error in result.errors)
