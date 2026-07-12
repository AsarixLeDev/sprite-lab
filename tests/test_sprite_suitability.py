from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spritelab.harvest.cli import main as harvest_main
from spritelab.harvest.suitability import (
    SuitabilityInput,
    audit_inputs,
    audit_sprite,
    load_config,
    write_audit_output,
)


def _save(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8), "RGBA").save(path)
    return path


def _blank() -> np.ndarray:
    return np.zeros((32, 32, 4), dtype=np.uint8)


def _good(path: Path, color: tuple[int, int, int] = (70, 150, 220)) -> Path:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.polygon([(16, 5), (25, 13), (22, 25), (10, 25), (7, 13)], fill=(*color, 255))
    draw.rectangle((5, 14, 8, 16), fill=(*color, 255))
    draw.polygon([(16, 8), (21, 13), (19, 20), (13, 20), (10, 13)], fill=(180, 225, 250, 255))
    image.save(path)
    return path


def _audit(path: Path, profile: str = "single_object_32px"):
    return audit_sprite(SuitabilityInput(path.stem, path), load_config(profile))


def test_empty_file_and_fully_transparent_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    transparent = _save(tmp_path / "transparent.png", _blank())
    assert _audit(empty).reason_codes == ["FILE_EMPTY"]
    result = _audit(transparent)
    assert result.status == "reject"
    assert "FULLY_TRANSPARENT" in result.reason_codes


def test_baked_checkerboard_is_hard_rejected(tmp_path: Path) -> None:
    array = np.zeros((32, 32, 4), dtype=np.uint8)
    for y in range(32):
        for x in range(32):
            value = 180 if (x // 4 + y // 4) % 2 else 230
            array[y, x] = (value, value, value, 255)
    result = _audit(_save(tmp_path / "checker.png", array))
    assert result.status == "reject"
    assert "BAKED_CHECKERBOARD" in result.reason_codes


def test_opaque_background_is_hard_rejected(tmp_path: Path) -> None:
    array = np.full((32, 32, 4), (20, 30, 40, 255), dtype=np.uint8)
    array[9:23, 11:21] = (220, 80, 60, 255)
    result = _audit(_save(tmp_path / "opaque_bg.png", array))
    assert result.status == "reject"
    assert "OPAQUE_RECTANGULAR_BACKGROUND" in result.reason_codes


def test_partial_alpha_halo_quarantines(tmp_path: Path) -> None:
    array = _blank()
    array[9:23, 9:23] = (100, 180, 240, 255)
    array[8, 9:23] = array[23, 9:23] = (100, 180, 240, 96)
    array[9:23, 8] = array[9:23, 23] = (100, 180, 240, 96)
    result = _audit(_save(tmp_path / "halo.png", array))
    assert result.status == "quarantine"
    assert "EXCESSIVE_PARTIAL_ALPHA" in result.reason_codes
    assert result.metrics["alpha_halo_ratio"] > 0


def test_interpolated_edge_quarantines(tmp_path: Path) -> None:
    small = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ImageDraw.Draw(small).ellipse((1, 1, 6, 6), fill=(210, 80, 50, 255))
    path = tmp_path / "bilinear.png"
    small.resize((32, 32), Image.Resampling.BILINEAR).save(path)
    result = _audit(path)
    assert result.status != "accept"
    assert {"EXCESSIVE_PARTIAL_ALPHA", "ANTIALIASED_EDGE_EVIDENCE"} & set(result.reason_codes)


def test_compact_pixel_art_palette_accepts(tmp_path: Path) -> None:
    result = _audit(_good(tmp_path / "good.png"))
    assert result.status == "accept"
    assert result.metrics["unique_rgb_colors"] == 2


def test_excessive_smooth_gradient_is_rejected(tmp_path: Path) -> None:
    yy, xx = np.indices((32, 32))
    array = np.empty((32, 32, 4), dtype=np.uint8)
    array[..., 0] = xx * 8
    array[..., 1] = yy * 8
    array[..., 2] = (xx + yy) * 4
    array[..., 3] = 255
    result = _audit(_save(tmp_path / "gradient.png", array))
    assert result.status == "reject"
    assert {"PHOTOGRAPHIC_OR_PAINTED", "SEVERE_INTERPOLATION"} & set(result.reason_codes)


def test_detached_shadow_is_not_hard_rejected(tmp_path: Path) -> None:
    array = _blank()
    array[5:22, 10:22] = (80, 170, 220, 255)
    array[24:26, 11:21] = (30, 40, 50, 255)
    result = _audit(_save(tmp_path / "shadow.png", array))
    assert result.status != "reject"
    assert result.metrics["detached_shadow_likely"] is True


def test_multiple_unrelated_objects_and_strip(tmp_path: Path) -> None:
    multi = _blank()
    for y, x in ((3, 3), (3, 21), (21, 3), (21, 21)):
        multi[y : y + 7, x : x + 7] = (200, 100, 30, 255)
    result = _audit(_save(tmp_path / "multi.png", multi))
    assert result.status == "reject"
    assert "MULTIPLE_UNRELATED_OBJECTS" in result.reason_codes

    strip = np.zeros((16, 80, 4), dtype=np.uint8)
    for x in (2, 22, 42, 62):
        strip[3:13, x : x + 12] = (50, 180, 120, 255)
    result = _audit(_save(tmp_path / "strip.png", strip))
    assert result.status == "reject"
    assert "SPRITE_SHEET_OR_ANIMATION_STRIP" in result.reason_codes


def test_clipped_object_rejects_and_padding_quarantines(tmp_path: Path) -> None:
    clipped = _blank()
    clipped[:, :25] = (100, 100, 220, 255)
    result = _audit(_save(tmp_path / "clipped.png", clipped))
    assert result.status == "reject"
    assert "SEVERELY_CLIPPED_FOREGROUND" in result.reason_codes

    padded = _blank()
    padded[14:18, 14:18] = (240, 180, 50, 255)
    result = _audit(_save(tmp_path / "padded.png", padded))
    assert result.status == "quarantine"
    assert "EXCESSIVE_PADDING" in result.reason_codes


def test_exact_alpha_recolor_translation_and_flip_groups(tmp_path: Path) -> None:
    first = _good(tmp_path / "first.png", (50, 130, 220))
    exact = tmp_path / "exact.png"
    exact.write_bytes(first.read_bytes())
    recolor = _good(tmp_path / "recolor.png", (220, 80, 80))
    original = np.asarray(Image.open(first).convert("RGBA"))
    translated = _blank()
    translated[1:, 2:] = original[:-1, :-2]
    translated_path = _save(tmp_path / "translated.png", translated)
    flipped_path = _save(tmp_path / "flipped.png", np.fliplr(original))
    items = [SuitabilityInput(path.stem, path) for path in (first, exact, recolor, translated_path, flipped_path)]
    output = audit_inputs(items, load_config())
    kinds = {group.kind for group in output.duplicate_groups}
    assert "exact_rgba" in kinds
    assert "exact_alpha_mask" in kinds
    assert "recolor_geometry" in kinds
    assert "padded_or_translated" in kinds
    assert "trivial_flip" in kinds


def test_deterministic_outputs(tmp_path: Path) -> None:
    path = _good(tmp_path / "good.png")
    output = audit_inputs([SuitabilityInput("good", path)], load_config())
    first, second = tmp_path / "one", tmp_path / "two"
    write_audit_output(output, first)
    write_audit_output(output, second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}


def test_profile_specific_behavior(tmp_path: Path) -> None:
    array = _blank()
    array[1:31, 1:31] = (80, 140, 90, 255)
    path = _save(tmp_path / "tile.png", array)
    assert _audit(path, "single_object_32px").status == "quarantine"
    assert _audit(path, "environment_tile").status == "accept"


def test_report_only_cli_execution(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    image = _good(run / "sprite.png")
    (run / "imported.jsonl").write_text(
        json.dumps({"sprite_id": "sprite", "final_png_path": str(image), "candidate_id": "candidate"}) + "\n",
        encoding="utf-8",
    )
    (run / "rejected.jsonl").write_text("", encoding="utf-8")
    (run / "candidates.jsonl").write_text("", encoding="utf-8")
    out = tmp_path / "audit"
    harvest_main(["suitability-audit", "--run", str(run), "--out-dir", str(out), "--report-only"])
    result = json.loads((out / "suitability_results.jsonl").read_text(encoding="utf-8"))
    assert result["sprite_id"] == "sprite"
    assert result["status"] == "accept"
    assert (run / "sprite.png").read_bytes() == image.read_bytes()
