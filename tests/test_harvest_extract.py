"""Tests for spritelab.harvest.extract."""

from __future__ import annotations

import os
import struct
import zlib

import pytest

from _harvest_testdata import make_source, make_sprite_png
from spritelab.harvest.extract import UnsafeSourceTreeError, discover_png_candidates, filter_candidate_basic


def _make_tree(root):
    make_sprite_png(root / "b.png")
    make_sprite_png(root / "sub" / "a.png")
    make_sprite_png(root / ".hidden" / "c.png")
    (root / "notes.txt").write_text("hi", encoding="utf-8")
    (root / "image.jpg").write_bytes(b"\xff\xd8\xff")


def test_discovers_recursively(tmp_path):
    _make_tree(tmp_path)
    candidates = discover_png_candidates(tmp_path, make_source())
    paths = [c.relative_path for c in candidates]
    assert paths == ["b.png", "sub/a.png"]


def test_ignores_hidden_by_default(tmp_path):
    _make_tree(tmp_path)
    default = discover_png_candidates(tmp_path, make_source())
    assert all(".hidden" not in c.relative_path for c in default)
    with_hidden = discover_png_candidates(tmp_path, make_source(), include_hidden=True)
    assert any(".hidden" in c.relative_path for c in with_hidden)


def test_deterministic_ordering(tmp_path):
    _make_tree(tmp_path)
    first = [c.relative_path for c in discover_png_candidates(tmp_path, make_source())]
    second = [c.relative_path for c in discover_png_candidates(tmp_path, make_source())]
    assert first == second == sorted(first)


def test_candidate_ids_stable(tmp_path):
    _make_tree(tmp_path)
    first = discover_png_candidates(tmp_path, make_source())
    second = discover_png_candidates(tmp_path, make_source())
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert all(c.candidate_id.startswith("test_source__") for c in first)


def test_dimensions_and_mode_captured(tmp_path):
    make_sprite_png(tmp_path / "s.png", size=64)
    candidate = discover_png_candidates(tmp_path, make_source())[0]
    assert (candidate.width, candidate.height) == (64, 64)
    assert candidate.mode == "RGBA"
    assert len(candidate.image_sha256) == 64


def test_non_png_ignored(tmp_path):
    _make_tree(tmp_path)
    candidates = discover_png_candidates(tmp_path, make_source())
    assert all(c.relative_path.endswith(".png") for c in candidates)


def test_basic_filter_size_limits(tmp_path):
    make_sprite_png(tmp_path / "tiny.png", size=4)
    candidate = discover_png_candidates(tmp_path, make_source())[0]
    filtered = filter_candidate_basic(candidate)
    assert filtered.status == "rejected"
    assert any("too small" in reason for reason in filtered.rejection_reasons)


def test_rejects_png_pixel_bomb_before_decoding_payload(tmp_path):
    def png_chunk(kind, payload):
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", 5000, 5000, 8, 6, 0, 0, 0)
    (tmp_path / "bomb.png").write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IEND", b""))

    candidate = discover_png_candidates(tmp_path, make_source())[0]

    assert candidate.status == "rejected"
    assert "safe decode limit" in candidate.rejection_reasons[0]


def test_rejects_symlinked_candidate_and_preserves_outside_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = make_sprite_png(tmp_path / "outside.png")
    original = outside.read_bytes()
    try:
        os.symlink(outside, root / "linked.png")
    except OSError:
        pytest.skip("symbolic links are unavailable in this test session")

    with pytest.raises(UnsafeSourceTreeError, match="link or reparse"):
        discover_png_candidates(root, make_source())

    assert outside.read_bytes() == original


def test_rejects_hardlinked_candidate_and_preserves_outside_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = make_sprite_png(tmp_path / "outside.png")
    original = outside.read_bytes()
    try:
        os.link(outside, root / "linked.png")
    except OSError:
        pytest.skip("hard links are unavailable in this test session")

    with pytest.raises(UnsafeSourceTreeError, match="hard-linked"):
        discover_png_candidates(root, make_source())

    assert outside.read_bytes() == original


def test_rejects_symlinked_source_root(tmp_path):
    root = tmp_path / "source"
    make_sprite_png(root / "sprite.png")
    alias = tmp_path / "alias"
    try:
        os.symlink(root, alias, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this test session")

    with pytest.raises(UnsafeSourceTreeError, match="candidate root"):
        discover_png_candidates(alias, make_source())
