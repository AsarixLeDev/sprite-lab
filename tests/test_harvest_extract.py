"""Tests for spritelab.harvest.extract."""

from __future__ import annotations

from _harvest_testdata import make_source, make_sprite_png

from spritelab.harvest.extract import discover_png_candidates, filter_candidate_basic


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
