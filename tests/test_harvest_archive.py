"""Tests for spritelab.harvest.archive and download helpers."""

from __future__ import annotations

import zipfile

import pytest

from _harvest_testdata import make_zip_of_pngs

from spritelab.harvest.archive import extract_archive, iter_archive_pngs
from spritelab.harvest.download import compute_sha256


def test_extracts_simple_zip(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png", "sub/b.png"])
    out = extract_archive(zip_path, tmp_path / "out")
    assert (out / "a.png").exists()
    assert (out / "sub" / "b.png").exists()


def test_prevents_zip_slip(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "bad")
        archive.writestr("/abs.txt", "bad")
        archive.writestr("ok.txt", "good")
    out = extract_archive(zip_path, tmp_path / "out")
    assert (out / "ok.txt").exists()
    assert not (tmp_path / "evil.txt").exists()
    extracted = [p.name for p in out.rglob("*") if p.is_file()]
    assert extracted == ["ok.txt"]


def test_iterates_png_names(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["z.png", "a.png"])
    with zipfile.ZipFile(zip_path, "a") as archive:
        archive.writestr("readme.txt", "hi")
    names = iter_archive_pngs(zip_path)
    assert names == ["a.png", "z.png"]


def test_computes_sha256(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"hello")
    assert compute_sha256(path) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_refuses_overwrite(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png"])
    out = extract_archive(zip_path, tmp_path / "out")
    with pytest.raises(FileExistsError):
        extract_archive(zip_path, out)
    extract_archive(zip_path, out, overwrite=True)
