from __future__ import annotations

import os
from pathlib import Path

import pytest

from spritelab.utils.safe_fs import (
    UnsafeFilesystemOperation,
    atomic_write_bytes,
    remove_confined_tree,
    require_confined_path,
)


def test_confined_path_rejects_root_and_lexical_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(UnsafeFilesystemOperation, match="root itself"):
        require_confined_path(root, root)
    with pytest.raises(UnsafeFilesystemOperation, match="escapes"):
        require_confined_path(root / ".." / "outside", root)


def test_confined_tree_removal_preserves_outside_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "owned"
    outside = tmp_path / "outside.txt"
    target.mkdir(parents=True)
    (target / "generated.txt").write_text("generated", encoding="utf-8")
    outside.write_text("preserve", encoding="utf-8")

    remove_confined_tree(target, root)

    assert not target.exists()
    assert outside.read_text(encoding="utf-8") == "preserve"


def test_confined_tree_removal_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    link = root / "linked"
    root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this test session")

    with pytest.raises(UnsafeFilesystemOperation, match=r"escapes|link|reparse"):
        remove_confined_tree(link, root)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_atomic_write_replaces_link_entry_without_mutating_its_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    target = tmp_path / "result.bin"
    outside.write_bytes(b"preserve")
    try:
        os.link(outside, target)
    except OSError:
        pytest.skip("hard links are unavailable in this test session")

    atomic_write_bytes(target, b"replacement")

    assert target.read_bytes() == b"replacement"
    assert outside.read_bytes() == b"preserve"
