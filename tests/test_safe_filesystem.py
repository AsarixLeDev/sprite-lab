from __future__ import annotations

import os
from pathlib import Path

import pytest

import spritelab.utils.safe_fs as safe_fs
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    atomic_write_bytes,
    open_anchored_directory,
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
    residues = list(root.glob(".spritelab-retired-tree-*"))
    assert len(residues) == 1
    assert (residues[0] / "generated.txt").read_text(encoding="utf-8") == "generated"
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


def test_anchored_atomic_write_cleans_owned_temp_after_fsync_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(safe_fs.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".spritelab-*.tmp"))


def test_anchored_atomic_write_cleans_owned_temp_after_write_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old")
    real_fdopen = safe_fs.os.fdopen

    class FailingWriter:
        def __init__(self, descriptor: int, mode: str) -> None:
            self._handle = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._handle.__exit__(exc_type, exc_value, traceback)

        def write(self, _content: bytes) -> int:
            raise OSError("injected write failure")

        def flush(self) -> None:
            self._handle.flush()

        def fileno(self) -> int:
            return self._handle.fileno()

    monkeypatch.setattr(safe_fs.os, "fdopen", lambda descriptor, mode: FailingWriter(descriptor, mode))

    with pytest.raises(OSError, match="injected write failure"):
        atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".spritelab-*.tmp"))


@pytest.mark.parametrize("operation", ["create", "append", "atomic", "publish"])
def test_anchored_mutations_do_not_follow_parent_rename_symlink_aba(tmp_path: Path, operation: str) -> None:
    root = tmp_path / "root"
    parent = root / "mutable"
    moved = root / "moved"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    (outside / "sentinel.bin").write_bytes(b"preserve")
    if operation == "append":
        (parent / "target.bin").write_bytes(b"old")
    if operation == "atomic":
        (parent / "target.bin").write_bytes(b"old")
    if operation == "publish":
        (parent / "source.bin").write_bytes(b"new")

    swapped = False
    with AnchoredDirectory(parent, root) as anchor:
        try:
            os.replace(parent, moved)
        except OSError:
            # Some filesystems refuse renaming held directories. Platforms
            # that allow it are protected by handle-relative child calls.
            if os.name != "nt":
                pytest.skip("the platform refused the parent-rename setup")
        else:
            try:
                os.symlink(outside, parent, target_is_directory=True)
            except OSError:
                os.replace(moved, parent)
                pytest.skip("directory symbolic links are unavailable in this test session")
            swapped = True
        try:
            if operation == "create":
                descriptor = anchor.open_file("target.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(b"new")
            elif operation == "append":
                descriptor = anchor.open_file("target.bin", os.O_WRONLY | os.O_APPEND)
                with os.fdopen(descriptor, "ab") as handle:
                    handle.write(b"+new")
            elif operation == "atomic":
                anchor.atomic_write_bytes("target.bin", b"new")
            else:
                anchor.link("source.bin", "target.bin")
            assert (outside / "sentinel.bin").read_bytes() == b"preserve"
            assert not (outside / "target.bin").exists()
        finally:
            if swapped:
                os.unlink(parent)
                os.replace(moved, parent)

    expected = b"old+new" if operation == "append" else b"new"
    assert (parent / "target.bin").read_bytes() == expected


def test_anchored_cleanup_refuses_raced_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owned = root / "owned.tmp"
    replacement = root / "replacement.tmp"
    owned.write_bytes(b"owned")
    replacement.write_bytes(b"sentinel")

    with AnchoredDirectory(root, root) as anchor:
        identity = OwnedFileIdentity.from_stat(anchor.lstat(owned.name))
        os.replace(replacement, owned)
        assert anchor.unlink_if_owned(owned.name, identity) is False

    assert owned.read_bytes() == b"sentinel"


def test_anchored_exact_mkdir_enumeration_and_nested_open(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with AnchoredDirectory(root, root) as anchor:
        identity = anchor.mkdir("artifacts")
        assert identity.matches(anchor.lstat("artifacts"))
        with pytest.raises(FileExistsError):
            anchor.mkdir("artifacts")
        assert anchor.mkdir("artifacts", exist_ok=True) == identity
        assert anchor.names() == ("artifacts",)
        with anchor.open_directory("artifacts") as artifacts:
            artifacts.mkdir("nested")
            artifacts.atomic_write_bytes("receipt.json", b"{}")
            assert artifacts.names() == ("nested", "receipt.json")

    assert (root / "artifacts" / "receipt.json").read_bytes() == b"{}"


def test_detached_anchor_duplicate_outlives_opening_chain(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "evidence.bin").write_bytes(b"evidence")

    with open_anchored_directory(child, root) as anchor:
        duplicate = anchor.detached_duplicate()

    try:
        assert duplicate.names() == ("evidence.bin",)
        descriptor = duplicate.open_file("evidence.bin", os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as handle:
            assert handle.read() == b"evidence"
    finally:
        duplicate.__exit__(None, None, None)


def test_held_directory_rename_noreplace_keeps_exact_anchor_live(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with AnchoredDirectory(root, root) as parent:
        parent.mkdir("candidate")
        with parent.open_directory("candidate") as candidate:
            candidate.atomic_write_bytes("evidence.bin", b"bound")
            parent.rename_held_directory_noreplace(candidate, "published")
            assert candidate.directory == root / "published"
            assert candidate.names() == ("evidence.bin",)
            candidate.atomic_write_bytes("receipt.json", b"{}")

    assert not (root / "candidate").exists()
    assert (root / "published" / "evidence.bin").read_bytes() == b"bound"
    assert (root / "published" / "receipt.json").read_bytes() == b"{}"


def test_held_directory_rename_never_replaces_destination(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with AnchoredDirectory(root, root) as parent:
        parent.mkdir("candidate")
        parent.mkdir("winner")
        with parent.open_directory("candidate") as candidate:
            with pytest.raises(FileExistsError):
                parent.rename_held_directory_noreplace(candidate, "winner")

    assert (root / "candidate").is_dir()
    assert (root / "winner").is_dir()


def test_immovable_child_anchor_refuses_held_rename_and_blocks_windows_path_swap(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child_path = root / "child"
    child_path.mkdir(parents=True)
    (child_path / "sentinel.bin").write_bytes(b"bound")

    with AnchoredDirectory(root, root) as parent:
        with parent.open_directory_immovable("child") as child:
            with pytest.raises(UnsafeFilesystemOperation, match="immovable"):
                parent.rename_held_directory_noreplace(child, "moved")
            if os.name == "nt":
                with pytest.raises(OSError):
                    os.replace(child_path, root / "moved")
            assert child.names() == ("sentinel.bin",)

    assert (child_path / "sentinel.bin").read_bytes() == b"bound"
    assert not (root / "moved").exists()


def test_anchored_public_operations_fail_closed_before_enter(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    anchor = AnchoredDirectory(root, root)

    with pytest.raises(UnsafeFilesystemOperation, match="not open"):
        anchor.names()
    with pytest.raises(UnsafeFilesystemOperation, match="not open"):
        anchor.mkdir("child")
    with pytest.raises(UnsafeFilesystemOperation, match="not open"):
        anchor.open_file("child", os.O_RDONLY)
    with pytest.raises(UnsafeFilesystemOperation, match="not open"):
        with anchor.open_directory("child"):
            pass


def test_quarantine_restores_foreign_entry_raced_between_check_and_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owned = root / "owned.tmp"
    foreign = root / "foreign.tmp"
    owned.write_bytes(b"owned")
    foreign.write_bytes(b"foreign sentinel")

    with AnchoredDirectory(root, root) as anchor:
        identity = OwnedFileIdentity.from_stat(anchor.lstat(owned.name))
        real_rename = anchor.rename
        raced = False

        def race_before_rename(source_name: str, destination_name: str, *, replace: bool) -> None:
            nonlocal raced
            if not raced and source_name == owned.name:
                raced = True
                os.replace(foreign, owned)
            real_rename(source_name, destination_name, replace=replace)

        monkeypatch.setattr(anchor, "rename", race_before_rename)
        residue = anchor.quarantine_if_owned(owned.name, identity, prefix=".residue-")

    assert residue is None
    assert owned.read_bytes() == b"foreign sentinel"
    assert not list(root.glob(".residue-*"))


def test_owned_cleanup_never_deletes_foreign_post_quarantine_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owned = root / "owned.tmp"
    foreign = root / "foreign.tmp"
    owned.write_bytes(b"owned")
    foreign.write_bytes(b"foreign sentinel")
    residue_path: Path | None = None

    with AnchoredDirectory(root, root) as anchor:
        identity = OwnedFileIdentity.from_stat(anchor.lstat(owned.name))
        real_quarantine = anchor.quarantine_if_owned

        def substitute_after_quarantine(
            name: str,
            owned_identity: OwnedFileIdentity,
            *,
            prefix: str,
        ) -> str | None:
            nonlocal residue_path
            residue = real_quarantine(name, owned_identity, prefix=prefix)
            assert residue is not None
            residue_path = root / residue
            os.replace(foreign, residue_path)
            return residue

        monkeypatch.setattr(anchor, "quarantine_if_owned", substitute_after_quarantine)
        if os.name == "nt":
            with pytest.raises(UnsafeFilesystemOperation, match="changed before deletion"):
                anchor.unlink_if_owned(owned.name, identity)
        else:
            assert anchor.unlink_if_owned(owned.name, identity) is True

    assert residue_path is not None
    assert residue_path.read_bytes() == b"foreign sentinel"
    assert not owned.exists()
