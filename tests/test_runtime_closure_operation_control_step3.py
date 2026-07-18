from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from spritelab.training import smoke_bundle
from spritelab.utils import runtime_closure
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation


class _Stopped(RuntimeError):
    pass


def test_stable_read_checks_operation_between_megabyte_chunks(tmp_path: Path) -> None:
    payload = b"a" * (3 * 1024 * 1024)
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)
    calls = 0

    def operation_check() -> None:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise _Stopped

    with pytest.raises(_Stopped):
        smoke_bundle.read_stable_single_link_bytes(
            source,
            boundary=tmp_path,
            max_bytes=len(payload),
            operation_check=operation_check,
        )

    assert calls == 7


def test_runtime_scan_checks_operation_after_directory_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    package = runtime_root / "package"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    armed = False
    lstat_names_after_enumeration: list[str] = []
    original_names = AnchoredDirectory.names
    original_lstat = AnchoredDirectory.lstat

    def names(anchor: AnchoredDirectory) -> list[str]:
        nonlocal armed
        result = original_names(anchor)
        if anchor.directory.name == "package":
            armed = True
        return result

    def lstat(anchor: AnchoredDirectory, name: str):  # type: ignore[no-untyped-def]
        if armed:
            lstat_names_after_enumeration.append(name)
        return original_lstat(anchor, name)

    def operation_check() -> None:
        if armed:
            raise _Stopped

    monkeypatch.setattr(AnchoredDirectory, "names", names)
    monkeypatch.setattr(AnchoredDirectory, "lstat", lstat)

    with pytest.raises(_Stopped):
        smoke_bundle._scan_runtime_files_with_identity(
            runtime_root,
            ["package"],
            operation_check=operation_check,
        )

    assert "module.py" not in lstat_names_after_enumeration


def test_exact_source_loader_checks_operation_before_exec(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = tmp_path / "module.py"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n".encode()
    source.write_bytes(payload)
    calls = 0

    def operation_check() -> None:
        nonlocal calls
        calls += 1
        if calls == 14:
            raise _Stopped

    loader = smoke_bundle._ExactRuntimeSourceLoader(
        "bound_test_module",
        str(source),
        hashlib.sha256(payload).hexdigest(),
        False,
        operation_check=operation_check,
    )
    spec = importlib.util.spec_from_loader("bound_test_module", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(_Stopped):
        loader.exec_module(module)

    assert calls == 14
    assert not marker.exists()


def test_publication_interruption_leaves_uncommitted_direct_final_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publications"
    parent.mkdir()
    sentinel = tmp_path / "outside-sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    original_write = smoke_bundle._write_exclusive_to_anchor

    def interrupted_write(anchor, name, content, *, operation_check=None):  # type: ignore[no-untyped-def]
        original_write(anchor, name, content, operation_check=operation_check)
        if name == "payload.json":
            raise _Stopped

    monkeypatch.setattr(smoke_bundle, "_write_exclusive_to_anchor", interrupted_write)

    with pytest.raises(_Stopped):
        smoke_bundle.publish_immutable_tree(
            parent,
            root=tmp_path,
            final_name="bundle",
            files={"payload.json": b"{}"},
        )

    publication = parent / "bundle"
    assert publication.is_dir()
    assert (publication / "payload.json").read_bytes() == b"{}"
    assert not (publication / smoke_bundle._PUBLICATION_COMPLETION_FILENAME).exists()
    with pytest.raises(smoke_bundle.SmokeBundleError, match="completion marker"):
        smoke_bundle._require_publication_complete(
            publication,
            boundary=tmp_path,
            publication_name="bundle",
            exact=True,
        )
    assert sentinel.read_bytes() == b"unchanged"


def test_publication_and_exclusive_file_use_no_path_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publications"
    parent.mkdir()

    def forbidden_rename(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("publication must not use a pathname directory/file rename")

    monkeypatch.setattr(AnchoredDirectory, "rename", forbidden_rename)
    publication = smoke_bundle.publish_immutable_tree(
        parent,
        root=tmp_path,
        final_name="bundle",
        files={"nested/payload.json": b"{}"},
    )
    smoke_bundle.write_exclusive_bytes(
        parent / "single.json",
        b"[]",
        boundary=tmp_path,
    )

    assert (publication / smoke_bundle._PUBLICATION_COMPLETION_FILENAME).is_file()
    assert (
        smoke_bundle._require_publication_complete(
            publication,
            boundary=tmp_path,
            publication_name="bundle",
            exact=True,
        )["status"]
        == "COMPLETE"
    )
    assert (parent / "single.json").read_bytes() == b"[]"
    assert os.stat(parent / "single.json").st_nlink == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only anonymous-file fallback")
def test_exclusive_file_falls_back_to_direct_final_o_excl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(_anchor: AnchoredDirectory, _mode: int = 0o600) -> int:
        raise UnsafeFilesystemOperation("anonymous files disabled by test")

    monkeypatch.setattr(AnchoredDirectory, "open_anonymous_file", unsupported)
    target = tmp_path / "direct-final.json"

    smoke_bundle.write_exclusive_bytes(target, b"exact", boundary=tmp_path)

    assert target.read_bytes() == b"exact"
    assert target.stat().st_nlink == 1


@pytest.mark.parametrize("replacement", ["hardlink", "symlink"])
def test_publication_rejects_file_substitution_without_touching_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    parent = tmp_path / "publications"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original_write = smoke_bundle._write_exclusive_to_anchor
    attacked = False

    def substitute(anchor, name, content, *, operation_check=None):  # type: ignore[no-untyped-def]
        nonlocal attacked
        original_write(anchor, name, content, operation_check=operation_check)
        if name != "payload.json" or attacked:
            return
        attacked = True
        payload = parent / "bundle" / name
        payload.rename(parent / "bundle" / ".owned-payload.json")
        try:
            if replacement == "hardlink":
                os.link(sentinel, payload)
            else:
                os.symlink(sentinel, payload)
        except OSError as exc:
            pytest.skip(f"{replacement} substitution is unavailable: {exc}")

    monkeypatch.setattr(smoke_bundle, "_write_exclusive_to_anchor", substitute)

    with pytest.raises((OSError, smoke_bundle.SmokeBundleError, UnsafeFilesystemOperation)):
        smoke_bundle.publish_immutable_tree(
            parent,
            root=tmp_path,
            final_name="bundle",
            files={"payload.json": b"trusted"},
        )

    assert attacked is True
    assert sentinel.read_bytes() == b"outside-byte-identical"
    assert not (parent / "bundle" / smoke_bundle._PUBLICATION_COMPLETION_FILENAME).exists()


def test_publication_rejects_directory_namespace_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publications"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    decoy = outside / "decoy"
    decoy.mkdir()
    original_write = smoke_bundle._write_exclusive_to_anchor
    original_lstat = AnchoredDirectory.lstat
    armed = False

    def arm_after_write(anchor, name, content, *, operation_check=None):  # type: ignore[no-untyped-def]
        nonlocal armed
        original_write(anchor, name, content, operation_check=operation_check)
        if name == "payload.json":
            armed = True

    def substituted_lstat(anchor: AnchoredDirectory, name: str):  # type: ignore[no-untyped-def]
        if armed and anchor.directory == parent and name == "bundle":
            return decoy.lstat()
        return original_lstat(anchor, name)

    monkeypatch.setattr(smoke_bundle, "_write_exclusive_to_anchor", arm_after_write)
    monkeypatch.setattr(AnchoredDirectory, "lstat", substituted_lstat)

    with pytest.raises((smoke_bundle.SmokeBundleError, UnsafeFilesystemOperation), match="changed"):
        smoke_bundle.publish_immutable_tree(
            parent,
            root=tmp_path,
            final_name="bundle",
            files={"payload.json": b"trusted"},
        )

    assert sentinel.read_bytes() == b"outside-byte-identical"
    assert not (parent / "bundle" / smoke_bundle._PUBLICATION_COMPLETION_FILENAME).exists()


def test_neutral_environment_path_api_forwards_operation_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def operation_check() -> None:
        observed["called"] = True

    def environment_paths(project_root: str | Path, *, operation_check=None):  # type: ignore[no-untyped-def]
        observed["root"] = project_root
        observed["check"] = operation_check
        operation_check()
        return (("imports",), ("runtime",))

    monkeypatch.setattr(runtime_closure, "smoke_runtime_environment_paths", environment_paths)

    result = runtime_closure.exact_python_runtime_environment_paths(
        tmp_path,
        operation_check=operation_check,
    )

    assert result == (("imports",), ("runtime",))
    assert observed == {"root": tmp_path, "check": operation_check, "called": True}


def test_nested_runtime_path_scan_inherits_outer_operation_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_calls = 0
    armed = False
    standard_roots_called = False

    def isolated_paths(project_root: Path) -> list[str]:
        nonlocal path_calls, armed
        assert project_root == tmp_path
        path_calls += 1
        if path_calls == 2:
            armed = True
        return [str(tmp_path), str(tmp_path / "runtime")]

    def standard_roots():  # type: ignore[no-untyped-def]
        nonlocal standard_roots_called
        standard_roots_called = True
        return []

    def operation_check() -> None:
        if armed:
            raise _Stopped

    monkeypatch.setattr(smoke_bundle, "_isolated_import_paths", isolated_paths)
    monkeypatch.setattr(smoke_bundle, "_standard_runtime_root_specs", standard_roots)

    with pytest.raises(_Stopped):
        runtime_closure.exact_python_runtime_environment_paths(
            tmp_path,
            operation_check=operation_check,
        )

    assert path_calls == 2
    assert standard_roots_called is False
