from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import spritelab.product_features.conditioned_v5.audit_runner as audit_runner
from spritelab.utils.safe_fs import open_anchored_directory

_STAGE_A = "a" * 32
_STAGE_B = "b" * 32


def _hard_link_or_skip(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable in this test session: {exc}")


def test_audit_read_and_inventory_accept_one_exact_retained_stage_alias(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    target = root / "payload.bin"
    payload = b"exact retained publication bytes"
    target.write_bytes(payload)
    alias = root / f".payload.bin.staging-{_STAGE_A}"
    _hard_link_or_skip(target, alias)

    with open_anchored_directory(root, root) as anchor:
        assert audit_runner._bound_directory_names(anchor) == ("payload.bin",)
        assert audit_runner._read_bound_file(anchor, "payload.bin", len(payload)) == payload
        audit_runner._verify_inventory_tree(
            anchor,
            {"payload.bin": (hashlib.sha256(payload).hexdigest(), len(payload))},
            cancelled=lambda: False,
        )

    assert target.stat().st_nlink == alias.stat().st_nlink == 2


def test_audit_rejects_two_link_target_without_its_named_retained_stage(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    target = root / "payload.bin"
    outside = tmp_path / "outside.bin"
    payload = b"outside hard link must not impersonate a retained stage"
    target.write_bytes(payload)
    _hard_link_or_skip(target, outside)

    with open_anchored_directory(root, root) as anchor:
        with pytest.raises(audit_runner.IndependentAuditError, match="sole retained publication stage"):
            audit_runner._read_bound_file(anchor, "payload.bin", len(payload))
        with pytest.raises(audit_runner.IndependentAuditError, match="unbound hard-link topology"):
            audit_runner._bound_directory_names(anchor)

    assert outside.read_bytes() == payload


def test_audit_rejects_malformed_reserved_stage_suffix(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    target = root / "payload.bin"
    payload = b"lowercase suffix contract"
    target.write_bytes(payload)
    malformed = root / f".payload.bin.staging-{'A' * 32}"
    _hard_link_or_skip(target, malformed)

    with open_anchored_directory(root, root) as anchor:
        with pytest.raises(audit_runner.IndependentAuditError, match="malformed or unexpected"):
            audit_runner._read_bound_file(anchor, "payload.bin", len(payload))
        with pytest.raises(audit_runner.IndependentAuditError, match="malformed reserved"):
            audit_runner._bound_directory_names(anchor)


def test_audit_rejects_validly_named_stage_on_a_different_inode(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    target = root / "payload.bin"
    payload = b"matching bytes do not establish inode authority"
    target.write_bytes(payload)
    (root / f".payload.bin.staging-{_STAGE_A}").write_bytes(payload)

    with open_anchored_directory(root, root) as anchor:
        with pytest.raises(audit_runner.IndependentAuditError, match="malformed or unexpected"):
            audit_runner._read_bound_file(anchor, "payload.bin", len(payload))
        with pytest.raises(audit_runner.IndependentAuditError, match="exact two-link inode alias"):
            audit_runner._bound_directory_names(anchor)


def test_audit_rejects_extra_stage_candidate_even_with_one_exact_alias(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    target = root / "payload.bin"
    payload = b"one alias only"
    target.write_bytes(payload)
    exact_alias = root / f".payload.bin.staging-{_STAGE_A}"
    wrong_alias = root / f".payload.bin.staging-{_STAGE_B}"
    _hard_link_or_skip(target, exact_alias)
    wrong_alias.write_bytes(payload)

    with open_anchored_directory(root, root) as anchor:
        with pytest.raises(audit_runner.IndependentAuditError, match="sole retained publication stage"):
            audit_runner._read_bound_file(anchor, "payload.bin", len(payload))
        with pytest.raises(audit_runner.IndependentAuditError, match="extra retained publication stage"):
            audit_runner._bound_directory_names(anchor)
