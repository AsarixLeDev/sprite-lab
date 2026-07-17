from __future__ import annotations

import os
from pathlib import Path

import pytest

import spritelab.product_features.harvest.trusted_backend as trusted_backend

REQUIRED_HARDENED_MODULES = {
    "spritelab.dataset_maker.exporter",
    "spritelab.harvest.archive",
    "spritelab.harvest.download",
    "spritelab.harvest.extract",
    "spritelab.harvest.sources",
    "spritelab.product_features.conditioned_v5.identity",
    "spritelab.product_features.conditioned_v5.audit_runner",
    "spritelab.product_features.dataset.intake",
    "spritelab.product_core.events",
    "spritelab.product_features.harvest",
    "spritelab.product_features.harvest.catalog",
    "spritelab.product_features.harvest.catalog_verifier",
    "spritelab.product_features.harvest.certification",
    "spritelab.product_features.harvest.service",
    "spritelab.product_features.harvest.storage",
    "spritelab.product_features.harvest.trusted_backend",
    "spritelab.product_features.harvest.web",
    "spritelab.product_runtime",
    "spritelab.product_web.app",
    "spritelab.training.campaign",
    "spritelab.utils.safe_fs",
}


def test_hardened_identity_covers_transitive_production_boundary_without_unused_pipeline() -> None:
    module_hashes = trusted_backend.hardened_backend_module_hashes()

    assert REQUIRED_HARDENED_MODULES <= set(module_hashes)
    assert "spritelab.harvest.pipeline" not in module_hashes
    assert all(len(digest) == 64 for digest in module_hashes.values())


def test_hardened_identity_opens_every_module_through_anchored_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = trusted_backend.AnchoredDirectory.open_file
    opened_paths: list[Path] = []

    def recording_open(anchor: trusted_backend.AnchoredDirectory, name: str, flags: int, mode: int = 0o600) -> int:
        opened_paths.append(anchor.directory / name)
        return real_open(anchor, name, flags, mode)

    monkeypatch.setattr(trusted_backend.AnchoredDirectory, "open_file", recording_open)

    module_hashes = trusted_backend.hardened_backend_module_hashes()

    assert len(opened_paths) == len(module_hashes)
    assert all(path.suffix == ".py" for path in opened_paths)


def test_hardened_identity_discovers_direct_first_party_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text("from spritelab.utils import safe_fs\n", encoding="utf-8")
    monkeypatch.setattr(trusted_backend, "_hardened_backend_module_paths", lambda: {"test.module": module_path})

    module_hashes = trusted_backend.hardened_backend_module_hashes()

    assert "test.module" in module_hashes
    assert "spritelab.utils.safe_fs" in module_hashes


def test_hardened_identity_binds_runtime_dependencies() -> None:
    dependencies = trusted_backend.hardened_backend_runtime_dependencies()

    assert set(dependencies) == {"NumPy", "OpenSSL", "Pillow", "PyYAML", "Python"}
    assert all(set(record) == {"version"} and record["version"] for record in dependencies.values())


def test_hardened_identity_binds_exact_conditioned_callback_code_and_runtime() -> None:
    from spritelab.product_features.conditioned_v5.identity import (
        conditioned_callback_runtime_inventory,
        conditioned_code_inventory,
    )

    inventory = conditioned_code_inventory()
    runtime = conditioned_callback_runtime_inventory(inventory)
    binding = trusted_backend.conditioned_dataset_import_callback_binding()

    assert binding == {
        "dataset_import_callback_id": "dataset.conditioned-intake",
        "dataset_import_callback_code_identity_sha256": inventory["inventory_sha256"],
        "dataset_import_callback_runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }


def test_hardened_identity_rejects_module_replaced_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "module.py"
    replacement = tmp_path / "replacement.py"
    module_path.write_text("ORIGINAL = True\n", encoding="utf-8")
    replacement.write_text("REPLACED = True\n", encoding="utf-8")
    real_open = trusted_backend.AnchoredDirectory.open_file
    swapped = False

    def swapping_open(
        anchor: trusted_backend.AnchoredDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal swapped
        if anchor.directory / name == module_path and not swapped:
            swapped = True
            os.replace(replacement, module_path)
        return real_open(anchor, name, flags, mode)

    monkeypatch.setattr(trusted_backend, "_hardened_backend_module_paths", lambda: {"test.module": module_path})
    monkeypatch.setattr(trusted_backend.AnchoredDirectory, "open_file", swapping_open)

    with pytest.raises(ValueError, match="changed while opening"):
        trusted_backend.hardened_backend_module_hashes()


def test_hardened_identity_rejects_changed_path_after_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "module.py"
    different_path = tmp_path / "different.py"
    module_path.write_text("ORIGINAL = True\n", encoding="utf-8")
    different_path.write_text("DIFFERENT = True\n", encoding="utf-8")
    real_lstat = trusted_backend.AnchoredDirectory.lstat
    module_lstat_calls = 0

    def changed_path_lstat(anchor: trusted_backend.AnchoredDirectory, name: str) -> os.stat_result:
        nonlocal module_lstat_calls
        if anchor.directory / name == module_path:
            module_lstat_calls += 1
            if module_lstat_calls == 2:
                return different_path.lstat()
        return real_lstat(anchor, name)

    monkeypatch.setattr(trusted_backend, "_hardened_backend_module_paths", lambda: {"test.module": module_path})
    monkeypatch.setattr(trusted_backend.AnchoredDirectory, "lstat", changed_path_lstat)

    with pytest.raises(ValueError, match="changed while reading"):
        trusted_backend.hardened_backend_module_hashes()
