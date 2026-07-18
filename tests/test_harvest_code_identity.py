from __future__ import annotations

import json
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


def test_hardened_identity_snapshot_captures_each_boundary_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from spritelab.product_features.conditioned_v5 import identity as conditioned_identity

    calls = {"modules": 0, "conditioned": 0, "python": 0, "openssl": 0, "runtime": 0}
    callback_binding = {
        "dataset_import_callback_id": "dataset.conditioned-intake",
        "dataset_import_callback_code_identity_sha256": "a" * 64,
        "dataset_import_callback_runtime_identity_sha256": "b" * 64,
    }
    runtime_dependencies = {
        "Python": {"runtime_identity_sha256": "c" * 64},
        "OpenSSL": {"runtime_identity_sha256": "d" * 64},
    }

    def modules() -> tuple[tuple[str, bytes], ...]:
        calls["modules"] += 1
        return (("test.module", b"VALUE = 1\n"),)

    def conditioned() -> dict[str, object]:
        calls["conditioned"] += 1
        return {"inventory_sha256": "a" * 64, "runtime_dependencies": {}}

    def python_runtime() -> dict[str, object]:
        calls["python"] += 1
        return runtime_dependencies["Python"]

    def openssl_runtime() -> dict[str, object]:
        calls["openssl"] += 1
        return runtime_dependencies["OpenSSL"]

    def runtime_dependencies_from_snapshot(**kwargs: object) -> dict[str, dict[str, object]]:
        calls["runtime"] += 1
        assert kwargs["python_runtime"] == runtime_dependencies["Python"]
        assert kwargs["openssl_runtime"] == runtime_dependencies["OpenSSL"]
        return runtime_dependencies

    monkeypatch.setattr(trusted_backend, "_read_hardened_backend_modules", modules)
    monkeypatch.setattr(conditioned_identity, "conditioned_code_inventory", conditioned)
    monkeypatch.setattr(
        conditioned_identity,
        "conditioned_callback_runtime_inventory",
        lambda _inventory: {"runtime_identity_sha256": "b" * 64},
    )
    monkeypatch.setattr(trusted_backend, "_python_runtime_identity", python_runtime)
    monkeypatch.setattr(trusted_backend, "_openssl_runtime_identity", openssl_runtime)
    monkeypatch.setattr(trusted_backend, "_hardened_backend_runtime_dependencies", runtime_dependencies_from_snapshot)

    snapshot = trusted_backend.hardened_backend_identity_snapshot()

    module_digest = trusted_backend.hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert snapshot.module_sha256 == {"test.module": module_digest}
    assert snapshot.runtime_dependencies == runtime_dependencies
    assert snapshot.callback_binding == callback_binding
    assert snapshot.code_identity_sha256 == trusted_backend._identity(
        {
            "schema_version": "spritelab.harvest.hardened-backend-code.v4",
            "modules": [
                {
                    "module": "test.module",
                    "sha256": module_digest,
                    "byte_count": len(b"VALUE = 1\n"),
                }
            ],
            "runtime_dependencies": runtime_dependencies,
            "dataset_import_callback": callback_binding,
        }
    )
    assert calls == {"modules": 1, "conditioned": 1, "python": 1, "openssl": 1, "runtime": 1}


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
    for name in ("NumPy", "Pillow", "PyYAML"):
        record = dependencies[name]
        assert record["schema_version"] == "spritelab.harvest.runtime-distribution.v2"
        assert record["inventory_schema_version"] == "spritelab.runtime.installed-distribution-inventory.v2"
        assert record["version"]
        assert len(record["inventory_sha256"]) == 64
        assert len(record["installation_root_identity_sha256"]) == 64
        assert record["record_file_count"] > 0
        assert record["owned_root_count"] > 0
        assert record["file_count"] > 0
        assert record["unrecorded_file_count"] >= 0
        assert record["record_file_count"] + record["unrecorded_file_count"] == record["file_count"]
        assert record["total_bytes"] > 0
        assert "files" not in record
        assert record["paths_exposed"] is False
    python = dependencies["Python"]
    assert python["stdlib_inventory"]["file_count"] > 100
    assert python["native_inventory"]["file_count"] > 0
    assert python["interpreter_libraries"]["paths_exposed"] is False
    assert {"executable", "ssl_stdlib", "ssl_native"} <= set(python)
    openssl = dependencies["OpenSSL"]
    assert {item["role"] for item in openssl["libraries"]} >= {"crypto", "ssl"}
    assert all(set(item) == {"role", "sha256", "byte_count", "metadata_sha256"} for item in openssl["libraries"])
    serialized = json.dumps(dependencies)
    assert len(serialized.encode("utf-8")) < 1 << 20
    assert os.fspath(Path.home()) not in serialized


def test_compact_distribution_identity_changes_on_same_version_same_size_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.product_features.conditioned_v5 import identity as conditioned_identity

    current = conditioned_identity.installed_distribution_inventory("PyYAML")
    replacement = dict(current)
    replacement_files = {name: dict(binding) for name, binding in current["files"].items()}
    changed_name = min(replacement_files)
    changed_binding = replacement_files[changed_name]
    changed_binding["sha256"] = "f" * 64 if changed_binding["sha256"] != "f" * 64 else "e" * 64
    replacement["files"] = replacement_files
    replacement["inventory_sha256"] = trusted_backend._identity(
        {key: value for key, value in replacement.items() if key != "inventory_sha256"}
    )
    monkeypatch.setattr(conditioned_identity, "installed_distribution_inventory", lambda _name: current)
    before = trusted_backend._compact_distribution_identity("PyYAML")
    monkeypatch.setattr(conditioned_identity, "installed_distribution_inventory", lambda _name: replacement)
    after = trusted_backend._compact_distribution_identity("PyYAML")

    assert before["version"] == after["version"]
    assert before["file_count"] == after["file_count"]
    assert before["total_bytes"] == after["total_bytes"]
    assert before["runtime_identity_sha256"] != after["runtime_identity_sha256"]


def test_python_stdlib_inventory_changes_on_same_size_non_ssl_module_drift(tmp_path: Path) -> None:
    module = tmp_path / "zipfile.py"
    bytecode = tmp_path / "__pycache__" / "zipfile.cpython-test.pyc"
    module.write_bytes(b"TRUST = 'original'\n")
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"same-size-pyc")
    before = trusted_backend._runtime_tree_inventory(
        tmp_path,
        label="test-stdlib",
        excluded_top_level=frozenset(),
    )
    module.write_bytes(b"TRUST = 'modified'\n")
    after_source = trusted_backend._runtime_tree_inventory(
        tmp_path,
        label="test-stdlib",
        excluded_top_level=frozenset(),
    )
    bytecode.write_bytes(b"drifted-pyc!!")
    after_bytecode = trusted_backend._runtime_tree_inventory(
        tmp_path,
        label="test-stdlib",
        excluded_top_level=frozenset(),
    )

    assert module.stat().st_size == len(b"TRUST = 'original'\n")
    assert before["total_bytes"] == after_source["total_bytes"] == after_bytecode["total_bytes"]
    assert len({before["inventory_sha256"], after_source["inventory_sha256"], after_bytecode["inventory_sha256"]}) == 3


@pytest.mark.parametrize(
    ("name", "role"),
    [("libssl.so.1.1", "ssl"), ("libcrypto.so.3.2.1", "crypto")],
)
def test_openssl_library_role_accepts_versioned_posix_sonames(name: str, role: str) -> None:
    assert trusted_backend._openssl_library_role(name) == role


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
