from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from spritelab.dev_features.artifacts import inspect_artifacts
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import ProjectState


def _config(root: Path) -> ProjectConfig:
    values = copy.deepcopy(DEFAULT_CONFIG)
    for section in ("dataset", "labeling", "training", "evaluation"):
        for key in values[section]:
            values[section][key] = [] if key == "review_queues" else ""
    return ProjectConfig(root, None, values)


def _state(root: Path) -> ProjectState:
    return ProjectState("synthetic", root, None, "a" * 40, [])


def test_artifact_missing_is_reported_without_rewrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.values["dataset"]["raw_inventory"] = "fixtures/missing inventory.jsonl"
    artifacts = inspect_artifacts(config, _state(tmp_path))
    missing = next(item for item in artifacts if item["reference"].endswith("missing inventory.jsonl"))
    assert missing["identity_status"] == "MISSING"
    assert not (tmp_path / "fixtures").exists()


def test_artifact_hash_mismatch_is_stale(tmp_path: Path) -> None:
    target = tmp_path / "artifact with spaces.bin"
    target.write_bytes(b"current")
    hashes = tmp_path / "hashes.json"
    hashes.write_text(
        json.dumps({"files": [{"path": target.name, "sha256_before": hashlib.sha256(b"old").hexdigest()}]}),
        encoding="utf-8",
    )
    config = _config(tmp_path)
    config.values["training"]["audit_hashes"] = hashes.name
    artifacts = inspect_artifacts(config, _state(tmp_path))
    mismatch = next(item for item in artifacts if item["reference"] == target.name)
    assert mismatch["identity_status"] == "HASH_MISMATCH"
    assert mismatch["stale"] is True
    assert target.read_bytes() == b"current"


def test_windows_paths_and_spaces_remain_single_references(tmp_path: Path) -> None:
    fixture = tmp_path / "Windows fixtures" / "sample report.json"
    fixture.parent.mkdir()
    fixture.write_text("{}", encoding="utf-8")
    config = _config(tmp_path)
    config.values["dataset"]["raw_provenance_report"] = str(fixture)
    artifacts = inspect_artifacts(config, _state(tmp_path))
    record = next(item for item in artifacts if item["reference"] == str(fixture))
    assert record["identity_status"] == "PRESENT"
    assert record["path"] == str(fixture.resolve())
