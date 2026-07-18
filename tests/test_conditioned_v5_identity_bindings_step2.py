from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spritelab.product_features.conditioned_v5 import intake as intake_module
from spritelab.utils.safe_fs import AnchoredDirectory
from spritelab.utils.write_confinement import DirectoryIdentity


def _harvest_documents() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run_id = "harvest-identity-binding"
    handoff: dict[str, object] = {
        "run_id": run_id,
        "provenance_identity": "b" * 64,
        "source_evidence_binding_identity": "c" * 64,
    }
    artifact_manifest: dict[str, object] = {"artifact_set_identity": "d" * 64}
    harvest: dict[str, object] = dict.fromkeys(intake_module._MANAGED_RECEIPT_HARVEST_KEYS, "a" * 64)
    harvest.update(
        {
            "run_id": run_id,
            "handoff_identity": intake_module.stable_hash(handoff),
            "request_handoff_identity": intake_module.stable_hash(handoff),
            "artifact_manifest_identity": intake_module.stable_hash(artifact_manifest),
            "artifact_set_identity": artifact_manifest["artifact_set_identity"],
            "provenance_identity": handoff["provenance_identity"],
            "source_evidence_binding_identity": handoff["source_evidence_binding_identity"],
            "backend_capability_issued_at": "2026-07-17T00:00:00Z",
            "backend_capability_expires_at": "2026-07-18T00:00:00Z",
        }
    )
    return harvest, handoff, artifact_manifest


def test_stored_request_handoff_identity_is_recomputed() -> None:
    harvest, handoff, artifact_manifest = _harvest_documents()
    intake_module._validate_managed_harvest_document(
        harvest,
        handoff_document=handoff,
        artifact_manifest=artifact_manifest,
    )

    forged = dict(harvest)
    forged["request_handoff_identity"] = "f" * 64
    with pytest.raises(intake_module.ConditionedIntakeError, match="Harvest bindings"):
        intake_module._validate_managed_harvest_document(
            forged,
            handoff_document=handoff,
            artifact_manifest=artifact_manifest,
        )

    with pytest.raises(intake_module.ConditionedIntakeError, match="changed during Dataset import"):
        intake_module._require_same_harvest_verification(
            {"request_handoff_identity": "1" * 64},
            {"request_handoff_identity": "2" * 64},
        )
    with pytest.raises(intake_module.ConditionedIntakeError, match="stale"):
        intake_module._require_receipt_harvest_bindings(
            {"request_handoff_identity": "1" * 64},
            {"request_handoff_identity": "2" * 64},
        )


def test_child_result_identity_is_bound_to_persisted_result_json(tmp_path: Path) -> None:
    output = tmp_path / "managed"
    output.mkdir()
    result = {
        "schema_version": "spritelab.product.result.v1",
        "status": "COMPLETE",
        "data": {"processed": 1},
    }
    (output / "result.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    expected = intake_module.stable_hash(result)

    with AnchoredDirectory(output, output) as anchor:
        intake_module._validate_persisted_intake_result(anchor, expected)
        with pytest.raises(intake_module.ConditionedIntakeError, match="persisted result"):
            intake_module._validate_persisted_intake_result(anchor, "e" * 64)


def test_stored_confinement_root_is_bound_to_reopened_work_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "intake-work"
    work.mkdir()
    identity = DirectoryIdentity.from_stat(os.stat(work))
    strategy = "test-exact-confinement"
    monkeypatch.setattr(intake_module, "write_confinement_strategy", lambda: strategy)

    with AnchoredDirectory(work, work) as anchor:
        intake_module._validate_persisted_confinement_binding(
            {"strategy": strategy, "root_identity_sha256": identity.identity_sha256},
            anchor,
        )
        with pytest.raises(intake_module.ConditionedIntakeError, match="transaction root"):
            intake_module._validate_persisted_confinement_binding(
                {"strategy": strategy, "root_identity_sha256": "0" * 64},
                anchor,
            )
        with pytest.raises(intake_module.ConditionedIntakeError, match="transaction root"):
            intake_module._validate_persisted_confinement_binding(
                {"strategy": "different", "root_identity_sha256": identity.identity_sha256},
                anchor,
            )
