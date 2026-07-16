from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from spritelab.product_features.evaluation import CheckpointAvailability, discover_checkpoint_candidates


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(
    root: Path,
    run_id: str,
    *,
    status: str = "COMPLETE",
    dataset_identity: str = "dataset-v1",
    view_identity: str = "view-v1",
    unsafe: bool = False,
    project_root: Path | None = None,
) -> Path:
    directory = root / "runs" / run_id
    (directory / "checkpoints").mkdir(parents=True)
    _write_json(
        directory / "state.json",
        {
            "schema_version": "spritelab.v3.run-state.v1",
            "run_id": run_id,
            "command": "train",
            "status": status,
            "started_at": "2026-07-12T10:00:00+00:00",
            "ended_at": "2026-07-12T11:00:00+00:00" if status == "COMPLETE" else None,
            "backend_identity": {
                "friendly_run_name": "baseline",
                "training_profile": "production-32px",
                "dataset_identity": dataset_identity,
                "dataset_identity_summary": "Frozen Dataset v5 · 2,048 sprites",
                "view_identity": view_identity,
                "unsafe_resume_record": {"reason": "inspection"} if unsafe else None,
            },
        },
    )
    _write_json(
        directory / "command.json",
        {"command": "train", "project_root": str(project_root or root)},
    )
    return directory


def _checkpoint_evidence(
    run: Path,
    *,
    state_claim: dict[str, object] | None = None,
    verification_path: str = "checkpoints/checkpoint.pt",
) -> tuple[Path, str]:
    checkpoint = run / "checkpoints" / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint evidence")
    checkpoint_sha256 = sha256(checkpoint.read_bytes()).hexdigest()
    good_claim: dict[str, object] = {
        "path": "checkpoints/checkpoint.pt",
        "dataset_identity": "dataset-v1",
        "view_identity": "view-v1",
        "sha256": checkpoint_sha256,
        "verified": True,
    }
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    state["checkpoints"] = [{**good_claim, **(state_claim or {})}]
    _write_json(run / "state.json", state)
    _write_json(
        run / "checkpoint_verification.json",
        {
            "checkpoints": [
                {
                    **good_claim,
                    "path": verification_path,
                }
            ]
        },
    )
    return checkpoint, checkpoint_sha256


def test_no_checkpoint_has_no_eligible_candidate(tmp_path: Path) -> None:
    _run(tmp_path, "train-empty")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.eligible == ()
    assert catalog.default_checkpoint_id is None
    assert catalog.unavailable[0].availability is CheckpointAvailability.MISSING


def test_incomplete_checkpoint_is_hidden_from_normal_selection(tmp_path: Path) -> None:
    run = _run(tmp_path, "train-running", status="RUNNING")
    (run / "checkpoints" / "checkpoint_step_000100_ema.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INCOMPLETE
    assert catalog.to_dict()["eligible"] == []


def test_unsafe_checkpoint_is_hidden_and_explainable(tmp_path: Path) -> None:
    run = _run(tmp_path, "unsafe-run", unsafe=True)
    (run / "checkpoints" / "checkpoint_step_000200.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.eligible == ()
    unavailable = catalog.to_dict(include_unavailable=True)["unavailable"][0]
    assert unavailable["availability"] == "UNSAFE_RESUME"
    assert "unsafe-resume" in unavailable["unavailable_reasons"][0]


def test_stale_dataset_identity_is_not_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "stale-run", dataset_identity="old-dataset")
    (run / "checkpoints" / "checkpoint_last_ema.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="current-dataset",
    )
    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.STALE_DATASET


def test_stale_training_view_identity_is_not_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "stale-view-run", view_identity="old-view")
    (run / "checkpoints" / "checkpoint_last_ema.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="current-view",
    )
    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.STALE_VIEW
    assert catalog.unavailable[0].view_identity == "old-view"


def test_conflicting_dataset_identity_aliases_are_invalid(tmp_path: Path) -> None:
    run = _run(tmp_path, "conflicting-identities")
    checkpoint = run / "checkpoints" / "checkpoint_last_ema.pt"
    checkpoint.write_bytes(b"checkpoint")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["dataset_identity"] = "conflicting-dataset"
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "aliases disagree" in catalog.unavailable[0].unavailable_reasons[0]


def test_conflicting_view_identity_aliases_are_invalid(tmp_path: Path) -> None:
    run = _run(tmp_path, "conflicting-view-identities")
    checkpoint = run / "checkpoints" / "checkpoint_last_ema.pt"
    checkpoint.write_bytes(b"checkpoint")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["training_view_identity"] = "conflicting-view"
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "aliases disagree" in catalog.unavailable[0].unavailable_reasons[0]


@pytest.mark.parametrize("value", ("", 7, True, ["view-v1"], " padded-view "))
def test_malformed_checkpoint_view_identity_alias_is_invalid(tmp_path: Path, value: object) -> None:
    run = _run(tmp_path, "malformed-view")
    checkpoint = run / "checkpoints" / "checkpoint_step_000100_ema.pt"
    checkpoint.write_bytes(b"checkpoint")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["training_view_identity"] = value
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "aliases are malformed" in catalog.unavailable[0].unavailable_reasons[0]


@pytest.mark.parametrize("value", ("", 7, True, ["dataset-v1"], " padded-dataset "))
def test_malformed_checkpoint_dataset_identity_alias_is_invalid(tmp_path: Path, value: object) -> None:
    run = _run(tmp_path, "malformed-dataset")
    checkpoint = run / "checkpoints" / "checkpoint_step_000100_ema.pt"
    checkpoint.write_bytes(b"checkpoint")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["training_dataset_identity"] = value
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "aliases are malformed" in catalog.unavailable[0].unavailable_reasons[0]


@pytest.mark.parametrize(
    ("bad_claim", "availability", "reason"),
    (
        ({"dataset_identity": "wrong-dataset"}, CheckpointAvailability.INVALID, "dataset identity aliases disagree"),
        ({"view_identity": "wrong-view"}, CheckpointAvailability.INVALID, "view identity aliases disagree"),
        ({"dataset_identity": 7}, CheckpointAvailability.INVALID, "dataset identity aliases are malformed"),
        ({"view_identity": True}, CheckpointAvailability.INVALID, "view identity aliases are malformed"),
        ({"sha256": "0" * 64}, CheckpointAvailability.INVALID, "SHA-256 evidence disagrees"),
        ({"sha256": 7}, CheckpointAvailability.INVALID, "evidence is malformed"),
        ({"verified": False}, CheckpointAvailability.INVALID, "not passing across all durable sources"),
        ({"verified": "yes"}, CheckpointAvailability.INVALID, "evidence is malformed"),
        ({"safe_resume": False}, CheckpointAvailability.INVALID, "not passing across all durable sources"),
        (
            {"unsafe_resume_record": {"reason": "retained synthetic revocation"}},
            CheckpointAvailability.UNSAFE_RESUME,
            "unsafe-resume",
        ),
    ),
)
def test_same_path_rows_cannot_hide_contradictory_or_malformed_evidence(
    tmp_path: Path,
    bad_claim: dict[str, object],
    availability: CheckpointAvailability,
    reason: str,
) -> None:
    run = _run(tmp_path, "same-path-conflict")
    _checkpoint_evidence(run, state_claim=bad_claim)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert len(catalog.unavailable) == 1
    assert catalog.unavailable[0].availability is availability
    assert any(reason in item for item in catalog.unavailable[0].unavailable_reasons)


def test_relative_and_absolute_aliases_are_one_fail_closed_candidate(tmp_path: Path) -> None:
    run = _run(tmp_path, "path-alias-conflict")
    checkpoint = run / "checkpoints" / "checkpoint.pt"
    _checkpoint_evidence(
        run,
        state_claim={"dataset_identity": "wrong-dataset"},
        verification_path=str(checkpoint.resolve()),
    )

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert len(catalog.unavailable) == 1
    assert catalog.unavailable[0].path == checkpoint.resolve()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert any("dataset identity aliases disagree" in item for item in catalog.unavailable[0].unavailable_reasons)


def test_completion_manifest_and_verification_rows_are_all_retained(tmp_path: Path) -> None:
    run = _run(tmp_path, "all-checkpoint-sources")
    checkpoint, checkpoint_sha256 = _checkpoint_evidence(run)
    good_claim = {
        "path": "checkpoints/checkpoint.pt",
        "dataset_identity": "dataset-v1",
        "view_identity": "view-v1",
        "sha256": checkpoint_sha256,
        "verified": True,
    }
    _write_json(
        run / "run_completion_marker.json",
        {"checkpoints": [{**good_claim, "dataset_identity": "wrong-dataset"}]},
    )
    _write_json(
        run / "checkpoint_manifest.json",
        {"checkpoints": [{**good_claim, "view_identity": "wrong-view"}]},
    )

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert catalog.eligible == ()
    assert len(catalog.unavailable) == 1
    assert catalog.unavailable[0].path == checkpoint.resolve()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID


def test_consistent_repeated_checkpoint_evidence_remains_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "consistent-checkpoint-sources")
    checkpoint, checkpoint_sha256 = _checkpoint_evidence(run)
    claim = {
        "path": "checkpoints/checkpoint.pt",
        "dataset_identity": "dataset-v1",
        "view_identity": "view-v1",
        "sha256": checkpoint_sha256,
        "verified": True,
    }
    _write_json(run / "run_completion_marker.json", {"checkpoints": [claim]})
    _write_json(run / "checkpoint_manifest.json", {"checkpoints": [claim]})

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert len(catalog.eligible) == 1
    assert catalog.unavailable == ()
    assert catalog.eligible[0].path == checkpoint.resolve()


def test_latest_complete_checkpoint_is_default_and_public_view_redacts_paths(tmp_path: Path) -> None:
    older = _run(tmp_path, "older")
    newer = _run(tmp_path, "newer")
    state = json.loads((newer / "state.json").read_text(encoding="utf-8"))
    state["ended_at"] = "2026-07-13T11:00:00+00:00"
    _write_json(newer / "state.json", state)
    (older / "checkpoints" / "checkpoint_step_000100.pt").write_bytes(b"old")
    (newer / "checkpoints" / "checkpoint_step_000200.pt").write_bytes(b"new")
    (newer / "checkpoints" / "checkpoint_step_000200_ema.pt").write_bytes(b"ema")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    default = catalog.find(None)
    assert default is not None
    assert default.run_id == "newer"
    assert default.weights == "ema"
    public = catalog.to_dict()
    assert public["label"] == "baseline — latest complete checkpoint"
    assert "checkpoint_path" not in json.dumps(public)
    assert catalog.find(default.checkpoint_id, weights="live").weights == "live"


def test_foreign_checkpoint_is_not_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "foreign", project_root=tmp_path / "somewhere-else")
    (run / "checkpoints" / "checkpoint_last.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.unavailable[0].availability is CheckpointAvailability.FOREIGN
