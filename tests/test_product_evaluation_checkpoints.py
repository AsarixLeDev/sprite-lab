from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_core import ProductEvent, ProductStatus
from spritelab.product_features.evaluation import CheckpointAvailability, discover_checkpoint_candidates
from spritelab.product_features.training.service import _evaluation_checkpoint_binding
from spritelab.product_web.events import EventRepository
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.run_state import RunState


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


def _product_training_run(
    root: Path,
    run_id: str,
    *,
    feature: str = "training",
    backend_completion: bool = True,
    bind_checkpoints: bool = True,
) -> Path:
    repository = EventRepository(root / "runs", private_roots=(root,))
    repository.create_run(
        run_id,
        feature=feature,
        command="training.start",
        status=ProductStatus.RUNNING.value,
        backend_identity={
            "dataset_identity": "dataset-v1",
            "view_identity": "view-v1",
            "training_view_identity": "view-v1",
            "dataset_view_manifest_hash": "view-v1",
        },
    )
    run = root / "runs" / run_id
    checkpoint = run / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"authenticated product checkpoint")
    checkpoint_sha256 = sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_row = {
        "checkpoint": str(checkpoint),
        "seed": 1,
        "optimizer_step": 100,
        "sha256": checkpoint_sha256,
        "backend_id": "local",
        "remote": False,
        "downloaded": False,
        "hash_verified": True,
        "remote_identity_verified": False,
        "safe_resume": True,
        "synchronization": "local",
        "verification": "verified",
        "dataset_identity": "dataset-v1",
        "view_identity": "view-v1",
        "training_view_identity": "view-v1",
    }
    terminal_binding = _evaluation_checkpoint_binding(
        SimpleNamespace(
            run_id=run_id,
            dataset_identity="dataset-v1",
            view_identity="view-v1",
            dashboard=SimpleNamespace(
                checkpoints=[
                    SimpleNamespace(
                        checkpoint=str(checkpoint),
                        seed=1,
                        optimizer_step=100,
                        sha256=checkpoint_sha256,
                        backend_id="local",
                        remote=False,
                        downloaded=False,
                        hash_verified=True,
                        remote_identity_verified=False,
                        safe_resume=True,
                        verification="verified",
                    )
                ]
            ),
        ),
        root / "runs",
    )
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-12T10:00:00+00:00",
            feature=feature,
            stage="campaign",
            event_type="training_started" if feature == "training" else "evaluation_started",
            status=ProductStatus.RUNNING,
            current=0,
            total=100,
            message="Authoritative product run started.",
        )
    )
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-12T11:00:00+00:00",
            feature=feature,
            stage="training",
            event_type="checkpoint",
            status=ProductStatus.RUNNING if backend_completion else ProductStatus.COMPLETE,
            current=100,
            total=100,
            message="Verified checkpoint retained.",
            metrics={
                "checkpoint": str(checkpoint),
                "seed": 1,
                "optimizer_step": 100,
                "sha256": checkpoint_sha256,
                "downloaded": False,
                "hash_verified": True,
                "identity_verified": True,
            },
        )
    )
    if backend_completion:
        repository.append(
            ProductEvent(
                run_id=run_id,
                timestamp="2026-07-12T11:01:00+00:00",
                feature=feature,
                stage="campaign",
                event_type="backend_state",
                status=ProductStatus.COMPLETE,
                current=100,
                total=100,
                message="Backend jobs are complete.",
                metrics={
                    "completion_validated": True,
                    **({"evaluation_checkpoint_binding": terminal_binding} if bind_checkpoints else {}),
                },
            )
        )
    repository.update_state(
        run_id,
        checkpoints=[checkpoint_row],
        dataset_identity="dataset-v1",
        view_identity="view-v1",
        training_view_identity="view-v1",
    )
    return run


def test_no_checkpoint_has_no_eligible_candidate(tmp_path: Path) -> None:
    _run(tmp_path, "train-empty")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.eligible == ()
    assert catalog.default_checkpoint_id is None
    assert catalog.unavailable[0].availability is CheckpointAvailability.MISSING


def test_current_v3_writer_event_authority_remains_checkpoint_compatible(tmp_path: Path) -> None:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    values["paths"]["runs"] = "runs"
    config = ProjectConfig(root=tmp_path, path=tmp_path / "spritelab.yaml", values=values)
    run = RunState.create(config, command="train", argv=["train"], source_commit="abc", dry_run=False)
    run.finish(
        command="train",
        status="COMPLETE",
        exit_code=0,
        message="Training completed.",
        stage="campaign",
        backend_identity={"dataset_identity": "dataset-v1", "view_identity": "view-v1"},
    )
    checkpoint, _checkpoint_sha256 = _checkpoint_evidence(run.directory)

    catalog = discover_checkpoint_candidates(
        config.runs_dir,
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert len(catalog.eligible) == 1
    assert catalog.unavailable == ()
    assert catalog.eligible[0].path == checkpoint.resolve()


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


def test_origin_bound_product_training_state_is_eligible(tmp_path: Path) -> None:
    checkpoint = _product_training_run(tmp_path, "product-training") / "checkpoints" / "checkpoint.pt"

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="dataset-v1",
        active_view_identity="view-v1",
    )

    assert len(catalog.eligible) == 1
    assert catalog.unavailable == ()
    assert catalog.eligible[0].path == checkpoint.resolve()


@pytest.mark.parametrize(
    "downgrade_schema",
    [False, True],
    ids=["product-schema", "legacy-schema-downgrade"],
)
def test_post_completion_state_rewrite_cannot_select_attacker_checkpoint(
    tmp_path: Path,
    downgrade_schema: bool,
) -> None:
    run = _product_training_run(tmp_path, "product-training-rewritten")
    attacker = run / "checkpoints" / "attacker-selected.pt"
    attacker.write_bytes(b"attacker-selected checkpoint")
    attacker_sha256 = sha256(attacker.read_bytes()).hexdigest()
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    row = state["checkpoints"][0]
    row.update(
        checkpoint=str(attacker),
        sha256=attacker_sha256,
        dataset_identity="attacker-dataset",
        view_identity="attacker-view",
        training_view_identity="attacker-view",
    )
    state.update(
        dataset_identity="attacker-dataset",
        view_identity="attacker-view",
        training_view_identity="attacker-view",
    )
    state["backend_identity"].update(
        dataset_identity="attacker-dataset",
        view_identity="attacker-view",
        training_view_identity="attacker-view",
        dataset_view_manifest_hash="attacker-view",
    )
    if downgrade_schema:
        state["schema_version"] = "spritelab.v3.run-state.v1"
        state["command"] = "train"
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(
        tmp_path / "runs",
        project_root=tmp_path,
        active_dataset_identity="attacker-dataset",
        active_view_identity="attacker-view",
    )

    assert catalog.eligible == ()
    assert len(catalog.unavailable) == 1
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    expected_reason = "legacy run-state schema" if downgrade_schema else "authenticated terminal event"
    assert expected_reason in catalog.unavailable[0].unavailable_reasons[0]


def test_unbound_product_training_schema_is_not_authenticated(tmp_path: Path) -> None:
    run = _run(tmp_path, "unbound-product-training")
    _checkpoint_evidence(run)
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        schema_version="spritelab.product.run-state.v1",
        feature="training",
        command="training.start",
        event_migration_required=False,
    )
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "authentication" in catalog.unavailable[0].unavailable_reasons[0]


def test_authenticated_legacy_completion_without_checkpoint_binding_fails_closed(tmp_path: Path) -> None:
    _product_training_run(tmp_path, "legacy-unbound-completion", bind_checkpoints=False)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "authenticated terminal event" in catalog.unavailable[0].unavailable_reasons[0]


def test_origin_bound_foreign_product_state_is_not_training(tmp_path: Path) -> None:
    _product_training_run(tmp_path, "product-evaluation", feature="evaluation")

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "TrainingService" in catalog.unavailable[0].unavailable_reasons[0]


def test_state_feature_relabel_cannot_authenticate_foreign_event_history(tmp_path: Path) -> None:
    run = _product_training_run(tmp_path, "relabelled-evaluation", feature="evaluation")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["feature"] == "evaluation"
    state["feature"] = "training"
    _write_json(state_path, state)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "event semantics" in catalog.unavailable[0].unavailable_reasons[0]


def test_terminal_checkpoint_without_validated_backend_completion_is_not_authenticated(tmp_path: Path) -> None:
    _product_training_run(tmp_path, "terminal-checkpoint-only", backend_completion=False)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "event semantics" in catalog.unavailable[0].unavailable_reasons[0]


def test_latest_complete_checkpoint_is_default_and_public_view_redacts_paths(tmp_path: Path) -> None:
    older = _run(tmp_path, "older")
    newer = _run(tmp_path, "newer")
    state = json.loads((newer / "state.json").read_text(encoding="utf-8"))
    state["ended_at"] = "2026-07-13T11:00:00+00:00"
    _write_json(newer / "state.json", state)
    older_checkpoint = older / "checkpoints" / "checkpoint_step_000100.pt"
    live_checkpoint = newer / "checkpoints" / "checkpoint_step_000200.pt"
    ema_checkpoint = newer / "checkpoints" / "checkpoint_step_000200_ema.pt"
    older_checkpoint.write_bytes(b"old")
    live_checkpoint.write_bytes(b"new")
    ema_checkpoint.write_bytes(b"ema")
    for run, rows in (
        (
            older,
            [
                {
                    "path": "checkpoints/checkpoint_step_000100.pt",
                    "step": 100,
                    "weights": "live",
                    "sha256": sha256(older_checkpoint.read_bytes()).hexdigest(),
                }
            ],
        ),
        (
            newer,
            [
                {
                    "path": "checkpoints/checkpoint_step_000200.pt",
                    "step": 200,
                    "weights": "live",
                    "sha256": sha256(live_checkpoint.read_bytes()).hexdigest(),
                },
                {
                    "path": "checkpoints/checkpoint_step_000200_ema.pt",
                    "step": 200,
                    "weights": "ema",
                    "sha256": sha256(ema_checkpoint.read_bytes()).hexdigest(),
                },
            ],
        ),
    ):
        durable_state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        durable_state["checkpoints"] = rows
        _write_json(run / "state.json", durable_state)
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    default = catalog.find(None)
    assert default is not None
    assert default.run_id == "newer"
    assert default.weights == "ema"
    public = catalog.to_dict()
    assert public["label"] == "baseline — latest complete checkpoint"
    assert "checkpoint_path" not in json.dumps(public)
    assert catalog.find(default.checkpoint_id, weights="live").weights == "live"


def test_catalog_projection_sanitizes_state_derived_text_and_keeps_technical_rows_pathless(tmp_path: Path) -> None:
    run = _run(tmp_path, "hostile-catalog")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["backend_identity"].update(
        {
            "friendly_run_name": "Nightly Authorization=Bearer FRIENDLYSECRET C:\\private\\name.txt",
            "training_profile": "profile api_key=PROFILESECRET file:///home/alice/profile.json",
            "dataset_identity_summary": f"Dataset password=DATASECRET {tmp_path / 'private' / 'dataset.json'}",
            "view_identity_summary": "View client_secret=VIEWSECRET /home/alice/private/view.json",
        }
    )
    _write_json(state_path, state)
    _checkpoint_evidence(run)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    public = catalog.to_dict(include_unavailable=True, private_roots=(tmp_path,))
    serialized = json.dumps(public)

    for private_value in (
        "FRIENDLYSECRET",
        "PROFILESECRET",
        "DATASECRET",
        "VIEWSECRET",
        str(tmp_path),
        tmp_path.as_posix(),
        "C:\\private",
        "/home/alice/private",
        "file:///home/alice",
    ):
        assert private_value not in serialized
    candidate = public["eligible"][0]
    assert "[redacted]" in candidate["friendly_run_name"]
    assert "[redacted]" in candidate["training_profile"]
    assert "[redacted]" in candidate["dataset_identity_summary"]
    assert "[redacted]" in candidate["view_identity_summary"]
    assert candidate["checkpoint_step"] is None
    assert candidate["eligible"] is True

    technical = catalog.to_dict(
        include_unavailable=True,
        technical_details=True,
        private_roots=(tmp_path,),
    )
    technical_text = json.dumps(technical)
    technical_candidate = technical["eligible"][0]
    assert "checkpoint_path" not in technical_text
    assert "run_directory" not in technical_text
    assert str(tmp_path) not in technical_text
    assert technical_candidate["checkpoint_reference"] == "checkpoint.pt"
    assert technical_candidate["run_reference"] == "hostile-catalog"


def test_foreign_checkpoint_is_not_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "foreign", project_root=tmp_path / "somewhere-else")
    (run / "checkpoints" / "checkpoint_last.pt").write_bytes(b"checkpoint")
    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)
    assert catalog.unavailable[0].availability is CheckpointAvailability.FOREIGN


def test_schema_or_verdict_without_per_file_hash_is_never_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "verdict-only")
    checkpoint = run / "checkpoints" / "checkpoint.pt"
    checkpoint.write_bytes(b"verdict-only")
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    state["checkpoints"] = [{"path": "checkpoints/checkpoint.pt", "verified": True}]
    _write_json(run / "state.json", state)

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.UNVERIFIED
    assert "SHA-256" in catalog.unavailable[0].unavailable_reasons[0]


def test_hard_linked_checkpoint_is_never_eligible(tmp_path: Path) -> None:
    run = _run(tmp_path, "hard-linked")
    checkpoint, _checkpoint_sha256 = _checkpoint_evidence(run)
    outside_alias = tmp_path / "checkpoint-alias.pt"
    try:
        outside_alias.hardlink_to(checkpoint)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")

    catalog = discover_checkpoint_candidates(tmp_path / "runs", project_root=tmp_path)

    assert catalog.eligible == ()
    assert catalog.unavailable[0].availability is CheckpointAvailability.INVALID
    assert "hard-link" in catalog.unavailable[0].unavailable_reasons[0]
