from __future__ import annotations

import json
from pathlib import Path

import pytest

from spritelab.product_features.evaluation import (
    CheckpointAvailability,
    CheckpointCandidate,
    CheckpointCatalog,
    GeneratedAsset,
    GenerationCancelledError,
    GenerationRequest,
    GenerationSafetyError,
    PlaygroundService,
)


def _candidate(tmp_path: Path, *, weights: str) -> CheckpointCandidate:
    path = tmp_path / f"checkpoint_step_000100{'_ema' if weights == 'ema' else ''}.pt"
    path.write_bytes(weights.encode())
    return CheckpointCandidate(
        checkpoint_id=f"checkpoint-{weights}",
        run_id="run-1",
        friendly_run_name="baseline",
        date="2026-07-13T10:00:00+00:00",
        training_profile="standard",
        completion_state="COMPLETE",
        dataset_identity="dataset-v1",
        dataset_identity_summary="Dataset v1",
        view_identity="view-v1",
        view_identity_summary="View v1",
        checkpoint_step=100,
        weights=weights,
        verification_state="VERIFIED",
        availability=CheckpointAvailability.ELIGIBLE,
        path=path,
        run_directory=tmp_path,
    )


def _catalog(tmp_path: Path) -> CheckpointCatalog:
    live = _candidate(tmp_path, weights="live")
    ema = _candidate(tmp_path, weights="ema")
    return CheckpointCatalog((ema, live), (), ema.checkpoint_id)


class FakeGenerator:
    remote = False
    billable = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [GeneratedAsset(f"fake-{index}".encode()) for index in range(kwargs["image_count"])]


class BillableFakeGenerator(FakeGenerator):
    remote = True
    billable = True


def test_prompt_playground_defaults_are_sensible(tmp_path: Path) -> None:
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground")
    defaults = service.defaults()
    assert defaults["seed"] == 42
    assert defaults["sampling_steps"] == 30
    assert defaults["guidance"] == 3.0
    assert defaults["image_count"] == 4
    assert defaults["weights"] == "ema"


def test_generation_requires_explicit_action_and_does_not_call_fake(tmp_path: Path) -> None:
    fake = FakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=fake)
    request = GenerationRequest(prompt="red sword", checkpoint_id="checkpoint-ema", image_count=1)
    with pytest.raises(GenerationSafetyError, match="explicit"):
        service.generate(request, explicit_action=False)
    assert fake.calls == []


def test_fake_generation_records_reproducibility_and_exploratory_scope(tmp_path: Path) -> None:
    fake = FakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=fake)
    request = GenerationRequest(
        prompt="red sword",
        checkpoint_id="checkpoint-ema",
        seed=71,
        sampling_steps=18,
        guidance=2.5,
        image_count=2,
    )
    result = service.generate(request, explicit_action=True)
    assert len(fake.calls) == 1
    assert result["scope"] == "EXPLORATORY"
    assert result["excluded_from_frozen_benchmark"] is True
    assert result["excluded_from_promotion_evidence"] is True
    first = result["results"][0]
    assert first["checkpoint_identity"] == "checkpoint-ema"
    assert first["prompt"] == "red sword"
    assert first["seed"] == 71
    assert first["generation_parameters"] == {"sampling_steps": 18, "guidance": 2.5, "image_count": 2}
    assert len(first["output_hash"]) == 64
    assert first["timestamp"]
    assert first["application_version"]
    assert first["frozen_benchmark_eligible"] is False
    run_directory = tmp_path / "playground" / result["run_id"]
    assert {"state.json", "events.jsonl", "command.json", "logs", "artifacts", "report"} <= {
        path.name for path in run_directory.iterdir()
    }
    report = json.loads((run_directory / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"].endswith("v1")


def test_live_ema_selection_is_passed_to_fake_generator(tmp_path: Path) -> None:
    fake = FakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=fake)
    request = GenerationRequest(
        prompt="blue potion",
        checkpoint_id="checkpoint-ema",
        weights="live",
        image_count=1,
    )
    service.generate(request, explicit_action=True)
    assert fake.calls[0]["weights"] == "live"
    assert fake.calls[0]["checkpoint"].name == "checkpoint_step_000100.pt"


def test_billable_generation_requires_confirmation(tmp_path: Path) -> None:
    fake = BillableFakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=fake)
    request = GenerationRequest(prompt="green shield", checkpoint_id="checkpoint-ema", image_count=1)
    with pytest.raises(GenerationSafetyError, match="cost confirmation"):
        service.generate(request, explicit_action=True)
    assert fake.calls == []
    service.generate(request, explicit_action=True, confirm_billable=True)
    assert len(fake.calls) == 1


def test_saved_prompt_preset_can_be_rerun(tmp_path: Path) -> None:
    fake = FakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=fake)
    request = GenerationRequest(prompt="violet wand", checkpoint_id="checkpoint-ema", image_count=1)
    service.presets.save("wand", request)
    result = service.rerun("wand", explicit_action=True, seed=99)
    assert result["results"][0]["seed"] == 99
    assert service.presets.list()[0]["name"] == "wand"


def test_completed_run_reconstructs_after_service_restart_without_generation(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "v3"
    fake = FakeGenerator()
    first = PlaygroundService(
        _catalog(tmp_path), output_root=tmp_path / "playground", runs_directory=runs, generator=fake
    )
    created = first.generate(
        GenerationRequest(prompt="silver key", checkpoint_id="checkpoint-ema", image_count=2),
        explicit_action=True,
    )
    recreated = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", runs_directory=runs)
    restored = recreated.latest_run()
    assert restored is not None
    assert restored["run_id"] == created["run_id"]
    assert restored["status"] == "COMPLETE"
    assert restored["progress"] == {"current": 2, "total": 2}
    assert restored["results"] == created["results"]
    assert restored["report_available"] is True
    assert len(fake.calls) == 1


def test_partial_run_reconstructs_after_abrupt_service_loss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import spritelab.product_features.evaluation.playground as playground_module

    runs = tmp_path / "runs" / "v3"
    service = PlaygroundService(
        _catalog(tmp_path), output_root=tmp_path / "playground", runs_directory=runs, generator=FakeGenerator()
    )
    original = playground_module._asset_bytes
    calls = 0

    def interrupt_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(value)

    monkeypatch.setattr(playground_module, "_asset_bytes", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        service.generate(
            GenerationRequest(prompt="partial shield", checkpoint_id="checkpoint-ema", image_count=2),
            explicit_action=True,
        )
    recreated = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", runs_directory=runs)
    restored = recreated.latest_run()
    assert restored is not None
    assert restored["status"] == "RUNNING"
    assert restored["stage"] == "image_completed"
    assert restored["progress"] == {"current": 1, "total": 2}
    assert len(restored["results"]) == 1
    assert restored["report_available"] is False


def test_failed_and_cancelled_runs_reconstruct_durable_terminal_state(tmp_path: Path) -> None:
    class FailingGenerator(FakeGenerator):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("synthetic adapter failure")

    class CancellingGenerator(FakeGenerator):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            raise GenerationCancelledError("synthetic cancellation")

    request = GenerationRequest(prompt="terminal wand", checkpoint_id="checkpoint-ema", image_count=1)
    failed = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "failed", generator=FailingGenerator())
    with pytest.raises(RuntimeError, match="synthetic adapter failure"):
        failed.generate(request, explicit_action=True)
    failed_run = failed.latest_run()
    assert failed_run is not None and failed_run["status"] == "FAILED"
    assert failed_run["failure"]["type"] == "RuntimeError"

    cancelled = PlaygroundService(
        _catalog(tmp_path), output_root=tmp_path / "cancelled", generator=CancellingGenerator()
    )
    with pytest.raises(GenerationCancelledError):
        cancelled.generate(request, explicit_action=True)
    cancelled_run = cancelled.latest_run()
    assert cancelled_run is not None and cancelled_run["status"] == "CANCELLED"
    assert cancelled_run["cancellation"]["reason"] == "synthetic cancellation"


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_missing_or_changed_image_is_stale_without_regeneration(tmp_path: Path, mutation: str) -> None:
    fake = FakeGenerator()
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / mutation, generator=fake)
    created = service.generate(
        GenerationRequest(prompt="stale potion", checkpoint_id="checkpoint-ema", image_count=1),
        explicit_action=True,
    )
    artifact = tmp_path / mutation / created["run_id"] / created["results"][0]["output_reference"]
    if mutation == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed-image-bytes")
    restored = service.reconstruct(created["run_id"])
    assert restored["status"] == "STALE"
    assert restored["comparability"] == "STALE"
    assert len(fake.calls) == 1


def test_changed_metadata_duplicate_events_and_malformed_state_fail_closed(tmp_path: Path) -> None:
    service = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", generator=FakeGenerator())
    created = service.generate(
        GenerationRequest(prompt="metadata sword", checkpoint_id="checkpoint-ema", image_count=1),
        explicit_action=True,
    )
    directory = tmp_path / "playground" / created["run_id"]
    events = directory / "events.jsonl"
    before = service.reconstruct(created["run_id"])["event_count"]
    last = events.read_text(encoding="utf-8").splitlines()[-1]
    with events.open("a", encoding="utf-8") as handle:
        handle.write(last + "\n")
    assert service.reconstruct(created["run_id"])["event_count"] == before

    command = json.loads((directory / "command.json").read_text(encoding="utf-8"))
    command["request"]["prompt"] = "modified prompt"
    (directory / "command.json").write_text(json.dumps(command), encoding="utf-8")
    assert service.reconstruct(created["run_id"])["status"] == "NOT_COMPARABLE"

    (directory / "state.json").write_text("{", encoding="utf-8")
    malformed = service.reconstruct(created["run_id"])
    assert malformed["status"] == "NOT_COMPARABLE"
    assert malformed["durable"] is False


def test_legacy_metadata_is_read_only_not_comparable_and_never_promotion_evidence(tmp_path: Path) -> None:
    root = tmp_path / "playground"
    legacy = root / "generations" / "legacy-one"
    legacy.mkdir(parents=True)
    (legacy / "metadata.json").write_text(
        json.dumps({"generation_id": "legacy-one", "results": [{"output_hash": "a" * 64}]}),
        encoding="utf-8",
    )
    service = PlaygroundService(_catalog(tmp_path), output_root=root)
    rows = service.legacy_generations()
    assert rows[0]["status"] == "NOT_COMPARABLE"
    assert rows[0]["durable"] is False
    assert rows[0]["frozen_benchmark_eligible"] is False
    assert rows[0]["promotion_evidence_eligible"] is False
    assert service.latest_run() is None


@pytest.mark.parametrize(
    "interruption_step",
    [
        "run_directory_created",
        "command_written",
        "events_created",
        "planned_event_appended",
        "before_state",
        "state_written",
    ],
)
def test_playground_initialization_is_interruption_safe_at_every_publication_step(
    tmp_path: Path, interruption_step: str
) -> None:
    runs = tmp_path / "runs" / "v3"
    run_id = f"playground-interrupt-{interruption_step.replace('_', '-')}"
    fake = FakeGenerator()

    def interrupt(step: str, _directory: Path) -> None:
        if step == interruption_step:
            raise RuntimeError(f"interrupted at {step}")

    service = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=fake,
        run_id_factory=lambda: run_id,
        initialization_hook=interrupt,
    )
    request = GenerationRequest(prompt="interruptible wand", checkpoint_id="checkpoint-ema", image_count=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        service.generate(request, explicit_action=True)

    directory = runs / run_id
    state_exists = (directory / "state.json").is_file()
    assert state_exists is (interruption_step == "state_written")
    assert fake.calls == []
    restored = service.reconstruct(run_id)
    if state_exists:
        assert restored["status"] == "RUNNING"
        assert restored["stage"] == "planned"
        assert restored["authoritative"] is True
        assert (directory / "events.jsonl").is_file()
        events = service.repository.replay(run_id).events
        assert [item.event.event_type for item in events] == ["planned"]
        assert {"logs", "artifacts", "report"} <= {path.name for path in directory.iterdir() if path.is_dir()}
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        assert state["exploratory"] is True
        assert state["benchmark_eligible"] is False
        assert state["promotion_evidence_eligible"] is False
    else:
        assert restored["status"] == "initialization_incomplete"
        assert restored["authoritative"] is False
        assert restored["resumable"] is False
        assert service.latest_run() is None

    restarted = PlaygroundService(_catalog(tmp_path), output_root=tmp_path / "playground", runs_directory=runs)
    assert fake.calls == []
    if state_exists:
        assert restarted.latest_run()["stage"] == "planned"  # type: ignore[index]
        assert restarted.startup_incomplete_run_ids == ()
    else:
        assert restarted.latest_run() is None
        assert restarted.startup_incomplete_run_ids == (run_id,)


def test_incomplete_initialization_retry_is_collision_safe_and_does_not_duplicate_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "v3"
    run_id = "playground-fixed-retry"
    request = GenerationRequest(prompt="retry shield", checkpoint_id="checkpoint-ema", image_count=1)

    def interrupt(step: str, _directory: Path) -> None:
        if step == "command_written":
            raise RuntimeError("synthetic initialization interruption")

    first = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=FakeGenerator(),
        run_id_factory=lambda: run_id,
        initialization_hook=interrupt,
    )
    with pytest.raises(RuntimeError):
        first.generate(request, explicit_action=True)

    retry_generator = FakeGenerator()
    retry = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=retry_generator,
        run_id_factory=lambda: run_id,
    )
    with pytest.raises(FileExistsError):
        retry.generate(request, explicit_action=True)
    assert retry_generator.calls == []
    assert [path.name for path in runs.iterdir() if path.is_dir()] == [run_id]


def test_complete_planned_run_restarts_without_generation_and_allows_explicit_continuation(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "v3"
    run_id = "playground-planned-continuation"
    request = GenerationRequest(prompt="continued potion", checkpoint_id="checkpoint-ema", image_count=1)

    def interrupt_after_state(step: str, _directory: Path) -> None:
        if step == "state_written":
            raise RuntimeError("process stopped after state publication")

    first_generator = FakeGenerator()
    first = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=first_generator,
        run_id_factory=lambda: run_id,
        initialization_hook=interrupt_after_state,
    )
    with pytest.raises(RuntimeError):
        first.generate(request, explicit_action=True)
    assert first_generator.calls == []

    continuation_generator = FakeGenerator()
    restarted = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=continuation_generator,
    )
    planned = restarted.reconstruct(run_id)
    assert planned["status"] == "RUNNING" and planned["stage"] == "planned"
    assert continuation_generator.calls == []

    completed = restarted.continue_run(run_id, explicit_action=True)

    assert completed["status"] == "COMPLETE"
    assert len(continuation_generator.calls) == 1
    event_types = [item.event.event_type for item in restarted.repository.replay(run_id).events]
    assert event_types.count("planned") == 1


def test_authoritative_playground_state_missing_events_is_not_comparable(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "v3"
    run_id = "playground-missing-events"

    def interrupt_after_state(step: str, _directory: Path) -> None:
        if step == "state_written":
            raise RuntimeError("stop")

    service = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        runs_directory=runs,
        generator=FakeGenerator(),
        run_id_factory=lambda: run_id,
        initialization_hook=interrupt_after_state,
    )
    with pytest.raises(RuntimeError):
        service.generate(
            GenerationRequest(prompt="missing events", checkpoint_id="checkpoint-ema", image_count=1),
            explicit_action=True,
        )
    (runs / run_id / "events.jsonl").unlink()

    restored = service.reconstruct(run_id)

    assert restored["status"] == "NOT_COMPARABLE"
    assert any("events.jsonl is missing" in reason for reason in restored["integrity_reasons"])
    assert restored["benchmark_eligible"] is False
    assert restored["promotion_evidence_eligible"] is False
