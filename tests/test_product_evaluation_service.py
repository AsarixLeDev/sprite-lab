from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import ProjectContext
from spritelab.product_core.web import create_product_app
from spritelab.product_features.evaluation import EvaluationRequest, EvaluationService, build_plugin
from spritelab.product_features.evaluation import plugin as evaluation_plugin
from spritelab.product_features.evaluation import service as evaluation_service_module
from spritelab.product_features.evaluation.web import create_evaluation_router
from spritelab.v3 import cli as product_cli


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _project(tmp_path: Path) -> tuple[dict, Path]:
    run = tmp_path / "runs" / "train-1"
    checkpoint = run / "checkpoints" / "checkpoint_step_000100_ema.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        run / "state.json",
        {
            "schema_version": "spritelab.v3.run-state.v1",
            "run_id": "train-1",
            "command": "train",
            "status": "COMPLETE",
            "started_at": "2026-07-13T10:00:00+00:00",
            "backend_identity": {"dataset_identity": "dataset-v1", "view_identity": "view-v1"},
            "checkpoints": [
                {
                    "path": "checkpoints/checkpoint_step_000100_ema.pt",
                    "step": 100,
                    "weights": "ema",
                    "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    _write_json(run / "command.json", {"command": "train", "project_root": str(tmp_path)})
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"prompt_id": "p1", "prompt": "red sword"}) + "\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    _write_json(audit, {"verdict": "FAIL", "authorization": {"checkpoint_promotion": False}})
    return {
        "paths": {"runs": str(tmp_path / "runs")},
        "dataset": {"identity": "dataset-v1", "view_identity": "view-v1"},
        "evaluation": {
            "benchmark": str(benchmark),
            "memorization_audit": str(audit),
            "review_log": "",
        },
    }, benchmark


class FakeEvaluationGenerator:
    remote = False
    billable = False

    def __init__(self) -> None:
        self.calls = 0

    def generate_benchmark(self, *, output_directory: Path, emit, **_kwargs) -> Path:
        self.calls += 1
        emit("generation", 1, 1, "Generated one fake sample.")
        return output_directory


class FakePlaygroundGenerator:
    remote = False
    billable = False

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return [b"fake"] * kwargs["image_count"]


class CapturingDefaultEvaluationService(EvaluationService):
    authored_context: tuple[str | None, str | None] | None = None

    def _existing_evaluator(
        self,
        generated: Path,
        out: Path,
        *,
        checkpoint: Path,
        benchmark: Path,
        training_dataset_identity=None,
        training_view_identity=None,
    ) -> dict:
        del generated, checkpoint, benchmark
        self.authored_context = (training_dataset_identity, training_view_identity)
        return _fake_evaluator()(Path(), out)


def _fake_evaluator(*, hard: int = 0, review: int = 0):
    def evaluate(_generated: Path, out: Path) -> dict:
        out.mkdir(parents=True)
        row = {
            "sample_id": "sample-1",
            "prompt_id": "p1",
            "prompt": "red sword",
            "seed": 7,
            "category": "weapon",
            "metrics": {"pixel_art": {"unique_palette_size": 8, "silhouette_occupancy": 0.4}},
            "conditional_adherence": 1.0,
            "memorization_evidence_class": "exact_rgba_nontrivial" if hard else "no_material_match",
        }
        (out / "per_image_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {
            "schema_version": "generation_benchmark_v1.0",
            "summary": {
                "sample_count": 1,
                "hard_validity": {"pass_rate": 1.0},
                "conditional": {"represented_rate": 1.0},
                "pixel_art": {"palette_size_mean": 8.0},
                "diversity": {"exact_duplicate_rate": 0.0},
                "memorization": {"hard_evidence_count": hard, "review_required_count": review},
            },
            "promotion": {"memorization_machine_status": "hard_fail" if hard else "pass"},
        }

    return evaluate


def test_benchmark_missing_blocks_without_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    config["evaluation"]["benchmark"] = str(tmp_path / "missing.jsonl")
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())
    result = service.run(EvaluationRequest(dry_run=True))
    assert result.status.value == "BLOCKED"
    assert fake.calls == 0


def test_missing_active_training_view_blocks_service_before_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    config["dataset"].pop("view_identity")
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.status.value == "BLOCKED"
    assert any("dataset and view identities" in item.message for item in result.blockers)
    assert result.data["generation_runs"] == 0
    assert result.data["promotion_actions"] == 0
    assert fake.calls == 0


def test_missing_active_training_view_blocks_api_before_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    config["dataset"].pop("view_identity")
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))

    response = TestClient(app).post("/evaluation/api/run", json={"explicit_action": True})

    assert response.status_code == 409
    assert response.json()["error_code"] == "evaluation_run_blocked"
    assert response.json()["recoverable"] is True
    assert fake.calls == 0


@pytest.mark.parametrize(
    ("identity", "value"),
    (
        ("dataset", ["not", "a", "string"]),
        ("dataset", " padded-dataset "),
        ("dataset", ""),
        ("view", {"not": "a string"}),
        ("view", " padded-view "),
        ("view", ""),
    ),
)
def test_malformed_product_training_identity_blocks_before_generation(
    tmp_path: Path,
    identity: str,
    value: object,
) -> None:
    config, _benchmark = _project(tmp_path)
    if identity == "dataset":
        config["dataset"].pop("identity")
        config["evaluation"]["dataset_identity"] = value
    else:
        config["dataset"].pop("view_identity")
        config["evaluation"]["training_view_identity"] = value
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.status.value == "BLOCKED"
    assert result.data["generation_runs"] == 0
    assert result.data["promotion_actions"] == 0
    assert fake.calls == 0


_INVALID_BENCHMARK_CONTENTS = (
    pytest.param("bench.json", '{"prompts": [NaN]}', id="json-nan"),
    pytest.param("bench.json", '{"prompts": [Infinity]}', id="json-infinity"),
    pytest.param("bench.json", '{"prompts": [-Infinity]}', id="json-negative-infinity"),
    pytest.param("bench.json", '{"prompts": [', id="json-malformed"),
    pytest.param("bench.json", '["not-an-object"]', id="json-wrong-schema"),
    pytest.param(
        "bench.jsonl",
        '{"prompt_id": "p1", "metric": NaN}\n{"prompt_id": "p2"}\n{"prompt_id": "p3"}\n',
        id="jsonl-invalid-first-row",
    ),
    pytest.param(
        "bench.jsonl",
        '{"prompt_id": "p1"}\n{"prompt_id": "p2", "metric": Infinity}\n{"prompt_id": "p3"}\n',
        id="jsonl-invalid-middle-row",
    ),
    pytest.param(
        "bench.jsonl",
        '{"prompt_id": "p1"}\n{"prompt_id": "p2"}\n{"prompt_id": "p3", "metric": -Infinity}\n',
        id="jsonl-invalid-final-row",
    ),
    pytest.param("bench.jsonl", '{"prompt_id": "p1"}\nnot json\n', id="jsonl-malformed-row"),
    pytest.param("bench.jsonl", '["not-an-object"]\n', id="jsonl-wrong-schema"),
)


@pytest.mark.parametrize(("filename", "content"), _INVALID_BENCHMARK_CONTENTS)
def test_nonfinite_and_malformed_benchmarks_return_controlled_results_without_generation(
    tmp_path: Path, filename: str, content: str
) -> None:
    config, _benchmark = _project(tmp_path)
    bad_benchmark = tmp_path / filename
    bad_benchmark.write_text(content, encoding="utf-8")
    config["evaluation"]["benchmark"] = str(bad_benchmark)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.status.value == "BLOCKED"
    blocker_messages = " ".join(item.message for item in result.blockers)
    assert "benchmark" in blocker_messages.casefold()
    assert "Traceback" not in blocker_messages
    assert fake.calls == 0
    assert result.data["generation_runs"] == 0
    assert result.data["promotion_actions"] == 0


def test_nonfinite_benchmark_returns_controlled_api_error_without_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    bad_benchmark = tmp_path / "api-nonfinite.json"
    bad_benchmark.write_text('{"prompts": [NaN]}', encoding="utf-8")
    config["evaluation"]["benchmark"] = str(bad_benchmark)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))

    response = TestClient(app).post("/evaluation/api/run", json={"explicit_action": True})

    assert response.status_code == 409
    assert response.json()["error_code"] == "evaluation_run_blocked"
    assert response.json()["recoverable"] is True
    assert "traceback" not in response.text.casefold()
    assert fake.calls == 0


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        pytest.param("good.json", '{"prompts": ["red sword"]}', id="valid-json"),
        pytest.param(
            "good.jsonl",
            '{"prompt_id": "p1", "prompt": "red sword"}\n{"prompt_id": "p2", "prompt": "blue shield"}\n',
            id="valid-jsonl",
        ),
    ],
)
def test_valid_json_and_jsonl_benchmarks_pass_validation(tmp_path: Path, filename: str, content: str) -> None:
    config, _benchmark = _project(tmp_path)
    good_benchmark = tmp_path / filename
    good_benchmark.write_text(content, encoding="utf-8")
    config["evaluation"]["benchmark"] = str(good_benchmark)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())

    result = service.run(EvaluationRequest(dry_run=True))

    assert result.status.value == "COMPLETE"
    benchmark_stage = next(stage for stage in result.data["stages"] if stage["key"] == "benchmark_validation")
    assert benchmark_stage["status"] == "COMPLETE"
    assert fake.calls == 0
    assert result.data["promotion_actions"] == 0


def test_benchmark_changed_to_nan_after_planning_blocks_before_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, benchmark = _project(tmp_path)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=fake,
        evaluator=_fake_evaluator(),
        output_root=tmp_path / "evaluation-output",
    )
    create_run = service.events.create_run

    def create_then_mutate(*args, **kwargs):
        value = create_run(*args, **kwargs)
        benchmark.write_text('{"prompts": [NaN]}', encoding="utf-8")
        return value

    monkeypatch.setattr(service.events, "create_run", create_then_mutate)

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.status.value == "FAILED"
    assert result.data["generation_runs"] == 0
    assert fake.calls == 0
    assert any("Benchmark changed after planning" in blocker.message for blocker in result.blockers)
    assert result.data["promotion_actions"] == 0


def test_evaluation_dry_run_reports_stage_progress_without_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())
    result = service.run(EvaluationRequest(dry_run=True))
    assert result.status.value == "COMPLETE"
    assert result.data["progress"] == {"completed": 2, "total": 10}
    assert result.data["stages"][0]["status"] == "COMPLETE"
    assert result.data["stages"][-1]["status"] == "BLOCKED"
    assert result.data["generation_runs"] == 0
    assert result.data["promotion_actions"] == 0
    assert fake.calls == 0


def test_fake_evaluation_runs_all_metric_stages_and_never_promotes(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=fake,
        evaluator=_fake_evaluator(),
        output_root=tmp_path / "evaluation-output",
    )
    result = service.run(EvaluationRequest(explicit_action=True))
    assert result.status.value == "BLOCKED"
    assert "incomplete" in result.message.casefold()
    assert fake.calls == 1
    assert result.data["dashboard"]["charts"][0]["status"] == "AVAILABLE"
    assert result.data["promotion"]["promotion_authorized"] is False
    assert result.data["promotion_actions"] == 0


def test_hard_memorization_blocks_final_status(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    service = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(hard=1),
        output_root=tmp_path / "evaluation-output",
    )
    result = service.run(EvaluationRequest(explicit_action=True))
    assert result.status.value == "BLOCKED"
    assert result.data["promotion_actions"] == 0


def test_opening_web_page_does_not_generate_and_exposes_user_controls(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    fake = FakePlaygroundGenerator()
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, playground_generator=fake))
    client = TestClient(app)
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert fake.calls == 0
    assert "Start evaluation" in response.text
    assert "EXPLORATORY" in response.text
    assert "Promotion integrity is not currently certified." in response.text
    assert "Sampling steps" in response.text
    assert "CFG / guidance" in response.text


def test_plugin_contract_registers_eval_without_promotion_action(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    config["dataset"].pop("view_identity", None)
    config["evaluation"].pop("training_view_identity", None)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    plugin = build_plugin()
    assert plugin.plugin_id == "evaluation.playground"
    assert plugin.cli_registration.__name__ == "register_cli"
    assert plugin.navigation[0].path == "/evaluation"
    status = plugin.status_provider(context)
    assert status.status.value == "BLOCKED"
    assert status.data["training_identity"]["bound"] is False
    assert status.data["promotion"]["actions"] == []
    capabilities = {item.capability_id: item for item in plugin.capability_probe(context)}
    assert capabilities["evaluation.checkpoint_selection"].status.value == "BLOCKED"
    assert capabilities["evaluation.promotion_display"].status.value == "BLOCKED"


def test_plugin_readiness_binds_active_training_dataset_and_view(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    state_path = tmp_path / "runs" / "train-1" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["backend_identity"]["view_identity"] = "training-view-v1"
    _write_json(state_path, state)
    config["evaluation"].update(
        {
            "dataset_identity": "dataset-v1",
            "training_view_identity": "training-view-v1",
        }
    )
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")

    plugin = build_plugin()
    status = plugin.status_provider(context)
    capabilities = {item.capability_id: item for item in plugin.capability_probe(context)}

    assert status.status.value == "READY"
    assert status.data["training_identity"] == {
        "dataset_identity": "dataset-v1",
        "view_identity": "training-view-v1",
        "bound": True,
    }
    assert capabilities["evaluation.checkpoint_selection"].status.value == "READY"

    config["evaluation"]["training_view_identity"] = "different-training-view"
    stale = plugin.status_provider(context)
    assert stale.status.value == "BLOCKED"
    assert stale.data["training_identity"]["bound"] is False


def test_composed_normal_v3_eval_cli_uses_plugin_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, benchmark = _project(tmp_path)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    monkeypatch.setattr(evaluation_plugin, "_project_context", lambda: context)
    with pytest.raises(SystemExit) as caught:
        product_cli.main(
            ["eval", "--dry-run", "--benchmark", str(benchmark), "--json"],
            plugins=[build_plugin()],
        )
    payload = json.loads(capsys.readouterr().out)
    assert caught.value.code == 0
    assert payload["status"] == "COMPLETE"
    assert payload["data"]["product_result"]["data"]["promotion_actions"] == 0


def test_incomplete_evaluation_reconstructs_without_recomputation_and_changes_become_stale(
    tmp_path: Path,
) -> None:
    config, _benchmark = _project(tmp_path)
    generator = FakeEvaluationGenerator()
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=generator,
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.status.value == "BLOCKED"
    assert result.data["memorization"]["evidence_state"] == "incomplete"
    assert generator.calls == 1
    run_id = result.run.run_id  # type: ignore[union-attr]
    state = first.events.state(run_id)
    assert state["evaluation_schema_version"] == "spritelab.product.evaluation-state.v2"
    assert state["training_dataset_identity"] == "dataset-v1"
    assert state["training_view_identity"] == "view-v1"

    def forbidden_evaluator(*_args, **_kwargs):
        raise AssertionError("restart must not recompute evaluation metrics")

    recreated = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=generator,
        evaluator=forbidden_evaluator,
        output_root=output,
    )
    assert recreated.latest_run_id == run_id
    assert recreated.latest_status == "BLOCKED"
    assert recreated.latest_report is not None
    assert generator.calls == 1
    assert recreated.dashboard()["stale"] is False
    assert recreated.dashboard()["memorization"]["evidence_state"] == "incomplete"

    generated = output / run_id / "generated"
    generated.rmdir()
    stale = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=generator,
        evaluator=forbidden_evaluator,
        output_root=output,
    )
    assert stale.latest_status == "STALE"
    assert stale.dashboard()["stale"] is True
    assert generator.calls == 1


def test_default_candidate_authoring_uses_context_persisted_before_generation(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    service = CapturingDefaultEvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        output_root=tmp_path / "evaluation-output",
    )

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.run is not None
    state = service.events.state(result.run.run_id)
    assert service.authored_context == ("dataset-v1", "view-v1")
    assert service.authored_context == (
        state["training_dataset_identity"],
        state["training_view_identity"],
    )


@pytest.mark.parametrize(
    ("section", "key", "replacement", "reason"),
    (
        ("dataset", "identity", "dataset-v2", "active training dataset identity changed"),
        ("dataset", "view_identity", "view-v2", "active training view identity changed"),
    ),
)
def test_reconstruction_fails_closed_when_active_training_context_drifts(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: str,
    reason: str,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None

    changed_config = deepcopy(config)
    changed_config[section][key] = replacement
    restored = EvaluationService(
        project_root=tmp_path,
        config=changed_config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    state = restored.events.state(result.run.run_id)
    assert any(reason in item for item in state["stale_reasons"])


def test_reconstruction_fails_closed_when_checkpoint_view_provenance_drifts(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None

    training_state_path = tmp_path / "runs" / "train-1" / "state.json"
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    training_state["backend_identity"]["view_identity"] = "view-v2"
    _write_json(training_state_path, training_state)
    config_without_active_view = deepcopy(config)
    del config_without_active_view["dataset"]["view_identity"]

    restored = EvaluationService(
        project_root=tmp_path,
        config=config_without_active_view,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    state = restored.events.state(result.run.run_id)
    assert "checkpoint training view identity changed after evaluation" in state["stale_reasons"]


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("status", "RUNNING"),
        ("schema_version", "unsupported.training-state.v0"),
        ("unsafe_resume_record", {"reason": "synthetic revocation"}),
        ("command", "foreign-command"),
    ),
)
def test_reconstruction_fails_closed_when_checkpoint_loses_eligibility(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None

    training_state_path = tmp_path / "runs" / "train-1" / "state.json"
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    if mutation == "unsafe_resume_record":
        training_state["backend_identity"][mutation] = value
    else:
        training_state[mutation] = value
    _write_json(training_state_path, training_state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    state = restored.events.state(result.run.run_id)
    assert "checkpoint is no longer eligible under the persisted dataset/view identity" in state["stale_reasons"]


@pytest.mark.parametrize("field", ("training_dataset_identity", "training_view_identity"))
def test_reconstruction_fails_closed_when_persisted_training_identity_is_deleted(
    tmp_path: Path,
    field: str,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None
    state_path = output / result.run.run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state[field]
    _write_json(state_path, state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    stale_state = restored.events.state(result.run.run_id)
    assert f"persisted {field} is missing" in stale_state["stale_reasons"]


@pytest.mark.parametrize("field", ("training_dataset_identity", "training_view_identity"))
@pytest.mark.parametrize("value", (None, "", 7, True, ["identity"], " padded-identity "))
def test_reconstruction_fails_closed_when_persisted_training_identity_is_malformed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None
    state_path = output / result.run.run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    _write_json(state_path, state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    stale_state = restored.events.state(result.run.run_id)
    assert f"persisted {field} is malformed" in stale_state["stale_reasons"]


def test_reconstruction_validates_candidate_against_persisted_training_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None
    run_id = result.run.run_id
    candidate = output / run_id / "metrics" / "candidate_evidence.json"
    candidate.write_text("{}\n", encoding="utf-8")
    state = first.events.state(run_id)
    expected_artifacts = dict(state["expected_artifacts"])
    expected_artifacts["candidate_evidence"] = {
        "path": "metrics/candidate_evidence.json",
        "kind": "file",
        "sha256": sha256(candidate.read_bytes()).hexdigest(),
    }
    first.events.update_state(run_id, expected_artifacts=expected_artifacts)
    observed_context: dict = {}

    def reject_candidate(_path: Path, *, expected_context):
        observed_context.update(expected_context)
        return SimpleNamespace(valid=False, reasons=("training view identity mismatch",))

    monkeypatch.setattr(evaluation_service_module, "load_candidate_bundle", reject_candidate)
    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    assert observed_context["training_dataset_identity"] == "dataset-v1"
    assert observed_context["training_view_identity"] == "view-v1"
    stale_state = restored.events.state(run_id)
    assert "candidate evidence identity invalid: training view identity mismatch" in stale_state["stale_reasons"]


@pytest.mark.parametrize(
    ("section", "key", "reason"),
    (
        ("dataset", "identity", "active training dataset identity is missing or malformed"),
        ("dataset", "view_identity", "active training view identity is missing or malformed"),
    ),
)
def test_reconstruction_fails_closed_when_active_training_identity_is_deleted(
    tmp_path: Path,
    section: str,
    key: str,
    reason: str,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None

    changed_config = deepcopy(config)
    del changed_config[section][key]
    restored = EvaluationService(
        project_root=tmp_path,
        config=changed_config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    state = restored.events.state(result.run.run_id)
    assert reason in state["stale_reasons"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_path", "persisted checkpoint_path is missing or malformed"),
        ("missing_hash", "persisted checkpoint_sha256 is missing or malformed"),
        ("missing_expected_record", "expected checkpoint identity record is missing or malformed"),
        ("changed_bytes", "persisted checkpoint SHA-256 changed after evaluation"),
        ("rewritten_expected_hash_only", "persisted checkpoint SHA-256 changed after evaluation"),
    ),
)
def test_reconstruction_enforces_checkpoint_launch_identity(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None
    state_path = output / result.run.run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "missing_path":
        del state["checkpoint_path"]
    elif mutation == "missing_hash":
        del state["checkpoint_sha256"]
    elif mutation == "missing_expected_record":
        del state["expected_artifacts"]["checkpoint"]
    else:
        checkpoint = Path(state["checkpoint_path"])
        checkpoint.write_bytes(b"changed checkpoint bytes")
        if mutation == "rewritten_expected_hash_only":
            state["expected_artifacts"]["checkpoint"]["sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    _write_json(state_path, state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    stale_state = restored.events.state(result.run.run_id)
    assert reason in stale_state["stale_reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("current", "not-an-integer"),
        ("total", {"invalid": True}),
        ("metrics", ["not", "a", "mapping"]),
        ("status", 7),
    ),
)
def test_malformed_persisted_evaluation_stage_is_controlled_stale(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    first = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=output,
    )
    result = first.run(EvaluationRequest(explicit_action=True))
    assert result.run is not None
    state_path = output / result.run.run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"][0][field] = value
    _write_json(state_path, state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    stale_state = restored.events.state(result.run.run_id)
    assert "persisted evaluation stages are malformed" in stale_state["stale_reasons"]
    assert all(stage.status == "BLOCKED" for stage in restored.latest_stages[:3])


def test_malformed_durable_evaluation_state_does_not_crash_page_reconstruction(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    output = tmp_path / "evaluation-output"
    malformed = output / "evaluation-malformed"
    malformed.mkdir(parents=True)
    (malformed / "state.json").write_text("{not-json", encoding="utf-8")
    service = EvaluationService(project_root=tmp_path, config=config, output_root=output)
    assert service.latest_run_id is None
    assert service.dashboard()["status"] == "NOT_STARTED"
