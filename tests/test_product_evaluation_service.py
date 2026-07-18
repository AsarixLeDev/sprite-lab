from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_core.web import create_product_app
from spritelab.product_features.evaluation import (
    EvaluationRequest,
    EvaluationService,
    GenerationCancelledError,
    build_plugin,
)
from spritelab.product_features.evaluation import plugin as evaluation_plugin
from spritelab.product_features.evaluation import service as evaluation_service_module
from spritelab.product_features.evaluation.playground import GenerationTimedOutError
from spritelab.product_features.evaluation.web import create_evaluation_router
from spritelab.utils.safe_fs import UnsafeFilesystemOperation
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


def test_benchmark_duplicate_authority_key_is_rejected(tmp_path: Path) -> None:
    config, benchmark = _project(tmp_path)
    benchmark.write_text('{"prompt_id":"p1","prompt_id":"p2","prompt":"sword"}\n', encoding="utf-8")
    service = EvaluationService(project_root=tmp_path, config=config, evaluator=_fake_evaluator())

    valid, message = service._validate_benchmark(benchmark)

    assert valid is False
    assert "malformed" in message.lower()


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
    ("flag", "hostile"),
    [
        ("dry_run", "false"),
        ("explicit_action", "false"),
        ("confirm_billable", "false"),
        ("allow_source_results", "false"),
        ("explicit_action", 1),
        ("confirm_billable", [False]),
    ],
)
def test_evaluation_api_requires_exact_json_booleans(
    tmp_path: Path,
    flag: str,
    hostile: object,
) -> None:
    config, _benchmark = _project(tmp_path)
    fake = FakeEvaluationGenerator()
    service = EvaluationService(project_root=tmp_path, config=config, generator=fake, evaluator=_fake_evaluator())
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))
    payload: dict[str, object] = {"explicit_action": True, flag: hostile}

    response = TestClient(app).post("/evaluation/api/run", json=payload)

    assert response.status_code == 422
    assert response.json()["error_code"] == "evaluation_boolean_invalid"
    assert fake.calls == 0


@pytest.mark.parametrize("flag", ["dry_run", "explicit_action", "confirm_billable", "allow_source_results"])
def test_evaluation_request_rejects_non_boolean_flags_before_service_use(flag: str) -> None:
    with pytest.raises(ValueError, match="exact boolean"):
        EvaluationRequest(**{flag: "false"})


def test_ordinary_checkpoint_api_ignores_technical_query_and_remains_pathless(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    service = EvaluationService(project_root=tmp_path, config=config)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))
    client = TestClient(app)

    ordinary = client.get("/evaluation/api/checkpoints?include_unavailable=true&technical_details=true")
    technical = client.get("/evaluation/api/technical/checkpoints?acknowledge=true")

    assert ordinary.status_code == 200
    assert "checkpoint_path" not in ordinary.text
    assert "run_directory" not in ordinary.text
    assert str(tmp_path) not in ordinary.text
    assert technical.status_code == 200
    assert "checkpoint_path" not in technical.text
    assert "run_directory" not in technical.text
    assert str(tmp_path) not in technical.text
    technical_candidate = technical.json()["eligible"][0]
    assert technical_candidate["checkpoint_reference"] == "checkpoint_step_000100_ema.pt"
    assert technical_candidate["run_reference"] == "train-1"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?acknowledge=false",
        "?acknowledge=True",
        "?acknowledge=TRUE",
        "?acknowledge=1",
        "?acknowledge=yes",
        "?acknowledge=on",
        "?acknowledge=true&acknowledge=true",
        "?acknowledge=true&acknowledge=false",
        "?acknowledge%5B%5D=true",
    ],
)
def test_technical_checkpoint_api_requires_one_exact_acknowledgement(tmp_path: Path, query: str) -> None:
    config, _benchmark = _project(tmp_path)
    service = EvaluationService(project_root=tmp_path, config=config)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))

    response = TestClient(app).get(f"/evaluation/api/technical/checkpoints{query}")

    assert response.status_code == 400
    assert "checkpoint_path" not in response.text
    assert "run_directory" not in response.text
    assert str(tmp_path) not in response.text


def test_checkpoint_catalog_api_and_initial_html_sanitize_state_derived_strings(tmp_path: Path) -> None:
    config, _benchmark = _project(tmp_path)
    state_path = tmp_path / "runs" / "train-1" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["title"] = "Title Authorization=Bearer TITLESECRET C:\\private\\title.txt"
    state["backend_identity"].update(
        {
            "training_profile": "api_key=PROFILESECRET file:///" + "home/alice/profile.json",
            "dataset_identity_summary": f"password=DATASECRET {tmp_path / 'private' / 'dataset.json'}",
            "view_identity_summary": "client_secret=VIEWSECRET /" + "home/alice/private/view.json",
        }
    )
    _write_json(state_path, state)
    service = EvaluationService(project_root=tmp_path, config=config)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))
    client = TestClient(app)

    page = client.get("/evaluation")
    catalog = client.get("/evaluation/api/checkpoints?include_unavailable=true")

    assert page.status_code == 200
    assert catalog.status_code == 200
    for response in (page, catalog):
        for private_value in (
            "TITLESECRET",
            "PROFILESECRET",
            "DATASECRET",
            "VIEWSECRET",
            str(tmp_path),
            tmp_path.as_posix(),
            "C:\\private",
            "/" + "home/alice/private",
            "file:///" + "home/alice",
        ):
            assert private_value not in response.text
        assert "[redacted]" in response.text
    candidate = catalog.json()["eligible"][0]
    assert candidate["friendly_run_name"].startswith("Title Authorization=[redacted]")
    assert candidate["eligible"] is True


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [(GenerationCancelledError, 409), (GenerationTimedOutError, 408)],
)
def test_playground_api_never_returns_raw_adapter_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_status: int,
) -> None:
    config, _benchmark = _project(tmp_path)
    service = EvaluationService(project_root=tmp_path, config=config)
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    router = create_evaluation_router(context, service=service)
    app.include_router(router)
    secret = "PLAYWEBSECRET"

    def explode(*_args, **_kwargs):
        raise error_type(f"Authorization=Bearer {secret} C:\\private\\adapter.txt /private/adapter.txt")

    monkeypatch.setattr(router.spritelab_playground_service, "generate", explode)
    response = TestClient(app).post(
        "/evaluation/api/playground/generate",
        json={
            "prompt": "red sword",
            "checkpoint_id": "checkpoint",
            "explicit_action": True,
            "confirm_billable": False,
        },
    )

    assert response.status_code == expected_status
    assert secret not in response.text
    assert "C:\\private" not in response.text
    assert "/private/adapter.txt" not in response.text


@pytest.mark.parametrize("failure_site", ["generator", "evaluator"])
def test_evaluation_adapter_exceptions_never_reach_durable_or_public_state(
    tmp_path: Path,
    failure_site: str,
) -> None:
    config, _benchmark = _project(tmp_path)
    secret = "EVALSECRET"
    private_detail = f"Authorization=Bearer {secret} C:\\private\\adapter.txt /private/adapter.txt"

    class ExplodingGenerator(FakeEvaluationGenerator):
        def generate_benchmark(self, **kwargs) -> Path:
            if failure_site == "generator":
                raise RuntimeError(private_detail)
            return super().generate_benchmark(**kwargs)

    def evaluator(_generated: Path, _out: Path) -> dict:
        raise RuntimeError(private_detail)

    service = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=ExplodingGenerator(),
        evaluator=evaluator if failure_site == "evaluator" else _fake_evaluator(),
        output_root=tmp_path / "evaluation-output",
    )

    result = service.run(EvaluationRequest(explicit_action=True))

    assert result.status is ProductStatus.FAILED
    assert secret not in json.dumps(result.to_dict())
    assert "C:\\private" not in json.dumps(result.to_dict())
    assert all(secret not in stage.message for stage in service.latest_stages)
    assert service.latest_run_id is not None
    durable_state = service.events.state(service.latest_run_id)
    assert secret not in json.dumps(durable_state)
    assert "adapter diagnostics remain private" in json.dumps(durable_state)

    recreated = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=_fake_evaluator(),
        output_root=tmp_path / "evaluation-output",
    )
    assert secret not in json.dumps([stage.to_dict() for stage in recreated.latest_stages])


def test_ordinary_evaluation_surfaces_share_one_recursive_public_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _benchmark = _project(tmp_path)
    secret = "EVALUATOR-ADAPTER-SECRET"
    posix_path = "/srv/private/evaluator/report.json"
    windows_path = r"C:\private\evaluator\report.json"
    file_uri = "file:///srv/private/evaluator/report.json"
    hostile_text = f"Authorization=Bearer {secret} api_key={secret} {posix_path} {windows_path} {file_uri}"

    def hostile_evaluator(_generated: Path, out: Path) -> dict:
        out.mkdir(parents=True)
        row = {
            "sample_id": "sample-public",
            "prompt_id": "prompt-public",
            "prompt": "red sword",
            "seed": 7,
            "category": "weapon",
            "checkpoint": windows_path,
            "image": posix_path,
            "inference_path": file_uri,
            "metrics": {
                "hard_validity": {"pass": True, "adapter_payload": {"password": secret}},
                "pixel_art": {
                    "unique_palette_size": 8,
                    "silhouette_occupancy": 0.4,
                    "border_clipping": False,
                    "adapter_payload": {
                        "api_key": secret,
                        "posix": posix_path,
                        "windows": windows_path,
                        "uri": file_uri,
                    },
                },
                "adapter_payload": {"credential": secret},
            },
            "conditional_adherence": 1.0,
            "memorization_evidence_class": "no_material_match",
            "low_evidence_reason": hostile_text,
            "adapter_payload": {"authorization": f"Bearer {secret}"},
        }
        (out / "per_image_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {
            "schema_version": "generation_benchmark_v1.0",
            "comparison_method": hostile_text,
            "generated": posix_path,
            "training_manifests": [windows_path, file_uri],
            "summary": {
                "sample_count": 1,
                "hard_validity": {"pass_rate": 1.0},
                "conditional": {"represented_rate": 1.0},
                "pixel_art": {
                    "palette_size_mean": 8.0,
                    "adapter_payload": {"secret": secret, "path": posix_path},
                },
                "diversity": {"exact_duplicate_rate": 0.0},
                "memorization": {
                    "hard_evidence_count": 0,
                    "review_required_count": 0,
                    "adapter_payload": {"file_uri": file_uri},
                },
            },
            "promotion": {
                "pass": True,
                "memorization_machine_status": "pass",
                "checks": {"palette": True, "adapter_secret": secret},
                "adapter_payload": {"password": secret},
            },
            "artifacts": {"report": posix_path, "adapter_uri": file_uri},
            "adapter_payload": {"api_key": secret},
        }

    hostile_memorization = {
        "schema_version": "spritelab.product.memorization-display.v2",
        "evidence_state": "complete",
        "review_required_count": 0,
        "review_message": hostile_text,
        "review_contract": "spritelab.evaluation.memorization_review.signed-v2",
        "review_action_available": False,
        "writes_review_log": False,
        "items": [
            {
                "pair_id": "pair-public",
                "evidence_class": "no_material_match",
                "display_state": "No material match",
                "current_review_state": "no_material_match",
                "review_authoritative": True,
                "event_chain_valid": True,
                "review_action_available": False,
                "clear_action_available": False,
                "action_unavailable_reason": hostile_text,
                "generated_image": posix_path,
                "training_comparison_image": windows_path,
                "candidate_bundle_path": file_uri,
                "diagnostics": {"api_key": secret},
                "metrics": {"adapter_secret": secret},
            }
        ],
        "validation_reasons": [hostile_text],
        "legacy_reviews": [{"password": secret, "path": posix_path}],
        "adapter_payload": {"credential": secret},
    }
    monkeypatch.setattr(
        evaluation_service_module,
        "memorization_display",
        lambda *_args, **_kwargs: deepcopy(hostile_memorization),
    )
    service = EvaluationService(
        project_root=tmp_path,
        config=config,
        generator=FakeEvaluationGenerator(),
        evaluator=hostile_evaluator,
        output_root=tmp_path / "evaluation-output",
    )
    context = ProjectContext(tmp_path, config=config, runs_directory=tmp_path / "runs")
    app = create_product_app(context)
    app.include_router(create_evaluation_router(context, service=service))
    client = TestClient(app)

    run_response = client.post("/evaluation/api/run", json={"explicit_action": True})
    planned_checkpoint, planned_benchmark, planned_stages = service.plan(EvaluationRequest(dry_run=True))
    hostile_stage = deepcopy(planned_stages[-1])
    hostile_stage.message = hostile_text
    hostile_stage.metrics = {"adapter_payload": {"api_key": secret, "path": posix_path, "uri": file_uri}}
    monkeypatch.setattr(
        service,
        "plan",
        lambda _request: (planned_checkpoint, planned_benchmark, [hostile_stage]),
    )
    report_response = client.get("/evaluation/api/report-data")
    dashboard_response = client.get("/evaluation/api/dashboard")
    gallery_response = client.get("/evaluation/api/gallery")
    plan_response = client.get("/evaluation/api/plan")
    page_response = client.get("/evaluation")

    assert run_response.status_code == 200
    assert (
        report_response.status_code
        == dashboard_response.status_code
        == gallery_response.status_code
        == plan_response.status_code
        == page_response.status_code
        == 200
    )
    run_payload = run_response.json()
    report_payload = report_response.json()
    dashboard_payload = dashboard_response.json()
    assert run_payload["data"]["report"]["summary"]["hard_validity"]["pass_rate"] == 1.0
    assert run_payload["data"]["report"]["promotion"]["pass"] is True
    assert run_payload["data"]["report"]["promotion"]["checks"]["palette"] is True
    assert run_payload["data"]["memorization"]["items"][0]["review_authoritative"] is True
    public_row = report_payload["per_image_metrics"][0]
    assert public_row["metrics"]["hard_validity"]["pass"] is True
    assert public_row["metrics"]["pixel_art"]["border_clipping"] is False
    assert public_row["metrics"]["pixel_art"]["unique_palette_size"] == 8
    assert dashboard_payload["gallery"][0]["metrics"]["pixel_art"]["silhouette_occupancy"] == 0.4
    assert gallery_response.json()["samples"][0]["metrics"]["pixel_art"]["unique_palette_size"] == 8
    assert plan_response.json()["stages"][0]["metrics"] == {}

    assert service.latest_report is not None and secret in json.dumps(service.latest_report)
    assert secret in json.dumps(service.latest_rows)
    assert secret in json.dumps(service.latest_memorization)
    combined_public = "\n".join(
        (
            json.dumps(run_payload, sort_keys=True),
            json.dumps(report_payload, sort_keys=True),
            json.dumps(dashboard_payload, sort_keys=True),
            json.dumps(gallery_response.json(), sort_keys=True),
            json.dumps(plan_response.json(), sort_keys=True),
            page_response.text,
        )
    ).replace("\\\\", "\\")
    assert "adapter_payload" not in combined_public
    assert "generated_image" not in combined_public
    assert "candidate_bundle_path" not in combined_public
    for private in (secret, posix_path, windows_path, file_uri):
        assert private not in combined_public


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
    assert any("adapter diagnostics remain private" in blocker.message for blocker in result.blockers)
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
    state_path = output / run_id / "state.json"
    state_before = state_path.read_bytes()
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
    assert state_path.read_bytes() == state_before


def test_malformed_hash_consistent_product_report_reconstructs_stale(tmp_path: Path) -> None:
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
    run_directory = output / result.run.run_id
    report_path = run_directory / "product_report.json"
    report_path.write_text('{"summary":{},"summary":{}}\n', encoding="utf-8")
    state_path = run_directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["expected_artifacts"]["report"]["sha256"] = sha256(report_path.read_bytes()).hexdigest()
    _write_json(state_path, state)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    assert "product report is malformed or unsafe" in restored.latest_stale_reasons


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
    assert any(reason in item for item in restored.latest_stale_reasons)


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
    assert "checkpoint training view identity changed after evaluation" in restored.latest_stale_reasons


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
    assert "checkpoint is no longer eligible under the persisted dataset/view identity" in restored.latest_stale_reasons


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
    assert f"persisted {field} is missing" in restored.latest_stale_reasons


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
    assert f"persisted {field} is malformed" in restored.latest_stale_reasons


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

    private_reason = f"private adapter diagnostic: {tmp_path / 'secret.txt'}"

    def reject_candidate(_path: Path, *, expected_context):
        observed_context.update(expected_context)
        return SimpleNamespace(valid=False, reasons=("training view identity mismatch", private_reason))

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
    assert "candidate evidence identity invalid: training view identity mismatch" in restored.latest_stale_reasons
    assert str(tmp_path) not in json.dumps(restored.latest_stale_reasons)
    public_dashboard = restored.dashboard()
    assert public_dashboard["stale_reasons"]
    assert str(tmp_path) not in json.dumps(public_dashboard)


def test_reconstruction_never_hashes_or_loads_unanchored_persisted_paths(
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
    state_path = output / result.run.run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    outside_file = tmp_path / "outside-secret.bin"
    outside_file.write_bytes(b"outside")
    outside_tree = tmp_path / "outside-tree"
    outside_tree.mkdir()
    (outside_tree / "secret.bin").write_bytes(b"secret")
    state["checkpoint_path"] = str(outside_file)
    state["expected_artifacts"]["checkpoint"]["path"] = str(outside_file)
    state["expected_artifacts"]["attacker_tree"] = {
        "path": str(outside_tree),
        "kind": "tree",
        "sha256": "a" * 64,
    }
    state["expected_artifacts"]["candidate_evidence"] = {
        "path": str(outside_file),
        "kind": "file",
        "sha256": "b" * 64,
    }
    _write_json(state_path, state)

    original_file_hash = evaluation_service_module._file_sha256
    original_tree_hash = evaluation_service_module._tree_sha256
    observed_hashes: list[Path] = []

    def file_hash(path: Path, **kwargs) -> str:
        observed_hashes.append(path)
        assert path != outside_file
        return original_file_hash(path, **kwargs)

    def tree_hash(path: Path) -> str:
        observed_hashes.append(path)
        assert path != outside_tree
        return original_tree_hash(path)

    def reject_load(_path: Path, **_kwargs):
        raise AssertionError("unanchored candidate evidence must not be loaded")

    monkeypatch.setattr(evaluation_service_module, "_file_sha256", file_hash)
    monkeypatch.setattr(evaluation_service_module, "_tree_sha256", tree_hash)
    monkeypatch.setattr(evaluation_service_module, "load_candidate_bundle", reject_load)

    restored = EvaluationService(
        project_root=tmp_path,
        config=config,
        evaluator=_fake_evaluator(),
        output_root=output,
    )

    assert restored.latest_status == "STALE"
    assert outside_file not in observed_hashes
    assert outside_tree not in observed_hashes
    assert any("validated checkpoint catalog" in reason for reason in restored.latest_stale_reasons)


@pytest.mark.parametrize("reference", (r"C:outside-secret.bin", r"\outside-secret.bin"))
def test_run_artifact_rejects_windows_anchored_forms_before_metadata_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reference: str,
) -> None:
    observed: list[Path] = []
    original_lstat = Path.lstat

    def tracked_lstat(path: Path):
        observed.append(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", tracked_lstat)

    assert evaluation_service_module._safe_run_artifact(tmp_path, reference, "file") is None
    assert observed == []


def test_descriptor_hash_rejects_postopen_hard_link_count_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")
    descriptor = os.open(path, os.O_RDONLY)
    expected = os.fstat(descriptor)
    original_fstat = os.fstat
    calls = 0

    def raced_fstat(fd: int):
        nonlocal calls
        metadata = original_fstat(fd)
        calls += 1
        if calls == 2:
            values = list(metadata)
            values[3] = int(metadata.st_nlink) + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(os, "fstat", raced_fstat)
    try:
        with pytest.raises(OSError, match="changed"):
            evaluation_service_module._descriptor_sha256(descriptor, expected)
    finally:
        os.close(descriptor)


def test_run_artifact_parent_reparse_rejection_prevents_unanchored_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "artifact.bin"
    artifact.write_bytes(b"artifact")
    opened = False

    @contextmanager
    def reject_reparse(_directory: Path, _root: Path):
        raise UnsafeFilesystemOperation("mutable parent identity changed while anchored")
        yield

    def unexpected_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("artifact must not be opened after parent reparse rejection")

    monkeypatch.setattr(evaluation_service_module, "open_anchored_directory", reject_reparse)
    monkeypatch.setattr(evaluation_service_module, "_anchored_file_sha256", unexpected_open)

    with pytest.raises(UnsafeFilesystemOperation, match="identity changed"):
        evaluation_service_module._run_artifact_sha256(run, artifact, "file")

    assert opened is False


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
    assert reason in restored.latest_stale_reasons


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
    assert reason in restored.latest_stale_reasons


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
    assert "persisted evaluation stages are malformed" in restored.latest_stale_reasons
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
