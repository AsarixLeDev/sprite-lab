from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter
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
from spritelab.product_web.events import EventRepository
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
        status_provider=lambda _context: result or _result(),
        capability_probe=lambda _context: capabilities,
        web_router_factory=router_factory if router_text is not None else None,
        navigation=(WebNavigationItem(plugin_id, title, path, 15), *extra_navigation),
        required_backend_capabilities=required,
        web_assets=assets,
    )


def _write_run(
    context: ProjectContext,
    *,
    run_id: str = "20260713T100000Z-train-demo",
    terminal: bool = False,
    secret_log: bool = False,
) -> tuple[str, list[ProductEvent]]:
    assert context.runs_directory is not None
    directory = context.runs_directory / run_id
    (directory / "logs").mkdir(parents=True)
    (directory / "report").mkdir()
    status = ProductStatus.COMPLETE if terminal else ProductStatus.RUNNING
    state = {
        "schema_version": "spritelab.v3.run-state.v1",
        "run_id": run_id,
        "command": "training",
        "stage": "optimize-model",
        "status": status.value,
        "started_at": "2026-07-13T10:00:00+00:00",
        "ended_at": "2026-07-13T10:02:00+00:00" if terminal else None,
        "resumable": not terminal,
        "message": "Training complete." if terminal else "Training sprites.",
    }
    (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
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
            metrics={"loss": 0.42, "pause_available": not terminal, "cancel_available": not terminal},
            artifact_references=(r"C:\example\checkpoint.safetensors",),
        ),
    ]
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event.to_dict()) + "\n" for event in events), encoding="utf-8"
    )
    log = "step=40 loss=0.42\n"
    if secret_log:
        log += f"API_KEY=super-secret Authorization=Bearer runtime-secret project_root={context.project_root}\n"
    (directory / "logs" / "run.log").write_text(log, encoding="utf-8")
    if terminal:
        (directory / "report" / "index.html").write_text("<h1>Offline report</h1>", encoding="utf-8")
    return run_id, events


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
    run_id, _events = _write_run(context)
    directory = context.runs_directory / run_id  # type: ignore[operator]
    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["metrics"]["auth_token"] = "do-not-return"
    (directory / "events.jsonl").write_text(lines[0] + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    client = TestClient(create_app(context, event_poll_interval=0.01))
    all_events = client.get(f"/api/runs/{run_id}/events?once=true")
    assert all_events.status_code == 200
    assert "event: product" in all_events.text
    assert "id: 1" in all_events.text and "id: 2" in all_events.text
    reconnect = client.get(f"/api/runs/{run_id}/events?once=true", headers={"Last-Event-ID": "1"})
    assert "Preparing images" not in reconnect.text
    assert "Processed batch 40" in reconnect.text
    assert "do-not-return" not in reconnect.text


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
    assert client.get(f"/runs/{run_id}/report").status_code == 200


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
