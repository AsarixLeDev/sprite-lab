from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import spritelab.product_runtime as product_runtime
import spritelab.product_web.app as product_app
from spritelab.product_core import ProductResult, ProductStatus, ProjectContext, WebServerSettings
from spritelab.product_features.dataset import cli as dataset_cli
from spritelab.product_features.dataset.cli import MetadataWizardOutcome
from spritelab.product_features.dataset.plugin import build_plugin
from spritelab.product_features.dataset.web import MetadataWizardSession
from spritelab.product_web import cli as web_cli
from spritelab.product_web.app import create_app
from test_product_dataset_helpers import make_png


def _metadata() -> dict[str, Any]:
    return {
        "creator_or_rights_holder": "Synthetic Artist",
        "pack_title": "Synthetic Pack",
        "source_type": "my_original_work",
        "source_page_url": None,
        "original_work_declaration": True,
        "license_identifier": "cc0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_evidence_file": None,
        "attribution_text": None,
        "permission_confirmed": False,
    }


def _wizard_client(project: Path, source: Path) -> tuple[TestClient, MetadataWizardSession]:
    runs = project / "runs" / "v3"
    runs.mkdir(parents=True)
    session = MetadataWizardSession()
    context = ProjectContext(
        project,
        config={"dataset": {"pending_input_root": str(source)}},
        runs_directory=runs,
    )
    app = create_app(context, plugins=(build_plugin(),))
    app.state.spritelab_metadata_wizard_session = session
    return TestClient(app), session


def _csrf(client: TestClient) -> dict[str, str]:
    return {"x-csrf-token": client.app.state.spritelab_csrf_token}


def _build_args(folder: Path, output: Path) -> Namespace:
    return Namespace(
        folder=str(folder),
        output=output,
        metadata_file=None,
        no_review=True,
        allow_hosted=False,
        provider_factory=None,
        json=False,
        no_color=False,
        quiet=False,
        debug=False,
    )


def _interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(dataset_cli.sys.stdout, "isatty", lambda: True)


def test_wizard_complete_is_server_authoritative_and_stops_server(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = tmp_path / "source"
    make_png(source / "sprite.png")
    client, session = _wizard_client(project, source)
    shutdowns: list[bool] = []
    client.app.state.spritelab_request_shutdown = lambda: shutdowns.append(True)

    page = client.get("/dataset/metadata")
    assert 'id="metadata-complete"' in page.text
    assert 'id="metadata-cancel"' in page.text
    assert 'id="metadata-complete" type="button" disabled' in page.text
    script = client.get("/plugins/dataset.intake/static/metadata.js").text
    assert 'post("/dataset/api/metadata/complete"' in script
    assert 'post("/dataset/api/metadata/cancel"' in script

    premature = client.post("/dataset/api/metadata/complete", headers=_csrf(client), json={})
    assert premature.status_code == 409
    assert session.outcome is MetadataWizardOutcome.PENDING
    assert shutdowns == []

    inspection = client.post("/dataset/api/metadata/inspect", headers=_csrf(client), json={}).json()
    saved = client.post(
        "/dataset/api/metadata/save",
        headers=_csrf(client),
        json={"pack_id": inspection["packs"][0]["pack_id"], "metadata": _metadata()},
    )
    assert saved.status_code == 200

    completed = client.post("/dataset/api/metadata/complete", headers=_csrf(client), json={})
    assert completed.status_code == 200
    assert completed.json()["outcome"] == "complete"
    assert session.outcome is MetadataWizardOutcome.COMPLETE
    assert shutdowns == [True]


def test_wizard_cancel_is_controlled_and_stops_server(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_png(source / "sprite.png")
    client, session = _wizard_client(tmp_path / "project", source)
    shutdowns: list[bool] = []
    client.app.state.spritelab_request_shutdown = lambda: shutdowns.append(True)

    cancelled = client.post("/dataset/api/metadata/cancel", headers=_csrf(client), json={})

    assert cancelled.status_code == 200
    assert cancelled.json()["outcome"] == "cancelled"
    assert session.outcome is MetadataWizardOutcome.CANCELLED
    assert shutdowns == [True]


@pytest.mark.parametrize("outcome", [MetadataWizardOutcome.CANCELLED, MetadataWizardOutcome.INTERRUPTED])
def test_cli_never_builds_after_wizard_does_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: MetadataWizardOutcome
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spritelab.yaml").write_text(
        "project:\n  name: lifecycle-test\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    make_png(source / "sprite.png")
    _interactive(monkeypatch)
    monkeypatch.setattr(dataset_cli, "launch_metadata_wizard", lambda *_args: outcome)

    def forbidden_build(*_args: Any, **_kwargs: Any) -> ProductResult:
        pytest.fail("dataset build must not start unless the wizard completed")

    monkeypatch.setattr(dataset_cli.DatasetIntakeService, "build", forbidden_build)
    result = dataset_cli._handle_build(_build_args(source, tmp_path / "out"), [])

    assert result.status is ProductStatus.BLOCKED
    assert result.data["wizard_outcome"] == outcome.value
    assert result.data["build_started"] is False
    assert not (tmp_path / "out" / "result.json").exists()


def test_cli_reinspects_and_builds_only_after_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spritelab.yaml").write_text(
        "project:\n  name: lifecycle-test\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    make_png(source / "sprite.png")
    _interactive(monkeypatch)
    sequence: list[str] = []
    inspections = iter(
        (
            {"wizard_required": True, "image_count": 1, "packs": []},
            {"wizard_required": False, "image_count": 1, "packs": []},
        )
    )

    def inspect(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        sequence.append("inspect")
        return next(inspections)

    def build(*_args: Any, **_kwargs: Any) -> ProductResult:
        sequence.append("build")
        return ProductResult(ProductStatus.COMPLETE, "built", feature="dataset", data={"counts": {}})

    monkeypatch.setattr(dataset_cli, "inspect_dataset_folder", inspect)
    monkeypatch.setattr(dataset_cli, "launch_metadata_wizard", lambda *_args: MetadataWizardOutcome.COMPLETE)
    monkeypatch.setattr(dataset_cli.DatasetIntakeService, "build", build)

    result = dataset_cli._handle_build(_build_args(source, tmp_path / "out"), [])

    assert result.status is ProductStatus.COMPLETE
    assert sequence == ["inspect", "inspect", "build"]


def test_cli_complete_still_blocks_when_reinspection_is_no_longer_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spritelab.yaml").write_text(
        "project:\n  name: lifecycle-test\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    make_png(source / "sprite.png")
    _interactive(monkeypatch)
    sequence: list[str] = []

    def inspect(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        sequence.append("inspect")
        return {"wizard_required": True, "image_count": 1, "packs": []}

    def forbidden_build(*_args: Any, **_kwargs: Any) -> ProductResult:
        pytest.fail("a stale Complete decision must not start preprocessing")

    monkeypatch.setattr(dataset_cli, "inspect_dataset_folder", inspect)
    monkeypatch.setattr(dataset_cli, "launch_metadata_wizard", lambda *_args: MetadataWizardOutcome.COMPLETE)
    monkeypatch.setattr(dataset_cli.DatasetIntakeService, "build", forbidden_build)

    result = dataset_cli._handle_build(_build_args(source, tmp_path / "out"), [])

    assert result.status is ProductStatus.BLOCKED
    assert result.data["build_started"] is False
    assert result.data["wizard_outcome"] == "complete"
    assert sequence == ["inspect", "inspect"]


def test_launch_metadata_wizard_returns_server_session_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}

    def create(context: ProjectContext, **_kwargs: Any) -> SimpleNamespace:
        captured["context"] = context
        app = SimpleNamespace(state=SimpleNamespace())
        captured["app"] = app
        return app

    def run(app: Any, _settings: WebServerSettings, **_kwargs: Any) -> None:
        app.state.spritelab_metadata_wizard_session.finish(MetadataWizardOutcome.CANCELLED)

    monkeypatch.setattr(product_runtime, "build_product_runtime", lambda: SimpleNamespace(plugins=()))
    monkeypatch.setattr(product_app, "create_app", create)
    monkeypatch.setattr(web_cli, "run_server", run)

    source = tmp_path / "source"
    source.mkdir()
    outcome = dataset_cli.launch_metadata_wizard(source, tmp_path / "out")

    assert outcome is MetadataWizardOutcome.CANCELLED
    assert captured["context"].config["dataset"]["pending_input_root"] == str(source.resolve())
    assert "metadata_wizard_session" not in captured["context"].config["dataset"]


def test_launch_metadata_wizard_converts_keyboard_interrupt_to_controlled_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(product_runtime, "build_product_runtime", lambda: SimpleNamespace(plugins=()))
    monkeypatch.setattr(
        product_app,
        "create_app",
        lambda *_args, **_kwargs: SimpleNamespace(state=SimpleNamespace()),
    )
    monkeypatch.setattr(
        web_cli,
        "run_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    source = tmp_path / "source"
    source.mkdir()

    assert dataset_cli.launch_metadata_wizard(source, None) is MetadataWizardOutcome.INTERRUPTED


def test_run_server_exposes_shutdown_and_closes_socket_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        closed = False

        def setsockopt(self, *_args: Any) -> None: ...

        def bind(self, address: tuple[str, int]) -> None:
            self.address = address

        def listen(self, _backlog: int) -> None: ...

        def getsockname(self) -> tuple[str, int]:
            return (self.address[0], 43123)

        def close(self) -> None:
            self.closed = True

    sock = FakeSocket()
    app = SimpleNamespace(state=SimpleNamespace())
    observed: list[bool] = []
    monkeypatch.setattr(web_cli.socket, "socket", lambda *_args: sock)

    def run(server: Any, **_kwargs: Any) -> None:
        app.state.spritelab_request_shutdown()
        observed.append(server.should_exit)

    monkeypatch.setattr("uvicorn.Server.run", run)
    web_cli.run_server(app, WebServerSettings(), open_browser=False)

    assert observed == [True]
    assert sock.closed is True


def test_run_server_closes_socket_when_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        closed = False

        def setsockopt(self, *_args: Any) -> None: ...

        def bind(self, address: tuple[str, int]) -> None:
            self.address = address

        def listen(self, _backlog: int) -> None: ...

        def getsockname(self) -> tuple[str, int]:
            return (self.address[0], 43123)

        def close(self) -> None:
            self.closed = True

    sock = FakeSocket()
    monkeypatch.setattr(web_cli.socket, "socket", lambda *_args: sock)
    monkeypatch.setattr("uvicorn.Server.run", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        web_cli.run_server(SimpleNamespace(state=SimpleNamespace()), WebServerSettings(), open_browser=False)

    assert sock.closed is True
