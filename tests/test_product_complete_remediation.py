from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from spritelab.product_core import (
    ApprovedFolderError,
    ApprovedFolderStore,
    ProductEvent,
    ProductPlugin,
    ProductResult,
    ProductSettingsError,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
    api_error_payload,
    product_api,
)
from spritelab.product_features.dataset.plugin import create_plugin as create_dataset_plugin
from spritelab.product_features.providers import DeterministicMockVisionProvider
from spritelab.product_features.providers.discovery import VisionProviderRegistry
from spritelab.product_features.providers.plugin import build_plugin as build_provider_plugin
from spritelab.product_features.providers.product_adapter import HubProductVisionProvider
from spritelab.product_features.providers.web import create_settings_router
from spritelab.product_features.training.config import ComputeSettings, effective_compute_context
from spritelab.product_features.training.service import backend_from_context
from spritelab.product_runtime import build_product_runtime
from spritelab.product_web.app import create_app
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    LEGACY_EVENT_FILENAME,
    EventRepository,
    LegacyEventMigrationError,
)
from spritelab.remote_compute import SSHComputeBackend
from spritelab.v3.config import DEFAULT_CONFIG


def _context(root: Path, values: dict[str, Any] | None = None) -> ProjectContext:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return ProjectContext(root, values or deepcopy(DEFAULT_CONFIG), root / "spritelab.yaml", runs)


def _make_pre_origin_legacy_fixture(directory: Path) -> None:
    (directory / EVENT_FILENAME).unlink()
    (directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in tuple(state):
        if (
            key.startswith("event_history_origin")
            or key.startswith("event_migration_")
            or key.startswith("event_canonical_")
        ):
            state.pop(key)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csrf(client: TestClient) -> dict[str, str]:
    return {"x-csrf-token": client.app.state.spritelab_csrf_token}


def test_complete_shell_rendering_has_zero_active_provider_or_compute_operations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    counters = {"provider": 0, "compute": 0, "training": 0, "generation": 0, "cloud": 0}

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> Any:
        counters["provider"] += 1
        raise AssertionError("passive rendering attempted provider discovery")

    def forbidden_compute(*_args: Any, **_kwargs: Any) -> Any:
        counters["compute"] += 1
        raise AssertionError("passive rendering attempted a compute probe")

    monkeypatch.setattr(
        "spritelab.product_features.providers.discovery.VisionProviderRegistry.discover", forbidden_provider
    )
    monkeypatch.setattr("spritelab.remote_compute.local.LocalComputeBackend.probe", forbidden_compute)
    client = TestClient(create_app(_context(tmp_path), plugins=build_product_runtime().plugins))
    for path in ("/", "/", "/settings", "/settings/vision", "/harvest", "/dataset", "/training", "/evaluation"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "skip-link" in response.text
    assert counters == {"provider": 0, "compute": 0, "training": 0, "generation": 0, "cloud": 0}


def test_provider_save_is_passive_and_detect_and_test_are_explicit_once(tmp_path: Path) -> None:
    class CountingProvider(DeterministicMockVisionProvider):
        def __init__(self) -> None:
            super().__init__()
            self.probe_count = 0

        def probe(self):
            self.probe_count += 1
            return super().probe()

    class CountingRegistry(VisionProviderRegistry):
        def __init__(self, provider: CountingProvider) -> None:
            super().__init__(providers=(provider,), plugin_entry_points=())
            self.discover_count = 0

        def discover(self):
            self.discover_count += 1
            return super().discover()

    context = _context(tmp_path)
    provider = CountingProvider()
    registry = CountingRegistry(provider)
    plugin = replace(
        build_provider_plugin(),
        web_router_factory=lambda supplied: create_settings_router(
            supplied, registry_factory=lambda _settings: registry
        ),
    )
    client = TestClient(create_app(context, plugins=(plugin,)))
    assert client.get("/settings/vision").status_code == 200
    assert client.get("/settings/vision").status_code == 200
    assert provider.probe_count == registry.discover_count == 0
    saved = client.post(
        "/settings/vision/api/settings",
        headers=_csrf(client),
        json={
            "type": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "model": "mock-vision-v1",
            "privacy_policy": "local_only",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["provider_requests"] == 0
    assert provider.probe_count == registry.discover_count == 0
    detected = client.post("/settings/vision/api/detect", headers=_csrf(client), json={})
    assert detected.status_code == 200
    assert registry.discover_count == 1
    assert provider.probe_count == 1
    tested = client.post("/settings/vision/api/test", headers=_csrf(client), json={})
    assert tested.status_code == 200
    assert registry.discover_count == 1
    assert provider.probe_count == 2
    assert tested.json()["image_inference_requests"] == 0
    assert tested.json()["available"] is True
    assert tested.json()["model_validation"]["state"] == "available"
    refreshed = client.post("/settings/vision/api/models/refresh", headers=_csrf(client), json={})
    assert refreshed.status_code == 200
    assert refreshed.json()["models"] == [
        {
            "model_id": "mock-vision-v1",
            "display_name": "Mock vision v1",
            "capabilities": ["vision", "structured_output"],
            "metadata": {},
        }
    ]


def test_provider_javascript_has_no_page_load_discovery() -> None:
    script = (Path(__file__).parents[1] / "src/spritelab/product_features/providers/static/providers.js").read_text(
        encoding="utf-8"
    )
    listener = script.index('button("detect")?.addEventListener("click"')
    request = script.index('request("/settings/vision/api/detect"')
    assert listener < request
    assert "DOMContentLoaded" not in script


def test_approved_folder_contract_is_opaque_bound_expiring_and_read_only(tmp_path: Path) -> None:
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    current = [now]
    context = _context(tmp_path / "project")
    source = tmp_path / "Outside folder with spaces Ω"
    source.mkdir()
    image = source / "sprite.png"
    original = b"synthetic-png-bytes"
    image.write_bytes(original)
    store = ApprovedFolderStore(context, session_id="session-a", clock=lambda: current[0])
    record = store.approve(source, source="native_picker", ttl_seconds=60)
    assert store.resolve(record.approval_id) == source.resolve()
    assert image.read_bytes() == original
    public = record.public_dict()
    assert "canonical_path" not in public and str(source) not in json.dumps(public, ensure_ascii=False)
    assert public["read_only"] is True
    with pytest.raises(ApprovedFolderError, match="another project"):
        store.resolve(record.approval_id, project_id="different-project")
    store._records[record.approval_id] = replace(record, session_id="session-b")
    with pytest.raises(ApprovedFolderError, match="another server session"):
        store.resolve(record.approval_id)
    store._records[record.approval_id] = record
    current[0] = now + timedelta(seconds=61)
    with pytest.raises(ApprovedFolderError, match="expired"):
        store.resolve(record.approval_id)


def test_approved_folder_rejects_traversal_files_malformed_and_sibling_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import_root = tmp_path / "imports"
    child = import_root / "chosen"
    sibling = tmp_path / "sibling"
    child.mkdir(parents=True)
    sibling.mkdir()
    file_path = import_root / "file.png"
    file_path.write_bytes(b"x")
    store = ApprovedFolderStore(_context(project), import_roots=(import_root,))
    approved = store.approve_import_root_child(0, "chosen")
    assert store.resolve(approved.approval_id) == child.resolve()
    for value in ("../sibling", "chosen/../../sibling"):
        with pytest.raises(ApprovedFolderError):
            store.approve_import_root_child(0, value)
    with pytest.raises(ApprovedFolderError, match="not a folder"):
        store.approve(file_path, source="native_picker")
    with pytest.raises(ApprovedFolderError):
        store.approve(str(child) + "\x00bad", source="native_picker")
    with pytest.raises(ApprovedFolderError):
        store.approve(r"C:relative", source="native_picker")
    with pytest.raises(ApprovedFolderError):
        store.approve("\\\\.\\C:\\", source="native_picker")


def test_import_root_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this Windows test session.")
    store = ApprovedFolderStore(_context(tmp_path / "project"), import_roots=(root,))
    with pytest.raises(ApprovedFolderError, match="escapes"):
        store.approve_import_root_child(0, "escape")


def test_dataset_web_rejects_raw_paths_and_accepts_only_explicit_opaque_approval(tmp_path: Path) -> None:
    project = tmp_path / "project"
    selected = tmp_path / "outside selected folder Ω"
    sibling = tmp_path / "outside sibling"
    selected.mkdir()
    sibling.mkdir()
    values = deepcopy(DEFAULT_CONFIG)
    values["dataset"]["import_roots"] = [str(tmp_path)]
    context = _context(project, values)
    client = TestClient(create_app(context, plugins=(create_dataset_plugin(folder_chooser=lambda: selected),)))
    raw = client.post(
        "/dataset/api/inspect",
        headers=_csrf(client),
        json={"folder": str(sibling)},
    )
    assert raw.status_code == 422
    assert raw.json()["error_code"] == "browser_path_not_allowed"
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={})
    assert chosen.status_code == 200
    payload = chosen.json()
    assert str(selected) not in json.dumps(payload, ensure_ascii=False)
    approval_id = payload["approval"]["approval_id"]
    inspected = client.post(
        "/dataset/api/inspect",
        headers=_csrf(client),
        json={"approval_id": approval_id},
    )
    assert inspected.status_code == 200
    assert inspected.json()["approval_id"] == approval_id
    unapproved = client.post(
        "/dataset/api/inspect",
        headers=_csrf(client),
        json={"approval_id": "opaque-but-not-approved"},
    )
    assert unapproved.status_code == 422
    fallback = client.post(
        "/dataset/api/folders/import-root",
        headers=_csrf(client),
        json={"root_index": 0, "relative": sibling.name},
    )
    assert fallback.status_code == 200
    assert fallback.json()["approval"]["source"] == "project_import_root"
    approvals = client.get("/dataset/api/folders/approved").json()
    assert approvals["paths_exposed"] is False
    assert str(tmp_path) not in json.dumps(approvals, ensure_ascii=False)


def test_settings_are_atomic_project_scoped_and_never_persist_secrets(tmp_path: Path) -> None:
    context = _context(tmp_path / "project-a")
    repository = ProductSettingsRepository(context)
    with pytest.raises(ProductSettingsError, match="Secrets are not allowed"):
        repository.save("provider", {"api_key": "do-not-save"})
    with pytest.raises(ProductSettingsError, match="Bearer"):
        repository.save("provider", {"endpoint": "Bearer do-not-save"})

    def save(index: int) -> int:
        return int(
            ProductSettingsRepository(context).save("provider", {"mode": "automatic", "model": f"model-{index}"})[
                "configuration_version"
            ]
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        versions = sorted(pool.map(save, range(12)))
    assert versions == list(range(1, 13))
    document = json.loads(repository.path.read_text(encoding="utf-8"))
    assert document["sections"]["provider"]["configuration_version"] == 12
    assert not list(repository.path.parent.glob("*.tmp"))
    other = _context(tmp_path / "project-b")
    foreign = ProductSettingsRepository(other)
    foreign.path.parent.mkdir(parents=True)
    foreign.path.write_bytes(repository.path.read_bytes())
    with pytest.raises(ProductSettingsError, match="another project"):
        foreign.section("provider")


def test_saved_provider_and_compute_settings_are_the_execution_representations(tmp_path: Path) -> None:
    context = _context(tmp_path)
    repository = ProductSettingsRepository(context)
    repository.save(
        "provider",
        {
            "mode": "specific",
            "adapter": "openai_compatible",
            "endpoint": "http://127.0.0.1:9000",
            "model": "local-model",
            "credential_env": "SPRITELAB_TEST_CREDENTIAL",
            "privacy_policy": "local_only",
            "location": "local",
        },
    )
    captured = []
    adapter = HubProductVisionProvider(
        context,
        registry_factory=lambda settings: (
            captured.append(settings)
            or VisionProviderRegistry(settings, providers=(DeterministicMockVisionProvider(),), plugin_entry_points=())
        ),
    )
    assert adapter.settings.model == "local-model"
    assert captured[0].endpoint == "http://127.0.0.1:9000"
    assert "SPRITELAB_TEST_CREDENTIAL" in repository.path.read_text(encoding="utf-8")
    repository.save(
        "compute",
        ComputeSettings.from_mapping(
            {
                "type": "ssh",
                "host": "compute.example.test",
                "username": "sprite",
                "remote_workspace": "/workspace/sprite-lab",
                "credential_reference": "ssh-agent",
                "environment_profile": "python3",
            }
        ).to_persisted_dict(),
    )
    effective, selected, version, saved = effective_compute_context(context)
    assert saved is True and version == 1 and selected.backend_type == "ssh"
    assert isinstance(backend_from_context(effective), SSHComputeBackend)
    with pytest.raises(ValueError, match="not available"):
        ComputeSettings.from_mapping({"type": "runpod"}, allow_unavailable=False)


def test_legacy_product_events_migrate_once_and_malformed_rows_fail_safely(tmp_path: Path) -> None:
    repository = EventRepository(tmp_path / "runs")
    run_id = "legacy-run"
    repository.create_run(run_id, feature="training", command="training.start")
    directory = repository.run_directory(run_id)
    assert directory is not None
    _make_pre_origin_legacy_fixture(directory)
    first = ProductEvent(
        run_id,
        "2026-07-13T10:00:00+00:00",
        "training",
        "campaign",
        "started",
        ProductStatus.RUNNING,
        message="Started.",
    )
    (directory / LEGACY_EVENT_FILENAME).write_text(
        json.dumps(first.to_dict()) + "\n",
        encoding="utf-8",
    )
    assert [item.event.event_type for item in repository.events(run_id)] == ["started"]
    second = replace(first, timestamp="2026-07-13T10:01:00+00:00", event_type="progress")
    third = replace(first, timestamp="2026-07-13T10:02:00+00:00", event_type="complete")
    assert repository.append(second) == 2
    assert repository.append(third) == 3
    assert (directory / EVENT_FILENAME).is_file()
    assert [item.event.event_type for item in repository.events(run_id)] == ["started", "progress", "complete"]
    assert (directory / LEGACY_EVENT_FILENAME).read_text(encoding="utf-8").startswith("{")

    malformed_id = "malformed-legacy-run"
    repository.create_run(malformed_id, feature="training", command="training.start")
    malformed_directory = repository.run_directory(malformed_id)
    assert malformed_directory is not None
    _make_pre_origin_legacy_fixture(malformed_directory)
    malformed_first = replace(first, run_id=malformed_id)
    (malformed_directory / LEGACY_EVENT_FILENAME).write_text(
        "not-json\n" + json.dumps(malformed_first.to_dict()) + "\n",
        encoding="utf-8",
    )
    assert [item.event.event_type for item in repository.events(malformed_id)] == ["started"]
    with pytest.raises(LegacyEventMigrationError, match="malformed"):
        repository.append(replace(second, run_id=malformed_id))
    assert not (malformed_directory / EVENT_FILENAME).exists()


def test_all_feature_and_plugin_api_failures_use_one_safe_envelope(tmp_path: Path) -> None:
    def router_factory(_context: ProjectContext) -> APIRouter:
        router = APIRouter()

        @product_api
        async def fail() -> None:
            raise RuntimeError("raw traceback bearer top-secret")

        for index, path in enumerate(
            (
                "/dataset/api/injected-failure",
                "/training/api/injected-failure",
                "/evaluation/api/injected-failure",
                "/settings/vision/api/injected-failure",
                "/plugin-owned/api/injected-failure",
            )
        ):
            router.add_api_route(path, fail, methods=["GET"], name=f"failure-{index}")
        return router

    plugin = ProductPlugin(
        plugin_id="error.injector",
        title="Error injector",
        cli_registration=lambda _registry: None,
        status_provider=lambda _context: ProductResult(ProductStatus.READY, "Ready."),
        capability_probe=lambda _context: (),
        web_router_factory=router_factory,
        api_prefixes=("/plugin-owned/api",),
    )
    client = TestClient(create_app(_context(tmp_path), plugins=(plugin,)), raise_server_exceptions=False)
    required = {
        "schema_version",
        "status",
        "error_code",
        "message",
        "error_reference",
        "recoverable",
        "next_action",
    }
    for path in (
        "/dataset/api/injected-failure",
        "/training/api/injected-failure",
        "/evaluation/api/injected-failure",
        "/settings/vision/api/injected-failure",
        "/plugin-owned/api/injected-failure",
    ):
        response = client.get(path)
        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        assert set(response.json()) == required
        assert response.json()["schema_version"] == "spritelab.product.api-error.v1"
        assert "traceback" not in response.text and "top-secret" not in response.text


def test_csrf_and_invalid_sse_ids_return_standard_api_errors(tmp_path: Path) -> None:
    client = TestClient(create_app(_context(tmp_path)))
    response = client.post("/dataset/api/review/confirm-exclusions", json={})
    assert response.status_code == 403
    assert response.json()["error_code"] == "csrf_validation_failed"
    invalid = client.get("/api/runs/not%2Fa%2Frun/events?once=true")
    assert invalid.status_code in {404, 422}
    assert invalid.headers["content-type"].startswith("application/json")
    assert invalid.json()["schema_version"] == "spritelab.product.api-error.v1"


def test_error_contract_has_only_allowlisted_fields() -> None:
    payload = api_error_payload(
        409,
        "blocked",
        "The action is blocked.",
        recoverable=True,
        next_action="Resolve the blocker.",
        details={"private": "not-authorized"},
    )
    assert "details" not in payload
    assert set(payload) == {
        "schema_version",
        "status",
        "error_code",
        "message",
        "error_reference",
        "recoverable",
        "next_action",
    }


def test_shared_pages_and_assets_expose_keyboard_chart_and_narrow_layout_contracts(tmp_path: Path) -> None:
    client = TestClient(create_app(_context(tmp_path), plugins=build_product_runtime().plugins))
    for path in ("/harvest", "/dataset", "/settings/vision", "/training", "/evaluation"):
        page = client.get(path)
        assert page.status_code == 200
        for marker in ("Skip to content", 'aria-label="Primary"', "Current run", "spritelab-csrf"):
            assert marker in page.text
    evaluation_js = client.get("/evaluation/static/evaluation.js").text
    assert '<button type="button" class="sample-card"' in evaluation_js
    assert "chart-summary" in evaluation_js and "chart-table" in evaluation_js
    a11y_css = client.get("/evaluation/static/evaluation-a11y.css").text
    assert "forced-colors" in a11y_css
    assert "prefers-reduced-motion" in a11y_css
    assert "max-width:390px" in a11y_css
    assert "min-height:44px" in a11y_css
