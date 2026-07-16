from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from spritelab import __main__ as package_main
from spritelab.dev import cli as dev_cli
from spritelab.product_core import (
    PRODUCT_EVENT_SCHEMA,
    DuplicatePluginIdError,
    ProductCapability,
    ProductEvent,
    ProductPlugin,
    ProductPluginRegistry,
    ProductResult,
    ProductStatus,
    ProjectContext,
    WebNavigationItem,
    WebSecurityError,
    WebServerSettings,
    load_plugin,
)
from spritelab.v3 import cli as product_cli
from spritelab.v3.config import ConfigError, ProjectConfig
from spritelab.v3.model import AuditStatus, Evidence, ProjectState, StageState, StageStatus


def _plugin(plugin_id: str = "dataset.intake") -> ProductPlugin:
    def register_cli(registry) -> None:
        registry.command(
            "providers",
            owner=plugin_id,
            help="Synthetic provider command.",
            handler=lambda *_: ProductResult(ProductStatus.COMPLETE, "Provider feature registered."),
            replace=True,
        )

    return ProductPlugin(
        plugin_id=plugin_id,
        title="Dataset intake",
        cli_registration=register_cli,
        status_provider=lambda _context: ProductResult(ProductStatus.READY, "Ready."),
        capability_probe=lambda _context: (ProductCapability("dataset.import", "Dataset import", ProductStatus.READY),),
        web_router_factory=lambda _context: object(),
        navigation=(WebNavigationItem("dataset", "Dataset", "/dataset"),),
        required_backend_capabilities=("storage.local",),
        settings_schema={"type": "object"},
    )


def _state(tmp_path: Path) -> ProjectState:
    return ProjectState(
        project_name="synthetic",
        project_root=tmp_path,
        config_path=tmp_path / "spritelab.yaml",
        source_commit="abc123-internal-commit",
        stages=[
            StageState(
                key="training-infrastructure-audit",
                title="Training audit",
                status=StageStatus.STALE,
                explanation="Internal evidence is stale.",
                blockers=["Internal hash changed."],
                evidence=[Evidence("audit.json", "f" * 64, "deadbeef")],
                source_commit="deadbeef",
                audit=AuditStatus.STALE,
            )
        ],
    )


def _invoke(main, args: list[str], capsys, **kwargs) -> tuple[int, str]:
    with pytest.raises(SystemExit) as caught:
        main(args, **kwargs)
    return int(caught.value.code), capsys.readouterr().out


def test_feature_module_build_plugin_contract() -> None:
    module = ModuleType("synthetic_product_plugin")
    module.build_plugin = _plugin  # type: ignore[attr-defined]
    plugin = load_plugin(module)
    context = ProjectContext(Path.cwd())
    assert plugin.plugin_id == "dataset.intake"
    assert plugin.status_provider(context).status == ProductStatus.READY
    assert plugin.capability_probe(context)[0].available is True
    assert plugin.web_plugin().navigation[0].path == "/dataset"


def test_duplicate_plugin_ids_are_rejected() -> None:
    with pytest.raises(DuplicatePluginIdError, match=r"dataset\.intake"):
        ProductPluginRegistry([_plugin(), _plugin()])


def test_plugin_can_register_reserved_cli_without_root_edit(capsys) -> None:
    code, output = _invoke(product_cli.main, ["providers", "--json"], capsys, plugins=[_plugin()])
    assert code == 0
    assert json.loads(output)["status"] == "COMPLETE"


def test_controlled_missing_product_feature(capsys) -> None:
    code, output = _invoke(product_cli.main, ["providers", "--json"], capsys)
    payload = json.loads(output)
    assert code == 3
    assert payload["status"] == "UNAVAILABLE"
    assert payload["data"]["product_result"]["data"]["feature_registered"] is False


def test_no_argument_v3_reserves_local_web_dispatch(capsys) -> None:
    code, output = _invoke(product_cli.main, [], capsys)
    assert code == 3
    assert "web interface is not registered" in output
    assert "usage:" not in output.lower()


def test_user_status_redacts_developer_evidence(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(product_cli, "build_project_state", lambda _config: _state(tmp_path))
    monkeypatch.setattr(product_cli, "_load", lambda **_kwargs: ProjectConfig(tmp_path, None, {}))
    code, output = _invoke(product_cli.main, ["status", "--json"], capsys)
    assert code == 0
    assert "abc123-internal-commit" not in output
    assert "deadbeef" not in output
    assert "sha256" not in output
    assert "audit" not in json.dumps(json.loads(output)["project_state"]).lower()


def test_developer_status_may_expose_detailed_evidence(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(dev_cli, "build_project_state", lambda _config: _state(tmp_path))
    monkeypatch.setattr(dev_cli, "_load", lambda: ProjectConfig(tmp_path, None, {}))
    code, output = _invoke(dev_cli.main, ["status", "--json"], capsys)
    assert code == 0
    assert "abc123-internal-commit" in output
    assert "deadbeef" in output
    assert "sha256" in output
    assert '"audit": "STALE"' in output


def test_loopback_web_server_is_default() -> None:
    settings = WebServerSettings()
    settings.validate()
    assert settings.host == "127.0.0.1"
    assert settings.is_loopback is True


def test_non_loopback_binding_requires_explicit_option_and_authentication() -> None:
    with pytest.raises(WebSecurityError, match="explicit"):
        WebServerSettings(host="0.0.0.0").validate()
    with pytest.raises(WebSecurityError, match="authentication"):
        WebServerSettings(host="0.0.0.0", allow_non_loopback=True).validate()
    WebServerSettings(host="0.0.0.0", allow_non_loopback=True, authentication_token="runtime-only").validate()


def test_secrets_are_rejected_from_persisted_configuration(tmp_path: Path) -> None:
    (tmp_path / "spritelab.yaml").write_text(
        "providers:\n  vision:\n    type: hosted\n    api_key: do-not-store\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Secrets are not allowed"):
        ProjectConfig.load(tmp_path)


def test_existing_configuration_remains_compatible(tmp_path: Path) -> None:
    (tmp_path / "spritelab.yaml").write_text("project:\n  name: old-project\n  schema_version: 3\n", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)
    assert config.values["ui"] == {"open_browser": True, "host": "127.0.0.1", "port": "auto"}
    assert config.values["providers"]["vision"]["type"] == "auto"
    assert config.values["compute"]["training"]["type"] == "local"


def test_event_roundtrip_uses_shared_schema() -> None:
    event = ProductEvent(
        run_id="run-1",
        timestamp="2026-07-13T10:00:00+00:00",
        feature="training",
        stage="prepare",
        event_type="progress",
        status=ProductStatus.RUNNING,
        current=2,
        total=5,
        message="Preparing.",
        metrics={"rate": 1.5},
        artifact_references=("artifact://plan",),
    )
    payload = event.to_dict()
    assert payload["schema_version"] == PRODUCT_EVENT_SCHEMA
    assert set(payload) == {
        "schema_version",
        "run_id",
        "timestamp",
        "feature",
        "stage",
        "event_type",
        "status",
        "current",
        "total",
        "message",
        "metrics",
        "artifact_references",
    }
    assert ProductEvent.from_dict(json.loads(json.dumps(payload))) == event


@pytest.mark.parametrize("status", list(ProductStatus))
def test_status_serialization(status: ProductStatus) -> None:
    payload = ProductResult(status, "Serialized.").to_dict()
    assert payload["status"] == status.value


def test_existing_low_level_commands_and_namespaces_remain_registered() -> None:
    assert {
        "v3",
        "dev",
        "curation",
        "train",
        "training",
        "dataset-maker",
        "harvest",
        "ml",
        "eval",
        "palette-report",
        "export-training",
        "readiness",
    } <= set(package_main._COMMANDS)
