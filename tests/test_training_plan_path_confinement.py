from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import spritelab.product_features.training.plans as plans_module
from spritelab.product_core import ProjectContext
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.plans import (
    TrainingPlanResolver,
    synthetic_training_path_contract_for_tests,
)
from spritelab.product_features.training.service import TrainingService
from spritelab.remote_compute import FakeComputeBackend
from spritelab.v3.config import DEFAULT_CONFIG
from spritelab.v3.model import AuditStatus


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight_only_resolver() -> TrainingPlanResolver:
    def unexpected_activation(*_args, **_kwargs):
        pytest.fail("conditioned activation must follow path preflight")

    return TrainingPlanResolver(activation_loader=unexpected_activation)


def _context(
    root: Path,
    campaign_config: str,
    *,
    real_config: bool,
    synthetic_contract: bool = False,
) -> ProjectContext:
    values = deepcopy(DEFAULT_CONFIG)
    values["training"]["campaign_config"] = campaign_config
    values["execution"]["allow_training"] = True
    if synthetic_contract:
        values.update(synthetic_training_path_contract_for_tests(root))
    config_path = root / "spritelab.yaml"
    if real_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# production-backed test config\n", encoding="utf-8")
    return ProjectContext(root, values, config_path, root / "runs/v3")


def _campaign_document(root: Path) -> tuple[dict, dict[str, Path]]:
    inputs = {name: root / "inputs" / f"{name}.json" for name in ("freeze", "view", "split", "vocabulary", "benchmark")}
    for name, path in inputs.items():
        _write_json(path, {"name": name})
    spec = {
        "campaign_id": "path-confinement-test",
        "identities": {
            "dataset_freeze_path": "inputs/freeze.json",
            "dataset_view_manifest_path": "inputs/view.json",
            "split_manifest_path": "inputs/split.json",
            "conditioning_vocabulary_path": "inputs/vocabulary.json",
        },
        "evaluation": {"benchmark_manifest_path": "inputs/benchmark.json"},
        "output_root": "runs/training",
    }
    return {"product_profiles": {"recommended": {"campaign": spec}}}, inputs


def _install_safe_planning_stubs(monkeypatch: pytest.MonkeyPatch, root: Path) -> TrainingPlanResolver:
    output = root / "runs" / "training" / "campaign" / "cell" / "seed_1"
    campaign = {
        "campaign_identity": "a" * 64,
        "seeds": [11, 29, 47],
        "checkpoint_schedule": {},
        "expected_runs": [{"run_id": "run-1", "output_root": str(output)}],
    }
    monkeypatch.setattr(
        plans_module,
        "validate_campaign",
        lambda _campaign: {"launch_ready": True, "errors": [], "blockers": []},
    )
    monkeypatch.setattr(
        plans_module,
        "audit_resume",
        lambda _campaign, **_kwargs: {
            "safe": True,
            "errors": [],
            "runs": [{"run_id": "run-1", "output_root": str(output), "status": "fresh"}],
            "foreign_run_roots": [],
        },
    )

    def load_activation(*_args, **kwargs):
        assert kwargs["require_audit"] is False
        return SimpleNamespace(
            audit_status=AuditStatus.PASS,
            campaign=deepcopy(campaign),
            manifest={"image_count": 2_417},
        )

    return TrainingPlanResolver(activation_loader=load_activation)


@pytest.mark.parametrize("shape", ["contained_absolute", "external_absolute", "traversal"])
def test_real_config_rejects_unsafe_campaign_path_before_any_read_and_redacts_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    root = tmp_path / shape / "project"
    root.mkdir(parents=True)
    contained = root / "campaign.json"
    outside = tmp_path / shape / "outside-sentinel.json"
    _write_json(contained, {"secret": "contained"})
    _write_json(outside, {"secret": "outside"})
    before = outside.read_bytes()
    raw = {
        "contained_absolute": str(contained),
        "external_absolute": str(outside),
        "traversal": "../outside-sentinel.json",
    }[shape]
    context = _context(root, raw, real_config=True, synthetic_contract=True)
    monkeypatch.setattr(plans_module, "_read_mapping", lambda *_args, **_kwargs: pytest.fail("unsafe read"))

    result = TrainingService(context, FakeComputeBackend()).status().to_dict()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "BLOCKED"
    assert any(item["code"] == "path_confinement" for item in result["blockers"])
    assert str(root) not in rendered and str(outside) not in rendered
    assert outside.read_bytes() == before


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("identities", "dataset_freeze_path"),
        ("identities", "dataset_view_manifest_path"),
        ("identities", "split_manifest_path"),
        ("identities", "conditioning_vocabulary_path"),
        ("evaluation", "benchmark_manifest_path"),
    ],
)
def test_external_bound_inputs_are_refused_before_status_or_campaign_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: str,
    field: str,
) -> None:
    root = tmp_path / f"{container}-{field}" / "project"
    document, _inputs = _campaign_document(root)
    outside = root.parent / "outside-sentinel.json"
    _write_json(outside, {"secret": field})
    before = outside.read_bytes()
    spec = document["product_profiles"]["recommended"]["campaign"]
    spec[container][field] = "../outside-sentinel.json"
    campaign_path = root / "campaign.json"
    _write_json(campaign_path, document)
    context = _context(root, "campaign.json", real_config=True)

    plan = _preflight_only_resolver().resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )

    assert [gate.gate_id for gate in plan.blockers] == ["path_confinement"]
    assert str(outside) not in json.dumps(plan.to_dict(), sort_keys=True)
    assert outside.read_bytes() == before


def test_nested_hard_link_is_refused_without_reading_the_outside_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside-sentinel.json"
    nested = root / "profiles" / "campaign.json"
    _write_json(outside, {"secret": "hard-link"})
    nested.parent.mkdir(parents=True)
    os.link(outside, nested)
    before = outside.read_bytes()
    top = root / "campaign.json"
    _write_json(top, {"product_profiles": {"recommended": {"campaign_path": "profiles/campaign.json"}}})
    context = _context(root, "campaign.json", real_config=True)
    reads: list[Path] = []
    original_read = plans_module._read_mapping

    def recording_read(path, **kwargs):
        reads.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(plans_module, "_read_mapping", recording_read)

    plan = _preflight_only_resolver().resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )

    assert reads == [top]
    assert [gate.gate_id for gate in plan.blockers] == ["path_confinement"]
    assert outside.read_bytes() == before


def test_hard_linked_campaign_config_is_refused_before_content_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside-campaign-sentinel.json"
    campaign = root / "campaign.json"
    _write_json(outside, {"secret": "hard-linked-campaign"})
    campaign.parent.mkdir(parents=True)
    os.link(outside, campaign)
    before = outside.read_bytes()
    context = _context(root, "campaign.json", real_config=True)
    monkeypatch.setattr(plans_module, "_read_mapping", lambda *_args, **_kwargs: pytest.fail("campaign read"))

    plan = _preflight_only_resolver().resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )

    assert [gate.gate_id for gate in plan.blockers] == ["path_confinement"]
    assert outside.read_bytes() == before


def test_linked_output_seam_is_refused_without_touching_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    document, _inputs = _campaign_document(root)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    linked = root / "linked-output"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory links are unavailable: {exc}")
    document["product_profiles"]["recommended"]["campaign"]["output_root"] = "linked-output/run"
    _write_json(root / "campaign.json", document)
    context = _context(root, "campaign.json", real_config=True)

    plan = _preflight_only_resolver().resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )

    assert [gate.gate_id for gate in plan.blockers] == ["path_confinement"]
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (
            ".json",
            '{"product_profiles":{"recommended":{"campaign":'
            '{"output_root":"private-first","output_root":"private-second"}}}}',
        ),
        (
            ".yaml",
            "product_profiles:\n"
            "  recommended:\n"
            "    campaign:\n"
            "      output_root: private-first\n"
            "      output_root: private-second\n",
        ),
    ],
)
def test_campaign_reader_refuses_nested_duplicate_keys_with_pathless_public_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    payload: str,
) -> None:
    root = tmp_path / "project"
    campaign_path = root / f"private-campaign{suffix}"
    campaign_path.parent.mkdir(parents=True)
    campaign_path.write_text(payload, encoding="utf-8")
    context = _context(root, campaign_path.name, real_config=True)
    monkeypatch.setattr(
        plans_module,
        "_select_campaign_spec_with_origin",
        lambda *_args, **_kwargs: pytest.fail("ambiguous campaign mapping was interpreted"),
    )

    plan = _preflight_only_resolver().resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )
    projection = json.dumps(plan.to_dict(), sort_keys=True)

    blocker_by_id = {gate.gate_id: gate for gate in plan.blockers}
    assert "profile_translation" in blocker_by_id
    assert blocker_by_id["profile_translation"].message == (
        "The configured training profile could not be selected safely."
    )
    assert str(campaign_path) not in projection
    assert "private-first" not in projection
    assert "private-second" not in projection


@pytest.mark.parametrize(
    "raw",
    ["foo/../bar.json", "../foo/../bar.json", ".", "..", "runs/./x", "runs//x", "runs/x/"],
)
def test_portable_campaign_paths_reject_nonleading_parent_or_reducible_spelling(
    tmp_path: Path,
    raw: str,
) -> None:
    root = tmp_path / "project"
    origin = root / "campaigns"
    origin.mkdir(parents=True)
    policy = plans_module._TrainingPathPolicy(root=root, allow_absolute_fixture_paths=False)

    with pytest.raises(plans_module.TrainingPathConfinementError) as captured:
        plans_module._confined_path(
            raw,
            origin=origin,
            policy=policy,
            allow_parent_parts=True,
            allow_absolute=False,
        )

    assert str(root) not in str(captured.value)


def test_portable_campaign_paths_allow_only_a_leading_parent_prefix(tmp_path: Path) -> None:
    root = tmp_path / "project"
    origin = root / "campaigns"
    target = root / "inputs" / "freeze.json"
    _write_json(target, {"ok": True})
    origin.mkdir(parents=True)
    policy = plans_module._TrainingPathPolicy(root=root, allow_absolute_fixture_paths=False)

    resolved = plans_module._confined_existing_file(
        "../inputs/freeze.json",
        origin=origin,
        policy=policy,
        allow_parent_parts=True,
        allow_absolute=False,
    )

    assert resolved == target.resolve()


def test_confined_mapping_read_is_capped_at_four_mib_with_pathless_refusal(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    policy = plans_module._TrainingPathPolicy(root=root, allow_absolute_fixture_paths=False)
    exact = root / "exact.json"
    prefix = b'{"ok":true}'
    exact.write_bytes(prefix + (b" " * (plans_module._MAX_CONFINED_MAPPING_BYTES - len(prefix))))

    assert plans_module._read_mapping(exact, path_policy=policy) == {"ok": True}

    oversized = root / "private-oversized.json"
    oversized.write_bytes(b"{" + (b" " * plans_module._MAX_CONFINED_MAPPING_BYTES))
    with pytest.raises(plans_module.TrainingPathConfinementError) as captured:
        plans_module._read_mapping(oversized, path_policy=policy)

    assert str(oversized) not in str(captured.value)


def test_explicit_synthetic_contract_allows_only_contained_absolute_fixture_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    document, inputs = _campaign_document(root)
    spec = document["product_profiles"]["recommended"]["campaign"]
    for field, key in (
        ("dataset_freeze_path", "freeze"),
        ("dataset_view_manifest_path", "view"),
        ("split_manifest_path", "split"),
        ("conditioning_vocabulary_path", "vocabulary"),
    ):
        spec["identities"][field] = str(inputs[key])
    spec["evaluation"]["benchmark_manifest_path"] = str(inputs["benchmark"])
    spec["output_root"] = str(root / "runs" / "training")
    campaign_path = root / "campaign.json"
    _write_json(campaign_path, document)
    context = _context(root, str(campaign_path), real_config=False, synthetic_contract=True)
    resolver = _install_safe_planning_stubs(monkeypatch, root)

    plan = resolver.resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )
    projection = json.dumps(plan.to_dict(), sort_keys=True)

    assert plan.ready is True
    assert "build_project_state" not in TrainingPlanResolver.resolve.__code__.co_names
    assert "path_confinement" not in {gate.gate_id for gate in plan.gates}
    assert str(root) not in projection
    assert '"paths_exposed": false' in projection

    outside = tmp_path / "external.json"
    _write_json(outside, {"secret": "still external"})
    document["product_profiles"]["recommended"]["campaign"]["identities"]["dataset_view_manifest_path"] = str(outside)
    _write_json(campaign_path, document)
    blocked = resolver.resolve(
        context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )
    assert [gate.gate_id for gate in blocked.blockers] == ["path_confinement"]
