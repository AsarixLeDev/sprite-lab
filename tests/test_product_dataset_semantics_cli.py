from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests._labeling_audit import audit_context, write_labeling_audit

from spritelab.product_core import (
    CONDITIONED_VIEW_CANDIDATES,
    CONSERVATIVE_PROPOSAL_GENERATION,
    ProductAction,
    ProductCapability,
    ProductResult,
    ProductStatus,
    ProjectContext,
)
from spritelab.product_features.dataset import cli as dataset_cli
from spritelab.product_features.dataset.intake import DatasetIntakeService
from spritelab.product_features.dataset.plugin import build_plugin
from spritelab.product_web.app import create_app
from spritelab.v3 import cli as v3_cli
from test_product_dataset_helpers import make_configured, make_opaque_background, make_png


class FakeVisionProvider:
    provider_id = "fake.vision"
    title = "Fake vision"

    def __init__(self, *, abstain: bool = False, healthy: bool = True, confidence: float = 0.95) -> None:
        self.abstain = abstain
        self.healthy = healthy
        self.confidence = confidence
        self.actions: list[ProductAction] = []

    def probe(self, _context: ProjectContext) -> tuple[ProductCapability, ...]:
        return (
            ProductCapability(
                "vision.proposals",
                "Vision proposals",
                ProductStatus.READY if self.healthy else ProductStatus.BLOCKED,
            ),
        )

    def execute(self, action: ProductAction, _context: ProjectContext, _emit) -> ProductResult:
        self.actions.append(action)
        proposals = []
        for item in action.parameters["items"]:
            if self.abstain:
                proposals.append({"item_id": item["item_id"], "abstained": True, "reason": "not enough evidence"})
            else:
                proposals.append(
                    {
                        "item_id": item["item_id"],
                        "labels": {"category": "object", "canonical_object": "unknown"},
                        "confidence": self.confidence,
                        "health_ok": True,
                    }
                )
        return ProductResult(ProductStatus.COMPLETE, "Synthetic proposals.", data={"proposals": proposals})


def _certified_context(tmp_path: Path, *scopes: str) -> ProjectContext:
    root = Path.cwd()
    report, manifest = write_labeling_audit(
        root,
        tmp_path / "labeling-audit",
        scopes=tuple(scopes) or (CONSERVATIVE_PROPOSAL_GENERATION,),
    )
    return audit_context(root, report, manifest)


def test_no_provider_completes_image_only_dataset(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    output = tmp_path / "out"
    context = ProjectContext(
        tmp_path,
        {
            "dataset": {"output_root": str(output)},
            "labeling": {
                "hierarchical_enabled": True,
                "hierarchical_profile": "fast_local",
                "reference_cohort_size": 400,
            },
        },
    )
    result = DatasetIntakeService().build(root, output_root=output, context=context)
    assert result.data["counts"]["image_only_eligible"] == 1
    assert result.data["semantic"]["provider_status"] == "not_configured"
    assert result.data["semantic"]["conditioned_dataset_ready"] is False
    client = TestClient(create_app(context, plugins=(build_plugin(),)))
    queue = client.get("/labeling/api/queue").json()
    assert queue["items"] == []
    assert queue["provider_prefill_pending"] == 1
    assert queue["human_truth_required_for_all"] is False


def test_provider_is_invoked_through_shared_contract(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    provider = FakeVisionProvider()
    context = _certified_context(tmp_path, CONSERVATIVE_PROPOSAL_GENERATION, CONDITIONED_VIEW_CANDIDATES)
    result = DatasetIntakeService(provider).build(root, output_root=tmp_path / "out", context=context)
    assert provider.actions[0].action_id == "dataset.semantic.propose"
    assert provider.actions[0].parameters["proposals_are_human_truth"] is False
    assert result.data["counts"]["semantically_labeled"] == 1
    assert result.data["semantic"]["conditioned_dataset_ready"] is True
    semantic = json.loads((tmp_path / "out" / "items.jsonl").read_text(encoding="utf-8").splitlines()[0])["semantic"]
    assert semantic["triage_state"] == "auto_prefilled"
    assert semantic["semantic_supervision_eligible"] is True
    assert semantic["keep_requires_human_truth"] is False
    queue = json.loads((tmp_path / "out" / "review_queue.json").read_text(encoding="utf-8"))
    assert not any(item["queue_kind"] == "semantic_exception" for item in queue["items"])


def test_low_certainty_prefill_is_rejected_unless_human_rescues_it(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    output = tmp_path / "out"
    base = _certified_context(tmp_path, CONSERVATIVE_PROPOSAL_GENERATION)
    context = ProjectContext(
        base.project_root,
        {
            **base.config,
            "dataset": {"output_root": str(output)},
            "labeling": {
                **base.config["labeling"],
                "hierarchical_enabled": True,
                "hierarchical_profile": "fast_local",
                "reference_cohort_size": 400,
            },
        },
    )
    DatasetIntakeService(FakeVisionProvider(confidence=0.4)).build(
        root,
        output_root=output,
        context=context,
    )
    item = json.loads((output / "items.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert item["current_disposition"] == "accepted"
    assert item["semantic"]["triage_state"] == "low_certainty_rejected"
    assert item["semantic"]["semantic_supervision_eligible"] is False
    assert item["semantic"]["keep_requires_human_truth"] is True
    intake_queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
    exception = next(row for row in intake_queue["items"] if row["queue_kind"] == "semantic_exception")
    assert exception["automatic_decision"] == "exclude_semantic_supervision"

    client = TestClient(create_app(context, plugins=(build_plugin(),)))
    queue = client.get("/labeling/api/queue").json()
    assert queue["human_truth_required_for_all"] is False
    assert queue["low_certainty_default"] == "exclude_semantic_supervision"
    assert queue["low_certainty_remaining"] == 1
    assert len(queue["items"]) == 1
    reviewed = client.post(
        f"/labeling/api/review/{item['item_id']}",
        headers={"x-csrf-token": client.app.state.spritelab_csrf_token},
        json={
            "action": "choose_alternative",
            "selected_node": "object",
            "reviewer_identity": "test-reviewer",
            "partition": "reference",
            "submission_token": "low-certainty-rescue-1",
        },
    )
    assert reviewed.status_code == 200
    assert client.get("/labeling/api/queue").json()["items"] == []


def test_provider_abstention_is_preserved_for_exception_review(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    context = _certified_context(tmp_path, CONSERVATIVE_PROPOSAL_GENERATION)
    result = DatasetIntakeService(FakeVisionProvider(abstain=True)).build(
        root, output_root=tmp_path / "out", context=context
    )
    assert result.data["counts"]["semantically_abstained"] == 1
    queue = json.loads((tmp_path / "out" / "review_queue.json").read_text(encoding="utf-8"))
    semantic = next(item for item in queue["items"] if item["queue_kind"] == "semantic_exception")
    assert semantic["semantic"]["state"] == "abstained"
    assert semantic["semantic"]["truth_status"] == "provider_proposal_not_human_truth"


def test_semantic_health_failure_preserves_image_only_dataset(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    provider = FakeVisionProvider(healthy=False)
    result = DatasetIntakeService(provider).build(
        root,
        output_root=tmp_path / "out",
        context=_certified_context(tmp_path, CONSERVATIVE_PROPOSAL_GENERATION),
    )
    assert provider.actions == []
    assert result.status == ProductStatus.COMPLETE
    assert result.data["counts"]["image_only_eligible"] == 1
    assert result.data["semantic"]["health_ok"] is False


@pytest.mark.parametrize("failure_stage", ["probe", "execute"])
def test_provider_failure_details_are_redacted_from_dataset_artifacts(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    secret = r"C:\private\provider-token=do-not-persist"

    class FailingProvider(FakeVisionProvider):
        def probe(self, context: ProjectContext) -> tuple[ProductCapability, ...]:
            if failure_stage == "probe":
                raise RuntimeError(secret)
            return super().probe(context)

        def execute(self, action: ProductAction, context: ProjectContext, emit) -> ProductResult:
            if failure_stage == "execute":
                raise RuntimeError(secret)
            return super().execute(action, context, emit)

    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    output = tmp_path / "out"
    result = DatasetIntakeService(FailingProvider()).build(
        root,
        output_root=output,
        context=_certified_context(tmp_path, CONSERVATIVE_PROPOSAL_GENERATION),
    )

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output / "result.json", output / "report_data.json", output / "items.jsonl")
    )
    assert result.status == ProductStatus.COMPLETE
    assert result.data["counts"]["image_only_eligible"] == 1
    assert secret not in persisted
    assert secret not in json.dumps(result.data)
    assert "RuntimeError" in persisted


def test_human_labels_csv_do_not_require_provider(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    (root / "labels.csv").write_text("filename,category,canonical_object\none.png,object,potion\n", encoding="utf-8")
    result = DatasetIntakeService().build(root, output_root=tmp_path / "out")
    assert result.data["counts"]["semantically_labeled"] == 1
    assert result.data["semantic"]["conditioned_dataset_ready"] is False


def _invoke(args: list[str], capsys) -> tuple[int, str]:
    with pytest.raises(SystemExit) as caught:
        v3_cli.main(args, plugins=(build_plugin(),))
    return int(caught.value.code), capsys.readouterr().out


def test_cli_folder_build_and_optional_review_skipped(tmp_path: Path, capsys, monkeypatch) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "exception.png")
    make_png(root / "accepted.png", color=(40, 180, 230, 255))
    output = tmp_path / "out"
    monkeypatch.setattr(dataset_cli, "launch_review_interface", lambda _path: pytest.fail("review must be optional"))
    code, text = _invoke(["dataset", "build", str(root), "--output", str(output), "--no-review", "--json"], capsys)
    payload = json.loads(text)
    assert code == 0
    assert payload["data"]["product_result"]["data"]["counts"]["processed"] == 2


def test_noninteractive_mode_never_opens_browser(tmp_path: Path, capsys, monkeypatch) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "exception.png")
    output = tmp_path / "out"
    DatasetIntakeService().build(root, output_root=output)
    monkeypatch.setattr(dataset_cli, "launch_review_interface", lambda _path: pytest.fail("browser launch forbidden"))
    code, text = _invoke(["review", "--result", str(output), "--json"], capsys)
    assert code == 4
    assert json.loads(text)["data"]["product_result"]["data"]["browser_opened"] is False


def test_exact_plugin_registration_function() -> None:
    plugin = build_plugin()
    assert plugin.plugin_id == "dataset.intake"
    assert callable(plugin.cli_registration)
    assert plugin.web_router_factory is not None
