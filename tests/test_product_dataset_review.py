from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import ProjectContext, create_product_app
from spritelab.product_features.dataset.intake import build_dataset
from spritelab.product_features.dataset.plugin import build_plugin
from spritelab.product_features.dataset.review import DatasetReviewStore, ReviewDecisionError
from test_product_dataset_helpers import make_configured, make_opaque_background, make_png


def _queue(output: Path) -> dict:
    return json.loads((output / "review_queue.json").read_text(encoding="utf-8"))


def test_review_gui_is_prefilled_and_keyboard_driven(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "rejected.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    context = ProjectContext(tmp_path, config={"dataset": {"output_root": str(output)}})
    client = TestClient(create_product_app(context, plugins=(build_plugin(),)))
    response = client.get("/dataset/review")
    assert response.status_code == 200
    assert "Keep" in response.text and "Exclude" in response.text
    assert 'data-current-decision="exclude"' in response.text
    assert "Contact sheet" in response.text
    assert "ArrowRight" in response.text and "confirm-exclusions" in response.text
    assert "Technical check" in response.text


def test_review_data_defaults_only_intake_exceptions_visible(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "rejected.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    items = _queue(output)["items"]
    assert all(item["default_visible"] for item in items if item["queue_kind"] == "intake_exception")
    assert {item["current_disposition"] for item in items if item["default_visible"]} <= {
        "rejected",
        "uncertain",
        "requires_special_extraction",
    }


def test_rescue_suitability_false_rejection_rebuilds_image_dataset(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "false-positive.png")
    output = tmp_path / "out"
    result = build_dataset(root, output_root=output)
    assert result.data["counts"]["accepted"] == 0
    item = _queue(output)["items"][0]
    decision = DatasetReviewStore(output).apply(item["item_id"], "keep")
    assert decision["current_disposition"] == "accepted"
    machine = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert machine["data"]["counts"]["accepted"] == 1
    assert (output / "raw_extraction" / "extraction_manifest.jsonl").is_file()


def test_batch_confirm_exclusions_is_append_only(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_opaque_background(root / "a.png")
    make_opaque_background(root / "b.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    store = DatasetReviewStore(output)
    first = store.confirm_all_current_exclusions()
    before = (output / "review_log.jsonl").read_text(encoding="utf-8")
    second = store.confirm_all_current_exclusions()
    after = (output / "review_log.jsonl").read_text(encoding="utf-8")
    assert first["confirmed"] == second["confirmed"] == 2
    assert after.startswith(before) and len(after) > len(before)


def test_review_cannot_override_missing_license(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "source.txt").write_text(
        "Name: Test Pack\nCreator: Test Artist\nhttps://example.test/source\n", encoding="utf-8"
    )
    make_png(root / "one.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    item = _queue(output)["items"][0]
    assert "legal" in item["reason_categories"]
    with pytest.raises(ReviewDecisionError, match="source/license evidence"):
        DatasetReviewStore(output).apply(item["item_id"], "keep")


def test_review_api_rejects_legal_fabrication(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "source.txt").write_text("Name: Test Pack\nCreator: Test Artist\n", encoding="utf-8")
    make_png(root / "one.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    item = _queue(output)["items"][0]
    context = ProjectContext(tmp_path, config={"dataset": {"output_root": str(output)}})
    client = TestClient(create_product_app(context, plugins=(build_plugin(),)))
    response = client.post(f"/dataset/review/items/{item['item_id']}/decision", json={"decision": "keep"})
    assert response.status_code == 409
    assert "evidence" in response.json()["detail"]


def test_plugin_status_cards_follow_latest_result(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    context = ProjectContext(tmp_path, config={"dataset": {"output_root": str(output)}})
    plugin = build_plugin()
    status = plugin.status_provider(context)
    assert status.data["counts"]["processed"] == 1
    assert {card["id"] for card in status.data["status_cards"]} >= {"processed", "accepted", "image-only"}
