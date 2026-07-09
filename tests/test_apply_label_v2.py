from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions
from spritelab.harvest.cli import main


def test_apply_label_v2_cli_reads_prediction_file_applies_safe_and_writes_reports(tmp_path: Path, capsys) -> None:
    run = _write_apply_run(tmp_path)

    main(
        [
            "apply-label-v2",
            "--run",
            str(run),
            "--prediction-file",
            "custom_predictions.jsonl",
            "--mode",
            "auto-only",
            "--accept-auto",
        ]
    )

    output = capsys.readouterr().out
    assert "Predictions: 4" in output
    assert "Matched imported sprites: 3" in output
    assert "Applied auto labels: 1" in output
    assert "Skipped review labels: 1" in output
    assert "Missing predictions: 1" in output
    assert "Missing imported sprites: 1" in output
    assert "Accepted auto labels: 1" in output
    assert "Human labels preserved: 1" in output

    imported = _read_by_id(run / "imported.jsonl")
    safe = imported["safe_sprite"]
    assert safe["status"] == "accepted"
    assert safe["category"] == "armor"
    assert safe["object_name"] == "chestplate"
    assert safe["tags"] == ["chestplate", "armor", "metal"]
    assert safe["notes"] == "A metal chestplate icon."
    assert safe["materials"] == ["metal"]
    assert safe["mood"] == ["defensive"]
    assert safe["source_name"] == "Source Pack"
    assert safe["license"] == "cc0"
    assert safe["palette_size"] == 7
    assert safe["has_role_map"] is True

    auto_metadata = safe["auto_metadata"]
    assert auto_metadata["qwen_suggestion"]["object_name"] == "old_qwen"
    assert auto_metadata["label_v2_applied"] is True
    assert auto_metadata["label_v2_prediction_file"] == "custom_predictions.jsonl"
    assert auto_metadata["label_v2_bucket"] == "auto_rpg_496_specialized"
    assert auto_metadata["label_v2_safe_prefill"]["object_name"] == "chestplate"
    assert auto_metadata["label_v2_vlm_descriptor"]["object_name"] == "armor"
    assert auto_metadata["label_v2_candidate_object_names"][:2] == ["chestplate", "armor"]

    review = imported["review_sprite"]
    assert review["category"] == "old_review"
    assert review["status"] == "quarantine"
    assert "label_v2_applied" not in review.get("auto_metadata", {})

    review_queue = [
        json.loads(line) for line in (run / "label_v2_review_queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(review_queue) == 1
    assert review_queue[0]["sprite_id"] == "review_sprite"
    assert review_queue[0]["bucket"] == "needs_review_candidate_conflict"
    assert review_queue[0]["safe_prefill"]["object_name"] == "mystery_item"
    assert review_queue[0]["candidate_object_names"] == ["mystery_item", "potion"]
    assert "object_name: filename=mystery_item, vlm=potion" in review_queue[0]["conflict_reasons"]

    report_json = json.loads((run / "label_v2_apply_report.json").read_text(encoding="utf-8"))
    assert report_json["counts_by_bucket"]["auto_rpg_496_specialized"] == 3
    assert report_json["counts_by_category"]["armor"] == 2
    assert report_json["missing_prediction_sprite_ids"] == ["missing_prediction_sprite"]
    assert report_json["missing_imported_sprite_ids"] == ["orphan_prediction"]
    assert (run / "label_v2_apply_report.md").exists()


def test_apply_label_v2_preserves_human_labels_by_default_and_overwrites_when_requested(tmp_path: Path) -> None:
    run = _write_apply_run(tmp_path)

    apply_label_v2_predictions(run, prediction_file="custom_predictions.jsonl")
    imported = _read_by_id(run / "imported.jsonl")
    human = imported["human_sprite"]
    assert human["category"] == "weapon"
    assert human["object_name"] == "reviewed_sword"
    assert human["tags"] == ["reviewed", "sword"]
    assert human["notes"] == "human note"
    assert "label_v2_applied" not in human["auto_metadata"]

    apply_label_v2_predictions(run, prediction_file="custom_predictions.jsonl", overwrite_human_labels=True)
    imported = _read_by_id(run / "imported.jsonl")
    human = imported["human_sprite"]
    assert human["category"] == "armor"
    assert human["object_name"] == "helmet"
    assert human["tags"] == ["helmet", "armor"]
    assert human["notes"] == "A helmet icon."
    assert human["auto_metadata"]["label_v2_applied"] is True


def test_apply_label_v2_dry_run_writes_no_mutated_imported_file(tmp_path: Path) -> None:
    run = _write_apply_run(tmp_path)
    before = (run / "imported.jsonl").read_text(encoding="utf-8")

    report = apply_label_v2_predictions(
        run,
        prediction_file="custom_predictions.jsonl",
        accept_auto=True,
        dry_run=True,
    )

    after = (run / "imported.jsonl").read_text(encoding="utf-8")
    assert after == before
    assert report["dry_run"] is True
    assert report["applied_auto_labels"] == 1
    assert report["wrote_imported"] is None
    assert not (run / "label_v2_review_queue.jsonl").exists()
    assert not (run / "label_v2_apply_report.json").exists()


def test_apply_label_v2_review_only_mode_applies_review_without_accepting(tmp_path: Path) -> None:
    run = _write_apply_run(tmp_path)

    report = apply_label_v2_predictions(
        run, prediction_file="custom_predictions.jsonl", mode="review-only", accept_auto=True
    )

    imported = _read_by_id(run / "imported.jsonl")
    review = imported["review_sprite"]
    assert report["applied_review_labels"] == 1
    assert report["applied_auto_labels"] == 0
    assert review["status"] == "quarantine"
    assert review["category"] == "item_icon"
    assert review["object_name"] == "mystery_item"
    assert review["auto_metadata"]["label_v2_bucket"] == "needs_review_candidate_conflict"


def test_accept_auto_quarantines_review_records_that_were_already_accepted(tmp_path: Path) -> None:
    run = _write_apply_run(tmp_path, review_status="accepted")

    report = apply_label_v2_predictions(run, prediction_file="custom_predictions.jsonl", accept_auto=True)

    imported = _read_by_id(run / "imported.jsonl")
    assert imported["safe_sprite"]["status"] == "accepted"
    assert imported["review_sprite"]["status"] == "quarantine"
    assert imported["review_sprite"]["category"] == "old_review"
    assert report["review_labels_quarantined"] == 1


def _write_apply_run(tmp_path: Path, *, review_status: str = "quarantine") -> Path:
    run = tmp_path / "run"
    run.mkdir()
    imported = [
        _imported_record("safe_sprite", category="old_auto", tags=["old"], notes="old note"),
        _imported_record(
            "review_sprite", status=review_status, category="old_review", tags=["review_old"], notes="review old"
        ),
        _imported_record(
            "human_sprite",
            category="weapon",
            object_name="reviewed_sword",
            tags=["reviewed", "sword"],
            notes="human note",
            auto_metadata={"human_label": True, "labeler": "tester"},
        ),
        _imported_record("missing_prediction_sprite", category="old_missing"),
    ]
    _write_jsonl(run / "imported.jsonl", imported)
    predictions = [
        _prediction(
            "safe_sprite",
            bucket="auto_rpg_496_specialized",
            category="armor",
            object_name="chestplate",
            tags=["chestplate", "armor", "metal"],
            short_description="A metal chestplate icon.",
            materials=["metal"],
            mood=["defensive"],
            flags=["auto_rpg_496_specialized"],
            vlm_object_name="armor",
            candidates=["chestplate", "armor"],
        ),
        _prediction(
            "review_sprite",
            bucket="needs_review_candidate_conflict",
            category="item_icon",
            object_name="mystery_item",
            tags=["mystery"],
            short_description="A mystery item.",
            flags=["filename_vlm_conflict"],
            conflict_reasons=["object_name: filename=mystery_item, vlm=potion"],
            needs_review=True,
            vlm_object_name="potion",
            candidates=["mystery_item", "potion"],
        ),
        _prediction(
            "human_sprite",
            bucket="auto_rpg_496_specialized",
            category="armor",
            object_name="helmet",
            tags=["helmet", "armor"],
            short_description="A helmet icon.",
            flags=["auto_rpg_496_specialized"],
            vlm_object_name="helmet",
            candidates=["helmet", "armor"],
        ),
        _prediction(
            "orphan_prediction",
            bucket="auto_rpg_496_specialized",
            category="material",
            object_name="gold_coin",
            tags=["coin", "gold"],
            short_description="A coin icon.",
            flags=["auto_rpg_496_specialized"],
            vlm_object_name="coin",
            candidates=["gold_coin", "coin"],
        ),
    ]
    _write_jsonl(run / "custom_predictions.jsonl", predictions)
    return run


def _imported_record(
    sprite_id: str,
    *,
    status: str = "quarantine",
    category: str = "unknown",
    object_name: str = "",
    tags: list[str] | None = None,
    notes: str = "",
    auto_metadata: dict | None = None,
) -> dict:
    metadata = {"qwen_suggestion": {"object_name": "old_qwen"}}
    metadata.update(auto_metadata or {})
    return {
        "sprite_id": sprite_id,
        "candidate_id": f"candidate_{sprite_id}",
        "source_id": "source_pack",
        "final_png_path": f"{sprite_id}.png",
        "relative_path": f"{sprite_id}.png",
        "status": status,
        "category": category,
        "object_name": object_name,
        "tags": list(tags or []),
        "notes": notes,
        "source_name": "Source Pack",
        "license": "cc0",
        "author": "Artist",
        "palette_size": 7,
        "has_role_map": True,
        "errors": [],
        "warnings": [],
        "auto_metadata": metadata,
    }


def _prediction(
    sprite_id: str,
    *,
    bucket: str,
    category: str,
    object_name: str,
    tags: list[str],
    short_description: str,
    flags: list[str],
    vlm_object_name: str,
    candidates: list[str],
    materials: list[str] | None = None,
    mood: list[str] | None = None,
    conflict_reasons: list[str] | None = None,
    needs_review: bool = False,
) -> dict:
    return {
        "sprite_id": sprite_id,
        "relative_path": f"{sprite_id}.png",
        "candidate_object_names": candidates,
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": tags,
            "short_description": short_description,
            "materials": list(materials or []),
            "mood": list(mood or []),
        },
        "label_quality": {
            "bucket": bucket,
            "flags": flags,
            "needs_review": needs_review,
            "conflict_reasons": list(conflict_reasons or []),
        },
        "bucket": bucket,
        "needs_review": needs_review,
        "flags": flags,
        "conflict_reasons": list(conflict_reasons or []),
        "vlm_descriptor": {
            "object_name": vlm_object_name,
            "alternative_object_names": [object_name],
            "source_consistency": "consistent",
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _read_by_id(path: Path) -> dict[str, dict]:
    return {
        record["sprite_id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }
