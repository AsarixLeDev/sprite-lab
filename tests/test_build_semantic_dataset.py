from __future__ import annotations

import json
from pathlib import Path

import pytest

from _harvest_testdata import make_sprite_png
from spritelab.harvest.build_semantic_dataset import (
    BuildError,
    build_semantic_dataset,
    is_raw_label_v2_prediction_file,
)
from spritelab.harvest.cli import main

_PNG_NAMES = (
    "red_potion.png",
    "blue_potion.png",
    "green_potion.png",
    "yellow_potion.png",
    "healing_potion.png",
    "mana_potion.png",
)


_COLOR_TOKENS = {"red", "blue", "green", "yellow"}


def _make_run(tmp_path: Path, *, run_name: str = "tiny_pack") -> Path:
    png_dir = tmp_path / "pngs"
    for index, name in enumerate(_PNG_NAMES):
        # Distinct colors so the sprites are not exact duplicates.
        make_sprite_png(png_dir / name, color=(30 + index * 30, 60, 200 - index * 20))
    run_root = tmp_path / "harvest_runs"
    main(
        [
            "import-dir",
            "--dir",
            str(png_dir),
            "--run-name",
            run_name,
            "--run-root",
            str(run_root),
            "--source-id",
            "tiny_source",
            "--source-name",
            "Tiny Source",
            "--license",
            "cc0",
            "--author",
            "Tester",
            "--user-confirmed-license",
        ]
    )
    run_dir = run_root / run_name
    _write_safe_auto_predictions(run_dir)
    return run_dir


def _write_safe_auto_predictions(run_dir: Path) -> None:
    """Write deterministic safe-auto label-v2 suggestions from filenames."""

    predictions: list[dict] = []
    for line in (run_dir / "imported.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        sprite_id = str(record["sprite_id"])
        stem = Path(str(record.get("relative_path") or record.get("final_png_path") or sprite_id)).stem
        object_name = stem.lower()
        tokens = object_name.split("_")
        base = tokens[-1] if tokens else object_name
        color = next((t for t in tokens if t in _COLOR_TOKENS), "red")
        predictions.append(
            {
                "sprite_id": sprite_id,
                "bucket": "auto_filename_trusted",
                "flags": ["auto_filename_trusted"],
                "candidate_object_names": [object_name, base],
                "label_quality": {
                    "bucket": "auto_filename_trusted",
                    "needs_review": False,
                    "flags": ["auto_filename_trusted"],
                },
                "safe_prefill": {
                    "category": "item_icon",
                    "object_name": object_name,
                    "short_description": object_name.replace("_", " "),
                    "tags": [base, color],
                    "materials": ["glass"],
                    "mood": ["fantasy"],
                    "dominant_colors": [color],
                    "candidate_object_names": [object_name, base],
                },
            }
        )
    (run_dir / "label_v2_suggestions.jsonl").write_text(
        "\n".join(json.dumps(p, sort_keys=True) for p in predictions) + "\n", encoding="utf-8"
    )


def _read_ids(path: Path) -> list[str]:
    return [json.loads(line)["sprite_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_missing_prediction_file_fails_clearly(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    with pytest.raises(BuildError, match="prediction file not found"):
        build_semantic_dataset(
            run_dir,
            dataset_name="tiny_pack_label_v2_semantic_v3",
            output_root=tmp_path / "datasets",
            prediction_file="does_not_exist.jsonl",
        )


def test_rejects_semantic_v3_prediction_file(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    semantic_path = run_dir / "label_v2_suggestions_semantic_v3.jsonl"
    semantic_path.write_text((run_dir / "label_v2_suggestions.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(BuildError, match="expects raw label-v2 predictions"):
        build_semantic_dataset(
            run_dir,
            dataset_name="tiny_pack_label_v2_semantic_v3",
            output_root=tmp_path / "datasets",
            prediction_file=semantic_path.name,
        )


def test_raw_prediction_file_helper_excludes_semantic_outputs() -> None:
    assert is_raw_label_v2_prediction_file(Path("label_v2_suggestions_fresh_novlm.jsonl"))
    assert not is_raw_label_v2_prediction_file(Path("label_v2_suggestions_semantic_v3.jsonl"))
    assert not is_raw_label_v2_prediction_file(Path("label_v2_suggestions_semantic_v3_semantic_v3.jsonl"))


def test_builds_on_tiny_synthetic_run(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    report = build_semantic_dataset(
        run_dir,
        dataset_name="tiny_pack_label_v2_semantic_v3",
        output_root=tmp_path / "datasets",
        prediction_file="label_v2_suggestions.jsonl",
        variants_per_sprite=4,
        overwrite=True,
    )
    assert report.ok, report.steps
    dataset_dir = Path(report.output_dir)
    assert dataset_dir.is_dir()
    assert (dataset_dir / "train.npz").is_file()
    assert (dataset_dir / "training_manifest.jsonl").is_file()
    assert (dataset_dir / "eval_prompts.jsonl").is_file()
    assert report.dataset_qa_errors == 0
    assert report.training_manifest_qa_errors == 0
    assert report.accepted_records >= 1


def test_build_report_is_written(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    report = build_semantic_dataset(
        run_dir,
        dataset_name="tiny_pack_label_v2_semantic_v3",
        output_root=tmp_path / "datasets",
        variants_per_sprite=4,
        overwrite=True,
    )
    dataset_dir = Path(report.output_dir)
    assert (dataset_dir / "semantic_dataset_build_report.json").is_file()
    assert (dataset_dir / "semantic_dataset_build_report.md").is_file()
    payload = json.loads((dataset_dir / "semantic_dataset_build_report.json").read_text())
    assert payload["ok"] is True
    step_names = [step["step"] for step in payload["steps"]]
    assert step_names == [
        "semantic_v3",
        "apply_label_v2",
        "export",
        "dataset_qa",
        "build_training_manifest",
        "training_manifest_qa",
        "build_eval_prompts",
    ]


def test_auto_only_excludes_review_records(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    suggestions_path = run_dir / "label_v2_suggestions.jsonl"
    records = [json.loads(line) for line in suggestions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    # Force the first sprite into the review bucket; it must not be exported.
    review_id = str(records[0]["sprite_id"])
    records[0]["bucket"] = "needs_review"
    records[0].setdefault("label_quality", {})
    records[0]["label_quality"]["needs_review"] = True
    records[0]["needs_review"] = True
    suggestions_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8")

    report = build_semantic_dataset(
        run_dir,
        dataset_name="tiny_pack_label_v2_semantic_v3",
        output_root=tmp_path / "datasets",
        variants_per_sprite=4,
        overwrite=True,
    )
    assert report.ok, report.steps
    assert report.review_queue_size >= 1
    dataset_dir = Path(report.output_dir)
    exported_ids: list[str] = []
    for split in ("train", "val", "test"):
        manifest = dataset_dir / f"manifest_{split}.jsonl"
        if manifest.is_file():
            exported_ids.extend(_read_ids(manifest))
    assert review_id not in exported_ids


def test_auto_only_does_not_export_stale_accepted_without_current_apply(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    imported = [
        json.loads(line)
        for line in (run_dir / "imported.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stale = imported[0]
    stale["status"] = "accepted"
    stale["auto_metadata"] = {
        "label_v2_applied": True,
        "label_v2_prediction_file": "old.jsonl",
        "label_v2_bucket": "auto_filename_trusted",
        "label_v2_safe_prefill": {"category": "item_icon", "object_name": "old_potion"},
        "semantic_v3": {"schema_version": "semantic_v3.0", "base_object": "potion"},
    }
    for record in imported[1:]:
        record["status"] = "quarantine"
    (run_dir / "imported.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in imported) + "\n",
        encoding="utf-8",
    )
    prediction = {
        "sprite_id": imported[1]["sprite_id"],
        "bucket": "needs_review",
        "needs_review": True,
        "label_quality": {"bucket": "needs_review", "needs_review": True},
        "safe_prefill": {"category": "item_icon", "object_name": "blue_potion", "tags": ["potion"]},
    }
    (run_dir / "label_v2_suggestions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    with pytest.raises(BuildError, match="no current auto-only accepted sprites"):
        build_semantic_dataset(
            run_dir,
            dataset_name="tiny_pack_label_v2_semantic_v3",
            output_root=tmp_path / "datasets",
            variants_per_sprite=4,
            overwrite=True,
        )


def test_auto_only_exports_only_current_applied_records(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    imported = [
        json.loads(line)
        for line in (run_dir / "imported.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    imported[0]["status"] = "accepted"
    imported[0]["auto_metadata"] = {
        "label_v2_applied": True,
        "label_v2_prediction_file": "old.jsonl",
        "label_v2_bucket": "auto_filename_trusted",
        "label_v2_safe_prefill": {"category": "item_icon", "object_name": "old_potion"},
        "semantic_v3": {"schema_version": "semantic_v3.0", "base_object": "potion"},
    }
    (run_dir / "imported.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in imported) + "\n",
        encoding="utf-8",
    )
    suggestions = [
        json.loads(line)
        for line in (run_dir / "label_v2_suggestions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (run_dir / "label_v2_suggestions.jsonl").write_text(
        json.dumps(suggestions[1], sort_keys=True) + "\n", encoding="utf-8"
    )

    report = build_semantic_dataset(
        run_dir,
        dataset_name="tiny_pack_label_v2_semantic_v3",
        output_root=tmp_path / "datasets",
        variants_per_sprite=4,
        overwrite=True,
    )

    dataset_dir = Path(report.output_dir)
    rows = []
    for split in ("train", "val", "test"):
        manifest = dataset_dir / f"manifest_{split}.jsonl"
        rows.extend(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    assert [row["sprite_id"] for row in rows] == [suggestions[1]["sprite_id"]]
    assert rows[0]["object_name"]
    assert rows[0]["category"]
    assert rows[0]["label_v2"]["applied"] is True
    assert rows[0]["label_v2"]["applied_at_build_id"]
    assert rows[0]["semantic_v3"]


def test_run_directory_missing_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="harvest run directory not found"):
        build_semantic_dataset(
            tmp_path / "nope",
            dataset_name="x_label_v2_semantic_v3",
            output_root=tmp_path / "datasets",
        )
