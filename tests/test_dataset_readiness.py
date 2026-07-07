from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.harvest.dataset_readiness import scan_readiness, write_readiness_reports

SPLITS = ("train", "val", "test")


def _semantic() -> dict:
    return {
        "schema_version": "semantic_v3.0",
        "category": "item_icon",
        "object_name": "ruby_gem",
        "base_object": "gem",
        "open_name": "ruby gem",
        "attributes": {"colors": ["red"], "materials": ["glass"]},
        "captions": ["ruby gem", "red gem", "32x32 pixel art ruby gem"],
        "negative_tags": ["photorealistic"],
    }


def _record(sprite_id: str, split: str, *, semantic: bool = True) -> dict:
    record = {
        "sprite_id": sprite_id,
        "split": split,
        "category": "item_icon",
        "category_id": 1,
        "object_name": "ruby_gem",
        "tags": ["gem", "red"],
        "source_name": "test",
        "source_path": f"data/{sprite_id}.png",
        "license": "cc0",
        "label_v2": {"applied": True, "bucket": "auto_filename_trusted", "flags": ["auto"]},
    }
    if semantic:
        record["semantic_v3"] = _semantic()
    return record


def _write_exported_dataset(
    root: Path,
    name: str,
    *,
    semantic: bool = True,
    dataset_qa: dict | None = None,
    training_manifest_qa: dict | None = None,
) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records_by_split = {
        "train": [_record(f"s{i}", "train", semantic=semantic) for i in range(4)],
        "val": [_record("v0", "val", semantic=semantic)],
        "test": [_record("t0", "test", semantic=semantic)],
    }
    for split in SPLITS:
        records = records_by_split[split]
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
        )
        count = len(records)
        np.savez_compressed(
            dataset_dir / f"{split}.npz",
            alpha=np.zeros((count, 32, 32), dtype=np.uint8),
            sprite_id=np.array([r["sprite_id"] for r in records], dtype=np.str_),
        )
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": name, "max_palette_slots": 32}), encoding="utf-8"
    )
    if dataset_qa is not None:
        (dataset_dir / "dataset_qa_report.json").write_text(json.dumps(dataset_qa), encoding="utf-8")
    if training_manifest_qa is not None:
        (dataset_dir / "training_manifest_qa_report.json").write_text(
            json.dumps(training_manifest_qa), encoding="utf-8"
        )
    return dataset_dir


def _write_run(
    root: Path,
    name: str,
    *,
    predictions: bool = True,
    semantic: bool = False,
    review_rate: float = 0.05,
    apply_accepted: int | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "imported.jsonl").write_text(
        "\n".join(json.dumps({"sprite_id": f"{name}_{i}"}) for i in range(10)) + "\n", encoding="utf-8"
    )
    if predictions:
        review_count = int(10 * review_rate)
        rows = []
        for i in range(10):
            review = i < review_count
            rows.append(
                {
                    "sprite_id": f"{name}_{i}",
                    "bucket": "needs_review" if review else "auto_filename_trusted",
                    "needs_review": review,
                    "safe_prefill": {"category": "material", "object_name": "ruby_gem"},
                    "label_quality": {
                        "bucket": "needs_review" if review else "auto_filename_trusted",
                        "needs_review": review,
                    },
                }
            )
        (run_dir / "label_v2_suggestions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        (run_dir / "label_v2_summary.json").write_text(
            json.dumps({"total": 10, "auto_count": 10 - review_count, "review_rate": review_rate, "top_categories": {"material": 10}}),
            encoding="utf-8",
        )
    if semantic:
        (run_dir / "label_v2_suggestions_semantic_v3.jsonl").write_text(
            "\n".join(json.dumps({"sprite_id": f"{name}_{i}", "semantic_v3": _semantic()}) for i in range(10)) + "\n",
            encoding="utf-8",
        )
    if apply_accepted is not None:
        (run_dir / "label_v2_apply_report.json").write_text(
            json.dumps({"applied_auto_labels": apply_accepted, "accepted_auto_labels": apply_accepted}),
            encoding="utf-8",
        )
    return run_dir


def _find(report, run_name: str):
    for pack in report.packs:
        if pack.run_name == run_name:
            return pack
    raise AssertionError(f"pack {run_name} not found")


# ---------------------------------------------------------------------------


def test_scanner_handles_empty_roots(tmp_path: Path) -> None:
    report = scan_readiness(tmp_path / "no_runs", tmp_path / "no_datasets")
    assert report.packs == []


def test_scanner_detects_exported_semantic_v3_dataset(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": []}, training_manifest_qa={"errors": []})
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA_label_v2_semantic_v3")
    assert pack.has_exported_dataset is True
    assert pack.semantic_v3_coverage == 1.0
    assert pack.base_object_coverage == 1.0
    assert pack.caption_count_average == 3.0


def test_scanner_detects_missing_training_manifest(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": []})
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA_label_v2_semantic_v3")
    assert pack.dataset_qa_status == "pass"
    assert pack.training_manifest_status == "missing"
    assert pack.recommended_action == "needs_training_manifest"


def test_scanner_detects_dataset_qa_report(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(
        datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": ["boom"]}
    )
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA_label_v2_semantic_v3")
    assert pack.dataset_qa_status == "fail"
    assert pack.recommended_action == "manual_attention_required"
    assert any("boom" in reason for reason in pack.top_reasons)


def test_scanner_missing_object_name_qa_failure_is_manual_attention(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(
        datasets,
        "packA_label_v2_semantic_v3",
        dataset_qa={"errors": ["record missing object_name: s0"]},
    )
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA_label_v2_semantic_v3")
    assert pack.recommended_action == "manual_attention_required"


def test_scanner_classifies_ready_for_merge(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(
        datasets,
        "packA_label_v2_semantic_v3",
        dataset_qa={"errors": []},
        training_manifest_qa={"errors": []},
    )
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA_label_v2_semantic_v3")
    assert pack.recommended_action == "ready_for_merge"
    assert pack.is_atomic_dataset is True
    assert pack.is_merge_input_candidate is True


def test_scanner_classifies_needs_semantic_v3(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _write_run(runs, "packB", predictions=True, semantic=False, review_rate=0.05)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packB")
    assert pack.has_label_v2_predictions is True
    assert pack.has_semantic_v3_predictions is False
    assert pack.recommended_action == "needs_semantic_v3"


def test_scanner_classifies_needs_export_with_semantic_predictions(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _write_run(runs, "packC", predictions=True, semantic=True, review_rate=0.05, apply_accepted=9)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packC")
    assert pack.has_semantic_v3_predictions is True
    assert pack.recommended_action == "needs_export"
    assert pack.semantic_v3_coverage == 1.0


def test_scanner_flags_too_many_review_items(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _write_run(runs, "packD", predictions=True, semantic=True, review_rate=0.5)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packD")
    assert pack.recommended_action == "too_many_review_items"


def test_scanner_all_needs_review_predictions_are_not_needs_export(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _write_run(runs, "packReview", predictions=True, semantic=True, review_rate=1.0)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packReview")
    assert pack.raw_prediction_review_count == 10
    assert pack.recommended_action == "needs_source_family_profile"


def test_scanner_zero_apply_acceptance(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _write_run(runs, "packZero", predictions=True, semantic=True, review_rate=1.0, apply_accepted=0)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packZero")
    assert pack.recommended_action == "zero_apply_acceptance"


def test_scanner_classifies_aggregate_dataset_not_merge_input(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    dataset_dir = _write_exported_dataset(
        datasets,
        "sprite_lab_multisource_v1",
        dataset_qa={"errors": []},
        training_manifest_qa={"errors": []},
    )
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": "sprite_lab_multisource_v1", "created_by": "spritelab.harvest.merge_datasets"}),
        encoding="utf-8",
    )
    report = scan_readiness(runs, datasets)
    pack = _find(report, "sprite_lab_multisource_v1")
    assert pack.is_aggregate_dataset is True
    assert pack.is_merge_input_candidate is False
    assert pack.recommended_action == "aggregate_dataset_not_merge_input"


def test_scanner_classifies_legacy_nonsemantic_dataset(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(datasets, "legacy_pack_label_v2", semantic=False, dataset_qa={"errors": []}, training_manifest_qa={"errors": []})
    report = scan_readiness(runs, datasets)
    pack = _find(report, "legacy_pack_label_v2")
    assert pack.recommended_action == "legacy_nonsemantic_dataset"


def test_scanner_flags_empty_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (runs / "empty_run").mkdir(parents=True)
    report = scan_readiness(runs, datasets)
    pack = _find(report, "empty_run")
    assert pack.recommended_action == "empty_or_import_broken"


def test_scanner_writes_json_and_markdown(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    _write_exported_dataset(
        datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": []}, training_manifest_qa={"errors": []}
    )
    report = scan_readiness(runs, datasets)
    out_md = tmp_path / "out" / "report.md"
    out_json = tmp_path / "out" / "report.json"
    write_readiness_reports(report, out_md=out_md, out_json=out_json)
    assert out_md.is_file()
    assert out_json.is_file()
    payload = json.loads(out_json.read_text())
    assert payload["pack_count"] == 1
    assert "packA_label_v2_semantic_v3" in payload["ready_for_merge"]
    assert "Dataset Readiness Report" in out_md.read_text()


def test_scanner_does_not_mutate_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    runs.mkdir()
    dataset_dir = _write_exported_dataset(
        datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": []}, training_manifest_qa={"errors": []}
    )
    _write_run(runs, "packB", predictions=True, semantic=True)
    before = {p.name: p.read_bytes() for p in dataset_dir.iterdir()}
    run_before = {p.name: p.read_bytes() for p in (runs / "packB").iterdir()}
    scan_readiness(runs, datasets)
    after = {p.name: p.read_bytes() for p in dataset_dir.iterdir()}
    run_after = {p.name: p.read_bytes() for p in (runs / "packB").iterdir()}
    assert before == after
    assert run_before == run_after


def test_run_matched_to_semantic_dataset(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    datasets = tmp_path / "datasets"
    _write_run(runs, "packA", predictions=True, semantic=True)
    # Two exported datasets map to the same pack; the semantic one wins.
    _write_exported_dataset(datasets, "packA_label_v2", dataset_qa={"errors": []})
    _write_exported_dataset(
        datasets, "packA_label_v2_semantic_v3", dataset_qa={"errors": []}, training_manifest_qa={"errors": []}
    )
    report = scan_readiness(runs, datasets)
    pack = _find(report, "packA")
    assert pack.exported_dataset_path.endswith("packA_label_v2_semantic_v3")
    assert pack.recommended_action == "ready_for_merge"
