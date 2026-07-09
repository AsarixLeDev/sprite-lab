from __future__ import annotations

import json
from pathlib import Path

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.qa import qa_dataset, write_reports
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.dataset_maker.training_manifest_qa import qa_training_manifest, write_training_manifest_qa_reports
from spritelab.harvest.build_multisource import build_multisource_dataset, select_atomic_ready_datasets


def test_build_multisource_selects_only_atomic_ready_and_passes_qa(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    atomic = _write_ready_dataset(datasets / "atomic_pack_label_v2_semantic_v3")
    aggregate = _write_ready_dataset(datasets / "sprite_lab_multisource_v9")
    (aggregate / "merge_report.json").write_text(json.dumps({"total_records": 1}), encoding="utf-8")
    legacy = _write_ready_dataset(datasets / "legacy_pack")

    selected, excluded = select_atomic_ready_datasets(datasets, only_atomic_ready=True)
    assert selected == [atomic]
    assert excluded["sprite_lab_multisource_v9"] == "aggregate_dataset"
    assert excluded["legacy_pack"] == "legacy_nonsemantic_dataset"

    report = build_multisource_dataset(
        datasets,
        datasets / "sprite_lab_multisource_v4",
        only_atomic_ready=True,
        variants_per_sprite=2,
        overwrite=True,
    )

    assert report.ok
    assert report.total_records == len(default_specs())
    assert str(atomic) in report.selected_datasets
    assert "sprite_lab_multisource_v9" not in "\n".join(report.selected_datasets)
    out = Path(report.output_dir)
    assert (out / "dataset_qa_report.json").is_file()
    assert not json.loads((out / "dataset_qa_report.json").read_text(encoding="utf-8"))["errors"]
    assert not json.loads((out / "training_manifest_qa_report.json").read_text(encoding="utf-8"))["errors"]
    assert (out / "eval_prompts.jsonl").is_file()
    assert (out / "build_multisource_report.json").is_file()


def _write_ready_dataset(dataset_dir: Path) -> Path:
    make_semantic_dataset(dataset_dir, default_specs())
    qa = qa_dataset(dataset_dir, require_semantic_v3=True)
    write_reports(qa, out_json=dataset_dir / "dataset_qa_report.json", out_md=dataset_dir / "dataset_qa_report.md")
    manifest = build_training_manifest(dataset_dir, variants_per_sprite=2, caption_policy="mixed", seed=1)
    manifest_path = dataset_dir / "training_manifest.jsonl"
    write_training_manifest(manifest_path, manifest.rows)
    tm_qa = qa_training_manifest(dataset_dir, manifest_path)
    write_training_manifest_qa_reports(
        tm_qa,
        out_json=dataset_dir / "training_manifest_qa_report.json",
        out_md=dataset_dir / "training_manifest_qa_report.md",
    )
    assert not qa.errors, qa.errors
    assert not tm_qa.errors, tm_qa.errors
    return dataset_dir
