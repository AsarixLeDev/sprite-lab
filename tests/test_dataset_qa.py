from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.dataset_maker.qa import (
    _color_tokens_compatible,
    build_contact_sheet,
    compare_dataset_manifest_parity,
    qa_dataset,
    write_reports,
)


def _bundle_arrays(sprite_ids: list[str]) -> dict[str, np.ndarray]:
    count = len(sprite_ids)
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, 33, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, 33), dtype=bool)
    for row in range(count):
        alpha[row, 10:14, 10:14] = 1
        index_map[row, 10:14, 10:14] = 1
        palette[row, 1] = [120, 50, 80]
        palette_mask[row, 0] = True
        palette_mask[row, 1] = True
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": role_map,
        "palette": palette,
        "palette_mask": palette_mask,
        "category_id": np.zeros((count,), dtype=np.int64),
        "sprite_id": np.array(sprite_ids, dtype=np.str_),
    }


def _record(sprite_id: str, split: str, **overrides) -> dict:
    record = {
        "sprite_id": sprite_id,
        "split": split,
        "category": "item_icon",
        "category_id": 1,
        "object_name": "potion",
        "tags": ["potion", "glass", "red"],
        "source_name": "test-source",
        "source_path": f"data/{sprite_id}.png",
        "license": "cc0",
        "label_v2": {"applied": True, "bucket": "auto_test", "flags": ["auto_test"]},
    }
    record.update(overrides)
    return record


def _write_dataset(
    tmp_path: Path,
    records_by_split: dict[str, list[dict]],
    *,
    ids_by_split: dict[str, list[str]] | None = None,
    with_config: bool = True,
) -> Path:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = records_by_split.get(split, [])
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        ids = ids_by_split[split] if ids_by_split else [str(r["sprite_id"]) for r in records]
        np.savez_compressed(dataset_dir / f"{split}.npz", **_bundle_arrays(ids))
    if with_config:
        (dataset_dir / "dataset_config.json").write_text(
            json.dumps({"dataset_name": "ds", "max_palette_slots": 32}), encoding="utf-8"
        )
        (dataset_dir / "vocab.json").write_text(
            json.dumps({"category_to_id": {"unknown": 0, "item_icon": 1}}), encoding="utf-8"
        )
    return dataset_dir


def _valid_dataset(tmp_path: Path) -> Path:
    records = {
        "train": [_record(f"s{i}", "train") for i in range(8)],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    return _write_dataset(tmp_path, records)


def test_qa_passes_on_valid_dataset(tmp_path: Path) -> None:
    result = qa_dataset(_valid_dataset(tmp_path))
    assert result.ok, result.errors
    assert result.errors == []
    assert result.total_records == 10
    assert result.total_images == 10
    assert result.splits == {"train": 8, "val": 1, "test": 1}


def test_missing_image_is_error(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    # Manifest references s0..s7 but npz only stores s0..s6.
    ids = [f"s{i}" for i in range(7)]
    np.savez_compressed(dataset_dir / "train.npz", **_bundle_arrays(ids))
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("no raster in npz" in e for e in result.errors)
    assert "s7" in result.image_checks["missing_images"]


def test_bad_dimensions_is_error(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    arrays = _bundle_arrays(["s0"])
    arrays["alpha"] = np.zeros((1, 16, 16), dtype=np.uint8)
    arrays["index_map"] = np.zeros((1, 16, 16), dtype=np.int16)
    np.savez_compressed(dataset_dir / "train.npz", **arrays)
    # keep manifest consistent with the single sprite
    (dataset_dir / "manifest_train.jsonl").write_text(
        json.dumps(_record("s0", "train"), sort_keys=True) + "\n", encoding="utf-8"
    )
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert result.image_checks["all_32x32"] is False
    assert any("dimensions" in e for e in result.errors)


def test_duplicate_sprite_id_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("dup", "train")],
        "val": [_record("dup", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "dup" in result.manifest_checks["duplicate_sprite_ids"]


def test_missing_object_name_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("s0", "train", object_name="")],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "s0" in result.manifest_checks["missing_object_name"]


def test_empty_tags_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("s0", "train", tags=[])],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "s0" in result.manifest_checks["missing_tags"]


def test_label_v2_not_applied_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("s0", "train", label_v2={"applied": False, "bucket": "b", "flags": []})],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "s0" in result.label_v2_checks["applied_not_true"]


def test_review_queue_overlap_is_error(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    queue = tmp_path / "review.jsonl"
    queue.write_text(json.dumps({"sprite_id": "s3"}) + "\n", encoding="utf-8")
    result = qa_dataset(dataset_dir, review_queue=queue)
    assert not result.ok
    assert "s3" in result.review_queue_overlap
    assert any("review queue" in e for e in result.errors)


def test_split_overlap_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("shared", "train"), _record("s1", "train")],
        "val": [_record("shared", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "shared" in result.split_checks["overlap"]


def test_split_count_must_match_npz(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    # npz has an extra sprite the manifest never lists -> count mismatch + unreferenced.
    ids = [f"s{i}" for i in range(8)] + ["ghost"]
    np.savez_compressed(dataset_dir / "train.npz", **_bundle_arrays(ids))
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("count does not match" in e for e in result.errors)


def test_forbidden_object_names_are_caught(tmp_path: Path) -> None:
    for bad in ("sho", "armour", "elm"):
        records = {
            "train": [_record("s0", "train", object_name=bad)],
            "val": [_record("v0", "val")],
            "test": [_record("t0", "test")],
        }
        dataset_dir = _write_dataset(tmp_path / bad, records)
        result = qa_dataset(dataset_dir)
        assert not result.ok, bad
        assert any(bad in entry for entry in result.manifest_checks["forbidden_object_names"])


def test_potion_color_only_names_are_caught_for_496(tmp_path: Path) -> None:
    records = {
        "train": [
            _record(
                "oga_496_rpg_icons_32fix_p_blue01",
                "train",
                object_name="blue",
                source_path="data_sources/fixed/oga_496_rpg_icons_32/P_Blue01.png",
            )
        ],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = tmp_path / "oga_496_rpg_icons_32fix_label_v2"
    dataset_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        recs = records[split]
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n", encoding="utf-8"
        )
        np.savez_compressed(dataset_dir / f"{split}.npz", **_bundle_arrays([str(r["sprite_id"]) for r in recs]))
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": "oga_496_rpg_icons_32fix_label_v2", "max_palette_slots": 32}),
        encoding="utf-8",
    )
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert result.manifest_checks["color_only_potion_object_names"]


def test_reports_and_contact_sheet_are_written(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    result = qa_dataset(dataset_dir)
    out_json = dataset_dir / "dataset_qa_report.json"
    out_md = dataset_dir / "dataset_qa_report.md"
    write_reports(result, out_json=out_json, out_md=out_md)
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_records"] == 10
    assert "## Splits" in out_md.read_text(encoding="utf-8")

    sheet = dataset_dir / "dataset_qa_contact_sheet.png"
    assert build_contact_sheet(dataset_dir, sheet, sample_limit=16) == sheet
    assert sheet.exists()


def test_review_status_leak_is_error(tmp_path: Path) -> None:
    records = {
        "train": [_record("s0", "train", needs_review=True)],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert result.manifest_checks["review_status_leaks"]


def test_qa_does_not_mutate_dataset(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in dataset_dir.iterdir()}
    qa_dataset(dataset_dir)
    after = {p.name: p.stat().st_mtime_ns for p in dataset_dir.iterdir()}
    assert before == after


def test_label_v2_audits_are_report_only_and_parity_ignores_new_metadata(tmp_path: Path) -> None:
    dataset_dir = _valid_dataset(tmp_path)
    result = qa_dataset(dataset_dir)
    assert result.label_audits["report_only"] is True
    assert "filename_vlm_category_mismatch" in result.label_audits["evidence_unavailable"]

    train_path = dataset_dir / "manifest_train.jsonl"
    row = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    row["label_confidence_tier"] = "T2"  # explicitly ignored by parity
    changed_dir = tmp_path / "changed"
    changed_dir.mkdir()
    for split in ("train", "val", "test"):
        source = dataset_dir / f"manifest_{split}.jsonl"
        target = changed_dir / source.name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    changed_train = [
        json.loads(line) for line in (changed_dir / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    changed_train[0]["label_confidence_tier"] = "T2"
    (changed_dir / "manifest_train.jsonl").write_text(
        "\n".join(json.dumps(value) for value in changed_train) + "\n", encoding="utf-8"
    )
    assert compare_dataset_manifest_parity(dataset_dir, changed_dir)["unexpected_changed_count"] == 0


def test_hallucination_audit_suppresses_trusted_same_potion_family(tmp_path: Path) -> None:
    record = _record(
        "potion",
        "train",
        label_v2={
            "applied": True,
            "bucket": "auto_rpg_496_specialized",
            "flags": ["vlm_conflicts_with_filename"],
            "safe_prefill": {"object_name": "blue_potion"},
            "vlm_descriptor": {"object_name": "potion"},
        },
    )
    dataset_dir = _write_dataset(
        tmp_path, {"train": [record], "val": [_record("v0", "val")], "test": [_record("t0", "test")]}
    )
    audits = qa_dataset(dataset_dir).label_audits["entries"]
    assert any(entry["code"] == "vlm_hallucination_denylist_suppressed" for entry in audits)
    assert not any(
        entry["code"] == "vlm_hallucination_denylist_hit" and "potion" in entry["sprite_ids"] for entry in audits
    )

    weak = _record(
        "weak_potion",
        "train",
        label_v2={
            "applied": True,
            "bucket": "auto_vlm_when_filename_weak",
            "flags": [],
            "safe_prefill": {"object_name": "blue_potion"},
            "vlm_descriptor": {"object_name": "potion"},
        },
    )
    weak_dir = _write_dataset(
        tmp_path / "weak", {"train": [weak], "val": [_record("v1", "val")], "test": [_record("t1", "test")]}
    )
    weak_audits = qa_dataset(weak_dir).label_audits["entries"]
    assert any(
        entry["code"] == "vlm_hallucination_denylist_hit" and "weak_potion" in entry["sprite_ids"]
        for entry in weak_audits
    )

    conflicted = _record(
        "conflicted_potion",
        "train",
        label_v2={
            "applied": True,
            "bucket": "auto_rpg_496_specialized",
            "flags": ["needs_review_candidate_conflict"],
            "safe_prefill": {"object_name": "blue_potion"},
            "vlm_descriptor": {"object_name": "potion"},
        },
    )
    conflict_dir = _write_dataset(
        tmp_path / "conflict", {"train": [conflicted], "val": [_record("v2", "val")], "test": [_record("t2", "test")]}
    )
    conflict_audits = qa_dataset(conflict_dir).label_audits["entries"]
    assert any(
        entry["code"] == "vlm_hallucination_denylist_hit" and "conflicted_potion" in entry["sprite_ids"]
        for entry in conflict_audits
    )


def test_color_family_matching_is_conservative_and_canonical() -> None:
    assert _color_tokens_compatible({"blue"}, ["teal"])
    assert _color_tokens_compatible({"pink"}, ["magenta"])
    assert _color_tokens_compatible({"purple"}, ["magenta"])
    assert _color_tokens_compatible({"gold"}, ["yellow"])
    assert _color_tokens_compatible({"gray"}, ["black"])
    assert not _color_tokens_compatible({"pink"}, ["purple"])
    assert not _color_tokens_compatible({"red"}, ["orange"])
