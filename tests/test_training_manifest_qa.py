from __future__ import annotations

import json
from pathlib import Path

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.dataset_maker.training_manifest_qa import qa_training_manifest


def _dataset(tmp_path: Path) -> Path:
    return make_semantic_dataset(tmp_path / "ds", default_specs())


def _build(dataset: Path, **kw) -> Path:
    policy = kw.pop("caption_policy", "mixed")
    variants = kw.pop("variants_per_sprite", 8)
    result = build_training_manifest(dataset, variants_per_sprite=variants, caption_policy=policy, seed=1)
    path = dataset / "training_manifest.jsonl"
    write_training_manifest(path, result.rows)
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def test_valid_manifest_passes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    result = qa_training_manifest(dataset, manifest)
    assert result.ok, result.errors
    assert result.total_rows == 48
    assert result.unique_sprites == 6


def test_missing_manifest_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = qa_training_manifest(dataset, dataset / "missing.jsonl")
    assert not result.ok


def test_out_of_range_npz_row_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    rows[0]["npz_row"] = 9999
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("out of range" in e for e in result.errors)


def test_sprite_id_mismatch_with_npz_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    # Point a train row at a different valid row index (0) but keep a wrong sprite id.
    for row in rows:
        if row["split"] == "train" and row["npz_row"] != 0:
            row["npz_row"] = 0
            row["sprite_id"] = "definitely_not_row_zero"
            break
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("does not match npz row" in e for e in result.errors)


def test_missing_caption_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    rows[0]["caption"] = ""
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("invalid caption" in e for e in result.errors)


def test_forbidden_positive_content_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    rows[0]["caption"] = "photorealistic golden sword"
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("forbidden caption content" in e for e in result.errors)


def test_wrong_split_reference_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    rows[0]["split"] = "bogus"
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("invalid split" in e for e in result.errors)


def test_missing_semantic_schema_version_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    rows[0]["audit"]["semantic_schema_version"] = ""
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("semantic_schema_version" in e for e in result.errors)


def test_missing_npz_file_is_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    (dataset / "train.npz").unlink()
    rows = _read(manifest)
    _write(manifest, rows)
    result = qa_training_manifest(dataset, manifest)
    assert any("npz_file problem" in e for e in result.errors)


def test_duplicate_caption_is_error_by_default_but_allowed_with_flag(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = _build(dataset)
    rows = _read(manifest)
    # Force a duplicate (sprite_id, caption) pair.
    train_rows = [r for r in rows if r["split"] == "train"]
    train_rows[1]["caption"] = train_rows[0]["caption"]
    train_rows[1]["sprite_id"] = train_rows[0]["sprite_id"]
    _write(manifest, rows)

    strict = qa_training_manifest(dataset, manifest)
    assert any("duplicate (sprite_id, caption)" in e for e in strict.errors)

    lenient = qa_training_manifest(dataset, manifest, allow_duplicate_captions=True)
    assert not any("duplicate (sprite_id, caption)" in e for e in lenient.errors)
    assert any("duplicate" in w for w in lenient.warnings)
