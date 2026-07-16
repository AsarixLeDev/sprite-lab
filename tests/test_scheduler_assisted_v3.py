from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from spritelab.harvest.label_v3.assisted_v3_gui import _resume_index
from spritelab.harvest.label_v3.record_decisions import RecordDecision, record_decision_to_json
from spritelab.harvest.label_v3.scheduler_input import (
    append_completed_id,
    append_quality_decision,
    prepare_scheduler_v3,
)
from spritelab.utils.jsonl import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pool = tmp_path / "pool"
    blobs = pool / "blobs"
    blobs.mkdir(parents=True)
    harvest = tmp_path / "harvest_runs"
    cohort_rows = []
    candidates = []
    for index, sprite_id in enumerate(("from_run_a", "from_blob", "from_run_b")):
        rgba = bytes([20 + index, 40, 60, 255] * 16)
        digest = hashlib.sha256(rgba).hexdigest()
        (blobs / f"{digest}.rgba").write_bytes(rgba)
        candidates.append(
            {
                "sprite_id": sprite_id,
                "source_id": f"source_{index}",
                "source_run": "run_a" if index == 0 else "run_b" if index == 2 else "missing_run",
                "source_runs": ["run_a" if index == 0 else "run_b" if index == 2 else "missing_run"],
                "source_image": f"{sprite_id}.png",
                "exported_width": 4,
                "exported_height": 4,
                "exported_rgba_hash": digest,
                "blob_path": f"blobs/{digest}.rgba",
                "broad_pack_type": "definitely_not_reviewed_truth",
            }
        )
        cohort_rows.append(
            {
                "sprite_id": sprite_id,
                "representative_id": sprite_id,
                "batch_number": 1,
                "broad_type": "definitely_not_reviewed_truth",
                "geometry_group": f"geometry_{index}",
                "propagation_count": index,
                "suitability": {"status": "accept", "reason_codes": []},
                "cohort_context": {
                    "mode": "semantic_accept_only",
                    "selection_index": index,
                    "original_batch_index": 10 - index,
                },
            }
        )
    _write_jsonl(pool / "candidate_manifest.jsonl", candidates)
    for run_name, sprite_id in (("run_a", "from_run_a"), ("run_b", "from_run_b")):
        run = harvest / run_name
        image = run / f"{sprite_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(image)
        _write_jsonl(
            run / "imported.jsonl",
            [{"sprite_id": sprite_id, "source_id": run_name, "final_png_path": str(image)}],
        )
    existing = RecordDecision(sprite_id="from_run_b")
    _write_jsonl(
        harvest / "run_b" / "v3_output" / "v3_v3_records.jsonl",
        [record_decision_to_json(existing)],
    )
    cohort = tmp_path / "cohort.jsonl"
    _write_jsonl(cohort, cohort_rows)
    return cohort, pool, harvest


def test_scheduler_prepare_preserves_order_resolves_multi_run_and_blob(tmp_path: Path):
    cohort, pool, harvest = _fixture(tmp_path)
    before_pool = (pool / "candidate_manifest.jsonl").read_bytes()
    before_cohort = cohort.read_bytes()
    result = prepare_scheduler_v3(cohort, pool, tmp_path / "work", harvest_root=harvest)
    records = read_jsonl(result.records_path)
    resolved = read_jsonl(result.resolved_candidates_path)
    expected = ["from_run_a", "from_blob", "from_run_b"]
    assert [row["sprite_id"] for row in records] == expected
    assert [row["sprite_id"] for row in resolved] == expected
    assert result.harvest_resolved == 2
    assert result.blob_resolved == 1
    assert result.reused_prefills == 1
    assert result.generated_deterministic_prefills == 2
    assert result.estimated_propagated_variants == 3
    assert all("broad_pack_type" not in row for row in resolved)
    assert all(row["prefill_metadata"]["scheduler_context"]["broad_type_is_reviewed_truth"] is False for row in records)
    assert (pool / "candidate_manifest.jsonl").read_bytes() == before_pool
    assert cohort.read_bytes() == before_cohort
    assert not (tmp_path / "work" / "completed_representative_ids.jsonl").exists()


def test_scheduler_prepare_resume_and_append_only_completion(tmp_path: Path):
    cohort, pool, harvest = _fixture(tmp_path)
    work = tmp_path / "work"
    first = prepare_scheduler_v3(cohort, pool, work, harvest_root=harvest)
    second = prepare_scheduler_v3(cohort, pool, work, harvest_root=harvest)
    assert second.candidate_ids == first.candidate_ids
    assert second.generated_deterministic_prefills == 0
    assert second.reused_prefills == 3
    saved = {"current_index": 2, "candidate_ids": list(first.candidate_ids), "cohort_hash": "abc"}
    assert _resume_index(saved, first.candidate_ids, "abc") == 2
    assert _resume_index(saved, tuple(reversed(first.candidate_ids)), "abc") == 0
    assert _resume_index(saved, first.candidate_ids, "different") == 0

    completed = work / "completed.jsonl"
    assert append_completed_id(completed, "from_run_a", cohort_hash="abc") is True
    before = completed.read_bytes()
    assert append_completed_id(completed, "from_run_a", cohort_hash="abc") is False
    assert completed.read_bytes() == before
    assert append_completed_id(completed, "from_blob", cohort_hash="abc") is True
    assert [json.loads(line)["representative_id"] for line in completed.read_text().splitlines()] == [
        "from_run_a",
        "from_blob",
    ]


def test_quality_decisions_are_separate_from_semantic_records(tmp_path: Path):
    output = tmp_path / "quality.jsonl"
    append_quality_decision(
        output,
        "sprite",
        "quality_uncertain",
        suitability_reason_codes=("LARGE_PALETTE",),
        reviewer_id="reviewer",
    )
    event = json.loads(output.read_text(encoding="utf-8"))
    assert event["quality_decision"] == "quality_uncertain"
    assert event["suitability_reason_codes"] == ["LARGE_PALETTE"]
    assert not ({"category", "canonical_object", "tags"} & event.keys())
