from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spritelab.annotation_scheduler.cli import main
from spritelab.annotation_scheduler.scheduler import (
    ScheduleConfig,
    ScheduleView,
    build_schedule,
    mark_issued,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _make_pool(root: Path, specs: list[tuple[str, str, str, str, int, str]]) -> Path:
    """Specs are id, type, pack, artist, family size, suitability."""

    root.mkdir()
    candidates = []
    groups = []
    queue = []
    quarantine = []
    for index, (sprite_id, broad_type, pack, artist, family_size, suitability) in enumerate(specs):
        geometry = f"geometry__{index:04d}"
        members = [sprite_id] + [f"{sprite_id}_variant_{number}" for number in range(1, family_size)]
        components = {
            "unique_geometry": 30,
            "underrepresented_broad_pack_type": 10,
            "underrepresented_source_artist": 10,
            "variant_propagation_value": family_size - 1,
            "provenance_completeness": 10,
        }
        row = {
            "sprite_id": sprite_id,
            "annotation_representative": True,
            "variant_geometry_group": geometry,
            "source_id": pack,
            "pack_id": pack,
            "author": artist,
            "sub_artist": artist,
            "broad_pack_type": broad_type,
            "suitability_status": suitability,
            "suitability_score": 100.0 if suitability == "accept" else 80.0,
            "suitability_reason_codes": [] if suitability == "accept" else ["TEST_QUARANTINE"],
            "annotation_priority_components": components,
            "annotation_priority_score": 1000 - index,
            "normalized_alpha_hash": hashlib.sha256(f"alpha-{index}".encode()).hexdigest(),
            "exported_rgba_hash": hashlib.sha256(f"rgba-{index}".encode()).hexdigest(),
        }
        candidates.append(row)
        for number, member in enumerate(members[1:], 1):
            candidates.append(
                {
                    **row,
                    "sprite_id": member,
                    "annotation_representative": False,
                    "annotation_priority_score": 500 - number,
                }
            )
        groups.extend(
            [
                {
                    "group_id": geometry,
                    "group_kind": "geometry_family",
                    "members": members,
                    "representative_sprite_id": sprite_id,
                    "variant_count": family_size,
                },
                {
                    "group_id": f"alpha_mask_recolor__{index:04d}",
                    "group_kind": "alpha_mask_recolor",
                    "members": members,
                },
            ]
        )
        for member in members:
            queue.append(
                {
                    "sprite_id": member,
                    "variant_geometry_group": geometry,
                    "queue": "high_priority_unique_geometry" if suitability == "accept" else "quality_quarantine",
                }
            )
        if suitability == "quarantine":
            quarantine.append(row)
    _write_jsonl(root / "candidate_manifest.jsonl", candidates)
    _write_jsonl(root / "group_manifest.jsonl", groups)
    _write_jsonl(root / "annotation_queue.jsonl", queue)
    _write_jsonl(root / "quarantine_manifest.jsonl", quarantine)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "annotation_representatives": len(specs),
                "pack_dominance": {"after_representative_selection": {"share": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    (root / "freeze_manifest.json").write_text(json.dumps({"content_manifest_hash": "test"}), encoding="utf-8")
    return root


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    specs = []
    types = ["armor", "plant", "gem", "material", "key", "tool", "weapon", "food", "potion"]
    for index in range(24):
        shade = index >= 12
        specs.append(
            (
                f"sprite_{index:02d}",
                types[index % len(types)],
                "shade_pack" if shade else f"pack_{index % 5}",
                "Shade" if shade else f"artist_{index % 5}",
                4 if index % 3 == 0 else 1,
                "quarantine" if index in {10, 23} else "accept",
            )
        )
    return _make_pool(tmp_path / "pool", specs)


def _rows(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((output / "batches").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_deterministic_scheduling(pool: Path, tmp_path: Path):
    one, two = tmp_path / "one", tmp_path / "two"
    build_schedule(pool, one, config=ScheduleConfig(batch_size=8))
    build_schedule(pool, two, config=ScheduleConfig(batch_size=8))
    assert (one / "annotation_plan.json").read_bytes() == (two / "annotation_plan.json").read_bytes()
    assert [path.read_bytes() for path in sorted((one / "batches").glob("*.jsonl"))] == [
        path.read_bytes() for path in sorted((two / "batches").glob("*.jsonl"))
    ]


def test_shade_pack_and_artist_soft_caps(pool: Path, tmp_path: Path):
    summary = build_schedule(pool, tmp_path / "out", config=ScheduleConfig(batch_size=10, max_shade_share=0.30))
    first = summary["batches"][0]
    assert 0 < first["shade_share"] <= 0.30
    rows = ScheduleView(tmp_path / "out").specific_batch(1)
    assert max(sum(row["pack"] == pack for row in rows) for pack in {row["pack"] for row in rows}) <= 3
    assert max(sum(row["artist"] == artist for row in rows) for artist in {row["artist"] for row in rows}) <= 3


def test_underrepresented_types_appear_early(pool: Path, tmp_path: Path):
    build_schedule(pool, tmp_path / "out", config=ScheduleConfig(batch_size=8))
    early = {row["broad_type"] for row in ScheduleView(tmp_path / "out").specific_batch(1)}
    assert {"armor", "plant", "gem", "material", "key"} <= early


def test_geometry_deduplication_and_recolor_propagation(pool: Path, tmp_path: Path):
    summary = build_schedule(pool, tmp_path / "out", config=ScheduleConfig(batch_size=8))
    rows = _rows(tmp_path / "out")
    assert len(rows) == len({row["geometry_group"] for row in rows}) == 24
    assert sum(row["propagation_count"] for row in rows) == 8 * 3
    assert summary["estimated_propagated_variants"] == 24
    assert all(row["recolor_family"] for row in rows)


def test_completed_id_resume(pool: Path, tmp_path: Path):
    completed = tmp_path / "completed.jsonl"
    completed.write_text('{"representative_id":"sprite_00"}\n', encoding="utf-8")
    summary = build_schedule(pool, tmp_path / "out", config=ScheduleConfig(batch_size=8), completed_ids_path=completed)
    assert summary["completed_count"] == 1
    assert "sprite_00" not in {row["sprite_id"] for row in _rows(tmp_path / "out")}

    live_output = tmp_path / "live"
    build_schedule(pool, live_output, config=ScheduleConfig(batch_size=8))
    first_id = ScheduleView(live_output).specific_batch(1)[0]["representative_id"]
    completed.write_text(first_id + "\n", encoding="utf-8")
    view = ScheduleView(live_output, completed)
    assert view.completed_count() == 1
    assert first_id not in {row["representative_id"] for row in view.next_batch()}


def test_issued_batch_stability(pool: Path, tmp_path: Path):
    output = tmp_path / "out"
    build_schedule(pool, output, config=ScheduleConfig(batch_size=8))
    mark_issued(output, 1)
    before = (output / "batches" / "batch_0001.jsonl").read_bytes()
    completed = tmp_path / "completed.txt"
    completed.write_text("sprite_20\n", encoding="utf-8")
    build_schedule(pool, output, config=ScheduleConfig(batch_size=6), completed_ids_path=completed)
    assert (output / "batches" / "batch_0001.jsonl").read_bytes() == before


def test_unsatisfiable_constraints_and_later_overflow_recovery(tmp_path: Path):
    specs = [(f"shade_{index}", "weapon", "only_pack", "Shade", 1, "accept") for index in range(9)]
    pool = _make_pool(tmp_path / "pool", specs)
    output = tmp_path / "out"
    summary = build_schedule(pool, output, config=ScheduleConfig(batch_size=4, max_shade_share=0.25))
    assert summary["soft_constraint_relaxations"]
    assert len(_rows(output)) == 9
    assert summary["total_batches"] == 3
    deferred = [json.loads(line) for line in (output / "deferred_candidates.jsonl").read_text().splitlines()]
    assert {row["sprite_id"] for row in deferred} == {row["sprite_id"] for row in _rows(output)[4:]}


def test_cli_execution_and_smoke_export(pool: Path, tmp_path: Path):
    output = tmp_path / "out"
    assert main(["build", "--pool", str(pool), "--output", str(output), "--batch-size", "8"]) == 0
    smoke = tmp_path / "smoke.jsonl"
    with pytest.raises(SystemExit):
        main(["export", "--schedule", str(output), "--batch", "1", "--limit", "5", "--output", str(smoke)])
    assert (
        main(
            [
                "export-cohort",
                "--schedule",
                str(output),
                "--batch",
                "1",
                "--size",
                "5",
                "--output",
                str(smoke),
            ]
        )
        == 0
    )
    assert len(smoke.read_text(encoding="utf-8").splitlines()) == 5
    assert main(["query", "--schedule", str(output), "remaining-batches"]) == 0


def test_balanced_25_from_50_is_not_prefix_and_is_deterministic(tmp_path: Path):
    types = ["armor", "plant", "gem", "material", "key", "tool", "weapon"]
    specs = [
        (
            f"sprite_{index:02d}",
            types[index % len(types)],
            f"pack_{index % 8}",
            f"artist_{index % 9}",
            1 + index % 4,
            "quarantine" if index in {3, 17, 31, 44} else "accept",
        )
        for index in range(50)
    ]
    pool = _make_pool(tmp_path / "pool50", specs)
    output = tmp_path / "schedule"
    build_schedule(pool, output, config=ScheduleConfig(batch_size=50))
    view = ScheduleView(output)
    first, manifest = view.export_cohort(1, 25)
    second, manifest_two = view.export_cohort(1, 25)
    assert first == second
    assert manifest == manifest_two
    assert [row["representative_id"] for row in first] != [
        row["representative_id"] for row in view.specific_batch(1)[:25]
    ]
    assert len(manifest["broad_type_distribution"]) >= 4
    assert len(manifest["pack_distribution"]) > 1
    assert len(manifest["artist_distribution"]) > 1
    assert manifest["suitability_distribution"] == {"accept": 25}
    assert len({row["geometry_group"] for row in first}) == 25
    assert all(row["cohort_context"]["broad_type_is_scheduling_metadata_only"] for row in first)


def test_semantic_and_quality_cohorts_are_separate(pool: Path, tmp_path: Path):
    output = tmp_path / "out"
    build_schedule(pool, output, config=ScheduleConfig(batch_size=24))
    view = ScheduleView(output)
    semantic, semantic_manifest = view.export_cohort(1, 10)
    quality, quality_manifest = view.export_cohort(1, 2, mode="quality_quarantine")
    assert all(row["suitability"]["status"] == "accept" for row in semantic)
    assert all(row["suitability"]["status"] == "quarantine" for row in quality)
    assert not ({row["sprite_id"] for row in semantic} & {row["sprite_id"] for row in quality})
    assert semantic_manifest["quality_decision_values"] == []
    assert quality_manifest["quality_decision_values"] == [
        "quality_accept",
        "quality_reject",
        "quality_uncertain",
    ]
    assert all(row["suitability"]["reason_codes"] for row in quality)


def test_propagation_metrics_name_estimated_completed_and_remaining(pool: Path, tmp_path: Path):
    output = tmp_path / "out"
    build_schedule(pool, output, config=ScheduleConfig(batch_size=8))
    first = ScheduleView(output).specific_batch(1)[0]
    completed = tmp_path / "completed.jsonl"
    completed.write_text(json.dumps({"representative_id": first["representative_id"]}) + "\n", encoding="utf-8")
    view = ScheduleView(output, completed)
    metrics = view.propagation_metrics()
    assert metrics["estimated_propagated_variants"] == 24
    assert metrics["completed_propagated_variants"] == first["propagation_count"]
    assert metrics["remaining_propagation_value"] == 24 - first["propagation_count"]
    assert view.propagated_variant_count()["metric"] == "completed_propagated_variants"
