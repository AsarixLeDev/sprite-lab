"""Tests for Auto-Labeling v3 Phase 6: sharding, resume, merge, failures, streaming."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
from spritelab.harvest.label_v3.pipeline_stages_v3 import (
    ShardPaths,
    merge_v3_shards,
    record_in_shard,
    retry_v3_failures,
    run_v3_shard,
    stable_shard,
    stream_v3_report,
)


def _make_run(tmp_path: Path, n: int = 30) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [
        {
            "sprite_id": f"sprite_{i:04d}",
            "source_id": f"src_{i % 4}",
            "source_name": f"Pack {i % 4}",
            "relative_path": f"pack{i % 4}/thing_{i:04d}.png",
            "final_png_path": f"pack{i % 4}/thing_{i:04d}.png",
            "status": "accepted",
        }
        for i in range(n)
    ]
    (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return run_dir


def _run_all_shards_and_merge(run_dir: Path, out_root: Path, shard_count: int) -> bytes:
    cfg = V3PipelineConfig()
    for i in range(shard_count):
        run_v3_shard(run_dir, out_root, cfg, shard_index=i, shard_count=shard_count, resume=False)
    merge_v3_shards(out_root)
    return (out_root / "v3_records.jsonl").read_bytes()


class TestSharding:
    def test_stable_shard_is_deterministic(self):
        assert stable_shard("sprite_0001", 4) == stable_shard("sprite_0001", 4)
        # Assignment does not depend on order, only on id + count.
        assert 0 <= stable_shard("sprite_0001", 4) < 4

    def test_every_record_assigned_exactly_once(self):
        ids = [f"sprite_{i:04d}" for i in range(200)]
        for n in (1, 2, 3, 5, 8):
            counts = [sum(record_in_shard(sid, i, n) for i in range(n)) for sid in ids]
            assert all(c == 1 for c in counts)


class TestMergeBitIdentical:
    def test_shard_count_does_not_change_merged_output(self, tmp_path):
        run_dir = _make_run(tmp_path, n=40)
        one = _run_all_shards_and_merge(run_dir, tmp_path / "out1", 1)
        three = _run_all_shards_and_merge(run_dir, tmp_path / "out3", 3)
        five = _run_all_shards_and_merge(run_dir, tmp_path / "out5", 5)
        assert one == three == five
        # And it is non-empty / one line per record.
        assert one.count(b"\n") == 40

    def test_repeated_run_is_bit_identical(self, tmp_path):
        run_dir = _make_run(tmp_path, n=20)
        a = _run_all_shards_and_merge(run_dir, tmp_path / "a", 2)
        b = _run_all_shards_and_merge(run_dir, tmp_path / "b", 2)
        assert a == b


class TestResume:
    def test_kill_and_resume_matches_full_run(self, tmp_path):
        run_dir = _make_run(tmp_path, n=30)
        cfg = V3PipelineConfig()

        # Full reference run.
        ref_root = tmp_path / "ref"
        run_v3_shard(run_dir, ref_root, cfg, shard_index=0, shard_count=1, resume=False)
        merge_v3_shards(ref_root)
        ref = (ref_root / "v3_records.jsonl").read_bytes()

        # Partial run (first 10), then resume the remainder.
        resume_root = tmp_path / "resume"
        partial = run_v3_shard(run_dir, resume_root, cfg, shard_count=1, resume=False, max_records=10)
        assert partial.processed == 10
        second = run_v3_shard(run_dir, resume_root, cfg, shard_count=1, resume=True)
        assert second.skipped == 10
        assert second.processed == 20
        merge_v3_shards(resume_root)
        assert (resume_root / "v3_records.jsonl").read_bytes() == ref

    def test_resume_is_noop_when_complete(self, tmp_path):
        run_dir = _make_run(tmp_path, n=15)
        cfg = V3PipelineConfig()
        root = tmp_path / "out"
        run_v3_shard(run_dir, root, cfg, shard_count=1, resume=False)
        again = run_v3_shard(run_dir, root, cfg, shard_count=1, resume=True)
        assert again.processed == 0
        assert again.skipped == 15

    def test_config_change_refuses_to_mix(self, tmp_path):
        run_dir = _make_run(tmp_path, n=5)
        root = tmp_path / "out"
        run_v3_shard(run_dir, root, V3PipelineConfig(), shard_count=1, resume=False)
        from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy

        changed = V3PipelineConfig(policy=V3LabelingPolicy(precision_target_category=0.5))
        with pytest.raises(ValueError):
            run_v3_shard(run_dir, root, changed, shard_count=1, resume=True)


class TestMergeDuplicates:
    def test_conflicting_duplicate_rejected(self, tmp_path):
        base = tmp_path / "out" / "shards"
        base.mkdir(parents=True)
        (base / "shard_0000_of_0002.records.jsonl").write_text(
            json.dumps({"sprite_id": "x", "record_state": "unknown"}) + "\n", encoding="utf-8"
        )
        (base / "shard_0001_of_0002.records.jsonl").write_text(
            json.dumps({"sprite_id": "x", "record_state": "auto_accept"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            merge_v3_shards(tmp_path / "out")

    def test_identical_duplicate_allowed(self, tmp_path):
        base = tmp_path / "out" / "shards"
        base.mkdir(parents=True)
        line = json.dumps({"sprite_id": "x", "record_state": "unknown"}) + "\n"
        (base / "shard_0000_of_0002.records.jsonl").write_text(line, encoding="utf-8")
        (base / "shard_0001_of_0002.records.jsonl").write_text(line, encoding="utf-8")
        info = merge_v3_shards(tmp_path / "out")
        assert info["merged_records"] == 1
        assert info["duplicate_lines"] == 1


class TestFailureRetry:
    def test_retry_processes_only_eligible(self, tmp_path):
        run_dir = _make_run(tmp_path, n=6)
        cfg = V3PipelineConfig()
        root = tmp_path / "out"
        paths = ShardPaths.for_shard(root, 0, 1)
        paths.failures.parent.mkdir(parents=True, exist_ok=True)
        # Ensure meta exists so retry writes are consistent.
        run_v3_shard(run_dir, root, cfg, shard_count=1, resume=False, max_records=0)

        failures = [
            {"sprite_id": "sprite_0000", "retryable": True, "retry_count": 0},  # recoverable
            {"sprite_id": "sprite_0001", "retryable": False, "retry_count": 0},  # terminal, stays
            {"sprite_id": "missing_id", "retryable": True, "retry_count": 0},  # not in imported, stays
        ]
        paths.failures.write_text("\n".join(json.dumps(f) for f in failures) + "\n", encoding="utf-8")

        info = retry_v3_failures(run_dir, root, cfg, shard_count=1)
        assert info["recovered"] == 1
        assert info["remaining"] == 2  # the terminal one + the missing one


class TestCLIRegistration:
    def test_phase6_commands_registered(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        for cmd in ("label-v3-shard", "label-v3-merge", "label-v3-retry"):
            assert cmd in subs.choices


class TestStreamingReport:
    def test_streaming_report_aggregates(self, tmp_path):
        run_dir = _make_run(tmp_path, n=24)
        root = tmp_path / "out"
        run_v3_shard(run_dir, root, V3PipelineConfig(), shard_count=1, resume=False)
        merge_v3_shards(root)
        report = stream_v3_report(root / "v3_records.jsonl")
        assert report["total_records"] == 24
        # Four packs (i % 4) should appear in per-pack aggregation.
        assert len(report["per_pack_state"]) == 4
        assert sum(report["record_states"].values()) == 24
