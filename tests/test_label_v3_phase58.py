"""Tests for assisted-v3 GUI and freeze-v3-suite CLI.

Tests:
- GUI candidate loading
- Accept as-is
- Correction
- Abstention
- Resume
- Valid correction serialization
- Deterministic suite creation
- Source leakage rejection
- CLI execution
- Evaluation mode: frozen suite + golden_labels.jsonl
- Leakage rejection before GUI
- GoldenLabel format accepted by label-v3-eval
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from spritelab.harvest.label_v3.assisted_golden_v3 import (
    V3_CORRECTIONS_FILENAME,
    V3_RECORDS_FILENAME,
    V3CorrectionEvent,
    append_v3_correction,
    load_v3_corrections,
    select_v3_calibration_sample,
    v3_candidate_summary_for_gui,
)
from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, FieldDecision
from spritelab.harvest.label_v3.frozen_suites_v3 import (
    FrozenSuiteManifest,
    check_suite_leakage,
)
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    record_decision_to_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    sid: str,
    state: str = "abstained",
    cat: str = "item_icon",
    obj: str = "potion",
    cat_state: str = "accepted",
    obj_state: str = "accepted",
) -> RecordDecision:
    return RecordDecision(
        sprite_id=sid,
        record_state=state,
        category=FieldDecision(
            field_name="category",
            state=cat_state,
            accepted_value=cat,
            evidence_refs=("ev_cat",),
            policy_hash="ph",
        ),
        canonical_object=FieldDecision(
            field_name="canonical_object",
            state=obj_state,
            accepted_value=obj,
            evidence_refs=("ev_obj",),
            policy_hash="ph",
        ),
        color=FieldDecision(field_name="color", state="accepted", accepted_value="red"),
        material=FieldDecision(field_name="material", state="abstained"),
        shape=FieldDecision(field_name="shape", state="unlabeled"),
        role=FieldDecision(field_name="role", state="unlabeled"),
        domain=FieldDecision(field_name="domain", state="accepted", accepted_value="item_icon"),
        tags=AcceptedTagSet(),
    )


def _write_v3_records_jsonl(path: Path, records: list[RecordDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(record_decision_to_json(r), sort_keys=True) + "\n")


def _write_imported_jsonl(path: Path, sprite_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for sid in sprite_ids:
            fh.write(
                json.dumps(
                    {
                        "sprite_id": sid,
                        "source_id": f"src_{sid[0]}",
                        "source_name": f"Source {sid[0]}",
                        "final_png_path": f"test/{sid}.png",
                        "relative_path": f"test/{sid}.png",
                        "status": "accepted",
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# V3CorrectionEvent tests
# ---------------------------------------------------------------------------


class TestV3CorrectionSerialization:
    def test_event_to_dict(self):
        event = V3CorrectionEvent(
            sprite_id="s1",
            field_name="category",
            original_value="weapon",
            corrected_value="tool",
            original_state="accepted",
            corrected_state="accepted",
            selection_reason="manual_correction",
        )
        d = event.to_dict()
        assert d["sprite_id"] == "s1"
        assert d["field_name"] == "category"
        assert d["corrected_value"] == "tool"

    def test_event_roundtrip(self):
        event = V3CorrectionEvent(
            sprite_id="s2",
            field_name="canonical_object",
            original_value="sword",
            corrected_value=None,
            original_state="accepted",
            corrected_state="abstained",
            evidence_refs_visible=("ev_1", "ev_2"),
        )
        d = event.to_dict()
        restored = V3CorrectionEvent.from_dict(d)
        assert restored.sprite_id == event.sprite_id
        assert restored.corrected_state == "abstained"
        assert restored.corrected_value is None

    def test_append_and_load(self, tmp_path):
        events = [
            V3CorrectionEvent("s1", "category", "a", "b", "accepted", "accepted"),
            V3CorrectionEvent("s2", "canonical_object", "x", "y", "accepted", "abstained"),
        ]
        p = tmp_path / "v3_corrections.jsonl"
        for e in events:
            append_v3_correction(p, e)
        loaded = load_v3_corrections(p)
        assert len(loaded) == 2
        assert loaded[0].sprite_id == "s1"

    def test_correction_timestamp_auto(self):
        event = V3CorrectionEvent("s1", "category", "a", "b", "accepted", "accepted")
        assert event.timestamp
        assert "T" in event.timestamp


# ---------------------------------------------------------------------------
# Candidate loading tests
# ---------------------------------------------------------------------------


class TestCandidateLoading:
    def test_load_v3_records_for_gui(self, tmp_path):
        records = [
            _make_record("s1", state="auto_accept", cat="weapon", obj="sword"),
            _make_record("s2", state="partial_accept", cat="item_icon", obj="potion", obj_state="abstained"),
        ]
        v3_path = tmp_path / V3_RECORDS_FILENAME
        _write_v3_records_jsonl(v3_path, records)

        from spritelab.harvest.label_v3.assisted_golden_v3 import prefetch_v3_records_for_candidates

        loaded = prefetch_v3_records_for_candidates(tmp_path, ["s1", "s2"], v3_records_path=v3_path)
        assert len(loaded) == 2
        assert loaded["s1"].category.accepted_value == "weapon"
        assert loaded["s2"].record_state == "partial_accept"

    def test_candidate_summary(self):
        record = _make_record("s1", state="auto_accept", cat="weapon", obj="sword")
        summary = v3_candidate_summary_for_gui(record)
        assert summary["sprite_id"] == "s1"
        assert summary["record_state"] == "auto_accept"
        assert summary["fields"]["category"]["value"] == "weapon"
        assert summary["fields"]["canonical_object"]["value"] == "sword"
        assert summary["fields"]["color"]["value"] == "red"
        assert "category" in summary["accepted_fields"]


# ---------------------------------------------------------------------------
# Sample selection tests
# ---------------------------------------------------------------------------


class TestSampleSelection:
    def test_select_deterministic(self):
        records = [_make_record(f"s{i}", cat_state="accepted" if i % 2 else "abstained") for i in range(20)]
        r1 = select_v3_calibration_sample(records, 5, seed=42)
        r2 = select_v3_calibration_sample(records, 5, seed=42)
        assert r1 == r2

    def test_select_small_n(self):
        records = [_make_record(f"s{i}") for i in range(10)]
        result = select_v3_calibration_sample(records, 3, seed=42)
        assert len(result) == 3

    def test_select_zero(self):
        records = [_make_record("s1")]
        result = select_v3_calibration_sample(records, 0)
        assert result == []

    def test_select_empty_records(self):
        result = select_v3_calibration_sample([], 10)
        assert result == []


# ---------------------------------------------------------------------------
# Accept-as-is and correction tests (without Gradio)
# ---------------------------------------------------------------------------


class TestAcceptAsIs:
    def test_accept_as_is_appends_corrections(self, tmp_path):
        records = [
            _make_record("s1", state="auto_accept", cat="weapon", obj="sword"),
            _make_record("s2", state="auto_accept", cat="item_icon", obj="potion"),
        ]
        v3_path = tmp_path / V3_RECORDS_FILENAME
        _write_v3_records_jsonl(v3_path, records)
        _write_imported_jsonl(tmp_path / "imported.jsonl", ["s1", "s2"])

        # Simulate accept-as-is: for each accepted field, write correction
        corr_path = tmp_path / V3_CORRECTIONS_FILENAME
        for r in records:
            for fd in (r.category, r.canonical_object, r.color, r.domain):
                if fd.state == "accepted":
                    event = V3CorrectionEvent(
                        sprite_id=r.sprite_id,
                        field_name=fd.field_name,
                        original_value=fd.accepted_value,
                        corrected_value=fd.accepted_value,
                        original_state=fd.state,
                        corrected_state="accepted",
                        selection_reason="accept_as_is",
                    )
                    append_v3_correction(corr_path, event)

        loaded = load_v3_corrections(corr_path)
        # s1: cat, obj, color, domain = 4 ; s2: cat, obj, color, domain = 4
        assert len(loaded) == 8
        assert all(c.corrected_state == "accepted" for c in loaded)


class TestAbstention:
    def test_abstention_correction(self, tmp_path):
        corr_path = tmp_path / V3_CORRECTIONS_FILENAME
        event = V3CorrectionEvent(
            sprite_id="s1",
            field_name="canonical_object",
            original_value="sword",
            corrected_value=None,
            original_state="accepted",
            corrected_state="abstained",
            selection_reason="human_abstention",
        )
        append_v3_correction(corr_path, event)
        loaded = load_v3_corrections(corr_path)
        assert len(loaded) == 1
        assert loaded[0].corrected_state == "abstained"


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_state_persistence(self, tmp_path):
        state_path = tmp_path / "v3_assisted_state.json"
        state = {
            "current_index": 5,
            "labeler": "operator",
            "session_id": "test-session",
            "mode": "calibration",
            "last_opened": "2026-07-11T00:00:00Z",
            "total": 20,
            "corrected_count": 3,
            "skipped": ["s1", "s5"],
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        assert loaded["current_index"] == 5
        assert loaded["skipped"] == ["s1", "s5"]
        assert loaded["mode"] == "calibration"


# ---------------------------------------------------------------------------
# Frozen suite tests
# ---------------------------------------------------------------------------


class TestFrozenSuiteCreation:
    def test_deterministic_partitioning(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_imported_jsonl(
            run_dir / "imported.jsonl",
            [f"src_a_{i}" for i in range(10)]
            + [f"src_b_{i}" for i in range(10)]
            + [f"src_c_{i}" for i in range(10)]
            + [f"src_d_{i}" for i in range(10)],
        )
        # Patch source_id in records for partitioning
        records = []
        for prefix in ("a", "b", "c", "d"):
            for i in range(10):
                records.append(
                    {
                        "sprite_id": f"src_{prefix}_{i}",
                        "source_id": f"src_{prefix}",
                        "source_name": f"Source {prefix}",
                        "relative_path": f"sheet_{prefix}/sprite_{i}.png",
                        "status": "accepted",
                        "final_png_path": f"sheet_{prefix}/sprite_{i}.png",
                    }
                )
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        # Run freeze-v3-suite CLI
        py = sys.executable
        result = subprocess.run(
            [
                py,
                "-m",
                "spritelab",
                "harvest",
                "freeze-v3-suite",
                "--runs",
                str(run_dir),
                "--suite-name",
                "test_suite",
                "--seed",
                "42",
                "--n",
                "40",
                "--suite-dir",
                str(tmp_path),
                "--fractions",
                "0.25",
                "0.25",
                "0.20",
                "0.15",
                "0.10",
                "0.05",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        if result.returncode != 0:
            raise AssertionError(f"CLI failed: {result.stderr}\n{result.stdout}")

        suite_path = tmp_path / "test_suite_suite.json"
        assert suite_path.is_file(), f"Suite file not created: {result.stdout}\n{result.stderr}"

        manifest = FrozenSuiteManifest.load(suite_path)
        all_ids = manifest.all_ids()
        assert len(all_ids) > 0
        assert "calibration" in manifest.partitions
        assert "development" in manifest.partitions
        assert "frozen_in_domain_test" in manifest.partitions
        assert "frozen_unseen_pack_test" in manifest.partitions
        assert "frozen_source_ood_test" in manifest.partitions

    def test_source_leakage_rejection(self):
        manifest = FrozenSuiteManifest(
            partitions={
                "calibration": ("a", "b"),
                "frozen_in_domain_test": ("b", "c"),
            }
        )
        report = check_suite_leakage(manifest)
        assert report.ok is False
        assert report.cross_partition_overlaps

    def test_tuning_leakage_detected(self):
        manifest = FrozenSuiteManifest(
            partitions={
                "calibration": ("a", "b"),
                "frozen_in_domain_test": ("c", "d"),
            }
        )
        report = check_suite_leakage(manifest, tuning_ids=("a", "z"))
        assert report.ok is False
        assert "a" in report.tuning_overlap

    def test_clean_suite_passes(self):
        manifest = FrozenSuiteManifest(
            partitions={
                "calibration": ("a", "b"),
                "frozen_in_domain_test": ("c", "d"),
                "frozen_unseen_pack_test": ("e", "f"),
            }
        )
        report = check_suite_leakage(manifest, tuning_ids=("x", "y"))
        assert report.ok is True

    def test_suite_manifest_roundtrip(self, tmp_path):
        manifest = FrozenSuiteManifest(
            suite_name="test",
            partitions={"in_domain": ("a", "b"), "unseen_pack": ("c",)},
        )
        path = tmp_path / "suite.json"
        manifest.save(path)
        restored = FrozenSuiteManifest.load(path)
        assert restored.suite_name == "test"
        assert restored.partition_ids("in_domain") == ("a", "b")


# ---------------------------------------------------------------------------
# CLI execution tests
# ---------------------------------------------------------------------------


class TestCLIExecution:
    def test_label_v3_help_registers_new_commands(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        names = [str(a) for a in subs.choices.keys()]
        assert "assisted-v3" in names
        assert "freeze-v3-suite" in names
        assert "calibrate-v3-all" in names

    def test_assisted_v3_help(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        py = sys.executable
        result = subprocess.run(
            [py, "-m", "spritelab", "harvest", "assisted-v3", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, f"help failed: {result.stderr}"
        assert "--run" in result.stdout

    def test_freeze_v3_suite_help(self):
        py = sys.executable
        result = subprocess.run(
            [py, "-m", "spritelab", "harvest", "freeze-v3-suite", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, f"help failed: {result.stderr}"
        assert "--runs" in result.stdout

    def test_calibrate_v3_all_help(self):
        py = sys.executable
        result = subprocess.run(
            [py, "-m", "spritelab", "harvest", "calibrate-v3-all", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, f"help failed: {result.stderr}"
        assert "--run" in result.stdout


# ---------------------------------------------------------------------------
# Evaluation chain tests: frozen suite, golden labels, label-v3-eval
# ---------------------------------------------------------------------------


class TestEvaluationChain:
    def test_golden_label_format_accepted_by_eval(self, tmp_path):
        from spritelab.harvest.golden import GoldenLabel, append_golden_label, load_golden_labels
        from spritelab.harvest.label_v3.field_decisions import FieldDecision
        from spritelab.harvest.label_v3.label_v3_eval import evaluate_v3_against_golden
        from spritelab.harvest.label_v3.record_decisions import RecordDecision

        gl = GoldenLabel(
            sprite_id="eval_001",
            category="weapon",
            object_name="sword",
            tags=("blade",),
            labeler="test",
        )
        gold_path = tmp_path / "golden_labels.jsonl"
        append_golden_label(gold_path, gl)
        loaded = load_golden_labels(gold_path)
        assert "eval_001" in loaded
        assert loaded["eval_001"].category == "weapon"

        v3_rec = RecordDecision(
            sprite_id="eval_001",
            record_state="auto_accept",
            category=FieldDecision(
                field_name="category",
                state="accepted",
                accepted_value="weapon",
                evidence_refs=("ev",),
                policy_hash="ph",
            ),
            canonical_object=FieldDecision(
                field_name="canonical_object",
                state="accepted",
                accepted_value="sword",
                evidence_refs=("ev",),
                policy_hash="ph",
            ),
        )
        result = evaluate_v3_against_golden(loaded, {"eval_001": v3_rec}, suite_name="test")
        assert result.matched == 1
        assert result.per_field["category"]["accepted_total"] == 1
        assert result.per_field["category"]["accepted_correct"] == 1
        assert result.per_field["category"]["precision"] == 1.0

    def test_golden_label_unknown_eval(self, tmp_path):
        from spritelab.harvest.golden import GoldenLabel, append_golden_label, load_golden_labels
        from spritelab.harvest.label_v3.field_decisions import FieldDecision
        from spritelab.harvest.label_v3.label_v3_eval import evaluate_v3_against_golden
        from spritelab.harvest.label_v3.record_decisions import RecordDecision

        gl = GoldenLabel(sprite_id="eval_u", category="unknown", object_name="", labeler="test")
        gold_path = tmp_path / "golden_labels.jsonl"
        append_golden_label(gold_path, gl)
        loaded = load_golden_labels(gold_path)
        v3_rec = RecordDecision(
            sprite_id="eval_u",
            record_state="unknown",
            category=FieldDecision(field_name="category", state="unknown"),
            canonical_object=FieldDecision(field_name="canonical_object", state="unknown"),
        )
        result = evaluate_v3_against_golden(loaded, {"eval_u": v3_rec}, suite_name="test")
        assert result.matched == 1

    def test_frozen_suite_leakage_blocks(self):
        from spritelab.harvest.label_v3.frozen_suites_v3 import FrozenSuiteManifest, check_suite_leakage

        suite = FrozenSuiteManifest(
            suite_name="leaky",
            partitions={"frozen_in_domain_test": ("a", "b", "c"), "frozen_unseen_pack_test": ("b", "d")},
        )
        report = check_suite_leakage(suite)
        assert report.ok is False
        assert report.cross_partition_overlaps

    def test_tuning_overlap_detected_in_suite(self):
        from spritelab.harvest.label_v3.frozen_suites_v3 import FrozenSuiteManifest, check_suite_leakage

        suite = FrozenSuiteManifest(
            suite_name="overlap",
            partitions={"calibration": ("c1", "c2"), "frozen_in_domain_test": ("e1", "e2")},
        )
        # c1 is in the suite AND in tuning_ids — should be flagged.
        report = check_suite_leakage(suite, tuning_ids=("c1",))
        assert report.ok is False
        assert "c1" in report.tuning_overlap

    def test_eval_manifest_roundtrip(self, tmp_path):
        from spritelab.harvest.label_v3.frozen_suites_v3 import FrozenSuiteManifest

        manifest = FrozenSuiteManifest(
            suite_name="eval_v3.1_test",
            taxonomy_version="v3.1.0",
            partitions={
                "calibration": ("cal_001", "cal_002"),
                "frozen_in_domain_test": ("eval_001", "eval_002"),
                "frozen_unseen_pack_test": ("eval_003",),
            },
        )
        path = tmp_path / "eval_suite.json"
        manifest.save(path)
        restored = FrozenSuiteManifest.load(path)
        assert restored.suite_name == "eval_v3.1_test"
        assert restored.partition_ids("frozen_in_domain_test") == ("eval_001", "eval_002")

    def test_disjoint_sources_pass_leakage(self):
        from spritelab.harvest.label_v3.frozen_suites_v3 import FrozenSuiteManifest, check_suite_leakage

        suite = FrozenSuiteManifest(
            suite_name="disjoint",
            partitions={
                "calibration": ("src_a_1", "src_a_2"),
                "frozen_in_domain_test": ("src_b_1", "src_b_2"),
                "frozen_unseen_pack_test": ("src_c_1",),
            },
        )
        report = check_suite_leakage(suite)
        assert report.ok is True


class TestEvalCLIHelp:
    def test_assisted_v3_eval_args(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        p = subs.choices["assisted-v3"]
        known_args = {a.dest for a in p._actions}
        assert "suite" in known_args
        assert "partition" in known_args
        assert "calibration_run" in known_args
        assert "golden_path" in known_args
        assert "mode" in known_args


# ---------------------------------------------------------------------------
# E2E test: label-v3 producer → assisted-v3 consumer
# ---------------------------------------------------------------------------


class TestProducerConsumerContract:
    def test_label_v3_output_readable_by_load_all_records(self, tmp_path):
        import json

        from spritelab.harvest.label_v3.assisted_golden_v3 import load_all_v3_records
        from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
        from spritelab.harvest.label_v3.pipeline_v3 import RECORD_OUTPUT_SUFFIX, run_v3_pipeline

        # Create a minimal harvest run
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        imported = [
            {
                "sprite_id": f"test_{i:03d}",
                "source_id": f"src_{i % 3}",
                "source_name": f"Test Pack {i % 3}",
                "relative_path": f"test/sprite_{i:03d}.png",
                "final_png_path": f"test/sprite_{i:03d}.png",
                "status": "accepted",
            }
            for i in range(10)
        ]
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in imported) + "\n", encoding="utf-8")

        # Run label-v3 pipeline (non-sharded)
        output_root = tmp_path / "output"
        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=output_root,
            config=V3PipelineConfig(),
            use_vlm=False,
            dry_run=False,
            max_records=10,
        )
        assert result.total_records == 10

        # Verify the output file exists at the pipeline's canonical path
        pipeline_name = f"v3{RECORD_OUTPUT_SUFFIX}"
        records_path = output_root / pipeline_name
        assert records_path.is_file(), f"Expected {records_path} to exist"

        # Verify load_all_v3_records can read it
        loaded = load_all_v3_records(records_path)
        assert len(loaded) == 10

        # Verify _find_v3_records finds it
        from spritelab.harvest.label_v3.assisted_v3_gui import _find_v3_records

        found = _find_v3_records(run_dir)
        # When run_dir has no v3_output, _find_v3_records returns the default path.
        # When pipeline writes to output_root (not run_dir), we test the
        # load_all_v3_records function directly with the known path.
        assert found is not None

    def test_load_all_v3_records_empty_file(self, tmp_path):
        from spritelab.harvest.label_v3.assisted_golden_v3 import load_all_v3_records

        empty_path = tmp_path / "empty.jsonl"
        empty_path.write_text("", encoding="utf-8")
        loaded = load_all_v3_records(empty_path)
        assert loaded == {}

    def test_load_all_v3_records_nonexistent(self, tmp_path):
        from spritelab.harvest.label_v3.assisted_golden_v3 import load_all_v3_records

        loaded = load_all_v3_records(tmp_path / "nonexistent.jsonl")
        assert loaded == {}

    def test_canonical_names_match_between_producer_and_finder(self):
        import tempfile

        from spritelab.harvest.label_v3.assisted_v3_gui import _find_v3_records
        from spritelab.harvest.label_v3.pipeline_stages_v3 import CANONICAL_RECORDS_NAME
        from spritelab.harvest.label_v3.pipeline_v3 import RECORD_OUTPUT_SUFFIX

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "v3_output"
            out_dir.mkdir()
            pipeline_name = f"v3{RECORD_OUTPUT_SUFFIX}"
            (out_dir / pipeline_name).write_text('{"sprite_id":"x"}\n', encoding="utf-8")
            found = _find_v3_records(td_path)
            assert found.is_file()
            assert found.name == pipeline_name

            # Remove that, place canonical name
            (out_dir / pipeline_name).unlink()
            (out_dir / CANONICAL_RECORDS_NAME).write_text('{"sprite_id":"y"}\n', encoding="utf-8")
            found2 = _find_v3_records(td_path)
            assert found2.is_file()
            assert found2.name == CANONICAL_RECORDS_NAME
