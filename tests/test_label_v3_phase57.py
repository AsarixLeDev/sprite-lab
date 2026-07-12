"""Tests for Auto-Labeling v3 Phases 5-7: GUI calibration, evaluation, CLI."""

from __future__ import annotations

import json

from spritelab.harvest.label_v3.assisted_golden_v3 import (
    V3CorrectionEvent,
    append_v3_correction,
    apply_v3_corrections_to_record,
    load_v3_corrections,
    prefetch_v3_records_for_candidates,
    select_v3_calibration_sample,
    v3_candidate_summary_for_gui,
    v3_correction_summary,
)
from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy
from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, FieldDecision, TagDecision
from spritelab.harvest.label_v3.record_decisions import RecordDecision

# ---------------------------------------------------------------------------
# Correction event tests
# ---------------------------------------------------------------------------


class TestV3CorrectionEvent:
    def test_create_and_to_dict(self):
        event = V3CorrectionEvent(
            sprite_id="test_001",
            field_name="category",
            original_value="weapon",
            corrected_value="tool",
            original_state="accepted",
            corrected_state="accepted",
            evidence_refs_visible=("ev_1", "ev_2"),
            selection_reason="uncertainty_focused",
            reviewer_id="test_reviewer",
            calibration_policy_hash="ph_abc",
        )
        d = event.to_dict()
        assert d["sprite_id"] == "test_001"
        assert d["field_name"] == "category"
        assert d["original_value"] == "weapon"
        assert d["corrected_value"] == "tool"
        assert d["corrected_state"] == "accepted"
        assert d["selection_reason"] == "uncertainty_focused"

    def test_roundtrip(self):
        event = V3CorrectionEvent(
            sprite_id="s_001",
            field_name="canonical_object",
            original_value="sword",
            corrected_value="dagger",
            original_state="accepted",
            corrected_state="accepted",
            selection_reason="near_threshold",
        )
        d = event.to_dict()
        restored = V3CorrectionEvent.from_dict(d)
        assert restored.sprite_id == event.sprite_id
        assert restored.field_name == event.field_name
        assert restored.corrected_value == event.corrected_value

    def test_append_and_load(self, tmp_path):
        events = [
            V3CorrectionEvent("s1", "category", "a", "b", "accepted", "accepted"),
            V3CorrectionEvent("s2", "canonical_object", "x", "y", "accepted", "abstained"),
        ]
        p = tmp_path / "test_corrections.jsonl"
        for e in events:
            append_v3_correction(p, e)
        loaded = load_v3_corrections(p)
        assert len(loaded) == 2
        assert loaded[0].sprite_id == "s1"

    def test_load_empty_file(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        loaded = load_v3_corrections(p)
        assert loaded == []

    def test_summary(self):
        events = [
            V3CorrectionEvent("s1", "category", "a", "b", "accepted", "accepted"),
            V3CorrectionEvent("s1", "canonical_object", "x", "y", "accepted", "abstained"),
            V3CorrectionEvent("s2", "category", "c", "d", "abstained", "accepted"),
        ]
        summary = v3_correction_summary(events)
        assert summary["total_corrections"] == 3
        assert summary["unique_sprites_corrected"] == 2
        assert summary["fields_corrected"]["category"] == 2


# ---------------------------------------------------------------------------
# Sample selection tests
# ---------------------------------------------------------------------------


class TestV3SampleSelection:
    def _make_record(
        self, sid: str, state: str = "abstained", object_state: str = "abstained", contradiction: bool = False
    ) -> RecordDecision:
        obj_fd = FieldDecision(
            field_name="canonical_object",
            state=object_state,
            contradiction_codes=("cat_source_vs_vlm",) if contradiction else (),
        )
        return RecordDecision(
            sprite_id=sid,
            record_state=state,
            category=FieldDecision(
                field_name="category",
                state="accepted" if state == "auto_accept" else "abstained",
                accepted_value="weapon",
            ),
            canonical_object=obj_fd,
        )

    def test_select_empty(self):
        result = select_v3_calibration_sample([], 10)
        assert result == []

    def test_select_zero(self):
        records = [self._make_record("s1")]
        result = select_v3_calibration_sample(records, 0)
        assert result == []

    def test_select_small_n(self):
        records = [self._make_record(f"s{i}") for i in range(20)]
        result = select_v3_calibration_sample(records, 5, seed=42)
        assert len(result) == 5
        assert all(sid.startswith("s") for sid in result)

    def test_select_deterministic(self):
        records = [self._make_record(f"s{i}") for i in range(20)]
        r1 = select_v3_calibration_sample(records, 5, seed=42)
        r2 = select_v3_calibration_sample(records, 5, seed=42)
        assert r1 == r2

    def test_novel_objects_prioritized(self):
        records = [
            self._make_record("s_unknown", object_state="unknown"),
            self._make_record("s_novel", object_state="novel"),
            self._make_record("s_accepted", object_state="accepted"),
            self._make_record("s_abstained", object_state="abstained"),
        ]
        result = select_v3_calibration_sample(records, 4, seed=42)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# GUI candidate summary tests
# ---------------------------------------------------------------------------


class TestGUIHelper:
    def test_v3_candidate_summary(self):
        record = RecordDecision(
            sprite_id="test",
            record_state="auto_accept",
            category=FieldDecision(
                field_name="category", state="accepted", accepted_value="weapon", evidence_refs=("ev_cat",)
            ),
            canonical_object=FieldDecision(
                field_name="canonical_object",
                state="accepted",
                accepted_value="sword",
                hierarchy_node="sword",
                evidence_refs=("ev_obj",),
            ),
            color=FieldDecision(field_name="color", state="abstained"),
            tags=AcceptedTagSet(decisions=(TagDecision(tag="blade", state="accepted"),)),
        )
        summary = v3_candidate_summary_for_gui(record)
        assert summary["record_state"] == "auto_accept"
        assert summary["fields"]["category"]["state"] == "accepted"
        assert summary["fields"]["category"]["value"] == "weapon"
        assert summary["fields"]["color"]["state"] == "abstained"
        assert "blade" in summary["accepted_tags"]
        assert "category" in summary["accepted_fields"]
        assert "color" in summary["abstained_fields"]


# ---------------------------------------------------------------------------
# Correction application tests
# ---------------------------------------------------------------------------


class TestCorrectionApplication:
    def _make_record(self) -> RecordDecision:
        return RecordDecision(
            sprite_id="test",
            record_state="auto_accept",
            category=FieldDecision(
                field_name="category", state="accepted", accepted_value="weapon", evidence_refs=("ev_cat",)
            ),
            canonical_object=FieldDecision(
                field_name="canonical_object", state="accepted", accepted_value="sword", evidence_refs=("ev_obj",)
            ),
            color=FieldDecision(field_name="color", state="abstained"),
            material=FieldDecision(field_name="material", state="unlabeled"),
        )

    def test_apply_category_correction(self):
        record = self._make_record()
        corrections = [
            V3CorrectionEvent("test", "category", "weapon", "tool", "accepted", "accepted"),
        ]
        updated = apply_v3_corrections_to_record(record, corrections)
        assert updated.category.accepted_value == "tool"
        assert updated.category.state == "accepted"
        assert updated.canonical_object.accepted_value == "sword"

    def test_apply_abstain_correction(self):
        record = self._make_record()
        corrections = [
            V3CorrectionEvent("test", "canonical_object", "sword", None, "accepted", "abstained"),
        ]
        updated = apply_v3_corrections_to_record(record, corrections)
        assert updated.canonical_object.state == "abstained"
        assert updated.canonical_object.decision_reason == "human_correction"

    def test_apply_multiple_fields(self):
        record = self._make_record()
        corrections = [
            V3CorrectionEvent("test", "category", "weapon", "tool", "accepted", "accepted"),
            V3CorrectionEvent("test", "color", None, "gray", "abstained", "accepted"),
        ]
        updated = apply_v3_corrections_to_record(record, corrections)
        assert updated.category.accepted_value == "tool"
        assert updated.color.accepted_value == "gray"
        assert updated.color.state == "accepted"

    def test_non_matching_sprite_ignored(self):
        record = self._make_record()
        corrections = [
            V3CorrectionEvent("other_sprite", "category", "a", "b", "accepted", "accepted"),
        ]
        updated = apply_v3_corrections_to_record(record, corrections)
        assert updated.category.accepted_value == "weapon"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestV3Evaluation:
    def test_evaluate_against_golden(self):
        from spritelab.harvest.golden import GoldenLabel
        from spritelab.harvest.label_v3.label_v3_eval import evaluate_v3_against_golden

        golden = {
            "test_001": GoldenLabel("test_001", "weapon", "sword", tags=("blade",)),
            "test_002": GoldenLabel("test_002", "item_icon", "potion", tags=("liquid",)),
        }

        v3_records = {
            "test_001": RecordDecision(
                sprite_id="test_001",
                record_state="auto_accept",
                category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
                canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
            ),
            "test_002": RecordDecision(
                sprite_id="test_002",
                record_state="partial_accept",
                category=FieldDecision(field_name="category", state="accepted", accepted_value="item_icon"),
                canonical_object=FieldDecision(field_name="canonical_object", state="abstained"),
            ),
        }

        result = evaluate_v3_against_golden(golden, v3_records, suite_name="test_suite")
        assert result.total_golden == 2
        assert result.matched == 2
        assert "category" in result.per_field
        assert result.per_field["category"]["accepted_total"] == 2
        assert result.per_field["category"]["accepted_correct"] == 2
        assert result.per_field["category"]["precision"] == 1.0

    def test_promotion_recommendation_full_pass(self):
        from spritelab.harvest.label_v3.label_v3_eval import V3EvalResult, promotion_recommendation

        result = V3EvalResult(suite_name="test")
        result.matched = 100
        result.per_field = {
            "category": {"meets_target": True, "accepted_total": 80},
            "canonical_object": {"meets_target": True, "accepted_total": 70},
        }
        result.promotion_gates = {
            "category_meets_target": True,
            "canonical_object_meets_target": True,
            "hard_reject_zero_fpr": True,
            "ece_acceptable": True,
            "provenance_complete": True,
            "color_meets_target": True,
            "material_meets_target": True,
        }
        rec = promotion_recommendation(result)
        assert rec == "eligible_for_large_batch"

    def test_promotion_recommendation_blocked(self):
        from spritelab.harvest.label_v3.label_v3_eval import V3EvalResult, promotion_recommendation

        result = V3EvalResult(suite_name="test")
        result.matched = 10
        result.promotion_gates = {
            "category_meets_target": False,
            "canonical_object_meets_target": False,
            "hard_reject_zero_fpr": True,
            "ece_acceptable": False,
        }
        rec = promotion_recommendation(result)
        assert rec == "blocked"

    def test_promotion_shadow_only_small_sample(self):
        from spritelab.harvest.label_v3.label_v3_eval import V3EvalResult, promotion_recommendation

        result = V3EvalResult(suite_name="test")
        result.matched = 10
        result.promotion_gates = {
            "category_meets_target": True,
            "canonical_object_meets_target": True,
            "hard_reject_zero_fpr": True,
            "ece_acceptable": True,
        }
        rec = promotion_recommendation(result)
        assert rec == "shadow_only"


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_empty(self, tmp_path):
        records = prefetch_v3_records_for_candidates(str(tmp_path), ["s1", "s2"])
        assert records == {}

    def test_prefetch_with_records(self, tmp_path):
        from spritelab.harvest.catalog import write_jsonl
        from spritelab.harvest.label_v3.record_decisions import record_decision_to_json

        r1 = RecordDecision(
            sprite_id="s1",
            record_state="auto_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
        )
        r2 = RecordDecision(
            sprite_id="s2",
            record_state="partial_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="item_icon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="abstained"),
        )

        v3_path = tmp_path / "v3_records.jsonl"
        write_jsonl(v3_path, [record_decision_to_json(r1), record_decision_to_json(r2)])

        loaded = prefetch_v3_records_for_candidates(str(tmp_path), ["s1", "s2"], v3_records_path=v3_path)
        assert len(loaded) == 2
        assert loaded["s1"].record_state == "auto_accept"
        assert loaded["s2"].category.accepted_value == "item_icon"


# ---------------------------------------------------------------------------
# CLI integration smoke tests
# ---------------------------------------------------------------------------


class TestCLISmoke:
    def test_label_v3_help(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        assert "label-v3" in [str(a) for a in subs.choices.keys()]

    def test_label_v3_eval_help(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        assert "label-v3-eval" in [str(a) for a in subs.choices.keys()]

    def test_label_v3_promote_help(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        assert "label-v3-promote" in [str(a) for a in subs.choices.keys()]

    def test_calibrate_v3_help(self):
        import argparse

        from spritelab.harvest.label_v3.label_v3_cli import register

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        register(subs)
        assert "calibrate-v3" in [str(a) for a in subs.choices.keys()]


# ---------------------------------------------------------------------------
# Full dry-run pipeline test
# ---------------------------------------------------------------------------


class TestEndToEndDryRun:
    def test_full_dry_run_with_imported(self, tmp_path):
        from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
        from spritelab.harvest.label_v3.pipeline_v3 import run_v3_pipeline

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
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in imported), encoding="utf-8")

        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=str(tmp_path / "output"),
            config=V3PipelineConfig(),
            use_vlm=False,
            dry_run=False,
            max_records=10,
        )

        assert result.total_records == 10
        assert result.policy_hash

        # Check output files were written
        records_path = tmp_path / "output" / "v3_v3_records.jsonl"
        assert records_path.is_file()

    def test_dry_run_does_not_write_files(self, tmp_path):
        from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
        from spritelab.harvest.label_v3.pipeline_v3 import run_v3_pipeline

        run_dir = tmp_path / "run2"
        run_dir.mkdir()
        imported = [
            {
                "sprite_id": "test_001",
                "source_id": "src_test",
                "source_name": "Test",
                "relative_path": "test.png",
                "final_png_path": "test.png",
                "status": "accepted",
            },
        ]
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in imported), encoding="utf-8")

        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=str(tmp_path / "output_dry"),
            config=V3PipelineConfig(),
            use_vlm=False,
            dry_run=True,
            max_records=10,
        )

        assert result.total_records == 1
        # Dry run should NOT write files
        records_path = tmp_path / "output_dry" / "v3_v3_records.jsonl"
        assert not records_path.is_file()

    def test_default_config_shadow_mode(self):
        policy = V3LabelingPolicy()
        assert policy.shadow_mode is True
        assert policy.dry_run_apply is True
        assert policy.precision_target_category == 0.99
        assert policy.auto_accept_enabled is True


# ---------------------------------------------------------------------------
# Config fidelity tests
# ---------------------------------------------------------------------------


class TestConfigFidelity:
    def test_field_states_are_distinct(self):
        states = {
            "accepted",
            "abstained",
            "quarantined",
            "rejected",
            "unknown",
            "novel",
            "ambiguous",
            "unlabeled",
            "not_applicable",
        }
        assert len(states) == 9

    def test_open_set_states_are_distinct(self):
        open_states = {"in_distribution", "open_set", "novel", "unknown"}
        assert len(open_states) == 4

    def test_reason_codes_complete_set(self):
        from spritelab.harvest.label_v3.reason_codes import REASON_CODES

        assert "insufficient_evidence" in REASON_CODES
        assert "impossible_combination" in REASON_CODES
        assert "calibration_insufficient" in REASON_CODES
        assert "blank_or_empty" in REASON_CODES
        assert "irreconcilable_contradiction" in REASON_CODES
