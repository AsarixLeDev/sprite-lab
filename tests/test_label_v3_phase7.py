"""Tests for Auto-Labeling v3 Phase 7: eval metric fixes, frozen suites, leakage."""

from __future__ import annotations

import pytest

from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.label_v3.field_decisions import FieldDecision
from spritelab.harvest.label_v3.frozen_suites_v3 import (
    FrozenSuiteManifest,
    check_suite_leakage,
)
from spritelab.harvest.label_v3.label_v3_eval import (
    V3EvalResult,
    evaluate_v3_against_golden,
    promotion_recommendation,
)
from spritelab.harvest.label_v3.record_decisions import RecordDecision


def _accepted(field_name: str, value: str) -> FieldDecision:
    return FieldDecision(
        field_name=field_name,
        state="accepted",
        accepted_value=value,
        evidence_refs=("ev_1",),
        policy_hash="ph",
        calibrated_estimate=0.99,
    )


class TestEvalMetricFixes:
    def test_domain_material_not_scored_against_tags(self):
        # domain/material have no golden truth; they must not appear in per_field
        # (previously they were scored against GoldenLabel.tags).
        golden = {"a": GoldenLabel("a", "weapon", "sword", tags=("blade",))}
        v3 = {
            "a": RecordDecision(
                sprite_id="a",
                record_state="auto_accept",
                category=_accepted("category", "weapon"),
                canonical_object=_accepted("canonical_object", "sword"),
                material=_accepted("material", "metal"),
                domain=_accepted("domain", "weapon"),
            )
        }
        result = evaluate_v3_against_golden(golden, v3)
        assert result.per_field["category"]["accepted_total"] == 1
        assert result.per_field["canonical_object"]["accepted_total"] == 1
        # material/domain are not golden-evaluable -> no accepted scoring.
        assert result.per_field["material"]["accepted_total"] == 0
        assert result.per_field["domain"]["accepted_total"] == 0

    def test_provenance_completeness_is_one_when_all_have_evidence(self):
        golden = {"a": GoldenLabel("a", "weapon", "sword")}
        v3 = {
            "a": RecordDecision(
                sprite_id="a",
                record_state="auto_accept",
                category=_accepted("category", "weapon"),
                canonical_object=_accepted("canonical_object", "sword"),
            )
        }
        result = evaluate_v3_against_golden(golden, v3)
        assert result.provenance_completeness == 1.0

    def test_provenance_completeness_flags_missing_evidence(self):
        golden = {"a": GoldenLabel("a", "weapon", "sword")}
        no_prov = FieldDecision(field_name="category", state="accepted", accepted_value="weapon")  # no evidence_refs
        v3 = {
            "a": RecordDecision(
                sprite_id="a",
                record_state="auto_accept",
                category=no_prov,
                canonical_object=_accepted("canonical_object", "sword"),
            )
        }
        result = evaluate_v3_against_golden(golden, v3)
        assert result.provenance_completeness < 1.0


class TestPromotionFromResult:
    def _result(self, **gates) -> V3EvalResult:
        r = V3EvalResult(suite_name="s")
        r.matched = 100
        r.promotion_gates = gates
        return r

    def test_recommendation_uses_gates(self):
        blocked = self._result(
            category_meets_target=False,
            canonical_object_meets_target=True,
            hard_reject_zero_fpr=True,
            ece_acceptable=True,
        )
        assert promotion_recommendation(blocked) == "blocked"


class TestApplyV3:
    def _records_file(self, tmp_path):
        from spritelab.harvest.label_v3.record_decisions import record_decision_to_json

        recs = [
            RecordDecision(
                sprite_id="ok",
                record_state="auto_accept",
                category=_accepted("category", "weapon"),
                canonical_object=_accepted("canonical_object", "sword"),
            ),
            RecordDecision(
                sprite_id="partial",
                record_state="partial_accept",
                category=_accepted("category", "item_icon"),
                canonical_object=FieldDecision(field_name="canonical_object", state="abstained"),
            ),
            RecordDecision(
                sprite_id="quar",
                record_state="quarantine",
                category=FieldDecision(field_name="category", state="quarantined"),
            ),
        ]
        p = tmp_path / "v3_records.jsonl"
        import json as _json

        p.write_text("\n".join(_json.dumps(record_decision_to_json(r)) for r in recs), encoding="utf-8")
        return p

    def test_dry_run_writes_nothing(self, tmp_path):
        from spritelab.harvest.label_v3.apply_v3 import apply_v3_records

        recs = self._records_file(tmp_path)
        out = tmp_path / "applied"
        report = apply_v3_records(recs, out, dry_run=True)
        assert report.applied == 2  # auto_accept + partial_accept
        assert report.excluded == 1  # quarantine
        assert not (out / "label_v3_suggestions.jsonl").exists()

    def test_apply_writes_sidecar_and_migration(self, tmp_path):
        from spritelab.harvest.label_v3.apply_v3 import apply_v3_records

        recs = self._records_file(tmp_path)
        out = tmp_path / "applied"
        apply_v3_records(recs, out, dry_run=False)
        assert (out / "label_v3_suggestions.jsonl").is_file()
        assert (out / "v3_migration_report.json").is_file()
        # Only applied records are in the sidecar.
        lines = (out / "label_v3_suggestions.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_refuses_to_overwrite_historical(self, tmp_path):
        from spritelab.harvest.label_v3.apply_v3 import apply_v3_records

        recs = self._records_file(tmp_path)
        out = tmp_path / "hist"
        out.mkdir()
        (out / "imported.jsonl").write_text("{}", encoding="utf-8")  # simulate a historical run
        with pytest.raises(ValueError):
            apply_v3_records(recs, out, dry_run=False)


class TestFrozenSuiteLeakage:
    def test_clean_suite_passes(self, tmp_path):
        manifest = FrozenSuiteManifest(
            suite_name="v1",
            taxonomy_version="v3.1",
            partitions={
                "in_domain": ("a", "b"),
                "unseen_pack": ("c", "d"),
                "source_ood": ("e", "f"),
            },
        )
        report = check_suite_leakage(manifest, tuning_ids=("t1", "t2"))
        assert report.ok is True
        assert report.tuning_overlap == ()

    def test_cross_partition_overlap_detected(self):
        manifest = FrozenSuiteManifest(partitions={"in_domain": ("a", "b"), "unseen_pack": ("b", "c")})
        report = check_suite_leakage(manifest)
        assert report.ok is False
        assert "in_domain&unseen_pack" in report.cross_partition_overlaps

    def test_tuning_leakage_detected(self):
        manifest = FrozenSuiteManifest(partitions={"in_domain": ("a", "b")})
        report = check_suite_leakage(manifest, tuning_ids=("b", "z"))
        assert report.ok is False
        assert "b" in report.tuning_overlap

    def test_manifest_roundtrip(self, tmp_path):
        manifest = FrozenSuiteManifest(
            suite_name="v1",
            taxonomy_version="v3.1",
            annotation_guidance="freeze",
            partitions={"in_domain": ("a",)},
        )
        path = tmp_path / "suite.json"
        manifest.save(path)
        restored = FrozenSuiteManifest.load(path)
        assert restored.suite_name == "v1"
        assert restored.partition_ids("in_domain") == ("a",)
