"""Auto-Labeling v3 completion pass: hierarchy-aware conflict, evaluator rigor,
stage cache, pluggable VLM, frozen-suite QA.

These tests exercise the correctness requirements added in the completion mission
without weakening any previously-fixed P0/P1 behavior.
"""

from __future__ import annotations

import pytest

from spritelab.harvest.label_v3.evidence import EvidenceItem
from spritelab.harvest.label_v3.fusion_v3 import (
    _field_conflict,
    fuse_hierarchical_object,
)
from spritelab.harvest.label_v3.taxonomy_v3 import taxonomy_relation


def _obj_ev(eid, group, value, score=0.85):
    return EvidenceItem(
        sprite_id="s",
        evidence_id=eid,
        evidence_family=group,
        dependency_group=group,
        target_fields=("canonical_object",),
        raw_score=score,
        proposed_value={"canonical_object": value},
    )


# ---------------------------------------------------------------------------
# Phase 1.1 — hierarchy-aware contradiction
# ---------------------------------------------------------------------------


class TestTaxonomyRelation:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("sword", "sword", "agree"),
            ("sword", "bladed_weapon", "compatible"),
            ("weapon", "sword", "compatible"),
            ("sword", "dagger", "sibling"),
            ("sword", "shield", "contradict"),
            ("weapon", "armor", "contradict"),
            ("rapier", "sword", "unknown"),  # rapier not a node -> cannot prove conflict
        ],
    )
    def test_relations(self, a, b, expected):
        assert taxonomy_relation(a, b) == expected


class TestFieldConflict:
    def test_compatible_is_not_conflict(self):
        items = (_obj_ev("a", "g1", "sword"), _obj_ev("b", "g2", "bladed_weapon"))
        assert _field_conflict(items, field="canonical_object") == "none"

    def test_siblings_are_ambiguous(self):
        items = (_obj_ev("a", "g1", "sword"), _obj_ev("b", "g2", "dagger"))
        assert _field_conflict(items, field="canonical_object") == "ambiguous"

    def test_cross_subtree_is_contradiction(self):
        items = (_obj_ev("a", "g1", "sword"), _obj_ev("b", "g2", "shield"))
        assert _field_conflict(items, field="canonical_object") == "contradiction"

    def test_unknown_child_not_contradiction(self):
        items = (_obj_ev("a", "g1", "rapier"), _obj_ev("b", "g2", "sword"))
        assert _field_conflict(items, field="canonical_object") == "none"

    def test_weak_signal_cannot_create_conflict(self):
        strong = _obj_ev("a", "sheet", "sword", score=0.96)
        weak = _obj_ev("b", "filename", "shield", score=0.3)
        assert _field_conflict((strong, weak), field="canonical_object") == "none"

    def test_flat_field_distinct_values_contradict(self):
        color_a = EvidenceItem(
            sprite_id="s",
            evidence_id="a",
            evidence_family="g1",
            dependency_group="g1",
            target_fields=("color",),
            raw_score=0.8,
            proposed_value={"color": "red"},
        )
        color_b = EvidenceItem(
            sprite_id="s",
            evidence_id="b",
            evidence_family="g2",
            dependency_group="g2",
            target_fields=("color",),
            raw_score=0.8,
            proposed_value={"color": "blue"},
        )
        assert _field_conflict((color_a, color_b), field="color") == "contradiction"


class TestEvaluatorRigor:
    def _golden_and_v3(self):
        from spritelab.harvest.golden import GoldenLabel
        from spritelab.harvest.label_v3.field_decisions import FieldDecision
        from spritelab.harvest.label_v3.record_decisions import RecordDecision

        def acc(name, val, correct_prob=0.99):
            return FieldDecision(
                field_name=name,
                state="accepted",
                accepted_value=val,
                evidence_refs=("ev",),
                policy_hash="ph",
                calibrated_estimate=correct_prob,
            )

        golden = {
            "a": GoldenLabel("a", "weapon", "sword"),
            "b": GoldenLabel("b", "item_icon", "potion"),
            "c": GoldenLabel("c", "weapon", "axe"),
        }
        v3 = {
            "a": RecordDecision(
                sprite_id="a",
                record_state="auto_accept",
                category=acc("category", "weapon"),
                canonical_object=acc("canonical_object", "sword"),
            ),
            "b": RecordDecision(
                sprite_id="b",
                record_state="partial_accept",
                category=acc("category", "item_icon"),
                canonical_object=FieldDecision(field_name="canonical_object", state="abstained"),
            ),
            "c": RecordDecision(
                sprite_id="c",
                record_state="auto_accept",
                category=acc("category", "weapon"),
                canonical_object=acc("canonical_object", "hammer"),
            ),  # wrong
        }
        return golden, v3

    def test_ratios_have_numerator_denominator_and_bounded(self):
        from spritelab.harvest.label_v3.label_v3_eval import evaluate_v3_against_golden

        golden, v3 = self._golden_and_v3()
        res = evaluate_v3_against_golden(golden, v3)
        cat = res.per_field["category"]
        assert cat["precision_numerator"] == 3 and cat["precision_denominator"] == 3
        assert 0.0 <= cat["precision"] <= 1.0
        obj = res.per_field["canonical_object"]
        # a=sword correct, c=hammer wrong -> 1/2 accepted correct.
        assert obj["precision_numerator"] == 1 and obj["precision_denominator"] == 2
        assert abs(obj["selective_risk"] - 0.5) < 1e-9

    def test_unsupported_field_marked_not_scored(self):
        from spritelab.harvest.label_v3.label_v3_eval import evaluate_v3_against_golden

        golden, v3 = self._golden_and_v3()
        res = evaluate_v3_against_golden(golden, v3)
        assert res.per_field["material"]["scored"] is False
        assert res.per_field["domain"]["scored"] is False

    def test_selective_risk_by_stratum(self):
        from spritelab.harvest.label_v3.label_v3_eval import selective_risk_by_stratum

        golden, v3 = self._golden_and_v3()
        strata = selective_risk_by_stratum(golden, v3, field="canonical_object", stratify_by="category")
        # weapon stratum: a correct, c wrong -> precision 0.5
        assert strata["weapon"]["precision_denominator"] == 2
        assert strata["weapon"]["correct"] == 1

    def test_calibration_eval_overlap_gate(self):
        from spritelab.harvest.label_v3.label_v3_eval import (
            CalibrationEvalOverlapError,
            assert_calibration_eval_disjoint,
        )

        assert_calibration_eval_disjoint(["a", "b"], ["c", "d"])  # ok
        with pytest.raises(CalibrationEvalOverlapError):
            assert_calibration_eval_disjoint(["a", "b"], ["b", "c"])


class TestStageCache:
    def _deps(self, **over):
        base = {
            "input_content_hash": "img1",
            "stage_version": "v1",
            "model_identity": "m1",
            "prompt_hash": "p1",
            "image_view": "magenta_matte",
            "preprocessing_hash": "pre1",
            "taxonomy_hash": "t1",
            "source_profiles_hash": "sp1",
            "sheet_mapping_hash": "sm1",
            "policy_hash": "pol1",
        }
        base.update(over)
        return base

    def _key(self, stage, deps):
        from spritelab.harvest.label_v3.stage_cache_v3 import STAGE_DEPENDENCIES, stage_cache_key

        return stage_cache_key(stage, {k: deps[k] for k in STAGE_DEPENDENCIES[stage]})

    def test_prompt_change_invalidates_vlm_not_deterministic(self):
        d0 = self._deps()
        d1 = self._deps(prompt_hash="p2")
        det0 = self._key("deterministic_evidence", d0)
        det1 = self._key("deterministic_evidence", d1)
        vlm0 = self._key("vlm_blind_descriptor", d0)
        vlm1 = self._key("vlm_blind_descriptor", d1)
        assert det0 == det1  # deterministic evidence does not consume the prompt
        assert vlm0 != vlm1  # VLM stage does

    def test_taxonomy_change_invalidates_fusion_not_blind_descriptor(self):
        d0 = self._deps()
        d1 = self._deps(taxonomy_hash="t2")
        assert self._key("fusion", d0) != self._key("fusion", d1)
        assert self._key("vlm_blind_descriptor", d0) == self._key("vlm_blind_descriptor", d1)

    def test_irrelevant_dependency_rejected(self):
        from spritelab.harvest.label_v3.stage_cache_v3 import IrrelevantDependencyError, stage_cache_key

        with pytest.raises(IrrelevantDependencyError):
            # deterministic evidence must not key on the prompt.
            stage_cache_key(
                "deterministic_evidence",
                {
                    "input_content_hash": "i",
                    "stage_version": "v",
                    "source_profiles_hash": "s",
                    "sheet_mapping_hash": "m",
                    "preprocessing_hash": "p",
                    "prompt_hash": "leak",
                },
            )

    def test_missing_dependency_rejected(self):
        from spritelab.harvest.label_v3.stage_cache_v3 import MissingDependencyError, stage_cache_key

        with pytest.raises(MissingDependencyError):
            stage_cache_key("fusion", {"input_content_hash": "i", "stage_version": "v", "taxonomy_hash": "t"})

    def test_get_or_compute_idempotent(self, tmp_path):
        from spritelab.harvest.label_v3.stage_cache_v3 import StageCache

        cache = StageCache(tmp_path / "cache")
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return {"result": 42}

        v1, hit1 = cache.get_or_compute("abc123", compute)
        v2, hit2 = cache.get_or_compute("abc123", compute)
        assert v1 == v2 == {"result": 42}
        assert hit1 is False and hit2 is True
        assert calls["n"] == 1  # computed once

    def test_shard_run_reuses_cache_across_output_roots(self, tmp_path):
        import json

        from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
        from spritelab.harvest.label_v3.pipeline_stages_v3 import merge_v3_shards, run_v3_shard

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        recs = [
            {
                "sprite_id": f"s{i}",
                "source_id": "x",
                "source_name": "P",
                "relative_path": f"p/{i}.png",
                "final_png_path": f"p/{i}.png",
                "status": "accepted",
            }
            for i in range(8)
        ]
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

        cache_dir = tmp_path / "cache"
        cfg = V3PipelineConfig()

        r1 = run_v3_shard(run_dir, tmp_path / "outA", cfg, shard_count=1, resume=False, cache_dir=cache_dir)
        assert r1.cache_hits == 0 and r1.processed == 8
        # Fresh output root, same cache -> all hits, identical merged output.
        r2 = run_v3_shard(run_dir, tmp_path / "outB", cfg, shard_count=1, resume=False, cache_dir=cache_dir)
        assert r2.cache_hits == 8
        merge_v3_shards(tmp_path / "outA")
        merge_v3_shards(tmp_path / "outB")
        assert (tmp_path / "outA" / "v3_records.jsonl").read_bytes() == (
            tmp_path / "outB" / "v3_records.jsonl"
        ).read_bytes()


class TestPluggableVlm:
    def _mock_backend(self):
        from spritelab.harvest.label_v3.vlm_orchestration import VlmUnavailable

        class MockBackend:
            model_identity = "mock-vlm-1"

            def __init__(self):
                self.calls = []

            def infer(self, *, stage_id, image_ref, prompt, prompt_hash, candidates=None):
                self.calls.append({"stage": stage_id, "candidates": candidates, "prompt": prompt})
                if stage_id == "stage_a_blind_descriptor":
                    return {"description": "a long thin metallic object", "confidence": 0.99}
                if stage_id == "stage_b_morphology":
                    return {"shape": "elongated", "confidence": 0.8}
                if stage_id == "stage_c_constrained_classification":
                    return {"canonical_object": "sword", "confidence": 0.95}
                if stage_id == "stage_d_open_set_verify":
                    return {"none_of_the_above": False, "canonical_object": "sword"}
                if stage_id == "stage_e_consistency":
                    return {"malformed": "yes", "_malformed": True}  # force a quarantine
                raise VlmUnavailable("unexpected stage")

        return MockBackend()

    def test_unavailable_backend_is_resumable(self):
        from spritelab.harvest.label_v3.vlm_orchestration import run_vlm_cascade

        res = run_vlm_cascade("s", backend=None)
        assert res.all_failed is True
        assert res.all_evidence() == ()

    def test_blind_stage_receives_no_candidates(self):
        from spritelab.harvest.label_v3.vlm_orchestration import run_vlm_cascade

        backend = self._mock_backend()
        run_vlm_cascade("s", backend=backend, candidates=("sword", "dagger"), image_hash="h1")
        blind_calls = [c for c in backend.calls if c["stage"] == "stage_a_blind_descriptor"]
        assert blind_calls and blind_calls[0]["candidates"] is None
        assert "sword" not in blind_calls[0]["prompt"]
        constrained = [c for c in backend.calls if c["stage"] == "stage_c_constrained_classification"]
        assert constrained and constrained[0]["candidates"] == ("sword", "dagger")

    def test_malformed_output_quarantined_not_trusted(self):
        from spritelab.harvest.label_v3.vlm_orchestration import run_vlm_cascade

        res = run_vlm_cascade("s", backend=self._mock_backend(), candidates=("sword",), image_hash="h1")
        # stage_e returned malformed -> unavailable, not an accepted decision.
        assert res.stage_e.available is False
        assert res.stage_e.failure_reason == "invalid_stage_contract"
        assert res.stage_c.available is True

    def test_self_reported_confidence_is_feature_not_weight(self):
        from spritelab.harvest.label_v3.vlm_orchestration import VLM_STAGE_RELIABILITY, run_vlm_cascade

        res = run_vlm_cascade("s", backend=self._mock_backend(), candidates=("sword",), image_hash="h1")
        ev = res.stage_c.evidence
        assert ev is not None
        # raw_score is the fixed stage reliability, NOT the reported 0.95.
        assert ev.raw_score == VLM_STAGE_RELIABILITY
        assert ev.provenance.get("self_reported_confidence") == 0.95
        assert ev.provenance.get("confidence_is_feature_only") is True

    def test_vlm_stages_share_dependency_group(self):
        from spritelab.harvest.label_v3.vlm_orchestration import run_vlm_cascade

        res = run_vlm_cascade("s", backend=self._mock_backend(), candidates=("sword",), image_hash="h1")
        groups = {e.dependency_group for e in res.all_evidence()}
        assert groups == {"vlm_mock-vlm-1"}  # one model -> one correlated group

    def test_stage_cache_reused(self, tmp_path):
        from spritelab.harvest.label_v3.stage_cache_v3 import StageCache
        from spritelab.harvest.label_v3.vlm_orchestration import run_vlm_cascade

        cache = StageCache(tmp_path / "vlmcache")
        b1 = self._mock_backend()
        run_vlm_cascade("s", backend=b1, candidates=("sword",), image_hash="h1", cache=cache)
        b2 = self._mock_backend()
        run_vlm_cascade("s", backend=b2, candidates=("sword",), image_hash="h1", cache=cache)
        # Second run hit the cache for the well-formed stages, so b2 was not
        # called for them (only the malformed stage_e is recomputed each time).
        called_stages = {c["stage"] for c in b2.calls}
        assert "stage_a_blind_descriptor" not in called_stages


class TestHierarchicalObjectFusion:
    def test_compatible_evidence_accepts_specific(self):
        items = (_obj_ev("a", "g1", "sword", 0.9), _obj_ev("b", "g2", "bladed_weapon", 0.8))
        res = fuse_hierarchical_object(items, sprite_id="s", auto_accept_enabled=True)
        # Compatible agreement -> accept the specific, calibrated node.
        assert res.decision.state == "accepted"
        assert res.decision.accepted_value == "sword"

    def test_contradiction_quarantines_object(self):
        items = (_obj_ev("a", "g1", "sword"), _obj_ev("b", "g2", "shield"))
        res = fuse_hierarchical_object(items, sprite_id="s", auto_accept_enabled=True)
        assert res.decision.state == "quarantined"

    def test_sibling_object_is_ambiguous(self):
        items = (_obj_ev("a", "g1", "sword"), _obj_ev("b", "g2", "dagger"))
        res = fuse_hierarchical_object(items, sprite_id="s", auto_accept_enabled=True)
        assert res.decision.state == "ambiguous"
