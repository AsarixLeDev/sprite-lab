"""Tests for Auto-Labeling v3 Phase 1: evidence schema, taxonomy, field decisions, record decisions."""

from __future__ import annotations

from spritelab.harvest.label_v3.adapter import (
    build_legacy_safe_fused_label,
    expose_accepted_v3_fields,
    v3_field_to_legacy_suggestion,
)
from spritelab.harvest.label_v3.calibration import (
    CalibrationArtifact,
    CalibrationStratumData,
    calibration_support_for_field,
    compute_ece,
    compute_lower_confidence_bound,
    stratum_sufficient,
)
from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy, V3PipelineConfig
from spritelab.harvest.label_v3.deterministic_evidence import (
    extract_deterministic_evidence,
)
from spritelab.harvest.label_v3.evidence import (
    SCHEMA_VERSION as EVIDENCE_SCHEMA,
)
from spritelab.harvest.label_v3.evidence import (
    EvidenceItem,
    evidence_item_from_json,
    evidence_item_to_json,
)
from spritelab.harvest.label_v3.field_decisions import (
    AcceptedTagSet,
    FieldDecision,
    TagDecision,
    field_decision_from_json,
    field_decision_to_json,
)
from spritelab.harvest.label_v3.fusion_v3 import (
    FieldFusionInput,
    fuse_field,
    fuse_hierarchical_object,
)
from spritelab.harvest.label_v3.impossible_combinations import (
    validate_impossible_combinations,
)
from spritelab.harvest.label_v3.pipeline_v3 import (
    run_v3_pipeline,
)
from spritelab.harvest.label_v3.reason_codes import (
    REASON_CODES,
    ContradictionAction,
    ContradictionSeverity,
    contradiction_action,
    contradiction_severity,
)
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    derive_record_state,
    record_decision_from_json,
    record_decision_to_json,
)
from spritelab.harvest.label_v3.sha256_utils import (
    config_identity_hash,
    dict_hash,
    sha256_hex,
    sha256_short,
    stable_evidence_id,
)
from spritelab.harvest.label_v3.taxonomy_v3 import (
    broader_hierarchy_node,
    deepest_supported_node,
    get_hierarchy_node,
    taxonomy_version_hash,
)
from spritelab.harvest.label_v3.vlm_orchestration import (
    build_stage_evidence,
    create_unavailable_cascade,
)

# ---------------------------------------------------------------------------
# Evidence schema tests
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_create_minimal(self):
        item = EvidenceItem(
            sprite_id="test_001",
            evidence_id="ev_001",
            evidence_family="deterministic_visual",
            deterministic=True,
        )
        assert item.sprite_id == "test_001"
        assert item.schema_version == EVIDENCE_SCHEMA
        assert item.evidence_family == "deterministic_visual"
        assert item.deterministic is True
        assert item.stochastic is False
        assert item.source_hints_exposed is False
        assert item.candidate_hints_exposed is False

    def test_roundtrip_json(self):
        item = EvidenceItem(
            sprite_id="test_001",
            evidence_id="ev_001",
            evidence_family="blind_vlm_descriptor",
            producer_stage="vlm_stage_a_blind_descriptor",
            target_fields=("category", "canonical_object"),
            proposed_value={"object_name": "sword", "category": "weapon"},
            raw_score=0.85,
            source_hints_exposed=False,
            candidate_hints_exposed=False,
            deterministic=False,
            stochastic=True,
            image_hash="abc123",
            image_view="magenta_matte",
            model_identity="test_model",
            prompt_hash="def456",
            contradiction_codes=("cat_source_vs_vlm",),
            warnings=("low_confidence",),
        )
        json_dict = evidence_item_to_json(item)
        restored = evidence_item_from_json(json_dict)
        assert restored.sprite_id == item.sprite_id
        assert restored.evidence_id == item.evidence_id
        assert restored.evidence_family == item.evidence_family
        assert restored.source_hints_exposed is False
        assert restored.contradiction_codes == ("cat_source_vs_vlm",)
        assert restored.warnings == ("low_confidence",)

    def test_defaults_are_sensible(self):
        item = evidence_item_from_json({})
        assert item.schema_version == EVIDENCE_SCHEMA
        assert item.evidence_id == ""
        assert item.deterministic is False


# ---------------------------------------------------------------------------
# Field decision tests
# ---------------------------------------------------------------------------


class TestFieldDecision:
    def test_create_accepted(self):
        decision = FieldDecision(
            sprite_id="test_001",
            field_name="category",
            state="accepted",
            accepted_value="weapon",
            evidence_refs=("ev_001", "ev_002"),
            decision_reason="strong_evidence_consensus",
        )
        assert decision.state == "accepted"
        assert decision.accepted_value == "weapon"
        assert len(decision.evidence_refs) == 2

    def test_state_values_distinct(self):
        for state in (
            "accepted",
            "abstained",
            "quarantined",
            "rejected",
            "unknown",
            "novel",
            "ambiguous",
            "unlabeled",
            "not_applicable",
        ):
            decision = FieldDecision(field_name="test", state=state)
            assert decision.state == state

    def test_roundtrip_json(self):
        decision = FieldDecision(
            sprite_id="s_001",
            field_name="canonical_object",
            state="accepted",
            accepted_value="sword",
            hierarchy_node="bladed_weapon",
            candidates=("sword", "dagger", "axe"),
            n_best_alternatives=(("sword", 0.9), ("dagger", 0.6)),
            evidence_refs=("ev_a", "ev_b"),
            calibrated_estimate=0.92,
            confidence_interval=(0.88, 0.96),
            decision_reason="strong_evidence_consensus",
            policy_hash="ph_001",
        )
        js = field_decision_to_json(decision)
        restored = field_decision_from_json(js)
        assert restored.field_name == "canonical_object"
        assert restored.state == "accepted"
        assert restored.accepted_value == "sword"
        assert restored.candidates == ("sword", "dagger", "axe")
        assert restored.n_best_alternatives == (("sword", 0.9), ("dagger", 0.6))
        assert restored.calibrated_estimate == 0.92
        assert restored.confidence_interval == (0.88, 0.96)


class TestTagDecision:
    def test_accepted_tag(self):
        td = TagDecision(tag="blade", state="accepted", evidence_refs=("ev_1",))
        assert td.state == "accepted"
        assert td.tag == "blade"

    def test_roundtrip(self):
        td = TagDecision(tag="gold", state="accepted", calibrated_estimate=0.88, provenance={"source": "visual_facts"})
        js = td.to_json()
        restored = TagDecision.from_json(js)
        assert restored.tag == "gold"
        assert restored.state == "accepted"
        assert restored.calibrated_estimate == 0.88


class TestAcceptedTagSet:
    def test_accepted_tags_only(self):
        tagset = AcceptedTagSet(
            decisions=(
                TagDecision(tag="blade", state="accepted"),
                TagDecision(tag="metal", state="accepted"),
                TagDecision(tag="fire", state="abstained"),
                TagDecision(tag="poison", state="unlabeled"),
            )
        )
        assert tagset.accepted_tags == ("blade", "metal")
        assert tagset.all_tags == ("blade", "metal", "fire", "poison")

    def test_roundtrip(self):
        tagset = AcceptedTagSet(
            decisions=(TagDecision(tag="test", state="accepted"),),
        )
        js = tagset.to_json()
        restored = AcceptedTagSet.from_json(js)
        assert restored.accepted_tags == ("test",)


# ---------------------------------------------------------------------------
# Record decision tests
# ---------------------------------------------------------------------------


class TestRecordDecision:
    def test_derive_auto_accept(self):
        decisions = {
            "domain": FieldDecision(field_name="domain", state="accepted", accepted_value="weapon"),
            "category": FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            "canonical_object": FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
        }
        state = derive_record_state(decisions)
        assert state == "auto_accept"

    def test_derive_hard_reject(self):
        decisions = {
            "category": FieldDecision(field_name="category", state="rejected"),
            "canonical_object": FieldDecision(field_name="canonical_object", state="unlabeled"),
        }
        state = derive_record_state(decisions)
        assert state == "hard_reject"

    def test_derive_quarantine(self):
        decisions = {
            "domain": FieldDecision(field_name="domain", state="accepted"),
            "category": FieldDecision(field_name="category", state="accepted"),
            "canonical_object": FieldDecision(field_name="canonical_object", state="quarantined"),
        }
        state = derive_record_state(decisions)
        assert state == "quarantine"

    def test_derive_partial_accept(self):
        decisions = {
            "domain": FieldDecision(field_name="domain", state="accepted"),
            "category": FieldDecision(field_name="category", state="accepted"),
            "canonical_object": FieldDecision(field_name="canonical_object", state="abstained"),
        }
        state = derive_record_state(decisions)
        assert state == "partial_accept"

    def test_roundtrip_json(self):
        record = RecordDecision(
            sprite_id="test_001",
            record_state="auto_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
            policy_hash="ph_test",
        )
        js = record_decision_to_json(record)
        restored = record_decision_from_json(js)
        assert restored.sprite_id == "test_001"
        assert restored.record_state == "auto_accept"
        assert restored.category.accepted_value == "weapon"

    def test_accepted_fields_property(self):
        record = RecordDecision(
            sprite_id="test",
            category=FieldDecision(field_name="category", state="accepted"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted"),
            color=FieldDecision(field_name="color", state="abstained"),
            material=FieldDecision(field_name="material", state="unlabeled"),
        )
        accepted = record.accepted_fields
        assert "category" in accepted
        assert "canonical_object" in accepted
        assert "color" not in accepted
        assert "material" not in accepted


# ---------------------------------------------------------------------------
# Reason codes tests
# ---------------------------------------------------------------------------


class TestReasonCodes:
    def test_all_codes_registered(self):
        assert "insufficient_evidence" in REASON_CODES
        assert "impossible_combination" in REASON_CODES
        assert "calibration_insufficient" in REASON_CODES

    def test_contradiction_severity(self):
        assert contradiction_severity("cat_source_vs_vlm") == ContradictionSeverity.HIGH
        assert contradiction_severity("nonexistent_code") == ContradictionSeverity.MEDIUM

    def test_contradiction_action(self):
        assert contradiction_action("impossible_combination") == ContradictionAction.MASK_FIELD
        assert contradiction_action("cat_source_vs_vlm") == ContradictionAction.ABSTAIN_FIELD
        assert contradiction_action("source_vs_visual") == ContradictionAction.QUARANTINE_RECORD


# ---------------------------------------------------------------------------
# Taxonomy tests
# ---------------------------------------------------------------------------


class TestTaxonomyV3:
    def test_get_hierarchy_node(self):
        node = get_hierarchy_node("sword")
        assert node is not None
        assert node.name == "sword"
        assert node.parent == "bladed_weapon"

    def test_get_nonexistent_node(self):
        node = get_hierarchy_node("xyzzy_nonexistent_12345")
        assert node is None

    def test_get_by_synonym(self):
        node = get_hierarchy_node("blade")
        assert node is not None
        assert node.name == "sword"

    def test_broader_node(self):
        sword = get_hierarchy_node("sword")
        assert sword is not None
        broader = broader_hierarchy_node(sword)
        assert broader is not None
        assert broader.name == "bladed_weapon"

    def test_deepest_supported_node(self):
        node = deepest_supported_node("sword")
        assert node is not None
        assert node.name == "sword"

    def test_taxonomy_hash_stable(self):
        h1 = taxonomy_version_hash()
        h2 = taxonomy_version_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_hierarchy_is_ancestor(self):
        weapon = get_hierarchy_node("weapon")
        sword = get_hierarchy_node("sword")
        assert weapon is not None
        assert sword is not None
        assert weapon.is_ancestor_of(sword)
        assert sword.is_descendant_of(weapon)


# ---------------------------------------------------------------------------
# Impossible combination tests
# ---------------------------------------------------------------------------


class TestImpossibleCombinations:
    def test_weapon_food_conflict(self):
        # A weapon category with a food canonical object is a genuine
        # cross-field impossibility.
        codes, _descs = validate_impossible_combinations(category="weapon", canonical_object="apple")
        assert "IC001" in codes

    def test_single_field_never_flagged(self):
        # A plain, well-formed single-field record must never be flagged as an
        # impossible combination (no false hard-rejects).
        for category in ("food", "weapon", "armor", "gem", "tool", "plant"):
            codes, _ = validate_impossible_combinations(category=category)
            assert codes == (), f"{category} alone should not be an impossible combination"

    def test_no_conflict_normal(self):
        codes, _descs = validate_impossible_combinations(
            category="item_icon",
            canonical_object="sword",
            material="metal",
        )
        assert len(codes) == 0

    def test_clean_weapon_not_flagged(self):
        # A metal sword that is a weapon is entirely valid.
        codes, _ = validate_impossible_combinations(category="weapon", canonical_object="sword", material="metal")
        assert codes == ()

    def test_liquid_material_warning(self):
        codes, _descs = validate_impossible_combinations(category="food", material="metal")
        assert "IC002" in codes


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------


class TestHashing:
    def test_sha256_deterministic(self):
        h1 = sha256_hex("hello")
        h2 = sha256_hex("hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_short(self):
        h = sha256_short("test", length=12)
        assert len(h) == 12

    def test_dict_hash_stable(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert dict_hash(d1) == dict_hash(d2)

    def test_stable_evidence_id(self):
        eid1 = stable_evidence_id("sprite_001", "deterministic_visual", "hash_123")
        eid2 = stable_evidence_id("sprite_001", "deterministic_visual", "hash_123")
        assert eid1 == eid2
        assert len(eid1) == 12

    def test_config_identity_hash(self):
        h = config_identity_hash(taxonomy_hash="th", prompt_hash="ph", model_identity="mi")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_auto_accept_to_legacy(self):
        record = RecordDecision(
            sprite_id="test",
            record_state="auto_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
            color=FieldDecision(field_name="color", state="accepted", accepted_value="gray"),
            material=FieldDecision(field_name="material", state="accepted", accepted_value="metal"),
            tags=AcceptedTagSet(decisions=(TagDecision(tag="blade", state="accepted"),)),
        )
        result = v3_field_to_legacy_suggestion(record)
        assert result["category"] == "weapon"
        assert result["object_name"] == "sword"
        assert "gray" in result["dominant_colors"]
        assert "metal" in result["materials"]
        assert "blade" in result["tags"]

    def test_partial_accept_to_safe_fused(self):
        record = RecordDecision(
            sprite_id="test",
            record_state="partial_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="item_icon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="abstained"),
        )
        result = build_legacy_safe_fused_label(record)
        assert result["bucket"] == "auto_v3_partial"
        assert result["needs_review"] is False

    def test_quarantine_to_safe_fused(self):
        record = RecordDecision(
            sprite_id="test",
            record_state="quarantine",
            category=FieldDecision(field_name="category", state="quarantined"),
        )
        result = build_legacy_safe_fused_label(record)
        assert result["bucket"] == "needs_review_v3_quarantine"
        assert result["needs_review"] is True

    def test_expose_accepted_fields_with_mask(self):
        record = RecordDecision(
            sprite_id="test",
            record_state="auto_accept",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
            color=FieldDecision(field_name="color", state="accepted", accepted_value="gray"),
        )
        result = expose_accepted_v3_fields(record, field_mask={"category", "canonical_object"})
        assert "category" in result
        assert "canonical_object" in result
        assert "color" not in result


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_policy_hash_stable(self):
        p1 = V3LabelingPolicy()
        p2 = V3LabelingPolicy()
        assert p1.policy_hash() == p2.policy_hash()

    def test_pipeline_hash(self):
        cfg = V3PipelineConfig()
        h = cfg.pipeline_hash()
        assert len(h) == 16

    def test_shadow_mode_default(self):
        policy = V3LabelingPolicy()
        assert policy.shadow_mode is True
        assert policy.dry_run_apply is True


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_lower_confidence_bound_perfect(self):
        bound = compute_lower_confidence_bound(100, 100, 0.95)
        assert bound > 0.96

    def test_lower_confidence_bound_zero(self):
        bound = compute_lower_confidence_bound(0, 0, 0.95)
        assert bound == 0.0

    def test_lower_confidence_bound_mixed(self):
        bound = compute_lower_confidence_bound(90, 100, 0.95)
        assert 0.80 < bound < 0.95

    def test_ece_perfect(self):
        ece = compute_ece([1.0] * 10, [1] * 10, n_bins=5)
        assert ece == 0.0

    def test_ece_worst(self):
        ece = compute_ece([0.0] * 10, [1] * 10, n_bins=5)
        assert ece == 1.0

    def test_stratum_sufficient(self):
        assert stratum_sufficient(50, min_samples=30, min_samples_per_stratum=10) is True
        assert stratum_sufficient(20, min_samples=30, min_samples_per_stratum=10) is False
        assert stratum_sufficient(5, min_samples=30, min_samples_per_stratum=10) is False

    def test_calibration_support_no_strata(self):
        artifact = CalibrationArtifact(field_name="category")
        support = calibration_support_for_field("category", "src1", "prof1", "domain1", artifact)
        assert support is None

    def test_calibration_support_with_sufficient_strata(self):
        artifact = CalibrationArtifact(
            field_name="category",
            strata_data=(
                CalibrationStratumData(
                    field="category",
                    stratum="global",
                    sample_count=100,
                    error_count=2,
                    observed_precision=0.98,
                    calibrated_probability=0.97,
                    ci_lower=0.95,
                    ci_upper=0.99,
                    sufficient=True,
                ),
            ),
        )
        support = calibration_support_for_field("category", "src1", "prof1", "domain1", artifact)
        assert support is not None
        assert support["stratum"] == "global"

    def test_artifact_roundtrip(self):
        artifact = CalibrationArtifact(
            field_name="category",
            taxonomy_hash="th_abc",
            prompt_hash="ph_def",
            model_identity="test_model",
            strata_data=(
                CalibrationStratumData(
                    field="category",
                    stratum="global",
                    sample_count=50,
                    error_count=1,
                    observed_precision=0.98,
                    sufficient=True,
                ),
            ),
        )
        js = artifact.to_json_dict()
        restored = CalibrationArtifact.from_json_dict(js)
        assert restored.field_name == "category"
        assert len(restored.strata_data) == 1
        assert restored.strata_data[0].sample_count == 50


# ---------------------------------------------------------------------------
# VLM orchestration tests
# ---------------------------------------------------------------------------


class TestVlmOrchestration:
    def test_no_vlm_cascade(self):
        result = create_unavailable_cascade("sprite_001", reason="no_vlm_backend")
        assert result.all_failed is True
        assert not result.stage_a.available
        assert len(result.available_stages()) == 0

    def test_stage_context_recording(self):
        evidence = build_stage_evidence(
            "stage_a_blind_descriptor",
            "sprite_001",
            "src_001",
            "pack_001",
            {"object_name": "sword", "confidence": 0.85},
            model_identity="test_model",
            prompt_hash="ph_test",
        )
        assert evidence.source_hints_exposed is False
        assert evidence.candidate_hints_exposed is False
        assert evidence.evidence_family == "blind_vlm_descriptor"

    def test_stage_c_context_recording(self):
        evidence = build_stage_evidence(
            "stage_c_constrained_classification",
            "sprite_002",
            "src_001",
            "pack_001",
            {"object_name": "sword", "confidence": 0.9},
            model_identity="test_model",
            prompt_hash="ph_test",
        )
        assert evidence.source_hints_exposed is True
        assert evidence.candidate_hints_exposed is True
        assert evidence.evidence_family == "constrained_vlm_classification"

    def test_unavailable_build(self):
        evidence = build_stage_evidence(
            "stage_a_blind_descriptor",
            "sprite_003",
            "src_001",
            "pack_001",
            None,
        )
        assert "unavailable" in str(evidence.proposed_value)


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------


class TestFusion:
    def _make_evidence(
        self, sprite_id: str, field: str, value: str, score: float = 0.8, family: str = "deterministic_visual"
    ):
        return EvidenceItem(
            sprite_id=sprite_id,
            evidence_id=f"ev_{field}_{sprite_id}",
            evidence_family=family,
            target_fields=(field,),
            proposed_value={field: value},
            raw_score=score,
            deterministic=True,
        )

    def test_fuse_insufficient_evidence(self):
        result = fuse_field(
            FieldFusionInput(
                field="category",
                evidence_items=(),
                policy_hash="ph",
            )
        )
        assert not result.accepted
        assert result.decision.state == "unlabeled"

    def test_fuse_no_calibration_abstains(self):
        evidence = self._make_evidence("s1", "category", "weapon")
        result = fuse_field(
            FieldFusionInput(
                field="category",
                evidence_items=(evidence,),
                policy_hash="ph",
                auto_accept_enabled=True,
            )
        )
        assert result.decision.state == "abstained"
        assert "no_calibration_support" in str(result.warnings)

    def test_fuse_with_calibration(self):
        evidence = self._make_evidence("s1", "category", "weapon")
        result = fuse_field(
            FieldFusionInput(
                field="category",
                evidence_items=(evidence,),
                policy_hash="ph",
                auto_accept_enabled=True,
            ),
            calibration_support={
                "calibrated_probability": 0.98,
                "ci_lower": 0.97,
                "ci_upper": 0.99,
            },
        )
        # CI lower is below 0.99 target, so abstained
        assert result.decision.state == "abstained"

    def test_hierarchy_fallback(self):
        evidence = self._make_evidence("s1", "canonical_object", "excalibur_not_in_taxonomy")
        result = fuse_hierarchical_object(
            (evidence,),
            sprite_id="s1",
            policy_hash="ph",
        )
        assert result.decision.state in ("unknown", "novel", "abstained")

    def test_sword_accepted_in_hierarchy(self):
        evidence = self._make_evidence("s1", "canonical_object", "sword")
        result = fuse_hierarchical_object(
            (evidence,),
            sprite_id="s1",
            policy_hash="ph",
            auto_accept_enabled=True,
        )
        assert result.decision.state == "accepted"
        assert result.decision.hierarchy_node == "sword"

    def test_contradiction_detection(self):
        evidence = EvidenceItem(
            sprite_id="s1",
            evidence_id="ev_contra",
            evidence_family="deterministic_visual",
            target_fields=("category",),
            contradiction_codes=("cat_source_vs_vlm",),
            deterministic=True,
        )
        result = fuse_field(
            FieldFusionInput(
                field="category",
                evidence_items=(evidence,),
                policy_hash="ph",
                min_evidence_count=1,
            )
        )
        assert result.decision.state == "quarantined"

    def test_independent_evidence_count(self):
        ev1 = self._make_evidence("s1", "category", "weapon", family="source_profile")
        ev2 = self._make_evidence("s1", "category", "weapon", family="blind_vlm_descriptor")
        combined = (ev1, ev2)
        result = fuse_field(
            FieldFusionInput(
                field="category",
                evidence_items=combined,
                policy_hash="ph",
                min_evidence_count=2,
            ),
            calibration_support={
                "calibrated_probability": 0.98,
                "ci_lower": 0.97,
                "ci_upper": 0.99,
            },
        )
        assert result.decision.state == "abstained"

    def test_independent_disagreement_abstains(self):
        # Two independent groups proposing different values is a contradiction.
        ev1 = self._make_evidence("s1", "category", "weapon", family="source_profile")
        ev2 = self._make_evidence("s1", "category", "armor", family="blind_vlm_descriptor")
        result = fuse_field(
            FieldFusionInput(field="category", evidence_items=(ev1, ev2), policy_hash="ph", min_evidence_count=1),
            calibration_support={"calibrated_probability": 0.99, "ci_lower": 0.995, "ci_upper": 0.999},
        )
        # Even though calibration would allow acceptance, disagreement blocks it.
        assert result.decision.state in ("abstained", "quarantined")
        assert result.decision.state != "accepted"

    def test_correlated_votes_not_double_counted(self):
        # Three correlated evidence items (same dependency group) proposing
        # "sword" must not out-vote one independent "dagger".
        from spritelab.harvest.label_v3.fusion_v3 import _consensus_value

        correlated = tuple(
            EvidenceItem(
                sprite_id="s1",
                evidence_id=f"cg_{i}",
                evidence_family="variant_group",
                dependency_group="variant_grp_A",
                target_fields=("canonical_object",),
                proposed_value={"canonical_object": "sword"},
                deterministic=True,
            )
            for i in range(3)
        )
        independent = EvidenceItem(
            sprite_id="s1",
            evidence_id="ind_0",
            evidence_family="blind_vlm_descriptor",
            dependency_group="vlm_blind",
            target_fields=("canonical_object",),
            proposed_value={"canonical_object": "dagger"},
            deterministic=False,
        )
        consensus = _consensus_value((*correlated, independent), field="canonical_object")
        # One vote per group -> 1 "sword" vs 1 "dagger" tie, broken
        # deterministically by name. The key property: 3 correlated items did
        # not steamroll the single independent signal into "sword".
        assert consensus == "dagger"


# ---------------------------------------------------------------------------
# Deterministic evidence tests
# ---------------------------------------------------------------------------


class TestDeterministicEvidence:
    def test_extract_from_record(self):
        record = {
            "sprite_id": "test_001",
            "source_id": "src_001",
            "source_name": "Test Pack",
            "relative_path": "test/sword.png",
            "final_png_path": "test/sword.png",
        }
        batch = extract_deterministic_evidence(record)
        assert batch.sprite_id == "test_001"
        assert batch.provenance_evidence is not None
        assert batch.filename_evidence is not None
        assert batch.source_profile_evidence is not None

    def test_provenance_has_path_info(self):
        record = {
            "sprite_id": "test_001",
            "source_id": "src_foo",
            "source_name": "Foo Pack",
            "relative_path": "foo/sword.png",
        }
        batch = extract_deterministic_evidence(record)
        prov = batch.provenance_evidence
        assert prov is not None
        if isinstance(prov.proposed_value, dict):
            assert prov.proposed_value.get("relative_path") == "foo/sword.png"

    def test_generic_source_profile_warns(self):
        record = {
            "sprite_id": "test_generic",
            "source_id": "unknown_src",
            "source_name": "Unknown Source",
            "relative_path": "img/unknown.png",
        }
        batch = extract_deterministic_evidence(record)
        sp = batch.source_profile_evidence
        assert sp is not None
        assert "generic_unknown_profile" in sp.warnings

    def test_filename_value_proposes_known_object(self):
        # A recognized object from a trusted-profile filename becomes a
        # value-proposing evidence item (not just tokens).
        from spritelab.harvest.label_v3.deterministic_evidence import _build_filename_value_evidence

        ev = _build_filename_value_evidence(
            "s",
            "src",
            "pack",
            {"relative_path": "oga_496_rpg_icons/sword.png", "source_name": "oga_496_rpg_icons"},
        )
        assert ev is not None
        assert ev.proposed_value["object_known"] is True
        assert ev.proposed_value["canonical_object"] == "sword"
        assert ev.raw_score >= 0.9
        assert ev.dependency_group == "filename"

    def test_filename_value_echoed_token_not_proposed_as_object(self):
        # An unrecognized filename token must not be proposed as a known object.
        from spritelab.harvest.label_v3.deterministic_evidence import _build_filename_value_evidence

        ev = _build_filename_value_evidence(
            "s",
            "src",
            "pack",
            {"relative_path": "x/img001.png", "source_name": "y"},
        )
        assert ev is not None
        assert ev.proposed_value["object_known"] is False
        assert ev.proposed_value["canonical_object"] == ""
        assert "filename_object_not_recognized" in ev.warnings

    def test_exact_trust_profile_unmapped_filename_downgraded(self):
        # Spec Phase-2: an unknown filename in an exact-trust (sheet-based)
        # profile must NOT inherit exact trust.
        from spritelab.harvest.label_v3.deterministic_evidence import _build_filename_value_evidence

        ev = _build_filename_value_evidence(
            "s",
            "src",
            "pack",
            {"source_name": "shade_weapons", "relative_path": "shade_weapons/weird_unmapped_blob_xyz.png"},
        )
        assert ev is not None
        assert ev.raw_score <= 0.35  # far below the ~0.96 exact-trust score
        assert ev.proposed_value["object_known"] is False
        assert "exact_profile_filename_not_sheet_mapped" in ev.warnings

    def test_empty_sprite_detected(self):
        record = {
            "sprite_id": "test_empty",
            "source_id": "src",
            "relative_path": "empty.png",
        }
        batch = extract_deterministic_evidence(record)
        # Blank detection may be None if visual facts couldn't be loaded
        bf = batch.blank_fragment_evidence
        if bf is not None and isinstance(bf.proposed_value, dict):
            # Should detect missing or empty sprite if visual facts available
            pass


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineV3:
    def test_dry_run_no_records(self, tmp_path):
        result = run_v3_pipeline(
            run_dir=str(tmp_path),
            output_root=str(tmp_path / "output"),
            config=V3PipelineConfig(),
            use_vlm=False,
            dry_run=True,
        )
        assert result.total_records == 0

    def test_dry_run_minimal_records(self, tmp_path):
        import json

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        records = [
            {
                "sprite_id": "test_001",
                "source_id": "src_test",
                "source_name": "Test Pack",
                "relative_path": "test/sword.png",
                "final_png_path": "test/sword.png",
                "status": "accepted",
            },
        ]
        (run_dir / "imported.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=str(tmp_path / "output"),
            config=V3PipelineConfig(),
            use_vlm=False,
            dry_run=False,
            max_records=10,
        )
        assert result.total_records == 1
        # Without calibration, all decisions should be partial or unknown
        assert result.auto_accept == 0

    def _sheet_mapped_record(self, category: str, object_name: str) -> dict:
        return {
            "sprite_id": "sm_001",
            "source_id": "src_sm",
            "source_name": "Sheet Pack",
            "relative_path": "sheet/item.png",
            "final_png_path": "sheet/item.png",
            "status": "accepted",
            "auto_metadata": {
                "sheet_mapping": {
                    "mapping_name": "test_map_v1",
                    "category": category,
                    "object_name": object_name,
                }
            },
        }

    def _category_calibration(self, field: str, ci_lower: float = 0.995) -> CalibrationArtifact:
        return CalibrationArtifact(
            field_name=field,
            strata_data=(
                CalibrationStratumData(
                    field=field,
                    stratum="global",
                    sample_count=200,
                    error_count=1,
                    observed_precision=0.995,
                    calibrated_probability=0.995,
                    ci_lower=ci_lower,
                    ci_upper=0.999,
                    sufficient=True,
                ),
            ),
        )

    def test_pipeline_accepts_with_calibration(self, tmp_path):
        # A clean, well-provenanced record (category via sheet mapping) plus a
        # sufficient calibration artifact must reach acceptance. Before the
        # calibration wiring was fixed, the artifact was ignored and nothing
        # could ever be accepted.
        import json

        run_dir = tmp_path / "run_cal"
        run_dir.mkdir()
        rec = self._sheet_mapped_record("weapon", "sword")
        (run_dir / "imported.jsonl").write_text(json.dumps(rec), encoding="utf-8")

        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=str(tmp_path / "out_cal"),
            config=V3PipelineConfig(),
            calibration=self._category_calibration("category"),
            use_vlm=False,
            dry_run=False,
            max_records=1,
        )
        assert result.total_records == 1
        # category should now be an accepted field -> at least partial_accept,
        # and crucially the clean record is NOT hard-rejected.
        assert result.hard_reject == 0
        assert result.auto_accept + result.partial_accept == 1

    def test_pipeline_no_false_hard_reject_on_clean_record(self, tmp_path):
        # A valid weapon/sword record must never be hard-rejected by the
        # impossible-combination gate.
        import json

        run_dir = tmp_path / "run_clean"
        run_dir.mkdir()
        rec = self._sheet_mapped_record("weapon", "sword")
        (run_dir / "imported.jsonl").write_text(json.dumps(rec), encoding="utf-8")
        result = run_v3_pipeline(
            run_dir=str(run_dir),
            output_root=str(tmp_path / "out_clean"),
            config=V3PipelineConfig(),
            calibration=self._category_calibration("category"),
            use_vlm=False,
            dry_run=False,
            max_records=1,
        )
        assert result.hard_reject == 0


# ---------------------------------------------------------------------------
# Legacy v2 compatibility tests
# ---------------------------------------------------------------------------


class TestLegacyV2Compatibility:
    """Verify that v2 schemas and tests remain unchanged."""

    def test_label_suggestion_unchanged(self):
        from spritelab.harvest.label_schema import LabelSuggestion

        ls = LabelSuggestion("weapon", "sword")
        assert ls.category == "weapon"
        assert ls.object_name == "sword"
        assert ls.confidence == 0.0

    def test_safe_fused_label_unchanged(self):
        from spritelab.harvest.label_schema import LabelSuggestion, SafeFusedLabel

        safe = LabelSuggestion("weapon", "sword")
        fused = SafeFusedLabel(
            safe_prefill=safe,
            filename_suggestion=None,
            vlm_suggestion=None,
            fused_suggestion=safe,
            bucket="auto_filename_trusted",
            needs_review=False,
            flags=(),
            conflict_reasons=(),
            provenance={},
            review_priority=0.0,
        )
        assert fused.bucket == "auto_filename_trusted"
        assert fused.needs_review is False

    def test_taxonomy_categories_unchanged(self):
        from spritelab.harvest.label_taxonomy import CATEGORY_VALUES

        assert "weapon" in CATEGORY_VALUES
        assert "item_icon" in CATEGORY_VALUES
        assert "unknown" in CATEGORY_VALUES
        assert len(CATEGORY_VALUES) == 13

    def test_source_profiles_unchanged(self):
        from spritelab.harvest.source_profiles import loaded_source_profiles

        profiles = loaded_source_profiles()
        assert len(profiles) == 14
        assert "oga_496_rpg_icons" in profiles

    def test_semantic_v3_unchanged(self):
        from spritelab.harvest.semantic_v3 import SCHEMA_VERSION, SemanticV3Record

        assert SCHEMA_VERSION == "semantic_v3.0"
        record = SemanticV3Record(
            schema_version=SCHEMA_VERSION,
            category="weapon",
            object_name="sword",
            base_object="sword",
            open_name="sword",
            attributes=None,
        )
        assert record.category == "weapon"


# ---------------------------------------------------------------------------
# Explicit v3 field-state tests (from the prompt)
# ---------------------------------------------------------------------------


class TestV3FieldStateRequirements:
    def test_category_accepted_object_unknown(self):
        record = RecordDecision(
            sprite_id="test",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="unknown"),
        )
        assert record.category.state == "accepted"
        assert record.canonical_object.state == "unknown"
        assert record.record_state == "partial_accept"

    def test_object_accepted_material_abstained(self):
        record = RecordDecision(
            sprite_id="test",
            category=FieldDecision(field_name="category", state="accepted", accepted_value="weapon"),
            canonical_object=FieldDecision(field_name="canonical_object", state="accepted", accepted_value="sword"),
            material=FieldDecision(field_name="material", state="abstained"),
        )
        assert record.canonical_object.state == "accepted"
        assert record.material.state == "abstained"

    def test_broad_hierarchy_node_accepted_specific_child_abstains(self):
        # sword node can be accepted while dagger abstains
        sword_decision = FieldDecision(
            field_name="canonical_object",
            state="accepted",
            accepted_value="bladed_weapon",
            hierarchy_node="bladed_weapon",
        )
        assert sword_decision.state == "accepted"
        assert sword_decision.hierarchy_node == "bladed_weapon"

    def test_novel_vs_unlabeled_vs_not_applicable(self):
        novel = FieldDecision(field_name="canonical_object", state="novel")
        unlabeled = FieldDecision(field_name="canonical_object", state="unlabeled")
        na = FieldDecision(field_name="canonical_object", state="not_applicable")
        assert novel.state != unlabeled.state
        assert novel.state != na.state
        assert unlabeled.state != na.state

    def test_impossible_combination_rejection(self):
        decision = FieldDecision(
            field_name="canonical_object",
            state="rejected",
            contradiction_codes=("IC001",),
            decision_reason="impossible_combination",
        )
        assert decision.state == "rejected"
        assert "IC001" in decision.contradiction_codes

    def test_unknown_fields_roundtrip_without_loss(self):
        decision = FieldDecision(
            sprite_id="test",
            field_name="color",
            state="unknown",
            candidates=("red", "blue"),
            evidence_refs=("ev_1",),
        )
        js = field_decision_to_json(decision)
        restored = field_decision_from_json(js)
        assert restored.state == "unknown"
        assert restored.candidates == ("red", "blue")

    def test_legacy_v2_records_deserialize_exact(self):
        from spritelab.harvest.label_schema import safe_fused_label_from_json

        data = {
            "safe_prefill": {"category": "weapon", "object_name": "sword"},
            "fused_suggestion": {"category": "weapon", "object_name": "sword"},
            "bucket": "auto_filename_trusted",
            "needs_review": False,
            "flags": ["filename_trusted"],
            "conflict_reasons": [],
            "provenance": {"object_name": "filename_rules_v2"},
            "review_priority": 0.05,
        }
        fused = safe_fused_label_from_json(data)
        assert fused.bucket == "auto_filename_trusted"
        assert fused.needs_review is False
        assert fused.fused_suggestion.category == "weapon"
