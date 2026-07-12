from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from PIL import Image

from spritelab.harvest.label_v4.assisted_v4_gui import gui_field_view
from spritelab.harvest.label_v4.cache import (
    INDEPENDENT_VERIFIER_NAMESPACE,
    CacheCollisionError,
    LabelV4Cache,
    blind_proposal_identity,
    exact_image_content_hash,
    verifier_identity,
)
from spritelab.harvest.label_v4.legacy import (
    adapt_v2_artifact,
    adapt_v3_artifact,
    validate_legacy_isolation,
)
from spritelab.harvest.label_v4.propagation import (
    EXACT_RGBA_DUPLICATE,
    GEOMETRY_FAMILY,
    MATERIAL_VARIANT,
    RECOLOR_VARIANT,
    propagate_fields,
)
from spritelab.harvest.label_v4.review import (
    ABSTAIN,
    ACCEPT_PROPOSAL,
    EDIT,
    MARK_UNSUITABLE_IMAGE,
    MARK_UNSUPPORTED,
    MARK_WRONG_TAXONOMY,
    SELECT_ALTERNATIVE,
    abstain_field,
    accept_proposal,
    compact_review_presenter,
    edit_field,
    immutable_proposal_digest,
    load_review_events,
    mark_unsuitable_image,
    mark_unsupported,
    mark_wrong_taxonomy,
    select_alternative,
)


def _review_record() -> dict:
    return {
        "sprite_id": "iron_buckler",
        "raw_proposal": '{"object_candidates":[{"value":"buckler"}]}',
        "proposal_schema_version": "rich_vlm_proposal_v1",
        "taxonomy_hash": "taxonomy-hash",
        "risk_model_version": "label_risk_v1",
        "field_proposals": {
            "canonical_object": {
                "value": "buckler",
                "alternatives": ["shield"],
                "support": ["filename", "vlm_visual"],
                "conflicts": [],
            },
            "category": {"value": "armor", "alternatives": ["misc_item"], "support": ["filename"]},
            "surface_alias": {"value": "iron buckler", "alternatives": [], "support": ["filename"]},
            "role": {"value": "defensive_equipment", "alternatives": [], "support": ["taxonomy"]},
            "explicit_material": {"value": "iron", "alternatives": [], "support": ["filename"]},
        },
        "label_quality": {
            "record_uncertainty_1_20": 5,
            "critical_field_max_uncertainty": 5,
            "fields": {
                "canonical_object": {
                    "uncertainty_1_20": 3,
                    "uncertainty_band": "strong",
                    "loss_weight": 0.9,
                    "calibration_state": "calibrated",
                },
                "category": {"uncertainty_1_20": 5, "uncertainty_band": "usable_weak"},
                "surface_alias": {"uncertainty_1_20": 4},
                "role": {"uncertainty_1_20": 4},
                "explicit_material": {"uncertainty_1_20": 6},
            },
        },
        "propagation": {"propagation_relation": "none"},
    }


def test_compact_review_presenter_exposes_risk_evidence_and_training_consequence() -> None:
    view = compact_review_presenter(_review_record())
    obj = view["fields"]["canonical_object"]
    assert obj == {
        "proposed_value": "buckler",
        "alternatives": ["shield"],
        "reviewed_value": None,
        "review_state": "unreviewed",
        "uncertainty_1_20": 3,
        "uncertainty_state": "calibrated",
        "risk_band": "strong",
        "evidence_summary": ["filename", "vlm_visual"],
        "conflicts": [],
        "propagation_scope": "none",
        "training_consequence": "full_supervised_weight",
        "loss_weight": 0.9,
    }
    assert view["critical_field_max_uncertainty"] == 5
    assert len(view["raw_proposal_hash"]) == 64


def test_assisted_gui_panel_is_compact_and_details_are_expandable(tmp_path) -> None:
    panel = gui_field_view(_review_record(), "canonical_object", tmp_path / "events.jsonl")
    assert panel["proposed_value"] == "buckler"
    assert panel["alternatives"] == ["shield"]
    assert panel["uncertainty"] == 3
    assert panel["risk_band"] == "strong"
    assert panel["evidence_summary"] == ["filename", "vlm_visual"]
    assert panel["propagation_scope"] == "none"
    assert panel["training_consequence"] == "full_supervised_weight"
    assert "raw_proposal_hash" in panel["details"]


def test_all_review_actions_append_history_without_mutating_raw_proposal(tmp_path) -> None:
    record = _review_record()
    original = copy.deepcopy(record)
    digest = immutable_proposal_digest(record)
    path = tmp_path / "review_events.jsonl"

    accept_proposal(path, record, "canonical_object", reviewer_id="reviewer")
    select_alternative(path, record, "canonical_object", "shield")
    edit_field(path, record, "surface_alias", "small iron buckler")
    abstain_field(path, record, "role")
    mark_unsupported(path, record, "explicit_material")
    mark_wrong_taxonomy(path, record, "category")
    mark_unsuitable_image(path, record, notes="cropped")

    events = load_review_events(path)
    assert [event.action for event in events] == [
        ACCEPT_PROPOSAL,
        SELECT_ALTERNATIVE,
        EDIT,
        ABSTAIN,
        MARK_UNSUPPORTED,
        MARK_WRONG_TAXONOMY,
        MARK_UNSUITABLE_IMAGE,
    ]
    assert len({event.event_id for event in events}) == len(events)
    assert all(event.proposal_hash == digest for event in events)
    assert events[1].alternatives_visible == ("shield",)
    assert events[-1].reviewed_state == "unsuitable_image"
    assert record == original

    view = compact_review_presenter(record, events)
    assert view["fields"]["canonical_object"]["reviewed_value"] == "shield"
    assert view["record_review_state"] == "unsuitable_image"


def test_review_event_append_is_single_record_safe_under_threads(tmp_path) -> None:
    record = _review_record()
    path = tmp_path / "concurrent.jsonl"

    def write(index: int) -> None:
        edit_field(path, record, "surface_alias", f"alias {index}", session_id=str(index))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(24)))
    events = load_review_events(path)
    assert len(events) == 24
    assert len(path.read_text(encoding="utf-8").splitlines()) == 24
    assert {event.reviewed_value for event in events} == {f"alias {index}" for index in range(24)}


def _source_fields(alias: str = "iron buckler") -> dict:
    return {
        "canonical_object": "buckler",
        "category": "armor",
        "domain": "equipment_icon",
        "role": "defensive_equipment",
        "surface_alias": alias,
        "explicit_material": "iron",
        "visual_material_cue": "metallic",
        "silhouette": "round",
        "structure": ["rimmed", "bossed"],
        "primary_colors": ["gray"],
        "description": "Source description must never be copied.",
    }


def _quality() -> dict:
    return {
        name: {"uncertainty_1_20": 3, "risk_upper_95": 0.12, "calibration_state": "calibrated"}
        for name in _source_fields()
    }


def test_exact_duplicate_propagates_semantics_but_regenerates_description() -> None:
    result = propagate_fields(
        _source_fields(),
        {"primary_colors": ["blue"]},
        EXACT_RGBA_DUPLICATE,
        source_id="source",
        source_quality=_quality(),
    )
    assert result.fields["canonical_object"] == "buckler"
    assert result.fields["explicit_material"] == "iron"
    assert result.fields["primary_colors"] == ["gray"]
    assert result.fields["description"] != _source_fields()["description"]
    assert result.propagation_risk_penalty == 0
    assert result.field_quality["canonical_object"]["uncertainty_1_20"] == 3
    assert result.field_quality["canonical_object"]["propagated_from"] == "source"


def test_recolor_propagation_keeps_target_colors_and_increases_field_risk() -> None:
    result = propagate_fields(
        _source_fields(),
        {"primary_colors": ["blue"], "explicit_material": "copper"},
        RECOLOR_VARIANT,
        source_id="gray-source",
        source_quality=_quality(),
    )
    assert result.fields["canonical_object"] == "buckler"
    assert result.fields["structure"] == ["rimmed", "bossed"]
    assert result.fields["primary_colors"] == ["blue"]
    assert result.fields["explicit_material"] == "copper"
    assert result.fields["description"] != _source_fields()["description"]
    assert result.blocked_fields["primary_colors"] == "target_variant_color_must_be_measured"
    assert result.blocked_fields["explicit_material"] == "material_not_safe_across_variant_relation"
    assert result.field_quality["canonical_object"]["uncertainty_1_20"] == 5
    assert result.field_quality["canonical_object"]["risk_upper_95"] == pytest.approx(0.22)


def test_geometry_family_requires_independent_relationship_evidence() -> None:
    blocked = propagate_fields(
        _source_fields(),
        {"canonical_object": "unknown", "description": "target"},
        GEOMETRY_FAMILY,
        source_id="source",
    )
    assert blocked.propagated_fields == ()
    assert blocked.fields["canonical_object"] == "unknown"
    assert blocked.fields["description"] == "target"
    assert set(blocked.blocked_fields.values()) == {"geometry_identity_requires_independent_relationship_evidence"}

    allowed = propagate_fields(
        _source_fields(),
        {"primary_colors": ["green"]},
        GEOMETRY_FAMILY,
        source_id="source",
        declared_relationship=True,
        source_quality=_quality(),
    )
    assert allowed.fields["canonical_object"] == "buckler"
    assert allowed.propagation_risk_penalty == 3


def test_material_variant_does_not_propagate_material_qualified_alias() -> None:
    result = propagate_fields(
        _source_fields(),
        {"surface_alias": "copper buckler", "explicit_material": "copper", "primary_colors": ["orange"]},
        MATERIAL_VARIANT,
        source_id="iron-source",
        source_quality=_quality(),
    )
    assert result.fields["canonical_object"] == "buckler"
    assert result.fields["surface_alias"] == "copper buckler"
    assert result.fields["explicit_material"] == "copper"
    assert result.blocked_fields["surface_alias"] == "material_qualified_alias"
    assert result.blocked_fields["explicit_material"] == "material_not_safe_across_variant_relation"


def test_legacy_adapters_are_explicitly_uncalibrated_and_never_fresh() -> None:
    v3 = {
        "schema_version": "record_decision_v3.1",
        "sprite_id": "agate",
        "policy_hash": "old-policy",
        "model_identity": "old-vlm",
        "prefills": {
            "material": {
                "value": "glass",
                "normalized_value": "glass",
                "raw_candidates": [{"value": "glass", "source": "legacy_v3"}],
                "warnings": ["legacy_v3_uncalibrated_score_unavailable"],
            }
        },
    }
    original = copy.deepcopy(v3)
    adapted = adapt_v3_artifact(v3)
    assert adapted["legacy_source"] is True
    assert adapted["uncalibrated"] is True
    assert adapted["fresh_model_evidence"] is False
    assert adapted["independent_evidence"] is False
    assert adapted["original_policy_hash"] == "old-policy"
    assert adapted["original_model_identity"] == "old-vlm"
    assert adapted["adapted_fields"]["material"]["uncertainty_floor_1_20"] == 13
    assert adapted["adapted_fields"]["material"]["fresh_model_evidence"] is False
    assert v3 == original

    v2 = adapt_v2_artifact({"sprite_id": "old", "object_name": "ring", "category": "jewelry"})
    assert v2["risk_penalty"] > adapted["risk_penalty"]
    assert v2["adapted_fields"]["canonical_object"]["value"] == "ring"
    invalid = dict(adapted, fresh_model_evidence=True)
    with pytest.raises(ValueError, match="fresh_model_evidence"):
        validate_legacy_isolation(invalid)


def _cache_identity(image: Image.Image):
    return blind_proposal_identity(
        stage="blind_visual",
        image=image,
        model_identity="mock-vlm-v1",
        prompt_version="blind-v1",
        prompt="Describe visible evidence only.",
        schema_version="proposal-v1",
        request={"temperature": 0, "image": "content-addressed"},
        provider="mock",
    )


def test_cache_identity_uses_exact_rgba_model_prompt_schema_and_namespace() -> None:
    image = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
    raw_hash = exact_image_content_hash(image.tobytes(), width=3, height=2)
    assert raw_hash == exact_image_content_hash(image)
    changed = image.copy()
    changed.putpixel((0, 0), (11, 20, 30, 255))
    assert exact_image_content_hash(changed) != raw_hash

    identity = _cache_identity(image)
    assert replace(identity, model_identity="mock-vlm-v2").key != identity.key
    assert replace(identity, prompt_hash="0" * 64).key != identity.key
    assert replace(identity, schema_version="proposal-v2").key != identity.key
    verifier = verifier_identity(
        stage="disputed_claims",
        image=image,
        model_identity="mock-vlm-v1",
        prompt_version="verify-v1",
        prompt="Verify only disputed claims.",
        schema_version="verifier-v1",
        request={"claims": ["buckler"]},
    )
    assert verifier.namespace == INDEPENDENT_VERIFIER_NAMESPACE
    assert verifier.key != identity.key


def test_cache_atomic_single_flight_and_immutable_collision(tmp_path) -> None:
    cache = LabelV4Cache(tmp_path / "cache")
    identity = _cache_identity(Image.new("RGBA", (2, 2), (1, 2, 3, 255)))
    count = 0
    guard = threading.Lock()

    def compute() -> dict:
        nonlocal count
        with guard:
            count += 1
        time.sleep(0.03)
        return {"object_candidates": [{"value": "buckler"}]}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: cache.get_or_compute(identity, compute), range(16)))
    assert count == 1
    assert sum(not cache_hit for _, cache_hit in results) == 1
    assert all(value == results[0][0] for value, _ in results)
    entry_path = cache.path_for(identity)
    envelope = json.loads(entry_path.read_text(encoding="utf-8"))
    assert envelope["identity"] == identity.to_dict()
    assert not list(entry_path.parent.glob("*.tmp"))
    with pytest.raises(CacheCollisionError):
        cache.put(identity, {"object_candidates": [{"value": "sword"}]})
