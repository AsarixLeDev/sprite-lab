from __future__ import annotations

import json

from spritelab.harvest.label_v3.assisted_golden_v3 import (
    V3CorrectionEvent,
    apply_v3_corrections_to_record,
    v3_candidate_summary_for_gui,
)
from spritelab.harvest.label_v3.assisted_v3_gui import V3GUIState, _view_outputs
from spritelab.harvest.label_v3.description_enrichment import (
    enrich_description,
    validate_enriched_description,
)
from spritelab.harvest.label_v3.evidence import EvidenceItem
from spritelab.harvest.label_v3.field_decisions import FieldDecision
from spritelab.harvest.label_v3.field_prefill import (
    SCHEMA_VERSION,
    build_prefills,
    filename_semantics,
    normalize_colors,
    normalize_shape,
)
from spritelab.harvest.label_v3.pack_context import analyze_pack_context
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    record_decision_from_json,
    record_decision_to_json,
)
from spritelab.harvest.label_v3.taxonomy_v3 import deepest_supported_node
from spritelab.harvest.label_v3.vlm_runtime import (
    OpenAICompatibleV3Backend,
    VlmRuntimeConfig,
    make_text_enricher,
)


def _evidence(family: str, proposed: dict, *, stage: str = "deterministic", targets=()) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{family}-{stage}",
        sprite_id="s",
        evidence_family=family,
        producer_stage=stage,
        target_fields=tuple(targets),
        proposed_value=proposed,
        dependency_group=family,
        deterministic=stage == "deterministic",
    )


def _gem_prefills(path: str, extra=()):
    record = {"sprite_id": "s", "relative_path": path}
    evidence = (
        _evidence(
            "source_profile",
            {"profile_name": "cc0_gem", "domain": "gem"},
            targets=("domain", "category", "canonical_object"),
        ),
        *extra,
    )
    return build_prefills("s", record, evidence)


def test_strong_prefill_without_calibration_is_separate_from_decision():
    prefills, tags, _ = _gem_prefills("32x32/jade.png")
    record = RecordDecision(
        sprite_id="s",
        canonical_object=FieldDecision(sprite_id="s", field_name="canonical_object", state="abstained"),
        prefills=prefills,
        prefill_tags=tags,
    )
    payload = record_decision_to_json(record)
    assert payload["canonical_object"]["accepted_value"] is None
    assert payload["prefills"]["canonical_object"]["value"] == "gem"
    assert payload["prefills"]["canonical_object"]["confidence_kind"] == "prefill_ranking_score"


def test_filename_alias_and_misspelling_normalization():
    assert filename_semantics({"relative_path": "x/agate.png"})["surface_alias"] == "agate"
    parsed = filename_semantics({"relative_path": "x/amethist.png"})
    assert parsed["original_token"] == "amethist"
    assert parsed["surface_alias"] == "amethyst"


def test_generic_filename_abstains():
    for name in ("tile_0012.png", "sprite_47.png", "asset_a.png"):
        assert filename_semantics({"relative_path": name})["generic"] is True


def test_pack_homogeneity_reduces_for_heterogeneous_pack():
    homogeneous = analyze_pack_context(
        [{"relative_path": f"{v}.png", "width": 32, "height": 32} for v in ("jade", "agate", "ruby", "diamond")]
    )
    heterogeneous = analyze_pack_context(
        [{"relative_path": f"{v}.png", "width": 32, "height": 32} for v in ("jade", "sword", "apple", "tile_1")]
    )
    assert homogeneous.pack_homogeneity_score > heterogeneous.pack_homogeneity_score
    assert homogeneous.pack_prior_strength > heterogeneous.pack_prior_strength


def test_jade_is_gem_and_egg_is_shape_not_object():
    blind = _evidence(
        "blind_vlm_descriptor",
        {"canonical_object": "egg", "shape": "egg-shaped oval"},
        stage="vlm_stage_a_blind_descriptor",
        targets=("canonical_object", "shape"),
    )
    prefills, _, metadata = _gem_prefills("32x32/jade.png", (blind,))
    assert prefills["canonical_object"].value == "gem"
    assert "oval" in prefills["shape"].value
    assert "egg" in metadata["raw_morphology"]


def test_crystal_cluster_has_taxonomy_fallback_before_novel():
    node = deepest_supported_node("crystal cluster")
    assert node is not None
    assert node.name == "crystal_cluster"


def test_domain_category_and_style_are_separate():
    visual = _evidence(
        "blind_vlm_descriptor",
        {"domain": "pixel_art", "category": "egg", "canonical_object": "egg"},
        stage="vlm_stage_a_blind_descriptor",
        targets=("domain", "category", "canonical_object"),
    )
    prefills, _, metadata = _gem_prefills("jade.png", (visual,))
    assert prefills["domain"].value == "inventory_icon"
    assert prefills["category"].value == "gem"
    assert "pixel_art" in metadata["style_tags"]


def test_shape_prose_normalization_and_raw_morphology():
    tags, raw, actions = normalize_shape("triangular, conical, pointed top, rounded base, pixelated edges")
    assert tags == ("triangular", "pointed", "rounded_base")
    assert raw.startswith("triangular")
    assert actions


def test_material_cue_is_conservative():
    cue = _evidence(
        "blind_vlm_descriptor",
        {"material": "glass"},
        stage="vlm_stage_a_blind_descriptor",
        targets=("material",),
    )
    prefills, _, _ = _gem_prefills("jade.png", (cue,))
    assert prefills["material"].confidence < 0.55
    assert "material_cue_not_fact" in prefills["material"].warnings


def test_normalized_multi_color_and_deterministic_tags():
    palette = _evidence("color_palette", {"dominant_colors": ["teal-green", "black", "cyan"]}, targets=("color",))
    prefills, tags, _ = _gem_prefills("jade.png", (palette,))
    assert normalize_colors(["teal-green", "black"]) == ("teal", "black")
    assert prefills["color"].value == ["teal", "black", "cyan"]
    assert {"gem", "jade", "teal", "inventory_icon"} <= set(tags)


def test_description_semantic_synonyms_and_unsupported_claims():
    facts = {"canonical_object": "gem", "color": "pink", "shape": ["round"], "style": ["pixel_art", "outlined"]}
    assert validate_enriched_description("A rounded pink pixel-art gem with a dark outline.", facts) == (True, ())
    valid, unsupported = validate_enriched_description("A legendary enchanted pink gem.", facts)
    assert valid is False
    assert {"legendary", "enchanted"} <= set(unsupported)


def test_enrichment_failure_falls_back_and_is_not_evidence():
    artifact = enrich_description(
        {"canonical_object": "gem", "color": "pink"},
        generator=lambda _facts, _literal: "An ancient magical healing gem.",
    )
    assert artifact.enriched_description == artifact.canonical_description
    assert artifact.dependency_group == "derived_description"
    assert artifact.valid is False


def test_gui_shows_prefill_while_decision_abstains_and_one_click_event_persists():
    prefills, tags, _ = _gem_prefills("jade.png")
    record = RecordDecision(
        sprite_id="s",
        canonical_object=FieldDecision(sprite_id="s", field_name="canonical_object", state="abstained"),
        prefills=prefills,
        prefill_tags=tags,
    )
    summary = v3_candidate_summary_for_gui(record)
    assert summary["fields"]["canonical_object"]["value"] == "gem"
    assert summary["fields"]["canonical_object"]["state"] == "abstained"
    event = V3CorrectionEvent(
        "s", "canonical_object", "gem", "gem", "abstained", "accepted", review_action="accepted_as_prefilled"
    )
    corrected = apply_v3_corrections_to_record(record, [event])
    assert corrected.canonical_object.accepted_value == "gem"
    assert event.to_dict()["review_action"] == "accepted_as_prefilled"
    outputs = _view_outputs(V3GUIState(candidates=(summary,)), lambda _candidate: None)
    assert len(outputs) == 27
    assert outputs[9] == "gem"


def test_old_v3_record_adapter_moves_abstained_value_to_prefill():
    old = {
        "schema_version": "record_decision_v3.1",
        "sprite_id": "old",
        "record_state": "unknown",
        "canonical_object": {
            "schema_version": "field_decision_v3.1",
            "sprite_id": "old",
            "field": "canonical_object",
            "accepted_value": "gem",
            "state": "abstained",
            "candidates": ["gem"],
        },
    }
    loaded = record_decision_from_json(old)
    assert loaded.canonical_object.accepted_value is None
    assert loaded.prefills["canonical_object"].value == "gem"
    assert loaded.prefills["canonical_object"].schema_version == SCHEMA_VERSION


class _MockHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_mocked_openai_compatible_runpod_vlm_path(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data)
        return _MockHTTPResponse({"choices": [{"message": {"content": json.dumps({"visible_colors": ["teal"]})}}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    config = VlmRuntimeConfig(
        backend="openai_compatible",
        model="Qwen/Qwen3-VL-32B-Instruct",
        base_url="https://api.runpod.ai/v2/mock/openai/v1",
        api_key="not-sent",
        retries=0,
    )
    view = "data:image/png;base64,AA=="
    backend = OpenAICompatibleV3Backend(
        config, {"checkerboard": view, "nearest_neighbor": view, "tight_foreground_crop": view}
    )
    result = backend.infer(stage_id="stage_a_blind_descriptor", image_ref=view, prompt="facts", prompt_hash="h")
    assert result == {"visible_colors": ["teal"]}
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["body"]["max_tokens"] == 320
    assert len(seen["body"]["messages"][0]["content"]) == 4  # prompt + three views


def test_mocked_ollama_enrichment_path(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _MockHTTPResponse({"message": {"content": json.dumps({"enriched_description": "A rounded pink gem."})}})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    enricher = make_text_enricher(
        VlmRuntimeConfig(
            backend="ollama",
            base_url="http://127.0.0.1:11434",
            enrichment_enabled=True,
            enrichment_model="qwen3:4b",
            retries=0,
        )
    )
    assert enricher is not None
    assert enricher({"canonical_object": "gem", "color": "pink"}, "") == "A rounded pink gem."
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["model"] == "qwen3:4b"
    assert seen["body"]["think"] is False
    assert seen["body"]["options"]["num_predict"] == 96
