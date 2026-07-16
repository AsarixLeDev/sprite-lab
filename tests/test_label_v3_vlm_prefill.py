from __future__ import annotations

from pathlib import Path

from spritelab.harvest.label_v3.assisted_golden_v3 import V3CorrectionEvent, v3_candidate_summary_for_gui
from spritelab.harvest.label_v3.assisted_v3_gui import (
    V3GUIState,
    _as_list,
    _display_candidate,
    _write_state_json,
    prefill_conflict_explanation,
)
from spritelab.harvest.label_v3.description_enrichment import (
    canonical_description_from_facts,
    enrich_description,
)
from spritelab.harvest.label_v3.evidence import EvidenceItem
from spritelab.harvest.label_v3.field_decisions import FieldDecision
from spritelab.harvest.label_v3.field_prefill import FieldPrefill, build_prefills
from spritelab.harvest.label_v3.pipeline_v3 import _record_png_path
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    record_decision_from_json,
    record_decision_to_json,
)
from spritelab.harvest.label_v3.stage_cache_v3 import stage_cache_key
from spritelab.harvest.label_v3.vlm_orchestration import (
    build_stage_evidence,
    build_stage_prompt,
    normalise_stage_fields,
    parse_stage_output,
    run_vlm_cascade,
)
from spritelab.harvest.label_v3.vlm_runtime import OpenAICompatibleV3Backend, VlmRuntimeConfig, make_text_enricher


class _Backend:
    model_identity = "test-vlm"

    def infer(self, *, stage_id, **_kwargs):
        values = {
            "stage_a_blind_descriptor": {
                "literal_description": "a red metal blade",
                "object_candidates": ["sword"],
                "colors": ["red"],
            },
            "stage_b_morphology": {"silhouette_family": "elongated", "aspect_ratio": 2.0, "complete_object": True},
            "stage_c_constrained_classification": {
                "top_1": "sword",
                "top_3": [{"value": "sword", "raw_score": 0.8}, {"value": "dagger", "raw_score": 0.2}],
                "category": "bladed_weapon",
            },
            "stage_d_open_set_verify": {"verification_result": "supported", "reason": "visible blade"},
            "stage_e_consistency": {"top_1": "sword"},
        }
        return values[stage_id]


def test_blind_prompt_never_contains_candidates_or_source_hints() -> None:
    prompt = build_stage_prompt("stage_a_blind_descriptor", ("sword", "source_filename"))
    assert "sword" not in prompt
    assert "source_filename" not in prompt
    assert "no filename" in prompt


def test_constrained_prompt_has_vocabulary_and_escape() -> None:
    prompt = build_stage_prompt("stage_c_constrained_classification", ("sword", "unknown", "none_of_the_above"))
    assert "sword" in prompt
    assert "none_of_the_above" in prompt


def test_structured_cascade_preserves_top_alternatives() -> None:
    result = run_vlm_cascade("sprite", backend=_Backend(), candidates=("sword", "dagger"), image_hash="image")
    proposed = result.stage_c.evidence.proposed_value
    assert proposed["canonical_object"] == "sword"
    assert proposed["candidate_object_names"] == ["sword", "dagger"]
    assert proposed["field_proposals"]["canonical_object"]["alternatives"]


def test_fast_cascade_uses_only_distinct_high_value_stages() -> None:
    class TrackingBackend(_Backend):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def infer(self, *, stage_id, **kwargs):
            self.calls.append(stage_id)
            return super().infer(stage_id=stage_id, **kwargs)

    backend = TrackingBackend()
    result = run_vlm_cascade(
        "sprite",
        backend=backend,
        candidates=("sword", "dagger"),
        image_hash="image",
        profile="fast",
    )
    assert backend.calls == ["stage_a_blind_descriptor", "stage_c_constrained_classification"]
    assert result.stage_a.available and result.stage_c.available
    assert result.stage_b.failure_reason == "skipped:fast_profile"
    assert result.stage_d.failure_reason == "skipped:fast_profile"
    assert result.stage_e.failure_reason == "skipped:fast_profile"


def test_malformed_output_abstains_and_is_not_a_valid_stage() -> None:
    assert parse_stage_output("stage_a_blind_descriptor", {"unrelated": True}) is None
    assert parse_stage_output("stage_d_open_set_verify", {"verification_result": "maybe"}) is None


def test_verification_contradiction_is_explicitly_marked_for_abstention() -> None:
    evidence = build_stage_evidence(
        "stage_d_open_set_verify", "sprite", "", "", {"verification_result": "contradicted"}
    )
    assert "verification_contradicted" in evidence.warnings
    assert evidence.contradiction_codes == ("vlm_verification_contradicted",)


def test_cache_scope_keeps_blind_descriptor_independent_of_taxonomy() -> None:
    deps = {
        "input_content_hash": "i",
        "stage_version": "v",
        "model_identity": "m",
        "prompt_hash": "p",
        "image_view": "view",
        "preprocessing_hash": "prep",
    }
    assert stage_cache_key("vlm_blind_descriptor", deps) == stage_cache_key("vlm_blind_descriptor", dict(deps))


def test_canonical_description_is_deterministic_and_enrichment_cannot_invent() -> None:
    facts = {"canonical_object": "battle axe", "color": "gold", "material": "metal"}
    assert canonical_description_from_facts(facts) == "A gold metal battle axe."
    artifact = enrich_description(
        {name: {"state": "accepted", "accepted_value": value} for name, value in facts.items()},
        generator=lambda *_: "A magical gold metal battle axe.",
    )
    assert not artifact.valid
    assert artifact.enriched_description == artifact.canonical_description
    assert "magical" in artifact.unsupported_claims_detected


def test_runtime_caps_runpod_workers_and_keeps_secret_out_of_identity() -> None:
    config = VlmRuntimeConfig(
        backend="openai_compatible",
        model="Qwen/Qwen3-VL-32B-Instruct",
        base_url="https://api.runpod.ai/v2/example/openai/v1",
        api_key="secret-never-in-cache-key",
        concurrency=99,
    )
    assert config.concurrency == 5
    assert "secret-never-in-cache-key" not in OpenAICompatibleV3Backend(config, {}).model_identity


def test_text_enrichment_refuses_unknown_backend_without_a_request() -> None:
    assert (
        make_text_enricher(VlmRuntimeConfig(backend="none", enrichment_enabled=True, enrichment_model="small-local"))
        is None
    )


def test_record_png_path_accepts_workspace_relative_harvest_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image = Path("harvest_runs/demo/extracted/item.png")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    assert _record_png_path({"final_png_path": str(image)}, Path("harvest_runs/demo")) == image


def test_color_roles_keep_legacy_flattened_color() -> None:
    result = normalise_stage_fields(
        "stage_a_blind_descriptor",
        {
            "primary_colors": ["red"],
            "secondary_colors": ["orange"],
            "highlight_colors": ["white"],
            "shadow_colors": ["dark red"],
            "outline_color": "black",
        },
    )
    assert result["color"] == "red"
    assert result["primary_colors"] == ["red"]
    assert result["outline_color"] == "black"
    assert result["colors"] == ["red", "orange", "white", "dark red", "black"]


def test_description_prefers_color_then_shape_and_grounded_detail() -> None:
    assert (
        canonical_description_from_facts(
            {
                "canonical_object": "gem",
                "primary_colors": ["red"],
                "shape": ["faceted", "oval"],
                "highlight_colors": ["pink"],
                "outline_color": "black",
            }
        )
        == "A red faceted oval gem with pink highlights and a black outline."
    )


def test_gui_multiselect_serialization_and_conflict_display(tmp_path) -> None:
    assert _as_list(["red", "blue"]) == ["red", "blue"]
    assert _as_list("red, blue") == ["red", "blue"]
    candidate = {
        "fields": {
            "color": {
                "prefill_confidence": 0.91,
                "confidence_kind": "prefill_ranking_score",
                "state": "abstained",
                "decision_reason": "calibration_insufficient",
                "conflicting_sources": ["blind_vlm", "filename"],
            }
        }
    }
    assert "prefill score: `0.91`" in prefill_conflict_explanation(candidate)
    state = V3GUIState(corrections=({"sprite_id": "one"},), index=2)
    output = tmp_path / "state.json"
    _write_state_json(output, state)
    stored = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert stored["labeled_count"] == 1
    assert stored["current_index"] == 2


def test_gem_role_prefill_exposes_alternatives_without_role_decision() -> None:
    evidence = (
        EvidenceItem(
            evidence_id="profile",
            sprite_id="gem",
            pack_id="cc0_gem",
            evidence_family="source_profile",
            proposed_value={"profile_name": "cc0_gem"},
        ),
    )
    prefills, _, _ = build_prefills("gem", {"relative_path": "ruby.png"}, evidence)
    assert prefills["role"].value == "resource"
    assert [a.value for a in prefills["role"].alternatives] == ["crafting_material", "item"]
    assert "role_prefill_not_auto_accepted" in prefills["role"].warnings


def test_v32_prefilled_abstained_record_binds_identically_through_correction_and_resume(tmp_path) -> None:
    fields = {
        "domain": "inventory_icon",
        "category": "gem",
        "canonical_object": "gem",
        "surface_alias": "ruby",
        "color": ["red", "pink", "black"],
        "material": "crystal",
        "shape": ["faceted", "oval"],
        "role": "resource",
        "description": "A red faceted oval gem.",
    }
    decisions = {name: FieldDecision(sprite_id="v32", field_name=name, state="abstained") for name in fields}
    record = RecordDecision(
        sprite_id="v32",
        domain=decisions["domain"],
        category=decisions["category"],
        canonical_object=decisions["canonical_object"],
        surface_alias=decisions["surface_alias"],
        color=decisions["color"],
        material=decisions["material"],
        shape=decisions["shape"],
        role=decisions["role"],
        description=decisions["description"],
        prefills={
            name: FieldPrefill(schema_version="field_prefill_v3.2", sprite_id="v32", field_name=name, value=value)
            for name, value in fields.items()
        },
        prefill_tags=("gem", "ruby", "red", "faceted", "oval", "resource"),
        description_artifact={
            "canonical_description": "A red faceted oval gem.",
            "enriched_description": "A red faceted oval gem.",
        },
    )
    loaded = record_decision_from_json(record_decision_to_json(record))
    correction = V3CorrectionEvent("v32", "color", ["red", "pink", "black"], ["blue", "cyan"], "abstained", "accepted")
    summary = v3_candidate_summary_for_gui(loaded, (correction,))
    assert summary["fields"]["domain"]["value"] == "inventory_icon"
    assert summary["fields"]["surface_alias"]["value"] == "ruby"
    assert summary["fields"]["shape"]["value"] == ["faceted", "oval"]
    assert summary["fields"]["color"]["value"] == ["blue", "cyan"]
    state = V3GUIState(candidates=(summary,), corrections=(correction.to_dict(),), index=0)
    assert _display_candidate(state)["fields"]["color"]["value"] == ["blue", "cyan"]
    path = tmp_path / "resume.json"
    _write_state_json(path, state)
    assert __import__("json").loads(path.read_text(encoding="utf-8"))["current_index"] == 0
