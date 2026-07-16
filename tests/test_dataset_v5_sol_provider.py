from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from spritelab.dataset_v5.blind import BlindInput
from spritelab.dataset_v5.identity import RecordBinding, decoded_rgba_sha256, make_record_id
from spritelab.dataset_v5.labeling import reconcile_sol_passes, reconcile_source_metadata
from spritelab.dataset_v5.sol import (
    SOL_MODEL_UNAVAILABLE,
    SolConfig,
    SolModelUnavailable,
    SolProvider,
)


def _rgba() -> np.ndarray:
    rgba = np.zeros((5, 7, 4), dtype=np.uint8)
    rgba[1:4, 2:6] = [90, 120, 160, 255]
    return rgba


def _blind() -> BlindInput:
    rgba = _rgba()
    binding = RecordBinding(
        source_archive_sha256="1" * 64,
        archive_member_path="source/member.png",
        extraction_operation="decode_member_rgba",
        crop_coordinates=None,
        decoded_rgba_sha256=decoded_rgba_sha256(rgba),
    )
    return BlindInput.from_rgba(make_record_id(binding), rgba)


def _config() -> SolConfig:
    return SolConfig(
        backend="openai_responses_v1",
        model="gpt-5.6-sol-2026-07-01",
        base_url="https://sol.example.test/v1",
        api_key="secret-never-recorded",
    )


def _valid_output() -> dict[str, Any]:
    fields = (
        "category",
        "canonical_object",
        "domain",
        "role",
        "visual_form",
        "material_applicability",
        "explicit_material",
        "color_roles",
        "description",
    )
    return {
        "category": "tool",
        "canonical_object": "hammer",
        "domain": "inventory_icon",
        "role": "functional_tool",
        "visual_form": ["compact head", "straight handle"],
        "material_applicability": "applicable",
        "explicit_material": "iron",
        "color_roles": {
            "primary": ["blue gray"],
            "secondary": [],
            "outline": ["dark gray"],
            "shadow": ["navy"],
            "highlight": ["pale blue"],
        },
        "description": "A compact blue-gray object with a straight handle.",
        "abstentions": {},
        "field_rationales": dict.fromkeys(fields, "Visible shape and pixels support this field."),
        "field_risk_signals": {
            **{field: [] for field in fields},
            "explicit_material": ["visually_justified_exact_material"],
        },
        "field_confidence": dict.fromkeys(fields, 0.7),
        "schema_version": "blind_semantic_output_v1",
    }


def _transport_with_output(raw_output: str):
    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        assert headers["Authorization"].startswith("Bearer ")
        assert timeout == 120.0
        if method == "GET":
            return {
                "status": 200,
                "json": {
                    "id": "gpt-5.6-sol-2026-07-01",
                    "family": "GPT-5.6 Sol",
                    "version": "2026-07-01.1",
                    "provider": "openai",
                    "request_schema_version": "responses-v1",
                },
                "final_url": url,
            }
        assert method == "POST" and url.endswith("/responses") and body is not None
        response = {
            "model": "gpt-5.6-sol-2026-07-01",
            "model_version": "2026-07-01.1",
            "provider": "openai",
            "output_text": raw_output,
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        return {
            "status": 200,
            "final_url": url,
            "json": response,
            "body": json.dumps(response).encode(),
        }

    return transport


def test_sol_environment_is_explicit_and_missing_configuration_stops_exactly() -> None:
    with pytest.raises(SolModelUnavailable) as exc:
        SolConfig.from_env({})
    assert str(exc.value) == SOL_MODEL_UNAVAILABLE


@pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "qwen3-vl", "ollama", "gpt-5.6"])
def test_sol_exact_model_name_enforcement_and_no_fallback(model: str) -> None:
    called = False

    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {}

    config = SolConfig(
        backend="openai_responses_v1",
        model=model,
        base_url="https://provider.example/v1",
        api_key="secret",
    )
    with pytest.raises(SolModelUnavailable) as exc:
        SolProvider(config, transport=transport)
    assert str(exc.value) == SOL_MODEL_UNAVAILABLE
    assert called is False


def test_sol_public_identity_never_contains_api_key() -> None:
    public = _config().public_identity()
    assert "secret" not in json.dumps(public)
    assert "api_key" not in public


def test_sol_preflight_requires_attested_family_version_provider_and_schema() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"status": 200, "json": {"id": "gpt-5.6-sol-2026-07-01"}}

    provider = SolProvider(_config(), transport=transport)
    with pytest.raises(SolModelUnavailable) as exc:
        provider.preflight()
    assert str(exc.value) == SOL_MODEL_UNAVAILABLE


def test_sol_success_records_exact_provider_identity_and_usage() -> None:
    output = _valid_output()
    provider = SolProvider(_config(), transport=_transport_with_output(json.dumps(output)))
    artifact = provider.call_blind_pass(_blind(), request_id="req_01", pass_kind="adjudication")
    assert artifact["authoritative"] is True
    assert artifact["model_identifier"] == "gpt-5.6-sol-2026-07-01"
    assert artifact["model_version"] == "2026-07-01.1"
    assert artifact["provider"] == "openai"
    assert artifact["token_usage"]["total_tokens"] == 150
    assert "secret" not in json.dumps(artifact)


def test_sol_fenced_json_is_rejected_without_silent_repair() -> None:
    raw = "```json\n" + json.dumps(_valid_output()) + "\n```"
    provider = SolProvider(_config(), transport=_transport_with_output(raw))
    artifact = provider.call_blind_pass(_blind(), request_id="req_bad", pass_kind="adjudication")
    assert artifact["authoritative"] is False
    assert artifact["error_code"] == "strict_response_rejected"
    assert "```json" in artifact["raw_provider_response"]


def test_critical_disagreement_quarantines_field_and_sol_labels_stay_weak() -> None:
    first = _valid_output()
    second = _valid_output()
    second["canonical_object"] = "pickaxe"
    reconciled = reconcile_sol_passes(first, second, deterministic_facts={"palette_size": 3})
    canonical = reconciled["fields"]["canonical_object"]
    assert canonical["state"] == "conflicted"
    assert canonical["supervision_class"] == "auxiliary_only"
    assert reconciled["inclusion_decision"] == "quarantine"
    assert reconciled["fields"]["category"]["supervision_class"] == "supervised_weak"
    assert reconciled["deterministic_fields"]["palette_size"]["supervision_class"] == "supervised_strong"


def test_abstention_is_preserved_and_never_becomes_negative_target() -> None:
    first = _valid_output()
    second = _valid_output()
    for value in (first, second):
        value["canonical_object"] = None
        value["abstentions"]["canonical_object"] = "Ambiguous silhouette."
    reconciled = reconcile_sol_passes(first, second, deterministic_facts={})
    field = reconciled["fields"]["canonical_object"]
    assert field["state"] == "abstained"
    assert field["supervision_class"] == "unlabeled"
    assert field["negative_target"] is False


def test_unsupported_exact_material_is_removed() -> None:
    first = _valid_output()
    second = _valid_output()
    first["field_risk_signals"]["explicit_material"] = []
    second["field_risk_signals"]["explicit_material"] = []
    reconciled = reconcile_sol_passes(first, second, deterministic_facts={})
    material = reconciled["fields"]["explicit_material"]
    assert material["value"] is None
    assert material["state"] == "unsupported_removed"
    assert material["supervision_class"] == "auxiliary_only"


def test_source_metadata_is_introduced_only_after_blind_freeze() -> None:
    reconciled = reconcile_sol_passes(_valid_output(), _valid_output(), deterministic_facts={})
    metadata = {
        "original_source_filename": "misleading_helmet.png",
        "declared_semantics": {"canonical_object": "helmet"},
    }
    with pytest.raises(ValueError, match="before blind labeling"):
        reconcile_source_metadata(reconciled, metadata, blind_labels_frozen=False)
    result = reconcile_source_metadata(reconciled, metadata, blind_labels_frozen=True)
    assert result["blind_label_unchanged"] is True
    assert result["filename_taint_status"] == "tainted_metadata"
    assert result["metadata_conflicts"]
