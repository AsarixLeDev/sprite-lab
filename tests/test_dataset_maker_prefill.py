from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.dataset_maker import prefill
from spritelab.dataset_maker.model import DatasetMakerItem
from spritelab.dataset_maker.prefill import (
    CachedPrefillBackend,
    MetadataPrefillBackend,
    MetadataSuggestion,
    NoopPrefillBackend,
    OllamaQwenPrefillBackend,
    OpenAICompatibleQwenPrefillBackend,
    PrefillConfig,
    PrefillRequest,
    RuleBasedPrefillBackend,
    apply_suggestion_to_item,
    compute_image_cache_key,
    create_prefill_backend,
    build_qwen_prefill_prompt,
    image_to_data_url,
    parse_metadata_suggestion,
    prepare_vlm_image,
)
from spritelab.dataset_maker.gui import _prefill_blocked_warning, _prefill_report


def test_prepare_vlm_image_upscales_with_nearest_neighbor() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 255))

    prepared = prepare_vlm_image(image, upscale=2, crop_to_content=False)

    assert prepared.size == (64, 64)
    assert prepared.getpixel((0, 0)) == (255, 0, 0)
    assert prepared.getpixel((1, 1)) == (255, 0, 0)
    assert prepared.getpixel((2, 0)) == (0, 0, 255)


def test_prepare_vlm_image_returns_larger_rgb_or_rgba_image() -> None:
    prepared = prepare_vlm_image(_image(), upscale=16, crop_to_content=False)

    assert prepared.size == (512, 512)
    assert prepared.mode in {"RGB", "RGBA"}


def test_prepare_vlm_image_crops_and_scales_small_content() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(12, 20):
        for x in range(12, 20):
            image.putpixel((x, y), (255, 0, 0, 255))

    prepared = prepare_vlm_image(image, upscale=16)

    # 8x8 content + 1px pad = 10x10 crop, scaled toward 512 (51x per side).
    assert max(prepared.size) >= 500
    assert prepared.size[0] == prepared.size[1]
    # Center is sprite content, corner is the magenta display background.
    center = prepared.getpixel((prepared.width // 2, prepared.height // 2))
    assert center == (255, 0, 0)
    assert prepared.getpixel((0, 0)) == (255, 0, 255)


def test_prepare_vlm_image_fully_transparent_uses_background() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

    prepared = prepare_vlm_image(image, upscale=16)

    assert prepared.getpixel((0, 0)) == (255, 0, 255)


def test_content_bbox_pads_and_clamps() -> None:
    from spritelab.dataset_maker.prefill import content_bbox

    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((5, 7), (255, 0, 0, 255))

    assert content_bbox(image) == (0, 0, 6, 8)
    assert content_bbox(image, pad=2) == (0, 0, 8, 10)
    assert content_bbox(Image.new("RGBA", (32, 32), (0, 0, 0, 0))) == (0, 0, 32, 32)


def test_image_to_data_url_returns_png_data_url() -> None:
    data_url = image_to_data_url(_image())

    assert data_url.startswith("data:image/png;base64,")


def test_parse_metadata_suggestion_parses_strict_json() -> None:
    suggestion = parse_metadata_suggestion(
        json.dumps(
            {
                "category": "item_icon",
                "object_name": "mushroom",
                "tags": ["purple", "glowing"],
                "confidence": 0.72,
            }
        )
    )

    assert suggestion.category == "item_icon"
    assert suggestion.object_name == "mushroom"
    assert suggestion.tags == ("purple", "glowing")
    assert suggestion.confidence == 0.72


def test_parse_metadata_suggestion_parses_contextual_fields() -> None:
    suggestion = parse_metadata_suggestion(
        json.dumps(
            {
                "category": "item_icon",
                "filename_agreement": "Partial",
                "visual_evidence": ["yellow crescent"],
                "disagreement_reason": "Filename says apple.",
            }
        )
    )

    assert suggestion.filename_agreement == "partial"
    assert suggestion.visual_evidence == ("yellow_crescent",)
    assert suggestion.disagreement_reason == "Filename says apple."


def test_qwen_prompt_includes_filename_hint_and_forbids_provenance_fields() -> None:
    prompt = build_qwen_prefill_prompt(
        {
            "category": "item_icon",
            "object_name": "banana",
            "tags": ["banana", "fruit"],
            "confidence": 0.98,
        }
    )

    assert "filename parser suggested" in prompt
    assert '"object_name": "banana"' in prompt
    assert "visual_evidence" in prompt
    assert "Do not include license, author, source, train split, or accept/reject status" in prompt


def test_qwen_prompt_contains_no_concrete_example_values() -> None:
    # A filled-in example JSON was regurgitated verbatim by the model.
    prompt = build_qwen_prefill_prompt()
    for leaked in ("purple_glowing_mushroom", "psychedelic", "0.72", "Object is ambiguous at 32x32"):
        assert leaked not in prompt
    assert "<one allowed category value>" in prompt
    assert "cannot_tell" in prompt


def test_qwen_prompt_includes_image_facts() -> None:
    prompt = build_qwen_prefill_prompt(
        image_facts={
            "content_width": 8,
            "content_height": 10,
            "opaque_palette_size": 5,
            "dominant_colors": ["dark_green", "brown"],
        }
    )
    assert "8x10 pixels" in prompt
    assert "5 opaque colors" in prompt
    assert "dark_green, brown" in prompt
    assert "magenta background" in prompt


def test_parser_tolerates_fenced_json_block() -> None:
    suggestion = parse_metadata_suggestion(
        """```json
{"category": "plant", "tags": ["Mushroom"]}
```"""
    )

    assert suggestion.category == "plant"
    assert suggestion.tags == ("mushroom",)


def test_parser_rejects_invalid_json_with_warning() -> None:
    suggestion = parse_metadata_suggestion("this is not json")

    assert suggestion.category == "unknown"
    assert any("invalid JSON" in warning for warning in suggestion.warnings)


def test_parser_normalizes_category() -> None:
    suggestion = parse_metadata_suggestion('{"category": "Item Icon"}')

    assert suggestion.category == "item_icon"


def test_parser_normalizes_tags_materials_mood_and_colors() -> None:
    suggestion = parse_metadata_suggestion(
        json.dumps(
            {
                "tags": ["Purple Glow", "purple_glow"],
                "materials": ["Dark Metal"],
                "mood": ["Mystical"],
                "dominant_colors": ["Dark Blue"],
            }
        )
    )

    assert suggestion.tags == ("purple_glow",)
    assert suggestion.materials == ("dark_metal",)
    assert suggestion.mood == ("mystical",)
    assert suggestion.dominant_colors == ("dark_blue",)


def test_parser_clamps_confidence() -> None:
    high = parse_metadata_suggestion('{"confidence": 9.0}')
    low = parse_metadata_suggestion('{"confidence": -1.0}')

    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_unknown_category_becomes_unknown_with_warning() -> None:
    suggestion = parse_metadata_suggestion('{"category": "spaceship"}')

    assert suggestion.category == "unknown"
    assert any("Unknown category" in warning for warning in suggestion.warnings)


def test_unsafe_suggested_sprite_id_is_normalized_or_discarded() -> None:
    normalized = parse_metadata_suggestion('{"suggested_sprite_id": "Purple Mushroom!!"}')
    discarded = parse_metadata_suggestion('{"suggested_sprite_id": "!!!"}')

    assert normalized.suggested_sprite_id == "purple_mushroom"
    assert discarded.suggested_sprite_id == ""
    assert any("sprite_id" in warning for warning in discarded.warnings)


def test_apply_suggestion_to_item_fills_unknown_category() -> None:
    item = _item(category="unknown")
    suggestion = MetadataSuggestion(category="item_icon")

    updated = apply_suggestion_to_item(item, suggestion)

    assert updated.category == "item_icon"


def test_apply_suggestion_does_not_overwrite_existing_category_by_default() -> None:
    item = _item(category="block")
    suggestion = MetadataSuggestion(category="item_icon")

    updated = apply_suggestion_to_item(item, suggestion)

    assert updated.category == "block"


def test_apply_suggestion_overwrite_mode_overwrites_category() -> None:
    item = _item(category="block")
    suggestion = MetadataSuggestion(category="item_icon")

    updated = apply_suggestion_to_item(item, suggestion, overwrite_existing=True)

    assert updated.category == "item_icon"


def test_apply_suggestion_merges_and_deduplicates_tags() -> None:
    item = _item(tags=("purple", "old"))
    suggestion = MetadataSuggestion(
        object_name="Purple Mushroom",
        tags=("purple", "glowing"),
        materials=("crystal",),
        mood=("mystical",),
        dominant_colors=("purple", "cyan"),
    )

    updated = apply_suggestion_to_item(item, suggestion)

    assert updated.tags == ("purple", "old", "glowing", "purple_mushroom", "crystal", "mystical", "cyan")


def test_apply_suggestion_never_modifies_license_author_source_status_or_split() -> None:
    item = _item(status="needs_fix", split="val")
    suggestion = MetadataSuggestion(
        category="item_icon",
        suggested_sprite_id="new_id",
        short_description="A small icon.",
    )

    updated = apply_suggestion_to_item(item, suggestion, overwrite_existing=True)

    assert updated.license == item.license
    assert updated.author == item.author
    assert updated.source_path == item.source_path
    assert updated.source_name == item.source_name
    assert updated.status == "needs_fix"
    assert updated.split == "val"


def test_cache_key_is_stable_for_same_image_model_and_prompt() -> None:
    left = compute_image_cache_key(_image(), model="qwen", prompt_version="v1")
    right = compute_image_cache_key(_image(), model="qwen", prompt_version="v1")

    assert left == right


def test_cached_backend_writes_cache_file(tmp_path: Path) -> None:
    backend = _CountingBackend()
    cached = CachedPrefillBackend(backend, tmp_path)

    cached.suggest(_request())

    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cached_backend_reads_cache_file(tmp_path: Path) -> None:
    backend = _CountingBackend()
    cached = CachedPrefillBackend(backend, tmp_path)

    cached.suggest(_request())
    cached.suggest(_request())

    assert backend.count == 1


def test_cached_backend_does_not_cache_warning_only_failures(tmp_path: Path) -> None:
    backend = _WarningOnlyBackend()
    cached = CachedPrefillBackend(backend, tmp_path)

    cached.suggest(_request())
    cached.suggest(_request())

    assert backend.count == 2
    assert list(tmp_path.glob("*.json")) == []


def test_gui_prefill_blocked_warning_explains_disabled_state() -> None:
    message = _prefill_blocked_warning(PrefillConfig(enabled=False, backend="openai_compatible"))

    assert message is not None
    assert "Auto-fill is disabled" in message


def test_gui_prefill_report_includes_backend_counts_and_warnings() -> None:
    report = _prefill_report(
        [("sprite_a", MetadataSuggestion(warnings=("connection failed",)))],
        attempted=1,
        applied=0,
        config=PrefillConfig(enabled=True, backend="openai_compatible", model="qwen", base_url="http://local/v1"),
        selected_ids=("sprite_a",),
    )

    assert "## Prefill report" in report
    assert "Backend: `openai_compatible`" in report
    assert "Attempted requests: 1" in report
    assert "`sprite_a`: connection failed" in report


def test_openai_compatible_backend_sends_expected_request_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"category": "item_icon", "tags": ["mushroom"]}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(
        PrefillConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="secret",
            timeout_seconds=12,
        )
    )

    backend.suggest(_request())

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert captured["timeout"] == 12
    assert payload["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_compatible_backend_sends_filename_hint_in_prompt(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "item_icon", "confidence": 0.8})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, include_filename_hint=True))

    backend.suggest(
        PrefillRequest(
            sprite_id="sprite",
            image=_image(),
            filename_suggestion={"category": "item_icon", "object_name": "banana"},
        )
    )

    prompt = captured["payload"]["messages"][0]["content"][0]["text"]
    assert "filename parser suggested" in prompt
    assert '"object_name": "banana"' in prompt
    assert "license" in prompt


def test_openai_compatible_backend_retries_invalid_json_then_success(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse({"choices": [{"message": {"content": "not json"}}]})
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "plant", "confidence": 0.85})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, retry_attempts=2))

    suggestion = backend.suggest(_request())

    assert calls["count"] == 2
    assert suggestion.category == "plant"
    assert any("Retried 1" in warning for warning in suggestion.warnings)


def test_openai_compatible_backend_retries_low_confidence(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls["count"] += 1
        confidence = 0.2 if calls["count"] == 1 else 0.9
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "item_icon", "confidence": confidence})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(
        PrefillConfig(enabled=True, retry_attempts=2, min_qwen_confidence=0.55)
    )

    suggestion = backend.suggest(_request())

    assert calls["count"] == 2
    assert suggestion.confidence == 0.9


def test_openai_compatible_backend_retries_warning_only_response(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse({"choices": [{"message": {"content": ""}}]})
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "item_icon", "confidence": 0.82})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, retry_attempts=2))

    suggestion = backend.suggest(_request())

    assert calls["count"] == 2
    assert suggestion.category == "item_icon"


def test_openai_compatible_backend_sends_json_schema_response_format(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "plant", "object_name": "fern", "uncertainty": "confident"})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True))

    backend.suggest(_request())

    response_format = captured["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert "unknown" in schema["properties"]["category"]["enum"]
    assert "uncertainty" in schema["required"]


def test_openai_compatible_backend_structured_output_off(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps({"category": "plant"})}}]})

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, structured_output="off"))

    backend.suggest(_request())

    assert "response_format" not in captured["payload"]


def test_openai_compatible_backend_falls_back_when_schema_rejected(monkeypatch) -> None:
    import urllib.error

    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, timeout: float):
        payload = json.loads(request.data.decode("utf-8"))
        payloads.append(payload)
        if "response_format" in payload:
            raise urllib.error.HTTPError(
                request.full_url, 400, "Bad Request", None, _FakeErrorBody(b'{"error": "response_format is not supported"}')
            )
        return _FakeResponse({"choices": [{"message": {"content": json.dumps({"category": "plant", "object_name": "fern"})}}]})

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, structured_output="auto"))

    suggestion = backend.suggest(_request())

    assert suggestion.category == "plant"
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    # The unsupported flag latches: the next request skips the schema immediately.
    backend.suggest(_request())
    assert "response_format" not in payloads[2]


def test_parse_metadata_suggestion_maps_uncertainty_to_confidence() -> None:
    suggestion = parse_metadata_suggestion(
        json.dumps({"category": "plant", "object_name": "fern", "uncertainty": "likely"})
    )
    assert suggestion.uncertainty == "likely"
    assert suggestion.confidence == 0.7

    explicit = parse_metadata_suggestion(
        json.dumps({"category": "plant", "uncertainty": "likely", "confidence": 0.42})
    )
    assert explicit.confidence == 0.42

    invalid = parse_metadata_suggestion(json.dumps({"category": "plant", "uncertainty": "definitely"}))
    assert invalid.uncertainty == "unsure"
    assert any("Unknown uncertainty" in warning for warning in invalid.warnings)


def test_degenerate_detection_background_mentions() -> None:
    from spritelab.dataset_maker.prefill import flag_degenerate_suggestion, is_degenerate_suggestion

    checker = MetadataSuggestion(
        category="ui_icon",
        object_name="checkerboard",
        short_description="A gray checkerboard pattern.",
        confidence=0.6,
    )
    assert is_degenerate_suggestion(checker)
    flagged = flag_degenerate_suggestion(checker)
    assert any(warning.startswith("degenerate_response") for warning in flagged.warnings)
    # Flagging twice does not duplicate the warning.
    assert flag_degenerate_suggestion(flagged).warnings == flagged.warnings


def test_degenerate_detection_confident_unknown() -> None:
    from spritelab.dataset_maker.prefill import is_degenerate_suggestion

    assert is_degenerate_suggestion(MetadataSuggestion(category="unknown", confidence=0.95))
    # cannot_tell legitimately pairs with unknown.
    assert not is_degenerate_suggestion(
        MetadataSuggestion(category="unknown", confidence=0.9, uncertainty="cannot_tell")
    )
    assert not is_degenerate_suggestion(
        MetadataSuggestion(category="plant", object_name="fern", confidence=0.9)
    )
    assert is_degenerate_suggestion(
        MetadataSuggestion(category="plant", object_name="unknown_object", confidence=0.9)
    )


def test_degenerate_response_is_retried_and_not_cached(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] < 3:
            content = {
                "category": "ui_icon",
                "object_name": "checkerboard",
                "short_description": "A checkered background pattern.",
                "uncertainty": "confident",
            }
        else:
            content = {"category": "plant", "object_name": "fern", "uncertainty": "confident"}
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(content)}}]})

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, retry_attempts=2))

    suggestion = backend.suggest(_request())
    assert calls["count"] == 3
    assert suggestion.category == "plant"

    # A backend that always answers degenerately must not populate the cache.
    calls["count"] = 0

    def always_degenerate(request: Any, timeout: float) -> _FakeResponse:
        content = {"category": "ui_icon", "object_name": "checkerboard", "uncertainty": "confident"}
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(content)}}]})

    monkeypatch.setattr(prefill.urllib.request, "urlopen", always_degenerate)
    cached_backend = CachedPrefillBackend(
        OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, retry_attempts=0)),
        tmp_path / "cache",
    )
    result = cached_backend.suggest(_request())
    assert any(warning.startswith("degenerate_response") for warning in result.warnings)
    assert not list((tmp_path / "cache").glob("*.json"))


def test_color_names_deterministic() -> None:
    from spritelab.codec.color_names import color_name

    assert color_name((0, 0, 0)) == "black"
    assert color_name((255, 255, 255)) == "white"
    assert color_name((250, 30, 30)) == "red"
    assert color_name((35, 100, 45)) == "dark_green"
    assert color_name((255, 0, 255)) == "magenta"


def test_dominant_colors_from_bundle() -> None:
    import numpy as np

    from spritelab.codec.bundle import SpriteBundle
    from spritelab.codec.color_names import dominant_colors_from_bundle

    palette = np.array([[0, 0, 0], [220, 40, 40], [60, 100, 220], [225, 45, 45]], dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.int64)
    index_map[0:16, :] = 1  # red half
    index_map[16:20, :] = 2  # blue band
    index_map[20:24, :] = 3  # near-identical red merges by name
    alpha = (index_map > 0).astype(np.uint8)
    bundle = SpriteBundle(alpha=alpha, palette=palette, index_map=index_map, role_map=None, metadata={})

    colors = dominant_colors_from_bundle(bundle)
    assert colors[0] == "red"
    assert "blue" in colors

    empty = SpriteBundle(
        alpha=np.zeros((32, 32), dtype=np.uint8),
        palette=np.array([[0, 0, 0]], dtype=np.uint8),
        index_map=np.zeros((32, 32), dtype=np.int64),
        role_map=None,
        metadata={},
    )
    assert dominant_colors_from_bundle(empty) == ()


def test_adjudication_prompt_and_parse() -> None:
    from spritelab.dataset_maker.prefill import build_adjudication_prompt, parse_adjudication_result

    prompt = build_adjudication_prompt(
        {"content_width": 8, "content_height": 8, "dominant_colors": ["red"]},
        candidate_a={"category": "plant", "object_name": "mushroom", "tags": ["mushroom"]},
        candidate_b={"category": "weapon", "object_name": "axe", "tags": ["axe"]},
    )
    assert "Candidate A" in prompt
    assert '"object_name": "mushroom"' in prompt
    assert '"object_name": "axe"' in prompt
    # Anonymity: the prompt must not reveal where candidates came from.
    assert "filename" not in prompt.lower()
    assert "qwen" not in prompt.lower()

    result = parse_adjudication_result(json.dumps({"choice": "B", "reason": "axe blade"}))
    assert result.choice == "b"
    assert result.reason == "axe blade"

    invalid = parse_adjudication_result("no json here")
    assert invalid.choice == "cannot_tell"
    assert invalid.warnings

    unknown_choice = parse_adjudication_result(json.dumps({"choice": "maybe", "reason": ""}))
    assert unknown_choice.choice == "cannot_tell"


def test_openai_backend_adjudicate_sends_schema_and_parses(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"choice": "a", "reason": "cap and stem visible"})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True))

    result = backend.adjudicate(
        _request(),
        {"category": "plant", "object_name": "mushroom"},
        {"category": "weapon", "object_name": "axe"},
    )

    assert result is not None
    assert result.choice == "a"
    schema = captured["payload"]["response_format"]["json_schema"]
    assert schema["name"] == "sprite_adjudication"
    assert "choice" in schema["schema"]["required"]


def test_cached_backend_caches_adjudications(tmp_path) -> None:
    from spritelab.dataset_maker.prefill import AdjudicationResult

    class _CountingBackend(MetadataPrefillBackend):
        model = "fake"

        def __init__(self) -> None:
            self.adjudicate_calls = 0

        def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
            return MetadataSuggestion(category="plant")

        def adjudicate(self, request, candidate_a, candidate_b):
            self.adjudicate_calls += 1
            return AdjudicationResult(choice="a", reason="visible cap")

    inner = _CountingBackend()
    backend = CachedPrefillBackend(inner, tmp_path / "cache")
    request = _request()
    a = {"category": "plant", "object_name": "mushroom"}
    b = {"category": "weapon", "object_name": "axe"}

    first = backend.adjudicate(request, a, b)
    second = backend.adjudicate(request, a, b)
    assert inner.adjudicate_calls == 1
    assert first == second or (first.choice == second.choice and first.reason == second.reason)

    # Different candidates use a different cache slot.
    backend.adjudicate(request, a, {"category": "tool", "object_name": "pickaxe"})
    assert inner.adjudicate_calls == 2


def test_base_backend_adjudicate_returns_none() -> None:
    assert NoopPrefillBackend().adjudicate(_request(), {}, {}) is None


class _ScriptedBackend(MetadataPrefillBackend):
    """Returns scripted suggestions keyed by request.sample_index."""

    model = "scripted"

    def __init__(self, by_sample: dict[int, MetadataSuggestion]) -> None:
        self.by_sample = by_sample
        self.calls: list[int] = []

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        self.calls.append(request.sample_index)
        return self.by_sample[request.sample_index]


def _voted(by_sample: dict[int, MetadataSuggestion], **config_kwargs) -> tuple[MetadataSuggestion, "_ScriptedBackend"]:
    from spritelab.dataset_maker.prefill import SelfConsistencyBackend

    inner = _ScriptedBackend(by_sample)
    backend = SelfConsistencyBackend(inner, PrefillConfig(enabled=True, **config_kwargs))
    return backend.suggest(_request()), inner


def test_adaptive_voting_skips_confident_anchor() -> None:
    anchor = MetadataSuggestion(category="plant", object_name="fern", tags=("fern",), confidence=0.9)
    result, inner = _voted({0: anchor}, votes=3, vote_mode="adaptive")
    assert inner.calls == [0]
    assert result.vote_stats is None
    assert result.category == "plant"


def test_adaptive_voting_escalates_on_uncertainty() -> None:
    by_sample = {
        0: MetadataSuggestion(category="plant", object_name="fern", uncertainty="unsure", confidence=0.45),
        1: MetadataSuggestion(category="plant", object_name="fern", confidence=0.7),
        2: MetadataSuggestion(category="plant", object_name="fern", confidence=0.7),
    }
    result, inner = _voted(by_sample, votes=3, vote_mode="adaptive")
    assert inner.calls == [0, 1, 2]
    assert result.category == "plant"
    assert result.object_name == "fern"
    assert result.vote_stats["k_used"] == 3
    assert result.vote_stats["category_agreement"] == 1.0
    assert result.confidence is not None and result.confidence > 0.5


def test_always_voting_votes_every_time() -> None:
    by_sample = {
        0: MetadataSuggestion(category="plant", object_name="fern", confidence=0.9),
        1: MetadataSuggestion(category="plant", object_name="fern", confidence=0.9),
        2: MetadataSuggestion(category="weapon", object_name="sword", confidence=0.9),
    }
    result, inner = _voted(by_sample, votes=3, vote_mode="always")
    assert inner.calls == [0, 1, 2]
    assert result.category == "plant"
    assert result.vote_stats["category_agreement"] == round(2 / 3, 4)


def test_voting_category_tie_falls_back_to_unknown() -> None:
    from spritelab.dataset_maker.prefill import merge_voted_suggestions

    samples = [
        MetadataSuggestion(category="plant", object_name="fern", confidence=0.8),
        MetadataSuggestion(category="weapon", object_name="sword", confidence=0.8),
    ]
    merged = merge_voted_suggestions(samples)
    assert merged.category == "unknown"
    assert any("vote_tie" in warning for warning in merged.warnings)
    assert merged.confidence is not None and merged.confidence < 0.6


def test_voting_excludes_degenerate_samples() -> None:
    from spritelab.dataset_maker.prefill import flag_degenerate_suggestion, merge_voted_suggestions

    degenerate = flag_degenerate_suggestion(
        MetadataSuggestion(category="ui_icon", object_name="checkerboard", confidence=0.9)
    )
    good = MetadataSuggestion(category="plant", object_name="fern", tags=("fern",), confidence=0.8)
    merged = merge_voted_suggestions([degenerate, good, good])
    assert merged.category == "plant"
    assert merged.vote_stats["k_used"] == 2
    assert merged.vote_stats["category_agreement"] == 1.0


def test_voting_tag_frequency_threshold() -> None:
    from spritelab.dataset_maker.prefill import merge_voted_suggestions

    samples = [
        MetadataSuggestion(category="plant", object_name="fern", tags=("fern", "green", "leafy"), confidence=0.8),
        MetadataSuggestion(category="plant", object_name="fern", tags=("fern", "green"), confidence=0.8),
        MetadataSuggestion(category="plant", object_name="fern", tags=("fern", "spiky"), confidence=0.8),
    ]
    merged = merge_voted_suggestions(samples)
    assert merged.tags == ("fern", "green")  # >= ceil(3/2) = 2 votes


def test_voting_transport_failure_not_escalated() -> None:
    failure = MetadataSuggestion(warnings=("prefill request could not connect to endpoint",))
    result, inner = _voted({0: failure}, votes=3, vote_mode="adaptive")
    assert inner.calls == [0]
    assert result.warnings


def test_create_backend_composes_voting_and_cache(tmp_path) -> None:
    from spritelab.dataset_maker.prefill import SelfConsistencyBackend

    backend = create_prefill_backend(
        PrefillConfig(enabled=True, backend="openai_compatible", cache_dir=tmp_path, votes=3)
    )
    assert isinstance(backend, SelfConsistencyBackend)
    assert isinstance(backend.backend, CachedPrefillBackend)

    off = create_prefill_backend(
        PrefillConfig(enabled=True, backend="openai_compatible", cache_dir=tmp_path, vote_mode="off")
    )
    assert isinstance(off, CachedPrefillBackend)


def test_vote_sample_index_changes_cache_key_and_payload_seed(monkeypatch, tmp_path) -> None:
    seeds: list[int] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        payload = json.loads(request.data.decode("utf-8"))
        seeds.append(payload["seed"])
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "plant", "object_name": "fern", "uncertainty": "unsure"})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = create_prefill_backend(
        PrefillConfig(
            enabled=True,
            backend="openai_compatible",
            cache_dir=tmp_path,
            votes=2,
            vote_mode="always",
            retry_attempts=0,
            min_qwen_confidence=0.0,
        )
    )
    backend.suggest(_request())
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]
    assert len(list(tmp_path.glob("*.json"))) == 2


class _FakeErrorBody:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


def test_openai_compatible_backend_uses_runpod_token_for_bearer_auth(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["request"] = request
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"category": "item_icon"})}}]}
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(
        PrefillConfig(enabled=True, api_key="ordinary", runpod_token="runpod-secret")
    )

    backend.suggest(_request())

    assert captured["request"].get_header("Authorization") == "Bearer runpod-secret"


def test_openai_compatible_backend_parses_mocked_successful_response(monkeypatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "category": "plant",
                                    "object_name": "mushroom",
                                    "tags": ["purple"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True))

    suggestion = backend.suggest(_request())

    assert suggestion.category == "plant"
    assert suggestion.object_name == "mushroom"
    assert suggestion.tags == ("purple",)


def test_openai_compatible_backend_returns_warning_on_timeout(monkeypatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise TimeoutError("slow")

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleQwenPrefillBackend(PrefillConfig(enabled=True, timeout_seconds=1))

    suggestion = backend.suggest(_request())

    assert any("timed out" in warning for warning in suggestion.warnings)


def test_ollama_backend_sends_native_chat_payload_and_parses_response(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "category": "plant",
                            "object_name": "mushroom",
                            "tags": ["purple"],
                        }
                    )
                }
            }
        )

    monkeypatch.setattr(prefill.urllib.request, "urlopen", fake_urlopen)
    backend = OllamaQwenPrefillBackend(
        PrefillConfig(
            enabled=True,
            backend="ollama",
            model="qwen2.5vl:7b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=7,
        )
    )

    suggestion = backend.suggest(_request())

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 7
    assert payload["model"] == "qwen2.5vl:7b"
    assert payload["stream"] is False
    assert payload["messages"][0]["images"][0]
    assert suggestion.category == "plant"
    assert suggestion.object_name == "mushroom"


def test_backend_factory_returns_noop_when_disabled() -> None:
    backend = create_prefill_backend(PrefillConfig(enabled=False))

    assert isinstance(backend, NoopPrefillBackend)


def test_backend_factory_returns_rule_based_backend() -> None:
    backend = create_prefill_backend(PrefillConfig(enabled=True, backend="rule_based"))

    assert isinstance(backend, RuleBasedPrefillBackend)


def test_backend_factory_returns_ollama_backend() -> None:
    backend = create_prefill_backend(PrefillConfig(enabled=True, backend="ollama", vote_mode="off"))

    assert isinstance(backend, OllamaQwenPrefillBackend)


def test_backend_factory_wraps_ollama_in_voting_by_default() -> None:
    from spritelab.dataset_maker.prefill import SelfConsistencyBackend

    backend = create_prefill_backend(PrefillConfig(enabled=True, backend="ollama"))
    assert isinstance(backend, SelfConsistencyBackend)
    assert isinstance(backend.backend, OllamaQwenPrefillBackend)


def test_backend_factory_rejects_unknown_backend() -> None:
    try:
        create_prefill_backend(PrefillConfig(enabled=True, backend="missing"))
    except ValueError as exc:
        assert "unknown prefill backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class _CountingBackend(MetadataPrefillBackend):
    model = "counting"

    def __init__(self) -> None:
        self.count = 0

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        self.count += 1
        return MetadataSuggestion(category="item_icon", tags=("cached",), raw_response='{"category":"item_icon"}')


class _WarningOnlyBackend(MetadataPrefillBackend):
    model = "warning_only"

    def __init__(self) -> None:
        self.count = 0

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        self.count += 1
        return MetadataSuggestion(warnings=("temporary failure",))


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _request() -> PrefillRequest:
    return PrefillRequest(sprite_id="sprite", image=_image(), source_path="sprite.png")


def _image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 255))
    return image


def _item(
    *,
    category: str = "unknown",
    tags: tuple[str, ...] = (),
    status: str = "accepted",
    split: str | None = None,
) -> DatasetMakerItem:
    return DatasetMakerItem(
        sprite_id="sprite",
        source_path=Path("sprite.png"),
        status=status,
        category=category,
        tags=tags,
        notes="",
        source_name="source.png",
        license="own_work",
        author="artist",
        split=split,
    )
