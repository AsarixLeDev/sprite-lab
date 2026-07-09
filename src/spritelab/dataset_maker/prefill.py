"""Optional local VLM metadata prefill for Dataset Maker sprites."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.dataset_maker.model import DatasetMakerItem, normalize_category, normalize_sprite_id, normalize_tag

PROMPT_VERSION = "qwen_prefill_v2"
DESCRIPTOR_PROMPT_VERSION = "qwen_descriptor_v2"
DESCRIPTOR_SCHEMA_VERSION = "descriptor_schema_v2"
VLM_IMAGE_PREP_VERSION = "vlm_image_prep_v2"
DESCRIPTOR_IMAGE_VIEWS = ("full", "crop", "both")
ALLOWED_CATEGORIES = {
    "unknown",
    "item_icon",
    "block",
    "plant",
    "ui_icon",
    "entity",
    "character",
    "weapon",
    "tool",
    "armor",
    "material",
    "effect_icon",
    "environment_prop",
}

_CATEGORY_DEFINITIONS = (
    ("unknown", "cannot tell what the object is"),
    ("item_icon", "inventory/collectible object icon (food, potion, coin, key, ...)"),
    ("block", "full-tile terrain or building tile"),
    ("plant", "vegetation (flower, mushroom, tree, herb, ...)"),
    ("ui_icon", "interface element (button, cursor, frame, arrow, ...)"),
    ("entity", "creature or monster"),
    ("character", "humanoid person, hero, or NPC"),
    ("weapon", "offensive equipment (sword, bow, axe, ...)"),
    ("tool", "utility equipment (pickaxe, hammer, fishing rod, ...)"),
    ("armor", "wearable defensive equipment (helmet, shield, ...)"),
    ("material", "crafting resource (ingot, ore, dust, gem, ...)"),
    ("effect_icon", "spell, status, buff, or debuff icon"),
    ("environment_prop", "scenery object (rock, barrel, fence, sign, ...)"),
)

UNCERTAINTY_LEVELS = ("confident", "likely", "unsure", "cannot_tell")
_UNCERTAINTY_TO_CONFIDENCE = {
    "confident": 0.9,
    "likely": 0.7,
    "unsure": 0.45,
    "cannot_tell": 0.2,
}

# Enforced at decode time on servers that support OpenAI response_format
# (vLLM guided JSON) or Ollama schema-valued format.
METADATA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
        "object_name": {"type": "string", "maxLength": 48},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 8},
        "materials": {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 4},
        "mood": {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 3},
        "short_description": {"type": "string", "maxLength": 160},
        "suggested_sprite_id": {"type": "string", "pattern": "^[a-z0-9_]*$", "maxLength": 48},
        "uncertainty": {"type": "string", "enum": list(UNCERTAINTY_LEVELS)},
        "visual_evidence": {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 5},
        "warnings": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 3},
    },
    "required": ["category", "object_name", "tags", "short_description", "uncertainty"],
}

DESCRIPTOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "visual_description": {"type": "string", "maxLength": 220},
        "visual_tags": {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 10},
        "source_consistency": {"type": "string", "enum": ["consistent", "unclear", "contradicted", "no_source"]},
        "evidence_for_source": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 5},
        "evidence_against_source": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 5},
        "possible_object_name": {"type": "string", "maxLength": 48},
        "alternative_object_names": {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 3},
        "possible_category": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
        "agrees_with_source": {"type": "string", "enum": ["yes", "no", "unclear", "no_source"]},
        "contradiction_reason": {"type": "string", "maxLength": 180},
        "uncertainty": {"type": "string", "enum": list(UNCERTAINTY_LEVELS)},
        "warnings": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 5},
    },
    "required": [
        "visual_description",
        "visual_tags",
        "source_consistency",
        "evidence_for_source",
        "evidence_against_source",
        "possible_object_name",
        "alternative_object_names",
        "possible_category",
        "uncertainty",
        "warnings",
    ],
}

ADJUDICATION_CHOICES = ("a", "b", "both_wrong", "cannot_tell")

# Forced-choice schema used when a blind label conflicts with filename rules.
ADJUDICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "choice": {"type": "string", "enum": list(ADJUDICATION_CHOICES)},
        "corrected_category": {"type": "string", "enum": ["", *sorted(ALLOWED_CATEGORIES)]},
        "corrected_object_name": {"type": "string", "maxLength": 48},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["choice", "reason"],
}

_DEGENERATE_TEXT_PATTERNS = (
    "checkerboard",
    "checkered",
    "checker board",
    "magenta background",
    "pink background",
    "grid pattern",
    "background pattern",
)
_DEGENERATE_OBJECTS = {
    "unknown",
    "unknown_object",
    "unidentified",
    "unidentified_object",
    "ambiguous",
    "ambiguous_object",
    "ambiguous_shape",
    "checkerboard",
    "background",
}
DEGENERATE_WARNING_PREFIX = "degenerate_response"
_RETRY_SEED_BASE = 1234


@dataclass(frozen=True)
class MetadataSuggestion:
    category: str = "unknown"
    object_name: str = ""
    tags: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    dominant_colors: tuple[str, ...] = ()
    short_description: str = ""
    suggested_sprite_id: str = ""
    confidence: float | None = None
    uncertainty: str = ""
    warnings: tuple[str, ...] = ()
    filename_agreement: str = ""
    visual_evidence: tuple[str, ...] = ()
    disagreement_reason: str = ""
    source_consistency: str = ""
    alternative_object_names: tuple[str, ...] = ()
    evidence_for_source: tuple[str, ...] = ()
    evidence_against_source: tuple[str, ...] = ()
    vote_stats: Mapping[str, Any] | None = None
    raw_response: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        object.__setattr__(self, "tags", _normalize_sequence(self.tags))
        object.__setattr__(self, "materials", _normalize_sequence(self.materials))
        object.__setattr__(self, "mood", _normalize_sequence(self.mood))
        object.__setattr__(self, "dominant_colors", _normalize_sequence(self.dominant_colors))
        object.__setattr__(self, "short_description", str(self.short_description).strip())
        object.__setattr__(self, "suggested_sprite_id", normalize_sprite_id(self.suggested_sprite_id))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "uncertainty", normalize_tag(self.uncertainty))
        object.__setattr__(
            self, "warnings", tuple(str(warning).strip() for warning in self.warnings if str(warning).strip())
        )
        object.__setattr__(self, "filename_agreement", normalize_tag(self.filename_agreement))
        object.__setattr__(self, "visual_evidence", _normalize_sequence(self.visual_evidence))
        object.__setattr__(self, "disagreement_reason", str(self.disagreement_reason).strip())
        source_consistency = _normalize_source_consistency(self.source_consistency or self.filename_agreement)
        object.__setattr__(self, "source_consistency", source_consistency)
        object.__setattr__(self, "alternative_object_names", _normalize_sequence(self.alternative_object_names)[:3])
        object.__setattr__(self, "evidence_for_source", _clean_string_tuple(self.evidence_for_source, max_items=5))
        object.__setattr__(
            self, "evidence_against_source", _clean_string_tuple(self.evidence_against_source, max_items=5)
        )
        if self.vote_stats is not None:
            object.__setattr__(self, "vote_stats", dict(self.vote_stats))
        object.__setattr__(self, "raw_response", str(self.raw_response))


@dataclass(frozen=True)
class AdjudicationResult:
    """Outcome of a forced-choice call between two candidate labels."""

    choice: str = "cannot_tell"
    corrected_category: str = ""
    corrected_object_name: str = ""
    reason: str = ""
    warnings: tuple[str, ...] = ()
    raw_response: str = ""

    def __post_init__(self) -> None:
        choice = normalize_tag(self.choice)
        object.__setattr__(self, "choice", choice if choice in ADJUDICATION_CHOICES else "cannot_tell")
        corrected_category = (
            normalize_category(str(self.corrected_category)) if str(self.corrected_category).strip() else ""
        )
        if corrected_category and corrected_category not in ALLOWED_CATEGORIES:
            corrected_category = ""
        object.__setattr__(self, "corrected_category", corrected_category)
        object.__setattr__(self, "corrected_object_name", normalize_tag(self.corrected_object_name))
        object.__setattr__(self, "reason", str(self.reason).strip())
        object.__setattr__(
            self, "warnings", tuple(str(warning).strip() for warning in self.warnings if str(warning).strip())
        )
        object.__setattr__(self, "raw_response", str(self.raw_response))


def adjudication_to_dict(result: AdjudicationResult) -> dict[str, Any]:
    return {
        "choice": result.choice,
        "corrected_category": result.corrected_category,
        "corrected_object_name": result.corrected_object_name,
        "reason": result.reason,
        "warnings": list(result.warnings),
    }


@dataclass(frozen=True)
class PrefillRequest:
    sprite_id: str
    image: Image.Image
    existing_category: str = "unknown"
    existing_tags: tuple[str, ...] = ()
    source_path: str = ""
    filename_suggestion: Mapping[str, Any] | None = None
    image_facts: Mapping[str, Any] | None = None
    candidate_object_names: tuple[str, ...] = ()
    # Self-consistency sample number; 0 is the deterministic anchor, higher
    # indices deterministically map to a different temperature/seed.
    sample_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_index", max(0, int(self.sample_index)))
        object.__setattr__(self, "sprite_id", normalize_sprite_id(self.sprite_id))
        object.__setattr__(self, "existing_category", normalize_category(self.existing_category))
        object.__setattr__(self, "existing_tags", _normalize_sequence(self.existing_tags))
        object.__setattr__(self, "source_path", str(self.source_path))
        if self.filename_suggestion is not None:
            object.__setattr__(self, "filename_suggestion", dict(self.filename_suggestion))
        if self.image_facts is not None:
            object.__setattr__(self, "image_facts", dict(self.image_facts))
        object.__setattr__(self, "candidate_object_names", _normalize_sequence(self.candidate_object_names))


@dataclass(frozen=True)
class PrefillConfig:
    enabled: bool = False
    backend: str = "openai_compatible"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"
    runpod_token: str = ""
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512
    upscale: int = 16
    cache_dir: Path | None = None
    # Blind-first: the filename hint measurably biased the model into copying
    # it. Agreement is computed in code and conflicts adjudicated separately.
    include_filename_hint: bool = False
    retry_attempts: int = 2
    retry_on_warning_only: bool = True
    min_qwen_confidence: float = 0.55
    fusion_policy: str = "weighted"
    structured_output: str = "auto"
    votes: int = 3
    vote_mode: str = "adaptive"
    vote_temperature: float = 0.5
    vlm_role: str = "labeler"
    vlm_image_view: str = "both"
    descriptor_full_size: int = 512
    descriptor_crop_size: int = 512
    descriptor_small_crop_size: int = 768

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", str(self.backend).strip().lower())
        object.__setattr__(self, "structured_output", str(self.structured_output).strip().lower() or "auto")
        vlm_role = str(self.vlm_role).strip().lower() or "labeler"
        if vlm_role not in {"labeler", "descriptor", "verifier"}:
            raise ValueError(f"vlm_role must be labeler, descriptor, or verifier, not {vlm_role!r}")
        object.__setattr__(self, "vlm_role", vlm_role)
        object.__setattr__(self, "votes", max(1, int(self.votes)))
        vote_mode = str(self.vote_mode).strip().lower() or "adaptive"
        if vote_mode not in {"adaptive", "always", "off"}:
            raise ValueError(f"vote_mode must be adaptive, always, or off, not {vote_mode!r}")
        object.__setattr__(self, "vote_mode", vote_mode)
        object.__setattr__(self, "vote_temperature", max(0.0, float(self.vote_temperature)))
        image_view = str(self.vlm_image_view).strip().lower() or "both"
        if image_view not in DESCRIPTOR_IMAGE_VIEWS:
            raise ValueError(f"vlm_image_view must be full, crop, or both, not {image_view!r}")
        object.__setattr__(self, "vlm_image_view", image_view)
        object.__setattr__(self, "descriptor_full_size", max(1, int(self.descriptor_full_size)))
        object.__setattr__(self, "descriptor_crop_size", max(1, int(self.descriptor_crop_size)))
        object.__setattr__(self, "descriptor_small_crop_size", max(1, int(self.descriptor_small_crop_size)))
        object.__setattr__(self, "model", str(self.model).strip() or "Qwen/Qwen3-VL-8B-Instruct")
        object.__setattr__(self, "base_url", str(self.base_url).strip().rstrip("/"))
        object.__setattr__(self, "api_key", str(self.api_key))
        object.__setattr__(self, "runpod_token", str(self.runpod_token).strip())
        object.__setattr__(self, "retry_attempts", max(0, int(self.retry_attempts)))
        object.__setattr__(self, "min_qwen_confidence", max(0.0, min(1.0, float(self.min_qwen_confidence))))
        object.__setattr__(self, "fusion_policy", str(self.fusion_policy).strip() or "weighted")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))


class MetadataPrefillBackend:
    """Protocol-style base class for metadata suggestion backends."""

    model = "none"
    prompt_version = PROMPT_VERSION

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        raise NotImplementedError

    def adjudicate(
        self,
        request: PrefillRequest,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
    ) -> AdjudicationResult | None:
        """Forced choice between two candidate labels; None when unsupported."""

        return None


class NoopPrefillBackend(MetadataPrefillBackend):
    """Disabled prefill backend."""

    model = "noop"

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        return MetadataSuggestion(warnings=("metadata prefill is disabled.",), raw_response="")


class RuleBasedPrefillBackend(MetadataPrefillBackend):
    """Small deterministic fallback for filename-derived metadata."""

    model = "rule_based"

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        source_stem = Path(request.source_path).stem if request.source_path else request.sprite_id
        suggested_sprite_id = normalize_sprite_id(source_stem or request.sprite_id)
        tokens = _tokens_from_text(suggested_sprite_id)
        category = _category_from_tokens(tokens)
        object_name = " ".join(tokens)
        tags = tuple(token for token in tokens if token not in {"sprite", "icon", "png"})
        return MetadataSuggestion(
            category=category,
            object_name=object_name,
            tags=tags,
            suggested_sprite_id=suggested_sprite_id,
            confidence=0.35 if category != "unknown" else 0.15,
            warnings=("Rule-based suggestion only; review manually.",),
            raw_response="",
        )


class OpenAICompatibleQwenPrefillBackend(MetadataPrefillBackend):
    """OpenAI-compatible local VLM metadata suggestion backend."""

    def __init__(self, config: PrefillConfig) -> None:
        self.config = config
        self.model = config.model
        prompt_version = DESCRIPTOR_PROMPT_VERSION if config.vlm_role == "descriptor" else PROMPT_VERSION
        self.prompt_version = _backend_prompt_version(prompt_version, config)
        self._structured_unsupported = False

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        images = _prepared_images_for_request(self.config, request.image)
        last_suggestion = _warning_suggestion("prefill did not run.")
        for attempt in range(self.config.retry_attempts + 1):
            retry_note = _retry_note(last_suggestion) if attempt else ""
            suggestion = self._suggest_once(endpoint, images, request, retry_note=retry_note, attempt=attempt)
            last_suggestion = suggestion
            if attempt >= self.config.retry_attempts or not _should_retry_suggestion(suggestion, self.config):
                return _with_retry_count(suggestion, attempt)
        return last_suggestion

    def _suggest_once(
        self,
        endpoint: str,
        images: Sequence[tuple[str, Image.Image]],
        request: PrefillRequest,
        *,
        retry_note: str = "",
        attempt: int = 0,
    ) -> MetadataSuggestion:
        try:
            payload = self._payload(images, request=request, retry_note=retry_note, attempt=attempt)
            status, response_text = self._post(endpoint, payload)
        except TimeoutError as exc:
            return _warning_suggestion(
                f"prefill request timed out after {self.config.timeout_seconds:g} seconds: {exc}"
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error(exc)
            if self._should_fall_back_to_plain(exc.code, detail):
                self._structured_unsupported = True
                return self._suggest_once(endpoint, images, request, retry_note=retry_note, attempt=attempt)
            return _warning_suggestion(f"prefill request returned HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            return _warning_suggestion(_format_url_error(exc, endpoint))
        except OSError as exc:
            return _warning_suggestion(f"prefill request failed for {endpoint}: {exc}")

        if status < 200 or status >= 300:
            return _warning_suggestion(f"prefill request returned HTTP {status}: {response_text[:500]}")

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            return _warning_suggestion(f"prefill server returned invalid JSON: {exc}", raw_response=response_text)

        content = _extract_response_text(data)
        if not content:
            return _warning_suggestion(
                "prefill server response did not contain message text.", raw_response=response_text
            )
        if self.config.vlm_role == "descriptor":
            return flag_degenerate_suggestion(parse_descriptor_suggestion(content))
        return flag_degenerate_suggestion(parse_metadata_suggestion(content))

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> tuple[int, str]:
        body = json.dumps(dict(payload)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = _bearer_token(self.config)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        http_request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode()))
            return status, response.read().decode("utf-8")

    def _structured_enabled(self) -> bool:
        if self.config.structured_output == "off":
            return False
        if self.config.structured_output == "auto" and self._structured_unsupported:
            return False
        return True

    def _should_fall_back_to_plain(self, code: int, detail: str) -> bool:
        if self.config.structured_output != "auto" or self._structured_unsupported:
            return False
        if code != 400:
            return False
        lowered = detail.lower()
        return any(marker in lowered for marker in ("response_format", "json_schema", "guided", "structured"))

    def _payload(
        self,
        images: Sequence[tuple[str, Image.Image]],
        *,
        request: PrefillRequest,
        retry_note: str = "",
        attempt: int = 0,
    ) -> dict[str, Any]:
        # Retries and vote samples nudge temperature and seed so a degenerate
        # or unlucky answer is not deterministically reproduced.
        temperature = float(self.config.temperature) if attempt == 0 else max(float(self.config.temperature), 0.3)
        if request.sample_index > 0:
            temperature = max(float(self.config.vote_temperature), 0.1)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _prompt_for_role(
                    self.config,
                    request.filename_suggestion
                    if self.config.include_filename_hint or self.config.vlm_role == "descriptor"
                    else None,
                    image_facts=request.image_facts,
                    retry_note=retry_note,
                    candidate_object_names=request.candidate_object_names,
                ),
            }
        ]
        content.extend({"type": "image_url", "image_url": {"url": image_to_data_url(image)}} for _, image in images)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": temperature,
            "max_tokens": int(self.config.max_tokens),
            "seed": _RETRY_SEED_BASE + 1000 * request.sample_index + attempt,
        }
        if self._structured_enabled():
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sprite_descriptor" if self.config.vlm_role == "descriptor" else "sprite_metadata",
                    "strict": True,
                    "schema": DESCRIPTOR_JSON_SCHEMA if self.config.vlm_role == "descriptor" else METADATA_JSON_SCHEMA,
                },
            }
        return payload

    def adjudicate(
        self,
        request: PrefillRequest,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
    ) -> AdjudicationResult | None:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        image = prepare_vlm_image(request.image, upscale=self.config.upscale)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_adjudication_prompt(
                                request.image_facts,
                                candidate_a=candidate_a,
                                candidate_b=candidate_b,
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    ],
                }
            ],
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
            "seed": _RETRY_SEED_BASE,
        }
        if self._structured_enabled():
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sprite_adjudication",
                    "strict": True,
                    "schema": ADJUDICATION_JSON_SCHEMA,
                },
            }
        try:
            status, response_text = self._post(endpoint, payload)
        except urllib.error.HTTPError as exc:
            detail = _read_http_error(exc)
            if self._should_fall_back_to_plain(exc.code, detail):
                self._structured_unsupported = True
                return self.adjudicate(request, candidate_a, candidate_b)
            return AdjudicationResult(warnings=(f"adjudication request returned HTTP {exc.code}: {detail}",))
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            return AdjudicationResult(warnings=(f"adjudication request failed: {exc}",))
        if status < 200 or status >= 300:
            return AdjudicationResult(warnings=(f"adjudication request returned HTTP {status}: {response_text[:500]}",))
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            return AdjudicationResult(
                warnings=(f"adjudication server returned invalid JSON: {exc}",), raw_response=response_text
            )
        content = _extract_response_text(data)
        if not content:
            return AdjudicationResult(
                warnings=("adjudication response did not contain message text.",), raw_response=response_text
            )
        return parse_adjudication_result(content)


class OllamaQwenPrefillBackend(MetadataPrefillBackend):
    """Native Ollama VLM backend using /api/chat with base64 images."""

    def __init__(self, config: PrefillConfig) -> None:
        self.config = config
        self.model = config.model
        self.base_url = _ollama_base_url(config.base_url)
        prompt_version = DESCRIPTOR_PROMPT_VERSION if config.vlm_role == "descriptor" else PROMPT_VERSION
        self.prompt_version = f"{_backend_prompt_version(prompt_version, config)}:ollama"

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        images = _prepared_images_for_request(self.config, request.image)
        last_suggestion = _warning_suggestion("ollama prefill did not run.")
        for attempt in range(self.config.retry_attempts + 1):
            retry_note = _retry_note(last_suggestion) if attempt else ""
            suggestion = self._suggest_once(endpoint, images, request, retry_note=retry_note, attempt=attempt)
            last_suggestion = suggestion
            if attempt >= self.config.retry_attempts or not _should_retry_suggestion(suggestion, self.config):
                return _with_retry_count(suggestion, attempt)
        return last_suggestion

    def _suggest_once(
        self,
        endpoint: str,
        images: Sequence[tuple[str, Image.Image]],
        request: PrefillRequest,
        *,
        retry_note: str = "",
        attempt: int = 0,
    ) -> MetadataSuggestion:
        try:
            temperature = float(self.config.temperature) if attempt == 0 else max(float(self.config.temperature), 0.3)
            if request.sample_index > 0:
                temperature = max(float(self.config.vote_temperature), 0.1)
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": _prompt_for_role(
                            self.config,
                            request.filename_suggestion
                            if self.config.include_filename_hint or self.config.vlm_role == "descriptor"
                            else None,
                            image_facts=request.image_facts,
                            retry_note=retry_note,
                            candidate_object_names=request.candidate_object_names,
                        ),
                        "images": [_image_to_base64_png(image) for _, image in images],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": _RETRY_SEED_BASE + 1000 * request.sample_index + attempt,
                },
            }
            if self.config.structured_output != "off":
                payload["format"] = (
                    DESCRIPTOR_JSON_SCHEMA if self.config.vlm_role == "descriptor" else METADATA_JSON_SCHEMA
                )
            body = json.dumps(payload).encode("utf-8")
            http_request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                response_text = response.read().decode("utf-8")
        except TimeoutError as exc:
            return _warning_suggestion(
                f"ollama prefill request timed out after {self.config.timeout_seconds:g} seconds: {exc}"
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error(exc)
            return _warning_suggestion(f"ollama prefill request returned HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            return _warning_suggestion(_format_ollama_url_error(exc, endpoint))
        except OSError as exc:
            return _warning_suggestion(f"ollama prefill request failed for {endpoint}: {exc}")

        if status < 200 or status >= 300:
            return _warning_suggestion(f"ollama prefill request returned HTTP {status}: {response_text[:500]}")
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            return _warning_suggestion(f"ollama returned invalid JSON: {exc}", raw_response=response_text)
        content = _extract_ollama_response_text(data)
        if not content:
            return _warning_suggestion("ollama response did not contain message content.", raw_response=response_text)
        if self.config.vlm_role == "descriptor":
            return flag_degenerate_suggestion(parse_descriptor_suggestion(content))
        return flag_degenerate_suggestion(parse_metadata_suggestion(content))

    def adjudicate(
        self,
        request: PrefillRequest,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
    ) -> AdjudicationResult | None:
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        image = prepare_vlm_image(request.image, upscale=self.config.upscale)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_adjudication_prompt(
                        request.image_facts,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                    ),
                    "images": [_image_to_base64_png(image)],
                }
            ],
            "stream": False,
            "options": {"temperature": float(self.config.temperature), "seed": _RETRY_SEED_BASE},
        }
        if self.config.structured_output != "off":
            payload["format"] = ADJUDICATION_JSON_SCHEMA
        try:
            body = json.dumps(payload).encode("utf-8")
            http_request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                response_text = response.read().decode("utf-8")
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            return AdjudicationResult(warnings=(f"ollama adjudication request failed: {exc}",))
        if status < 200 or status >= 300:
            return AdjudicationResult(warnings=(f"ollama adjudication returned HTTP {status}: {response_text[:500]}",))
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            return AdjudicationResult(
                warnings=(f"ollama adjudication returned invalid JSON: {exc}",), raw_response=response_text
            )
        content = _extract_ollama_response_text(data)
        if not content:
            return AdjudicationResult(
                warnings=("ollama adjudication response had no content.",), raw_response=response_text
            )
        return parse_adjudication_result(content)


class CachedPrefillBackend(MetadataPrefillBackend):
    """Cache wrapper for repeated local VLM suggestions."""

    def __init__(self, backend: MetadataPrefillBackend, cache_dir: Path) -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.model = getattr(backend, "model", "unknown")
        self.prompt_version = getattr(backend, "prompt_version", PROMPT_VERSION)
        self._locks_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        context = json.dumps(
            {
                "filename_suggestion": request.filename_suggestion or {},
                "image_facts": request.image_facts or {},
                "candidate_object_names": request.candidate_object_names,
                "sample_index": request.sample_index,
            },
            sort_keys=True,
            default=str,
        )
        backend_config = getattr(self.backend, "config", None)
        descriptor_cache = getattr(backend_config, "vlm_role", "") == "descriptor"
        key = compute_image_cache_key(
            request.image,
            model=self.model,
            prompt_version=self.prompt_version,
            context=context,
            image_prep_version=VLM_IMAGE_PREP_VERSION if descriptor_cache else "legacy_v1",
            image_view=getattr(backend_config, "vlm_image_view", "crop") if descriptor_cache else "legacy_crop",
            target_size=getattr(backend_config, "descriptor_crop_size", 512) if descriptor_cache else 32 * 16,
        )
        path = self.cache_dir / f"{key}.json"
        with self._lock_for_key(key):
            cached = _read_cached_suggestion(path)
            if cached is not None:
                return cached

            suggestion = self.backend.suggest(request)
            if suggestion.warnings and not _suggestion_has_content(suggestion):
                return suggestion
            if _has_degenerate_warning(suggestion):
                # Degenerate answers must stay retryable on the next run.
                return suggestion
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "suggestion": _suggestion_to_dict(suggestion, include_raw=True),
                "raw_response": suggestion.raw_response,
                "created": _utc_timestamp(),
                "model": self.model,
                "prompt_version": self.prompt_version,
            }
            tmp_path = path.with_name(f"{path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
            return suggestion

    def adjudicate(
        self,
        request: PrefillRequest,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
    ) -> AdjudicationResult | None:
        context = json.dumps(
            {
                "mode": "adjudicate",
                "candidate_a": dict(candidate_a),
                "candidate_b": dict(candidate_b),
                "image_facts": request.image_facts or {},
            },
            sort_keys=True,
            default=str,
        )
        key = compute_image_cache_key(
            request.image,
            model=self.model,
            prompt_version=self.prompt_version,
            context=context,
        )
        path = self.cache_dir / f"{key}.json"
        with self._lock_for_key(key):
            cached = _read_cached_adjudication(path)
            if cached is not None:
                return cached
            result = self.backend.adjudicate(request, candidate_a, candidate_b)
            if result is None:
                return None
            if result.warnings:
                # Transport failures stay retryable on the next run.
                return result
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "adjudication": adjudication_to_dict(result),
                "raw_response": result.raw_response,
                "created": _utc_timestamp(),
                "model": self.model,
                "prompt_version": self.prompt_version,
            }
            tmp_path = path.with_name(f"{path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
            return result

    def _lock_for_key(self, key: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock


class SelfConsistencyBackend(MetadataPrefillBackend):
    """Adaptive self-consistency voting over an inner (usually cached) backend.

    Sample 0 is the deterministic anchor. In ``adaptive`` mode extra samples
    are drawn only when the anchor looks weak; in ``always`` mode every sprite
    gets ``votes`` samples. The vote agreement replaces the model's
    self-reported confidence, which was measurably uninformative.
    """

    def __init__(self, backend: MetadataPrefillBackend, config: PrefillConfig) -> None:
        self.backend = backend
        self.config = config
        self.model = getattr(backend, "model", "unknown")
        self.prompt_version = getattr(backend, "prompt_version", PROMPT_VERSION)

    def suggest(self, request: PrefillRequest) -> MetadataSuggestion:
        anchor = self.backend.suggest(replace(request, sample_index=0))
        if self.config.vote_mode == "off" or self.config.votes <= 1:
            return anchor
        if self.config.vote_mode == "adaptive" and not self._needs_votes(anchor):
            return anchor
        samples = [anchor]
        for sample_index in range(1, self.config.votes):
            samples.append(self.backend.suggest(replace(request, sample_index=sample_index)))
        return merge_voted_suggestions(samples)

    def adjudicate(
        self,
        request: PrefillRequest,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
    ) -> AdjudicationResult | None:
        return self.backend.adjudicate(request, candidate_a, candidate_b)

    def _needs_votes(self, anchor: MetadataSuggestion) -> bool:
        if _has_degenerate_warning(anchor):
            return True
        if anchor.warnings and not _suggestion_has_content(anchor):
            # Transport failures will fail identically; do not multiply them.
            return False
        if anchor.uncertainty in {"unsure", "cannot_tell"}:
            return True
        if anchor.confidence is None or anchor.confidence < self.config.min_qwen_confidence:
            return True
        if anchor.category == "unknown" or not normalize_tag(anchor.object_name):
            return True
        return False


def merge_voted_suggestions(samples: Sequence[MetadataSuggestion]) -> MetadataSuggestion:
    """Merge self-consistency samples: majority fields, agreement as confidence."""

    if not samples:
        raise ValueError("merge_voted_suggestions requires at least one sample.")
    usable = [sample for sample in samples if _suggestion_has_content(sample) and not _has_degenerate_warning(sample)]
    if not usable:
        return samples[0]

    k = len(usable)
    category, category_count, category_tie = _majority(sample.category for sample in usable)
    warnings: list[str] = []
    if category_tie:
        category = "unknown"
        warnings.append("vote_tie: no category majority across samples.")
    object_name, object_count, object_tie = _majority(normalize_tag(sample.object_name) for sample in usable)
    if object_tie:
        object_name = ""

    category_agreement = category_count / k
    object_agreement = object_count / k if object_name else 0.0
    mean_confidence = sum(sample.confidence or 0.5 for sample in usable) / k
    confidence = (0.6 * category_agreement + 0.4 * object_agreement) * mean_confidence

    winner = next(
        (
            sample
            for sample in usable
            if sample.category == category and normalize_tag(sample.object_name) == object_name
        ),
        usable[0],
    )

    threshold = (k + 1) // 2  # ceil(k/2)
    vote_stats = {
        "k_requested": len(samples),
        "k_used": k,
        "category_agreement": round(category_agreement, 4),
        "object_agreement": round(object_agreement, 4),
        "sample_categories": [sample.category for sample in usable],
        "sample_objects": [normalize_tag(sample.object_name) for sample in usable],
    }
    return MetadataSuggestion(
        category=category,
        object_name=object_name or winner.object_name,
        tags=_frequent_tokens((sample.tags for sample in usable), threshold),
        materials=_frequent_tokens((sample.materials for sample in usable), threshold),
        mood=_frequent_tokens((sample.mood for sample in usable), threshold),
        dominant_colors=_frequent_tokens((sample.dominant_colors for sample in usable), threshold),
        short_description=winner.short_description,
        suggested_sprite_id=winner.suggested_sprite_id,
        confidence=confidence,
        uncertainty=winner.uncertainty,
        warnings=(*winner.warnings, *warnings),
        visual_evidence=winner.visual_evidence,
        vote_stats=vote_stats,
        raw_response=winner.raw_response,
    )


def _majority(values: Any) -> tuple[str, int, bool]:
    """Return (winner, count, tied); first-seen order breaks equal counts."""

    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    best = max(order, key=lambda value: counts[value])
    best_count = counts[best]
    tied = sum(1 for value in order if counts[value] == best_count) > 1
    return best, best_count, tied


def _frequent_tokens(groups: Any, threshold: int) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for group in groups:
        for token in group:
            if token not in counts:
                counts[token] = 0
                order.append(token)
            counts[token] += 1
    kept = [token for token in order if counts[token] >= threshold]
    kept.sort(key=lambda token: (-counts[token], order.index(token)))
    return tuple(kept)


def content_bbox(image: Image.Image, *, pad: int = 0) -> tuple[int, int, int, int]:
    """Return the (left, top, right, bottom) box of non-transparent pixels.

    Falls back to the full image when everything is transparent. ``pad``
    expands the box, clamped to the image bounds.
    """

    rgba = image.convert("RGBA")
    box = rgba.getchannel("A").getbbox()
    if box is None:
        return (0, 0, rgba.width, rgba.height)
    left, top, right, bottom = box
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(rgba.width, right + pad),
        min(rgba.height, bottom + pad),
    )


def prepare_vlm_image(
    image: Image.Image,
    upscale: int = 16,
    *,
    background: tuple[int, int, int] = (255, 0, 255),
    crop_to_content: bool = True,
    pad: int = 1,
) -> Image.Image:
    """Crop to content, composite over a solid background, nearest-upscale.

    The solid background (magenta by default) is explicitly described in the
    prompt as not part of the sprite; a checkerboard was previously mistaken
    for sprite content. Cropping keeps small sprites from being dwarfed by
    empty canvas: the scale factor grows so the longest content edge reaches
    roughly ``32 * upscale`` pixels.
    """

    if upscale < 1:
        raise ValueError("upscale must be at least 1.")
    rgba = image.convert("RGBA")
    if crop_to_content:
        rgba = rgba.crop(content_bbox(rgba, pad=pad))
    target = 32 * upscale
    scale = max(upscale, target // max(rgba.width, rgba.height))
    backdrop = Image.new("RGBA", rgba.size, (*background, 255))
    composited = Image.alpha_composite(backdrop, rgba).convert("RGB")
    return composited.resize((rgba.width * scale, rgba.height * scale), resample=Image.Resampling.NEAREST)


def prepare_vlm_image_view(
    image: Image.Image,
    *,
    view: str,
    background: tuple[int, int, int] = (255, 0, 255),
    crop_pad: int = 3,
    full_size: int = 512,
    crop_size: int = 512,
    small_crop_size: int = 768,
) -> Image.Image:
    """Prepare one descriptor image view with solid matte and nearest scaling."""

    view = str(view).strip().lower()
    if view not in {"full", "crop"}:
        raise ValueError(f"view must be full or crop, not {view!r}")
    rgba = image.convert("RGBA")
    if view == "full":
        target_size = int(full_size)
    else:
        raw_bbox = content_bbox(rgba, pad=0)
        raw_width = raw_bbox[2] - raw_bbox[0]
        raw_height = raw_bbox[3] - raw_bbox[1]
        rgba = rgba.crop(content_bbox(rgba, pad=crop_pad))
        target_size = int(small_crop_size if raw_width < 20 or raw_height < 20 else crop_size)
    backdrop = Image.new("RGBA", rgba.size, (*background, 255))
    composited = Image.alpha_composite(backdrop, rgba).convert("RGB")
    return composited.resize((target_size, target_size), resample=Image.Resampling.NEAREST)


def prepare_vlm_image_views(
    image: Image.Image,
    *,
    image_view: str = "both",
    background: tuple[int, int, int] = (255, 0, 255),
    full_size: int = 512,
    crop_size: int = 512,
    small_crop_size: int = 768,
) -> tuple[tuple[str, Image.Image], ...]:
    """Prepare descriptor image views in the order the prompt describes them."""

    image_view = str(image_view).strip().lower() or "both"
    if image_view == "both":
        views = ("full", "crop")
    elif image_view in {"full", "crop"}:
        views = (image_view,)
    else:
        raise ValueError(f"image_view must be full, crop, or both, not {image_view!r}")
    return tuple(
        (
            view,
            prepare_vlm_image_view(
                image,
                view=view,
                background=background,
                full_size=full_size,
                crop_size=crop_size,
                small_crop_size=small_crop_size,
            ),
        )
        for view in views
    )


def image_to_data_url(image: Image.Image) -> str:
    """Return a PNG data URL for an image."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _prepared_images_for_request(config: PrefillConfig, image: Image.Image) -> tuple[tuple[str, Image.Image], ...]:
    if config.vlm_role == "descriptor":
        return prepare_vlm_image_views(
            image,
            image_view=config.vlm_image_view,
            full_size=config.descriptor_full_size,
            crop_size=config.descriptor_crop_size,
            small_crop_size=config.descriptor_small_crop_size,
        )
    return (("legacy_crop", prepare_vlm_image(image, upscale=config.upscale)),)


def _backend_prompt_version(base_prompt_version: str, config: PrefillConfig) -> str:
    if config.vlm_role == "descriptor":
        return (
            f"{base_prompt_version}:role={config.vlm_role}:schema={DESCRIPTOR_SCHEMA_VERSION}:"
            f"prep={VLM_IMAGE_PREP_VERSION}:view={config.vlm_image_view}:full={config.descriptor_full_size}:"
            f"crop={config.descriptor_crop_size}:small={config.descriptor_small_crop_size}:"
            f"structured={config.structured_output}"
        )
    return f"{base_prompt_version}:role={config.vlm_role}:upscale={config.upscale}:img=crop_magenta:structured={config.structured_output}"


def _prompt_for_role(
    config: PrefillConfig,
    filename_suggestion: Mapping[str, Any] | None = None,
    *,
    image_facts: Mapping[str, Any] | None = None,
    retry_note: str = "",
    candidate_object_names: Sequence[str] = (),
) -> str:
    if config.vlm_role == "descriptor":
        return build_vlm_descriptor_prompt(
            filename_suggestion,
            image_facts=image_facts,
            retry_note=retry_note,
            candidate_object_names=candidate_object_names,
            image_view_mode=config.vlm_image_view,
        )
    return build_qwen_prefill_prompt(filename_suggestion, image_facts=image_facts, retry_note=retry_note)


def build_vlm_descriptor_prompt(
    filename_suggestion: Mapping[str, Any] | None = None,
    *,
    image_facts: Mapping[str, Any] | None = None,
    candidate_object_names: Sequence[str] = (),
    image_view_mode: str = "crop",
    retry_note: str = "",
) -> str:
    """Return the descriptor/verifier prompt used by label-v2."""

    facts_block = ""
    if image_facts:
        parts = []
        width = image_facts.get("content_width")
        height = image_facts.get("content_height")
        if width and height:
            parts.append(f"content bbox is {width}x{height} pixels")
        colors = image_facts.get("dominant_colors")
        if colors:
            parts.append(f"computed dominant colors are {', '.join(str(c) for c in colors)}")
        shape = image_facts.get("aspect_hint") or image_facts.get("shape_hints")
        if shape:
            parts.append(f"computed shape hint is {shape}")
        if parts:
            facts_block = "\nDeterministic facts: " + "; ".join(parts) + ".\n"

    candidate_values = tuple(normalize_tag(str(value)) for value in candidate_object_names if normalize_tag(str(value)))
    prefix_family_source = False
    if filename_suggestion:
        filename_trust = normalize_tag(str(filename_suggestion.get("filename_trust", "")))
        profile_name = normalize_tag(str(filename_suggestion.get("source_profile_name", "")))
        prefix_family_source = bool(candidate_values) and (
            filename_trust == "prefix_family" or profile_name == "oga_496_rpg_icons"
        )

    source_block = '\nNo trusted object name is available.\nIdentify cautiously from visible evidence.\nUse uncertainty="cannot_tell" if ambiguous.\nPrefer candidate object names if provided.\n'
    if filename_suggestion:
        confidence = _confidence_or_none(filename_suggestion.get("confidence")) or 0.0
        object_name = normalize_tag(str(filename_suggestion.get("object_name", "")))
        if prefix_family_source:
            source_block = (
                "\nThis source uses compact RPG icon filenames.\n"
                "The filename gives a family hint, not necessarily an exact object label.\n"
                "Candidate object names are provided from the source/profile.\n"
                "Prefer one candidate if visually plausible.\n"
                "Do not invent unrelated objects outside the candidate family unless all candidates are clearly contradicted.\n"
                "If the exact subtype is unclear, keep the broad family candidate and put plausible subtypes in alternative_object_names.\n"
                'If none of the candidates fit, set source_consistency="contradicted" or "unclear" and explain briefly.\n'
                f"Filename/source family hint:\n{json.dumps(dict(filename_suggestion), sort_keys=True)}\n"
            )
        elif confidence >= 0.85 and object_name:
            source_block = (
                f"\nThe source filename/path suggests this sprite is: {object_name!r}.\n"
                "Treat this as strong source metadata.\n"
                "Your job is to describe visible evidence and verify whether the image is consistent with this source label.\n"
                f"If the source metadata is plausible, possible_object_name MUST equal the source object_name: {object_name!r}.\n"
                "Only propose a different possible_object_name if the source is clearly contradicted.\n"
                "If uncertain but plausible, keep the source object and put alternatives in alternative_object_names.\n"
            )
        else:
            source_block = (
                "\nNo trusted object name is available.\n"
                "A weak filename/path hypothesis is available; do not copy it unless the visible sprite supports it:\n"
                f"{json.dumps(dict(filename_suggestion), sort_keys=True)}\n"
                'Identify cautiously from visible evidence. Use uncertainty="cannot_tell" if ambiguous. Prefer candidate object names if provided.\n'
            )
    candidate_block = ""
    if candidate_values:
        candidate_lines = "\n".join(f"- {candidate}" for candidate in candidate_values)
        candidate_block = (
            "\nCandidate object names from this source/profile:\n"
            f"{candidate_lines}\n"
            "\nPrefer one of these candidates if visually plausible.\n"
            "If none fit, use cannot_tell or propose a cautious alternative.\n"
        )
        if prefix_family_source:
            candidate_block += (
                "\nCandidate-specific guidance:\n"
                "- For armor/clothing/helmet/shoes/accessories, do not call them food, mushrooms, animals, or random tools unless visually impossible.\n"
                "- For gold/metal/materials, do not call them shield/weapon unless the source family is clearly contradicted.\n"
                "- For jewelry/accessories, do not call them food or magnifying_glass unless candidates are clearly impossible.\n"
            )
    view_block = ""
    if str(image_view_mode).strip().lower() == "both":
        view_block = (
            "\nThe image input may contain two views of the same sprite:\n"
            "- full canvas view: preserves 32x32 placement\n"
            "- cropped close-up view: makes small details easier to inspect\n"
            "Both views show the same sprite.\n"
        )
    retry_block = f"\nPrevious attempt issue: {retry_note}\nLook again carefully.\n" if retry_note else ""
    category_lines = "\n".join(f"- {name}: {definition}" for name, definition in _CATEGORY_DEFINITIONS)

    return (
        """You are a visual verifier for tiny 32x32 pixel-art sprites.
You are not the final label authority.
You describe visible evidence and verify whether source metadata is visually plausible.

The image shows one pixel-art sprite, nearest-neighbor upscaled. The solid magenta or neutral background/matte was added for display and is not sprite content. Do not mention the magenta background.
"""
        + view_block
        + facts_block
        + source_block
        + candidate_block
        + retry_block
        + f"""
Allowed category values:
{category_lines}

Known tiny-sprite traps:
- Yellow rectangular food may be butter, cheese, corn, or lemon; do not call it gold/coin/currency unless there are clear coin/metal markings.
- Round fruit or cheese may look like coins; do not call them coins unless there are explicit currency details.
- Drinks, soda cans, mugs, milk cartons, juice, and bottles may look like potions; do not call them potions unless fantasy/alchemy source metadata supports it.
- Simple dark round food may look like an orb; prefer the source food label if plausible.
- If the source profile is food/tool/gem and the source object is plausible, do not replace it with a fantasy RPG object.

Return strict JSON only with exactly these fields:
{{
  "visual_description": "<one factual sentence about visible color/shape/object cues>",
  "visual_tags": ["<short snake_case visual tags such as yellow, rectangular, roundish>"],
  "source_consistency": "<consistent | unclear | contradicted | no_source>",
  "evidence_for_source": ["<visible evidence supporting source label>"],
  "evidence_against_source": ["<visible evidence contradicting source label>"],
  "possible_object_name": "<snake_case object name, empty if cannot tell>",
  "alternative_object_names": ["<0 to 3 alternatives, most likely first>"],
  "possible_category": "<one allowed category value>",
  "uncertainty": "<confident | likely | unsure | cannot_tell>",
  "warnings": ["<real issues only; usually empty>"]
}}

Rules:
- If strong source metadata is present and visually plausible, possible_object_name MUST equal the source object name.
- If strong source metadata is present, only use a different possible_object_name when the source is clearly contradicted.
- If the source is ambiguous but plausible, use source_consistency="unclear", keep the source object name, and put alternatives in alternative_object_names.
- If no trusted source exists, identify cautiously and use cannot_tell when needed.
- source_consistency must be consistent, unclear, contradicted, or no_source.
- Do not mention the magenta background.
- Do not estimate dominant colors if deterministic colors were provided; use them as facts.
- Do not describe the upscaled pixel dimensions as original sprite size.
- If uncertainty is cannot_tell, possible_category must be "unknown" and possible_object_name must be ""."""
    )


def build_qwen_prefill_prompt(
    filename_suggestion: Mapping[str, Any] | None = None,
    *,
    image_facts: Mapping[str, Any] | None = None,
    retry_note: str = "",
) -> str:
    """Return the strict JSON prompt used for VLM metadata suggestions.

    Deliberately contains no filled-in example: a concrete example JSON was
    measurably regurgitated verbatim by the model.
    """

    facts_block = ""
    if image_facts:
        parts = []
        width = image_facts.get("content_width")
        height = image_facts.get("content_height")
        if width and height:
            parts.append(f"the sprite's actual content is {width}x{height} pixels")
        palette_size = image_facts.get("opaque_palette_size")
        if palette_size:
            parts.append(f"it uses {palette_size} opaque colors")
        colors = image_facts.get("dominant_colors")
        if colors:
            parts.append(f"its dominant colors (computed, trust these) are: {', '.join(str(c) for c in colors)}")
        if parts:
            facts_block = "\nKnown facts about this sprite: " + "; ".join(parts) + ".\n"

    filename_block = ""
    if filename_suggestion:
        filename_block = (
            "\nA deterministic filename parser suggested the following. It may be wrong; "
            "confirm or reject it from the image instead of copying it:\n"
            f"{json.dumps(dict(filename_suggestion), sort_keys=True)}\n"
        )
    retry_block = (
        f"\nPrevious attempt issue: {retry_note}\nLook again carefully and answer from the image.\n"
        if retry_note
        else ""
    )

    category_lines = "\n".join(f"- {name}: {definition}" for name, definition in _CATEGORY_DEFINITIONS)

    return (
        """You label tiny pixel-art sprites for a machine-learning dataset.

The image shows one pixel-art sprite, nearest-neighbor upscaled. The solid magenta background was added for display and is NOT part of the sprite. Never describe or mention the background.
"""
        + facts_block
        + filename_block
        + retry_block
        + f"""
Identify the object and describe only what is visible.
Do not invent source, author, license, or copyright status.
Do not decide whether the sprite is valid.

Allowed category values (pick exactly one):
{category_lines}

Respond with a single JSON object of exactly this shape, replacing the <placeholders>:
{{
  "category": "<one allowed category value>",
  "object_name": "<snake_case name of the depicted object>",
  "tags": ["<up to 8 short snake_case tags>"],
  "materials": ["<up to 4 visible materials such as wood, metal, glass>"],
  "mood": ["<up to 3 mood words, only if clearly conveyed>"],
  "short_description": "<one factual sentence describing the sprite>",
  "suggested_sprite_id": "<snake_case identifier for this sprite>",
  "uncertainty": "<confident | likely | unsure | cannot_tell>",
  "visual_evidence": ["<up to 5 short cues you can actually see>"],
  "warnings": ["<real problems only; usually an empty list>"]
}}

Rules:
- Return strict JSON only, no prose around it.
- Use snake_case for object_name, tags, and suggested_sprite_id.
- Prefer concrete object tags over generic style tags. Never use the tag pixel_art.
- uncertainty meaning: confident = object clearly identifiable; likely = probably right; unsure = a guess; cannot_tell = shape not identifiable.
- If uncertainty is cannot_tell, category must be "unknown" and object_name must be "".
- Do not include license, author, source, train split, or accept/reject status."""
    )


def build_adjudication_prompt(
    image_facts: Mapping[str, Any] | None,
    *,
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
) -> str:
    """Forced choice between two candidate labels for the shown sprite.

    Candidates are presented anonymously (no provenance) so the model judges
    them purely against the image.
    """

    facts_line = ""
    if image_facts:
        colors = image_facts.get("dominant_colors")
        parts = []
        width = image_facts.get("content_width")
        height = image_facts.get("content_height")
        if width and height:
            parts.append(f"content is {width}x{height} pixels")
        if colors:
            parts.append(f"dominant colors are {', '.join(str(c) for c in colors)}")
        if parts:
            facts_line = "Known facts: " + "; ".join(parts) + ".\n"

    return (
        """You verify labels for tiny pixel-art sprites.

The image shows one pixel-art sprite, nearest-neighbor upscaled. The solid magenta background was added for display and is NOT part of the sprite.
"""
        + facts_line
        + f"""
Two candidate labels were proposed for this sprite:
Candidate A: {json.dumps(_compact_candidate(candidate_a), sort_keys=True)}
Candidate B: {json.dumps(_compact_candidate(candidate_b), sort_keys=True)}

Look at the image and decide which candidate matches what is actually visible.

Respond with a single JSON object of exactly this shape, replacing the <placeholders>:
{{
  "choice": "<a | b | both_wrong | cannot_tell>",
  "corrected_category": "<only when both_wrong: the correct category, else empty string>",
  "corrected_object_name": "<only when both_wrong: the correct snake_case object name, else empty string>",
  "reason": "<one short sentence grounded in what you see>"
}}

Rules:
- Return strict JSON only.
- Pick "a" or "b" only if that candidate matches the visible object.
- Pick "both_wrong" if you can identify the object but neither candidate matches.
- Pick "cannot_tell" if the sprite is too ambiguous to decide."""
    )


def parse_adjudication_result(text: str) -> AdjudicationResult:
    """Parse the forced-choice JSON answer into an AdjudicationResult."""

    raw = str(text)
    candidate = _extract_json_candidate(raw)
    if candidate is None:
        return AdjudicationResult(
            warnings=("invalid JSON from adjudication: expected a JSON object.",), raw_response=raw
        )
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return AdjudicationResult(warnings=(f"invalid JSON from adjudication: {exc}",), raw_response=raw)
    if not isinstance(data, dict):
        return AdjudicationResult(warnings=("invalid JSON from adjudication: expected an object.",), raw_response=raw)
    return AdjudicationResult(
        choice=str(data.get("choice", "")),
        corrected_category=str(data.get("corrected_category", "")),
        corrected_object_name=str(data.get("corrected_object_name", "")),
        reason=str(data.get("reason", "")),
        raw_response=raw,
    )


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": normalize_category(str(candidate.get("category", "unknown"))),
        "object_name": normalize_tag(str(candidate.get("object_name", ""))),
        "tags": [normalize_tag(str(tag)) for tag in (candidate.get("tags") or ())][:8],
        "short_description": str(candidate.get("short_description", "")).strip()[:160],
    }


def _read_cached_adjudication(path: Path) -> AdjudicationResult | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result_data = data.get("adjudication")
        if not isinstance(result_data, dict):
            return None
        return AdjudicationResult(
            choice=str(result_data.get("choice", "")),
            corrected_category=str(result_data.get("corrected_category", "")),
            corrected_object_name=str(result_data.get("corrected_object_name", "")),
            reason=str(result_data.get("reason", "")),
            warnings=_warning_tuple(result_data.get("warnings")),
            raw_response=str(data.get("raw_response", "")),
        )
    except Exception:
        return None


def parse_metadata_suggestion(text: str) -> MetadataSuggestion:
    """Parse and normalize strict JSON model output into a suggestion."""

    raw = str(text)
    candidate = _extract_json_candidate(raw)
    if candidate is None:
        return _warning_suggestion("invalid JSON from model: expected a JSON object.", raw_response=raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return _warning_suggestion(f"invalid JSON from model: {exc}", raw_response=raw)
    if not isinstance(data, dict):
        return _warning_suggestion(
            "invalid JSON from model: expected an object, not an array or scalar.", raw_response=raw
        )

    warnings = _warning_tuple(data.get("warnings"))
    category = normalize_category(str(data.get("category", "unknown")))
    if category not in ALLOWED_CATEGORIES:
        warnings = (*warnings, f"Unknown category {category!r}; normalized to 'unknown'.")
        category = "unknown"

    uncertainty = normalize_tag(str(data.get("uncertainty", "")))
    if uncertainty and uncertainty not in UNCERTAINTY_LEVELS:
        warnings = (*warnings, f"Unknown uncertainty {uncertainty!r}; treated as 'unsure'.")
        uncertainty = "unsure"
    confidence = _confidence_or_none(data.get("confidence"))
    if confidence is None and uncertainty:
        confidence = _UNCERTAINTY_TO_CONFIDENCE[uncertainty]
    suggested_sprite_id = normalize_sprite_id(str(data.get("suggested_sprite_id", "")))
    if data.get("suggested_sprite_id") and not suggested_sprite_id:
        warnings = (*warnings, "Suggested sprite_id was empty or unsafe after normalization.")

    return MetadataSuggestion(
        category=category,
        object_name=_string_value(data.get("object_name")),
        tags=_string_tuple(data.get("tags")),
        materials=_string_tuple(data.get("materials")),
        mood=_string_tuple(data.get("mood")),
        dominant_colors=_string_tuple(data.get("dominant_colors")),
        short_description=_string_value(data.get("short_description")),
        suggested_sprite_id=suggested_sprite_id,
        confidence=confidence,
        uncertainty=uncertainty,
        warnings=warnings,
        filename_agreement=_string_value(data.get("filename_agreement")),
        visual_evidence=_string_tuple(data.get("visual_evidence")),
        disagreement_reason=_string_value(data.get("disagreement_reason")),
        raw_response=raw,
    )


def parse_descriptor_suggestion(text: str) -> MetadataSuggestion:
    """Parse descriptor-mode JSON into the compatibility MetadataSuggestion shape."""

    raw = str(text)
    candidate = _extract_json_candidate(raw)
    if candidate is None:
        return _warning_suggestion("invalid JSON from descriptor: expected a JSON object.", raw_response=raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return _warning_suggestion(f"invalid JSON from descriptor: {exc}", raw_response=raw)
    if not isinstance(data, dict):
        return _warning_suggestion(
            "invalid JSON from descriptor: expected an object, not an array or scalar.", raw_response=raw
        )

    warnings = _warning_tuple(data.get("warnings"))[:5]
    source_consistency = _normalize_source_consistency(
        str(data.get("source_consistency") or data.get("agrees_with_source") or "")
    )
    if source_consistency == "contradicted":
        warnings = (*warnings, "vlm_conflicts_with_trusted_filename")
    uncertainty = normalize_tag(str(data.get("uncertainty", "")))
    if uncertainty not in UNCERTAINTY_LEVELS:
        warnings = (*warnings, f"Unknown uncertainty {uncertainty!r}; treated as 'unsure'.")
        uncertainty = "unsure"
    confidence = {
        "confident": 0.85,
        "likely": 0.65,
        "unsure": 0.4,
        "cannot_tell": 0.2,
    }.get(uncertainty, 0.0)
    category = normalize_category(str(data.get("possible_category", "unknown")))
    object_name = normalize_tag(_string_value(data.get("possible_object_name")))
    if uncertainty == "cannot_tell":
        category = "unknown"
        object_name = ""
    alternatives = _string_tuple(data.get("alternative_object_names"))[:3]
    evidence_for = _clean_string_tuple(data.get("evidence_for_source"), max_items=5)
    evidence_against = _clean_string_tuple(data.get("evidence_against_source"), max_items=5)
    contradiction_reason = _string_value(data.get("contradiction_reason"))
    if contradiction_reason and not evidence_against:
        evidence_against = (contradiction_reason,)
    return MetadataSuggestion(
        category=category,
        object_name=object_name,
        tags=_string_tuple(data.get("visual_tags")),
        short_description=_string_value(data.get("visual_description")),
        confidence=confidence,
        uncertainty=uncertainty,
        warnings=warnings,
        filename_agreement=source_consistency,
        visual_evidence=_string_tuple(data.get("visual_tags")),
        disagreement_reason=contradiction_reason,
        source_consistency=source_consistency,
        alternative_object_names=alternatives,
        evidence_for_source=evidence_for,
        evidence_against_source=evidence_against,
        raw_response=raw,
    )


def compute_image_cache_key(
    image: Image.Image,
    *,
    model: str,
    prompt_version: str,
    context: str = "",
    image_prep_version: str = VLM_IMAGE_PREP_VERSION,
    image_view: str = "legacy_crop",
    target_size: int = 512,
) -> str:
    """Compute a stable SHA256 key for a prepared image/model/prompt."""

    if image_view in {"full", "crop", "both"}:
        prepared_images = prepare_vlm_image_views(image, image_view=image_view)
    else:
        prepared_images = (("legacy_crop", prepare_vlm_image(image, upscale=16)),)
    digest = hashlib.sha256()
    for view_name, prepared in prepared_images:
        digest.update(str(view_name).encode("utf-8"))
        digest.update(_png_bytes(prepared))
    digest.update(str(model).encode("utf-8"))
    digest.update(str(prompt_version).encode("utf-8"))
    digest.update(str(image_prep_version).encode("utf-8"))
    digest.update(str(image_view).encode("utf-8"))
    digest.update(str(target_size).encode("utf-8"))
    digest.update(str(context).encode("utf-8"))
    return digest.hexdigest()


def create_prefill_backend(config: PrefillConfig) -> MetadataPrefillBackend:
    """Create a configured metadata prefill backend."""

    if not config.enabled or config.backend == "none":
        return NoopPrefillBackend()
    if config.backend == "rule_based":
        backend: MetadataPrefillBackend = RuleBasedPrefillBackend()
    elif config.backend == "openai_compatible":
        backend = OpenAICompatibleQwenPrefillBackend(config)
    elif config.backend == "ollama":
        backend = OllamaQwenPrefillBackend(config)
    else:
        raise ValueError(f"unknown prefill backend: {config.backend}")
    if config.cache_dir is not None:
        backend = CachedPrefillBackend(backend, config.cache_dir)
    # Voting sits outside the cache so each sample is cached individually and
    # a rerun resumes mid-vote instead of repeating finished samples.
    if config.backend in {"openai_compatible", "ollama"} and config.votes > 1 and config.vote_mode != "off":
        backend = SelfConsistencyBackend(backend, config)
    return backend


def apply_suggestion_to_item(
    item: DatasetMakerItem,
    suggestion: MetadataSuggestion,
    *,
    overwrite_existing: bool = False,
) -> DatasetMakerItem:
    """Return a copy of an item with safe model-suggested metadata applied."""

    category = item.category
    if suggestion.category and (item.category == "unknown" or overwrite_existing):
        category = suggestion.category

    tags = _merge_tags(
        item.tags,
        suggestion.tags,
        (suggestion.object_name,),
        suggestion.materials,
        suggestion.mood,
        suggestion.dominant_colors,
    )

    sprite_id = item.sprite_id
    if suggestion.suggested_sprite_id and (overwrite_existing or _is_generated_sprite_id(item)):
        sprite_id = suggestion.suggested_sprite_id

    notes = item.notes
    if suggestion.short_description and not item.notes.strip():
        notes = suggestion.short_description

    return DatasetMakerItem(
        sprite_id=sprite_id,
        source_path=item.source_path,
        status=item.status,
        category=category,
        tags=tags,
        notes=notes,
        source_name=item.source_name,
        license=item.license,
        author=item.author,
        split=item.split,
        quality_issues=item.quality_issues,
        palette_size=item.palette_size,
        has_role_map=item.has_role_map,
    )


def suggestion_to_json_dict(suggestion: MetadataSuggestion, *, include_raw: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable suggestion dictionary."""

    return _suggestion_to_dict(suggestion, include_raw=include_raw)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_base64_png(image: Image.Image) -> str:
    return base64.b64encode(_png_bytes(image)).decode("ascii")


def _extract_json_candidate(text: str) -> str | None:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return None


def _extract_response_text(data: Mapping[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def _extract_ollama_response_text(data: Mapping[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = data.get("response")
    if isinstance(response, str):
        return response
    return ""


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:500]
    except Exception:
        return str(exc)


def _format_url_error(exc: urllib.error.URLError, endpoint: str) -> str:
    reason = exc.reason
    reason_text = str(reason)
    lower = reason_text.lower()
    if "connection refused" in lower or "actively refused" in lower:
        return (
            f"prefill request could not connect to {endpoint}. "
            "Start your local OpenAI-compatible Qwen server, check the base URL, or use the rule_based backend."
        )
    if "name or service not known" in lower or "getaddrinfo" in lower:
        return f"prefill request could not resolve {endpoint}. Check the base URL host name."
    return f"prefill request failed for {endpoint}: {reason_text}"


def _format_ollama_url_error(exc: urllib.error.URLError, endpoint: str) -> str:
    reason = exc.reason
    reason_text = str(reason)
    lower = reason_text.lower()
    if "connection refused" in lower or "actively refused" in lower:
        return (
            f"ollama prefill could not connect to {endpoint}. "
            "On Windows, start Ollama, run `ollama serve` if needed, and pull a vision model such as `ollama pull qwen2.5vl:7b`."
        )
    return f"ollama prefill request failed for {endpoint}: {reason_text}"


def _bearer_token(config: PrefillConfig) -> str:
    if config.runpod_token:
        return config.runpod_token
    if config.api_key and config.api_key != "not-needed":
        return config.api_key
    return os.environ.get("RUNPOD_API_KEY", "").strip() or os.environ.get("RUNPOD_TOKEN", "").strip()


def _ollama_base_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value or value == "http://127.0.0.1:8000/v1":
        return "http://127.0.0.1:11434"
    if value.endswith("/api/chat"):
        return value[: -len("/api/chat")]
    return value


def is_degenerate_suggestion(suggestion: MetadataSuggestion) -> bool:
    """Detect answers that describe the display background, echo boilerplate,
    or pair an unusable object with high confidence."""

    text = " ".join(
        (
            suggestion.object_name,
            suggestion.short_description,
            suggestion.suggested_sprite_id,
            *suggestion.tags,
        )
    ).lower()
    if any(pattern in text for pattern in _DEGENERATE_TEXT_PATTERNS):
        return True

    high_confidence = suggestion.confidence is not None and suggestion.confidence >= 0.85
    object_token = normalize_tag(suggestion.object_name)
    if high_confidence and suggestion.category == "unknown" and suggestion.uncertainty != "cannot_tell":
        return True
    if high_confidence and object_token and object_token in _DEGENERATE_OBJECTS and suggestion.category != "unknown":
        return True
    return False


def flag_degenerate_suggestion(suggestion: MetadataSuggestion) -> MetadataSuggestion:
    """Append the degenerate warning so retry/cache logic can react."""

    if not is_degenerate_suggestion(suggestion) or _has_degenerate_warning(suggestion):
        return suggestion
    return replace_suggestion_warnings(
        suggestion,
        (*suggestion.warnings, f"{DEGENERATE_WARNING_PREFIX}: background/boilerplate answer or unusable object."),
    )


def replace_suggestion_warnings(suggestion: MetadataSuggestion, warnings: tuple[str, ...]) -> MetadataSuggestion:
    return MetadataSuggestion(
        category=suggestion.category,
        object_name=suggestion.object_name,
        tags=suggestion.tags,
        materials=suggestion.materials,
        mood=suggestion.mood,
        dominant_colors=suggestion.dominant_colors,
        short_description=suggestion.short_description,
        suggested_sprite_id=suggestion.suggested_sprite_id,
        confidence=suggestion.confidence,
        uncertainty=suggestion.uncertainty,
        warnings=warnings,
        filename_agreement=suggestion.filename_agreement,
        visual_evidence=suggestion.visual_evidence,
        disagreement_reason=suggestion.disagreement_reason,
        source_consistency=suggestion.source_consistency,
        alternative_object_names=suggestion.alternative_object_names,
        evidence_for_source=suggestion.evidence_for_source,
        evidence_against_source=suggestion.evidence_against_source,
        raw_response=suggestion.raw_response,
    )


def _has_degenerate_warning(suggestion: MetadataSuggestion) -> bool:
    return any(warning.startswith(DEGENERATE_WARNING_PREFIX) for warning in suggestion.warnings)


def _warning_suggestion(message: str, *, raw_response: str = "") -> MetadataSuggestion:
    return MetadataSuggestion(warnings=(message,), raw_response=raw_response)


def _should_retry_suggestion(suggestion: MetadataSuggestion, config: PrefillConfig) -> bool:
    if _has_degenerate_warning(suggestion):
        return True
    if config.retry_on_warning_only and suggestion.warnings and not _suggestion_has_content(suggestion):
        return True
    if suggestion.confidence is not None and suggestion.confidence < config.min_qwen_confidence:
        return True
    return False


def _retry_note(suggestion: MetadataSuggestion) -> str:
    if suggestion.warnings:
        return "; ".join(suggestion.warnings)
    if suggestion.confidence is not None:
        return f"confidence {suggestion.confidence:.2f} was below the requested threshold."
    return "response had no usable metadata."


def _with_retry_count(suggestion: MetadataSuggestion, attempts_used: int) -> MetadataSuggestion:
    if attempts_used <= 0:
        return suggestion
    return replace_suggestion_warnings(
        suggestion,
        (*suggestion.warnings, f"Retried {attempts_used} time(s) during prefill."),
    )


def _string_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _normalize_sequence((value,))
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return _normalize_sequence(tuple(str(item) for item in value))


def _clean_string_tuple(value: object, *, max_items: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        values = (str(value),)
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        result.append(text[:160])
        if len(result) >= max_items:
            break
    return tuple(result)


def _normalize_sequence(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = normalize_tag(str(value))
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def _normalize_source_consistency(value: object) -> str:
    normalized = normalize_tag(str(value))
    mapped = {
        "yes": "consistent",
        "true": "consistent",
        "consistent": "consistent",
        "no": "contradicted",
        "false": "contradicted",
        "contradicted": "contradicted",
        "conflict": "contradicted",
        "unclear": "unclear",
        "maybe": "unclear",
        "unknown": "unclear",
        "": "unclear",
        "no_source": "no_source",
        "none": "no_source",
    }
    return mapped.get(normalized, "unclear")


def _confidence_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _tokens_from_text(value: str) -> tuple[str, ...]:
    return tuple(token for token in normalize_sprite_id(value).split("_") if token)


def _category_from_tokens(tokens: Sequence[str]) -> str:
    token_set = set(tokens)
    if token_set & {"ui", "button", "cursor", "menu"}:
        return "ui_icon"
    if token_set & {"weapon", "sword", "axe", "bow", "dagger"}:
        return "weapon"
    if token_set & {"tool", "pickaxe", "hammer", "shovel"}:
        return "tool"
    if token_set & {"armor", "helmet", "chestplate", "boots"}:
        return "armor"
    if token_set & {"block", "brick", "tile", "ore"}:
        return "block"
    if token_set & {"plant", "mushroom", "flower", "leaf", "tree"}:
        return "plant"
    if token_set & {"entity", "mob", "creature", "slime"}:
        return "entity"
    if token_set & {"material", "ingot", "dust", "gem"}:
        return "material"
    if token_set & {"effect", "spell", "buff"}:
        return "effect_icon"
    if token_set & {"prop", "rock", "barrel", "crate"}:
        return "environment_prop"
    if token_set & {"item", "icon", "vial", "crystal", "potion"}:
        return "item_icon"
    return "unknown"


def _merge_tags(*groups: Sequence[str]) -> tuple[str, ...]:
    return _normalize_sequence(tuple(value for group in groups for value in group))


def _is_generated_sprite_id(item: DatasetMakerItem) -> bool:
    return not item.sprite_id or item.sprite_id == normalize_sprite_id(Path(item.source_path).stem)


def _suggestion_to_dict(suggestion: MetadataSuggestion, *, include_raw: bool) -> dict[str, Any]:
    data = {
        "category": suggestion.category,
        "object_name": suggestion.object_name,
        "tags": list(suggestion.tags),
        "materials": list(suggestion.materials),
        "mood": list(suggestion.mood),
        "dominant_colors": list(suggestion.dominant_colors),
        "short_description": suggestion.short_description,
        "suggested_sprite_id": suggestion.suggested_sprite_id,
        "confidence": suggestion.confidence,
        "uncertainty": suggestion.uncertainty,
        "warnings": list(suggestion.warnings),
        "filename_agreement": suggestion.filename_agreement,
        "visual_evidence": list(suggestion.visual_evidence),
        "disagreement_reason": suggestion.disagreement_reason,
        "source_consistency": suggestion.source_consistency,
        "alternative_object_names": list(suggestion.alternative_object_names),
        "evidence_for_source": list(suggestion.evidence_for_source),
        "evidence_against_source": list(suggestion.evidence_against_source),
    }
    if suggestion.vote_stats is not None:
        data["vote_stats"] = dict(suggestion.vote_stats)
    if include_raw:
        data["raw_response"] = suggestion.raw_response
    return data


def _suggestion_has_content(suggestion: MetadataSuggestion) -> bool:
    # Transport wrappers and raw model text are not usable metadata. This keeps
    # invalid JSON and empty-message responses retryable instead of treating the
    # raw response envelope as a successful suggestion.
    return any(
        [
            suggestion.category != "unknown",
            bool(suggestion.object_name),
            bool(suggestion.tags),
            bool(suggestion.materials),
            bool(suggestion.mood),
            bool(suggestion.dominant_colors),
            bool(suggestion.short_description),
            bool(suggestion.suggested_sprite_id),
            suggestion.confidence is not None,
        ]
    )


def _read_cached_suggestion(path: Path) -> MetadataSuggestion | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        suggestion_data = data.get("suggestion", data)
        if not isinstance(suggestion_data, dict):
            return None
        return MetadataSuggestion(
            category=str(suggestion_data.get("category", "unknown")),
            object_name=str(suggestion_data.get("object_name", "")),
            tags=_string_tuple(suggestion_data.get("tags")),
            materials=_string_tuple(suggestion_data.get("materials")),
            mood=_string_tuple(suggestion_data.get("mood")),
            dominant_colors=_string_tuple(suggestion_data.get("dominant_colors")),
            short_description=str(suggestion_data.get("short_description", "")),
            suggested_sprite_id=str(suggestion_data.get("suggested_sprite_id", "")),
            confidence=suggestion_data.get("confidence"),
            uncertainty=str(suggestion_data.get("uncertainty", "")),
            warnings=_warning_tuple(suggestion_data.get("warnings")),
            filename_agreement=str(suggestion_data.get("filename_agreement", "")),
            visual_evidence=_string_tuple(suggestion_data.get("visual_evidence")),
            disagreement_reason=str(suggestion_data.get("disagreement_reason", "")),
            source_consistency=str(
                suggestion_data.get("source_consistency", suggestion_data.get("agrees_with_source", ""))
            ),
            alternative_object_names=_string_tuple(suggestion_data.get("alternative_object_names"))[:3],
            evidence_for_source=_clean_string_tuple(suggestion_data.get("evidence_for_source"), max_items=5),
            evidence_against_source=_clean_string_tuple(suggestion_data.get("evidence_against_source"), max_items=5),
            raw_response=str(suggestion_data.get("raw_response", data.get("raw_response", ""))),
        )
    except Exception:
        return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _warning_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())
