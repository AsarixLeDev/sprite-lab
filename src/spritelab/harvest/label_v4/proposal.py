"""Rich, blind, open-vocabulary VLM proposal artifacts for Labeling v4.

The visual model proposes; it never decides.  Its output contains no model
self-confidence field and is kept separate from calibrated label risk.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spritelab.harvest.label_v4.semantic_axes import ColorAttributes, normalize_semantic_term

VLM_PROPOSAL_SCHEMA_VERSION = "vlm_proposal_v4.2"
BLIND_VLM_PROMPT_VERSION = "blind_vlm_proposal_v4.2"

GENERIC_VISUAL_FORMS: frozenset[str] = frozenset(
    {
        "cylinder",
        "elongated_cylinder",
        "elongated_form",
        "elongated_object",
        "orb",
        "rectangle",
        "rectangular_form",
        "rod",
        "round_object",
        "stick",
        "stick_like_object",
        "stick_like_weapon",
    }
)


def is_generic_visual_form(value: Any) -> bool:
    return normalize_semantic_term(value) in GENERIC_VISUAL_FORMS


class ProposalValidationError(ValueError):
    """Raised when a response is JSON but violates the blind proposal contract."""


@dataclass(frozen=True)
class ObjectCandidate:
    value: str
    visual_support: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", normalize_semantic_term(self.value))
        object.__setattr__(
            self,
            "visual_support",
            tuple(str(value).strip() for value in self.visual_support if str(value).strip()),
        )
        if not self.value or self.value == "unknown":
            raise ProposalValidationError("object candidates must contain a supported non-unknown value")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "visual_support": list(self.visual_support)}

    @classmethod
    def from_value(cls, value: Any) -> ObjectCandidate:
        if isinstance(value, Mapping):
            return cls(str(value.get("value", "")), tuple(value.get("visual_support") or ()))
        return cls(str(value), ())


@dataclass(frozen=True)
class ProposalShape:
    """Open-vocabulary visual geometry from the blind proposal stage.

    Controlled shape mapping happens during reconciliation.  This proposal
    contract performs only bounded structural normalization: a provider may
    return either one string or a list of strings for each declared axis.
    """

    silhouette: tuple[str, ...] = ()
    aspect: tuple[str, ...] = ()
    orientation: tuple[str, ...] = ()
    structure: tuple[str, ...] = ()
    edge_profile: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _string_tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalShape:
        return cls(**{name: _string_tuple(data.get(name)) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BlindVLMProposal:
    """Parsed high-freedom visual proposal; all fields are non-authoritative."""

    schema_version: str = VLM_PROPOSAL_SCHEMA_VERSION
    object_candidates: tuple[ObjectCandidate, ...] = ()
    visual_form: tuple[str, ...] = ()
    category_candidates: tuple[str, ...] = ()
    surface_alias_candidates: tuple[str, ...] = ()
    role_candidates: tuple[str, ...] = ()
    shape: ProposalShape = field(default_factory=ProposalShape)
    color_roles: ColorAttributes = field(default_factory=ColorAttributes)
    raw_visual_color_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    material_visual_cues: tuple[str, ...] = ()
    description_candidates: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    alternative_interpretations: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "visual_form",
            "category_candidates",
            "role_candidates",
            "material_visual_cues",
            "unsupported_fields",
        ):
            values = _unique(normalize_semantic_term(value) for value in getattr(self, name))
            object.__setattr__(self, name, tuple(value for value in values if value and value != "unknown"))
        object.__setattr__(
            self,
            "raw_visual_color_roles",
            {
                str(role): _raw_string_tuple(values)
                for role, values in self.raw_visual_color_roles.items()
                if _raw_string_tuple(values)
            },
        )
        for name in (
            "surface_alias_candidates",
            "description_candidates",
            "uncertainties",
            "alternative_interpretations",
        ):
            values = _unique(str(value).strip() for value in getattr(self, name))
            object.__setattr__(self, name, tuple(value for value in values if value))

        ambiguity_words = ("ambiguous", "could be", "either", "unclear identity", "uncertain identity")
        genuinely_ambiguous = any(
            any(word in uncertainty.lower() for word in ambiguity_words) for uncertainty in self.uncertainties
        )
        if genuinely_ambiguous and not self.alternative_interpretations:
            raise ProposalValidationError("ambiguous visual proposals must preserve an alternative interpretation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_object_candidates": [candidate.to_dict() for candidate in self.object_candidates],
            "visual_form": list(self.visual_form),
            "category_candidates": list(self.category_candidates),
            "surface_alias_candidates": list(self.surface_alias_candidates),
            "role_candidates": list(self.role_candidates),
            "shape": self.shape.to_dict(),
            "color_roles": {
                "primary": list(self.color_roles.primary_colors),
                "secondary": list(self.color_roles.secondary_colors),
                "outline": list(self.color_roles.outline_colors),
                "shadow": list(self.color_roles.shadow_colors),
                "highlight": list(self.color_roles.highlight_colors),
            },
            "raw_visual_color_roles": {role: list(values) for role, values in self.raw_visual_color_roles.items()},
            "material_visual_cues": list(self.material_visual_cues),
            "description_candidates": list(self.description_candidates),
            "uncertainties": list(self.uncertainties),
            "alternative_interpretations": list(self.alternative_interpretations),
            "unsupported_fields": list(self.unsupported_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BlindVLMProposal:
        _reject_self_confidence(data)
        raw_shape = data.get("shape") if isinstance(data.get("shape"), Mapping) else {}
        raw_colors = data.get("color_roles") if isinstance(data.get("color_roles"), Mapping) else {}
        object_candidates: list[ObjectCandidate] = []
        raw_candidates = data.get("canonical_object_candidates")
        if raw_candidates is None:
            # Read-only compatibility for frozen v4.1 artifacts. New provider
            # requests and normalized outputs use canonical_object exclusively.
            raw_candidates = data.get("object_candidates") or ()
        # Qwen sometimes emits one candidate object whose ``value`` is a list
        # and whose visual support applies to every item.  Explode only this
        # explicit schema variant; do not guess at arbitrary nested shapes.
        if isinstance(raw_candidates, Mapping):
            candidate_values = raw_candidates.get("value")
            shared_support = _string_tuple(raw_candidates.get("visual_support"))
            if isinstance(candidate_values, (list, tuple)):
                raw_candidates = tuple(
                    {"value": candidate_value, "visual_support": shared_support} for candidate_value in candidate_values
                )
            elif candidate_values not in (None, ""):
                raw_candidates = ({"value": candidate_values, "visual_support": shared_support},)
            else:
                raise ProposalValidationError("canonical_object_candidates mapping must contain value")
        elif isinstance(raw_candidates, str):
            raw_candidates = (raw_candidates,)
        elif not isinstance(raw_candidates, (list, tuple)):
            raise ProposalValidationError("canonical_object_candidates must be a list or candidate mapping")
        for raw in raw_candidates:
            if (isinstance(raw, str) and normalize_semantic_term(raw) == "unknown") or (
                isinstance(raw, Mapping) and normalize_semantic_term(raw.get("value")) == "unknown"
            ):
                continue
            object_candidates.append(ObjectCandidate.from_value(raw))
        explicit_visual_form = _string_tuple(data.get("visual_form", data.get("visual_forms")))
        derived_visual_form = tuple(
            candidate.value for candidate in object_candidates if is_generic_visual_form(candidate.value)
        )
        raw_visual_colors = (
            data.get("raw_visual_color_roles")
            if isinstance(data.get("raw_visual_color_roles"), Mapping)
            else raw_colors
        )
        return cls(
            schema_version=str(data.get("schema_version") or VLM_PROPOSAL_SCHEMA_VERSION),
            object_candidates=tuple(object_candidates),
            visual_form=tuple(_unique((*explicit_visual_form, *derived_visual_form))),
            category_candidates=_string_tuple(data.get("category_candidates")),
            surface_alias_candidates=_string_tuple(data.get("surface_alias_candidates")),
            role_candidates=_string_tuple(data.get("role_candidates")),
            shape=ProposalShape.from_dict(raw_shape),
            color_roles=ColorAttributes(
                primary_colors=_string_tuple(raw_colors.get("primary", raw_colors.get("primary_colors"))),
                secondary_colors=_string_tuple(raw_colors.get("secondary", raw_colors.get("secondary_colors"))),
                outline_colors=_string_tuple(raw_colors.get("outline", raw_colors.get("outline_colors"))),
                shadow_colors=_string_tuple(raw_colors.get("shadow", raw_colors.get("shadow_colors"))),
                highlight_colors=_string_tuple(raw_colors.get("highlight", raw_colors.get("highlight_colors"))),
            ),
            raw_visual_color_roles={
                _canonical_color_role(str(role)): _raw_string_tuple(values)
                for role, values in dict(raw_visual_colors or {}).items()
                if _canonical_color_role(str(role))
            },
            material_visual_cues=_string_tuple(data.get("material_visual_cues", data.get("material_visual_cue"))),
            description_candidates=_string_tuple(data.get("description_candidates")),
            uncertainties=_string_tuple(data.get("uncertainties")),
            alternative_interpretations=_string_tuple(data.get("alternative_interpretations")),
            unsupported_fields=_string_tuple(data.get("unsupported_fields")),
        )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> TokenUsage:
        raw = dict(data or {})

        def number(*names: str) -> int | None:
            for name in names:
                if raw.get(name) is not None:
                    try:
                        return max(0, int(raw[name]))
                    except (TypeError, ValueError):
                        return None
            return None

        return cls(
            input_tokens=number("input_tokens", "prompt_tokens"),
            output_tokens=number("output_tokens", "completion_tokens"),
            total_tokens=number("total_tokens"),
        )


@dataclass(frozen=True)
class FailureDiagnostics:
    failure_type: str
    message: str
    retryable: bool = False
    response_sha256: str = ""
    safe_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "message": self.message,
            "retryable": self.retryable,
            "response_sha256": self.response_sha256,
            "safe_excerpt": self.safe_excerpt,
        }


@dataclass(frozen=True)
class VLMProposalArtifact:
    """Auditable envelope around one raw and parsed blind proposal response."""

    schema_version: str = VLM_PROPOSAL_SCHEMA_VERSION
    proposal: BlindVLMProposal | None = None
    raw_output: str = ""
    parsed_output: dict[str, Any] | None = None
    model_identity: str = ""
    request_hash: str = ""
    image_hash: str = ""
    prompt_version: str = BLIND_VLM_PROMPT_VERSION
    latency_ms: float | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    failure: FailureDiagnostics | None = None

    @property
    def available(self) -> bool:
        return self.proposal is not None and self.failure is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "raw_output": self.raw_output,
            "parsed_output": dict(self.parsed_output) if self.parsed_output is not None else None,
            "model_identity": self.model_identity,
            "request_hash": self.request_hash,
            "image_hash": self.image_hash,
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage.to_dict(),
            "failure": self.failure.to_dict() if self.failure else None,
        }


def _unique(values: Sequence[Any] | Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, "", "unknown"):
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(str(item).strip() for item in values if str(item).strip() and str(item).strip() != "unknown")


def _raw_string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _canonical_color_role(value: str) -> str:
    aliases = {
        "primary": "primary_colors",
        "secondary": "secondary_colors",
        "outline": "outline_colors",
        "shadow": "shadow_colors",
        "highlight": "highlight_colors",
    }
    normalized = normalize_semantic_term(value)
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in set(aliases.values()) else ""


def _reject_self_confidence(value: Any, path: str = "proposal") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = normalize_semantic_term(key)
            if normalized in {
                "confidence",
                "probability",
                "probabilities",
                "raw_score",
                "score",
            } or normalized.endswith("_confidence"):
                raise ProposalValidationError(f"model self-confidence is forbidden at {path}.{key}")
            _reject_self_confidence(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_self_confidence(nested, f"{path}[{index}]")


def _safe_excerpt(raw: str, limit: int = 320) -> str:
    sanitized = re.sub(r"(?i)bearer\s+[a-z0-9._-]+", "Bearer [REDACTED]", raw)
    sanitized = re.sub(r"(?i)rpa_[a-z0-9]+", "[REDACTED_TOKEN]", sanitized)
    sanitized = re.sub(r"data:image/[^;]+;base64,[a-z0-9+/=]+", "[REDACTED_IMAGE]", sanitized, flags=re.I)
    return sanitized[:limit]


def parse_blind_vlm_response(
    raw_output: str | Mapping[str, Any],
    *,
    model_identity: str,
    request_hash: str,
    image_hash: str,
    prompt_version: str = BLIND_VLM_PROMPT_VERSION,
    latency_ms: float | None = None,
    token_usage: Mapping[str, Any] | None = None,
) -> VLMProposalArtifact:
    """Parse a provider response into a safe artifact without forcing success."""

    raw_text = (
        json.dumps(dict(raw_output), sort_keys=True, ensure_ascii=False)
        if isinstance(raw_output, Mapping)
        else str(raw_output)
    )
    try:
        parsed = dict(raw_output) if isinstance(raw_output, Mapping) else json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ProposalValidationError("blind VLM response must be a JSON object")
        proposal = BlindVLMProposal.from_dict(parsed)
    except (json.JSONDecodeError, ProposalValidationError, TypeError, ValueError) as exc:
        failure_type = "json_parse_failure" if isinstance(exc, json.JSONDecodeError) else "schema_validation_failure"
        failure = FailureDiagnostics(
            failure_type=failure_type,
            message=str(exc)[:240],
            retryable=True,
            response_sha256=hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest(),
            safe_excerpt=_safe_excerpt(raw_text),
        )
        return VLMProposalArtifact(
            raw_output=raw_text,
            parsed_output=None,
            model_identity=model_identity,
            request_hash=request_hash,
            image_hash=image_hash,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
            token_usage=TokenUsage.from_mapping(token_usage),
            failure=failure,
        )
    return VLMProposalArtifact(
        proposal=proposal,
        raw_output=raw_text,
        parsed_output=parsed,
        model_identity=model_identity,
        request_hash=request_hash,
        image_hash=image_hash,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
        token_usage=TokenUsage.from_mapping(token_usage),
    )


def build_blind_vlm_prompt() -> str:
    """Return the context-free Stage-B prompt contract.

    The prompt accepts pixels only. No scheduling, source, taxonomy, or
    candidate vocabulary is interpolated into it.
    """

    return (
        "Inspect only the supplied sprite pixels. Return one JSON object. Separate literal visible evidence from "
        "semantic interpretation. Use unknown or an empty list when unsupported. Never infer an exact material from "
        "color or sheen. Preserve a plausible alternative when the identity is genuinely ambiguous. Do not emit "
        "confidence, probability, or score fields. Put literal generic geometry such as rod, cylinder, rectangle, "
        "orb, or stick-like object in visual_form; do not present it as established functional identity. Required "
        "keys: canonical_object_candidates (value and visual_support), visual_form. Never use object or object_name "
        "as field names. Also return "
        "category_candidates, surface_alias_candidates, role_candidates, shape (silhouette, aspect, orientation, "
        "structure, edge_profile, parts), color_roles (primary, secondary, outline, shadow, highlight), "
        "material_visual_cues, description_candidates, uncertainties, alternative_interpretations, unsupported_fields."
    )


# Concise alias used by provider adapters.
parse_vlm_proposal = parse_blind_vlm_response
