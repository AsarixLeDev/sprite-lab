"""Evidence-aware text reconciliation contracts for Labeling v4.

Reconciliation normalizes proposals and records disagreements.  It does not
promote model output to truth, erase novel terms, or turn a visual material cue
into an explicit material fact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spritelab.harvest.label_v4.filename_parser import (
    COMPOUND_OBJECTS,
    OBJECT_TOKENS,
    FilenameParseResult,
)
from spritelab.harvest.label_v4.proposal import (
    BlindVLMProposal,
    VLMProposalArtifact,
    is_generic_visual_form,
)
from spritelab.harvest.label_v4.semantic_axes import (
    CATEGORY_VALUES,
    DOMAIN_VALUES,
    ROLE_VALUES,
    normalize_semantic_term,
)

RECONCILIATION_SCHEMA_VERSION = "reconciliation_v4.2"
RECONCILIATION_PROMPT_VERSION = "text_reconciliation_v4.3"

RECONCILIATION_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "domain",
        "category",
        "canonical_object",
        "visual_form",
        "surface_alias",
        "role",
        "explicit_material",
        "visual_material_cue",
        "filename_color_hints",
        "raw_visual_color_roles",
        "color_roles",
        "shape",
        "description",
    }
)
PROVIDER_FIELD_ALIASES = {
    "object": "canonical_object",
    "object_name": "canonical_object",
    "material": "explicit_material",
}
CALIBRATED_CANONICAL_PROMOTIONS = {
    ("crystal", "crystal_cluster"): "calibrated_crystal_cluster_refinement",
}


class ReconciliationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FieldProposal:
    raw_open_vocabulary_value: Any = None
    normalized_controlled_value: Any = None
    alternatives: tuple[Any, ...] = ()
    support: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    decision: str = "accepted"
    promotion_basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in {"accepted", "rejected", "abstained"}:
            raise ReconciliationValidationError(f"invalid field proposal decision: {self.decision!r}")

    @property
    def value(self) -> Any:
        if self.decision != "accepted":
            return None
        return (
            self.normalized_controlled_value
            if self.normalized_controlled_value is not None
            else self.raw_open_vocabulary_value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw_open_vocabulary_value": self.raw_open_vocabulary_value,
            "normalized_controlled_value": self.normalized_controlled_value,
            "alternatives": list(self.alternatives),
            "support": list(self.support),
            "conflicts": list(self.conflicts),
            "decision": self.decision,
            "promotion_basis": list(self.promotion_basis),
        }


@dataclass(frozen=True)
class ReconciliationResult:
    schema_version: str = RECONCILIATION_SCHEMA_VERSION
    field_proposals: dict[str, FieldProposal] = field(default_factory=dict)
    taxonomy_mapping_actions: tuple[dict[str, Any], ...] = ()
    unresolved_conflicts: tuple[dict[str, Any], ...] = ()
    open_set_terms: tuple[str, ...] = ()
    claims_accepted: tuple[dict[str, Any], ...] = ()
    claims_rejected: tuple[dict[str, Any], ...] = ()
    claims_unresolved: tuple[dict[str, Any], ...] = ()
    prompt_version: str = RECONCILIATION_PROMPT_VERSION

    def __post_init__(self) -> None:
        terminal = {
            "accepted": self.claims_accepted,
            "rejected": self.claims_rejected,
            "unresolved": self.claims_unresolved,
        }
        seen: dict[tuple[str, str], str] = {}
        for state, claims in terminal.items():
            for claim in claims:
                identity = _claim_identity(claim)
                previous = seen.get(identity)
                if previous is not None and previous != state:
                    raise ReconciliationValidationError(
                        f"claim {identity!r} appears in both {previous} and {state} terminal collections"
                    )
                seen[identity] = state

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field_proposals": {name: proposal.to_dict() for name, proposal in self.field_proposals.items()},
            "taxonomy_mapping_actions": [dict(action) for action in self.taxonomy_mapping_actions],
            "unresolved_conflicts": [dict(conflict) for conflict in self.unresolved_conflicts],
            "open_set_terms": list(self.open_set_terms),
            "claims_accepted": [dict(claim) for claim in self.claims_accepted],
            "claims_rejected": [dict(claim) for claim in self.claims_rejected],
            "claims_unresolved": [dict(claim) for claim in self.claims_unresolved],
            "prompt_version": self.prompt_version,
        }


_CATEGORY_NORMALIZATION = {
    "shield": "armor",
    "helmet": "armor",
    "head_armor": "armor",
    "body_armor": "armor",
    "gemstone": "gem",
    "jewel": "gem",
    "crafting_material": "material",
    "item": "misc_item",
}
_DOMAIN_NORMALIZATION = {
    "item_icon": "inventory_icon",
    "weapon_icon": "equipment_icon",
    "armor_icon": "equipment_icon",
    "gem_icon": "resource_icon",
    "material_icon": "resource_icon",
}
_ROLE_NORMALIZATION = {
    "equipment": "wearable_equipment",
    "equippable": "wearable_equipment",
    "defensive": "defensive_equipment",
    "decoration": "decorative_item",
    "item": "unknown",
}


@dataclass(frozen=True)
class CanonicalPromotionDecision:
    canonical_object: str | None
    promoted: bool
    decision: str
    promotion_basis: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    visual_form: tuple[str, ...] = ()


def decide_canonical_object_promotion(
    deterministic_object: Any,
    visual_candidates: Sequence[Any],
    *,
    object_source: str = "",
    visual_form: Sequence[Any] = (),
    promotion_evidence: Mapping[str, Any] | None = None,
) -> CanonicalPromotionDecision:
    """Apply the conservative, auditable canonical-object promotion gate."""

    deterministic = normalize_semantic_term(deterministic_object)
    candidates = tuple(_dedupe(normalize_semantic_term(value) for value in visual_candidates if value))
    forms = tuple(
        _dedupe(
            normalize_semantic_term(value)
            for value in (*visual_form, *(value for value in candidates if is_generic_visual_form(value)))
            if value
        )
    )
    primary = candidates[0] if candidates else ""
    calibrated_basis = CALIBRATED_CANONICAL_PROMOTIONS.get((deterministic, primary))
    if deterministic and calibrated_basis:
        return CanonicalPromotionDecision(
            canonical_object=primary,
            promoted=True,
            decision="accepted",
            promotion_basis=(calibrated_basis,),
            alternatives=tuple(_dedupe((deterministic, *candidates[1:]))),
            visual_form=forms,
        )
    explicit_basis = {
        "sprite_filename": "explicit_sprite_filename_identity",
        "member_filename": "explicit_sprite_filename_identity",
        "explicit_cell_mapping": "explicit_per_cell_mapping",
        "reviewed_variant_metadata": "reviewed_family_mapping",
    }.get(object_source)
    if deterministic and explicit_basis:
        return CanonicalPromotionDecision(
            canonical_object=deterministic,
            promoted=True,
            decision="accepted",
            promotion_basis=(explicit_basis,),
            alternatives=tuple(value for value in candidates if value != deterministic),
            visual_form=forms,
        )

    evidence = dict(promotion_evidence or {})
    bases: list[str] = []
    for key in (
        "explicit_sprite_filename_identity",
        "explicit_per_cell_mapping",
        "reviewed_family_mapping",
        "calibrated_rule",
    ):
        if evidence.get(key):
            bases.append(key)
    groups = max(0, int(evidence.get("independent_evidence_groups", 0) or 0))
    # Unreviewed source-record identity is an evidence group, not a promotion
    # credential.  It may participate in independent agreement with visual
    # evidence, but it cannot establish canonical identity by itself.
    if deterministic and primary and deterministic == primary and object_source == "source_record_metadata":
        groups = max(groups + 1, 2)
    if groups >= 2:
        bases.append("independent_evidence_group_agreement")
    margin = float(evidence.get("visual_identity_margin", 0.0) or 0.0)
    if bool(evidence.get("strong_visual_identity")) and margin >= float(
        evidence.get("minimum_visual_identity_margin", 0.35)
    ):
        bases.append("strong_visual_identity_with_margin")

    # Generic geometry cannot be promoted merely because one visual prompt
    # listed it first. Only explicit/reviewed/calibrated identity evidence can
    # establish it as the functional object name.
    if primary and is_generic_visual_form(primary):
        bases = [
            basis
            for basis in bases
            if basis
            in {
                "explicit_sprite_filename_identity",
                "explicit_per_cell_mapping",
                "reviewed_family_mapping",
                "calibrated_rule",
            }
        ]
    if primary and bases:
        return CanonicalPromotionDecision(
            canonical_object=primary,
            promoted=True,
            decision="accepted",
            promotion_basis=tuple(_dedupe(bases)),
            alternatives=tuple(candidates[1:]),
            visual_form=forms,
        )
    alternatives = tuple(_dedupe((deterministic, *candidates))) if deterministic else candidates
    return CanonicalPromotionDecision(
        canonical_object=None,
        promoted=False,
        decision="abstained",
        promotion_basis=(),
        alternatives=alternatives,
        visual_form=forms,
    )


def _claim_identity(claim: Mapping[str, Any]) -> tuple[str, str]:
    field_name = normalize_semantic_term(claim.get("field"))
    value = claim.get("value")
    claim_text = str(claim.get("claim") or "").strip()
    if not field_name and ":" in claim_text:
        raw_field, raw_value = claim_text.split(":", 1)
        field_name = normalize_semantic_term(raw_field)
        value = raw_value.strip()
    if value is None and claim.get("values") is not None:
        value = claim.get("values")
    if value is None:
        value = claim_text
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return field_name or "claim", canonical


def _claim(field_name: str, value: Any, reason: str) -> dict[str, Any]:
    return {"field": field_name, "value": value, "reason": reason}


def _dedupe(values: Sequence[Any] | Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def normalize_reconciled_value(field_name: str, value: Any) -> tuple[Any, dict[str, Any] | None]:
    """Normalize one reconciled value and describe any controlled mapping."""

    if value is None:
        return None, None
    if field_name in {"shape", "color_roles"} and isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, (list, tuple)):
        normalized = [normalize_semantic_term(item) for item in value if normalize_semantic_term(item)]
        return list(_dedupe(normalized)), None
    raw = str(value).strip()
    token = normalize_semantic_term(raw)
    if field_name == "domain":
        normalized = _DOMAIN_NORMALIZATION.get(token, token)
        if normalized not in DOMAIN_VALUES:
            return None, {"field": field_name, "raw": raw, "normalized": None, "action": "invalid_axis_value"}
    elif field_name == "category":
        normalized = _CATEGORY_NORMALIZATION.get(token, token)
        if normalized not in CATEGORY_VALUES:
            return None, {"field": field_name, "raw": raw, "normalized": None, "action": "open_set_axis_term"}
    elif field_name == "role":
        normalized = _ROLE_NORMALIZATION.get(token, token)
        if normalized not in ROLE_VALUES:
            return None, {"field": field_name, "raw": raw, "normalized": None, "action": "invalid_axis_value"}
    elif field_name == "surface_alias" or field_name == "description":
        normalized = raw
    else:
        normalized = token or None
    action = None
    if normalized is not None and str(normalized) != raw and normalize_semantic_term(raw) != normalized:
        action = {"field": field_name, "raw": raw, "normalized": normalized, "action": "normalize"}
    return normalized, action


def _as_dict(deterministic: FilenameParseResult | Mapping[str, Any]) -> dict[str, Any]:
    return deterministic.to_dict() if isinstance(deterministic, FilenameParseResult) else dict(deterministic)


def _compact_deterministic_evidence(
    deterministic: FilenameParseResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Project full deterministic provenance into a lossless semantic digest.

    The complete parser artifact remains stored on the label record. Stage C
    only needs the extracted values, their transformations, and which source
    groups supplied them. Repeated paths and token positions are grouped here
    so a model cannot burn its response budget echoing thousands of characters
    of duplicate provenance.
    """

    raw = _as_dict(deterministic)
    semantic_keys = (
        "schema_version",
        "canonical_object",
        "surface_alias",
        "category",
        "domain",
        "role",
        "explicit_material",
        "explicit_material_candidates",
        "filename_color_hints",
        "size_hint",
        "size_hints",
        "condition_hints",
        "style_modifiers",
        "orientation_hints",
        "variant_suffixes",
        "sequence_numbers",
        "open_set_tokens",
        "generic",
        "object_source",
        "surface_alias_source",
        "field_sources",
    )
    compact = {key: raw.get(key) for key in semantic_keys if key in raw}

    source_values = raw.get("source_values")
    if isinstance(source_values, Mapping):
        source_context = {
            key: source_values.get(key)
            for key in ("filename", "member_path", "sheet_name", "pack_name")
            if source_values.get(key) not in (None, "", [], {})
        }
        if source_context:
            compact["source_context"] = source_context

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in raw.get("token_provenance") or ():
        if not isinstance(item, Mapping):
            continue
        classification = str(item.get("classification") or "")
        value = str(item.get("value") or item.get("normalized_token") or "")
        raw_token = str(item.get("raw_token") or value)
        transformation = str(item.get("transformation") or "identity")
        source = str(item.get("source") or "unknown")
        key = (classification, value, raw_token, transformation)
        evidence = grouped.setdefault(
            key,
            {
                "classification": classification,
                "value": value,
                "raw_token": raw_token,
                "transformation": transformation,
                "sources": [],
            },
        )
        if source not in evidence["sources"]:
            evidence["sources"].append(source)
    if grouped:
        compact["token_evidence"] = list(grouped.values())
        compact["full_provenance_stored_out_of_band"] = True
    return compact


def _as_proposal(vlm: BlindVLMProposal | VLMProposalArtifact | Mapping[str, Any] | None) -> BlindVLMProposal | None:
    if vlm is None:
        return None
    if isinstance(vlm, VLMProposalArtifact):
        return vlm.proposal
    if isinstance(vlm, BlindVLMProposal):
        return vlm
    return BlindVLMProposal.from_dict(vlm)


def _field(
    raw: Any,
    field_name: str,
    *,
    alternatives: Sequence[Any] = (),
    support: Sequence[str] = (),
    conflicts: Sequence[str] = (),
    decision: str = "accepted",
    promotion_basis: Sequence[str] = (),
    actions: list[dict[str, Any]],
) -> FieldProposal:
    normalized, action = normalize_reconciled_value(field_name, raw)
    if action is not None:
        actions.append(action)
    return FieldProposal(
        raw_open_vocabulary_value=raw,
        normalized_controlled_value=normalized,
        alternatives=_dedupe(tuple(alternatives)),
        support=_dedupe(tuple(support)),
        conflicts=_dedupe(tuple(conflicts)),
        decision=decision,
        promotion_basis=_dedupe(tuple(promotion_basis)),
    )


def reconcile_evidence(
    deterministic: FilenameParseResult | Mapping[str, Any],
    vlm: BlindVLMProposal | VLMProposalArtifact | Mapping[str, Any] | None,
    *,
    known_object_vocabulary: Sequence[str] | None = None,
    promotion_evidence: Mapping[str, Any] | None = None,
) -> ReconciliationResult:
    """Reconcile deterministic semantics with a blind visual proposal.

    The deterministic value is retained as the primary proposal when explicit,
    while disagreement is carried forward for risk estimation and verification.
    """

    det = _as_dict(deterministic)
    visual = _as_proposal(vlm)
    actions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    claims_accepted: list[dict[str, Any]] = []
    claims_rejected: list[dict[str, Any]] = []
    claims_unresolved: list[dict[str, Any]] = []
    open_terms: list[str] = [normalize_semantic_term(value) for value in det.get("open_set_tokens") or ()]
    fields: dict[str, FieldProposal] = {}

    known = {semantics.canonical_object for semantics in OBJECT_TOKENS.values()} | {
        semantics.canonical_object for semantics in COMPOUND_OBJECTS.values()
    }
    if known_object_vocabulary is not None:
        known |= {normalize_semantic_term(value) for value in known_object_vocabulary}

    det_domain = normalize_semantic_term(det.get("domain"))
    if det_domain and det_domain != "unknown":
        normalized_domain, domain_action = normalize_reconciled_value("domain", det_domain)
        if domain_action is not None:
            actions.append(domain_action)
        if normalized_domain:
            fields["domain"] = FieldProposal(
                raw_open_vocabulary_value=det_domain,
                normalized_controlled_value=normalized_domain,
                support=("deterministic_source_context",),
            )
            claims_accepted.append(_claim("domain", normalized_domain, "deterministic_source_context"))
        else:
            claims_rejected.append({"field": "domain", "value": det_domain, "reason": "invalid_domain_axis_value"})

    det_object = normalize_semantic_term(det.get("canonical_object"))
    visual_objects = [candidate.value for candidate in visual.object_candidates] if visual else []
    visual_object = visual_objects[0] if visual_objects else ""
    visual_forms = list(visual.visual_form) if visual else []
    promotion = decide_canonical_object_promotion(
        det_object,
        visual_objects,
        object_source=str(det.get("object_source") or ""),
        visual_form=visual_forms,
        promotion_evidence=promotion_evidence,
    )
    object_conflicts: list[str] = []
    calibrated_refinement = bool(
        promotion.promotion_basis and promotion.promotion_basis[0] in set(CALIBRATED_CANONICAL_PROMOTIONS.values())
    )
    if det_object and visual_object and det_object != visual_object and not calibrated_refinement:
        code = "filename_vlm_object_disagreement"
        object_conflicts.append(code)
        conflict = {
            "field": "canonical_object",
            "code": code,
            "values": [det_object, visual_object],
            "support": [str(det.get("object_source") or "deterministic_identity"), "vlm_visual"],
        }
        conflicts.append(conflict)
        claims_unresolved.append(_claim("canonical_object", [det_object, visual_object], code))
    if calibrated_refinement:
        actions.append(
            {
                "field": "canonical_object",
                "raw": det_object,
                "normalized": promotion.canonical_object,
                "action": promotion.promotion_basis[0],
            }
        )
    if promotion.canonical_object:
        fields["canonical_object"] = _field(
            promotion.canonical_object,
            "canonical_object",
            alternatives=promotion.alternatives,
            support=tuple(promotion.promotion_basis),
            conflicts=object_conflicts,
            promotion_basis=promotion.promotion_basis,
            actions=actions,
        )
        claims_accepted.append(
            _claim("canonical_object", promotion.canonical_object, "+".join(promotion.promotion_basis))
        )
        if normalize_semantic_term(promotion.canonical_object) not in known:
            open_terms.append(normalize_semantic_term(promotion.canonical_object))
    elif visual_objects or det_object:
        abstained_identity = visual_object or det_object
        fields["canonical_object"] = FieldProposal(
            raw_open_vocabulary_value=abstained_identity,
            normalized_controlled_value=None,
            alternatives=tuple(promotion.alternatives),
            support=tuple(
                value
                for value in (
                    str(det.get("object_source") or "") if det_object else "",
                    "vlm_visual" if visual_objects else "",
                )
                if value
            ),
            conflicts=("canonical_object_promotion_abstained",),
            decision="abstained",
        )
        actions.append(
            {
                "field": "canonical_object",
                "raw": abstained_identity,
                "normalized": None,
                "action": "canonical_object_promotion_abstained",
            }
        )
        claims_unresolved.append(
            _claim("canonical_object", list(promotion.alternatives), "promotion_gate_not_satisfied")
        )
    if promotion.visual_form:
        fields["visual_form"] = _field(
            list(promotion.visual_form),
            "visual_form",
            support=("vlm_visual",),
            actions=actions,
        )
    for candidate in visual_objects:
        if candidate not in known:
            open_terms.append(candidate)

    det_category = normalize_semantic_term(det.get("category"))
    if det_category == "unknown":
        det_category = ""
    raw_visual_categories = list(visual.category_candidates) if visual else []
    normalized_visual_categories: list[str] = []
    for candidate in raw_visual_categories:
        normalized, action = normalize_reconciled_value("category", candidate)
        if action is not None:
            actions.append(action)
        if normalized:
            normalized_visual_categories.append(str(normalized))
        elif candidate:
            open_terms.append(candidate)
    chosen_category = det_category or (normalized_visual_categories[0] if normalized_visual_categories else "")
    category_conflicts: list[str] = []
    if det_category and normalized_visual_categories and det_category not in normalized_visual_categories:
        code = "filename_vlm_category_disagreement"
        category_conflicts.append(code)
        conflict = {
            "field": "category",
            "code": code,
            "values": [det_category, *normalized_visual_categories],
            "support": [
                str((det.get("field_sources") or {}).get("category") or "deterministic_source_context"),
                "vlm_visual",
            ],
        }
        conflicts.append(conflict)
        claims_unresolved.append(_claim("category", conflict["values"], code))
    if chosen_category:
        fields["category"] = _field(
            chosen_category,
            "category",
            alternatives=[value for value in normalized_visual_categories if value != chosen_category],
            support=("filename",) if det_category else ("vlm_visual",),
            conflicts=category_conflicts,
            actions=actions,
        )
        claims_accepted.append(
            _claim("category", chosen_category, "deterministic_precedence" if det_category else "visual_proposal")
        )

    alias = str(det.get("surface_alias") or "").strip()
    visual_aliases = list(visual.surface_alias_candidates) if visual else []
    if alias:
        fields["surface_alias"] = _field(
            alias,
            "surface_alias",
            alternatives=visual_aliases,
            support=(str(det.get("surface_alias_source") or "deterministic_identity"),),
            actions=actions,
        )
        claims_accepted.append(_claim("surface_alias", alias, "eligible_identity_source"))
    elif visual_aliases:
        fields["surface_alias"] = FieldProposal(
            raw_open_vocabulary_value=visual_aliases[0],
            normalized_controlled_value=None,
            alternatives=tuple(visual_aliases[1:]),
            support=("vlm_visual",),
            conflicts=("visual_alias_not_identity_evidence",),
            decision="abstained",
        )

    det_role = normalize_semantic_term(det.get("role"))
    if det_role == "unknown":
        det_role = ""
    visual_roles = list(visual.role_candidates) if visual else []
    normalized_visual_roles: list[str] = []
    for candidate in visual_roles:
        normalized, action = normalize_reconciled_value("role", candidate)
        if action is not None:
            actions.append(action)
        if normalized:
            normalized_visual_roles.append(str(normalized))
    chosen_role = det_role or (normalized_visual_roles[0] if normalized_visual_roles else "")
    if chosen_role:
        role_conflicts = (
            ("filename_vlm_role_disagreement",)
            if det_role and normalized_visual_roles and det_role not in normalized_visual_roles
            else ()
        )
        if role_conflicts:
            conflict = {
                "field": "role",
                "code": role_conflicts[0],
                "values": [det_role, *normalized_visual_roles],
                "support": ["deterministic_source_context", "vlm_visual"],
            }
            conflicts.append(conflict)
            claims_unresolved.append(_claim("role", conflict["values"], role_conflicts[0]))
        fields["role"] = _field(
            chosen_role,
            "role",
            alternatives=[value for value in normalized_visual_roles if value != chosen_role],
            support=("filename",) if det_role else ("vlm_visual",),
            conflicts=role_conflicts,
            actions=actions,
        )
        claims_accepted.append(
            _claim("role", chosen_role, "deterministic_precedence" if det_role else "visual_proposal")
        )

    explicit_material = normalize_semantic_term(det.get("explicit_material"))
    if explicit_material:
        fields["explicit_material"] = _field(
            explicit_material,
            "explicit_material",
            alternatives=tuple(det.get("explicit_material_candidates") or ())[1:],
            support=("filename_explicit",),
            actions=actions,
        )
        claims_accepted.append(_claim("explicit_material", explicit_material, "explicit_source_metadata"))
    if visual and visual.material_visual_cues:
        fields["visual_material_cue"] = _field(
            list(visual.material_visual_cues),
            "visual_material_cue",
            support=("vlm_visual",),
            actions=actions,
        )

    filename_colors = tuple(normalize_semantic_term(value) for value in det.get("filename_color_hints") or ())
    if filename_colors:
        fields["filename_color_hints"] = _field(
            list(filename_colors), "filename_color_hints", support=("filename_explicit",), actions=actions
        )
    if visual:
        raw_visual_color_roles = {role: list(values) for role, values in visual.raw_visual_color_roles.items()}
        fields["raw_visual_color_roles"] = FieldProposal(
            raw_open_vocabulary_value=raw_visual_color_roles,
            normalized_controlled_value=None,
            support=("vlm_visual",),
        )
        color_roles = visual.color_roles.to_dict()
        fields["color_roles"] = _field(color_roles, "color_roles", support=("vlm_visual",), actions=actions)
        visual_primary = set(visual.color_roles.primary_colors) | set(visual.color_roles.secondary_colors)
        if filename_colors and visual_primary and set(filename_colors).isdisjoint(visual_primary):
            code = "filename_visual_color_conflict"
            fields["filename_color_hints"] = FieldProposal(
                raw_open_vocabulary_value=list(filename_colors),
                normalized_controlled_value=list(filename_colors),
                support=("filename_explicit",),
                conflicts=(code,),
            )
            fields["color_roles"] = FieldProposal(
                raw_open_vocabulary_value=color_roles,
                normalized_controlled_value=color_roles,
                support=("vlm_visual",),
                conflicts=(code,),
            )
            conflicts.append(
                {
                    "field": "color",
                    "code": code,
                    "values": [list(filename_colors), sorted(visual_primary)],
                    "support": ["filename_explicit", "vlm_visual"],
                }
            )

        shape = visual.shape.to_dict()
        if any(shape.values()):
            fields["shape"] = _field(shape, "shape", support=("vlm_visual",), actions=actions)
        if visual.description_candidates:
            fields["description"] = _field(
                visual.description_candidates[0],
                "description",
                alternatives=visual.description_candidates[1:],
                support=("vlm_visual",),
                actions=actions,
            )

    if isinstance(vlm, VLMProposalArtifact) and isinstance(vlm.parsed_output, Mapping):
        visual_explicit = vlm.parsed_output.get("explicit_material")
        if visual_explicit and not explicit_material:
            claims_rejected.append(
                {
                    "field": "explicit_material",
                    "value": visual_explicit,
                    "reason": "visual_cue_cannot_establish_explicit_material",
                }
            )

    return ReconciliationResult(
        field_proposals=fields,
        taxonomy_mapping_actions=tuple(_dedupe_dicts(actions)),
        unresolved_conflicts=tuple(_dedupe_dicts(conflicts)),
        open_set_terms=tuple(_dedupe(value for value in open_terms if value)),
        claims_accepted=tuple(_dedupe_dicts(claims_accepted)),
        claims_rejected=tuple(_dedupe_dicts(claims_rejected)),
        claims_unresolved=tuple(_dedupe_dicts(claims_unresolved)),
    )


def _dedupe_dicts(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        item = dict(value)
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def parse_reconciliation_response(raw_output: str | Mapping[str, Any]) -> ReconciliationResult:
    """Parse Stage-C output into a strict normalized view.

    Provider artifacts retain the original response outside this function.
    Legacy list-shaped proposals and object-name aliases are accepted only at
    this boundary and are immediately canonicalized.
    """

    data = dict(raw_output) if isinstance(raw_output, Mapping) else json.loads(str(raw_output))
    if not isinstance(data, dict):
        raise ReconciliationValidationError("reconciliation response must be a JSON object")
    raw_fields = data.get("field_proposals", {})
    if isinstance(raw_fields, Mapping):
        field_items = list(raw_fields.items())
    elif isinstance(raw_fields, Sequence) and not isinstance(raw_fields, (str, bytes, bytearray)):
        field_items = []
        for index, item in enumerate(raw_fields):
            if not isinstance(item, Mapping) or not item.get("field"):
                raise ReconciliationValidationError(f"legacy field proposal at index {index} must contain a field name")
            field_items.append((str(item.get("field")), item))
    else:
        raise ReconciliationValidationError("reconciliation response must contain field proposals")

    # Provider-supplied taxonomy prose remains in the raw ProviderArtifact.
    # Normalized actions are rebuilt from validated field values so a model
    # cannot assert that an out-of-enum value is valid taxonomy.
    actions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = [
        dict(value)
        if isinstance(value, Mapping)
        else {"field": "unknown", "code": "provider_conflict", "reason": str(value)}
        for value in data.get("unresolved_conflicts") or ()
    ]
    open_terms = [str(value) for value in data.get("open_set_terms") or ()]
    claims_accepted = [_provider_claim(value) for value in data.get("claims_accepted") or ()]
    claims_rejected = [_provider_claim(value) for value in data.get("claims_rejected") or ()]
    claims_unresolved = [_provider_claim(value) for value in data.get("claims_unresolved") or ()]
    fields: dict[str, FieldProposal] = {}
    for original_field_name, raw_proposal in field_items:
        if not isinstance(raw_proposal, Mapping):
            raise ReconciliationValidationError(f"field proposal {original_field_name!r} must be an object")
        normalized_name = normalize_semantic_term(original_field_name)
        field_name = PROVIDER_FIELD_ALIASES.get(normalized_name, normalized_name)
        if field_name != normalized_name:
            actions.append(
                {
                    "field": field_name,
                    "raw_field": original_field_name,
                    "action": "legacy_provider_field_alias_canonicalized",
                }
            )
        if field_name not in RECONCILIATION_FIELD_NAMES:
            actions.append(
                {
                    "field": original_field_name,
                    "normalized": None,
                    "action": "invalid_provider_field_name",
                }
            )
            conflicts.append(
                {
                    "field": original_field_name,
                    "code": "invalid_reconciliation_provider_field",
                    "values": [raw_proposal.get("raw_open_vocabulary_value", raw_proposal.get("value"))],
                }
            )
            continue
        if field_name in fields:
            conflicts.append(
                {
                    "field": field_name,
                    "code": "duplicate_reconciliation_provider_field",
                    "values": [original_field_name],
                }
            )
            continue

        raw_value = raw_proposal.get("raw_open_vocabulary_value", raw_proposal.get("value"))
        requested_normalized = raw_proposal.get("normalized_controlled_value")
        value_to_validate = requested_normalized if requested_normalized is not None else raw_value
        normalized, action = normalize_reconciled_value(field_name, value_to_validate)
        invalid_controlled = (
            field_name in {"domain", "category", "role"}
            and normalized is None
            and bool(normalize_semantic_term(value_to_validate))
        )
        field_conflicts = [str(value) for value in raw_proposal.get("conflicts") or ()]
        decision = {
            "accept": "accepted",
            "accepted": "accepted",
            "reject": "rejected",
            "rejected": "rejected",
            "unresolved": "abstained",
            "abstain": "abstained",
            "abstained": "abstained",
        }.get(str(raw_proposal.get("decision") or "accepted").lower(), str(raw_proposal.get("decision") or "accepted"))
        raw_basis = raw_proposal.get("promotion_basis") or ()
        if isinstance(raw_basis, str):
            raw_basis = (raw_basis,)
        promotion_basis = tuple(str(value) for value in raw_basis)
        support = tuple(str(value) for value in raw_proposal.get("support") or ())
        if action is not None:
            actions.append(action)
            if action.get("action") == "open_set_axis_term" and raw_value:
                open_terms.append(str(raw_value))
        if invalid_controlled:
            code = "invalid_taxonomy_provider_output"
            decision = "rejected"
            support = ()
            field_conflicts.append(code)
            conflict = {
                "field": field_name,
                "code": code,
                "values": [value_to_validate],
                "allowed_values": list(
                    DOMAIN_VALUES
                    if field_name == "domain"
                    else CATEGORY_VALUES
                    if field_name == "category"
                    else ROLE_VALUES
                ),
            }
            conflicts.append(conflict)
            claims_rejected.append(_claim(field_name, value_to_validate, code))
        elif field_name == "canonical_object" and is_generic_visual_form(normalized) and not promotion_basis:
            decision = "abstained"
            normalized = None
            field_conflicts.append("canonical_object_promotion_abstained")
            actions.append(
                {
                    "field": "canonical_object",
                    "raw": raw_value,
                    "normalized": None,
                    "action": "canonical_object_promotion_abstained",
                }
            )
            claims_unresolved.append(_claim("canonical_object", raw_value, "promotion_gate_not_satisfied"))

        fields[field_name] = FieldProposal(
            raw_open_vocabulary_value=raw_value,
            normalized_controlled_value=normalized,
            alternatives=tuple(raw_proposal.get("alternatives") or ()),
            support=support,
            conflicts=tuple(_dedupe(field_conflicts)),
            decision=decision,
            promotion_basis=promotion_basis,
        )

    # Backward-compatible repair for the exact semantic inversion observed in
    # the first canary. New providers must emit claims_accepted directly.
    retained_rejected: list[dict[str, Any]] = []
    for claim in claims_rejected:
        reason = normalize_semantic_term(claim.get("reason"))
        if "accepted_not_rejected" in reason or "claim_is_accepted" in reason:
            claims_accepted.append(claim)
            actions.append({"claim": dict(claim), "action": "terminal_claim_reclassified_as_accepted"})
        else:
            retained_rejected.append(claim)

    return ReconciliationResult(
        schema_version=str(data.get("schema_version") or RECONCILIATION_SCHEMA_VERSION),
        field_proposals=fields,
        taxonomy_mapping_actions=tuple(_dedupe_dicts(actions)),
        unresolved_conflicts=tuple(_dedupe_dicts(conflicts)),
        open_set_terms=tuple(_dedupe(open_terms)),
        claims_accepted=tuple(_dedupe_dicts(claims_accepted)),
        claims_rejected=tuple(_dedupe_dicts(retained_rejected)),
        claims_unresolved=tuple(_dedupe_dicts(claims_unresolved)),
        prompt_version=str(data.get("prompt_version") or RECONCILIATION_PROMPT_VERSION),
    )


def _provider_claim(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value)
    separator = ":" if ":" in text else "=" if "=" in text else ""
    if separator:
        field_name, raw_value = text.split(separator, 1)
        return {
            "field": normalize_semantic_term(field_name),
            "value": raw_value.strip(),
            "reason": "provider_terminal_claim",
        }
    return {"field": "claim", "value": text, "reason": "provider_terminal_claim"}


def build_reconciliation_prompt(
    deterministic: FilenameParseResult | Mapping[str, Any],
    vlm: BlindVLMProposal | VLMProposalArtifact | Mapping[str, Any] | None,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    declarative_mappings: Mapping[str, Any] | None = None,
    known_conflicts: Sequence[Mapping[str, Any]] = (),
    candidate_vocabularies: Mapping[str, Sequence[str]] | None = None,
    approved_prior_labels: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build the text-only Stage-C request from explicit evidence objects."""

    proposal = _as_proposal(vlm)
    payload = {
        "deterministic_evidence": _compact_deterministic_evidence(deterministic),
        "source_metadata": dict(source_metadata or {}),
        "declarative_mappings": dict(declarative_mappings or {}),
        "blind_vlm_proposal": proposal.to_dict() if proposal else None,
        "taxonomy": {"domains": DOMAIN_VALUES, "categories": CATEGORY_VALUES, "roles": ROLE_VALUES},
        "known_conflicts": [dict(value) for value in known_conflicts],
        "candidate_vocabularies": {key: list(value) for key, value in (candidate_vocabularies or {}).items()},
        "approved_prior_labels": [dict(value) for value in approved_prior_labels],
    }
    return (
        "You are the text-only Stage C evidence reconciler. Return exactly one complete JSON object and nothing "
        "else. field_proposals MUST be an object keyed only by canonical field names; use canonical_object, never "
        "object or object_name. Required top-level keys: field_proposals, taxonomy_mapping_actions, "
        "unresolved_conflicts, open_set_terms, claims_accepted, claims_rejected, claims_unresolved. Every claim "
        "must occur in exactly one terminal claim collection. Each field proposal must contain "
        "raw_open_vocabulary_value, normalized_controlled_value, alternatives, support, conflicts, decision, and "
        "promotion_basis. A domain is valid only when it is exactly one of the supplied taxonomy.domains values; "
        "weapon, gem, jewelry, pixel_art, and rpg_icons are not domains. Preserve invalid raw values but set their "
        "normalized value to null and emit invalid_taxonomy_provider_output. Generic visual geometry belongs in "
        "visual_form and must not become canonical_object without an explicit promotion basis. Never promote a "
        "visual material cue to explicit material; never silently resolve a filename/visual color conflict. Do "
        "not copy, quote, summarize, or echo the INPUT object. INPUT="
        + json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    )


# Convenient aliases for provider and pipeline layers.
reconcile = reconcile_evidence
parse_text_reconciliation = parse_reconciliation_response
