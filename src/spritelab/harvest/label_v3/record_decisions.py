"""Auto-Labeling v3: record-level decision contract.

Derives a record state from per-field decisions using conservative precedence.
Every non-accepted record carries structured, machine-readable reason codes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, FieldDecision
from spritelab.harvest.label_v3.field_prefill import FieldPrefill

SCHEMA_VERSION = "record_decision_v3.1"

RecordState = Literal[
    "auto_accept",
    "partial_accept",
    "quarantine",
    "hard_reject",
    "unknown",
]

ReasonCode = Literal[
    "fatal_provenance_failure",
    "image_integrity_failure",
    "unresolved_high_severity_contradiction",
    "impossible_combination_violation",
    "open_set_unknown",
    "insufficient_evidence",
    "calibration_insufficient",
    "blank_or_empty_sprite",
    "below_minimum_information",
    "tiny_ambiguous_fragment",
    "environment_tile_out_of_scope",
    "malformed_alpha",
    "destructive_resize",
    "missing_provenance",
    "unsupported_domain",
    "uncertain_sheet_mapping",
    "inconsistent_variant_group",
    "irreconcilable_deterministic_contradiction",
]


@dataclass(frozen=True)
class RecordDecision:
    """One v3 record decision with per-field detail and structured reasons."""

    schema_version: str = SCHEMA_VERSION
    sprite_id: str = ""
    record_state: RecordState = "unknown"
    reason_codes: tuple[ReasonCode, ...] = ()
    reason_details: tuple[str, ...] = ()

    domain: FieldDecision = field(default_factory=FieldDecision)
    category: FieldDecision = field(default_factory=FieldDecision)
    canonical_object: FieldDecision = field(default_factory=FieldDecision)
    surface_alias: FieldDecision = field(default_factory=FieldDecision)
    color: FieldDecision = field(default_factory=FieldDecision)
    material: FieldDecision = field(default_factory=FieldDecision)
    shape: FieldDecision = field(default_factory=FieldDecision)
    role: FieldDecision = field(default_factory=FieldDecision)
    tags: AcceptedTagSet = field(default_factory=AcceptedTagSet)
    description: FieldDecision = field(default_factory=FieldDecision)
    description_artifact: dict[str, Any] = field(default_factory=dict)
    prefills: dict[str, FieldPrefill] = field(default_factory=dict)
    prefill_tags: tuple[str, ...] = ()
    prefill_metadata: dict[str, Any] = field(default_factory=dict)

    policy_hash: str = ""
    config_hash: str = ""
    lineage: dict[str, str] = field(default_factory=dict)
    _auto_derived: bool = False

    def __post_init__(self) -> None:
        if not self._auto_derived and self.record_state == "unknown":
            decisions = {
                "domain": self.domain,
                "category": self.category,
                "canonical_object": self.canonical_object,
                "color": self.color,
                "material": self.material,
                "shape": self.shape,
            }
            derived = derive_record_state(decisions)
            if derived != self.record_state:
                object.__setattr__(self, "record_state", derived)
                object.__setattr__(self, "_auto_derived", True)

    def all_required_accepted(self) -> bool:
        return (
            self.domain.state == "accepted"
            and self.category.state == "accepted"
            and self.canonical_object.state in {"accepted", "not_applicable"}
        )

    def has_any_quarantined(self) -> bool:
        return any(
            d.state == "quarantined"
            for d in [self.domain, self.category, self.canonical_object, self.color, self.material, self.shape]
        )

    def has_any_hard_reject(self) -> bool:
        return any(d.state == "rejected" for d in [self.domain, self.category, self.canonical_object])

    @property
    def accepted_fields(self) -> tuple[str, ...]:
        fields = {
            "domain": self.domain,
            "category": self.category,
            "canonical_object": self.canonical_object,
            "surface_alias": self.surface_alias,
            "color": self.color,
            "material": self.material,
            "shape": self.shape,
            "role": self.role,
            "description": self.description,
        }
        return tuple(name for name, decision in fields.items() if decision.state == "accepted")

    @property
    def abstained_fields(self) -> tuple[str, ...]:
        fields = {
            "domain": self.domain,
            "category": self.category,
            "canonical_object": self.canonical_object,
            "surface_alias": self.surface_alias,
            "color": self.color,
            "material": self.material,
            "shape": self.shape,
            "role": self.role,
            "description": self.description,
        }
        return tuple(name for name, decision in fields.items() if decision.state == "abstained")


def derive_record_state(decisions: dict[str, FieldDecision]) -> RecordState:
    """Derive a conservative record state from per-field decisions.

    Precedence: fatal > quarantine > auto_accept > partial_accept > unknown.

    The required *core* is ``category`` and ``canonical_object``. ``domain`` is
    treated as a supporting field, not a hard requirement: it is rarely
    acceptable on its own (there is usually no independent domain evidence), so
    gating acceptance on it would force otherwise-clean records into
    ``unknown``. A record whose category and object are both accepted is
    ``auto_accept`` regardless of domain.
    """

    # Fatal reasons: a rejected core field (e.g. validated impossible
    # combination) hard-rejects the whole record.
    for field_name in ("domain", "category", "canonical_object"):
        decision = decisions.get(field_name)
        if decision is not None and decision.state == "rejected":
            return "hard_reject"

    # Quarantine: any high-severity unresolved contradiction on any field.
    for decision in decisions.values():
        if decision.state == "quarantined":
            return "quarantine"

    category = decisions.get("category", FieldDecision())
    canonical_object = decisions.get("canonical_object", FieldDecision())

    category_accepted = category.state == "accepted"
    object_accepted = canonical_object.state in {"accepted", "not_applicable"}

    # Fully-supported required core -> auto_accept.
    if category_accepted and object_accepted:
        return "auto_accept"

    # Safe accepted category with an abstained/unknown object -> partial_accept.
    if category_accepted and canonical_object.state in {"abstained", "unknown", "novel"}:
        return "partial_accept"

    return "unknown"


def record_decision_to_json(record: RecordDecision) -> dict[str, Any]:
    from spritelab.harvest.label_v3.field_decisions import field_decision_to_json

    return {
        "schema_version": record.schema_version,
        "sprite_id": record.sprite_id,
        "record_state": record.record_state,
        "reason_codes": list(record.reason_codes),
        "reason_details": list(record.reason_details),
        "domain": field_decision_to_json(record.domain),
        "category": field_decision_to_json(record.category),
        "canonical_object": field_decision_to_json(record.canonical_object),
        "surface_alias": field_decision_to_json(record.surface_alias),
        "color": field_decision_to_json(record.color),
        "material": field_decision_to_json(record.material),
        "shape": field_decision_to_json(record.shape),
        "role": field_decision_to_json(record.role),
        "tags": record.tags.to_json(),
        "description": field_decision_to_json(record.description),
        "description_artifact": dict(record.description_artifact),
        "prefills": {name: prefill.to_json() for name, prefill in record.prefills.items()},
        "prefill_tags": list(record.prefill_tags),
        "prefill_metadata": dict(record.prefill_metadata),
        "policy_hash": record.policy_hash,
        "config_hash": record.config_hash,
        "lineage": dict(record.lineage),
    }


def record_decision_from_json(data: Mapping[str, Any]) -> RecordDecision:
    from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, field_decision_from_json
    from spritelab.harvest.label_v3.field_prefill import FieldPrefill, prefill_from_legacy_decision

    sprite_id = str(data.get("sprite_id", ""))
    field_names = (
        "domain",
        "category",
        "canonical_object",
        "surface_alias",
        "color",
        "material",
        "shape",
        "role",
        "description",
    )
    raw_prefills = data.get("prefills") or {}
    prefills: dict[str, FieldPrefill] = {}
    for name in field_names:
        raw = raw_prefills.get(name) if isinstance(raw_prefills, Mapping) else None
        prefills[name] = (
            FieldPrefill.from_json(raw)
            if isinstance(raw, Mapping)
            else prefill_from_legacy_decision(sprite_id, name, data.get(name) or {})
        )

    def _decision(name: str) -> FieldDecision:
        decision = field_decision_from_json(data.get(name) or {})
        # v3.1 serialized an uncalibrated proposal in accepted_value even when
        # state was abstained.  Preserve it in the compatibility prefill above,
        # but keep the calibrated decision contract truthful in memory.
        if decision.state != "accepted" and decision.accepted_value is not None:
            decision = replace(decision, accepted_value=None, accepted_values=())
        return decision

    return RecordDecision(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        sprite_id=str(data.get("sprite_id", "")),
        record_state=str(data.get("record_state", "unknown")),
        reason_codes=tuple(str(v) for v in data.get("reason_codes") or ()),
        reason_details=tuple(str(v) for v in data.get("reason_details") or ()),
        domain=_decision("domain"),
        category=_decision("category"),
        canonical_object=_decision("canonical_object"),
        surface_alias=_decision("surface_alias"),
        color=_decision("color"),
        material=_decision("material"),
        shape=_decision("shape"),
        role=_decision("role"),
        tags=AcceptedTagSet.from_json(data.get("tags") or {}),
        description=_decision("description"),
        description_artifact=dict(data.get("description_artifact") or {}),
        prefills=prefills,
        prefill_tags=tuple(str(v) for v in data.get("prefill_tags") or ()),
        prefill_metadata=dict(data.get("prefill_metadata") or {}),
        policy_hash=str(data.get("policy_hash", "")),
        config_hash=str(data.get("config_hash", "")),
        lineage={str(k): str(v) for k, v in (data.get("lineage") or {}).items()},
    )
