"""Typed evidence, decision, calibration, and export contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_finite,
    require_probability,
    require_sha256,
    require_text,
    require_unique_text,
)


class EvidenceChannel(str, Enum):
    TECHNICAL = "technical"
    VISUAL_ONLY = "visual_only"
    RETRIEVAL = "retrieval"
    METADATA = "metadata"
    PACK_CONTEXT = "pack_context"
    HUMAN = "human"


class ValidityState(str, Enum):
    VALID = "valid"
    HEURISTIC = "heuristic"
    NOT_APPLICABLE = "not_applicable"
    INVALID = "invalid"


class EvidenceStrength(str, Enum):
    STRONG_DETERMINISTIC = "strong_deterministic"
    HEURISTIC_TECHNICAL = "heuristic_technical"


class DecisionState(str, Enum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"
    HUMAN_ABSTAINED = "human_abstained"
    MODEL_ABSTAINED = "model_abstained"
    HUMAN_VERIFIED = "human_verified"
    INVALID = "invalid"


# Runtime capability minted only after review.py has verified the complete append-only
# event chain, authoritative event, taxonomy path, and cohort membership.  The seal is
# deliberately not a dataclass field, so it cannot be supplied by constructors or JSON
# deserialization and is not persisted as self-attested provenance.
_HUMAN_TRUTH_PROJECTION_SEAL = object()


class CalibrationState(str, Enum):
    NOT_READY = "not_ready"
    INSUFFICIENT_TRUTH = "insufficient_truth"
    READY_FOR_EXPERIMENT = "ready_for_experiment"
    VALIDATED_FOR_SCOPE = "validated_for_scope"


class SupervisionTier(str, Enum):
    HUMAN_VERIFIED = "human_verified"
    HUMAN_ABSTAINED = "human_abstained"
    CALIBRATED_CORE = "calibrated_core"
    RETRIEVAL_SUPPORTED_WEAK = "retrieval_supported_weak"
    MODEL_PROPOSAL = "model_proposal"
    AUXILIARY_VISUAL = "auxiliary_visual"
    TECHNICAL_DETERMINISTIC = "technical_deterministic"
    UNUSABLE = "unusable"


@dataclass(frozen=True, eq=False)
class FeatureValue(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.feature-value.v1"
    IDENTITY_FIELDS = ("name", "method", "method_version", "source_image_identity")

    name: str
    value: Any
    method: str
    method_version: str
    validity: ValidityState
    strength: EvidenceStrength
    source_image_identity: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.name, "feature name", identifier=True)
        require_text(self.method, "feature method")
        require_text(self.method_version, "feature method version")
        require_text(self.source_image_identity, "source image identity")
        require_probability(self.confidence, "feature confidence", optional=True)
        self.validate_record()


@dataclass(frozen=True, eq=False)
class TechnicalVisualEvidence(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.technical-visual-evidence.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity")

    record_identity: str
    image_identity: str
    features: tuple[FeatureValue, ...]
    extraction_identity: str

    def __post_init__(self) -> None:
        require_text(self.record_identity, "record identity")
        require_text(self.image_identity, "image identity")
        require_text(self.extraction_identity, "extraction identity")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ContractValidationError("technical feature names cannot repeat")
        if any(feature.source_image_identity != self.image_identity for feature in self.features):
            raise ContractValidationError("every technical feature must bind the same source image identity")
        self.validate_record()

    def feature(self, name: str, default: Any = None) -> Any:
        return next((feature.value for feature in self.features if feature.name == name), default)


@dataclass(frozen=True, eq=False)
class VisualDescription(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.visual-description.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "render_bundle_identity", "prompt_identity")

    record_identity: str
    image_identity: str
    render_bundle_identity: str
    provider_identity: str
    model_identity: str
    prompt_identity: str
    visible_observations: tuple[str, ...]
    visible_entities: tuple[str, ...]
    entity_count: int | None
    shape_terms: tuple[str, ...]
    visual_forms: tuple[str, ...]
    dominant_colors: tuple[str, ...]
    secondary_colors: tuple[str, ...]
    material_like_cues: tuple[str, ...]
    orientation: tuple[str, ...]
    symmetry: tuple[str, ...]
    visible_parts: tuple[str, ...]
    possible_interpretations: tuple[str, ...]
    ambiguities: tuple[str, ...]
    resolution_limitations: tuple[str, ...]
    scene_or_icon_context: str
    caption_short: str
    caption_detailed: str

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "image_identity",
            "render_bundle_identity",
            "provider_identity",
            "model_identity",
            "prompt_identity",
            "scene_or_icon_context",
            "caption_short",
            "caption_detailed",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        if self.entity_count is not None and (type(self.entity_count) is not int or self.entity_count < 0):
            raise ContractValidationError("entity_count must be a non-negative integer or null")
        for name in (
            "visible_observations",
            "visible_entities",
            "shape_terms",
            "visual_forms",
            "dominant_colors",
            "secondary_colors",
            "material_like_cues",
            "orientation",
            "symmetry",
            "visible_parts",
            "possible_interpretations",
            "ambiguities",
            "resolution_limitations",
        ):
            require_unique_text(getattr(self, name), name)
        allowed_material_cues = {"metal-like", "wood-like", "stone-like", "glass-like", "fabric-like"}
        invalid_cues = sorted(set(self.material_like_cues) - allowed_material_cues)
        if invalid_cues:
            raise ContractValidationError(f"material cues overclaim or are uncontrolled: {', '.join(invalid_cues)}")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class StructuredVisualAttributes(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.structured-visual-attributes.v1"
    IDENTITY_FIELDS = ("description_identity", "image_identity")

    description_identity: str
    image_identity: str
    entity_count: int | None
    colors: tuple[str, ...]
    forms: tuple[str, ...]
    parts: tuple[str, ...]
    orientations: tuple[str, ...]
    material_like_cues: tuple[str, ...]
    uncertainty_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.description_identity, "description identity")
        require_text(self.image_identity, "image identity")
        for name in ("colors", "forms", "parts", "orientations", "material_like_cues", "uncertainty_terms"):
            require_unique_text(getattr(self, name), name)
        if self.entity_count is not None and (type(self.entity_count) is not int or self.entity_count < 0):
            raise ContractValidationError("entity_count must be a non-negative integer or null")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class SemanticHypothesis(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.semantic-hypothesis.v1"
    IDENTITY_FIELDS = ("node_id", "taxonomy_identity", "provider_identity", "prompt_identity")

    node_id: str
    depth: int
    rank: int
    raw_model_confidence: float | None
    evidence_citations: tuple[str, ...]
    contradicting_observations: tuple[str, ...]
    abstention_recommended: bool
    provider_identity: str
    model_identity: str
    prompt_identity: str
    render_bundle_identity: str
    taxonomy_identity: str

    def __post_init__(self) -> None:
        require_text(self.node_id, "hypothesis node", identifier=True)
        if type(self.depth) is not int or self.depth < 0 or type(self.rank) is not int or self.rank < 1:
            raise ContractValidationError("hypothesis depth and rank must be non-negative/positive integers")
        require_probability(self.raw_model_confidence, "raw model confidence", optional=True)
        for name in (
            "provider_identity",
            "model_identity",
            "prompt_identity",
            "render_bundle_identity",
            "taxonomy_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        require_unique_text(self.evidence_citations, "evidence citations")
        require_unique_text(self.contradicting_observations, "contradicting observations")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class TaxonomyPathHypothesis(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.taxonomy-path-hypothesis.v1"
    IDENTITY_FIELDS = ("record_identity", "taxonomy_identity", "description_identity")

    record_identity: str
    taxonomy_identity: str
    description_identity: str
    path: tuple[str, ...]
    hypotheses: tuple[SemanticHypothesis, ...]
    no_safe_hypothesis: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        require_text(self.record_identity, "record identity")
        require_text(self.taxonomy_identity, "taxonomy identity")
        require_text(self.description_identity, "description identity")
        require_unique_text(self.path, "taxonomy hypothesis path")
        if self.no_safe_hypothesis:
            if self.path or self.hypotheses:
                raise ContractValidationError("no_safe_hypothesis cannot carry candidates")
            require_text(self.reason, "no-safe-hypothesis reason")
        elif not self.hypotheses:
            raise ContractValidationError("a safe hypothesis result requires at least one ranked candidate")
        identities = [(item.depth, item.rank) for item in self.hypotheses]
        if len(identities) != len(set(identities)):
            raise ContractValidationError("hypothesis depth/rank pairs cannot repeat")
        if any(item.taxonomy_identity != self.taxonomy_identity for item in self.hypotheses):
            raise ContractValidationError("hypotheses must bind the same taxonomy identity")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class RetrievalNeighbor(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.retrieval-neighbor.v1"
    IDENTITY_FIELDS = ("neighbor_record_identity", "image_identity", "embedding_identity")

    neighbor_record_identity: str
    image_identity: str
    embedding_identity: str
    taxonomy_identity: str
    distance: float
    similarity: float
    review_status: str
    verified_taxonomy_path: tuple[str, ...]
    reference_cohort_identity: str | None
    truth_projection_identity: str | None
    review_log_identity: str | None
    proposal_taxonomy_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("neighbor_record_identity", "image_identity", "embedding_identity", "taxonomy_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        require_finite(self.distance, "neighbor distance", minimum=0.0)
        similarity = require_finite(self.similarity, "neighbor similarity")
        if not -1.0 <= similarity <= 1.0:
            raise ContractValidationError("neighbor similarity must be from -1 through 1")
        if self.review_status not in {"reviewed", "proposal", "unreviewed"}:
            raise ContractValidationError("review status is not controlled")
        require_unique_text(self.verified_taxonomy_path, "verified taxonomy path")
        require_unique_text(self.proposal_taxonomy_path, "proposal taxonomy path")
        if self.review_status != "reviewed" and self.verified_taxonomy_path:
            raise ContractValidationError("only reviewed neighbors may expose a verified taxonomy path")
        if self.review_status == "reviewed" and not self.verified_taxonomy_path:
            raise ContractValidationError("reviewed neighbors require a verified taxonomy path")
        if self.review_status == "reviewed":
            if (
                self.reference_cohort_identity is None
                or self.truth_projection_identity is None
                or self.review_log_identity is None
            ):
                raise ContractValidationError("reviewed neighbors require cohort, log, and truth projection identities")
            require_text(self.reference_cohort_identity, "reference cohort identity")
            require_text(self.truth_projection_identity, "truth projection identity")
            require_text(self.review_log_identity, "review log identity")
        elif (
            self.reference_cohort_identity is not None
            or self.truth_projection_identity is not None
            or self.review_log_identity is not None
        ):
            raise ContractValidationError("non-reviewed neighbors cannot carry authoritative truth identities")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class RetrievalEvidence(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.retrieval-evidence.v1"
    IDENTITY_FIELDS = ("record_identity", "query_embedding_identity", "index_identity")

    record_identity: str
    query_image_identity: str
    query_embedding_identity: str
    index_identity: str
    taxonomy_identity: str
    reference_cohort_identity: str | None
    review_log_identity: str | None
    neighbors: tuple[RetrievalNeighbor, ...]
    fusion_weights: tuple[tuple[str, float], ...]
    novelty_score: float

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "query_image_identity",
            "query_embedding_identity",
            "index_identity",
            "taxonomy_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        if self.reference_cohort_identity is not None:
            require_text(self.reference_cohort_identity, "retrieval reference cohort identity")
        if self.review_log_identity is not None:
            require_text(self.review_log_identity, "retrieval review log identity")
        require_probability(self.novelty_score, "novelty score")
        names = [name for name, _weight in self.fusion_weights]
        if len(names) != len(set(names)) or not names:
            raise ContractValidationError("fusion weights require unique representation names")
        for name, weight in self.fusion_weights:
            require_text(name, "fusion representation", identifier=True)
            require_finite(weight, "fusion weight", minimum=0.0)
        if sum(weight for _name, weight in self.fusion_weights) <= 0:
            raise ContractValidationError("at least one fusion weight must be positive")
        for neighbor in self.neighbors:
            if neighbor.taxonomy_identity != self.taxonomy_identity:
                raise ContractValidationError("retrieval neighbor taxonomy does not match the evidence taxonomy")
            if neighbor.review_status == "reviewed" and (
                self.reference_cohort_identity is None
                or neighbor.reference_cohort_identity != self.reference_cohort_identity
                or self.review_log_identity is None
                or neighbor.review_log_identity != self.review_log_identity
            ):
                raise ContractValidationError("reviewed neighbor does not bind the retrieval reference truth snapshot")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class MetadataEvidence(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.metadata-evidence.v1"
    IDENTITY_FIELDS = ("record_identity", "metadata_identity")

    record_identity: str
    metadata_identity: str
    claims: tuple[tuple[str, str], ...]
    verified: bool = False

    def __post_init__(self) -> None:
        require_text(self.record_identity, "record identity")
        require_text(self.metadata_identity, "metadata identity")
        names = [name for name, _value in self.claims]
        if len(names) != len(set(names)):
            raise ContractValidationError("metadata claim fields cannot repeat")
        for name, value in self.claims:
            require_text(name, "metadata field", identifier=True)
            require_text(value, "metadata claim")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class ContextEvidence(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.context-evidence.v1"
    IDENTITY_FIELDS = ("record_identity", "context_identity")

    record_identity: str
    context_identity: str
    context_type: str
    claims: tuple[tuple[str, str], ...]
    permitted_by_policy: bool

    def __post_init__(self) -> None:
        require_text(self.record_identity, "record identity")
        require_text(self.context_identity, "context identity")
        require_text(self.context_type, "context type", identifier=True)
        for name, value in self.claims:
            require_text(name, "context claim field", identifier=True)
            require_text(value, "context claim")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class HumanTruthVerification(StrictRecord):
    """Identity-bound proof that a truth projection came from a controlled source."""

    SCHEMA_VERSION = "spritelab.labeling.human-truth-verification.v1"
    IDENTITY_FIELDS = (
        "source",
        "record_identity",
        "taxonomy_identity",
        "event_identity",
        "review_log_identity",
        "cohort_identity",
    )

    source: str
    record_identity: str
    taxonomy_identity: str
    taxonomy_path: tuple[str, ...]
    explicit_abstentions: tuple[str, ...]
    partition: str
    reviewer_identity: str
    event_identity: str
    review_log_identity: str
    chain_tip_identity: str
    image_identity: str
    render_identities: tuple[str, ...]
    evidence_bundle_identity: str
    cohort_identity: str
    source_identity: str
    cluster_identity: str
    leakage_group_identity: str
    duplicate_cluster_identity: str | None = None
    near_duplicate_cluster_identity: str | None = None

    def __post_init__(self) -> None:
        if self.source != "append_only_human_review":
            raise ContractValidationError("human truth verification must come from append-only human review")
        for name in (
            "record_identity",
            "taxonomy_identity",
            "reviewer_identity",
            "image_identity",
            "evidence_bundle_identity",
            "cohort_identity",
            "source_identity",
            "cluster_identity",
            "leakage_group_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        for name in ("event_identity", "review_log_identity", "chain_tip_identity"):
            require_sha256(getattr(self, name), name.replace("_", " "))
        require_unique_text(self.render_identities, "verified human render identities")
        if not self.render_identities:
            raise ContractValidationError("verified human truth requires at least one bound render identity")
        require_unique_text(self.taxonomy_path, "verified human taxonomy path")
        require_unique_text(self.explicit_abstentions, "verified human explicit abstentions")
        if self.partition not in {"reference", "calibration", "holdout"}:
            raise ContractValidationError("verified human truth partition is not controlled")
        for name in ("duplicate_cluster_identity", "near_duplicate_cluster_identity"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name.replace("_", " "))
        self.validate_record()

    @property
    def is_verified_projection(self) -> bool:
        """Whether this instance was minted by the verified review-log projector."""

        return getattr(self, "_projection_seal", None) is _HUMAN_TRUTH_PROJECTION_SEAL

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HumanTruthVerification:
        expected = {"schema_version", *(item.name for item in fields(cls))}
        if set(value) != expected or value.get("schema_version") != cls.SCHEMA_VERSION:
            raise ContractValidationError("human truth verification does not match the exact schema")
        for name in ("taxonomy_path", "explicit_abstentions", "render_identities"):
            if not isinstance(value[name], list) or not all(isinstance(item, str) for item in value[name]):
                raise ContractValidationError(f"human truth verification field {name} must be an array of strings")
        return cls(
            value["source"],
            value["record_identity"],
            value["taxonomy_identity"],
            tuple(value["taxonomy_path"]),
            tuple(value["explicit_abstentions"]),
            value["partition"],
            value["reviewer_identity"],
            value["event_identity"],
            value["review_log_identity"],
            value["chain_tip_identity"],
            value["image_identity"],
            tuple(value["render_identities"]),
            value["evidence_bundle_identity"],
            value["cohort_identity"],
            value["source_identity"],
            value["cluster_identity"],
            value["leakage_group_identity"],
            value["duplicate_cluster_identity"],
            value["near_duplicate_cluster_identity"],
        )


@dataclass(frozen=True, eq=False)
class HumanReferenceLabel(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.human-reference-label.v1"
    IDENTITY_FIELDS = ("record_identity", "review_event_identity", "taxonomy_identity")

    record_identity: str
    review_event_identity: str
    taxonomy_identity: str
    taxonomy_path: tuple[str, ...]
    deepest_accepted_node: str | None
    explicit_abstentions: tuple[str, ...]
    partition: str
    reviewer_identity: str
    verification: HumanTruthVerification | None = None

    def __post_init__(self) -> None:
        for name in ("record_identity", "review_event_identity", "taxonomy_identity", "reviewer_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        require_unique_text(self.taxonomy_path, "human taxonomy path")
        require_unique_text(self.explicit_abstentions, "human explicit abstentions")
        if self.partition not in {"reference", "calibration", "holdout"}:
            raise ContractValidationError("human truth partition is not controlled")
        if self.deepest_accepted_node is None and not self.explicit_abstentions:
            raise ContractValidationError("human reference must accept a node or explicitly abstain")
        if self.deepest_accepted_node is not None:
            require_text(self.deepest_accepted_node, "deepest accepted node", identifier=True)
            if not self.taxonomy_path or self.taxonomy_path[-1] != self.deepest_accepted_node:
                raise ContractValidationError("deepest accepted node must terminate the human taxonomy path")
        overlap = set(self.taxonomy_path) & set(self.explicit_abstentions)
        if overlap:
            raise ContractValidationError("human accepted path and explicit abstentions cannot overlap")
        if self.verification is not None:
            expected = (
                self.record_identity,
                self.taxonomy_identity,
                self.taxonomy_path,
                self.explicit_abstentions,
                self.partition,
                self.reviewer_identity,
                self.review_event_identity,
            )
            observed = (
                self.verification.record_identity,
                self.verification.taxonomy_identity,
                self.verification.taxonomy_path,
                self.verification.explicit_abstentions,
                self.verification.partition,
                self.verification.reviewer_identity,
                self.verification.event_identity,
            )
            if observed != expected:
                raise ContractValidationError("human truth verification does not bind the label fields")
        self.validate_record()

    @property
    def truth_source(self) -> str | None:
        return self.verification.source if self.verification is not None else None

    @property
    def verified(self) -> bool:
        return bool(
            self.verification is not None
            and self.verification.source == "append_only_human_review"
            and self.verification.is_verified_projection
        )

    @property
    def verified_append_only(self) -> bool:
        return self.verified


def _seal_verified_human_truth_projection(verification: HumanTruthVerification) -> HumanTruthVerification:
    """Mint the non-serializable capability after review-log projection validation."""

    verification.__post_init__()
    object.__setattr__(verification, "_projection_seal", _HUMAN_TRUTH_PROJECTION_SEAL)
    return verification


@dataclass(frozen=True, eq=False)
class SyntheticOracleLabel(StrictRecord):
    """Explicit fixture truth that can never satisfy a human-review boundary."""

    SCHEMA_VERSION = "spritelab.labeling.synthetic-oracle-label.v1"
    IDENTITY_FIELDS = ("record_identity", "oracle_event_identity", "taxonomy_identity", "oracle_set_identity")

    record_identity: str
    oracle_event_identity: str
    taxonomy_identity: str
    taxonomy_path: tuple[str, ...]
    deepest_accepted_node: str | None
    explicit_abstentions: tuple[str, ...]
    partition: str
    oracle_set_identity: str
    image_identity: str
    evidence_bundle_identity: str
    cohort_identity: str
    source_identity: str
    cluster_identity: str
    leakage_group_identity: str
    duplicate_cluster_identity: str | None = None
    near_duplicate_cluster_identity: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "taxonomy_identity",
            "oracle_set_identity",
            "image_identity",
            "evidence_bundle_identity",
            "cohort_identity",
            "source_identity",
            "cluster_identity",
            "leakage_group_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        require_sha256(self.oracle_event_identity, "oracle event identity")
        require_unique_text(self.taxonomy_path, "synthetic oracle taxonomy path")
        require_unique_text(self.explicit_abstentions, "synthetic oracle explicit abstentions")
        if self.partition not in {"reference", "calibration", "holdout"}:
            raise ContractValidationError("synthetic oracle partition is not controlled")
        if self.deepest_accepted_node is not None:
            require_text(self.deepest_accepted_node, "synthetic oracle deepest accepted node", identifier=True)
            if not self.taxonomy_path or self.taxonomy_path[-1] != self.deepest_accepted_node:
                raise ContractValidationError("synthetic oracle deepest node must terminate its taxonomy path")
        elif not self.explicit_abstentions:
            raise ContractValidationError("synthetic oracle must accept a node or explicitly abstain")
        if set(self.taxonomy_path) & set(self.explicit_abstentions):
            raise ContractValidationError("synthetic oracle accepted path and abstentions cannot overlap")
        for name in ("duplicate_cluster_identity", "near_duplicate_cluster_identity"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name.replace("_", " "))
        self.validate_record()

    @property
    def review_event_identity(self) -> str:
        """Compatibility name used only by explicitly oracle-scoped diagnostics."""

        return self.oracle_event_identity

    @property
    def truth_source(self) -> str:
        return "synthetic_oracle"


@dataclass(frozen=True, eq=False)
class LabelEvidenceBundle(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.evidence-bundle.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "taxonomy_identity")

    record_identity: str
    image_identity: str
    taxonomy_identity: str
    technical: TechnicalVisualEvidence
    visual_description: VisualDescription | None = None
    visual_attributes: StructuredVisualAttributes | None = None
    taxonomy_hypotheses: tuple[TaxonomyPathHypothesis, ...] = ()
    retrieval: RetrievalEvidence | None = None
    metadata: MetadataEvidence | None = None
    context: ContextEvidence | None = None
    human: HumanReferenceLabel | None = None

    def __post_init__(self) -> None:
        for name in ("record_identity", "image_identity", "taxonomy_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        if (
            self.technical.record_identity != self.record_identity
            or self.technical.image_identity != self.image_identity
        ):
            raise ContractValidationError("technical evidence identity does not match the bundle")
        for evidence in (self.metadata, self.context, self.human, self.retrieval):
            if evidence is not None and evidence.record_identity != self.record_identity:
                raise ContractValidationError("evidence channel record identity does not match the bundle")
        if self.retrieval is not None and self.retrieval.taxonomy_identity != self.taxonomy_identity:
            raise ContractValidationError("retrieval taxonomy identity does not match the bundle")
        if self.retrieval is not None and self.retrieval.query_image_identity != self.image_identity:
            raise ContractValidationError("retrieval query image identity does not match the bundle")
        if self.visual_description is not None:
            if (
                self.visual_description.record_identity != self.record_identity
                or self.visual_description.image_identity != self.image_identity
            ):
                raise ContractValidationError("visual description record/image identity does not match the bundle")
        if self.visual_attributes is not None:
            if self.visual_description is None:
                raise ContractValidationError("structured visual attributes require their bound description")
            if (
                self.visual_attributes.image_identity != self.image_identity
                or self.visual_attributes.description_identity != self.visual_description.identity
            ):
                raise ContractValidationError("structured visual attributes do not bind the bundle description")
        for hypothesis in self.taxonomy_hypotheses:
            if self.visual_description is None:
                raise ContractValidationError("taxonomy hypotheses require their bound visual description")
            if (
                hypothesis.record_identity != self.record_identity
                or hypothesis.taxonomy_identity != self.taxonomy_identity
                or hypothesis.description_identity != self.visual_description.identity
            ):
                raise ContractValidationError("taxonomy hypothesis identity does not match the evidence bundle")
            if any(
                item.taxonomy_identity != self.taxonomy_identity
                or item.render_bundle_identity != self.visual_description.render_bundle_identity
                for item in hypothesis.hypotheses
            ):
                raise ContractValidationError("semantic hypotheses do not bind the bundle taxonomy/render identities")
            response_identities = {
                (
                    item.provider_identity,
                    item.model_identity,
                    item.prompt_identity,
                    item.render_bundle_identity,
                    item.taxonomy_identity,
                )
                for item in hypothesis.hypotheses
            }
            if len(response_identities) > 1:
                raise ContractValidationError(
                    "one semantic response cannot mix provider/model/prompt/render identities"
                )
        if self.human is not None:
            if (
                not isinstance(self.human, HumanReferenceLabel)
                or not self.human.verified
                or self.human.verification is None
                or self.human.truth_source != "append_only_human_review"
            ):
                raise ContractValidationError("human evidence requires a verified append-only review projection")
            if self.human.taxonomy_identity != self.taxonomy_identity:
                raise ContractValidationError("human evidence taxonomy identity does not match the bundle")
            if (
                self.human.verification.image_identity != self.image_identity
                or self.human.verification.evidence_bundle_identity != self.nonhuman_identity
                or (
                    self.visual_description is not None
                    and self.visual_description.render_bundle_identity not in self.human.verification.render_identities
                )
            ):
                raise ContractValidationError("human verification does not bind the bundle image/evidence identity")
        self.validate_record()

    @property
    def nonhuman_identity(self) -> str:
        payload = self.to_dict()
        payload["human"] = None
        return content_identity(self.SCHEMA_VERSION, payload)

    @property
    def contributed_channels(self) -> tuple[str, ...]:
        channels = [EvidenceChannel.TECHNICAL.value]
        if self.visual_description or self.taxonomy_hypotheses:
            channels.append(EvidenceChannel.VISUAL_ONLY.value)
        if self.retrieval:
            channels.append(EvidenceChannel.RETRIEVAL.value)
        if self.metadata:
            channels.append(EvidenceChannel.METADATA.value)
        if self.context:
            channels.append(EvidenceChannel.PACK_CONTEXT.value)
        if self.human:
            channels.append(EvidenceChannel.HUMAN.value)
        return tuple(channels)


@dataclass(frozen=True, eq=False)
class ScoreComponent(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.score-component.v1"
    IDENTITY_FIELDS = ("channel", "name")

    channel: EvidenceChannel
    name: str
    value: float
    weight: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.name, "component name", identifier=True)
        require_finite(self.value, "component value")
        require_finite(self.weight, "component weight")
        require_unique_text(self.reasons, "component reasons")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class FieldDecision(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.field-decision.v1"
    IDENTITY_FIELDS = ("node_id", "decision_policy_identity", "evidence_bundle_identity")

    node_id: str
    depth: int
    state: DecisionState
    raw_score: float | None
    calibrated_probability: float | None
    acceptance_threshold: float | None
    components: tuple[ScoreComponent, ...]
    reasons: tuple[str, ...]
    decision_policy_identity: str
    evidence_bundle_identity: str

    def __post_init__(self) -> None:
        require_text(self.node_id, "decision node", identifier=True)
        if type(self.depth) is not int or self.depth < 0:
            raise ContractValidationError("decision depth must be a non-negative integer")
        if self.raw_score is not None:
            require_probability(self.raw_score, "raw decision score")
        require_probability(self.calibrated_probability, "calibrated probability", optional=True)
        require_probability(self.acceptance_threshold, "acceptance threshold", optional=True)
        require_unique_text(self.reasons, "decision reasons")
        require_text(self.decision_policy_identity, "decision policy identity")
        require_text(self.evidence_bundle_identity, "evidence bundle identity")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class HierarchicalLabelDecision(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.hierarchical-decision.v1"
    IDENTITY_FIELDS = ("record_identity", "taxonomy_identity", "evidence_bundle_identity")

    record_identity: str
    taxonomy_identity: str
    evidence_bundle_identity: str
    taxonomy_path: tuple[str, ...]
    deepest_accepted_node: str | None
    abstained_below_node: str | None
    top_k_alternatives: tuple[str, ...]
    level_decisions: tuple[FieldDecision, ...]
    contributed_channels: tuple[str, ...]
    conflicts: tuple[str, ...]
    calibration_state: CalibrationState

    def __post_init__(self) -> None:
        for name in ("record_identity", "taxonomy_identity", "evidence_bundle_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        require_unique_text(self.taxonomy_path, "accepted taxonomy path")
        require_unique_text(self.top_k_alternatives, "top-k alternatives")
        require_unique_text(self.contributed_channels, "contributed channels")
        require_unique_text(self.conflicts, "decision conflicts")
        if self.deepest_accepted_node is not None:
            require_text(self.deepest_accepted_node, "deepest accepted node", identifier=True)
            if not self.taxonomy_path or self.taxonomy_path[-1] != self.deepest_accepted_node:
                raise ContractValidationError("deepest accepted node must terminate the accepted path")
        if self.abstained_below_node is not None:
            require_text(self.abstained_below_node, "abstained-below node", identifier=True)
        self.validate_record()


@dataclass(frozen=True, eq=False)
class CalibrationResult(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.calibration-result.v1"
    IDENTITY_FIELDS = ("calibration_identity", "taxonomy_identity", "truth_set_identity")

    calibration_identity: str
    taxonomy_identity: str
    truth_set_identity: str
    state: CalibrationState
    fit_partition: str | None
    evaluation_partition: str | None
    sample_size: int
    accepted_count: int
    correct_accepted_count: int
    precision: float | None
    coverage: float | None
    risk: float | None
    calibration_error: float | None
    thresholds: tuple[tuple[str, float], ...]
    bins: tuple[dict[str, Any], ...]
    truth_source: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("calibration_identity", "taxonomy_identity", "truth_set_identity", "truth_source"):
            require_text(getattr(self, name), name.replace("_", " "))
        for name in ("sample_size", "accepted_count", "correct_accepted_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractValidationError(f"{name} must be a non-negative integer")
        if self.correct_accepted_count > self.accepted_count or self.accepted_count > self.sample_size:
            raise ContractValidationError("calibration counts are inconsistent")
        for name in ("precision", "coverage", "risk", "calibration_error"):
            require_probability(getattr(self, name), name, optional=True)
        names = [name for name, _value in self.thresholds]
        if len(names) != len(set(names)):
            raise ContractValidationError("calibration threshold keys cannot repeat")
        for name, value in self.thresholds:
            require_text(name, "threshold key")
            require_probability(value, f"threshold {name}")
        if self.sample_size == 0 and any(value is not None for value in (self.precision, self.coverage, self.risk)):
            raise ContractValidationError("zero human truth rows cannot produce accuracy metrics")
        require_unique_text(self.limitations, "calibration limitations")
        self.validate_record()


@dataclass(frozen=True, eq=False)
class SupervisionExport(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.supervision-export.v1"
    IDENTITY_FIELDS = ("record_identity", "decision_identity", "taxonomy_identity")

    record_identity: str
    decision_identity: str
    taxonomy_identity: str
    taxonomy_path: tuple[str, ...]
    deepest_accepted_node: str | None
    canonical_object: str | None
    role: str | None
    technical_attributes: dict[str, Any]
    visual_attributes: dict[str, Any]
    caption: str | None
    keywords: tuple[str, ...]
    top_k_alternatives: tuple[str, ...]
    evidence_identities: tuple[str, ...]
    calibration_state: CalibrationState
    supervision_tier: SupervisionTier
    recommended_training_weight: float

    def __post_init__(self) -> None:
        for name in ("record_identity", "decision_identity", "taxonomy_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        require_unique_text(self.taxonomy_path, "export taxonomy path")
        require_unique_text(self.keywords, "export keywords")
        require_unique_text(self.top_k_alternatives, "export alternatives")
        require_unique_text(self.evidence_identities, "export evidence identities")
        weight = require_finite(self.recommended_training_weight, "training weight", minimum=0.0)
        if weight > 1.0:
            raise ContractValidationError("recommended training weight cannot exceed human-verified weight 1.0")
        if self.canonical_object is not None:
            require_text(self.canonical_object, "canonical object", identifier=True)
        if self.role is not None:
            require_text(self.role, "role", identifier=True)
        self.validate_record()
