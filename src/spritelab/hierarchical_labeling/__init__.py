"""Hierarchical, retrieval-assisted semantic labeling for Sprite Lab.

The package is intentionally additive.  Existing conservative labeling and
blind-audit contracts remain authoritative for their certified scopes while
this package provides the versioned architecture needed to improve useful
broad-label coverage after independent human calibration.
"""

ARCHITECTURE_VERSION = "sprite_lab_hierarchical_labeling_v1"

__all__ = [
    "ARCHITECTURE_VERSION",
    "CalibrationResult",
    "CascadeProfile",
    "FieldDecision",
    "HierarchicalLabelDecision",
    "HumanReviewEvent",
    "LabelEvidenceBundle",
    "SemanticHypothesis",
    "SupervisionExport",
    "TaxonomyGraph",
    "TaxonomyNode",
    "TechnicalVisualEvidence",
    "VisualDescription",
    "load_default_taxonomy",
]


def __getattr__(name: str) -> object:
    if name == "CascadeProfile":
        from spritelab.hierarchical_labeling.cascade import CascadeProfile

        return CascadeProfile
    if name in {
        "CalibrationResult",
        "FieldDecision",
        "HierarchicalLabelDecision",
        "LabelEvidenceBundle",
        "SemanticHypothesis",
        "SupervisionExport",
        "TechnicalVisualEvidence",
        "VisualDescription",
    }:
        from spritelab.hierarchical_labeling.contracts import (
            CalibrationResult,
            FieldDecision,
            HierarchicalLabelDecision,
            LabelEvidenceBundle,
            SemanticHypothesis,
            SupervisionExport,
            TechnicalVisualEvidence,
            VisualDescription,
        )

        return {
            "CalibrationResult": CalibrationResult,
            "FieldDecision": FieldDecision,
            "HierarchicalLabelDecision": HierarchicalLabelDecision,
            "LabelEvidenceBundle": LabelEvidenceBundle,
            "SemanticHypothesis": SemanticHypothesis,
            "SupervisionExport": SupervisionExport,
            "TechnicalVisualEvidence": TechnicalVisualEvidence,
            "VisualDescription": VisualDescription,
        }[name]
    if name == "HumanReviewEvent":
        from spritelab.hierarchical_labeling.review import HumanReviewEvent

        return HumanReviewEvent
    if name in {"TaxonomyGraph", "TaxonomyNode", "load_default_taxonomy"}:
        from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph, TaxonomyNode, load_default_taxonomy

        return {
            "TaxonomyGraph": TaxonomyGraph,
            "TaxonomyNode": TaxonomyNode,
            "load_default_taxonomy": load_default_taxonomy,
        }[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
