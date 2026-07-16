"""Hierarchical, retrieval-assisted semantic labeling for Sprite Lab.

The package is intentionally additive.  Existing conservative labeling and
blind-audit contracts remain authoritative for their certified scopes while
this package provides the versioned architecture needed to improve useful
broad-label coverage after independent human calibration.
"""

from spritelab.hierarchical_labeling.cascade import CascadeProfile
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
from spritelab.hierarchical_labeling.review import HumanReviewEvent
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph, TaxonomyNode, load_default_taxonomy

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
