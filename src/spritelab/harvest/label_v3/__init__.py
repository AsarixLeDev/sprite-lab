"""Auto-Labeling v3 package.

Versioned, precision-first labeling system for pixel-art sprites.
Operates alongside Labeling v2 without modifying its behavior.
"""

from spritelab.harvest.label_v3.adapter import (
    build_legacy_safe_fused_label,
    expose_accepted_v3_fields,
    v3_field_to_legacy_suggestion,
)
from spritelab.harvest.label_v3.config_v3 import (
    V3LabelingPolicy,
    V3PipelineConfig,
)
from spritelab.harvest.label_v3.evidence import (
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
)
from spritelab.harvest.label_v3.evidence import (
    CalibrationStratum,
    EvidenceFamily,
    EvidenceItem,
    EvidenceProducer,
    TargetField,
)
from spritelab.harvest.label_v3.field_decisions import (
    AcceptedTagSet,
    FieldDecision,
    FieldState,
    TagDecision,
)
from spritelab.harvest.label_v3.field_prefill import (
    FieldPrefill,
    PrefillAlternative,
    build_prefills,
)
from spritelab.harvest.label_v3.reason_codes import (
    CONTRADICTION_CODES,
    REASON_CODES,
    contradiction_action,
    contradiction_severity,
)
from spritelab.harvest.label_v3.record_decisions import (
    ReasonCode,
    RecordDecision,
    RecordState,
    derive_record_state,
)
from spritelab.harvest.label_v3.taxonomy_v3 import (
    TAXONOMY_VERSION,
    HierarchyNode,
    broader_hierarchy_node,
    deepest_supported_node,
    get_hierarchy_node,
)

__all__ = [
    "CONTRADICTION_CODES",
    "EVIDENCE_SCHEMA_VERSION",
    "REASON_CODES",
    "TAXONOMY_VERSION",
    "AcceptedTagSet",
    "CalibrationStratum",
    "EvidenceFamily",
    "EvidenceItem",
    "EvidenceProducer",
    "FieldDecision",
    "FieldPrefill",
    "FieldState",
    "HierarchyNode",
    "PrefillAlternative",
    "ReasonCode",
    "RecordDecision",
    "RecordState",
    "TagDecision",
    "TargetField",
    "V3LabelingPolicy",
    "V3PipelineConfig",
    "broader_hierarchy_node",
    "build_legacy_safe_fused_label",
    "build_prefills",
    "contradiction_action",
    "contradiction_severity",
    "deepest_supported_node",
    "derive_record_state",
    "expose_accepted_v3_fields",
    "get_hierarchy_node",
    "v3_field_to_legacy_suggestion",
]
