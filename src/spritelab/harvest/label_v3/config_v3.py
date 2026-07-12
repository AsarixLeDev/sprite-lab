"""Auto-Labeling v3: configuration and policy hashing."""

from __future__ import annotations

from dataclasses import dataclass, field

from spritelab.harvest.label_v3.impossible_combinations import impossible_combinations_hash
from spritelab.harvest.label_v3.sha256_utils import dict_hash
from spritelab.harvest.label_v3.taxonomy_v3 import taxonomy_version_hash


@dataclass(frozen=True)
class V3LabelingPolicy:
    """Versioned operating policy for a v3 labeling run."""

    policy_version: str = "v3.1.0"
    taxonomy_version: str = "v3.1"
    precision_target_category: float = 0.99
    precision_target_canonical_object: float = 0.99
    precision_target_color: float = 0.95
    precision_target_material: float = 0.95
    precision_target_shape: float = 0.90
    precision_target_tags: float = 0.90
    confidence_level: float = 0.95
    min_calibration_samples: int = 30
    min_calibration_samples_per_stratum: int = 10
    auto_accept_enabled: bool = True
    partial_accept_enabled: bool = True
    shadow_mode: bool = True
    dry_run_apply: bool = True
    max_vlm_calls_per_sprite: int = 6
    promote_threshold_coverage: float = 0.05

    def policy_hash(self) -> str:
        data = {
            "policy_version": self.policy_version,
            "taxonomy_version": self.taxonomy_version,
            "precision_target_category": self.precision_target_category,
            "precision_target_canonical_object": self.precision_target_canonical_object,
            "precision_target_color": self.precision_target_color,
            "precision_target_material": self.precision_target_material,
            "precision_target_shape": self.precision_target_shape,
            "precision_target_tags": self.precision_target_tags,
            "confidence_level": self.confidence_level,
            "min_calibration_samples": self.min_calibration_samples,
            "auto_accept_enabled": self.auto_accept_enabled,
            "partial_accept_enabled": self.partial_accept_enabled,
            "taxonomy_hash": taxonomy_version_hash(),
            "impossible_combinations_hash": impossible_combinations_hash(),
        }
        return dict_hash(data)


@dataclass(frozen=True)
class V3PipelineConfig:
    """Configuration for a v3 labeling pipeline run."""

    policy: V3LabelingPolicy = field(default_factory=V3LabelingPolicy)
    vlm_backend: str = "none"
    vlm_model: str = ""
    vlm_base_url: str = "http://127.0.0.1:8000/v1"
    vlm_api_key: str = "not-needed"
    vlm_structured_output: str = "auto"
    vlm_prompt_version: str = "vlm_prefill_v3_2"
    # ``fast`` keeps the two stages that add distinct review value: a blind
    # visual description and constrained classification. Deterministic image
    # analysis already supplies morphology, while verification calls from the
    # same model are correlated evidence and do not increase fusion weight.
    vlm_cascade_profile: str = "fast"
    vlm_disable_thinking: bool = True
    vlm_timeout_seconds: float = 60.0
    vlm_retries: int = 1
    vlm_concurrency: int = 1
    vlm_retry_backoff_seconds: float = 1.0
    vlm_cache_dir: str = ""
    vlm_failure_diagnostics_enabled: bool = True
    vlm_failure_diagnostics_dir: str = ""
    text_enrichment_enabled: bool = False
    text_enrichment_model: str = ""
    text_enrichment_backend: str = "none"
    text_enrichment_base_url: str = ""
    text_enrichment_api_key: str = ""
    text_enrichment_timeout_seconds: float = 60.0
    text_enrichment_retries: int = 1
    vlm_image_view: str = "magenta_matte"
    vlm_include_filename_hint: bool = False
    blind_description_enabled: bool = True
    morphology_extraction_enabled: bool = True
    constrained_classification_enabled: bool = True
    open_set_verification_enabled: bool = True
    consistency_verification_enabled: bool = True
    embeddings_enabled: bool = False
    embeddings_model: str = ""
    calibration_artifact_path: str = ""
    use_existing_calibration: bool = True
    workers: int = 1
    shard_index: int = 0
    shard_count: int = 1
    max_records: int | None = None
    run_dir: str = ""

    def pipeline_hash(self) -> str:
        data = {
            "vlm_backend": self.vlm_backend,
            "vlm_model": self.vlm_model,
            "vlm_base_url": self.vlm_base_url,
            "vlm_structured_output": self.vlm_structured_output,
            "vlm_prompt_version": self.vlm_prompt_version,
            "vlm_cascade_profile": self.vlm_cascade_profile,
            "vlm_disable_thinking": self.vlm_disable_thinking,
            "text_enrichment_backend": self.text_enrichment_backend,
            "text_enrichment_model": self.text_enrichment_model,
            "text_enrichment_base_url": self.text_enrichment_base_url,
            "vlm_image_view": self.vlm_image_view,
            "vlm_include_filename_hint": self.vlm_include_filename_hint,
            "blind_description_enabled": self.blind_description_enabled,
            "morphology_extraction_enabled": self.morphology_extraction_enabled,
            "constrained_classification_enabled": self.constrained_classification_enabled,
            "open_set_verification_enabled": self.open_set_verification_enabled,
            "consistency_verification_enabled": self.consistency_verification_enabled,
            "embeddings_enabled": self.embeddings_enabled,
            "embeddings_model": self.embeddings_model,
            "policy_hash": self.policy.policy_hash(),
        }
        return dict_hash(data)
