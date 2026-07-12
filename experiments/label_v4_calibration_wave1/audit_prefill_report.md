# Labeling-v4 calibration audit prefill

The GUI consumes `label_v4_prefilled_audit_record_v1`; the frozen audit manifest is selection metadata, not a prediction record.

Inference policy: `cached-only`. Provider calls allowed: `false`. Provider calls made: `0`.

## Current coverage

- Records total: 100
- Records with complete deterministic critical semantics: 0
- Records with compatible cached rich-VLM predictions: 2
- Records requiring missing B/C inference: 98
- Records with genuine model abstentions: 1
- Records with quality quarantine: 60

## Reported mineral record

`acq_craftpix_minerals_icon29` resolved to `C:\Users\Mathieu\Documents\sprite-lab\harvest_runs\acq_diversity_v1_craftpix_minerals\extracted\acq_craftpix_minerals\PNG\Transperent\Icon29.png`. Its prediction state is `missing_required_model_stage` and its canonical-object state is `missing_prediction` because `rich_vlm_stage_not_executed`. The original GUI null was a raw-selection/prediction schema mismatch plus a missing prefill and missing B/C stages, not a genuine abstention and not a resolver failure.

Missing model stages are represented as `missing_prediction`, never as semantic abstention.
