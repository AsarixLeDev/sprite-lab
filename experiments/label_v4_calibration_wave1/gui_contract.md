# Labeling-v4 calibration GUI contract

## Root cause

Before this fix, `assisted-v4 --records` had no schema validation. Its presenter accepted any mapping containing `field_proposals` or `reconciliation.field_proposals`, so it effectively accepted audit selection rows, resolved candidate rows, Labeling-v4 prediction records, and human-golden-like rows. The frozen wave-1 selection happens to contain provisional sampling projections under `field_proposals`; the GUI therefore rendered those projections as if they were completed predictions. Missing B/C output became a null value with the sampling uncertainty of 20 instead of an explicit missing-stage state.

For `acq_craftpix_minerals_icon29`, the image resolver succeeds and Stage A runs. Its generic filename does not yield a canonical object, and no compatible cached B/C artifact exists. The observed null was therefore caused by a schema mismatch and missing prefill record, with B/C not executed. It was not a genuine model abstention and not a resolver failure.

## Separate schemas

- `label_v4_audit_selection_v1`: frozen representative selection and sampling metadata. It is input to preparation, never directly reviewable. The already-frozen wave-1 file retains its historical `label_v4_calibration_wave1_v1` marker and is recognized as this contract without mutation.
- `label_v4_prefilled_audit_record_v1`: complete review projection containing resolved image/source metadata, Stage A, compatible cached evidence, all review fields with value states, field/record risks, suitability context, and immutable model provenance. This is the only normal `assisted-v4` input.
- `label_v4_human_truth_v1`: append-only field or record review outcomes with proposal provenance, review mode, visibility timing, start/completion timestamps, and duration. It is output only.

Ordinary Labeling-v4 prediction records (`label_record_v4.*`) remain pipeline artifacts and must be prepared before review. Resolved candidate rows have no review schema and are rejected.

## Fail-closed behavior

Raw selection input produces:

```text
Input schema is label_v4_audit_selection_v1.
The assisted review GUI requires label_v4_prefilled_audit_record_v1.
Run label-v4-prepare-audit first.
```

Diagnostic selection mode only confirms the contract mismatch; it does not silently convert the selection into a review record.

## Null states

Every field uses one of: `known`, `model_abstained`, `not_applicable`, `not_scorable`, `missing_prediction`, `provider_failed`, or `unsupported`. Every null carries a reason. `missing_prediction` means a required model stage did not run; `model_abstained` is used only when that model stage completed and did not promote a value.

## Review flow

Suitability is decided before semantics. Unsuitable and uncertain-quality records are saved at record level as `not_scorable_due_to_image`; semantic proposals remain unchanged inside model provenance. Suitable records support critical-field bulk acceptance or field-level acceptance, alternative choice, edit, abstention, unsupported, wrong-taxonomy, and not-applicable outcomes. Human truth is append-only and never overwrites the prefill record.

Every fifth record is a deterministic blind subset. Proposal, uncertainty, alternatives, and evidence remain hidden until its first critical-field judgment.

## Viewer

The browser receives the original PNG bytes as a data URI. CSS uses `image-rendering: pixelated`/`crisp-edges`, a transparency checkerboard, 1×/8×/12×/16× choices, native dimensions, and an optional tight-alpha crop. Default 12× display guarantees at least 384×384 for a 32×32 sprite. No resized image is written and Labeling-v4 continues to use the source image.
