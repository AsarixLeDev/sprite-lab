# Product training

The Training feature is a feature-owned `ProductPlugin`. Integration code registers it by calling:

```python
from spritelab.product_features.training import build_plugin

plugin = build_plugin()
```

The feature does not register itself globally and does not modify the product shell. Its navigation item points to `/training`, and the existing command remains:

```text
python -m spritelab v3 train
```

## Routine experience

The normal page asks for a profile and compute target, not campaign manifests, hashes, random seeds, output paths, or checkpoint paths. Advanced configuration is a collapsed `<details>` section. The four profile names are `recommended`, `fast_preview`, `quality`, and `custom`.

A profile is only a selector. It selects an existing campaign specification from `training.campaign_config` under `product_profiles.<profile>.campaign` or `campaign_path`. The feature passes that specification to the existing `plan_campaign`, `validate_campaign`, `audit_resume`, and `execute_campaign` functions. It does not copy optimizer, schedule, loss, determinism, checkpoint, or resume semantics.

The recommended profile may also select a campaign document directly when the configured document has no `product_profiles` mapping. A custom request must exactly match the configured `product_profiles.custom` campaign; it is not an unbound launch-time override. `fast_preview` remains available for non-production planning, but it is not eligible for conditioned production activation.

## Preparing an accepted dataset

The Training page can publish an optional immutable image-only baseline after one explicit confirmation. This path verifies source hashes, encodes canonical 32x32 arrays, runs dataset and training-manifest QA, checks the trainer loader when available, freezes a deterministic generic vocabulary, and validates a non-launchable three-seed campaign draft.

The baseline is never a production freeze. Its manifest records `artifact_kind: image_only_baseline`, `production_authorized: false`, `training_eligible: false`, and `activation_forbidden: true`. Its campaign draft records both `executable: false` and `launch_authorized: false`. Preparation does not write `dataset.view_manifest`, `dataset.freeze_manifest`, `training.dataset_freeze`, `training.campaign_config`, `execution.allow_dataset_production_freeze`, or `execution.allow_training` and therefore cannot make **Start training** available.

Preparation writes content-addressed artifacts under `.spritelab/training-preparation/baseline-*`, binds their identity to stable item IDs, verified source bytes, and exact preparation-recipe source hashes, and records a whole-publication hash and byte-size inventory. Semantic and hierarchical proposals are intentionally not consumed by this baseline. Persisted artifacts and API responses do not expose absolute local paths. Reuse reconstructs the expected publication independently and refuses changed, re-signed, or semantically promoted content.

Background state is durable under `.spritelab/training-preparation/jobs`. It binds source, config, code, input, result, worker, and job identities. Cross-process locking refuses concurrent starts, a live worker is not interrupted by a second server process, and a dead worker is reconstructed as safely retryable. The privacy-safe event history is atomically rewritten and capped at 200 records. Linked, reparse-point, or hard-linked state and history entries fail closed.

If an image was manually accepted even though the canonical training encoder does not support it, preparation stops without publishing a partial baseline. The Training page shows that current, hash-verified image together with the encoder reasons and directs the user to exclude it from the currently built dataset before retrying. The preview and durable error state expose no source path, and removing an image remains an explicit dataset-review action.

## Conditioned Dataset-v5 activation contract

The web start action fails closed until the exact selected profile satisfies `spritelab.training.conditioned-dataset-contract.v2`. `dataset.freeze_manifest` and `training.dataset_freeze` must contain the same canonical project-relative path; absolute paths, traversal, links, reparse points, and hard links are rejected.

The activation manifest uses `spritelab.dataset.freeze.conditioned.v5`, declares Dataset version 5, `dataset_kind: conditioned`, semantic-label dependence, complete production authorization, and 2,000 through 3,000 images. Its exact ten-artifact set is the view manifest, split manifest, conditioning vocabulary, benchmark manifest, and each independent audit's report, server-managed receipt, and no-replace job action record. Every binding records a canonical publication-relative path, SHA-256, and byte count. Training verifies each report → receipt → action chain and requires the copied action bytes to match the original job-owned record. The full publication inventory records the same data for every regular file other than the activation manifest, plus an exact file count, total bytes, and canonical inventory identity. Unexpected, missing, linked, changed, cross-job, or re-signed entries invalidate activation.

The selected recommended, quality, or custom campaign must be executable and launch-authorized with exactly three standard seeds and 5,000 optimizer steps. It binds the activation bytes plus the exact view, split, vocabulary, and benchmark identities. The campaign config lives outside the frozen publication, avoiding a campaign/freeze hash cycle. `build_conditioned_three_seed_campaign` is the shared no-write builder for the Dataset-v5 publisher.

Phase-J activation durably publishes the immutable activation receipt, record, and `PREPARED` journal, then publishes one fixed no-replace project commit marker that binds those exact documents and the exact prospective configuration bytes. That marker is authoritative. Windows also performs an exact held-handle six-key `spritelab.yaml` compare-and-swap before the marker; POSIX passively loads the marker-bound configuration overlay because portable POSIX has no rename-by-held-descriptor primitive. A canonical file whose SHA-256 matches neither boundary is refused. Passive readiness, Start, and Resume require the marker to bind the current boundary, publication, freeze, and campaign. Restart recovery projects `PREPARED`, finalizes a Windows CAS that preceded the marker, or loads the exact committed overlay without an unsafe pathname replacement. A fixed repository-local interprocess action lock serializes activation with Start and Resume through this commit boundary.

This structural contract does not replace the independent training-infrastructure audit. Applicable reports use `spritelab.training.infrastructure-audit.v2` and `spritelab.training.infrastructure-audit-hashes.v2`, bind the exact activation manifest, campaign config, campaign identity, and training-code identity, and contain exactly these 18 `PASS` gates: tracked code inventory; no untracked production Python; dataset/view/freeze/campaign/vocabulary identity; dataset and training-manifest QA; production-loader coverage; campaign/experiment compatibility; CPU/CUDA smoke evidence; CUDA/driver/Torch/device compatibility; determinism environment qualification; launch-receipt/execution binding; backend command safety; idempotency/concurrency refusal; output-root/resume safety; event-history/migration identity; publication/config atomicity and restart; filesystem containment and link defenses; API/UI privacy; and curated/full test results.

Audit applicability re-hashes every recorded file and dynamically scans all training-bound production roots. Adding even one untracked production `.py` file makes a prior `PASS` stale. Start and every safe resume re-load the selected profile, activation, campaign, audit, and code bindings before any backend operation. Baseline preparation cannot manufacture or activate this contract. A report and hash inventory written by a caller are never sufficient, even when every unkeyed hash is recomputed: applicability also requires the immutable server-managed `spritelab.training.infrastructure-audit-receipt.v1` and the conditioned service's fixed-path `spritelab.training.infrastructure-audit-action-record.v1`. The record is committed under the source job only after the service re-reads all three artifacts while the project configuration is still byte-identical. It binds the operation, current runner and tracked test-harness inventories, prospective configuration, exact activation and smoke artifacts, and fixed command-result identities.

Start, safe Resume, and cloud-challenge issuance open one coherent retained snapshot of the exact report, hash inventory, receipt, and action record while holding the shared Training action lock. The snapshot's aggregate identity is bound into the one-use cloud challenge, durable `spritelab.training.backend-operation.v2` claim, validator context, launch receipt, and serialized compute request. The same live, non-serializable snapshot capability remains open through every adapter `prepare`, `upload`, `launch`, or `resume` seam and is reverified immediately before dispatch. A swapped file, changed byte, mismatched aggregate, closed capability, or hash-only request fails closed; reconstructed durable requests retain the identity for control comparisons but cannot launch a process.

Phase I is an explicit action on the published Conditioned Dataset-v5 job, before Phase J changes configuration:

```text
POST /dataset-v5/api/jobs/{job_id}/training-audit
```

```json
{
  "candidate_identity": "<sha256>",
  "publication_identity_sha256": "<sha256>",
  "activation_manifest_sha256": "<sha256>",
  "campaign_config_sha256": "<sha256>",
  "campaign_identity_sha256": "<sha256>",
  "expected_config_sha256": "<sha256>",
  "smoke_id": "<completed-server-smoke-id>",
  "operation_nonce": "<fresh-8-to-80-character-id>",
  "explicit_action": true
}
```

The browser supplies selectors and exact observed identities only. Gate verdicts, evidence documents, output paths, commands, and test results are rejected. The server holds the configuration lock, derives in memory the same six-key overlay that Phase J would activate, independently reloads that prospective activation, rechecks the CPU/CUDA smoke bundle, scans tracked production Python, runs fixed curated and full test plans, and publishes report, hash inventory, then receipt with exclusive no-replace writes. Each output is first written to an unpredictable same-directory staging inode while its descriptor remains open, then identity-checked, published without replacement from that exact held descriptor, and reread as canonical bytes through the held directory. Windows renames the held source; POSIX links the held inode and either uses anonymous staging or retains and validates the sole named stage alias. Failure handling can quarantine only that owned inode and never a substituted foreign entry. Every audit-bound inventory hash is likewise streamed from an anchored descriptor and requires stable type, device, inode, size, link count, modification time, and filesystem boundary before and after the read. Pytest starts from a fixed isolated bootstrap with caller Python/pytest environment controls removed; every tracked test/config byte is inventoried and an untracked executable test/bootstrap file blocks the audit. The complete registered smoke plan/evidence and run trees—including configs, manifests, bootstrap, reports, metrics, checkpoints, qualification, state, and receipts—are inventoried exactly, so changed, removed, or added files make the audit stale without importing Torch on passive status. The service then rechecks `spritelab.yaml` under the configuration lock and commits the immutable job action record. It starts no training process.

Every completed attempt is immutable, including `FAIL` and `INCONCLUSIVE`. Outputs use fixed filenames either directly under `artifacts/training` or in one managed attempt directory. To retry safely, preserve the prior attempt and first configure fresh absent paths such as `artifacts/training/audits/<new-id>/audit_report.json` and `artifacts/training/audits/<new-id>/audit_hashes.json`; the receipt is derived beside them as `audit_receipt.json`. Other repository, source, dataset, smoke, and publication trees are rejected as audit destinations. Reload the job to obtain the new `expected_config_sha256`, then submit a fresh operation nonce. Never delete or overwrite the earlier evidence. Phase J still writes only the documented six activation keys, so the selected fresh audit paths must already be present in the inactive configuration.

## Pre-activation exploratory smoke registration

The Evaluation Playground exposes a separate, web-operated infrastructure-smoke lane for a completed conditioned publication before configuration activation. The server derives all six publication bindings from the selected conditioned job and prepares immutable plan/config/manifest artifacts under `artifacts/training/smokes/<smoke-id>`. Its CPU and CUDA outputs are fixed under `runs/v3/training-smokes/<smoke-id>/<device>`; every full 5,000-step campaign root remains an absent sentinel.

Every smoke plan, run container, device output root, and exploratory checkpoint snapshot is created directly at its canonical final name while its parent and new directory remain identity-held; publication does not depend on a directory rename. A canonical recursive inventory marker, `.spritelab-publication-complete.json`, is written last and binds the seed files. Missing, partial, substituted, or tampered markers make the directory incomplete and unusable: loaders never adopt or repair it. Device execution may add only its separately validated outputs after the marker, while an exploratory checkpoint snapshot must continue to match its marker-bound inventory exactly.

Explicit CSRF-protected POST actions run CPU and then CUDA, never concurrently. Execution uses a fixed argument array with `shell=False`; CPU binds `CUDA_VISIBLE_DEVICES=-1` and `SPRITELAB_PROGRESS=0`, while CUDA binds `CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `SPRITELAB_PROGRESS=0` before Python starts. Mutable execution state is atomically published beside the immutable plan, retains bounded path-scrubbed log tails, and reconstructs a final receipt after restart. A dead owner without a receipt is `INTERRUPTED` and the bundle is permanently nonresumable.

Each child receives the exact plan-bound minimal environment: provider credentials, `PYTHONPATH`, user-site startup, and unrelated host variables are excluded, while temporary and cache paths point inside the bundle. Linux worker and trainer processes start with `-I -B`; Windows starts the trainer directly through the same compact, content-bound loader. A stdlib-only preflight verifies the immutable plan, environment, exact interpreter bytes, orchestration sources, and the complete production-Python inventory before any Sprite Lab import. A bound source loader then rechecks every imported `spritelab` source file, so a source change between preflight and import fails closed.

On Windows the execution and output roots are created exclusively and labeled inheritable Untrusted integrity while still empty. The one direct trainer process is created suspended with a Low-integrity startup token, assigned to a kill-on-close Job with an active-process limit of one, verified against the held interpreter, and only then resumed. Its stdlib-only outer bootstrap carries the caller's exact restricting-SID set when Codex is already restricted (or creates the audited restriction set otherwise), lowers to Untrusted before the trainer loader executes, proves the inherited Job/token/desktop boundary and protected outside-write probes, and records pathless confinement identities in durable execution state. It never starts a nested smoke worker, so the Job's one-process rule is not weakened. Closing the owning application kills that Job; restart without an exact completion receipt becomes `INTERRUPTED` after the bounded startup grace and requires a fresh bundle.

On Linux the exact interpreter target remains open across worker and trainer process creation. Each child executes the held `/proc/self/fd` target with a parent-death signal and post-`prctl` parent-race check. The outer worker publishes launch-bound heartbeats and a terminal outcome; a receipt alone never marks a still-running worker complete. On either platform, failed image, confinement, heartbeat/outcome where applicable, or receipt checks require a fresh bundle.

Registration revalidates the current publication, freeze, campaign, full training-code identity, real configuration hash, absent campaign roots, device environment, reports, finite step-2 metrics, strict CUDA qualification, and every checkpoint through weights-only loading. It reads both immutable receipts server-side and snapshots the CUDA live/EMA pair beneath `runs/v3/playground/exploratory-checkpoints/<content-id>`. The exploratory catalog is never merged into the production Evaluation catalog and records production, evaluation, resume, campaign-execution, and promotion eligibility as false. Before activation it uses prospective activation validation without an audit; after activation it remains available only when the exact activated config has a current applicable `PASS` infrastructure audit.

The command argv is displayed only in a transparency disclosure. Manual CLI execution, browser-supplied paths, hash transcription, receipt pasting, resume, and promotion are not part of the workflow. Passive page/catalog/status reads launch no subprocess, import no Torch, initialize no CUDA, and create no directories.

## Mandatory launch checks

Before any backend `prepare`, `upload`, or `launch` call, the feature verifies:

- the exact conditioned Dataset-v5 activation contract and freeze-to-campaign hash binding;
- the durable committed activation record bound to the current exact configuration;
- dataset and split identity bindings through campaign validation;
- applicability and `PASS` status of the independent training-infrastructure audit;
- campaign identity and resolved configuration identity;
- project and campaign launch authorization;
- fresh, owned, complete, or safely resumable output roots;
- safe-resume identity and checkpoint schedule rules;
- disk requirements when the campaign/backend supplies an estimate;
- backend device/environment capability without importing Torch or initializing CUDA;
- the complete campaign artifact and completion-marker contract.

Immediately before process creation, a validated launch opens one retained filesystem capability over the exact resolved run config, logical/physical output root, training manifest, conditioning vocabulary, every manifest-referenced NPZ, optional resume checkpoint, and a deterministic ZIP of the receipt-bound production sources. The child starts with `python -I` and a minimal receipt-bound environment; `PYTHON*`, dynamic-loader injection variables, and caller search paths are rejected. A stdlib-only bootstrap verifies the inherited code bundle and its complete source inventory before any `spritelab` import, installs the retained source loader, consumes the input descriptors, clears inheritance and boundary environment state, and hands the trainer immutable bytes/descriptors plus the exact campaign-run contract. The authoritative top-level campaign schedule must exactly match its optimizer projection.

The trainer writes only through the retained physical output root while artifacts continue to record the logical campaign root. Direct control JSON, checkpoint sidecars, event/history metadata, and completion/state documents are content-bound at the parent and child boundary; large checkpoint payloads are separately descriptor-retained and hash-bound for resume. The local adapter keeps the output-root capability until terminal-event capture or explicit cleanup and streams events through descriptor-relative reads, so a renamed root, link substitution, equal-length content swap, or lexical-path redirection fails closed.

The existing repository is intentionally blocked: its Dataset-v5 production freeze is absent, no applicable training-infrastructure audit exists, `execution.allow_training` is false, and no campaign config is selected. The Training page reports these blockers and launches nothing. This feature does not fabricate or bypass audit evidence.

An authoritative campaign refusal that occurs before the first backend `prepare`, `upload`, or `launch` operation claims no durable product run, so the exact campaign remains retryable after its inputs are corrected. The service claims durable run state immediately before the first validated `prepare` call; any failure from that seam onward remains recorded and a new Start is refused in favor of verified safe resume when available.

Cloud backends require a fresh literal `confirm_cloud: true` after the plan has passed. The confirmation is bound to the exact persisted compute-configuration version and backend identity; unsaved compute edits disable Start and Resume, and stale, mismatched, or non-boolean confirmations fail closed before preparation. Pause and Cancel never require or consume cost confirmation. Page load only returns status data; it does not run a connection test or allocate a resource. Connection tests use their own explicit endpoint.

For the local backend, `device_policy: cpu` hides CUDA, while explicit `device_policy: cuda` performs a bounded, read-only `nvidia-smi` visibility check before launch and fails closed when the requested device cannot be verified. `device_policy: auto` remains usable with a disclosed CPU fallback; it does not itself certify CUDA readiness. Torch import, CUDA initialization, and strict-determinism qualification remain explicit smoke or launch operations.

## Dashboard and events

Local and hosted jobs use the same `spritelab.product.event.v1` `ProductEvent` schema. The dashboard aggregates:

- campaign and per-seed progress;
- optimizer step and total steps;
- training and validation loss curves;
- learning rate and optional gradient norm;
- optional GPU utilization and VRAM;
- estimated completion;
- checkpoint schedule and timeline;
- logs, warnings, pause state, and safe-resume availability;
- remote resource uncertainty, possible continuing cost, and shutdown guidance.

Remote checkpoints are not safe-resume points until the backend identity is verified and the artifact has been downloaded and hash-verified. Unsafe resume is never exposed.

The initial `/training` HTML response does not replay run events. The browser lazily reconstructs the latest dashboard once through the pathless API, then starts event streaming only after a successful Start or an explicit **Follow live** action. The returned durable event cursor prevents historical events from being applied twice. Every Training public text projection uses the shared event redactor, removing project/private absolute paths, credential assignments, bearer/basic tokens, URL credentials, and recognizable provider-token values without mutating durable backend evidence.

## Intermediate previews

`PreviewScheduler` runs only at the intersection of the configured checkpoint schedule and preview interval. Prompts and generation seeds are fixed in `PreviewConfiguration`; each event records the checkpoint, training seed, prompt, generation seed, parameters, and output path. Outputs live under the run's `previews/checkpoint_<step>/seed_<seed>/` directory.

Every preview is marked `exploratory: true`, `benchmark_evidence: false`, and `promotion_evidence: false`. Preview generation can be disabled. A preview exception emits a warning event and never changes the training status.

## Feature routes

- `GET /training/api/preparation` - passive background-preparation state and privacy-safe logs.
- `POST /training/api/preparation` - explicitly authorized immutable image-only baseline preparation using `authorize_baseline: true`; rejects freeze or training authorization and starts no training process.
- `GET /training/api/preparation/error-image` - current hash-verified source preview for the one accepted image that blocked preparation; available only while that failure remains current.

- `GET /training` — page shell; no backend probe or launch.
- `GET /training/api/state` — current plan, exact-profile conditioned activation contract, and blockers; device check deferred.
- `GET /training/api/settings` — redacted compute settings.
- `POST /training/api/connection-test` — explicit backend probe.
- `POST /training/api/start` — revalidates the selected profile, custom input, activation, audit, and campaign before backend work.
- `POST /dataset-v5/api/jobs/{job_id}/training-audit` — runs the server-managed Phase-I audit against the exact prospective six-key activation overlay without changing configuration or starting training.
- `GET /training/api/runs/{run_id}` — refreshes events and dashboard data.
- `POST /training/api/runs/{run_id}/pause` — graceful interruption request.
- `POST /training/api/runs/{run_id}/resume` — revalidates the retained activation and campaign, then performs safe resume only.
