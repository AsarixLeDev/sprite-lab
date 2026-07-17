# Conditioned Dataset-v5 web workflow

Open **Dataset v5** at `/dataset-v5`. The feature is offline and repository-local after the explicit Harvest **Import into Dataset** action: passive inventory does not construct a backend, contact a provider, initialize CUDA, or mutate an input.

## Flow

1. In Harvest, explicitly import each completed `spritelab.harvest.dataset-handoff.v2` into Dataset. The callback revalidates the final handoff and every raw artifact, copies rather than moves the bytes into unique managed work, rehashes the raw tree and copy, binds a pack sidecar, runs the ordinary `DatasetIntakeService`, and validates the completed managed output. Only then does it atomically publish an opaque `dataset.<identity>` receipt. Failed unique work remains safely unreferenced for inspection; raw Harvest bytes are never rewritten or removed.
2. Select one or more of those opaque managed Dataset imports. Preview and build revalidate the import receipt, complete copied-source and intake-output inventories, provenance sidecar/grouping, original artifact manifest and handoff, and the Harvest Dataset-import receipt. Browser requests never accept a Harvest run ID or filesystem path as candidate input.
3. Preview deterministic conditioning. Only Dataset-intake-accepted exact 32×32 PNGs with hard alpha, strict CC0-1.0/public-domain provenance, and a non-conflicting known taxonomy category are eligible. Unknown and category-disagreement rows are excluded. Selection uses equal source quotas and deterministic per-source category round-robin; no augmentation is used.
4. Build a durable candidate. Exact byte/pixel duplicates are removed. Conservative near duplicates and normalized source parent/family groups are kept in one train/validation/test split. The output contains Phase-7 NPZ arrays, portable manifests, `semantic_v3`, conditioning vocabulary, a source-group-disjoint benchmark, provenance, coverage, duplicate/split reports, and local dataset/training-manifest/loader checks.
5. Supply independent reports with schemas `spritelab.audit.conditioned-labels.v1` and `spritelab.audit.conditioned-dataset.v1`. Each PASS report must identify an independent auditor/code/run, bind the exact candidate identity, enumerate the complete `{sha256, byte_count}` candidate inventory, and PASS every mandatory gate. The candidate builder never emits its own PASS evidence.
6. Explicitly authorize one exact freeze once. Publication refuses replacement and creates `datasets/conditioned-v5-<content identity>/activation.json` with schema `spritelab.dataset.freeze.conditioned.v5`, complete byte/hash inventory, evidence bindings, and 2,000–3,000 image count. It then calls the public Training activation helper to create an executable, launch-authorized, exact three-seed 5,000-step campaign bound to the activation hash.

Publication does **not** modify `spritelab.yaml`, enable `execution.allow_training`, or start training. Activation and the independent training-infrastructure audit remain separate explicit actions.

## Durable states and API

Build jobs live below `runs/v3/conditioned-dataset-v5/`. A persisted `RUNNING` job with no live worker is projected as `INTERRUPTED`; retrying creates a fresh job and never deletes the interrupted artifacts. Browser writes use the product shell's CSRF middleware and accept no filesystem paths or URLs.

- `GET /dataset-v5/api/inventory`
- `POST /dataset-v5/api/preview`
- `POST /dataset-v5/api/jobs`
- `GET /dataset-v5/api/jobs/{job_id}`
- `POST /dataset-v5/api/jobs/{job_id}/cancel`
- `POST /dataset-v5/api/jobs/{job_id}/evidence`
- `POST /dataset-v5/api/jobs/{job_id}/publish`

The production policy targets 2,500 images and permits 2,000–3,000 inclusive. Smaller bounds are injectable only when constructing a service directly for isolated tests; the built-in plugin always uses production bounds, and the Training activation loader independently enforces the same range.
