# Conditioned Dataset-v5 web workflow

Open **Dataset v5** at `/dataset-v5`. The feature is offline and repository-local after the explicit Harvest **Import into Dataset** action: passive inventory does not construct a backend, contact a provider, initialize CUDA, or mutate an input.

## Flow

1. In Harvest, explicitly import each completed `spritelab.harvest.dataset-handoff.v2` into Dataset. The callback revalidates the final handoff and every raw artifact, copies rather than moves the bytes into unique managed work, rehashes the raw tree and copy, binds transaction-local grouping and pack sidecars, runs the ordinary `DatasetIntakeService` below `work/datasets/managed`, and validates the completed managed output. The callback holds the project-to-work directory chain while it works. Only an atomic no-replace `dataset.<identity>` receipt makes the import visible; that rename is the commit point, so a post-commit reload fault leaves no contradictory unpublished failure record and an exact retry loads the immutable receipt. Failed unique work remains safely unreferenced for inspection; raw Harvest bytes are never rewritten or removed.
2. Select one or more of those opaque managed Dataset imports. Preview and build revalidate the import receipt, complete copied-source and intake-output inventories, provenance sidecar/grouping, original artifact manifest and handoff, and the Harvest Dataset-import receipt. Browser requests never accept a Harvest run ID or filesystem path as candidate input.
3. Preview deterministic conditioning. Only Dataset-intake-accepted exact 32×32 PNGs with hard alpha, strict CC0-1.0/public-domain provenance, and a non-conflicting known taxonomy category are eligible. Unknown and category-disagreement rows are excluded. Selection uses equal source quotas and deterministic per-source category round-robin; no augmentation is used.
4. Build a durable candidate. Exact byte/pixel duplicates are removed. Conservative near duplicates and normalized source parent/family groups are kept in one train/validation/test split. The output contains Phase-7 NPZ arrays, portable manifests, `semantic_v3`, conditioning vocabulary, a source-group-disjoint benchmark, provenance, coverage, duplicate/split reports, and local dataset/training-manifest/loader checks.
5. Supply independent reports with schemas `spritelab.audit.conditioned-labels.v1` and `spritelab.audit.conditioned-dataset.v1`. Each PASS report must identify an independent auditor/code/run, bind the exact candidate identity, enumerate the complete `{sha256, byte_count}` candidate inventory, and PASS every mandatory gate. The candidate builder never emits its own PASS evidence.
6. Explicitly authorize one exact freeze once. Publication refuses replacement and creates `datasets/conditioned-v5-<content identity>/activation.json` with schema `spritelab.dataset.freeze.conditioned.v5`, complete byte/hash inventory, evidence bindings, and 2,000–3,000 image count. It then calls the public Training activation helper to create an executable, launch-authorized, exact three-seed 5,000-step campaign bound to the activation hash.
7. Explicitly activate the exact published freeze and campaign. Activation compare-and-swaps the observed `spritelab.yaml` SHA-256, revalidates the applicable training-infrastructure audit, writes only the six Dataset/training readiness keys, publishes an immutable activation receipt, and records the one-time authorization. A stale configuration or changed freeze/campaign is refused. Activation never starts training.

The browser controller runs one request at a time, ignores duplicate clicks while an action is in flight, and resets the unchecked publication/activation authorizations whenever a different job is selected.

Publication does **not** modify `spritelab.yaml`, enable `execution.allow_training`, or start training. Activation and the independent training-infrastructure audit remain separate explicit actions. Fresh publication and activation outputs are renamed through held parent-directory handles. Rollback never recursively deletes a tree: an exact still-owned inode is moved to an unpredictable residue, content drift is retained byte-for-byte in a drift residue, and a substituted foreign inode is refused and left untouched while other owned outputs are still rolled back.

The candidate and evidence bindings include `spritelab.dataset.conditioned-code-inventory.v2`. It hashes the full recursively resolved first-party import closure, including integration-only imports and every executed parent-package initializer, and binds the installed `numpy`, `Pillow`, and `PyYAML` versions. Label-audit and dataset-validation trust use distinct transitive auditor inventories, so changing a helper invalidates the applicable prior report.

The pathname-oriented Dataset-intake/sidecar implementation runs only in an OS-confined child. On Windows, the parent applies one inheritable Untrusted mandatory-integrity `NO_WRITE_UP` SACL to the exact newly-created, empty, unique work root before creating or copying any descendants. Launch is verify-only over the bounded link-free tree; it never recursively changes ACLs, ownership, or modes. Only three exact private stdio files and the fixed, repository-local Medium/Low probe roots receive additional individually enumerated labels. A pinned interpreter starts Low, an audited prefix lowers its primary token to Untrusted before worker imports, and a one-process kill-on-close Job prevents descendants. This protects ordinary unlabeled, Low, and Medium objects; an object deliberately granting World write access at Untrusted integrity remains explicitly outside the guarantee. Linux uses inherited directory descriptors plus Landlock and `no_new_privs`; unsupported platforms fail closed.

Worker stderr is bounded and retained only in the unique private scratch tree. It is never copied into receipts, jobs, audit reports, API responses, or other portable artifacts; durable failures contain only a controlled error code.

## Durable states and API

Build jobs live below `runs/v3/conditioned-dataset-v5/`. A persisted `RUNNING` job with no live worker is projected as `INTERRUPTED`; retrying creates a fresh job and never deletes the interrupted artifacts. Browser writes use the product shell's CSRF middleware and accept no filesystem paths or URLs.

- `GET /dataset-v5/api/inventory`
- `POST /dataset-v5/api/preview`
- `POST /dataset-v5/api/jobs`
- `GET /dataset-v5/api/jobs/{job_id}`
- `POST /dataset-v5/api/jobs/{job_id}/cancel`
- `POST /dataset-v5/api/jobs/{job_id}/evidence`
- `POST /dataset-v5/api/jobs/{job_id}/publish`
- `POST /dataset-v5/api/jobs/{job_id}/activate`

The production policy targets 2,500 images and permits 2,000–3,000 inclusive. Smaller bounds are injectable only when constructing a service directly for isolated tests; the built-in plugin always uses production bounds, and the Training activation loader independently enforces the same range.
