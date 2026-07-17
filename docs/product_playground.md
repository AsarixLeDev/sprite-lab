# Exploratory prompt playground

The prompt playground is a small, explicit generation surface inside the evaluation feature. Every output is labeled `EXPLORATORY` and stored separately from evaluation results.

## Controls and defaults

The visible controls are prompt, eligible checkpoint, live or EMA weights, seed, and number of images. Sampling steps and CFG/guidance remain under Advanced settings.

Defaults are:

- seed `42`;
- EMA weights;
- `30` sampling steps;
- guidance `3.0`;
- `4` images.

The Generate button is the explicit action boundary. Merely opening the page, loading defaults, listing presets, or viewing a saved preset does not invoke a generator. A remote or billable adapter adds a separate cost-confirmation requirement.

## Typed generator boundary

The playground accepts only a typed `PlaygroundGenerator`. There is no Python, shell, command, provider-string, or arbitrary-code input. The product configures a local-only, non-billable challenger adapter. It imports the sampler and Torch only after the explicit Generate action; constructing the router, opening the page, inspecting checkpoints, and loading presets remain passive.

The local adapter creates a fresh work directory below the project run root for every request. It never invokes a shell or a provider. A durable cross-process lease permits one local sampler at a time, records heartbeats, and marks a dead predecessor as an orphan that is safe to retry. The prompt becomes bounded JSONL data, never a path. The adapter validates the complete tracked production-Python inventory (and rejects untracked production Python), checkpoint confinement, sampler manifest semantics, Unicode-normalized path collisions, single-link output files, PNG signature, decoded frame count from the exact returned bytes, exact 32x32 dimensions, output count, and per-file byte limit before returning any bytes to the durable Playground run. Sampler diagnostics remain private exploratory work material and are not exposed through the product API, promoted, or copied into Dataset-v5.

The selected checkpoint must have an explicit durable per-file SHA-256 and still be eligible under a freshly discovered evaluation checkpoint catalog. If the user changes live/EMA, an eligible sibling at the same verified run and step must exist. The adapter copies the selected regular single-link file once into an exclusive snapshot while hashing it, requires the catalog hash, and loads only that snapshot with PyTorch's safe weights-only loader. It verifies challenger model type, step/global-step agreement, and live/EMA metadata before sampling. Eligibility and adapter code identity are checked again before completion.

## Reproducibility record

Each image record contains:

- opaque checkpoint identity, run identity, step, and live/EMA variant;
- prompt and effective seed;
- sampling steps, guidance, and requested image count;
- aware UTC timestamp;
- SHA-256 output hash;
- application version and media type.

Records explicitly set `scope: EXPLORATORY`, `frozen_benchmark_eligible: false`, and `promotion_evidence_eligible: false`. A generation-level metadata file repeats those exclusions. Outputs live under the playground generation directory and are never appended to a benchmark manifest.

The generation adapter identity also binds all tracked production Python under `src/spritelab`; any untracked production Python fails closed. A later code change therefore changes the recorded identity rather than silently appearing to be the same generator. The report records bounded Python, PyTorch, CUDA-availability/runtime, platform, and selected-device identity without exposing sampler filesystem paths.

## Prompt presets

Presets store only validated `GenerationRequest` fields. They contain no executable data. Saving a preset does not generate. Rerun is itself an explicit generation action and repeats billable confirmation when applicable; the seed may be overridden for a new reproducible variant.
