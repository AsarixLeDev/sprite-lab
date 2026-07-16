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

The playground accepts only a typed `PlaygroundGenerator`. There is no Python, shell, command, provider-string, or arbitrary-code input. No real generator is configured by the plugin itself. Tests supply a fake adapter and never initialize CUDA, a model, a provider, or a checkpoint loader.

The selected checkpoint must still be eligible under the evaluation checkpoint catalog. If the user changes live/EMA, an eligible sibling at the same verified run and step must exist.

## Reproducibility record

Each image record contains:

- opaque checkpoint identity, run identity, step, and live/EMA variant;
- prompt and effective seed;
- sampling steps, guidance, and requested image count;
- aware UTC timestamp;
- SHA-256 output hash;
- application version and media type.

Records explicitly set `scope: EXPLORATORY`, `frozen_benchmark_eligible: false`, and `promotion_evidence_eligible: false`. A generation-level metadata file repeats those exclusions. Outputs live under the playground generation directory and are never appended to a benchmark manifest.

## Prompt presets

Presets store only validated `GenerationRequest` fields. They contain no executable data. Saving a preset does not generate. Rerun is itself an explicit generation action and repeats billable confirmation when applicable; the seed may be overridden for a new reproducible variant.
