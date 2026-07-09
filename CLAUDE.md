# sprite-lab developer reference

## Commands

```powershell
# Install (CPU torch for tests)
pip install -e ".[dev,ml,harvest,prefill]"

# Full test suite
python -m pytest -q --basetemp=.pytest_tmp_spritelab -p no:cacheprovider

# Lint / format
ruff check . && ruff format --check .

# Type-check (lenient baseline)
mypy src

# Pre-commit (installed, runs on commit)
pre-commit run --all-files

# CLI entry points
spritelab train <subcommand>
spritelab harvest <subcommand>
spritelab ml <subcommand>
spritelab dataset-maker [import-export|prefill|...]
```

## Architecture map

```
src/spritelab/
  __main__.py              — dict-registry dispatcher (curation|train|harvest|ml|dataset-maker|...)
  codec/                   — sprite encode/decode/palette/canonicalize/role-infer
  curation/                — SpriteBundle curation browser + manifest decisions
  data/                    — quality reports, dedupe, preview grids, ingest
  dataset_maker/
    cli.py                 — dataset-maker CLI subcommands
    model.py               — shared types (SpriteBundle, normalize_sprite_id)
    prefill.py             — VLM-backed metadata prefill (large, pending split)
    gui.py                 — Gradio dataset-maker GUI
    exporter.py            — export to training dataset
    qa.py                  — dataset QA gate
  harvest/
    cli/                   — harvest CLI package (set_defaults dispatch)
    assisted_golden*.py    — assisted golden-set labeling
    filename_rules*.py     — v1 + v2 filename→metadata rules
    label_v2_*.py          — v2 labeling pipeline (candidates, fusion, pipeline, eval, schema)
    source_profiles.py     — source/profile detection with capability fields
    sheet_specializations.py — per-sheet specialization rules (rpg_496)
  ml/                      — ML baselines, dataset, masking, metrics
  training/
    cli/                   — training CLI package (set_defaults dispatch)
    generator_challenger.py — rectified-flow UNet (primary model)
    device.py              — resolve_device, move_batch_to_device
    checkpoint_io.py       — load_checkpoint, tokenizer_from_checkpoint
    inspect_data.py        — inspect_training_data, describe_array
    prompt_records.py      — read_prompt_records
    report_utils.py        — jsonable, fmt_float, fmt_int
    data.py                — SpriteTrainingDataset + collate
    conditioning.py        — conditioning mode helpers
    palette_*.py           — palette projection/swap/report
    generator_audits.py    — challenger audit orchestration (large, pending split)
    v1_gallery*.py         — v1 demo gallery (challenger-based)
    v2_phase0_eval.py      — v2 Phase 0 evaluation harness
  utils/
    jsonl.py               — canonical read_jsonl / write_jsonl / iter_jsonl
```

## Conventions

- **Lazy CLI imports**: all heavy imports inside command handlers (torch-optional strategy)
- **CLI dispatch**: `set_defaults(func=handler)` pattern, never if/elif chains
- **Re-exports**: old modules re-import from new canonical homes
- **JSONL**: use `spritelab.utils.jsonl` for all read/write
- **Reporting**: `jsonable`, `fmt_float`, `fmt_int` from `training.report_utils`
- **Device ops**: `resolve_device`, `move_batch_to_device` from `training.device`
- **Checkpoint I/O**: `load_checkpoint`, `tokenizer_from_checkpoint` from `training.checkpoint_io`
- **Logging**: `logger = logging.getLogger(__name__)` per module; `--verbose` on CLIs
- **No large files**: pre-commit guards blobs > 2 MB
- **Tests**: CPU-only, `torch = pytest.importorskip("torch")`, use `_semantic_dataset` test helpers
- **Training changes**: A/B bit-identical pattern required (see `tests/test_training_speed_options.py`)

## Constraints

- Windows dev box (PowerShell); GPU tests skip on CPU; basetemp must be `.pytest_tmp_*`
- Frozen release branches (`main`, `v1`, `v1.1`) — never format or modify
- Active workbench: `generator_challenger.py`, `generator_audits.py`, `v2_phase0_eval.py`
- Challenger hot loop must stay sync-free (all `.cpu()` calls are cached or hoisted)
- `palette_scale_needs_normalize` is a module-level cache, reset per training run
