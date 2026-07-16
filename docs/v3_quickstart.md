# Sprite Lab v3 quick start

Start with these four commands:

```powershell
python -m spritelab v3 status
python -m spritelab v3 dataset build
python -m spritelab v3 train
python -m spritelab v3 eval
```

The first command is always safe and shows implementation readiness, independent-audit status, and production authorization as separate facts. The other commands stop at the first mandatory gate. Invoking a friendly command does not authorize Dataset-v5 production freezing, training, generation, or checkpoint promotion.

For a new project, create the single configuration file and inspect the environment:

```powershell
python -m spritelab v3 init
python -m spritelab v3 doctor
python -m spritelab v3 status
```

`init` creates `spritelab.yaml` with every production action disabled. It never overwrites an existing configuration. Use `--dry-run` to preview state-changing commands:

```powershell
python -m spritelab v3 dataset build --dry-run
python -m spritelab v3 train --dry-run
python -m spritelab v3 eval --dry-run
```

When work stops, Sprite Lab preserves a run under `runs/v3/<run-id>/`, explains the smallest next action, and writes an offline report. Common follow-ups are:

```powershell
python -m spritelab v3 review
python -m spritelab v3 resume
python -m spritelab v3 runs
python -m spritelab v3 logs
python -m spritelab v3 report --open
python -m spritelab v3 explain training-audit
```

Add `--json` for automation, `--no-color` for plain output, `--quiet` for minimal output, or `--debug` to show a traceback that is hidden by default. No command prints credentials or calls a provider during status discovery.

The older low-level commands remain available for backend specialists. They are documented separately in [v3_migration_from_low_level_commands.md](v3_migration_from_low_level_commands.md); ordinary operators should use `v3`.
