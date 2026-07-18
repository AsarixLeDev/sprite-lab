"""Package-level command dispatcher with a dict registry (Phase 6c)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[list[str]], None]

_DEFAULT_V3_APP_ARGS = ("--host", "127.0.0.1", "--port", "8765")


def _usage() -> str:
    return "Usage: spritelab <v3|dev|curation|training|train|dataset-maker|harvest|ml|eval> ..."


# ── Handler implementations ───────────────────────────────────────────────────


def _run_train(args: list[str]) -> None:
    from spritelab.training.cli import main as train_main

    train_main(args)


def _run_harvest(args: list[str]) -> None:
    from spritelab.harvest.cli import main as harvest_main

    harvest_main(args)


def _run_ml(args: list[str]) -> None:
    from spritelab.ml.cli import main as ml_main

    ml_main(args)


def _run_eval(args: list[str]) -> None:
    from spritelab.evaluation.cli import main as eval_main

    eval_main(args)


def _run_v3(args: list[str]) -> None:
    if not args:
        args = ["app", *_DEFAULT_V3_APP_ARGS]
    if args[0] == "app":
        from spritelab.product_web.cli import main as product_web_main

        product_web_main(args[1:])
        return
    from spritelab.product_runtime import build_product_runtime
    from spritelab.v3.cli import main as v3_main

    runtime = build_product_runtime()
    v3_main(args, plugins=runtime.plugins)


def _run_dev(args: list[str]) -> None:
    from spritelab.dev.cli import main as dev_main

    dev_main(args)


def _run_curation(args: list[str]) -> None:
    if not args:
        print("Usage: python -m spritelab curation <summary|validate|decide|browser> ...")
        raise SystemExit(2)

    subcommand = args[0]
    rest = args[1:]
    if subcommand == "browser":
        from spritelab.curation.browser import main as browser_main

        browser_main(rest)
        return

    from spritelab.curation.manifest import main as manifest_main

    manifest_main([subcommand, *rest])


def _run_palette_report(args: list[str]) -> None:
    from spritelab.training.palette_report import main as palette_report_main

    palette_report_main(args)


def _run_export_training(args: list[str]) -> None:
    from spritelab.training.export import main as export_main

    export_main(args)


def _run_readiness(args: list[str]) -> None:
    from spritelab.training.readiness import main as readiness_main

    readiness_main(args)


def _run_dataset_maker(args: list[str]) -> None:
    from spritelab.dataset_maker.cli import run_dataset_maker_gui

    run_dataset_maker_gui(args)


def _run_dataset_maker_import_export(args: list[str]) -> None:
    from spritelab.dataset_maker.cli import run_dataset_maker_import_export

    run_dataset_maker_import_export(args)


def _run_dataset_maker_prefill(args: list[str]) -> None:
    from spritelab.dataset_maker.cli import run_dataset_maker_prefill

    run_dataset_maker_prefill(args)


# ── Command registry ─────────────────────────────────────────────────────────
# Keys are the CLI command names (first argument after `spritelab`).
# All handlers are lazy-importing (no heavy imports at module load time).

_COMMANDS: dict[str, Handler] = {
    "v3": _run_v3,
    "dev": _run_dev,
    "curation": _run_curation,
    "train": _run_train,
    "training": _run_train,  # alias
    "harvest": _run_harvest,
    "ml": _run_ml,
    "eval": _run_eval,
    "dataset-maker": _run_dataset_maker,
    "dataset-maker-import-export": _run_dataset_maker_import_export,
    "dataset-maker-prefill": _run_dataset_maker_prefill,
    "palette-report": _run_palette_report,
    "export-training": _run_export_training,
    "readiness": _run_readiness,
}


# ── Public entry point ───────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("-v", "--verbose"):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s")
        args = args[1:]
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

    if not args:
        print(_usage())
        raise SystemExit(2)
    if args[0] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)

    command = args[0]
    handler = _COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command}")
        raise SystemExit(2)

    handler(args[1:])


if __name__ == "__main__":
    main()
