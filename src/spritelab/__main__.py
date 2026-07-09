"""Small package-level command dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def _usage() -> str:
    return "Usage: spritelab <curation|training|train|dataset-maker|harvest|ml> ..."


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_usage())
        raise SystemExit(2)
    if args[0] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)

    command = args[0]
    if command == "curation":
        _run_curation(args[1:])
        return
    if command == "training":
        _run_training(args[1:])
        return
    if command == "train":
        from spritelab.training.cli import main as train_main

        train_main(args[1:])
        return
    if command == "dataset-maker":
        _run_dataset_maker(args[1:])
        return
    if command == "dataset-maker-import-export":
        _run_dataset_maker_import_export(args[1:])
        return
    if command == "dataset-maker-prefill":
        _run_dataset_maker_prefill(args[1:])
        return
    if command == "palette-report":
        from spritelab.training.palette_report import main as palette_report_main

        palette_report_main(args[1:])
        return
    if command == "export-training":
        from spritelab.training.export import main as export_main

        export_main(args[1:])
        return
    if command == "harvest":
        from spritelab.harvest.cli import main as harvest_main

        harvest_main(args[1:])
        return
    if command == "ml":
        from spritelab.ml.cli import main as ml_main

        ml_main(args[1:])
        return
    if command == "readiness":
        from spritelab.training.readiness import main as readiness_main

        readiness_main(args[1:])
        return

    print(f"Unknown command: {command}")
    raise SystemExit(2)


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


def _run_training(args: list[str]) -> None:
    if not args:
        print("Usage: python -m spritelab training <palette-report|export|readiness> ...")
        raise SystemExit(2)

    subcommand = args[0]
    rest = args[1:]
    if subcommand == "palette-report":
        from spritelab.training.palette_report import main as palette_report_main

        palette_report_main(rest)
        return
    if subcommand == "export":
        from spritelab.training.export import main as export_main

        export_main(rest)
        return
    if subcommand == "readiness":
        from spritelab.training.readiness import main as readiness_main

        readiness_main(rest)
        return

    print(f"Unknown training command: {subcommand}")
    raise SystemExit(2)


def _run_dataset_maker(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the local Dataset Maker GUI.")
    parser.add_argument("--output-root", default="datasets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parsed = parser.parse_args(args)

    from spritelab.dataset_maker.gui import launch_dataset_maker_gui

    try:
        launch_dataset_maker_gui(output_root=parsed.output_root, host=parsed.host, port=parsed.port)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc


def _run_dataset_maker_import_export(args: list[str]) -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Import a PNG directory and export a Dataset Maker dataset.")
    parser.add_argument("--png-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-root", default="datasets", type=Path)
    parser.add_argument("--category", default="unknown")
    parser.add_argument("--tags", default="")
    parser.add_argument("--max-palette-slots", type=int, default=32)
    parser.add_argument("--quantize-overcolor", action="store_true")
    parser.add_argument("--infer-role-map", action="store_true")
    parser.add_argument("--no-canonicalize-palette", action="store_false", dest="canonicalize_palette")
    parser.add_argument("--allow-nearest-resize", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(args)

    from spritelab.dataset_maker.exporter import DatasetMakerExportConfig, export_dataset_from_imported_sprites
    from spritelab.dataset_maker.importer import ImportOptions, import_png_directory

    imported = import_png_directory(
        parsed.png_dir,
        options=ImportOptions(
            max_palette_slots=parsed.max_palette_slots,
            allow_quantize_overcolor=True,
            quantize_overcolor=parsed.quantize_overcolor,
            allow_nearest_resize=parsed.allow_nearest_resize,
            infer_role_map=parsed.infer_role_map,
            canonicalize_palette=parsed.canonicalize_palette,
            recursive=parsed.recursive,
        ),
        default_category=parsed.category,
        default_tags=tuple(tag.strip() for tag in parsed.tags.split(",") if tag.strip()),
    )
    result = export_dataset_from_imported_sprites(
        imported,
        DatasetMakerExportConfig(
            dataset_name=parsed.dataset_name,
            output_root=parsed.output_root,
            max_palette_slots=parsed.max_palette_slots,
            train_fraction=parsed.train_fraction,
            val_fraction=parsed.val_fraction,
            test_fraction=parsed.test_fraction,
            seed=parsed.seed,
            overwrite=parsed.overwrite,
        ),
    )
    print(f"Output: {result.output_dir}")
    print(f"Accepted: {result.accepted_count}")
    print(f"Train: {result.train_count}")
    print(f"Val: {result.val_count}")
    print(f"Test: {result.test_count}")
    print(f"Excluded: {result.excluded_count}")


def _run_dataset_maker_prefill(args: list[str]) -> None:
    import argparse
    import json
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser(description="Suggest Dataset Maker metadata for PNGs using an optional local VLM.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--png", type=Path)
    source.add_argument("--png-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--backend", default="openai_compatible", choices=["openai_compatible", "ollama", "rule_based", "none"]
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--runpod-token", default="")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of concurrent prefill requests for directory mode."
    )
    parsed = parser.parse_args(args)

    from spritelab.dataset_maker.model import normalize_sprite_id
    from spritelab.dataset_maker.prefill import (
        MetadataSuggestion,
        PrefillConfig,
        PrefillRequest,
        create_prefill_backend,
        suggestion_to_json_dict,
    )

    backend = create_prefill_backend(
        PrefillConfig(
            enabled=parsed.backend != "none",
            backend=parsed.backend,
            model=parsed.model,
            base_url=parsed.base_url,
            api_key=parsed.api_key,
            runpod_token=parsed.runpod_token,
            timeout_seconds=parsed.timeout_seconds,
            cache_dir=parsed.cache_dir,
        )
    )
    png_paths = [parsed.png] if parsed.png is not None else _prefill_png_paths(parsed.png_dir)
    worker_count = max(1, int(parsed.workers or 1))

    def prefill_line(path: Path) -> str:
        try:
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
            suggestion = backend.suggest(
                PrefillRequest(
                    sprite_id=normalize_sprite_id(path.stem),
                    image=rgba,
                    source_path=str(path),
                )
            )
        except Exception as exc:
            suggestion = MetadataSuggestion(warnings=(f"prefill failed: {exc}",))
        return json.dumps(
            {
                "source_path": str(path),
                "suggestion": suggestion_to_json_dict(suggestion, include_raw=False),
            },
            sort_keys=True,
        )

    if worker_count == 1 or len(png_paths) <= 1:
        lines = [prefill_line(path) for path in png_paths]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        line_by_index: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(prefill_line, path): item_index for item_index, path in enumerate(png_paths)
            }
            for future in as_completed(future_to_index):
                line_by_index[future_to_index[future]] = future.result()
        lines = [line_by_index[item_index] for item_index in range(len(png_paths))]

    output = "\n".join(lines) + ("\n" if lines else "")
    if parsed.out is None:
        print(output, end="")
    else:
        parsed.out.parent.mkdir(parents=True, exist_ok=True)
        parsed.out.write_text(output, encoding="utf-8")


def _prefill_png_paths(root: Path) -> list[Path]:
    if root is None:
        return []
    return sorted(
        (path for path in root.glob("*.png") if path.is_file() and not path.name.startswith(".")),
        key=lambda path: path.as_posix().lower(),
    )


if __name__ == "__main__":
    main()
