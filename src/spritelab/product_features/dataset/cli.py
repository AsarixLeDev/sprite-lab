"""Plugin-owned CLI registration for simple folder intake and review."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from spritelab.product_core import ProductResult, ProductStatus, ProjectContext, VisionProvider, WebServerSettings
from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_features.dataset.intake import (
    DatasetIntakeService,
    discover_source_packs,
    inspect_dataset_folder,
    preprocessing_prompt,
)
from spritelab.product_features.dataset.sidecar import (
    PackMetadataError,
    apply_metadata_file,
    ensure_dataset_writes_outside_input,
    metadata_file_template,
)
from spritelab.product_features.dataset.web import (
    MetadataWizardOutcome,
    MetadataWizardSession,
    discover_review_queues,
    find_dataset_output,
)
from spritelab.v3.run_state import RunState, atomic_write_json

ProviderFactory = Callable[[ProjectContext, Callable[[str], bool] | None], VisionProvider | None]


def register_cli(registry: ProductCliRegistry, *, provider_factory: ProviderFactory | None = None) -> None:
    """Replace foundation placeholders without editing central CLI registration."""

    registry.register(
        "dataset",
        lambda target: _install_dataset(target, provider_factory=provider_factory),
        owner="dataset.intake",
        replace=True,
    )
    registry.register("review", _install_review, owner="dataset.intake", replace=True)


def _install_dataset(target: argparse._SubParsersAction, *, provider_factory: ProviderFactory | None = None) -> None:
    parser = target.add_parser("dataset", help="Build an image dataset from a normal folder.")
    commands = parser.add_subparsers(dest="dataset_command", required=True)
    build = commands.add_parser("build", help="Import, check, and prepare a folder of PNG images.")
    build.add_argument("folder", help="Folder containing PNG images and simple source/license files.")
    build.add_argument("--output", type=Path, help="Optional output folder; defaults to the project's datasets folder.")
    build.add_argument(
        "--metadata-file",
        type=Path,
        help="Apply controlled pack declarations from spritelab.dataset.pack_metadata_batch.v2 JSON.",
    )
    build.add_argument("--no-review", action="store_true", help="Do not offer to open exception review.")
    build.add_argument(
        "--allow-hosted",
        action="store_true",
        help="Confirm image transfer when the configured privacy policy requires hosted confirmation.",
    )
    _add_output_flags(build)
    build.set_defaults(handler=_handle_build, provider_factory=provider_factory)


def _install_review(target: argparse._SubParsersAction) -> None:
    parser = target.add_parser("review", help="Review excluded dataset images and rescue false rejections.")
    parser.add_argument("--result", type=Path, help="Dataset result/output directory to review.")
    _add_output_flags(parser)
    parser.set_defaults(handler=_handle_review)


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)


def _handle_build(args: argparse.Namespace, _argv: list[str]) -> ProductResult:
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    context = ProjectContext(config.root, config.values, config.path, config.runs_dir)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        args.json = True
    try:
        ensure_dataset_writes_outside_input(
            context.project_root,
            Path(args.folder).expanduser().resolve(),
            output_root=args.output,
            runs_directory=context.runs_directory,
        )
    except PackMetadataError as exc:
        return ProductResult(
            ProductStatus.BLOCKED,
            f"Dataset source safety check blocked automatic writes: {exc}",
            feature="dataset",
            data={"build_started": False, "input_mutated": False, "browser_opened": False},
        )
    run = RunState.create(
        config,
        command="dataset build",
        argv=list(_argv),
        source_commit=None,
        dry_run=False,
    )
    run.update(stage="pack-discovery", resumable=True)
    run.append_event(
        command="dataset build",
        stage="pack-discovery",
        event_type="stage_started",
        status="RUNNING",
        message="Discovering PNG files and independently licensed pack boundaries.",
    )

    result: ProductResult | None = None
    try:
        root, _paths, packs = discover_source_packs(args.folder, context=context)
        if args.metadata_file:
            apply_metadata_file(context.project_root, root, packs, args.metadata_file.expanduser().resolve())
        inspection = inspect_dataset_folder(root, context=context)
        if interactive and inspection["wizard_required"] and not args.metadata_file:
            wizard_outcome = launch_metadata_wizard(root, args.output)
            if wizard_outcome is MetadataWizardOutcome.COMPLETE:
                inspection = inspect_dataset_folder(root, context=context)
                if inspection["wizard_required"]:
                    result = ProductResult(
                        ProductStatus.BLOCKED,
                        "Pack information changed after the wizard completed. Dataset build did not start.",
                        feature="dataset",
                        data={
                            "wizard_outcome": wizard_outcome.value,
                            "browser_opened": True,
                            "build_started": False,
                            "next_command": f'python -m spritelab v3 dataset build "{args.folder}"',
                        },
                    )
            else:
                message = (
                    "Dataset build cancelled in the metadata wizard. No preprocessing was started."
                    if wizard_outcome is MetadataWizardOutcome.CANCELLED
                    else "Metadata wizard was interrupted. Dataset build did not start."
                )
                result = ProductResult(
                    ProductStatus.BLOCKED,
                    message,
                    feature="dataset",
                    data={
                        "wizard_outcome": wizard_outcome.value,
                        "browser_opened": True,
                        "build_started": False,
                        "next_command": f'python -m spritelab v3 dataset build "{args.folder}"',
                    },
                )
        if result is None:
            run.update(stage="technical-preprocessing", resumable=True)
            run.append_event(
                command="dataset build",
                stage="technical-preprocessing",
                event_type="stage_started",
                status="RUNNING",
                message="Running deterministic Dataset-v5 preprocessing without modifying source files.",
                current_count=0,
                total_count=int(inspection["image_count"]),
            )

            def confirm_hosted(prompt: str) -> bool:
                if args.allow_hosted:
                    return True
                return interactive and input(f"{prompt} [y/N] ").strip().casefold() in {"y", "yes"}

            factory: ProviderFactory | None = args.provider_factory
            provider = factory(context, confirm_hosted) if factory else None
            result = DatasetIntakeService(provider).build(
                root,
                output_root=args.output,
                context=context,
            )
    except PackMetadataError as exc:
        result = ProductResult(
            ProductStatus.BLOCKED,
            f"Metadata file was not applied: {exc}",
            feature="dataset",
            data={"next_command": f'python -m spritelab v3 dataset build "{args.folder}"'},
        )
    except Exception as exc:
        run.finish(
            command="dataset build",
            status="FAILED",
            exit_code=5,
            message=f"Dataset intake failed safely: {exc}",
            stage="technical-preprocessing",
            resumable=True,
        )
        raise

    assert result is not None
    counts = result.data.get("counts", {})
    output_root = (
        Path(str(result.data.get("output_root")))
        if result.data.get("output_root")
        else None
        if result.data.get("build_started") is False
        else args.output
    )
    inspection = inspect_dataset_folder(args.folder, context=context) if result.data.get("output_root") else {}
    if not interactive and inspection.get("wizard_required") and output_root is not None:
        template_path = Path(output_root) / "metadata-required.json"
        _root, _paths, packs = discover_source_packs(args.folder, context=context)
        atomic_write_json(
            template_path,
            metadata_file_template(_root, packs),
        )
        data = dict(result.data)
        data.update(
            {
                "metadata_template": str(template_path),
                "next_command": (
                    f'python -m spritelab v3 dataset build "{Path(args.folder).expanduser().resolve()}" '
                    f'--metadata-file "{template_path}"'
                ),
                "noninteractive": True,
                "browser_opened": False,
            }
        )
        result = replace(result, data=data)
    excluded = int(counts.get("excluded", 0))
    if excluded and interactive and not args.no_review and result.data.get("output_root"):
        print(preprocessing_prompt({"counts": counts}))
        answer = input("\nReview excluded images now? [Y/n] ").strip().casefold()
        if answer in {"", "y", "yes"}:
            launch_review_interface(Path(str(result.data["output_root"])))
    result_data = dict(result.data)
    result_data["durable_run"] = {
        "run_id": run.run_id,
        "state": str(run.state_path),
        "events": str(run.events_path),
        "command": str(run.command_path),
    }
    result = replace(result, data=result_data)
    run.finish(
        command="dataset build",
        status=result.status.value,
        exit_code=0 if result.status in {ProductStatus.COMPLETE, ProductStatus.READY} else 4,
        message=result.message,
        stage="metadata-wizard" if result.data.get("build_started") is False else "review" if excluded else "complete",
        resumable=False,
        backend_identity={"output_root": str(output_root) if output_root else None},
    )
    return result


def _handle_review(args: argparse.Namespace, _argv: list[str]) -> ProductResult:
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    values = copy.deepcopy(config.values)
    if args.result:
        values.setdefault("dataset", {})["output_root"] = str(args.result)
    context = ProjectContext(config.root, values, config.path, config.runs_dir)
    queues = discover_review_queues(context)
    if not queues["available"]:
        return ProductResult(
            ProductStatus.BLOCKED,
            "No product review queue is available.",
            feature="review",
            data={"browser_opened": False, "queues": queues},
        )
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return ProductResult(
            ProductStatus.NEEDS_REVIEW,
            "Product review is ready; noninteractive mode did not open a browser.",
            feature="review",
            data={"browser_opened": False, "queues": queues},
        )
    output = find_dataset_output(context)
    launch_review_interface(output)
    return ProductResult(
        ProductStatus.COMPLETE,
        "Product review interface closed. Authoritative decisions were saved by their owning review backends.",
        feature="review",
        data={"browser_opened": True, "queues": queues},
    )


def launch_review_interface(output_root: Path | None) -> None:
    """Open the common review route in the same unified product application."""

    from spritelab.product_runtime import build_product_runtime
    from spritelab.product_web.app import create_app
    from spritelab.product_web.cli import run_server
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    values = copy.deepcopy(config.values)
    if output_root is not None:
        values.setdefault("dataset", {})["output_root"] = str(output_root)
    context = ProjectContext(config.root, values, config.path, config.runs_dir)
    settings = WebServerSettings(host="127.0.0.1", port="auto", open_browser=True)
    app = create_app(context, plugins=build_product_runtime().plugins, settings=settings)
    run_server(app, settings, open_browser=True, open_path="/review")


def launch_metadata_wizard(input_root: Path, output_root: Path | None) -> MetadataWizardOutcome:
    """Open the local prefilled pack wizard; source paths stay server-side."""

    from spritelab.product_runtime import build_product_runtime
    from spritelab.product_web.app import create_app
    from spritelab.product_web.cli import run_server
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    values = copy.deepcopy(config.values)
    dataset = values.setdefault("dataset", {})
    dataset["pending_input_root"] = str(input_root.resolve())
    session = MetadataWizardSession()
    if output_root is not None:
        dataset["pending_output_root"] = str(output_root.expanduser().resolve())
    context = ProjectContext(config.root, values, config.path, config.runs_dir)
    settings = WebServerSettings(host="127.0.0.1", port="auto", open_browser=True)
    app = create_app(context, plugins=build_product_runtime().plugins, settings=settings)
    app.state.spritelab_metadata_wizard_session = session
    try:
        run_server(app, settings, open_browser=True, open_path="/dataset/metadata")
    except KeyboardInterrupt:
        session.finish(MetadataWizardOutcome.INTERRUPTED)
    finally:
        if session.outcome is MetadataWizardOutcome.PENDING:
            session.finish(MetadataWizardOutcome.INTERRUPTED)
    return session.outcome
