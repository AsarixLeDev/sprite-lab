"""Build a trainer-ready, identity-bound dataset from the active product dataset."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from spritelab.dataset_maker.exporter import DatasetMakerExportConfig, export_dataset_from_imported_sprites
from spritelab.dataset_maker.importer import ImportOptions, import_png_as_dataset_item
from spritelab.dataset_maker.qa import qa_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.dataset_maker.training_manifest_qa import qa_training_manifest
from spritelab.product_core import ProjectContext
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, plan_campaign, stable_hash, validate_campaign
from spritelab.training.tokenization import SpriteTextTokenizer
from spritelab.utils.safe_fs import atomic_write_text, remove_confined_tree, require_confined_path
from spritelab.v3.config import ProjectConfig

Progress = Callable[[int, int, str], None]
PUBLICATION_MANIFEST_NAME = "publication_manifest.json"
PUBLICATION_MANIFEST_SCHEMA = "spritelab.training_preparation.publication.v1"
PREPARATION_RECIPE_SCHEMA = "spritelab.training_preparation.recipe.v1"
PREPARATION_RECIPE_SOURCES = (
    "src/spritelab/product_features/training/preparation.py",
    "src/spritelab/dataset_maker/exporter.py",
    "src/spritelab/dataset_maker/importer.py",
    "src/spritelab/dataset_maker/qa.py",
    "src/spritelab/dataset_maker/training_manifest.py",
    "src/spritelab/dataset_maker/training_manifest_qa.py",
    "src/spritelab/training/data.py",
    "src/spritelab/training/tokenization.py",
    "src/spritelab/utils/safe_fs.py",
)


class TrainingPreparationError(RuntimeError):
    """A controlled preparation failure safe to return through the web API."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def prepare_active_dataset(
    context: ProjectContext,
    *,
    authorize_freeze: bool,
    authorize_training: bool,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Publish and configure an explicitly authorized image-only training freeze."""

    if not authorize_freeze:
        raise TrainingPreparationError(
            "freeze_authorization_required",
            "Explicit image-only production-freeze authorization is required.",
        )
    dataset = _find_dataset_output(context)
    if dataset is None:
        raise TrainingPreparationError("active_dataset_missing", "No active accepted dataset is selected.")
    rows = _read_jsonl(dataset / "items.jsonl")
    accepted = [row for row in rows if row.get("current_disposition") == "accepted"]
    if not accepted:
        raise TrainingPreparationError("accepted_images_missing", "The active dataset has no accepted images.")
    accepted.sort(key=_accepted_item_id)

    total = len(accepted) * 2 + 8
    _notify(progress, 0, total, f"Found {len(accepted)} accepted images; verifying immutable source identities.")
    verified: list[tuple[dict[str, Any], Path, str]] = []
    for index, row in enumerate(accepted, start=1):
        source = Path(str(row.get("source_path") or "")).expanduser()
        digest = _verify_source(source, str(row.get("byte_sha256") or ""))
        verified.append((row, source, digest))
        _notify(progress, index, total, f"Verified source identity {index}/{len(accepted)}.")

    identity = stable_hash(
        {
            "preparation_recipe_sha256": _preparation_recipe_identity(),
            "items": [
                {
                    "item_id": _accepted_item_id(row),
                    "byte_sha256": digest,
                }
                for row, _source, digest in verified
            ],
        }
    )
    root = require_confined_path(
        context.project_root / ".spritelab" / "training-preparation",
        context.project_root,
    )
    root.mkdir(parents=True, exist_ok=True)
    name = f"dataset-{identity[:16]}"
    output = require_confined_path(root / name, root)
    if output.exists():
        _validate_publication(output, identity=identity, image_count=len(accepted))
        _notify(progress, total - 1, total, "Reused the verified content-addressed training publication.")
    else:
        _build_publication(
            context,
            root=root,
            output=output,
            name=name,
            identity=identity,
            verified=verified,
            total=total,
            progress=progress,
        )

    view = output / "view_manifest.json"
    freeze = output / "freeze_manifest.json"
    campaign = output / "campaign.json"
    training_authorized = _update_project_config(
        context,
        view=view,
        freeze=freeze,
        campaign=campaign,
        authorize_training=authorize_training,
    )
    _notify(
        progress, total, total, "Preparation complete. Training readiness was refreshed without bypassing audit gates."
    )
    return {
        "dataset_identity": identity,
        "image_count": len(accepted),
        "publication_id": name,
        "freeze_kind": "image_only",
        "campaign_profile": "recommended",
        "training_authorized": training_authorized,
        "remaining_gate": "independent_training_infrastructure_audit",
        "paths_exposed": False,
    }


def _build_publication(
    context: ProjectContext,
    *,
    root: Path,
    output: Path,
    name: str,
    identity: str,
    verified: list[tuple[dict[str, Any], Path, str]],
    total: int,
    progress: Progress | None,
) -> None:
    staging_root = require_confined_path(root / f".staging-{uuid.uuid4().hex}", root)
    staging_root.mkdir()
    staged_output = staging_root / name
    try:
        imported = []
        options = ImportOptions(
            max_palette_slots=32,
            allow_quantize_overcolor=True,
            quantize_overcolor=True,
            allow_nearest_resize=True,
        )
        for index, (row, source, _digest) in enumerate(verified, start=1):
            category = "sprite"
            sprite = import_png_as_dataset_item(
                source,
                options=options,
                default_category=category,
                default_tags=(category,),
            )
            if sprite.errors or sprite.bundle is None:
                raise TrainingPreparationError(
                    "canonical_encoding_failed",
                    "An accepted image could not be encoded into the canonical 32 by 32 training format.",
                )
            item_id = _accepted_item_id(row)
            public_source_name = f"{_digest}.png"
            sprite = replace(
                sprite,
                item=replace(
                    sprite.item,
                    sprite_id=item_id,
                    source_path=Path("accepted-sources") / public_source_name,
                    source_name=public_source_name,
                    notes="pixel art sprite",
                ),
                auto_metadata={
                    "label_v2_safe_prefill": {
                        "object_name": "sprite",
                        "short_description": "pixel art sprite",
                    }
                },
            )
            imported.append(sprite)
            _notify(
                progress,
                len(verified) + index,
                total,
                f"Encoded canonical training array {index}/{len(verified)}.",
            )
        export_dataset_from_imported_sprites(
            imported,
            DatasetMakerExportConfig(dataset_name=name, output_root=staging_root, overwrite=False),
        )
        _notify(progress, len(verified) * 2 + 1, total, "Built deterministic split arrays and base manifests.")

        manifest_result = build_training_manifest(
            staged_output,
            variants_per_sprite=1,
            caption_policy="mixed",
            seed=1337,
        )
        manifest = staged_output / "training_manifest.jsonl"
        training_rows = _portable_training_rows(manifest_result.rows)
        write_training_manifest(manifest, training_rows)
        _notify(progress, len(verified) * 2 + 2, total, "Built the canonical combined conditioning manifest.")

        dataset_qa = qa_dataset(staged_output)
        training_qa = qa_training_manifest(staged_output, manifest)
        if dataset_qa.errors or training_qa.errors:
            raise TrainingPreparationError(
                "training_dataset_qa_failed",
                "The prepared dataset did not pass its deterministic dataset and training-manifest checks.",
            )
        _write_qa_reports(staged_output, dataset_qa.to_json_dict(), training_qa.to_json_dict())
        loader_validated = _validate_training_loader(staged_output, manifest)
        loader_message = (
            "Validated dataset QA, manifest QA, and production trainer data loading."
            if loader_validated
            else "Validated dataset QA and manifest QA; trainer loading awaits the audited PyTorch environment."
        )
        _notify(progress, len(verified) * 2 + 3, total, loader_message)

        vocabulary = staged_output / "conditioning_vocabulary.json"
        training_rows = _read_jsonl(manifest)
        tokenizer = SpriteTextTokenizer.build_from_records(
            (row for row in training_rows if row.get("split") == "train"),
            max_length=32,
        )
        _write_json_once(vocabulary, tokenizer.to_json_dict())
        _notify(progress, len(verified) * 2 + 4, total, "Frozen the deterministic conditioning vocabulary.")

        benchmark = staged_output / "benchmark_manifest.json"
        _write_json_once(
            benchmark,
            {
                "schema_version": "spritelab.training_benchmark.v1",
                "dataset_identity": identity,
                "prompts": [str(row.get("caption") or "sprite") for row in training_rows[:16]],
            },
        )
        view = staged_output / "view_manifest.json"
        _write_json_once(
            view,
            {
                "schema_version": "spritelab.training_view.v1",
                "status": "complete",
                "view": "image_only",
                "dataset_identity": identity,
                "image_count": len(verified),
                "training_manifest": manifest.name,
                "training_manifest_sha256": file_sha256(manifest),
            },
        )
        freeze = staged_output / "freeze_manifest.json"
        _write_json_once(
            freeze,
            {
                "schema_version": "spritelab.dataset.freeze.image_only.v1",
                "status": "complete",
                "production_authorized": True,
                "freeze_kind": "image_only",
                "dataset_kind": "image_only",
                "requires_semantic_labels": False,
                "dataset_identity": identity,
                "image_count": len(verified),
                "view_manifest": view.name,
                "view_manifest_sha256": file_sha256(view),
                "training_manifest": manifest.name,
                "training_manifest_sha256": file_sha256(manifest),
            },
        )
        _notify(progress, len(verified) * 2 + 5, total, "Built the explicitly authorized image-only freeze.")

        validation_spec = _campaign_spec(
            freeze=freeze,
            view=view,
            manifest=manifest,
            vocabulary=vocabulary,
            benchmark=benchmark,
            identity=identity,
            training_record_count=sum(row.get("split") == "train" for row in training_rows),
            output_root=context.project_root / "training-runs",
        )
        validation = validate_campaign(plan_campaign(validation_spec, execution_root=context.project_root))
        if validation["errors"] or validation["blockers"] or not validation["launch_ready"]:
            raise TrainingPreparationError(
                "campaign_validation_failed",
                "The recommended campaign could not be bound to the prepared training artifacts.",
            )
        campaign = staged_output / "campaign.json"
        _write_json_once(
            campaign,
            {
                "product_profiles": {
                    "recommended": {
                        "display": {"display_name": "Recommended baseline"},
                        "campaign": _portable_campaign_spec(
                            validation_spec,
                            publication_root=output,
                            output_root=context.project_root / "training-runs",
                        ),
                    }
                }
            },
        )
        _notify(progress, len(verified) * 2 + 6, total, "Validated the recommended three-seed campaign schema.")

        _write_publication_manifest(staged_output, identity=identity, image_count=len(verified))
        _assert_private_paths_absent(
            staged_output,
            private_paths=(context.project_root, *(source for _row, source, _digest in verified)),
        )

        try:
            staged_output.replace(output)
        except OSError:
            if not output.exists():
                raise
            _validate_publication(output, identity=identity, image_count=len(verified))
        _notify(progress, total - 1, total, "Published the immutable content-addressed training artifacts.")
    except TrainingPreparationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TrainingPreparationError(
            "training_preparation_failed",
            "Training preparation failed before project configuration was changed.",
        ) from exc
    finally:
        remove_confined_tree(staging_root, root, missing_ok=True)


def _campaign_spec(
    *,
    freeze: Path,
    view: Path,
    manifest: Path,
    vocabulary: Path,
    benchmark: Path,
    identity: str,
    training_record_count: int,
    output_root: Path,
) -> dict[str, Any]:
    def bound(path: Path) -> str:
        return str(path)

    model = {
        "architecture": "rectified_flow",
        "sprite_size": 32,
        "base_channels": 32,
        "channel_mults": [1, 2],
        "res_blocks_per_level": 1,
        "embed_dim": 32,
        "film_conditioning": False,
        "bottleneck_attention": False,
        "auxiliary_heads_mode": "absent",
    }
    optimizer = {
        "name": "adamw",
        "learning_rate": 0.0002,
        "schedule": "none",
        "warmup_steps": 0,
        "gradient_clip": 0.0,
    }
    schedule = {"name": "none", "warmup_steps": 0}
    loss = {
        "name": "uniform_velocity",
        "strategy": "uniform_velocity",
        "foreground_rgb_weight": 1.0,
        "background_rgb_weight": 1.0,
        "palette_aux_weight": 0.0,
        "auxiliary_heads": False,
        "index_head_weight": 0.0,
        "palette_head_weight": 0.0,
        "palette_presence_weight": 0.0,
    }
    determinism = {"mode": "strict"}
    evaluation = {
        "cadence": 250,
        "include_step_zero": False,
        "benchmark_manifest_hash": file_sha256(benchmark),
        "benchmark_manifest_path": bound(benchmark),
        "cfg_value": 3.0,
        "sampling_steps": 30,
        "ema_policy": "both",
        "live_weight_evaluation_policy": "required",
    }
    evaluation["evaluation_config_hash"] = stable_hash(
        {key: value for key, value in evaluation.items() if not key.startswith("benchmark_manifest_")}
    )
    return {
        "campaign_id": f"recommended_{identity[:12]}",
        "purpose": "Recommended image-only baseline prepared by the Sprite Lab web workflow.",
        "architecture_cells": [{"cell_id": "baseline", "comparison_values": {}}],
        "identities": {
            "dataset_view_manifest_hash": file_sha256(view),
            "dataset_view_manifest_path": bound(view),
            "split_manifest_hash": file_sha256(manifest),
            "split_manifest_path": bound(manifest),
            "conditioning_vocabulary_hash": file_sha256(vocabulary),
            "conditioning_vocabulary_path": bound(vocabulary),
            "model_config_hash": stable_hash(model),
            "optimizer_config_hash": stable_hash(optimizer),
            "schedule_config_hash": stable_hash(schedule),
            "loss_config_hash": stable_hash(loss),
            "determinism_config_hash": stable_hash(determinism),
            "dataset_freeze_hash": file_sha256(freeze),
        },
        "seeds": list(DEFAULT_SEEDS),
        "model": model,
        "training": {
            "device": "auto",
            "max_optimizer_steps": 5000,
            "micro_batch_size": 4,
            "gradient_accumulation": 1,
            "effective_batch_size": 4,
            "precision": "fp32",
            "sampler_policy": "weighted_replacement_v1",
            "positive_sampling_mass_records": float(max(1, training_record_count)),
        },
        "optimizer": optimizer,
        "schedule": schedule,
        "loss": loss,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": {"cadence": 1000},
        "output_root": bound(output_root),
        "executable": True,
        "launch_authorized": True,
    }


def _portable_campaign_spec(
    spec: Mapping[str, Any],
    *,
    publication_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    result = deepcopy(dict(spec))
    identities = result["identities"]
    identities["dataset_view_manifest_path"] = "view_manifest.json"
    identities["split_manifest_path"] = "training_manifest.jsonl"
    identities["conditioning_vocabulary_path"] = "conditioning_vocabulary.json"
    result["evaluation"]["benchmark_manifest_path"] = "benchmark_manifest.json"
    result["output_root"] = os.path.relpath(output_root, start=publication_root).replace("\\", "/")
    return result


def _validate_publication(output: Path, *, identity: str, image_count: int) -> None:
    required = (
        "train.npz",
        "training_manifest.jsonl",
        "conditioning_vocabulary.json",
        "dataset_qa_report.json",
        "training_manifest_qa_report.json",
        "view_manifest.json",
        "benchmark_manifest.json",
        "freeze_manifest.json",
        "campaign.json",
        PUBLICATION_MANIFEST_NAME,
    )
    if not output.is_dir() or any(not (output / name).is_file() for name in required):
        raise TrainingPreparationError(
            "training_publication_incomplete",
            "An existing identity-bound training publication is incomplete and was not replaced.",
        )
    try:
        view = json.loads((output / "view_manifest.json").read_text(encoding="utf-8"))
        freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
        publication = json.loads((output / PUBLICATION_MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingPreparationError(
            "training_publication_invalid",
            "An existing identity-bound training publication is unreadable and was not replaced.",
        ) from exc
    if not all(isinstance(value, Mapping) for value in (view, freeze, publication)):
        raise TrainingPreparationError(
            "training_publication_invalid",
            "An existing identity-bound training publication is malformed and was not replaced.",
        )
    try:
        expected_publication = _publication_manifest(output, identity=identity, image_count=image_count)
    except (OSError, ValueError) as exc:
        raise TrainingPreparationError(
            "training_publication_invalid",
            "An existing identity-bound training publication could not be verified safely.",
        ) from exc
    manifest = output / "training_manifest.jsonl"
    if (
        publication != expected_publication
        or view.get("dataset_identity") != identity
        or freeze.get("dataset_identity") != identity
        or view.get("image_count") != image_count
        or freeze.get("image_count") != image_count
        or view.get("training_manifest_sha256") != file_sha256(manifest)
        or freeze.get("training_manifest_sha256") != file_sha256(manifest)
    ):
        raise TrainingPreparationError(
            "training_publication_identity_mismatch",
            "An existing identity-bound training publication failed immutable identity verification.",
        )
    dataset_qa = qa_dataset(output)
    training_qa = qa_training_manifest(output, manifest)
    if dataset_qa.errors or training_qa.errors:
        raise TrainingPreparationError(
            "training_publication_qa_failed",
            "An existing identity-bound training publication no longer passes dataset validation.",
        )
    _validate_training_loader(output, manifest)


def _write_publication_manifest(output: Path, *, identity: str, image_count: int) -> None:
    _write_json_once(
        output / PUBLICATION_MANIFEST_NAME,
        _publication_manifest(output, identity=identity, image_count=image_count),
    )


def _publication_manifest(output: Path, *, identity: str, image_count: int) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for directory, directory_names, file_names in os.walk(output, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = require_confined_path(directory_path / name, output)
            if not child.is_dir():
                raise TrainingPreparationError(
                    "training_publication_invalid",
                    "An identity-bound training publication contains an unsafe filesystem entry.",
                )
        for name in sorted(file_names):
            child = require_confined_path(directory_path / name, output)
            relative = child.relative_to(output).as_posix()
            if relative == PUBLICATION_MANIFEST_NAME:
                continue
            if not child.is_file():
                raise TrainingPreparationError(
                    "training_publication_invalid",
                    "An identity-bound training publication contains an unsafe filesystem entry.",
                )
            artifacts[relative] = {
                "byte_size": child.stat().st_size,
                "sha256": file_sha256(child),
            }
    return {
        "schema_version": PUBLICATION_MANIFEST_SCHEMA,
        "dataset_identity": identity,
        "image_count": image_count,
        "artifacts": dict(sorted(artifacts.items())),
    }


def _assert_private_paths_absent(output: Path, *, private_paths: tuple[Path, ...]) -> None:
    tokens = {
        spelling
        for path in private_paths
        for spelling in (str(path.resolve()), str(path.resolve()).replace("\\", "/"))
        if spelling
    }
    try:
        for path in sorted(output.iterdir()):
            if path.suffix not in {".json", ".jsonl", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in tokens):
                raise TrainingPreparationError(
                    "training_artifact_privacy_failed",
                    "A prepared text artifact contained a private local path and was not published.",
                )
    except (OSError, UnicodeDecodeError) as exc:
        raise TrainingPreparationError(
            "training_artifact_privacy_failed",
            "Prepared text artifacts could not be checked for private local paths.",
        ) from exc


def _validate_training_loader(output: Path, manifest: Path) -> bool:
    try:
        from spritelab.training import data as training_data

        if training_data.torch is None:
            return False
        dataset = training_data.SpriteTrainingDataset(output, manifest, split="train", max_records=1)
        if len(dataset) < 1:
            raise ValueError("empty train split")
        dataset[0]
    except (OSError, ValueError, IndexError, KeyError, RuntimeError) as exc:
        raise TrainingPreparationError(
            "training_loader_validation_failed",
            "The prepared arrays could not be loaded through the production training dataset.",
        ) from exc
    return True


def _write_qa_reports(output: Path, dataset_report: dict[str, Any], training_report: dict[str, Any]) -> None:
    dataset_report = deepcopy(dataset_report)
    training_report = deepcopy(training_report)
    dataset_report["dataset_dir"] = "."
    training_report["dataset_dir"] = "."
    training_report["manifest_path"] = "training_manifest.jsonl"
    _write_json_once(output / "dataset_qa_report.json", dataset_report)
    _write_json_once(output / "training_manifest_qa_report.json", training_report)


def _portable_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    portable = deepcopy(rows)
    for row in portable:
        source = row.get("source")
        if isinstance(source, dict):
            source["dataset_dir"] = "."
            source["inference_path"] = ""
    return portable


def _update_project_config(
    context: ProjectContext,
    *,
    view: Path,
    freeze: Path,
    campaign: Path,
    authorize_training: bool,
) -> bool:
    config = ProjectConfig.load(context.project_root)
    if config.path is None or not config.path.is_file():
        raise TrainingPreparationError(
            "project_configuration_missing",
            "The project configuration could not be updated because it is missing.",
        )
    target = config.path
    before = target.read_bytes()
    values = deepcopy(config.values)
    values["dataset"]["view_manifest"] = _project_relative(context.project_root, view)
    values["dataset"]["freeze_manifest"] = _project_relative(context.project_root, freeze)
    values["training"]["dataset_freeze"] = _project_relative(context.project_root, freeze)
    values["training"]["campaign_config"] = _project_relative(context.project_root, campaign)
    values["execution"]["allow_dataset_production_freeze"] = True
    if authorize_training:
        values["execution"]["allow_training"] = True
    payload = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    if target.read_bytes() != before:
        raise TrainingPreparationError(
            "project_configuration_changed",
            "Project configuration changed concurrently; the prepared artifacts were not activated.",
        )
    atomic_write_text(target, payload)
    return values["execution"]["allow_training"] is True


def _project_relative(project_root: Path, path: Path) -> str:
    confined = require_confined_path(path, project_root)
    return confined.relative_to(project_root.resolve()).as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingPreparationError("active_dataset_invalid", "The active dataset metadata is incomplete.")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("row is not an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise TrainingPreparationError("active_dataset_invalid", "The active dataset metadata is unreadable.") from exc
    return rows


def _find_dataset_output(context: ProjectContext) -> Path | None:
    """Resolve only the committed project-config/datasets contract used by training."""

    dataset_config = context.config.get("dataset", {}) if isinstance(context.config, Mapping) else {}
    if isinstance(dataset_config, Mapping):
        configured = dataset_config.get("output_root") or dataset_config.get("result_path")
        if configured:
            path = Path(str(configured)).expanduser()
            if path.name == "result.json":
                path = path.parent
            if path.is_dir() and (path / "review_queue.json").is_file():
                return path.resolve()
    datasets = context.project_root / "datasets"
    if not datasets.is_dir():
        return None
    candidates = [
        path.parent for path in datasets.glob("*/result.json") if (path.parent / "review_queue.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "result.json").stat().st_mtime_ns).resolve()


def _verify_source(path: Path, expected: str) -> str:
    if not expected or not path.is_file():
        raise TrainingPreparationError(
            "accepted_source_missing",
            "An accepted source image is missing or has no immutable source identity.",
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingPreparationError(
            "accepted_source_unreadable",
            "An accepted source image could not be read for identity verification.",
        ) from exc
    actual = digest.hexdigest()
    if actual != expected:
        raise TrainingPreparationError(
            "accepted_source_changed",
            "An accepted source image changed after review; preparation was refused.",
        )
    return actual


def _accepted_item_id(row: Mapping[str, Any]) -> str:
    value = row.get("item_id")
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TrainingPreparationError(
            "accepted_item_identity_missing",
            "An accepted image is missing its stable item identity.",
        )
    return value


def _preparation_recipe_identity() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    records: list[dict[str, str]] = []
    for relative in PREPARATION_RECIPE_SOURCES:
        path = repo_root / relative
        if not path.is_file():
            raise TrainingPreparationError(
                "preparation_recipe_incomplete",
                "The training preparation recipe is incomplete and cannot be identity-bound.",
            )
        records.append({"path": relative, "sha256": file_sha256(path)})
    return stable_hash({"schema_version": PREPARATION_RECIPE_SCHEMA, "sources": records})


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise TrainingPreparationError(
                "identity_artifact_conflict",
                "An identity-bound training artifact already exists with different content.",
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _notify(progress: Progress | None, current: int, total: int, message: str) -> None:
    if progress is not None:
        progress(current, total, message)


__all__ = [
    "TrainingPreparationError",
    "prepare_active_dataset",
]
