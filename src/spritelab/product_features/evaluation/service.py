"""Evaluation planning and durable execution over the existing benchmark backend."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from spritelab.evaluation.candidate_bundle import load_candidate_bundle
from spritelab.evaluation.memorization import resolve_training_context_identities
from spritelab.product_core import (
    ProductBlocker,
    ProductEvent,
    ProductResult,
    ProductRun,
    ProductStatus,
    StrictJSONError,
    strict_json_dumps,
    strict_json_loads,
)
from spritelab.product_features.evaluation.checkpoints import (
    discover_checkpoint_candidates,
    expected_dataset_identity,
    expected_training_view_identity,
)
from spritelab.product_features.evaluation.dashboard import build_dashboard, public_evaluation_projection
from spritelab.product_features.evaluation.memorization_display import (
    INCOMPLETE_EVIDENCE_MESSAGE,
    memorization_display,
    promotion_integrity_display,
)
from spritelab.product_features.evaluation.models import (
    CheckpointCandidate,
    CheckpointCatalog,
    EvaluationStage,
    new_evaluation_stages,
)
from spritelab.product_features.evaluation.playground import GenerationSafetyError
from spritelab.product_web.events import EventRepository

EVALUATION_STATE_SCHEMA = "spritelab.product.evaluation-state.v2"
_UNSET = object()


class EvaluationGenerator(Protocol):
    remote: bool
    billable: bool

    def generate_benchmark(
        self,
        *,
        checkpoint: Path,
        benchmark: Path,
        output_directory: Path,
        weights: str,
        emit: Callable[[str, int, int | None, str], None],
    ) -> Path: ...


Evaluator = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class EvaluationRequest:
    checkpoint_id: str | None = None
    benchmark: Path | None = None
    weights: str = "ema"
    dry_run: bool = False
    explicit_action: bool = False
    confirm_billable: bool = False
    allow_source_results: bool = False

    def __post_init__(self) -> None:
        for name in ("dry_run", "explicit_action", "confirm_billable", "allow_source_results"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"EvaluationRequest.{name} must be an exact boolean.")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    for line in lines:
        try:
            value = strict_json_loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _configured_path(config: Mapping[str, Any], project_root: Path, section: str, key: str) -> Path | None:
    values = config.get(section)
    raw = values.get(key) if isinstance(values, Mapping) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _runs_directory(config: Mapping[str, Any], project_root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied.resolve()
    paths = config.get("paths") if isinstance(config.get("paths"), Mapping) else {}
    raw = paths.get("runs", "runs/v3")
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _stage(stages: list[EvaluationStage], key: str) -> EvaluationStage:
    return next(item for item in stages if item.key == key)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.strip()
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_stage_payload(raw: object) -> bool:
    expected = new_evaluation_stages()
    if not isinstance(raw, list) or len(raw) != len(expected):
        return False
    for value, expected_stage in zip(raw, expected, strict=True):
        if not isinstance(value, Mapping):
            return False
        current = value.get("current")
        total = value.get("total")
        if (
            value.get("key") != expected_stage.key
            or value.get("title") != expected_stage.title
            or not isinstance(value.get("status"), str)
            or not isinstance(value.get("message"), str)
            or not isinstance(current, int)
            or isinstance(current, bool)
            or current < 0
            or (total is not None and (not isinstance(total, int) or isinstance(total, bool) or total < 0))
            or not isinstance(value.get("metrics"), Mapping)
        ):
            return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationService:
    """Product evaluation service with no promotion mutation or implicit generation."""

    def __init__(
        self,
        *,
        project_root: Path,
        config: Mapping[str, Any],
        runs_directory: Path | None = None,
        generator: EvaluationGenerator | None = None,
        evaluator: Evaluator | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        self.runs_directory = _runs_directory(config, self.project_root, runs_directory)
        self.generator = generator
        self._uses_default_evaluator = evaluator is None
        self.evaluator: Evaluator = evaluator or self._existing_evaluator
        self.output_root = (output_root or self.runs_directory).resolve()
        self.events = EventRepository(self.output_root, private_roots=(self.project_root,))
        self.latest_report: dict[str, Any] | None = None
        self.latest_rows: list[dict[str, Any]] = []
        self.latest_candidate_evidence: Path | None = None
        self.latest_memorization: dict[str, Any] = memorization_display(None)
        self.latest_stages: list[EvaluationStage] = new_evaluation_stages()
        self.latest_run_id: str | None = None
        self.latest_status: str = ProductStatus.NOT_STARTED.value
        self.latest_message = "No durable evaluation run is available."
        self._reconstruct_latest()

    @property
    def catalog(self) -> CheckpointCatalog:
        return discover_checkpoint_candidates(
            self.runs_directory,
            project_root=self.project_root,
            active_dataset_identity=expected_dataset_identity(self.config),
            active_view_identity=expected_training_view_identity(self.config),
        )

    @property
    def configured_benchmark(self) -> Path | None:
        return _configured_path(self.config, self.project_root, "evaluation", "benchmark")

    @property
    def review_log(self) -> Path | None:
        return _configured_path(self.config, self.project_root, "evaluation", "review_log")

    @property
    def memorization_audit(self) -> Path | None:
        return _configured_path(self.config, self.project_root, "evaluation", "memorization_audit")

    def _existing_evaluator(
        self,
        generated: Path,
        out: Path,
        *,
        checkpoint: Path,
        benchmark: Path,
        training_dataset_identity: str | None | object = _UNSET,
        training_view_identity: str | None | object = _UNSET,
    ) -> dict[str, Any]:
        from spritelab.evaluation.suite import score_suite

        manifests = self._training_manifests()
        dataset_identity = (
            expected_dataset_identity(self.config) if training_dataset_identity is _UNSET else training_dataset_identity
        )
        view_identity = (
            expected_training_view_identity(self.config) if training_view_identity is _UNSET else training_view_identity
        )
        return score_suite(
            generated,
            out,
            training_manifests=manifests or None,
            checkpoint=checkpoint,
            benchmark_manifest=benchmark,
            training_dataset_identity=str(dataset_identity) if dataset_identity is not None else None,
            training_view_identity=str(view_identity) if view_identity is not None else None,
        )

    def _training_manifests(self) -> list[Path]:
        configured = self.config.get("evaluation")
        raw_manifests = configured.get("training_manifests", ()) if isinstance(configured, Mapping) else ()
        if isinstance(raw_manifests, (str, Path)):
            raw_manifests = (raw_manifests,)
        manifests: list[Path] = []
        for raw in raw_manifests if isinstance(raw_manifests, (list, tuple)) else ():
            path = Path(str(raw)).expanduser()
            path = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
            if path.is_file():
                manifests.append(path)
        return manifests

    def _resolved_training_context(self) -> tuple[str | None, str | None]:
        """Freeze the evaluation's dataset/view authority before generation starts."""

        dataset_identity = expected_dataset_identity(self.config)
        view_identity = expected_training_view_identity(self.config)
        manifests = self._training_manifests()
        if len(manifests) != 1 or (dataset_identity is not None and view_identity is not None):
            return dataset_identity, view_identity

        manifest = manifests[0]
        dataset_values: list[str] = []
        view_values: list[str] = []
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = strict_json_loads(line)
                if not isinstance(row, Mapping):
                    continue
                source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
                raw_dataset = row.get("dataset_identity") or source.get("dataset_identity")
                raw_view = row.get("view_identity") or source.get("view_identity")
                if raw_dataset:
                    dataset_values.append(str(raw_dataset))
                if raw_view:
                    view_values.append(str(raw_view))
        except (OSError, UnicodeError, StrictJSONError, json.JSONDecodeError):
            dataset_values = []
            view_values = []
        return resolve_training_context_identities(
            dataset_identities=dataset_values,
            view_identities=view_values,
            manifest_sha256=_file_sha256(manifest),
            explicit_dataset_identity=dataset_identity,
            explicit_view_identity=view_identity,
        )

    def _validate_benchmark(self, path: Path | None) -> tuple[bool, str]:
        if path is None:
            return False, "Standard Sprite Lab benchmark is not configured."
        if not path.is_file():
            return False, "Standard Sprite Lab benchmark is missing."
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return False, "Standard Sprite Lab benchmark cannot be read."
        if not content:
            return False, "Standard Sprite Lab benchmark is empty."
        if path.suffix.lower() == ".jsonl":
            try:
                if not all(isinstance(strict_json_loads(line), dict) for line in content.splitlines() if line.strip()):
                    return False, "Standard Sprite Lab benchmark records must be objects."
            except (StrictJSONError, json.JSONDecodeError):
                return False, "Standard Sprite Lab benchmark is malformed."
        elif path.suffix.lower() == ".json":
            try:
                value = strict_json_loads(content)
            except (StrictJSONError, json.JSONDecodeError):
                return False, "Standard Sprite Lab benchmark is malformed."
            if not isinstance(value, dict):
                return False, "Standard Sprite Lab benchmark document must be an object."
        return True, "Standard Sprite Lab benchmark is valid."

    def plan(self, request: EvaluationRequest) -> tuple[CheckpointCandidate | None, Path | None, list[EvaluationStage]]:
        stages = new_evaluation_stages()
        dataset_identity = expected_dataset_identity(self.config)
        view_identity = expected_training_view_identity(self.config)
        identities_configured = bool(dataset_identity and view_identity)
        checkpoint = (
            self.catalog.find(request.checkpoint_id, weights=request.weights) if identities_configured else None
        )
        checkpoint_stage = _stage(stages, "checkpoint_validation")
        if not identities_configured:
            checkpoint_stage.status = "BLOCKED"
            checkpoint_stage.message = (
                "Active training dataset and view identities must both be configured before evaluation."
            )
        elif checkpoint is None:
            checkpoint_stage.status = "BLOCKED"
            checkpoint_stage.message = (
                "No eligible complete, verified checkpoint is bound to the active training dataset and view."
            )
        else:
            checkpoint_stage.status = "COMPLETE"
            checkpoint_stage.current = checkpoint_stage.total = 1
            checkpoint_stage.message = (
                f"{checkpoint.friendly_run_name}, step {checkpoint.checkpoint_step}, "
                f"{checkpoint.weights.upper()} is eligible."
            )
        benchmark = request.benchmark or self.configured_benchmark
        benchmark_valid, benchmark_message = self._validate_benchmark(benchmark)
        benchmark_stage = _stage(stages, "benchmark_validation")
        benchmark_stage.status = "COMPLETE" if benchmark_valid else "BLOCKED"
        benchmark_stage.current = 1 if benchmark_valid else 0
        benchmark_stage.total = 1
        benchmark_stage.message = benchmark_message
        ready = checkpoint is not None and benchmark_valid
        for stage in stages[2:-1]:
            stage.status = "NOT_STARTED" if ready else "BLOCKED"
            stage.message = "Ready after explicit Start evaluation action." if ready else "Blocked by validation."
        promotion = _stage(stages, "promotion_decision_report")
        integrity = promotion_integrity_display(self.memorization_audit, repository_root=self.project_root)
        promotion.status = "BLOCKED"
        promotion.message = str(integrity["message"])
        return checkpoint, benchmark, stages

    def run(self, request: EvaluationRequest) -> ProductResult:
        checkpoint, benchmark, stages = self.plan(request)
        self.latest_stages = stages
        integrity = promotion_integrity_display(self.memorization_audit, repository_root=self.project_root)
        blockers = [stage for stage in stages[:2] if stage.status == "BLOCKED"]
        if blockers:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="evaluation",
                message="Evaluation prerequisites are not satisfied.",
                blockers=tuple(ProductBlocker(stage.key, stage.message) for stage in blockers),
                data=self._result_data(stages, integrity, generation_runs=0),
            )
        if request.dry_run:
            return ProductResult(
                status=ProductStatus.COMPLETE,
                feature="evaluation",
                message="Evaluation plan validated; dry-run generated nothing and authorized no promotion.",
                data=self._result_data(stages, integrity, generation_runs=0, dry_run=True),
            )
        if not request.explicit_action:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="evaluation",
                message="Starting evaluation requires an explicit action.",
                blockers=(ProductBlocker("explicit_action_required", "Select Start evaluation to begin generation."),),
                data=self._result_data(stages, integrity, generation_runs=0),
            )
        if self.generator is None:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="evaluation",
                message="No typed evaluation generation adapter is configured.",
                blockers=(ProductBlocker("generator_unavailable", "Configure a typed generation adapter."),),
                data=self._result_data(stages, integrity, generation_runs=0),
            )
        billable = bool(getattr(self.generator, "remote", False) or getattr(self.generator, "billable", False))
        if billable and not request.confirm_billable:
            raise GenerationSafetyError("Remote or billable evaluation generation requires explicit confirmation.")
        assert checkpoint is not None and checkpoint.path is not None and benchmark is not None

        checkpoint_path = checkpoint.path.resolve()
        benchmark_path = benchmark.resolve()
        try:
            checkpoint_sha256 = _file_sha256(checkpoint_path)
            benchmark_sha256 = _file_sha256(benchmark_path)
        except OSError:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="evaluation",
                message="Evaluation inputs could not be bound to immutable launch identities.",
                blockers=(
                    ProductBlocker(
                        "evaluation_identity",
                        "Evaluation input identity verification failed safely.",
                    ),
                ),
                data=self._result_data(stages, integrity, generation_runs=0),
            )

        run_id = f"evaluation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        started_at = _utc_now()
        training_dataset_identity, training_view_identity = self._resolved_training_context()
        directory = self.events.run_directory(run_id, create=True)
        if directory is None:
            raise OSError("Evaluation runs directory is unavailable.")
        generated_directory = directory / "generated"
        metrics_directory = directory / "metrics"
        generated_directory.mkdir(parents=True, exist_ok=False)
        self.events.create_run(
            run_id,
            feature="evaluation",
            command="evaluation.run",
            status=ProductStatus.RUNNING.value,
            stage="checkpoint_validation",
            started_at=started_at,
            resumable=False,
            backend_id=type(self.generator).__name__,
            backend_identity={"remote": bool(getattr(self.generator, "remote", False))},
            report_reference="product_report.json",
            extra={
                "evaluation_schema_version": EVALUATION_STATE_SCHEMA,
                "selected_checkpoint": checkpoint.to_dict(technical_details=False),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "benchmark_path": str(benchmark_path),
                "benchmark_sha256": benchmark_sha256,
                "training_dataset_identity": training_dataset_identity,
                "training_view_identity": training_view_identity,
                "weights": request.weights,
                "stages": [stage.to_dict() for stage in stages],
                "result_status": ProductStatus.RUNNING.value,
                "review_requirement": None,
                "generated_reference": "generated",
                "metrics_reference": "metrics/per_image_metrics.jsonl",
                "expected_artifacts": {
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "kind": "file",
                        "sha256": checkpoint_sha256,
                    },
                    "benchmark": {
                        "path": str(benchmark_path),
                        "kind": "file",
                        "sha256": benchmark_sha256,
                    },
                },
            },
        )
        self.latest_run_id = run_id
        self._append_event(
            run_id,
            stage="checkpoint_validation",
            event_type="evaluation_started",
            status=ProductStatus.RUNNING,
            current=2,
            total=len(stages),
            message="Evaluation started after checkpoint and benchmark validation.",
        )

        def emit(stage_key: str, current: int, total: int | None, message: str) -> None:
            stage = _stage(stages, stage_key)
            stage.status = "RUNNING"
            stage.current = current
            stage.total = total
            stage.message = message
            self.events.update_state(run_id, stage=stage_key, stages=[item.to_dict() for item in stages])
            self._append_event(
                run_id,
                stage=stage_key,
                event_type="stage_progress",
                status=ProductStatus.RUNNING,
                current=current,
                total=total,
                message=message,
            )

        generation = _stage(stages, "generation")
        generation.status = "RUNNING"
        generation_runs = 0
        try:
            benchmark_valid, benchmark_message = self._validate_benchmark(benchmark_path)
            if not benchmark_valid:
                raise ValueError(f"Benchmark changed after planning: {benchmark_message}")
            if _file_sha256(benchmark_path) != benchmark_sha256:
                raise ValueError("Benchmark identity changed after planning.")
            if _file_sha256(checkpoint_path) != checkpoint_sha256:
                raise ValueError("Checkpoint identity changed after planning.")
            current_checkpoint = self.catalog.find(request.checkpoint_id, weights=request.weights)
            if (
                current_checkpoint is None
                or current_checkpoint.path is None
                or current_checkpoint.path.resolve() != checkpoint_path
            ):
                raise ValueError("Checkpoint eligibility changed after planning.")
            generation_runs = 1
            generated = self.generator.generate_benchmark(
                checkpoint=checkpoint_path,
                benchmark=benchmark_path,
                output_directory=generated_directory,
                weights=request.weights,
                emit=emit,
            )
            generated = generated.resolve()
            try:
                generated.relative_to(directory)
            except ValueError as error:
                raise ValueError(
                    "Evaluation generator returned artifacts outside its durable run directory."
                ) from error
            generation.status = "COMPLETE"
            generation.current = generation.total = 1
            generation.message = "Benchmark generation completed."
            self._append_event(
                run_id,
                stage="generation",
                event_type="stage_complete",
                status=ProductStatus.RUNNING,
                current=3,
                total=len(stages),
                message=generation.message,
                artifacts=("generated",),
            )
            if self._uses_default_evaluator:
                report = self.evaluator(
                    generated,
                    metrics_directory,
                    checkpoint=checkpoint.path,
                    benchmark=benchmark,
                    training_dataset_identity=training_dataset_identity,
                    training_view_identity=training_view_identity,
                )
            else:
                report = self.evaluator(generated, metrics_directory)
        except Exception:
            active = next((stage for stage in stages if stage.status == "RUNNING"), generation)
            active.status = "FAILED"
            active.message = "Evaluation stage failed safely; adapter diagnostics remain private."
            for stage in stages[stages.index(active) + 1 : -1]:
                stage.status = "BLOCKED"
                stage.message = "Blocked by an earlier failed stage."
            message = "Evaluation stopped at a failed stage; completed stage data was preserved."
            self.events.update_state(
                run_id,
                status=ProductStatus.FAILED.value,
                result_status=ProductStatus.FAILED.value,
                stage=active.key,
                message=message,
                stages=[stage.to_dict() for stage in stages],
            )
            self._append_event(
                run_id,
                stage=active.key,
                event_type="evaluation_failed",
                status=ProductStatus.FAILED,
                current=sum(stage.status == "COMPLETE" for stage in stages),
                total=len(stages),
                message=message,
            )
            self.latest_status = ProductStatus.FAILED.value
            self.latest_message = message
            return ProductResult(
                status=ProductStatus.FAILED,
                feature="evaluation",
                run=self._product_run(run_id, ProductStatus.FAILED, started_at),
                message=message,
                blockers=(ProductBlocker(active.key, active.message),),
                data=self._result_data(stages, integrity, generation_runs=generation_runs),
            )

        self.latest_report = report
        self.latest_rows = _read_jsonl(metrics_directory / "per_image_metrics.jsonl")
        candidate_path = metrics_directory / "candidate_evidence.json"
        self.latest_candidate_evidence = candidate_path if candidate_path.is_file() else None
        for key in (
            "structural_metrics",
            "conditional_metrics",
            "diversity",
            "palette_analysis",
            "memorization_detector",
        ):
            stage = _stage(stages, key)
            stage.status = "COMPLETE"
            stage.current = stage.total = 1
            stage.message = f"{stage.title} completed from existing evaluation backend data."
            self._append_event(
                run_id,
                stage=key,
                event_type="stage_complete",
                status=ProductStatus.RUNNING,
                current=sum(item.status == "COMPLETE" for item in stages),
                total=len(stages),
                message=stage.message,
            )
        memo = report.get("summary", {}).get("memorization", {}) if isinstance(report.get("summary"), Mapping) else {}
        reported_review_count = int(memo.get("review_required_count", 0)) if isinstance(memo, Mapping) else 0
        hard_count = int(memo.get("hard_evidence_count", 0)) if isinstance(memo, Mapping) else 0
        memo_display = memorization_display(
            self.latest_candidate_evidence,
            review_log=self.review_log,
            expected_context={
                "checkpoint_path": checkpoint.path,
                "benchmark_manifest_path": benchmark,
                "training_dataset_identity": training_dataset_identity,
                "training_view_identity": training_view_identity,
            },
        )
        self.latest_memorization = memo_display
        evidence_complete = memo_display.get("evidence_state") == "complete"
        review_count = int(memo_display["review_required_count"]) if evidence_complete else reported_review_count
        review = _stage(stages, "review_completeness")
        if not evidence_complete:
            review.status = "BLOCKED"
            review.current = 0
            review.total = 1
            review.message = str(memo_display.get("review_message") or INCOMPLETE_EVIDENCE_MESSAGE)
        else:
            review.status = "NEEDS_REVIEW" if review_count else "COMPLETE"
            review.current = 0 if review_count else 1
            review.total = 1
            review.message = (
                f"{review_count} memorization candidates require review"
                if review_count
                else "Review queue is complete."
            )
        dashboard = build_dashboard(
            report,
            self.latest_rows,
            allow_source_results=request.allow_source_results,
            private_roots=(self.project_root,),
        )
        status = (
            ProductStatus.BLOCKED
            if hard_count or not evidence_complete
            else (ProductStatus.NEEDS_REVIEW if review_count else ProductStatus.COMPLETE)
        )
        message = (
            "Hard memorization evidence blocks the final gate."
            if hard_count
            else str(memo_display.get("review_message") or INCOMPLETE_EVIDENCE_MESSAGE)
            if not evidence_complete
            else "Evaluation requires memorization review."
            if review_count
            else "Evaluation completed; promotion remains unauthorized."
        )
        report_path = directory / "product_report.json"
        report_path.write_text(
            strict_json_dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        expected_artifacts = {
            "checkpoint": {
                "path": str(checkpoint_path),
                "kind": "file",
                "sha256": checkpoint_sha256,
            },
            "benchmark": {"path": str(benchmark_path), "kind": "file", "sha256": benchmark_sha256},
            "generated": {"path": "generated", "kind": "tree", "sha256": _tree_sha256(generated_directory)},
            "report": {"path": "product_report.json", "kind": "file", "sha256": _file_sha256(report_path)},
        }
        rows_path = metrics_directory / "per_image_metrics.jsonl"
        if rows_path.is_file():
            expected_artifacts["per_image_metrics"] = {
                "path": "metrics/per_image_metrics.jsonl",
                "kind": "file",
                "sha256": _file_sha256(rows_path),
            }
        summary_path = metrics_directory / "summary.json"
        if summary_path.is_file():
            expected_artifacts["machine_report"] = {
                "path": "metrics/summary.json",
                "kind": "file",
                "sha256": _file_sha256(summary_path),
            }
        detector_policy_path = metrics_directory / "detector_policy.json"
        if detector_policy_path.is_file():
            expected_artifacts["detector_policy"] = {
                "path": "metrics/detector_policy.json",
                "kind": "file",
                "sha256": _file_sha256(detector_policy_path),
            }
        if self.latest_candidate_evidence is not None:
            expected_artifacts["candidate_evidence"] = {
                "path": "metrics/candidate_evidence.json",
                "kind": "file",
                "sha256": _file_sha256(self.latest_candidate_evidence),
            }
        self.events.update_state(
            run_id,
            status=status.value,
            result_status=status.value,
            stage="review_completeness",
            message=message,
            stages=[stage.to_dict() for stage in stages],
            review_requirement={
                "required": not evidence_complete or bool(review_count),
                "count": review_count,
                "hard_block_count": hard_count,
                "evidence_state": memo_display.get("evidence_state"),
            },
            expected_artifacts=expected_artifacts,
            report_available=True,
            ended_at=_utc_now(),
        )
        self._append_event(
            run_id,
            stage="review_completeness",
            event_type="evaluation_complete",
            status=status,
            current=sum(stage.status == "COMPLETE" for stage in stages),
            total=len(stages),
            message=message,
            artifacts=("generated", "product_report.json", "metrics/per_image_metrics.jsonl"),
        )
        self.latest_status = status.value
        self.latest_message = message
        data = self._result_data(stages, integrity, generation_runs=1)
        data.update({"dashboard": dashboard, "memorization": memo_display, "report": report})
        data = public_evaluation_projection(data, surface="run_data", private_roots=(self.project_root,))
        return ProductResult(
            status=status,
            feature="evaluation",
            run=self._product_run(run_id, status, started_at),
            message=message,
            data=data,
        )

    def _append_event(
        self,
        run_id: str,
        *,
        stage: str,
        event_type: str,
        status: ProductStatus,
        current: int,
        total: int | None,
        message: str,
        artifacts: tuple[str, ...] = (),
    ) -> None:
        self.events.append(
            ProductEvent(
                run_id=run_id,
                timestamp=_utc_now(),
                feature="evaluation",
                stage=stage,
                event_type=event_type,
                status=status,
                current=current,
                total=total,
                message=message,
                artifact_references=artifacts,
            )
        )

    def _product_run(self, run_id: str, status: ProductStatus, started_at: str) -> ProductRun:
        state = self.events.state(run_id)
        return ProductRun(
            run_id=run_id,
            feature="evaluation",
            action_id="evaluation.run",
            status=status,
            backend_id=str(state.get("backend_id") or "evaluation"),
            started_at=started_at,
            ended_at=str(state.get("ended_at")) if state.get("ended_at") else None,
            artifact_references=("product_report.json",),
        )

    def _reconstruct_latest(self) -> None:
        for run_id in self.events.recent_run_ids(feature="evaluation"):
            state = self.events.state(run_id)
            if state.get("evaluation_schema_version") != EVALUATION_STATE_SCHEMA:
                continue
            self.latest_run_id = run_id
            self.latest_stages = self._stages_from_state(state)
            stale_reasons = self._artifact_reasons(run_id, state)
            if stale_reasons:
                self.latest_status = "STALE"
                self.latest_message = (
                    "Evaluation artifacts are missing or changed; results are stale and not comparable."
                )
                self.events.update_state(
                    run_id,
                    status="STALE",
                    message=self.latest_message,
                    stale_reasons=stale_reasons,
                )
                for stage in self.latest_stages:
                    if stage.key in {"checkpoint_validation", "benchmark_validation", "generation"}:
                        stage.status = "BLOCKED"
                        stage.message = self.latest_message
                return
            directory = self.events.run_directory(run_id)
            if directory is None:
                return
            report_path = directory / "product_report.json"
            try:
                report = strict_json_loads(report_path.read_bytes())
            except (OSError, UnicodeError, json.JSONDecodeError):
                report = None
            if isinstance(report, dict):
                self.latest_report = report
            metrics_directory = directory / "metrics"
            self.latest_rows = _read_jsonl(metrics_directory / "per_image_metrics.jsonl")
            candidate_path = metrics_directory / "candidate_evidence.json"
            self.latest_candidate_evidence = candidate_path if candidate_path.is_file() else None
            expected_context: dict[str, Any] = {
                "training_dataset_identity": state.get("training_dataset_identity"),
                "training_view_identity": state.get("training_view_identity"),
            }
            checkpoint_path = state.get("checkpoint_path")
            benchmark_path = state.get("benchmark_path")
            if checkpoint_path:
                expected_context["checkpoint_path"] = Path(str(checkpoint_path))
            if benchmark_path:
                expected_context["benchmark_manifest_path"] = Path(str(benchmark_path))
            self.latest_memorization = memorization_display(
                self.latest_candidate_evidence,
                review_log=self.review_log,
                expected_context=expected_context,
            )
            self.latest_status = str(state.get("result_status") or state.get("status") or "NOT_STARTED")
            self.latest_message = str(state.get("message") or "Durable evaluation state reconstructed.")
            if self.latest_memorization.get("evidence_state") != "complete":
                self.latest_status = ProductStatus.BLOCKED.value
                self.latest_message = str(self.latest_memorization.get("review_message") or INCOMPLETE_EVIDENCE_MESSAGE)
                review_stage = _stage(self.latest_stages, "review_completeness")
                review_stage.status = "BLOCKED"
                review_stage.current = 0
                review_stage.total = 1
                review_stage.message = self.latest_message
            return

    def _stages_from_state(self, state: Mapping[str, Any]) -> list[EvaluationStage]:
        raw = state.get("stages")
        if not _valid_stage_payload(raw):
            return new_evaluation_stages()
        assert isinstance(raw, list)
        return [
            EvaluationStage(
                key=str(value["key"]),
                title=str(value["title"]),
                status=str(value["status"]),
                message=(
                    "Evaluation stage failed safely; adapter diagnostics remain private."
                    if value["status"] == "FAILED"
                    else str(value["message"])
                ),
                current=int(value["current"]),
                total=int(value["total"]) if value["total"] is not None else None,
                metrics=dict(value["metrics"]),
            )
            for value in raw
            if isinstance(value, Mapping)
        ]

    def _artifact_reasons(self, run_id: str, state: Mapping[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not _valid_stage_payload(state.get("stages")):
            reasons.append("persisted evaluation stages are malformed")
        for field in ("training_dataset_identity", "training_view_identity"):
            if field not in state:
                reasons.append(f"persisted {field} is missing")
            elif not isinstance(state.get(field), str) or not state[field] or state[field] != state[field].strip():
                reasons.append(f"persisted {field} is malformed")

        persisted_dataset = state.get("training_dataset_identity")
        persisted_view = state.get("training_view_identity")
        current_dataset = expected_dataset_identity(self.config)
        current_view = expected_training_view_identity(self.config)
        if current_dataset is None:
            reasons.append("active training dataset identity is missing or malformed")
        elif persisted_dataset != current_dataset:
            reasons.append("active training dataset identity changed after evaluation")
        if current_view is None:
            reasons.append("active training view identity is missing or malformed")
        elif persisted_view != current_view:
            reasons.append("active training view identity changed after evaluation")

        checkpoint_path = state.get("checkpoint_path")
        checkpoint_sha256 = state.get("checkpoint_sha256")
        selected_path: Path | None = None
        if not isinstance(checkpoint_path, str) or not checkpoint_path or checkpoint_path != checkpoint_path.strip():
            reasons.append("persisted checkpoint_path is missing or malformed")
        else:
            try:
                selected_path = Path(checkpoint_path).resolve()
            except (OSError, RuntimeError, ValueError):
                reasons.append("persisted checkpoint_path cannot be resolved")
        if not _valid_sha256(checkpoint_sha256):
            reasons.append("persisted checkpoint_sha256 is missing or malformed")
        elif selected_path is not None:
            try:
                if not selected_path.is_file():
                    reasons.append("persisted checkpoint artifact is missing")
                elif _file_sha256(selected_path) != checkpoint_sha256:
                    reasons.append("persisted checkpoint SHA-256 changed after evaluation")
            except OSError:
                reasons.append("persisted checkpoint artifact cannot be read")

        if selected_path is not None:
            try:
                catalog = discover_checkpoint_candidates(
                    self.runs_directory,
                    project_root=self.project_root,
                    active_dataset_identity=persisted_dataset if isinstance(persisted_dataset, str) else None,
                    active_view_identity=persisted_view if isinstance(persisted_view, str) else None,
                )
                current_checkpoint = next(
                    (
                        item
                        for item in (*catalog.eligible, *catalog.unavailable)
                        if item.path is not None and item.path.resolve() == selected_path
                    ),
                    None,
                )
            except (OSError, RuntimeError, ValueError):
                current_checkpoint = None
            if current_checkpoint is None:
                reasons.append("checkpoint durable identity is unavailable")
            else:
                remains_eligible = any(
                    item.path is not None and item.path.resolve() == selected_path for item in catalog.eligible
                )
                if not remains_eligible:
                    reasons.append("checkpoint is no longer eligible under the persisted dataset/view identity")
                if persisted_dataset != current_checkpoint.dataset_identity:
                    reasons.append("checkpoint training dataset identity changed after evaluation")
                if persisted_view != current_checkpoint.view_identity:
                    reasons.append("checkpoint training view identity changed after evaluation")

        benchmark_path = state.get("benchmark_path")
        benchmark_sha256 = state.get("benchmark_sha256")
        selected_benchmark: Path | None = None
        if not isinstance(benchmark_path, str) or not benchmark_path or benchmark_path != benchmark_path.strip():
            reasons.append("persisted benchmark_path is missing or malformed")
        else:
            try:
                selected_benchmark = Path(benchmark_path).resolve()
            except (OSError, RuntimeError, ValueError):
                reasons.append("persisted benchmark_path cannot be resolved")
        if not _valid_sha256(benchmark_sha256):
            reasons.append("persisted benchmark_sha256 is missing or malformed")
        elif selected_benchmark is not None:
            try:
                if not selected_benchmark.is_file():
                    reasons.append("persisted benchmark artifact is missing")
                elif _file_sha256(selected_benchmark) != benchmark_sha256:
                    reasons.append("persisted benchmark SHA-256 changed after evaluation")
            except OSError:
                reasons.append("persisted benchmark artifact cannot be read")

        expected = state.get("expected_artifacts")
        if not isinstance(expected, Mapping):
            reasons.append("artifact identity manifest is missing")
            return reasons
        directory = self.events.run_directory(run_id)
        if directory is None:
            return [*reasons, "durable run directory is unavailable"]

        def require_expected_file(
            name: str,
            persisted_path: Path | None,
            persisted_sha256: object,
        ) -> None:
            raw = expected.get(name)
            if not isinstance(raw, Mapping):
                reasons.append(f"expected {name} identity record is missing or malformed")
                return
            raw_path = raw.get("path")
            if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
                reasons.append(f"expected {name} path is missing or malformed")
            else:
                try:
                    reference = Path(raw_path)
                    expected_path = (
                        reference.resolve() if reference.is_absolute() else (directory / reference).resolve()
                    )
                    if persisted_path is not None and expected_path != persisted_path:
                        reasons.append(f"expected {name} path disagrees with persisted launch identity")
                except (OSError, RuntimeError, ValueError):
                    reasons.append(f"expected {name} path cannot be resolved")
            if raw.get("kind") != "file":
                reasons.append(f"expected {name} kind is malformed")
            expected_sha256 = raw.get("sha256")
            if not _valid_sha256(expected_sha256):
                reasons.append(f"expected {name} SHA-256 is missing or malformed")
            elif _valid_sha256(persisted_sha256) and expected_sha256 != persisted_sha256:
                reasons.append(f"expected {name} SHA-256 disagrees with persisted launch identity")

        require_expected_file("checkpoint", selected_path, checkpoint_sha256)
        require_expected_file("benchmark", selected_benchmark, benchmark_sha256)
        for name, raw in expected.items():
            if not isinstance(raw, Mapping):
                reasons.append(f"{name} identity is malformed")
                continue
            raw_reference = raw.get("path")
            if not isinstance(raw_reference, str) or not raw_reference or raw_reference != raw_reference.strip():
                reasons.append(f"{name} reference is malformed")
                continue
            reference = Path(raw_reference)
            path = reference.resolve() if reference.is_absolute() else (directory / reference).resolve()
            if not reference.is_absolute():
                try:
                    path.relative_to(directory)
                except ValueError:
                    reasons.append(f"{name} reference escapes the run directory")
                    continue
            kind = raw.get("kind")
            if kind not in {"file", "tree"}:
                reasons.append(f"{name} artifact kind is malformed")
                continue
            if not _valid_sha256(raw.get("sha256")):
                reasons.append(f"{name} artifact SHA-256 is malformed")
                continue
            if (kind == "tree" and not path.is_dir()) or (kind == "file" and not path.is_file()):
                reasons.append(f"{name} artifact is missing")
                continue
            try:
                actual = _tree_sha256(path) if kind == "tree" else _file_sha256(path)
            except OSError:
                reasons.append(f"{name} artifact cannot be read")
                continue
            if actual != raw.get("sha256"):
                reasons.append(f"{name} artifact identity changed")

        candidate_raw = expected.get("candidate_evidence")
        if isinstance(candidate_raw, Mapping):
            candidate_reference = Path(str(candidate_raw.get("path") or ""))
            candidate_path = (
                candidate_reference.resolve()
                if candidate_reference.is_absolute()
                else (directory / candidate_reference).resolve()
            )
            if not persisted_dataset or not persisted_view:
                reasons.append("candidate evidence lacks persisted training dataset/view authority")
            elif candidate_path.is_file():
                candidate_context: dict[str, Any] = {
                    "training_dataset_identity": persisted_dataset,
                    "training_view_identity": persisted_view,
                }
                if checkpoint_path:
                    candidate_context["checkpoint_path"] = Path(str(checkpoint_path))
                benchmark_path = state.get("benchmark_path")
                if benchmark_path:
                    candidate_context["benchmark_manifest_path"] = Path(str(benchmark_path))
                validation = load_candidate_bundle(candidate_path, expected_context=candidate_context)
                if not validation.valid:
                    reasons.extend(f"candidate evidence identity invalid: {reason}" for reason in validation.reasons)
        return reasons

    def dashboard(self, *, allow_source_results: bool = False) -> dict[str, Any]:
        value = build_dashboard(
            self.latest_report or {},
            self.latest_rows,
            allow_source_results=allow_source_results,
            private_roots=(self.project_root,),
        )
        value.update(
            {
                "run_id": self.latest_run_id,
                "status": self.latest_status,
                "message": self.latest_message,
                "stale": self.latest_status == "STALE",
                "memorization": self.latest_memorization,
            }
        )
        return public_evaluation_projection(value, surface="dashboard", private_roots=(self.project_root,))

    def _result_data(
        self,
        stages: list[EvaluationStage],
        integrity: Mapping[str, Any],
        *,
        generation_runs: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "spritelab.product.evaluation-run.v1",
            "stages": [stage.to_dict() for stage in stages],
            "progress": {
                "completed": sum(stage.status == "COMPLETE" for stage in stages),
                "total": len(stages),
            },
            "promotion": dict(integrity),
            "generation_runs": generation_runs,
            "promotion_actions": 0,
            "dry_run": dry_run,
        }
        return public_evaluation_projection(payload, surface="run_data", private_roots=(self.project_root,))
