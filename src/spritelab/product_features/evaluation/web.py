"""FastAPI router for the evaluation dashboard and prompt playground."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from starlette.requests import Request

from spritelab.product_core import ProductStatus, ProjectContext, api_error, product_api, strict_json_dumps
from spritelab.product_features.evaluation.dashboard import (
    IncompatibleMetricDefinitions,
    build_dashboard,
    compare_evaluations,
    filter_gallery,
    public_evaluation_projection,
)
from spritelab.product_features.evaluation.exploratory_smoke import (
    ExploratorySmokeWorkflow,
)
from spritelab.product_features.evaluation.local_generator import LocalCheckpointPlaygroundGenerator
from spritelab.product_features.evaluation.playground import (
    GenerationCancelledError,
    GenerationRequest,
    GenerationSafetyError,
    GenerationTimedOutError,
    GeneratorUnavailableError,
    PlaygroundGenerator,
    PlaygroundService,
)
from spritelab.product_features.evaluation.service import EvaluationRequest, EvaluationService
from spritelab.training.smoke_runner import ExploratorySmokeRunner


def _resource_text(name: str) -> str:
    return files("spritelab.product_features.evaluation").joinpath(name).read_text(encoding="utf-8")


def _strict_payload_bool(payload: dict[str, Any], name: str, *, default: bool = False) -> bool:
    value = payload.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def create_evaluation_router(
    context: ProjectContext,
    *,
    service: EvaluationService | None = None,
    playground_generator: PlaygroundGenerator | None = None,
    smoke_workflow: ExploratorySmokeWorkflow | None = None,
    smoke_runner: ExploratorySmokeRunner | None = None,
) -> Any:
    """Build an isolated router; GET routes never invoke a generator."""

    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

    router = APIRouter()
    evaluation = service or EvaluationService(
        project_root=context.project_root,
        config=context.config,
        runs_directory=context.runs_directory,
    )
    exploratory_smoke_workflow = smoke_workflow or ExploratorySmokeWorkflow(context.project_root)
    exploratory_smoke_runner = smoke_runner or ExploratorySmokeRunner(context.project_root)
    exploratory_catalog = exploratory_smoke_workflow.catalog()
    playground = PlaygroundService(
        evaluation.catalog,
        output_root=evaluation.output_root / "playground",
        generator=playground_generator
        or LocalCheckpointPlaygroundGenerator(
            project_root=context.project_root,
            work_root=context.runs_directory / "playground-sampler-work",
        ),
        runs_directory=evaluation.runs_directory,
        catalog_provider=lambda: evaluation.catalog,
        exploratory_catalog=exploratory_catalog,
        exploratory_catalog_provider=exploratory_smoke_workflow.catalog,
    )

    @router.get("/evaluation", response_class=HTMLResponse)
    def evaluation_page(request: Request) -> Any:
        promotion = public_evaluation_projection(
            evaluation.plan(EvaluationRequest(dry_run=True))[2][-1].to_dict(),
            surface="stage",
            private_roots=(context.project_root,),
        )
        durable_run = public_evaluation_projection(
            {
                "run_id": evaluation.latest_run_id,
                "status": evaluation.latest_status,
                "message": evaluation.latest_message,
                "stages": [stage.to_dict() for stage in evaluation.latest_stages],
                "dashboard": evaluation.dashboard(),
            },
            surface="durable_run",
            private_roots=(context.project_root,),
        )
        initial = {
            "checkpoints": evaluation.catalog.to_dict(private_roots=(context.project_root,)),
            "exploratory_checkpoints": exploratory_catalog.to_dict(),
            "smoke_publications": exploratory_smoke_workflow.eligible_publications(),
            "smoke_plans": exploratory_smoke_workflow.prepared_plans(),
            "playground_defaults": playground.defaults(),
            "playground_run": playground.latest_run(),
            "promotion": promotion,
            "durable_run": durable_run,
        }
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(
                request,
                "evaluation.playground",
                "evaluation.html",
                {"evaluation_initial_state": initial},
            )
        standalone = _resource_text("templates/evaluation_standalone.html")
        return standalone.replace(
            "__INITIAL_STATE__", strict_json_dumps(initial, ensure_ascii=False).replace("<", "\\u003c")
        )

    @router.get("/playground")
    def playground_page() -> RedirectResponse:
        return RedirectResponse("/evaluation#playground", status_code=303)

    @router.get("/evaluation/static/evaluation.css")
    def evaluation_css() -> Response:
        return Response(_resource_text("static/evaluation.css"), media_type="text/css")

    @router.get("/evaluation/static/evaluation-a11y.css")
    def evaluation_a11y_css() -> Response:
        return Response(_resource_text("static/evaluation-a11y.css"), media_type="text/css")

    @router.get("/evaluation/static/exploratory-smoke.css")
    def exploratory_smoke_css() -> Response:
        return Response(_resource_text("static/exploratory-smoke.css"), media_type="text/css")

    @router.get("/evaluation/static/evaluation.js")
    def evaluation_js() -> Response:
        return Response(_resource_text("static/evaluation.js"), media_type="application/javascript")

    @router.get("/evaluation/static/exploratory-smoke-standalone.js")
    def exploratory_smoke_standalone_js() -> Response:
        return Response(_resource_text("static/exploratory-smoke-standalone.js"), media_type="application/javascript")

    @router.get("/evaluation/api/checkpoints")
    @product_api
    def checkpoints(
        include_unavailable: bool = False,
    ) -> dict[str, Any]:
        return evaluation.catalog.to_dict(
            include_unavailable=include_unavailable,
            technical_details=False,
            private_roots=(context.project_root,),
        )

    @router.get("/evaluation/api/plan")
    @product_api
    def plan(checkpoint_id: str | None = None, weights: str = "ema") -> dict[str, Any]:
        _checkpoint, _benchmark, stages = evaluation.plan(
            EvaluationRequest(checkpoint_id=checkpoint_id, weights=weights, dry_run=True)
        )
        return {
            "stages": [
                public_evaluation_projection(
                    stage.to_dict(),
                    surface="stage",
                    private_roots=(context.project_root,),
                )
                for stage in stages
            ]
        }

    @router.post("/evaluation/api/run")
    @product_api
    def start_evaluation(payload: dict[str, Any]) -> JSONResponse:
        if any(key in payload for key in ("benchmark", "benchmark_path", "path")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Evaluation uses the project-configured benchmark; browser-supplied server paths are not accepted.",
                next_action="Configure the benchmark through the local project configuration.",
            )
        try:
            request = EvaluationRequest(
                checkpoint_id=payload.get("checkpoint_id"),
                weights=str(payload.get("weights") or "ema"),
                dry_run=_strict_payload_bool(payload, "dry_run"),
                explicit_action=_strict_payload_bool(payload, "explicit_action"),
                confirm_billable=_strict_payload_bool(payload, "confirm_billable"),
                allow_source_results=_strict_payload_bool(payload, "allow_source_results"),
            )
        except ValueError:
            return api_error(
                422,
                "evaluation_boolean_invalid",
                "Evaluation action flags must be JSON booleans.",
                next_action="Reload the Evaluation page and submit the action again.",
            )
        try:
            result = evaluation.run(request)
        except GenerationSafetyError as exc:
            raise HTTPException(status_code=409, detail="Evaluation generation was refused safely.") from exc
        if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
            next_action = (
                result.blockers[0].resolution
                if result.blockers and result.blockers[0].resolution
                else "Resolve the displayed evaluation prerequisite, then try again."
            )
            return api_error(
                409,
                "evaluation_run_blocked",
                result.message,
                recoverable=True,
                next_action=next_action,
            )
        return JSONResponse(
            public_evaluation_projection(
                result.to_dict(),
                surface="result",
                private_roots=(context.project_root,),
            )
        )

    @router.get("/evaluation/api/dashboard")
    @product_api
    def dashboard(allow_source_results: bool = False) -> dict[str, Any]:
        return evaluation.dashboard(allow_source_results=allow_source_results)

    @router.get("/evaluation/api/gallery")
    @product_api
    def gallery(
        prompt: str | None = None,
        seed: int | None = None,
        checkpoint: str | None = None,
        weights: str | None = None,
        category: str | None = None,
        sort_metric: str | None = None,
        descending: bool = True,
    ) -> dict[str, Any]:
        raw = build_dashboard(
            evaluation.latest_report or {},
            evaluation.latest_rows,
            private_roots=(context.project_root,),
        )["gallery"]
        return {
            "samples": filter_gallery(
                raw,
                prompt=prompt,
                seed=seed,
                checkpoint=checkpoint,
                weights=weights,
                category=category,
                sort_metric=sort_metric,
                descending=descending,
            )
        }

    @router.post("/evaluation/api/compare")
    @product_api
    def compare(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return compare_evaluations(
                payload.get("left_report") or {},
                payload.get("right_report") or {},
                payload.get("left_rows") or (),
                payload.get("right_rows") or (),
            )
        except IncompatibleMetricDefinitions as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/evaluation/api/report-data")
    @product_api
    def report_data() -> JSONResponse:
        payload = public_evaluation_projection(
            {
                "report": evaluation.latest_report,
                "per_image_metrics": evaluation.latest_rows,
                "stages": [stage.to_dict() for stage in evaluation.latest_stages],
                "run_id": evaluation.latest_run_id,
                "status": evaluation.latest_status,
                "message": evaluation.latest_message,
                "promotion_actions": 0,
            },
            surface="report_data",
            private_roots=(context.project_root,),
        )
        return JSONResponse(
            payload,
            headers={"Content-Disposition": 'attachment; filename="sprite-lab-evaluation-report.json"'},
        )

    @router.get("/evaluation/api/playground/defaults")
    @product_api
    def playground_defaults() -> dict[str, Any]:
        return {
            "defaults": playground.defaults(),
            "scope": "EXPLORATORY",
            "billable_confirmation_required": playground.confirmation_required,
            "exploratory_checkpoints": exploratory_smoke_workflow.catalog().to_dict(),
        }

    @router.get("/evaluation/api/playground/exploratory-checkpoints")
    @product_api
    def exploratory_checkpoints() -> dict[str, Any]:
        return exploratory_smoke_workflow.catalog().to_dict()

    @router.get("/evaluation/api/playground/smoke-publications")
    @product_api
    def smoke_publications() -> dict[str, Any]:
        return exploratory_smoke_workflow.eligible_publications()

    @router.get("/evaluation/api/playground/smoke-plans")
    @product_api
    def smoke_plans() -> dict[str, Any]:
        return exploratory_smoke_workflow.prepared_plans()

    @router.post("/evaluation/api/playground/smokes/prepare")
    @product_api
    def prepare_exploratory_smoke(payload: dict[str, Any]) -> Any:
        allowed = {
            "conditioned_job_id",
            "preparation_nonce",
            "explicit_action",
        }
        if set(payload) != allowed:
            return api_error(422, "smoke_prepare_payload", "Use only the exact opaque smoke preparation fields.")
        try:
            return exploratory_smoke_workflow.prepare_job(
                str(payload["conditioned_job_id"]),
                str(payload["preparation_nonce"]),
                explicit_action=payload["explicit_action"] is True,
            )
        except (OSError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "smoke_prepare_blocked")
            message = getattr(exc, "public_message", "Exploratory smoke preparation was refused safely.")
            return api_error(409, str(code), str(message))

    def run_smoke_device(payload: dict[str, Any], device: str) -> Any:
        allowed = {"conditioned_job_id", "smoke_id", "plan_identity", "explicit_action"}
        if set(payload) != allowed:
            return api_error(422, "smoke_run_payload", "Use only the selected job and server-prepared smoke identity.")
        try:
            plan = exploratory_smoke_workflow.validate_job_plan(
                str(payload["conditioned_job_id"]),
                str(payload["smoke_id"]),
                str(payload["plan_identity"]),
            )
            return exploratory_smoke_runner.launch(
                str(plan["smoke_id"]),
                str(plan["plan_identity"]),
                device,
                explicit_action=payload["explicit_action"] is True,
            )
        except (OSError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "smoke_run_blocked")
            message = getattr(exc, "public_message", "The fixed smoke process was refused safely.")
            return api_error(409, str(code), str(message))

    @router.post("/evaluation/api/playground/smokes/run-cpu")
    @product_api
    def run_cpu_smoke(payload: dict[str, Any]) -> Any:
        return run_smoke_device(payload, "cpu")

    @router.post("/evaluation/api/playground/smokes/run-cuda")
    @product_api
    def run_cuda_smoke(payload: dict[str, Any]) -> Any:
        return run_smoke_device(payload, "cuda")

    def cancel_smoke_device(payload: dict[str, Any], device: str) -> Any:
        allowed = {"conditioned_job_id", "smoke_id", "plan_identity", "explicit_action"}
        if set(payload) != allowed:
            return api_error(422, "smoke_cancel_payload", "Use only the selected server-prepared smoke identity.")
        try:
            plan = exploratory_smoke_workflow.validate_job_plan(
                str(payload["conditioned_job_id"]),
                str(payload["smoke_id"]),
                str(payload["plan_identity"]),
            )
            return exploratory_smoke_runner.cancel(
                str(plan["smoke_id"]),
                str(plan["plan_identity"]),
                device,
                explicit_action=payload["explicit_action"] is True,
            )
        except (OSError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "smoke_cancel_blocked")
            message = getattr(exc, "public_message", "The contained smoke cancellation was refused safely.")
            return api_error(409, str(code), str(message))

    @router.post("/evaluation/api/playground/smokes/cancel-cpu")
    @product_api
    def cancel_cpu_smoke(payload: dict[str, Any]) -> Any:
        return cancel_smoke_device(payload, "cpu")

    @router.post("/evaluation/api/playground/smokes/cancel-cuda")
    @product_api
    def cancel_cuda_smoke(payload: dict[str, Any]) -> Any:
        return cancel_smoke_device(payload, "cuda")

    @router.get("/evaluation/api/playground/smokes/{smoke_id}/status")
    @product_api
    def smoke_status(smoke_id: str, conditioned_job_id: str, plan_identity: str) -> Any:
        try:
            exploratory_smoke_workflow.validate_job_plan(conditioned_job_id, smoke_id, plan_identity)
            return exploratory_smoke_runner.bundle_status(smoke_id)
        except (OSError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "smoke_status_unavailable")
            message = getattr(exc, "public_message", "The smoke execution status is unavailable.")
            return api_error(409, str(code), str(message))

    @router.post("/evaluation/api/playground/smokes/register")
    @product_api
    def register_exploratory_smoke(payload: dict[str, Any]) -> Any:
        allowed = {
            "conditioned_job_id",
            "smoke_id",
            "plan_identity",
            "explicit_action",
        }
        if set(payload) != allowed:
            return api_error(422, "smoke_registration_payload", "Use only exact opaque registration identities.")
        try:
            executions = exploratory_smoke_runner.require_complete(str(payload["smoke_id"]))
            return exploratory_smoke_workflow.register_job(
                str(payload["conditioned_job_id"]),
                str(payload["smoke_id"]),
                str(payload["plan_identity"]),
                explicit_action=payload["explicit_action"] is True,
                server_execution_identities={
                    device: str(value["execution_identity"]) for device, value in executions.items()
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "smoke_registration_blocked")
            message = getattr(exc, "public_message", "Exploratory smoke registration was refused safely.")
            return api_error(409, str(code), str(message))

    @router.post("/evaluation/api/playground/generate")
    @product_api
    def playground_generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = GenerationRequest(
                prompt=str(payload.get("prompt") or ""),
                checkpoint_id=str(payload.get("checkpoint_id") or playground.defaults()["checkpoint_id"]),
                weights=str(payload.get("weights") or "ema"),
                seed=int(payload.get("seed", 42)),
                sampling_steps=int(payload.get("sampling_steps", 30)),
                guidance=float(payload.get("guidance", 3.0)),
                image_count=int(payload.get("image_count", 4)),
            )
            return playground.generate(
                request,
                explicit_action=_strict_payload_bool(payload, "explicit_action"),
                confirm_billable=_strict_payload_bool(payload, "confirm_billable"),
            )
        except GenerationTimedOutError as exc:
            raise HTTPException(status_code=408, detail="Playground generation reached its fixed deadline.") from exc
        except GenerationCancelledError as exc:
            raise HTTPException(status_code=409, detail="Playground generation was cancelled.") from exc
        except (ValueError, GenerationSafetyError) as exc:
            raise HTTPException(status_code=409, detail="Playground generation was refused safely.") from exc
        except GeneratorUnavailableError as exc:
            raise HTTPException(status_code=503, detail="The Playground generator is unavailable.") from exc

    @router.get("/evaluation/api/playground/runs/latest")
    @product_api
    def latest_playground_run() -> dict[str, Any]:
        return {"run": playground.latest_run(), "legacy": playground.legacy_generations()}

    @router.get("/evaluation/api/playground/runs/{run_id}")
    @product_api
    def playground_run(run_id: str) -> dict[str, Any]:
        try:
            return playground.reconstruct(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/evaluation/api/playground/runs/{run_id}/cancel")
    @product_api
    def cancel_playground_run(run_id: str) -> dict[str, Any]:
        try:
            return playground.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/evaluation/api/playground/presets")
    @product_api
    def list_presets() -> dict[str, Any]:
        return {"presets": playground.presets.list()}

    @router.post("/evaluation/api/playground/presets")
    @product_api
    def save_preset(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = GenerationRequest(**dict(payload.get("request") or {}))
            return playground.presets.save(str(payload.get("name") or ""), request)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/evaluation/api/playground/presets/{name}/rerun")
    @product_api
    def rerun_preset(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return playground.rerun(
                name,
                explicit_action=_strict_payload_bool(payload, "explicit_action"),
                confirm_billable=_strict_payload_bool(payload, "confirm_billable"),
                seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except GenerationTimedOutError as exc:
            raise HTTPException(status_code=408, detail="Playground generation reached its fixed deadline.") from exc
        except GenerationCancelledError as exc:
            raise HTTPException(status_code=409, detail="Playground generation was cancelled.") from exc
        except (ValueError, GenerationSafetyError) as exc:
            raise HTTPException(status_code=409, detail="Playground generation was refused safely.") from exc
        except GeneratorUnavailableError as exc:
            raise HTTPException(status_code=503, detail="The Playground generator is unavailable.") from exc

    @router.get("/evaluation/api/technical/checkpoints")
    @product_api
    def technical_checkpoints(
        request: Request,
        acknowledge: str | None = Query(None),
    ) -> dict[str, Any]:
        if acknowledge != "true" or request.query_params.getlist("acknowledge") != ["true"]:
            raise HTTPException(status_code=400, detail="Technical details require explicit acknowledgement.")
        return evaluation.catalog.to_dict(
            include_unavailable=True,
            technical_details=True,
            private_roots=(context.project_root,),
        )

    router.spritelab_evaluation_service = evaluation
    router.spritelab_playground_service = playground
    router.spritelab_exploratory_smoke_workflow = exploratory_smoke_workflow
    router.spritelab_exploratory_smoke_runner = exploratory_smoke_runner
    return router
