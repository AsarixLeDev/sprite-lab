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
)
from spritelab.product_features.evaluation.playground import (
    GenerationCancelledError,
    GenerationRequest,
    GenerationSafetyError,
    GeneratorUnavailableError,
    PlaygroundGenerator,
    PlaygroundService,
)
from spritelab.product_features.evaluation.service import EvaluationRequest, EvaluationService


def _resource_text(name: str) -> str:
    return files("spritelab.product_features.evaluation").joinpath(name).read_text(encoding="utf-8")


def create_evaluation_router(
    context: ProjectContext,
    *,
    service: EvaluationService | None = None,
    playground_generator: PlaygroundGenerator | None = None,
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
    playground = PlaygroundService(
        evaluation.catalog,
        output_root=evaluation.output_root / "playground",
        generator=playground_generator,
        runs_directory=evaluation.runs_directory,
    )

    @router.get("/evaluation", response_class=HTMLResponse)
    def evaluation_page(request: Request) -> Any:
        initial = {
            "checkpoints": evaluation.catalog.to_dict(),
            "playground_defaults": playground.defaults(),
            "playground_run": playground.latest_run(),
            "promotion": evaluation.plan(EvaluationRequest(dry_run=True))[2][-1].to_dict(),
            "durable_run": {
                "run_id": evaluation.latest_run_id,
                "status": evaluation.latest_status,
                "message": evaluation.latest_message,
                "stages": [stage.to_dict() for stage in evaluation.latest_stages],
                "dashboard": evaluation.dashboard(),
            },
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

    @router.get("/evaluation/static/evaluation.js")
    def evaluation_js() -> Response:
        return Response(_resource_text("static/evaluation.js"), media_type="application/javascript")

    @router.get("/evaluation/api/checkpoints")
    @product_api
    def checkpoints(
        include_unavailable: bool = False,
        technical_details: bool = False,
    ) -> dict[str, Any]:
        return evaluation.catalog.to_dict(
            include_unavailable=include_unavailable,
            technical_details=technical_details,
        )

    @router.get("/evaluation/api/plan")
    @product_api
    def plan(checkpoint_id: str | None = None, weights: str = "ema") -> dict[str, Any]:
        _checkpoint, _benchmark, stages = evaluation.plan(
            EvaluationRequest(checkpoint_id=checkpoint_id, weights=weights, dry_run=True)
        )
        return {"stages": [stage.to_dict() for stage in stages]}

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
        request = EvaluationRequest(
            checkpoint_id=payload.get("checkpoint_id"),
            weights=str(payload.get("weights") or "ema"),
            dry_run=bool(payload.get("dry_run", False)),
            explicit_action=bool(payload.get("explicit_action", False)),
            confirm_billable=bool(payload.get("confirm_billable", False)),
            allow_source_results=bool(payload.get("allow_source_results", False)),
        )
        try:
            result = evaluation.run(request)
        except GenerationSafetyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        return JSONResponse(result.to_dict())

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
        raw = build_dashboard(evaluation.latest_report or {}, evaluation.latest_rows)["gallery"]
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
        return JSONResponse(
            {
                "report": evaluation.latest_report,
                "per_image_metrics": evaluation.latest_rows,
                "stages": [stage.to_dict() for stage in evaluation.latest_stages],
                "run_id": evaluation.latest_run_id,
                "status": evaluation.latest_status,
                "message": evaluation.latest_message,
                "promotion_actions": 0,
            },
            headers={"Content-Disposition": 'attachment; filename="sprite-lab-evaluation-report.json"'},
        )

    @router.get("/evaluation/api/playground/defaults")
    @product_api
    def playground_defaults() -> dict[str, Any]:
        return {
            "defaults": playground.defaults(),
            "scope": "EXPLORATORY",
            "billable_confirmation_required": playground.confirmation_required,
        }

    @router.post("/evaluation/api/playground/generate")
    @product_api
    def playground_generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = GenerationRequest(
                prompt=str(payload.get("prompt") or ""),
                checkpoint_id=str(payload.get("checkpoint_id") or playground.catalog.default_checkpoint_id or ""),
                weights=str(payload.get("weights") or "ema"),
                seed=int(payload.get("seed", 42)),
                sampling_steps=int(payload.get("sampling_steps", 30)),
                guidance=float(payload.get("guidance", 3.0)),
                image_count=int(payload.get("image_count", 4)),
            )
            return playground.generate(
                request,
                explicit_action=bool(payload.get("explicit_action", False)),
                confirm_billable=bool(payload.get("confirm_billable", False)),
            )
        except (ValueError, GenerationSafetyError, GenerationCancelledError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except GeneratorUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
                explicit_action=bool(payload.get("explicit_action", False)),
                confirm_billable=bool(payload.get("confirm_billable", False)),
                seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, GenerationSafetyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except GeneratorUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/evaluation/api/technical/checkpoints")
    @product_api
    def technical_checkpoints(acknowledge: bool = Query(False)) -> dict[str, Any]:
        if not acknowledge:
            raise HTTPException(status_code=400, detail="Technical details require explicit acknowledgement.")
        return evaluation.catalog.to_dict(include_unavailable=True, technical_details=True)

    router.spritelab_evaluation_service = evaluation
    router.spritelab_playground_service = playground
    return router
