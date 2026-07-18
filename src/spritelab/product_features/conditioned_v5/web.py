"""Web surface for the conditioned Dataset-v5 workflow."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from starlette.requests import Request

from spritelab.product_core import ProjectContext, api_error, product_api, strict_json_dumps
from spritelab.product_features.conditioned_v5.service import (
    ConditionedDatasetError,
    ConditionedDatasetService,
)

_PREVIEW_KEYS = frozenset({"dataset_references"})
_BUILD_KEYS = frozenset({"dataset_references", "idempotency_key", "explicit_action"})
_CANCEL_KEYS = frozenset({"explicit_action"})
_EVIDENCE_KEYS = frozenset({"kind", "explicit_action"})
_PUBLISH_KEYS = frozenset(
    {
        "candidate_identity",
        "label_audit_sha256",
        "dataset_validation_sha256",
        "authorization_id",
        "explicit_action",
        "authorize_one_time_freeze",
    }
)
_TRAINING_AUDIT_KEYS = frozenset(
    {
        "candidate_identity",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "expected_config_sha256",
        "smoke_id",
        "operation_nonce",
        "explicit_action",
    }
)
_ACTIVATE_KEYS = frozenset(
    {
        "candidate_identity",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "expected_config_sha256",
        "activation_authorization_id",
        "explicit_action",
        "authorize_dataset_freeze",
        "authorize_training",
    }
)
_PATH_FRAGMENTS = ("path", "directory", "folder", "output", "destination", "url", "uri")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DATASET_REFERENCE_PATTERN = re.compile(r"dataset\.[0-9a-f]{24}")
_JOB_ID_PATTERN = re.compile(r"conditioned-[0-9a-f]{20}")
_SMOKE_ID_PATTERN = re.compile(r"smoke-[0-9a-f]{20}")
_REGISTRATION_ID_PATTERN = re.compile(r"exploratory-[0-9a-f]{24}")
_JOB_STATUSES = frozenset({"RUNNING", "CANCELLING", "CANCELLED", "INTERRUPTED", "NEEDS_REVIEW", "COMPLETE", "FAILED"})
_JOB_STAGES = frozenset(
    {
        "queued",
        "conditioning",
        "cancelling",
        "cancelled",
        "failed",
        "independent_evidence",
        "publishing",
        "publication_failed",
        "publication_state_failed",
        "published",
        "activation_prepared",
        "activated",
    }
)
_AUDIT_VERDICTS = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
_JOB_STATUS_MESSAGES = {
    "RUNNING": "Conditioned job is running.",
    "CANCELLING": "Conditioned job cancellation is pending.",
    "CANCELLED": "Conditioned job was cancelled.",
    "INTERRUPTED": "Conditioned job was interrupted.",
    "NEEDS_REVIEW": "Conditioned candidate requires independent review.",
    "COMPLETE": "Conditioned job is complete.",
    "FAILED": "Conditioned job failed closed.",
}


def _resource_text(name: str) -> str:
    return files("spritelab.product_features.conditioned_v5").joinpath(name).read_text(encoding="utf-8")


def create_router(
    context: ProjectContext,
    *,
    service: ConditionedDatasetService | None = None,
) -> Any:
    """Create the local, CSRF-protected JSON and page routes."""

    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    router = APIRouter()
    conditioned = service or ConditionedDatasetService(context.project_root)

    @router.get("/dataset-v5", response_class=HTMLResponse)
    def page(request: Request) -> Any:
        initial = _public_inventory(conditioned.inventory())
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(
                request,
                "dataset.conditioned_v5",
                "conditioned_v5.html",
                {"conditioned_initial_state": initial},
            )
        standalone = _resource_text("templates/conditioned_v5_standalone.html")
        return standalone.replace(
            "__CONDITIONED_INITIAL_STATE__",
            strict_json_dumps(initial, ensure_ascii=False).replace("<", "\\u003c"),
        )

    @router.get("/dataset-v5/static/conditioned-v5.css")
    def css() -> Response:
        return Response(_resource_text("static/conditioned-v5.css"), media_type="text/css")

    @router.get("/dataset-v5/static/conditioned-v5.js")
    def javascript() -> Response:
        return Response(_resource_text("static/conditioned-v5.js"), media_type="application/javascript")

    @router.get("/dataset-v5/api/inventory")
    @product_api
    def inventory() -> dict[str, Any]:
        return _public_inventory(conditioned.inventory())

    @router.post("/dataset-v5/api/preview")
    @product_api
    def preview(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PREVIEW_KEYS)
        if rejected is not None:
            return rejected
        return _call(lambda: _public_preview(conditioned.preview(_dataset_references(payload))))

    @router.post("/dataset-v5/api/jobs")
    @product_api
    def start(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _BUILD_KEYS)
        if rejected is not None:
            return rejected
        try:
            job, created = conditioned.start_build(
                _dataset_references(payload),
                idempotency_key=_required_string(payload, "idempotency_key"),
                explicit_action=payload.get("explicit_action") is True,
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()
        return JSONResponse(
            status_code=202 if created else 200,
            content={"created": created is True, "job": _public_job(job)},
        )

    @router.get("/dataset-v5/api/jobs/{job_id}")
    @product_api
    def job(job_id: str) -> Any:
        return _call(lambda: _public_job(conditioned.job(job_id)))

    @router.get("/dataset-v5/api/jobs/{job_id}/training-audit-options")
    @product_api
    def training_audit_options(job_id: str) -> Any:
        return _call(lambda: _training_audit_options(context, conditioned, job_id))

    @router.post("/dataset-v5/api/jobs/{job_id}/cancel")
    @product_api
    def cancel(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _CANCEL_KEYS)
        if rejected is not None:
            return rejected
        return _call(
            lambda: _public_job(conditioned.cancel(job_id, explicit_action=payload.get("explicit_action") is True))
        )

    @router.post("/dataset-v5/api/jobs/{job_id}/evidence")
    @product_api
    def evidence(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _EVIDENCE_KEYS)
        if rejected is not None:
            return rejected
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind.strip() or kind != kind.strip():
            return _invalid()
        return _call(
            lambda: _public_job(
                conditioned.run_independent_audit(
                    job_id,
                    kind=kind,
                    explicit_action=payload.get("explicit_action") is True,
                )
            )
        )

    @router.post("/dataset-v5/api/jobs/{job_id}/publish")
    @product_api
    def publish(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PUBLISH_KEYS)
        if rejected is not None:
            return rejected
        try:
            return _public_job(
                conditioned.publish(
                    job_id,
                    candidate_identity=_required_string(payload, "candidate_identity"),
                    label_audit_sha256=_required_string(payload, "label_audit_sha256"),
                    dataset_validation_sha256=_required_string(payload, "dataset_validation_sha256"),
                    authorization_id=_required_string(payload, "authorization_id"),
                    explicit_action=payload.get("explicit_action") is True,
                    authorize_one_time_freeze=payload.get("authorize_one_time_freeze") is True,
                )
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()

    @router.post("/dataset-v5/api/jobs/{job_id}/activate")
    @product_api
    def activate(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _ACTIVATE_KEYS)
        if rejected is not None:
            return rejected
        try:
            return _public_job(
                conditioned.activate(
                    job_id,
                    candidate_identity=_required_string(payload, "candidate_identity"),
                    publication_identity_sha256=_required_string(payload, "publication_identity_sha256"),
                    activation_manifest_sha256=_required_string(payload, "activation_manifest_sha256"),
                    campaign_config_sha256=_required_string(payload, "campaign_config_sha256"),
                    campaign_identity_sha256=_required_string(payload, "campaign_identity_sha256"),
                    expected_config_sha256=_required_string(payload, "expected_config_sha256"),
                    activation_authorization_id=_required_string(payload, "activation_authorization_id"),
                    explicit_action=payload.get("explicit_action") is True,
                    authorize_dataset_freeze=payload.get("authorize_dataset_freeze") is True,
                    authorize_training=payload.get("authorize_training") is True,
                )
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()

    @router.post("/dataset-v5/api/jobs/{job_id}/training-audit")
    @product_api
    def training_audit(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _TRAINING_AUDIT_KEYS)
        if rejected is not None:
            return rejected
        try:
            return _public_training_audit_result(
                job_id,
                conditioned.run_training_infrastructure_audit(
                    job_id,
                    candidate_identity=_required_string(payload, "candidate_identity"),
                    publication_identity_sha256=_required_string(payload, "publication_identity_sha256"),
                    activation_manifest_sha256=_required_string(payload, "activation_manifest_sha256"),
                    campaign_config_sha256=_required_string(payload, "campaign_config_sha256"),
                    campaign_identity_sha256=_required_string(payload, "campaign_identity_sha256"),
                    expected_config_sha256=_required_string(payload, "expected_config_sha256"),
                    smoke_id=_required_string(payload, "smoke_id"),
                    operation_nonce=_required_string(payload, "operation_nonce"),
                    explicit_action=payload.get("explicit_action") is True,
                ),
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()

    return router


def _call(function: Any) -> Any:
    try:
        return function()
    except ConditionedDatasetError as exc:
        return _error(exc)


def _public_inventory(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    managed_intakes: list[dict[str, Any]] = []
    raw_intakes = source.get("managed_intakes")
    if isinstance(raw_intakes, list):
        for raw in raw_intakes[:1_000]:
            if not isinstance(raw, Mapping):
                continue
            dataset_reference = _opaque_id(raw.get("dataset_reference"), _DATASET_REFERENCE_PATTERN)
            if dataset_reference is None or raw.get("status") != "COMPLETE":
                continue
            managed_intakes.append(
                {
                    "dataset_reference": dataset_reference,
                    "accepted_count": _public_count(raw.get("accepted_count")),
                    "quarantined_count": _public_count(raw.get("quarantined_count")),
                    "status": "COMPLETE",
                    "paths_exposed": False,
                }
            )
    jobs: list[dict[str, Any]] = []
    raw_jobs = source.get("jobs")
    if isinstance(raw_jobs, list):
        jobs = [_public_job_summary(raw) for raw in raw_jobs[:200] if isinstance(raw, Mapping)]
    raw_policy = source.get("count_policy")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    return {
        "schema_version": "spritelab.dataset.conditioned-inventory-public.v1",
        "managed_intakes": managed_intakes,
        "jobs": jobs,
        "count_policy": {
            "minimum": _public_count(policy.get("minimum")),
            "target": _public_count(policy.get("target")),
            "maximum": _public_count(policy.get("maximum")),
        },
        "config_sha256": _public_sha256(source.get("config_sha256")),
        "network_actions": 0,
        "paths_exposed": False,
    }


def _public_preview(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    ready = source.get("ready_to_build") is True
    references = source.get("dataset_references")
    public_references = (
        [
            reference
            for reference in references[:8]
            if isinstance(reference, str) and _DATASET_REFERENCE_PATTERN.fullmatch(reference)
        ]
        if isinstance(references, list)
        else []
    )
    return {
        "schema_version": "spritelab.dataset.conditioned-preview-public.v1",
        "dataset_references": public_references,
        "eligible_unique_images": _public_count(source.get("eligible_unique_images")),
        "selected_images": _public_count(source.get("selected_images")),
        "ready_to_build": ready,
        "blockers": []
        if ready
        else ["A conditioned Dataset-v5 candidate requires 2,000-3,000 unique eligible images."],
        "labels_are_human_truth": False,
        "paths_exposed": False,
    }


def _public_job_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    status_value = value.get("status")
    status = status_value if isinstance(status_value, str) and status_value in _JOB_STATUSES else "FAILED"
    stage_value = value.get("stage")
    stage = stage_value if isinstance(stage_value, str) and stage_value in _JOB_STAGES else "unavailable"
    return {
        "job_id": _opaque_id(value.get("job_id"), _JOB_ID_PATTERN) or "",
        "status": status,
        "stage": stage,
        "current": _public_count(value.get("current")),
        "total": _public_count(value.get("total")),
        "message": _JOB_STATUS_MESSAGES[status],
        "paths_exposed": False,
    }


def _public_job(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    summary = _public_job_summary(source)
    return {
        "schema_version": "spritelab.dataset.conditioned-job-public.v1",
        **summary,
        "candidate": _public_candidate(source.get("candidate")),
        "evidence": _public_evidence(source.get("evidence")),
        "publication": _public_publication(source.get("publication")),
        "activated_config_sha256": _public_activated_config_sha256(source.get("activation_authorization")),
        "events": [],
        "paths_exposed": False,
    }


def _public_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    candidate_identity = _public_sha256(value.get("candidate_identity"))
    if candidate_identity is None:
        return None
    return {
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": _public_sha256(value.get("payload_inventory_sha256")),
        "image_count": _public_count(value.get("image_count")),
        "paths_exposed": False,
    }


def _public_evidence(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    evidence: dict[str, Any] = {}
    for kind in ("label_audit", "dataset_validation"):
        raw = source.get(kind)
        if not isinstance(raw, Mapping):
            continue
        report_sha256 = _public_sha256(raw.get("sha256"))
        receipt = raw.get("receipt") if isinstance(raw.get("receipt"), Mapping) else {}
        action = raw.get("action") if isinstance(raw.get("action"), Mapping) else {}
        receipt_identity = _public_sha256(receipt.get("receipt_identity"))
        action_identity = _public_sha256(action.get("record_identity"))
        if report_sha256 is None or receipt_identity is None or action_identity is None:
            continue
        evidence[kind] = {
            "sha256": report_sha256,
            "byte_count": _public_count(raw.get("byte_count")),
            "audit_run_identity": _public_sha256(raw.get("audit_run_identity")),
            "receipt_identity": receipt_identity,
            "action_record_identity": action_identity,
            "verdict": "PASS",
            "server_managed": True,
            "paths_exposed": False,
        }
    return evidence


def _public_publication(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    identities = {
        key: _public_sha256(value.get(key))
        for key in (
            "publication_identity_sha256",
            "activation_manifest_sha256",
            "campaign_config_sha256",
            "campaign_identity_sha256",
        )
    }
    if any(identity is None for identity in identities.values()):
        return None
    raw_seeds = value.get("campaign_seeds")
    seeds = (
        [seed for seed in raw_seeds[:16] if type(seed) is int and 0 <= seed <= 2**63 - 1]
        if isinstance(raw_seeds, list)
        else []
    )
    return {
        **identities,
        "campaign_launch_ready": value.get("campaign_launch_ready") is True,
        "campaign_seeds": seeds,
        "campaign_steps": _public_count(value.get("campaign_steps")),
        "configuration_activated": _exact_bool(value.get("configuration_activated"), default=True),
        "training_started": _exact_bool(value.get("training_started"), default=True),
        "paths_exposed": False,
    }


def _public_activated_config_sha256(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    config_after_sha256 = _public_sha256(value.get("config_after_sha256"))
    if value.get("status") != "COMMITTED" or config_after_sha256 is None:
        return None
    return config_after_sha256


def _public_training_audit_result(job_id: str, value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    verdict_value = source.get("verdict")
    verdict = verdict_value if isinstance(verdict_value, str) and verdict_value in _AUDIT_VERDICTS else "INCONCLUSIVE"
    return {
        "schema_version": "spritelab.training.infrastructure-audit-action-public.v1",
        "job_id": _opaque_id(job_id, _JOB_ID_PATTERN) or "",
        "smoke_id": _opaque_id(source.get("smoke_id"), _SMOKE_ID_PATTERN),
        "operation_identity": _public_sha256(source.get("operation_identity")),
        "prospective_configuration_identity_sha256": _public_sha256(
            source.get("prospective_configuration_identity_sha256")
        ),
        "base_config_sha256": _public_sha256(source.get("base_config_sha256")),
        "action_record_identity": _public_sha256(source.get("action_record_identity")),
        "verdict": verdict,
        "config_unchanged": source.get("config_unchanged") is True,
        "configuration_activated": _exact_bool(source.get("configuration_activated"), default=True),
        "training_started": _exact_bool(source.get("training_started"), default=True),
        "paths_exposed": False,
    }


def _opaque_id(value: Any, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _public_sha256(value: Any) -> str | None:
    return _opaque_id(value, _SHA256_PATTERN)


def _public_count(value: Any) -> int:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else 0


def _exact_bool(value: Any, *, default: bool) -> bool:
    return value if type(value) is bool else default


def _training_audit_options(
    context: ProjectContext,
    conditioned: ConditionedDatasetService,
    job_id: str,
) -> dict[str, Any]:
    job = conditioned.job(job_id)
    candidate = job.get("candidate")
    publication = job.get("publication")
    public_job_id = _opaque_id(job_id, _JOB_ID_PATTERN) or ""
    base = {
        "schema_version": "spritelab.training.conditioned-audit-options.v1",
        "job_id": public_job_id,
        "eligible": [],
        "count": 0,
        "ready": False,
        "paths_exposed": False,
    }
    if (
        job.get("status") != "COMPLETE"
        or not isinstance(candidate, Mapping)
        or not isinstance(publication, Mapping)
        or publication.get("configuration_activated") is not False
    ):
        return base

    # Import lazily so passive Dataset-v5 page/status rendering does not load
    # the Training or optional Torch runtime.
    from spritelab.product_features.evaluation.exploratory_smoke import ExploratorySmokeWorkflow

    workflow = ExploratorySmokeWorkflow(
        context.project_root,
        job_loader=conditioned.job,
    )
    prepared = workflow.prepared_plans()
    raw_prepared = prepared.get("eligible")
    job_smoke_ids: set[str] = set()
    if isinstance(raw_prepared, list):
        for item in raw_prepared[:200]:
            if not isinstance(item, Mapping) or item.get("conditioned_job_id") != job_id:
                continue
            smoke_id = _opaque_id(item.get("smoke_id"), _SMOKE_ID_PATTERN)
            if smoke_id is not None:
                job_smoke_ids.add(smoke_id)
    catalog = workflow.catalog()
    selected: dict[str, dict[str, str]] = {}
    for item in catalog.eligible[:200]:
        smoke_id = _opaque_id(getattr(item, "smoke_id", None), _SMOKE_ID_PATTERN)
        registration_id = _opaque_id(getattr(item, "registration_id", None), _REGISTRATION_ID_PATTERN)
        if (
            getattr(item, "weights", None) != "ema"
            or smoke_id is None
            or registration_id is None
            or smoke_id not in job_smoke_ids
            or getattr(item, "freeze_identity", None) != publication.get("activation_manifest_sha256")
            or getattr(item, "campaign_identity", None) != publication.get("campaign_identity_sha256")
        ):
            continue
        selected[smoke_id] = {
            "smoke_id": smoke_id,
            "registration_id": registration_id,
            "status": "PROVISIONALLY_VERIFIED",
            "purpose": "exploratory",
        }
    eligible = [selected[smoke_id] for smoke_id in sorted(selected, reverse=True)]
    return {**base, "eligible": eligible, "count": len(eligible), "ready": bool(eligible)}


def _error(exc: ConditionedDatasetError) -> Any:
    return api_error(
        exc.status_code,
        exc.code,
        exc.public_message,
        recoverable=exc.status_code < 500,
        next_action="Review the managed handoffs, candidate evidence, and exact authorization before retrying.",
    )


def _validate_payload(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    reject_path_keys: bool = True,
) -> Any | None:
    if set(payload) - allowed:
        return _invalid()
    if reject_path_keys and any(fragment in str(key).casefold() for key in payload for fragment in _PATH_FRAGMENTS):
        return api_error(
            422,
            "browser_path_not_allowed",
            "Dataset-v5 uses managed repository-local artifacts; browser paths and URLs are not accepted.",
        )
    return None


def _dataset_references(payload: dict[str, Any]) -> list[str]:
    value = payload.get("dataset_references")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ConditionedDatasetError(
            "managed_intake_selection", "Select completed managed Dataset imports.", status_code=422
        )
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{key} is required")
    return value


def _invalid() -> Any:
    return api_error(
        422,
        "invalid_conditioned_v5_payload",
        "The Dataset-v5 request fields are missing or not recognized.",
    )


__all__ = ["create_router"]
