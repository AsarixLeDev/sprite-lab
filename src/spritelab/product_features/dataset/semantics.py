"""Optional semantic proposals through the shared VisionProvider contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spritelab.product_core import (
    ProductAction,
    ProductResult,
    ProductStatus,
    ProjectContext,
    VisionProvider,
)
from spritelab.product_core.audit_evidence import (
    CONDITIONED_VIEW_CANDIDATES,
    CONSERVATIVE_PROPOSAL_GENERATION,
)
from spritelab.product_features.dataset.certification import (
    authorize_labeling_scope,
    labeling_audit_verification,
)

SEMANTIC_PROPOSAL_SCHEMA = "spritelab.dataset.semantic_proposal.v1"
SEMANTIC_CONFIDENCE_THRESHOLD = 0.8


def propose_semantics(
    items: list[dict[str, Any]],
    provider: VisionProvider | None,
    context: ProjectContext,
) -> dict[str, Any]:
    """Obtain conservative, explicitly non-human proposals for accepted items."""

    eligible = [item for item in items if item["current_disposition"] == "accepted"]
    supplied = _apply_human_labels(eligible)
    pending = [item for item in eligible if not item.get("semantic")]
    verification = labeling_audit_verification(context)
    authorized_scopes = verification.authorized_scopes
    if provider is not None and not pending:
        summary = _summary(
            eligible,
            provider_status="configured_not_needed",
            health_ok=None,
            authorized_scopes=authorized_scopes,
        )
        summary["provider_id"] = provider.provider_id
        summary["human_supplied"] = supplied
        return summary
    if provider is None:
        for item in pending:
            item["semantic"] = {"state": "pending", "truth_status": "unavailable", "provider_id": None}
        return _summary(
            eligible,
            provider_status="not_configured",
            health_ok=None,
            authorized_scopes=authorized_scopes,
        )
    authorization = authorize_labeling_scope(context, CONSERVATIVE_PROPOSAL_GENERATION)
    if not authorization.authorized:
        return _certification_blocked(
            pending,
            eligible,
            provider.provider_id,
            authorization.reason,
            authorized_scopes=authorized_scopes,
        )
    try:
        capabilities = tuple(provider.probe(context))
    except Exception as exc:  # provider failures are data, not intake failures
        return _health_failure(
            pending,
            eligible,
            provider,
            f"Provider health probe failed: {exc}",
            authorized_scopes=authorized_scopes,
        )
    if not capabilities or not any(capability.available for capability in capabilities):
        return _health_failure(
            pending,
            eligible,
            provider,
            "Vision provider health gate is not ready.",
            authorized_scopes=authorized_scopes,
        )
    action = ProductAction(
        action_id="dataset.semantic.propose",
        feature="dataset",
        title="Propose conservative semantic labels",
        parameters={
            "schema_version": SEMANTIC_PROPOSAL_SCHEMA,
            "items": [
                {
                    "item_id": item["item_id"],
                    "image_path": item["source_path"],
                    "decoded_rgba_sha256": item.get("decoded_rgba_sha256"),
                }
                for item in pending
            ],
            "preserve_abstentions": True,
            "proposals_are_human_truth": False,
        },
    )
    try:
        result = provider.execute(action, context, lambda _event: None)
    except Exception as exc:  # provider failures do not discard the image-only dataset
        return _health_failure(
            pending,
            eligible,
            provider,
            f"Vision provider execution failed: {exc}",
            authorized_scopes=authorized_scopes,
        )
    if not isinstance(result, ProductResult) or result.status not in {
        ProductStatus.READY,
        ProductStatus.COMPLETE,
    }:
        return _health_failure(
            pending,
            eligible,
            provider,
            "Vision provider returned an unhealthy result.",
            authorized_scopes=authorized_scopes,
        )
    proposals = result.data.get("proposals", ())
    by_id = {
        str(value.get("item_id")): value for value in proposals if isinstance(value, Mapping) and value.get("item_id")
    }
    for item in pending:
        raw = by_id.get(item["item_id"])
        item["semantic"] = _validated_proposal(raw, provider.provider_id)
    summary = _summary(
        eligible,
        provider_status="available",
        health_ok=True,
        authorized_scopes=authorized_scopes,
    )
    summary["human_supplied"] = supplied
    summary["provider_id"] = provider.provider_id
    return summary


def _apply_human_labels(items: Sequence[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        labels = item.pop("_human_labels", None)
        if labels:
            item["semantic"] = {
                "state": "labeled",
                "truth_status": "human_supplied",
                "labels": labels,
                "needs_review": False,
            }
            count += 1
    return count


def _validated_proposal(value: Mapping[str, Any] | None, provider_id: str) -> dict[str, Any]:
    if value is None or bool(value.get("abstained", value.get("abstain", False))):
        return {
            "state": "abstained",
            "truth_status": "provider_proposal_not_human_truth",
            "provider_id": provider_id,
            "needs_review": True,
            "reason": str(value.get("reason", "provider_abstained")) if value else "provider_omitted_item",
        }
    labels = value.get("labels")
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    valid = isinstance(labels, Mapping) and bool(labels) and 0.0 <= confidence <= 1.0
    conflicts = list(value.get("conflicts", ())) if isinstance(value.get("conflicts", ()), Sequence) else []
    health_ok = bool(value.get("health_ok", True))
    needs_review = not valid or confidence < SEMANTIC_CONFIDENCE_THRESHOLD or bool(conflicts) or not health_ok
    return {
        "state": "proposed" if valid else "abstained",
        "truth_status": "provider_proposal_not_human_truth",
        "provider_id": provider_id,
        "labels": dict(labels) if isinstance(labels, Mapping) else {},
        "confidence": confidence,
        "conflicts": conflicts,
        "health_ok": health_ok,
        "needs_review": needs_review,
    }


def _health_failure(
    pending: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    provider: VisionProvider,
    message: str,
    *,
    authorized_scopes: Sequence[str],
) -> dict[str, Any]:
    for item in pending:
        item["semantic"] = {
            "state": "pending",
            "truth_status": "unavailable",
            "provider_id": provider.provider_id,
            "needs_review": True,
            "health_failure": message,
        }
    summary = _summary(
        eligible,
        provider_status="health_failed",
        health_ok=False,
        authorized_scopes=authorized_scopes,
    )
    summary["provider_id"] = provider.provider_id
    summary["health_failure"] = message
    return summary


def _certification_blocked(
    pending: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    provider_id: str,
    reason: str,
    *,
    authorized_scopes: Sequence[str],
) -> dict[str, Any]:
    for item in pending:
        item["semantic"] = {
            "state": "pending",
            "truth_status": "unavailable",
            "provider_id": provider_id,
            "needs_review": False,
            "certification_blocker": reason,
        }
    summary = _summary(
        eligible,
        provider_status="certification_blocked",
        health_ok=None,
        authorized_scopes=authorized_scopes,
    )
    summary["provider_id"] = provider_id
    summary["certification_blocker"] = reason
    return summary


def _summary(
    eligible: Sequence[dict[str, Any]],
    *,
    provider_status: str,
    health_ok: bool | None,
    authorized_scopes: Sequence[str],
) -> dict[str, Any]:
    labeled = sum(item.get("semantic", {}).get("state") in {"labeled", "proposed"} for item in eligible)
    abstained = sum(item.get("semantic", {}).get("state") in {"abstained", "pending"} for item in eligible)
    exception_count = sum(bool(item.get("semantic", {}).get("needs_review")) for item in eligible)
    scope_set = set(authorized_scopes)
    conditioned_ready = (
        bool(eligible)
        and labeled == len(eligible)
        and exception_count == 0
        and CONDITIONED_VIEW_CANDIDATES in scope_set
    )
    return {
        "schema_version": SEMANTIC_PROPOSAL_SCHEMA,
        "provider_status": provider_status,
        "health_ok": health_ok,
        "semantically_labeled": labeled,
        "semantically_abstained": abstained,
        "semantic_review_exceptions": exception_count,
        "conditioned_dataset_ready": conditioned_ready,
        "conditioned_view_authorized": CONDITIONED_VIEW_CANDIDATES in scope_set,
        "authorized_scopes": sorted(scope_set),
        "proposals_are_human_truth": False,
    }
