"""Append-only exception review decisions and safe dataset updates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.dataset_v5.raw_inventory import file_sha256
from spritelab.product_core.audit_evidence import CONDITIONED_VIEW_CANDIDATES
from spritelab.product_core.contracts import ProjectContext
from spritelab.product_features.dataset.certification import authorize_labeling_scope
from spritelab.product_features.dataset.intake import (
    rebuild_raw_extraction,
    recompute_summary_from_items,
    terminal_message,
)


class ReviewDecisionError(ValueError):
    """A requested review decision would violate an intake safety gate."""


class DatasetReviewStore:
    """Mutable current decisions backed by an immutable append-only audit log."""

    def __init__(self, output_root: str | Path, *, context: ProjectContext | None = None) -> None:
        self.output_root = Path(output_root).resolve()
        self.context = context
        self.queue_path = self.output_root / "review_queue.json"
        self.items_path = self.output_root / "items.jsonl"
        self.result_path = self.output_root / "result.json"
        self.report_path = self.output_root / "report_data.json"
        self.log_path = self.output_root / "review_log.jsonl"
        if not all(path.is_file() for path in (self.queue_path, self.items_path, self.result_path, self.report_path)):
            raise FileNotFoundError(f"No completed dataset intake was found at {self.output_root}")

    def queue(self) -> dict[str, Any]:
        return _read_json(self.queue_path)

    def apply(self, item_id: str, decision: str) -> dict[str, Any]:
        normalized = decision.strip().casefold()
        if normalized not in {"keep", "exclude"}:
            raise ReviewDecisionError("Review decision must be 'keep' or 'exclude'.")
        queue = self.queue()
        queue_item = next((item for item in queue["items"] if item.get("item_id") == item_id), None)
        if queue_item is None:
            raise ReviewDecisionError(f"Unknown review item: {item_id}")
        if queue_item.get("queue_kind") != "intake_exception":
            raise ReviewDecisionError(
                "Semantic exceptions require semantic confirmation, not an intake keep/exclude override."
            )
        if normalized == "keep" and not queue_item.get("review_rescuable", False):
            legal = {
                "conflicting_license_evidence",
                "conflicting_source_evidence",
                "missing_source",
                "missing_license",
                "unverified_license",
            } & set(queue_item.get("reasons", ()))
            if legal:
                raise ReviewDecisionError(
                    "Keep cannot override missing or unverified source/license evidence. Add the evidence and rebuild."
                )
            raise ReviewDecisionError("This item cannot be made technically usable by a review decision.")
        if normalized == "keep":
            source_path = Path(str(queue_item["source_path"]))
            extraction = queue_item.get("sheet_extraction")
            expected_hash = (
                extraction.get("source_byte_sha256")
                if isinstance(extraction, Mapping)
                else queue_item.get("byte_sha256")
            )
            if not source_path.is_file() or file_sha256(source_path) != expected_hash:
                raise ReviewDecisionError(
                    "The source image changed after preprocessing. Rebuild the dataset before review."
                )
        before = {
            "current_decision": queue_item.get("current_decision"),
            "current_disposition": queue_item.get("current_disposition"),
        }
        queue_item["current_decision"] = normalized
        queue_item["current_disposition"] = (
            "accepted" if normalized == "keep" else str(queue_item["automatic_disposition"])
        )
        queue_item["human_override"] = "rescued" if normalized == "keep" else "confirmed_excluded"
        queue_item["review_confirmed"] = True
        queue_item["default_visible"] = normalized == "exclude" and queue_item["current_disposition"] in {
            "rejected",
            "uncertain",
            "requires_special_extraction",
        }
        items = _read_jsonl(self.items_path)
        item = next((value for value in items if value.get("item_id") == item_id), None)
        if item is None:
            raise ReviewDecisionError(f"Review item is not present in the current dataset state: {item_id}")
        for key in ("current_decision", "current_disposition", "human_override", "review_confirmed"):
            item[key] = queue_item[key]
        _write_json(self.queue_path, queue)
        _write_jsonl(self.items_path, items)
        self._append_log(
            {
                "event": "item_decision",
                "item_id": item_id,
                "before": before,
                "after": {
                    "current_decision": queue_item["current_decision"],
                    "current_disposition": queue_item["current_disposition"],
                },
                "automatic_disposition": queue_item["automatic_disposition"],
                "reasons": list(queue_item.get("reasons", ())),
            }
        )
        self._refresh_result(items)
        rebuild_raw_extraction(self.output_root, items)
        return {
            "item_id": item_id,
            "decision": normalized,
            "current_disposition": queue_item["current_disposition"],
            "review_log": str(self.log_path),
        }

    def confirm_all_current_exclusions(self, *, reason: str | None = None) -> dict[str, Any]:
        queue = self.queue()
        selected = [
            item
            for item in queue["items"]
            if item.get("queue_kind") == "intake_exception"
            and item.get("current_decision") == "exclude"
            and (reason is None or reason in item.get("reasons", ()))
        ]
        ids = {str(item["item_id"]) for item in selected}
        for item in selected:
            item["review_confirmed"] = True
            item["human_override"] = "confirmed_excluded"
        items = _read_jsonl(self.items_path)
        for item in items:
            if item.get("item_id") in ids:
                item["review_confirmed"] = True
                item["human_override"] = "confirmed_excluded"
        _write_json(self.queue_path, queue)
        _write_jsonl(self.items_path, items)
        self._append_log(
            {
                "event": "confirm_all_current_exclusions",
                "item_ids": sorted(ids),
                "count": len(ids),
                "reason_filter": reason,
            }
        )
        return {"confirmed": len(ids), "item_ids": sorted(ids), "review_log": str(self.log_path)}

    def _refresh_result(self, items: list[dict[str, Any]]) -> None:
        result = _read_json(self.result_path)
        report = _read_json(self.report_path)
        previous_summary = dict(report.get("summary", {}))
        summary = recompute_summary_from_items(items, previous_summary)
        conditioned_authorized = bool(
            self.context and authorize_labeling_scope(self.context, CONDITIONED_VIEW_CANDIDATES).authorized
        )
        if not conditioned_authorized:
            summary["semantic"]["conditioned_dataset_ready"] = False
            summary["semantic"]["conditioned_view_authorized"] = False
            summary["conditioned_dataset"] = {
                "status": "NOT_READY",
                "reason": "current_labeling_scope_not_authorized",
            }
        counts = dict(summary["counts"])
        result["status"] = "COMPLETE" if counts["accepted"] else "NEEDS_REVIEW"
        result["message"] = terminal_message(Path(str(result["data"]["input_root"])), summary)
        result["data"]["counts"] = counts
        result["data"]["dispositions"] = dict(summary["disposition_counts"])
        for capability in result.get("capabilities", ()):
            if capability.get("capability_id") == "dataset.image_only":
                capability["status"] = "READY" if counts["accepted"] else "BLOCKED"
                capability["details"] = {"eligible": counts["accepted"]}
            elif capability.get("capability_id") == "dataset.review":
                capability["status"] = "NEEDS_REVIEW" if counts["excluded"] else "COMPLETE"
                capability["details"] = {"items": counts["excluded"]}
            elif capability.get("capability_id") == "dataset.conditioned":
                capability["status"] = "READY" if conditioned_authorized else "UNAVAILABLE"
                capability["details"] = dict(summary["conditioned_dataset"])
        report["summary"] = summary
        report["status_cards"] = _status_cards(summary)
        result["data"]["status_cards"] = report["status_cards"]
        _write_json(self.result_path, result)
        _write_json(self.report_path, report)
        _write_json(self.output_root / "status_cards.json", {"cards": report["status_cards"]})

    def _append_log(self, event: Mapping[str, Any]) -> None:
        record = {
            "schema_version": "spritelab.dataset.review_event.v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **dict(event),
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _status_cards(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = summary["counts"]
    return [
        {"id": "processed", "title": "Processed", "value": counts["processed"], "status": "COMPLETE"},
        {"id": "accepted", "title": "Accepted", "value": counts["accepted"], "status": "READY"},
        {
            "id": "excluded",
            "title": "Excluded",
            "value": counts["excluded"],
            "status": "NEEDS_REVIEW" if counts["excluded"] else "COMPLETE",
        },
        {
            "id": "image-only",
            "title": "Image-only",
            "value": counts["image_only_eligible"],
            "status": summary["image_only_dataset"]["status"],
        },
        {
            "id": "conditioned",
            "title": "Conditioned",
            "value": counts["semantically_labeled"],
            "status": summary["conditioned_dataset"]["status"],
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for value in values
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
