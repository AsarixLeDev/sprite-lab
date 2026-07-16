"""Resumable human review for generation-benchmark training matches."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from spritelab.evaluation.candidate_bundle import (
    CANDIDATE_SCHEMA_VERSION,
    candidate_bundle_sha256,
    load_candidate_bundle,
)
from spritelab.evaluation.memorization import (
    HARD_EVIDENCE_CLASSES,
    REVIEW_REQUIRED_EVIDENCE_CLASSES,
    parse_evidence_class,
    reconstruct_rgba,
)
from spritelab.evaluation.suite import read_jsonl

REVIEW_CHOICES = (
    "same_sprite_or_memorized",
    "same_silhouette_different_render",
    "common_generic_shape",
    "likely_false_positive",
    "uncertain",
)
SCHEMA_VERSION = "memorization_review_v1.0"
BOUND_REVIEW_SCHEMA_VERSION = "sprite_lab_memorization_review_event_v2"
BOUND_CANDIDATE_SCHEMA_VERSION = CANDIDATE_SCHEMA_VERSION
BOUND_REVIEW_GENESIS_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
BOUND_REVIEW_OUTCOMES = frozenset(
    {
        "same_sprite_or_memorized",
        "uncertain",
        "different_sprite",
        "common_generic_shape",
        "likely_false_positive",
    }
)
BOUND_REVIEW_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_sha256",
        "pair_id",
        "revision",
        "previous_event_sha256",
        "reviewer_id",
        "created_at_utc",
        "review_outcome",
        "human_note",
        "checkpoint_path",
        "checkpoint_sha256",
        "benchmark_manifest_path",
        "benchmark_manifest_sha256",
        "generated_report_path",
        "generated_report_sha256",
        "generated_manifest_path",
        "generated_manifest_sha256",
        "generated_sample_id",
        "prompt_id",
        "seed",
        "generated_png_path",
        "generated_png_sha256",
        "generated_decoded_rgba_sha256",
        "training_dataset_identity",
        "training_view_identity",
        "training_manifest_path",
        "training_manifest_sha256",
        "training_source_sprite_id",
        "training_row_or_index",
        "training_image_path",
        "training_source_blob_sha256",
        "training_decoded_rgba_sha256",
        "evidence_class",
        "detector_policy_version",
        "detector_policy_sha256",
        "comparison_method",
        "comparison_parameters_sha256",
        "candidate_evidence_sha256",
        "pair_evidence_sha256",
    }
)
BOUND_REVIEW_IDENTITY_FIELDS = frozenset(
    BOUND_REVIEW_REQUIRED_FIELDS
    - {
        "schema_version",
        "event_id",
        "event_sha256",
        "revision",
        "previous_event_sha256",
        "reviewer_id",
        "created_at_utc",
        "review_outcome",
        "human_note",
    }
)


class LegacyReviewReadOnlyError(RuntimeError):
    """Raised when an obsolete v1 authoring entry point is invoked."""


def _legacy_read_only() -> None:
    raise LegacyReviewReadOnlyError(
        "Legacy memorization reviews are read-only. New reviews must use "
        "python -m spritelab eval review-memorization-v2 ..."
    )


@dataclass(frozen=True)
class ReviewChainResult:
    """Structured result for one pair's signed append-only event chain."""

    chain_status: str
    authoritative_event: dict[str, Any] | None
    latest_valid_revision: int
    invalid_events: tuple[dict[str, Any], ...]
    blocking_reasons: tuple[str, ...]
    pending_review: bool


@dataclass(frozen=True)
class ReviewReplay:
    """Fail-closed replay result for an append-only review-event log."""

    chains: dict[str, ReviewChainResult]
    legacy_events: tuple[dict[str, Any], ...]
    global_invalid_events: tuple[dict[str, Any], ...]
    log_status: str
    seen_pair_ids: frozenset[str]

    @property
    def current(self) -> dict[str, dict[str, Any]]:
        """Return only fully valid authoritative events for compatibility."""
        return {
            pair_id: result.authoritative_event
            for pair_id, result in self.chains.items()
            if result.chain_status == "valid" and result.authoritative_event is not None
        }

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        reasons = [str(item["reason"]) for item in self.global_invalid_events]
        for result in self.chains.values():
            if result.chain_status in {"incomplete", "invalid", "contradictory", "identity_mismatch"}:
                reasons.extend(result.blocking_reasons)
        return tuple(sorted(set(reasons)))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical JSON as UTF-8, sorted compact keys, and preserved Unicode.

    This representation deliberately has no insignificant whitespace. Event hashes
    use this serialization after removing only the top-level ``event_sha256`` key.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of canonical JSON."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def review_event_sha256(event: Mapping[str, Any]) -> str:
    """Hash a bound review event, excluding only its computed hash field."""
    payload = dict(event)
    payload.pop("event_sha256", None)
    return canonical_sha256(payload)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_bound_event(event: Mapping[str, Any], line_number: int) -> list[str]:
    reasons: list[str] = []
    missing = sorted(BOUND_REVIEW_REQUIRED_FIELDS - event.keys())
    if missing:
        reasons.append(f"line {line_number}: missing required fields: {', '.join(missing)}")
    if event.get("schema_version") != BOUND_REVIEW_SCHEMA_VERSION:
        reasons.append(f"line {line_number}: wrong review schema")
    if not isinstance(event.get("event_id"), str) or not event.get("event_id"):
        reasons.append(f"line {line_number}: invalid event_id")
    if not isinstance(event.get("pair_id"), str) or not event.get("pair_id"):
        reasons.append(f"line {line_number}: invalid pair_id")
    revision = event.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        reasons.append(f"line {line_number}: revision must be a positive integer")
    if event.get("review_outcome") not in BOUND_REVIEW_OUTCOMES:
        reasons.append(f"line {line_number}: unknown review_outcome")
    if not isinstance(event.get("human_note"), str):
        reasons.append(f"line {line_number}: human_note must be a string")
    if not isinstance(event.get("reviewer_id"), str) or not event.get("reviewer_id"):
        reasons.append(f"line {line_number}: invalid reviewer_id")
    for field in (
        "checkpoint_path",
        "benchmark_manifest_path",
        "generated_report_path",
        "generated_manifest_path",
        "generated_sample_id",
        "prompt_id",
        "generated_png_path",
        "training_dataset_identity",
        "training_view_identity",
        "training_manifest_path",
        "training_source_sprite_id",
        "training_image_path",
        "detector_policy_version",
        "comparison_method",
    ):
        if not isinstance(event.get(field), str) or not event.get(field):
            reasons.append(f"line {line_number}: invalid {field}")
    if isinstance(event.get("seed"), bool) or not isinstance(event.get("seed"), int):
        reasons.append(f"line {line_number}: seed must be an integer but not bool")
    row_or_index = event.get("training_row_or_index")
    if row_or_index is None or isinstance(row_or_index, bool) or not isinstance(row_or_index, (int, str)):
        reasons.append(f"line {line_number}: invalid training_row_or_index")
    try:
        evidence_class = parse_evidence_class(event.get("evidence_class"))
        if evidence_class not in REVIEW_REQUIRED_EVIDENCE_CLASSES:
            reasons.append(f"line {line_number}: evidence_class is not review-required")
    except ValueError as error:
        reasons.append(f"line {line_number}: {error}")
    try:
        timestamp = datetime.fromisoformat(str(event.get("created_at_utc", "")).replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError
    except ValueError:
        reasons.append(f"line {line_number}: created_at_utc must be an aware UTC timestamp")
    for field in sorted(name for name in BOUND_REVIEW_REQUIRED_FIELDS if name.endswith("_sha256")):
        if not _valid_sha256(event.get(field)):
            reasons.append(f"line {line_number}: invalid {field}")
    previous = event.get("previous_event_sha256")
    if not _valid_sha256(previous):
        reasons.append(f"line {line_number}: invalid previous_event_sha256")
    if event.get("event_sha256") != review_event_sha256(event):
        reasons.append(f"line {line_number}: invalid event_sha256")
    return reasons


def _missing_chain() -> ReviewChainResult:
    return ReviewChainResult("missing", None, 0, (), ("missing authoritative review event",), True)


def replay_review_events(
    results_path: Path,
    *,
    expected_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> ReviewReplay:
    """Replay v2 chains without allowing malformed, legacy, or competing rows to win.

    Historical v1 rows are returned for display with ``promotion_authority=false``
    and ``identity_status=unbound_legacy``. They are never current decisions.
    """
    expected = dict(expected_identities or {})
    if not results_path.exists():
        return ReviewReplay(
            {pair_id: _missing_chain() for pair_id in sorted(expected)},
            (),
            (),
            "missing",
            frozenset(),
        )
    if not results_path.is_file():
        reason = "review event log is not a regular file"
        invalid = ({"line_number": None, "reason": reason},)
        return ReviewReplay(
            {pair_id: ReviewChainResult("invalid", None, 0, invalid, (reason,), True) for pair_id in sorted(expected)},
            (),
            invalid,
            "unreadable",
            frozenset(),
        )
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        reason = f"review event log cannot be read: {error}"
        invalid = ({"line_number": None, "reason": reason},)
        return ReviewReplay(
            {pair_id: ReviewChainResult("invalid", None, 0, invalid, (reason,), True) for pair_id in sorted(expected)},
            (),
            invalid,
            "unreadable",
            frozenset(),
        )
    records: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    global_invalid: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            global_invalid.append(
                {"line_number": line_number, "reason": f"line {line_number}: malformed JSON: {error.msg}"}
            )
            continue
        if not isinstance(raw, dict):
            global_invalid.append(
                {"line_number": line_number, "reason": f"line {line_number}: review event must be an object"}
            )
            continue
        pair_id = raw.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            seen_pairs.add(pair_id)
        if raw.get("schema_version") == SCHEMA_VERSION:
            legacy.append({**raw, "promotion_authority": False, "identity_status": "unbound_legacy"})
            continue
        records.append(
            {
                "line_number": line_number,
                "event": dict(raw),
                "reasons": _validate_bound_event(raw, line_number),
            }
        )

    event_id_lines: dict[str, list[int]] = {}
    for record in records:
        event_id = record["event"].get("event_id")
        if isinstance(event_id, str):
            event_id_lines.setdefault(event_id, []).append(int(record["line_number"]))
    duplicate_ids = {event_id for event_id, event_lines in event_id_lines.items() if len(event_lines) > 1}
    for record in records:
        if record["event"].get("event_id") in duplicate_ids:
            record["reasons"].append(f"line {record['line_number']}: duplicate event_id")

    records_by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pair_id = record["event"].get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            records_by_pair.setdefault(pair_id, []).append(record)
        else:
            global_invalid.extend(
                {"line_number": record["line_number"], "reason": reason} for reason in record["reasons"]
            )

    chains: dict[str, ReviewChainResult] = {}
    for pair_id in sorted(set(records_by_pair) | set(expected)):
        pair_records = records_by_pair.get(pair_id, [])
        if not pair_records:
            chains[pair_id] = _missing_chain()
            continue
        invalid_events: list[dict[str, Any]] = []
        by_revision: dict[int, list[dict[str, Any]]] = {}
        for record in pair_records:
            revision = record["event"].get("revision")
            if isinstance(revision, int) and not isinstance(revision, bool):
                by_revision.setdefault(revision, []).append(record)
            if record["reasons"]:
                invalid_events.append(
                    {
                        "line_number": record["line_number"],
                        "event_id": record["event"].get("event_id"),
                        "revision": revision,
                        "reasons": tuple(record["reasons"]),
                    }
                )

        competing = [revision for revision, items in by_revision.items() if len(items) > 1]
        revisions = sorted(by_revision)
        status = "valid"
        reasons: list[str] = []
        if competing:
            status = "contradictory"
            reasons.extend(f"pair {pair_id}: competing events for revision {revision}" for revision in competing)
            validation_revisions = [revision for revision in revisions if revision < min(competing)]
        elif not revisions or revisions[0] != 1 or revisions != list(range(1, revisions[-1] + 1)):
            status = "incomplete"
            reasons.append(f"pair {pair_id}: revision sequence must start at 1 and have no gaps")
            validation_revisions = []
            expected_revision = 1
            for revision in revisions:
                if revision != expected_revision:
                    break
                validation_revisions.append(revision)
                expected_revision += 1
        else:
            validation_revisions = revisions

        latest: dict[str, Any] | None = None
        latest_revision = 0
        previous_hash = BOUND_REVIEW_GENESIS_SHA256
        baseline_identity: dict[str, Any] | None = None
        identity_mismatches: list[str] = []
        for revision in validation_revisions:
            record = by_revision[revision][0]
            event = record["event"]
            if record["reasons"]:
                if status == "valid":
                    status = "invalid"
                reasons.extend(record["reasons"])
                break
            if event.get("previous_event_sha256") != previous_hash:
                reason = f"pair {pair_id}: invalid previous-event hash at revision {revision}"
                if status == "valid":
                    status = "invalid"
                reasons.append(reason)
                invalid_events.append(
                    {
                        "line_number": record["line_number"],
                        "event_id": event.get("event_id"),
                        "revision": revision,
                        "reasons": (reason,),
                    }
                )
                break
            identity = {field: event.get(field) for field in BOUND_REVIEW_IDENTITY_FIELDS}
            if baseline_identity is None:
                baseline_identity = identity
            else:
                identity_mismatches.extend(
                    field for field, value in baseline_identity.items() if identity.get(field) != value
                )
            expected_identity = expected.get(pair_id)
            if expected_identity is not None:
                identity_mismatches.extend(
                    field for field, value in expected_identity.items() if event.get(field) != value
                )
            latest = dict(event)
            latest_revision = revision
            previous_hash = str(event["event_sha256"])
        if status == "valid" and identity_mismatches:
            status = "identity_mismatch"
            reasons.append(f"pair {pair_id}: event identity mismatch: {', '.join(sorted(set(identity_mismatches)))}")
        if status == "valid" and latest is not None and latest.get("review_outcome") == "same_sprite_or_memorized":
            reasons.append(f"pair {pair_id}: authoritative review blocks memorization clearance")
        pending_review = status != "valid" or latest is None or latest.get("review_outcome") == "uncertain"
        chains[pair_id] = ReviewChainResult(
            status,
            latest,
            latest_revision,
            tuple(invalid_events),
            tuple(sorted(set(reasons))),
            pending_review,
        )
    log_status = "malformed" if global_invalid else "valid"
    return ReviewReplay(
        chains,
        tuple(legacy),
        tuple(global_invalid),
        log_status,
        frozenset(seen_pairs),
    )


@dataclass(frozen=True)
class BoundReviewTask:
    """One identity-bound review task coalesced from all evidence reasons."""

    pair_id: str
    evidence_classes: tuple[str, ...]
    evidence_reasons: tuple[dict[str, Any], ...]
    identity: dict[str, Any]
    pair_evidence_sha256: str
    hard_evidence_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "evidence_classes": list(self.evidence_classes),
            "evidence_reasons": [dict(reason) for reason in self.evidence_reasons],
            "identity": dict(self.identity),
            "pair_evidence_sha256": self.pair_evidence_sha256,
            "hard_evidence_present": self.hard_evidence_present,
            "pending_review": True,
        }


def bound_event_identity(bundle: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact immutable identity projection required on every event."""
    return {
        "pair_id": pair.get("pair_id"),
        "checkpoint_path": bundle.get("checkpoint_path"),
        "checkpoint_sha256": bundle.get("checkpoint_sha256"),
        "benchmark_manifest_path": bundle.get("benchmark_manifest_path"),
        "benchmark_manifest_sha256": bundle.get("benchmark_manifest_sha256"),
        "generated_report_path": bundle.get("generated_report_path"),
        "generated_report_sha256": bundle.get("generated_report_sha256"),
        "generated_manifest_path": bundle.get("generated_manifest_path"),
        "generated_manifest_sha256": bundle.get("generated_manifest_sha256"),
        "generated_sample_id": pair.get("generated_sample_id"),
        "prompt_id": pair.get("prompt_id"),
        "seed": pair.get("seed"),
        "generated_png_path": pair.get("generated_png_path"),
        "generated_png_sha256": pair.get("generated_png_sha256"),
        "generated_decoded_rgba_sha256": pair.get("generated_decoded_rgba_sha256"),
        "training_dataset_identity": pair.get("training_dataset_identity"),
        "training_view_identity": pair.get("training_view_identity"),
        "training_manifest_path": bundle.get("training_manifest_path"),
        "training_manifest_sha256": bundle.get("training_manifest_sha256"),
        "training_source_sprite_id": pair.get("training_source_sprite_id"),
        "training_row_or_index": pair.get("training_row_or_index"),
        "training_image_path": pair.get("training_image_path"),
        "training_source_blob_sha256": pair.get("training_source_blob_sha256"),
        "training_decoded_rgba_sha256": pair.get("training_decoded_rgba_sha256"),
        "evidence_class": pair.get("evidence_class"),
        "detector_policy_version": bundle.get("detector_policy_version"),
        "detector_policy_sha256": bundle.get("detector_policy_sha256"),
        "comparison_method": bundle.get("comparison_method"),
        "comparison_parameters_sha256": bundle.get("comparison_parameters_sha256"),
        "candidate_evidence_sha256": bundle.get("candidate_evidence_sha256"),
        "pair_evidence_sha256": canonical_sha256(pair),
        **({"noise_seed": pair.get("noise_seed")} if "noise_seed" in pair else {}),
    }


def _candidate_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    return candidate_bundle_sha256(bundle)


def _candidate_reasons(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons = [
        {
            "evidence_class": pair.get("evidence_class"),
            "evidence_metrics": pair.get("evidence_metrics"),
            "evidence_diagnostics": pair.get("evidence_diagnostics"),
        }
    ]
    additional = pair.get("evidence_reasons", [])
    if not isinstance(additional, list):
        raise ValueError("evidence_reasons must be a list when present")
    for reason in additional:
        if isinstance(reason, str):
            reasons.append(
                {
                    "evidence_class": reason,
                    "evidence_metrics": pair.get("evidence_metrics"),
                    "evidence_diagnostics": pair.get("evidence_diagnostics"),
                }
            )
        elif isinstance(reason, dict):
            reasons.append(
                {
                    "evidence_class": reason.get("evidence_class"),
                    "evidence_metrics": reason.get("evidence_metrics", pair.get("evidence_metrics")),
                    "evidence_diagnostics": reason.get("evidence_diagnostics", pair.get("evidence_diagnostics")),
                }
            )
        else:
            raise ValueError("each evidence reason must be a string or object")
    unique = {canonical_sha256(reason): reason for reason in reasons}
    return [unique[key] for key in sorted(unique)]


def load_bound_review_tasks(
    candidate_evidence: Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[BoundReviewTask]]:
    """Load and strictly validate the complete v2 bundle into a review queue."""
    validation = load_candidate_bundle(candidate_evidence, expected_context=expected_context)
    if not validation.valid:
        raise ValueError("candidate evidence is invalid or outdated: " + "; ".join(validation.reasons))
    bundle = validation.bundle
    pairs = list(validation.pairs)

    grouped: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"candidate pair {index} must be an object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"candidate pair {index} has invalid pair_id")
        identity = bound_event_identity(bundle, pair)
        missing_identity = sorted(field for field in BOUND_REVIEW_IDENTITY_FIELDS if field not in identity)
        if missing_identity:
            raise ValueError(f"candidate pair {pair_id} missing identity fields: {', '.join(missing_identity)}")
        for field in sorted(name for name in BOUND_REVIEW_IDENTITY_FIELDS if name.endswith("_sha256")):
            if not _valid_sha256(identity.get(field)):
                raise ValueError(f"candidate pair {pair_id} has invalid {field}")
        for field in (
            "pair_id",
            "checkpoint_path",
            "benchmark_manifest_path",
            "generated_report_path",
            "generated_manifest_path",
            "generated_sample_id",
            "prompt_id",
            "generated_png_path",
            "training_dataset_identity",
            "training_view_identity",
            "training_manifest_path",
            "training_source_sprite_id",
            "training_image_path",
            "evidence_class",
            "detector_policy_version",
            "comparison_method",
        ):
            if not isinstance(identity.get(field), str) or not identity[field]:
                raise ValueError(f"candidate pair {pair_id} has invalid {field}")
        if isinstance(identity.get("seed"), bool) or not isinstance(identity.get("seed"), int):
            raise ValueError(f"candidate pair {pair_id} has invalid seed")
        if pair.get("training_manifest_sha256") != bundle.get("training_manifest_sha256"):
            raise ValueError(f"candidate pair {pair_id} training manifest identity mismatch")
        row_or_index = identity.get("training_row_or_index")
        if row_or_index is None or isinstance(row_or_index, bool) or not isinstance(row_or_index, (int, str)):
            raise ValueError(f"candidate pair {pair_id} has invalid training_row_or_index")
        reasons = _candidate_reasons(pair)
        parsed_classes = [parse_evidence_class(reason.get("evidence_class")) for reason in reasons]
        for reason in reasons:
            if not isinstance(reason.get("evidence_metrics"), dict) or not isinstance(
                reason.get("evidence_diagnostics"), dict
            ):
                raise ValueError(f"candidate pair {pair_id} has malformed evidence diagnostics")
        review_required = any(item in REVIEW_REQUIRED_EVIDENCE_CLASSES for item in parsed_classes)
        if not review_required:
            continue
        group = grouped.get(pair_id)
        if group is None:
            grouped[pair_id] = {
                "identity": identity,
                "pairs": [pair],
                "reasons": reasons,
                "classes": parsed_classes,
            }
        else:
            differing = sorted(field for field, value in group["identity"].items() if identity.get(field) != value)
            if differing:
                raise ValueError(f"candidate pair {pair_id} has conflicting identities: {', '.join(differing)}")
            group["pairs"].append(pair)
            group["reasons"].extend(reasons)
            group["classes"].extend(parsed_classes)

    tasks: list[BoundReviewTask] = []
    for pair_id, group in grouped.items():
        unique_reasons = {canonical_sha256(reason): reason for reason in group["reasons"]}
        evidence_reasons = tuple(unique_reasons[key] for key in sorted(unique_reasons))
        evidence_classes = tuple(sorted({str(item.value) for item in group["classes"]}))
        source_pairs = group["pairs"]
        pair_hash = (
            canonical_sha256(source_pairs[0])
            if len(source_pairs) == 1
            else canonical_sha256({"coalesced_pair_evidence": source_pairs})
        )
        identity = dict(group["identity"])
        identity["pair_evidence_sha256"] = pair_hash
        tasks.append(
            BoundReviewTask(
                pair_id,
                evidence_classes,
                evidence_reasons,
                identity,
                pair_hash,
                any(item in HARD_EVIDENCE_CLASSES for item in group["classes"]),
            )
        )
    tasks.sort(key=lambda task: task.pair_id)
    return bundle, tasks


def append_bound_review_event(
    candidate_evidence: Path,
    review_event_log: Path,
    *,
    pair_id: str,
    review_outcome: str,
    reviewer_id: str,
    human_note: str = "",
    created_at_utc: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a fully bound and signed v2 event for one controlled task."""
    if review_outcome not in BOUND_REVIEW_OUTCOMES:
        raise ValueError(f"unknown bound review outcome: {review_outcome!r}")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be a nonempty string")
    if not isinstance(human_note, str):
        raise ValueError("human_note must be a string")
    _, tasks = load_bound_review_tasks(candidate_evidence, expected_context=expected_context)
    task_by_id = {task.pair_id: task for task in tasks}
    task = task_by_id.get(pair_id)
    if task is None:
        raise ValueError(f"pair is not a review-required candidate: {pair_id}")
    if task.hard_evidence_present and review_outcome in {
        "different_sprite",
        "common_generic_shape",
        "likely_false_positive",
    }:
        raise ValueError("hard exact-RGBA evidence cannot be cleared by review")
    expected_identities = {candidate.pair_id: candidate.identity for candidate in tasks}
    replay = replay_review_events(review_event_log, expected_identities=expected_identities)
    if replay.global_invalid_events or replay.legacy_events:
        raise ValueError("refusing to append to a malformed or legacy-mixed review log")
    chain = replay.chains.get(pair_id, _missing_chain())
    if chain.chain_status == "missing":
        revision = 1
        previous_hash = BOUND_REVIEW_GENESIS_SHA256
    elif chain.chain_status == "valid" and chain.authoritative_event is not None:
        revision = chain.latest_valid_revision + 1
        previous_hash = str(chain.authoritative_event["event_sha256"])
    else:
        raise ValueError(f"refusing to append to {chain.chain_status} chain for pair {pair_id}")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    event = {
        "schema_version": BOUND_REVIEW_SCHEMA_VERSION,
        "event_id": f"review-{uuid4()}",
        "pair_id": pair_id,
        "revision": revision,
        "previous_event_sha256": previous_hash,
        "reviewer_id": reviewer_id.strip(),
        "created_at_utc": timestamp,
        "review_outcome": review_outcome,
        "human_note": human_note,
        **task.identity,
    }
    event["event_sha256"] = review_event_sha256(event)
    reasons = _validate_bound_event(event, revision)
    if reasons:
        raise ValueError("refusing malformed bound event: " + "; ".join(reasons))
    review_event_log.parent.mkdir(parents=True, exist_ok=True)
    with review_event_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


@dataclass(frozen=True)
class ReviewPair:
    """Benchmark evidence and read-only images for one suspicious pair."""

    pair_id: str
    benchmark: dict[str, Any]
    nearest: dict[str, Any]
    training_provenance: dict[str, Any]
    generated_rgba: np.ndarray
    training_rgba: np.ndarray

    @property
    def nearest_match_reason(self) -> str:
        evidence: list[str] = []
        if self.nearest.get("exact_rgba"):
            evidence.append("exact RGBA pixels")
        if self.nearest.get("exact_alpha"):
            evidence.append("exact alpha mask")
        if self.nearest.get("translated_duplicate"):
            evidence.append("translation-normalized alpha match")
        evidence.extend(
            (
                f"RGBA pixel distance {float(self.nearest.get('pixel_distance', 0.0)):.8f}",
                f"geometry IoU {float(self.nearest.get('geometry_iou', 0.0)):.6f}",
                f"perceptual distance {float(self.nearest.get('perceptual_distance', 0.0)):.8f}",
            )
        )
        return "; ".join(evidence)


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGBA"))


def _manifest_provenance(manifest: Path, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(manifest):
        key = (str(row.get("npz_file") or ""), int(row.get("npz_row", -1)))
        if key in wanted and key not in found:
            found[key] = {
                "training_manifest": str(manifest),
                "split": row.get("split"),
                "sprite_id": row.get("sprite_id") or row.get("source_sprite_id"),
                "source": row.get("source") or {},
                "schema_version": row.get("schema_version"),
            }
            if len(found) == len(wanted):
                break
    return found


def load_review_pairs(report_dir: Path, *, project_root: Path | None = None) -> list[ReviewPair]:
    """Load exact-alpha pairs already reported by generation benchmark v1."""
    report_dir = report_dir.resolve()
    root = (project_root or Path.cwd()).resolve()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != "generation_benchmark_v1.0":
        raise ValueError("review input must be a generation benchmark v1 report")
    rows = [
        row
        for row in read_jsonl(report_dir / "per_image_metrics.jsonl")
        if row.get("suspicious_memorization") == "exact_alpha"
    ]
    if not rows:
        return []

    manifests = [_resolve(root, value) for value in summary.get("training_manifests", [])]
    wanted_by_manifest: dict[Path, set[tuple[str, int]]] = {path: set() for path in manifests}
    for row in rows:
        nearest = row["training_neighbors"][0]
        key = (str(nearest["npz_file"]), int(nearest["npz_row"]))
        for manifest in manifests:
            if manifest.parent.resolve() == _resolve(root, nearest["dataset"]).resolve():
                wanted_by_manifest[manifest].add(key)
    provenance: dict[tuple[str, str, int], dict[str, Any]] = {}
    for manifest, wanted in wanted_by_manifest.items():
        for key, value in _manifest_provenance(manifest, wanted).items():
            provenance[(str(manifest.parent.resolve()), *key)] = value

    npz_cache: dict[Path, Any] = {}
    pairs: list[ReviewPair] = []
    try:
        for row in rows:
            nearest = dict(row["training_neighbors"][0])
            dataset = _resolve(root, nearest["dataset"]).resolve()
            npz_path = dataset / str(nearest["npz_file"])
            if npz_path not in npz_cache:
                npz_cache[npz_path] = np.load(npz_path, mmap_mode="r")
            generated_path = _resolve(root, row["image"]).resolve()
            train_key = (str(dataset), str(nearest["npz_file"]), int(nearest["npz_row"]))
            pair_id = f"{row['sample_id']}__{nearest['sprite_id']}"
            pairs.append(
                ReviewPair(
                    pair_id=pair_id,
                    benchmark={**row, "image": str(generated_path), "report": str(report_dir)},
                    nearest=nearest,
                    training_provenance=provenance.get(train_key, {}),
                    generated_rgba=_load_rgba(generated_path),
                    training_rgba=reconstruct_rgba(npz_cache[npz_path], int(nearest["npz_row"])),
                )
            )
    finally:
        for archive in npz_cache.values():
            archive.close()
    return pairs


def load_latest_reviews(results_path: Path) -> dict[str, dict[str, Any]]:
    """Replay the append-only log and retain the newest decision per pair."""
    if not results_path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != SCHEMA_VERSION or not row.get("pair_id"):
            raise ValueError(f"invalid review event at line {line_number}")
        latest[str(row["pair_id"])] = row
    return latest


def resume_index(pairs: Sequence[ReviewPair], latest: Mapping[str, Mapping[str, Any]]) -> int:
    """Resume at the first pair without a saved human decision."""
    for index, pair in enumerate(pairs):
        if pair.pair_id not in latest:
            return index
    return max(0, len(pairs) - 1)


def append_review(
    output_dir: Path,
    pair: ReviewPair,
    *,
    classification: str,
    notes: str,
    block_promotion: bool,
    rule_needs_review: bool,
    current_index: int,
    pair_count: int,
) -> dict[str, Any]:
    """Durably append a review event, then refresh resumable state and summaries."""
    _legacy_read_only()
    if classification not in REVIEW_CHOICES:
        raise ValueError(f"unknown classification: {classification}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "review_results.jsonl"
    previous = load_latest_reviews(results_path).get(pair.pair_id)
    event = {
        "schema_version": SCHEMA_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "revision": int(previous.get("revision", 0)) + 1 if previous else 1,
        "pair_id": pair.pair_id,
        "sample_id": pair.benchmark["sample_id"],
        "training_sprite_id": pair.nearest["sprite_id"],
        "classification": classification,
        "notes": notes,
        "block_promotion": bool(block_promotion),
        "threshold_or_rule_needs_review": bool(rule_needs_review),
        "prompt": pair.benchmark.get("prompt", ""),
        "seed": pair.benchmark.get("seed"),
        "noise_seed": pair.benchmark.get("noise_seed"),
        "checkpoint": pair.benchmark.get("checkpoint", ""),
        "nearest_match_reason": pair.nearest_match_reason,
        "nearest": pair.nearest,
        "generated_provenance": {
            "report": pair.benchmark["report"],
            "run": pair.benchmark.get("run"),
            "image": pair.benchmark["image"],
        },
        "training_provenance": pair.training_provenance,
    }
    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    latest = load_latest_reviews(results_path)
    next_index = min(current_index + 1, max(0, pair_count - 1))
    state = {
        "schema_version": SCHEMA_VERSION,
        "current_index": next_index,
        "pair_count": pair_count,
        "completed_pair_ids": sorted(latest),
        "completed_count": len(latest),
    }
    _atomic_json(output_dir / "review_state.json", state)
    write_summaries(output_dir, pair_count=pair_count)
    return event


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_summaries(output_dir: Path, *, pair_count: int) -> dict[str, Any]:
    """Write JSON and Markdown summaries from the latest decision per pair."""
    latest = load_latest_reviews(output_dir / "review_results.jsonl")
    classifications = Counter(row["classification"] for row in latest.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "pair_count": pair_count,
        "reviewed_count": len(latest),
        "remaining_count": max(0, pair_count - len(latest)),
        "classification_counts": {choice: classifications.get(choice, 0) for choice in REVIEW_CHOICES},
        "block_promotion_count": sum(bool(row["block_promotion"]) for row in latest.values()),
        "threshold_or_rule_review_count": sum(bool(row["threshold_or_rule_needs_review"]) for row in latest.values()),
        "reviews": [latest[key] for key in sorted(latest)],
    }
    _atomic_json(output_dir / "review_summary.json", summary)
    lines = [
        "# Exact-alpha match review",
        "",
        f"- Reviewed: {summary['reviewed_count']} / {pair_count}",
        f"- Remaining: {summary['remaining_count']}",
        f"- Block promotion: {summary['block_promotion_count']}",
        f"- Threshold/rule review: {summary['threshold_or_rule_review_count']}",
        "",
        "## Classification counts",
        "",
        *(f"- `{choice}`: {classifications.get(choice, 0)}" for choice in REVIEW_CHOICES),
        "",
        "## Latest decisions",
        "",
    ]
    for row in summary["reviews"]:
        lines.extend(
            (
                f"### {row['sample_id']} / {row['training_sprite_id']}",
                "",
                f"- Classification: `{row['classification']}`",
                f"- Block promotion: {row['block_promotion']}",
                f"- Threshold/rule review: {row['threshold_or_rule_needs_review']}",
                f"- Notes: {row['notes'] or '(none)'}",
                "",
            )
        )
    (output_dir / "review_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def initialize_review(output_dir: Path, pairs: Sequence[ReviewPair]) -> int:
    """Materialize resumable state and empty/current summaries before opening the GUI."""
    _legacy_read_only()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = load_latest_reviews(output_dir / "review_results.jsonl")
    start = resume_index(pairs, latest)
    _atomic_json(
        output_dir / "review_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "current_index": start,
            "pair_count": len(pairs),
            "completed_pair_ids": sorted(latest),
            "completed_count": len(latest),
        },
    )
    write_summaries(output_dir, pair_count=len(pairs))
    return start


def _display_image(array: np.ndarray, *, alpha_mask: bool = False, difference: np.ndarray | None = None) -> Image.Image:
    if alpha_mask:
        alpha = array[..., 3]
        rgba = np.stack((alpha, alpha, alpha, np.full_like(alpha, 255)), axis=-1)
    elif difference is not None:
        delta = np.abs(array.astype(np.int16) - difference.astype(np.int16)).astype(np.uint8)
        delta[..., 3] = 255
        rgba = delta
    else:
        rgba = array
    return Image.fromarray(rgba, "RGBA").resize((224, 224), Image.Resampling.NEAREST)


def launch_gui(pairs: Sequence[ReviewPair], output_dir: Path) -> None:
    """Launch the Tk review UI. Tk imports stay optional until this call."""
    _legacy_read_only()
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import ImageTk

    if not pairs:
        raise ValueError("the report contains no exact-alpha suspicious pairs")
    start = initialize_review(output_dir, pairs)

    root = tk.Tk()
    root.title("Generation benchmark v1 — exact-alpha human review")
    root.geometry("1220x870")
    index = tk.IntVar(value=start)
    classification = tk.StringVar(value="uncertain")
    block = tk.BooleanVar(value=False)
    rule_review = tk.BooleanVar(value=False)
    header = tk.StringVar()
    details = tk.StringVar()
    status = tk.StringVar()
    image_labels: list[ttk.Label] = []
    image_refs: list[Any] = []

    top = ttk.Frame(root, padding=10)
    top.pack(fill="both", expand=True)
    ttk.Label(top, textvariable=header, font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
    images = ttk.Frame(top)
    images.pack(fill="x", pady=8)
    for title in ("Generated", "Nearest training", "Generated alpha", "Training alpha", "Pixel difference"):
        cell = ttk.Frame(images)
        cell.pack(side="left", padx=4)
        ttk.Label(cell, text=title).pack()
        label = ttk.Label(cell)
        label.pack()
        image_labels.append(label)
    ttk.Label(top, textvariable=details, justify="left", wraplength=1170).pack(anchor="w", pady=4)

    choices = ttk.LabelFrame(
        top, text="Human classification (exact alpha is evidence, not an automatic verdict)", padding=8
    )
    choices.pack(fill="x", pady=6)
    for choice in REVIEW_CHOICES:
        ttk.Radiobutton(choices, text=choice, variable=classification, value=choice).pack(side="left", padx=6)
    flags = ttk.Frame(top)
    flags.pack(fill="x", pady=4)
    ttk.Checkbutton(flags, text="Pair should block promotion", variable=block).pack(side="left", padx=4)
    ttk.Checkbutton(flags, text="Threshold/rule needs review", variable=rule_review).pack(side="left", padx=16)
    ttk.Label(top, text="Notes").pack(anchor="w")
    notes = tk.Text(top, height=5, wrap="word")
    notes.pack(fill="x")

    def show(position: int) -> None:
        nonlocal image_refs
        position = max(0, min(position, len(pairs) - 1))
        index.set(position)
        pair = pairs[position]
        prior = load_latest_reviews(output_dir / "review_results.jsonl").get(pair.pair_id)
        classification.set(str(prior["classification"]) if prior else "uncertain")
        block.set(bool(prior and prior["block_promotion"]))
        rule_review.set(bool(prior and prior["threshold_or_rule_needs_review"]))
        notes.delete("1.0", "end")
        if prior:
            notes.insert("1.0", str(prior.get("notes") or ""))
        header.set(f"Pair {position + 1} / {len(pairs)} — {pair.benchmark['sample_id']} ↔ {pair.nearest['sprite_id']}")
        source = pair.training_provenance.get("source") or {}
        details.set(
            f"Prompt: {pair.benchmark.get('prompt')} | seed: {pair.benchmark.get('seed')} | "
            f"noise seed: {pair.benchmark.get('noise_seed')}\nCheckpoint: {pair.benchmark.get('checkpoint')}\n"
            f"Nearest-match reason: {pair.nearest_match_reason}\nGenerated: {pair.benchmark.get('image')}\n"
            f"Training: {pair.nearest.get('dataset')}/{pair.nearest.get('npz_file')} row {pair.nearest.get('npz_row')} | "
            f"split: {pair.training_provenance.get('split')} | source manifest: {source.get('manifest_file')} row {source.get('manifest_row')}"
        )
        rendered = (
            _display_image(pair.generated_rgba),
            _display_image(pair.training_rgba),
            _display_image(pair.generated_rgba, alpha_mask=True),
            _display_image(pair.training_rgba, alpha_mask=True),
            _display_image(pair.generated_rgba, difference=pair.training_rgba),
        )
        image_refs = [ImageTk.PhotoImage(image) for image in rendered]
        for label, photo in zip(image_labels, image_refs, strict=True):
            label.configure(image=photo)
        reviewed = len(load_latest_reviews(output_dir / "review_results.jsonl"))
        status.set(f"Saved: {reviewed}/{len(pairs)} | append-only log: {output_dir / 'review_results.jsonl'}")

    def save() -> None:
        pair = pairs[index.get()]
        append_review(
            output_dir,
            pair,
            classification=classification.get(),
            notes=notes.get("1.0", "end").strip(),
            block_promotion=block.get(),
            rule_needs_review=rule_review.get(),
            current_index=index.get(),
            pair_count=len(pairs),
        )
        latest_now = load_latest_reviews(output_dir / "review_results.jsonl")
        if len(latest_now) == len(pairs):
            show(index.get())
            messagebox.showinfo(
                "Review complete", "All pairs have saved decisions. JSON and Markdown summaries are current."
            )
        else:
            show(resume_index(pairs, latest_now))

    controls = ttk.Frame(top)
    controls.pack(fill="x", pady=8)
    ttk.Button(controls, text="← Previous", command=lambda: show(index.get() - 1)).pack(side="left")
    ttk.Button(controls, text="Save decision and continue", command=save).pack(side="left", padx=10)
    ttk.Button(controls, text="Next →", command=lambda: show(index.get() + 1)).pack(side="left")
    ttk.Label(controls, textvariable=status).pack(side="right")
    show(start)
    root.mainloop()
