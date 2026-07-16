"""Read-only, fail-closed promotion decisions for bound memorization evidence."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.evaluation.candidate_bundle import (
    CANDIDATE_SCHEMA_VERSION,
    load_candidate_bundle,
)
from spritelab.evaluation.memorization import (
    COMPARISON_METHOD,
    COMPARISON_PARAMETERS,
    COMPARISON_PARAMETERS_SHA256,
    DETECTOR_POLICY,
    DETECTOR_POLICY_SHA256,
    DETECTOR_POLICY_VERSION,
    HARD_EVIDENCE_CLASSES,
    REVIEW_REQUIRED_EVIDENCE_CLASSES,
    EvidenceClass,
    MemorizationMachineStatus,
    detector_policy_record,
    evaluate_memorization_outcome,
    parse_evidence_class,
    recompute_memorization_status,
)
from spritelab.evaluation.memorization_review import (
    bound_event_identity,
    canonical_sha256,
    replay_review_events,
)

DECISION_SCHEMA_VERSION = "sprite_lab_promotion_decision_v2"
CLEARING_OUTCOMES = frozenset({"different_sprite", "common_generic_shape", "likely_false_positive"})
HARD_BLOCK_OUTCOMES = frozenset({"same_sprite_or_memorized"})
PENDING_OUTCOMES = frozenset({"uncertain"})
MEMORIZATION_CHECKS = frozenset(
    {
        "detector_policy_supported",
        "memorization_hard_evidence",
        "memorization_reviews_resolved",
        "exact_train_duplicates",
        "near_train_duplicates",
    }
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_rgba_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image.load()
        rgba = image.convert("RGBA")
        dimensions = rgba.width.to_bytes(8, "big") + rgba.height.to_bytes(8, "big")
        return sha256(dimensions + rgba.tobytes()).hexdigest()


def candidate_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    """Hash the complete ordered bundle, excluding only its self-hash field."""
    payload = dict(bundle)
    payload.pop("candidate_evidence_sha256", None)
    return canonical_sha256(payload)


def pair_evidence_sha256(pair: Mapping[str, Any]) -> str:
    return canonical_sha256(pair)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _load_object(path: Path, label: str, reasons: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        reasons.append(f"{label} is missing or malformed: {error}")
        return {}
    if not isinstance(value, dict):
        reasons.append(f"{label} must be a JSON object")
        return {}
    return value


def _load_manifest(path: Path, label: str, reasons: list[str]) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                for key in ("records", "samples", "images", "rows"):
                    if isinstance(value.get(key), list):
                        return [row for row in value[key] if isinstance(row, dict)]
                return [value]
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        reasons.append(f"{label} is missing or malformed: {error}")
    return []


def _path_identity(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def _resolve_manifest_path(manifest: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else manifest.parent / path


def _validate_bound_file(
    actual_path: Path,
    evidence: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    label: str,
    reasons: list[str],
) -> str | None:
    try:
        actual_hash = file_sha256(actual_path)
    except OSError as error:
        reasons.append(f"{label} cannot be read: {error}")
        return None
    expected_path = evidence.get(path_field)
    if not isinstance(expected_path, str) or _path_identity(Path(expected_path)) != _path_identity(actual_path):
        reasons.append(f"{label} path identity mismatch")
    if evidence.get(hash_field) != actual_hash:
        reasons.append(f"{label} SHA-256 mismatch")
    return actual_hash


def _validate_policy(path: Path, bundle: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    policy = _load_object(path, "detector policy artifact", reasons)
    expected = detector_policy_record()
    if policy != expected:
        reasons.append("detector policy artifact content is unsupported or changed")
    core = {key: policy.get(key) for key in DETECTOR_POLICY}
    if canonical_sha256(core) != policy.get("detector_policy_sha256"):
        reasons.append("detector policy artifact hash does not match its content")
    if policy.get("detector_policy_version") != DETECTOR_POLICY_VERSION:
        reasons.append("unknown detector policy version")
    if policy.get("detector_policy_sha256") != DETECTOR_POLICY_SHA256:
        reasons.append("detector policy SHA-256 is incompatible")
    if policy.get("comparison_method") != COMPARISON_METHOD:
        reasons.append("comparison method is incompatible")
    if policy.get("comparison_parameters") != COMPARISON_PARAMETERS:
        reasons.append("comparison parameters changed")
    if policy.get("comparison_parameters_sha256") != COMPARISON_PARAMETERS_SHA256:
        reasons.append("comparison parameters SHA-256 is incompatible")
    for field in (
        "detector_policy_version",
        "detector_policy_sha256",
        "comparison_method",
        "comparison_parameters_sha256",
    ):
        if bundle.get(field) != policy.get(field):
            reasons.append(f"candidate bundle {field} mismatch")
    try:
        if bundle.get("detector_policy_artifact_sha256") != file_sha256(path):
            reasons.append("detector policy artifact file SHA-256 mismatch")
    except OSError:
        pass
    return policy


def _pair_reasons(pair: Any, index: int, bundle: Mapping[str, Any]) -> list[str]:
    if not isinstance(pair, dict):
        return [f"candidate pair {index} must be an object"]
    required = {
        "pair_id",
        "generated_sample_id",
        "prompt_id",
        "seed",
        "generated_png_path",
        "generated_png_sha256",
        "generated_decoded_rgba_sha256",
        "training_dataset_identity",
        "training_view_identity",
        "training_source_sprite_id",
        "training_row_or_index",
        "training_image_path",
        "training_source_blob_sha256",
        "training_decoded_rgba_sha256",
        "training_manifest_sha256",
        "evidence_class",
        "evidence_metrics",
        "evidence_diagnostics",
    }
    missing = sorted(required - pair.keys())
    reasons = [f"candidate pair {index} missing fields: {', '.join(missing)}"] if missing else []
    pair_id = pair.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        reasons.append(f"candidate pair {index} has invalid pair_id")
    try:
        evidence_class = parse_evidence_class(pair.get("evidence_class"))
    except ValueError as error:
        reasons.append(f"candidate pair {index} {error}")
        evidence_class = None
    if pair.get("exact_rgba") is True and evidence_class not in {
        EvidenceClass.EXACT_RGBA_NONTRIVIAL,
        EvidenceClass.EXACT_RGBA_LOW_EVIDENCE_COLLISION,
    }:
        reasons.append(f"candidate pair {index} silently downgrades an exact-RGBA match")
    if pair.get("training_manifest_sha256") != bundle.get("training_manifest_sha256"):
        reasons.append(f"candidate pair {index} training manifest hash mismatch")
    for field in ("evidence_metrics", "evidence_diagnostics"):
        if not isinstance(pair.get(field), dict):
            reasons.append(f"candidate pair {index} {field} must be an object")
    return reasons


def _record_value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _validate_generated_source(
    pair: Mapping[str, Any], manifest: Path, records: list[dict[str, Any]], reasons: list[str]
) -> None:
    pair_id = str(pair.get("pair_id"))
    sample_id = pair.get("generated_sample_id")
    matches = [row for row in records if _record_value(row, "sample_id", "generated_sample_id", "id") == sample_id]
    if len(matches) != 1:
        reasons.append(f"generated manifest identity mismatch for pair {pair_id}")
        return
    record = matches[0]
    record_path = _resolve_manifest_path(
        manifest, _record_value(record, "image_path", "generated_png_path", "png_path", "image")
    )
    pair_path = Path(str(pair.get("generated_png_path")))
    if record_path is None or _path_identity(record_path) != _path_identity(pair_path):
        reasons.append(f"generated manifest image path mismatch for pair {pair_id}")
    try:
        blob_hash = file_sha256(pair_path)
        decoded_hash = decoded_rgba_sha256(pair_path)
    except (OSError, ValueError) as error:
        reasons.append(f"generated image cannot be validated for pair {pair_id}: {error}")
        return
    if blob_hash != pair.get("generated_png_sha256"):
        reasons.append(f"generated image file hash mismatch for pair {pair_id}")
    if decoded_hash != pair.get("generated_decoded_rgba_sha256"):
        reasons.append(f"generated decoded-RGBA hash mismatch for pair {pair_id}")
    for names, expected in (
        (("png_sha256", "generated_png_sha256", "blob_sha256"), blob_hash),
        (("decoded_rgba_sha256", "generated_decoded_rgba_sha256"), decoded_hash),
    ):
        if _record_value(record, *names) != expected:
            reasons.append(f"generated manifest hash mismatch for pair {pair_id}")


def _validate_training_source(
    pair: Mapping[str, Any], manifest: Path, records: list[dict[str, Any]], reasons: list[str]
) -> None:
    pair_id = str(pair.get("pair_id"))
    matches = [
        row
        for row in records
        if _record_value(row, "dataset_identity", "training_dataset_identity") == pair.get("training_dataset_identity")
        and _record_value(row, "view_identity", "training_view_identity") == pair.get("training_view_identity")
        and _record_value(row, "source_sprite_id", "training_source_sprite_id", "sprite_id")
        == pair.get("training_source_sprite_id")
        and _record_value(row, "row_or_index", "training_row_or_index", "row", "npz_row")
        == pair.get("training_row_or_index")
    ]
    if len(matches) != 1:
        reasons.append(f"training manifest source identity mismatch for pair {pair_id}")
        return
    record = matches[0]
    record_path = _resolve_manifest_path(
        manifest, _record_value(record, "image_path", "training_image_path", "source_image_path", "blob_path")
    )
    pair_path = Path(str(pair.get("training_image_path")))
    if record_path is None or _path_identity(record_path) != _path_identity(pair_path):
        reasons.append(f"training source image path mismatch for pair {pair_id}")
    try:
        blob_hash = file_sha256(pair_path)
        decoded_hash = decoded_rgba_sha256(pair_path)
    except (OSError, ValueError) as error:
        reasons.append(f"training image cannot be validated for pair {pair_id}: {error}")
        return
    if blob_hash != pair.get("training_source_blob_sha256"):
        reasons.append(f"training source blob hash mismatch for pair {pair_id}")
    if decoded_hash != pair.get("training_decoded_rgba_sha256"):
        reasons.append(f"training decoded-RGBA hash mismatch for pair {pair_id}")
    if _record_value(record, "source_blob_sha256", "training_source_blob_sha256", "blob_sha256") != blob_hash:
        reasons.append(f"training manifest blob hash mismatch for pair {pair_id}")
    if _record_value(record, "decoded_rgba_sha256", "training_decoded_rgba_sha256") != decoded_hash:
        reasons.append(f"training manifest decoded hash mismatch for pair {pair_id}")


def _event_identity_reasons(event: Mapping[str, Any], pair: Mapping[str, Any], bundle: Mapping[str, Any]) -> list[str]:
    mappings = bound_event_identity(bundle, pair)
    return [field for field, expected in mappings.items() if event.get(field) != expected]


def _reported_status(machine: Mapping[str, Any]) -> Any:
    promotion = machine.get("promotion") if isinstance(machine.get("promotion"), dict) else {}
    summary = machine.get("summary") if isinstance(machine.get("summary"), dict) else {}
    memo = summary.get("memorization") if isinstance(summary.get("memorization"), dict) else {}
    return promotion.get("memorization_machine_status", memo.get("machine_status"))


def decide_promotion(
    *,
    checkpoint: Path,
    benchmark_manifest: Path,
    machine_report: Path,
    generated_report: Path,
    generated_manifest: Path,
    training_dataset_identity: str,
    training_view_identity: str,
    training_manifest: Path,
    candidate_evidence: Path,
    review_event_log: Path,
    detector_policy: Path,
    generated_images: Path | None = None,
    training_images: Path | None = None,
    detector_policy_version: str | None = None,
) -> dict[str, Any]:
    """Apply the fixed validation/recomputation/review precedence without writes."""
    not_comparable: list[str] = []
    hard_blocks: list[str] = []
    pending: list[str] = []
    expected_context: dict[str, Any] = {
        "checkpoint_path": checkpoint,
        "benchmark_manifest_path": benchmark_manifest,
        "machine_report_path": machine_report,
        "generated_report_path": generated_report,
        "generated_manifest_path": generated_manifest,
        "training_manifest_path": training_manifest,
        "detector_policy_artifact_path": detector_policy,
        "training_dataset_identity": training_dataset_identity,
        "training_view_identity": training_view_identity,
    }
    if detector_policy_version is not None:
        expected_context["detector_policy_version"] = detector_policy_version
    strict = load_candidate_bundle(candidate_evidence, expected_context=expected_context)
    not_comparable.extend(strict.reasons)
    bundle = strict.bundle
    if bundle.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        not_comparable.append("candidate evidence has an unsupported schema")

    bindings = (
        (checkpoint, "checkpoint_path", "checkpoint_sha256", "checkpoint"),
        (benchmark_manifest, "benchmark_manifest_path", "benchmark_manifest_sha256", "benchmark manifest"),
        (machine_report, "machine_report_path", "machine_report_sha256", "machine scoring report"),
        (generated_report, "generated_report_path", "generated_report_sha256", "generated report"),
        (generated_manifest, "generated_manifest_path", "generated_manifest_sha256", "generated manifest"),
        (training_manifest, "training_manifest_path", "training_manifest_sha256", "training manifest"),
        (detector_policy, "detector_policy_artifact_path", "detector_policy_artifact_sha256", "detector policy"),
    )
    actual_hashes: dict[str, str | None] = {}
    for path, path_field, hash_field, label in bindings:
        actual_hashes[hash_field] = _validate_bound_file(path, bundle, path_field, hash_field, label, not_comparable)
    _validate_policy(detector_policy, bundle, not_comparable)
    if detector_policy_version is not None and detector_policy_version != bundle.get("detector_policy_version"):
        not_comparable.append("detector policy version is incompatible")
    if bundle.get("training_dataset_identity") != training_dataset_identity:
        not_comparable.append("training dataset identity mismatch")
    if bundle.get("training_view_identity") != training_view_identity:
        not_comparable.append("training view identity mismatch")
    for root, field, label in (
        (generated_images, "generated_images_root", "generated images root"),
        (training_images, "training_images_root", "training images root"),
    ):
        if root is not None:
            if not root.is_dir():
                not_comparable.append(f"{label} is missing")
            if bundle.get(field) != str(root.resolve()):
                not_comparable.append(f"{label} identity mismatch")
    expected_bundle_hash = candidate_bundle_sha256(bundle)
    if bundle.get("candidate_evidence_sha256") != expected_bundle_hash:
        not_comparable.append("candidate evidence bundle SHA-256 mismatch")

    raw_pairs = bundle.get("pairs")
    pairs = raw_pairs if isinstance(raw_pairs, list) else []
    if not isinstance(raw_pairs, list):
        not_comparable.append("candidate evidence pairs must be a list")
    pair_order = [pair.get("pair_id") if isinstance(pair, dict) else None for pair in pairs]
    if bundle.get("candidate_order") != pair_order:
        not_comparable.append("candidate ordering contract mismatch")
    if bundle.get("candidate_count") != len(pairs):
        not_comparable.append("candidate count mismatch")
    pair_by_id: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs):
        not_comparable.extend(_pair_reasons(pair, index, bundle))
        if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
            continue
        pair_id = pair["pair_id"]
        if pair_id in pair_by_id:
            not_comparable.append(f"duplicate candidate pair: {pair_id}")
            continue
        pair_by_id[pair_id] = pair
        for root, field, label in (
            (generated_images, "generated_png_path", "generated image"),
            (training_images, "training_image_path", "training image"),
        ):
            if root is not None:
                try:
                    Path(str(pair.get(field))).resolve().relative_to(root.resolve())
                except ValueError:
                    not_comparable.append(f"{label} is outside the supplied image root for pair {pair_id}")

    machine = _load_object(machine_report, "machine scoring report", not_comparable)
    summary = machine.get("summary") if isinstance(machine.get("summary"), dict) else {}
    memo = summary.get("memorization") if isinstance(summary.get("memorization"), dict) else {}
    machine_pair_ids = memo.get("candidate_pair_ids")
    if machine_pair_ids != pair_order:
        not_comparable.append("machine report candidate set or ordering mismatch")
    classes = [pair.get("evidence_class") for pair in pairs if isinstance(pair, dict)]
    recomputed_status = recompute_memorization_status(classes)
    reported_status = _reported_status(machine)
    machine_outcome = evaluate_memorization_outcome(memo, expected_total=len(pairs))
    if machine_outcome.status in {
        MemorizationMachineStatus.INCOMPLETE,
        MemorizationMachineStatus.NOT_COMPARABLE,
    }:
        not_comparable.extend(f"machine report: {reason}" for reason in machine_outcome.reasons)
    if reported_status not in {status.value for status in MemorizationMachineStatus}:
        not_comparable.append("machine report has no controlled memorization status")
    elif reported_status != machine_outcome.status.value:
        not_comparable.append("reported machine status disagrees with validated evidence counts")
    elif reported_status != recomputed_status.value:
        not_comparable.append("reported and recomputed memorization status mismatch")
    expected_counts = {
        "hard_evidence_count": sum(value in {item.value for item in HARD_EVIDENCE_CLASSES} for value in classes),
        "review_required_count": sum(
            parse_evidence_class(value) in REVIEW_REQUIRED_EVIDENCE_CLASSES
            for value in classes
            if isinstance(value, str) and value in {item.value for item in EvidenceClass}
        ),
        "candidate_count": len(pairs),
        "warning_count": sum(
            value
            in {
                EvidenceClass.EXACT_RGBA_LOW_EVIDENCE_COLLISION.value,
                EvidenceClass.GENERIC_SPARSE_COLLISION.value,
                EvidenceClass.BLANK_COLLISION.value,
            }
            for value in classes
        ),
    }
    for field, expected in expected_counts.items():
        if memo.get(field) != expected:
            not_comparable.append(f"machine report {field} disagrees with candidate evidence")
    if memo.get("evidence_class_counts") != dict(Counter(classes)):
        not_comparable.append("machine report evidence_class_counts disagrees with candidate evidence")
    if memo.get("detector_policy_sha256") != bundle.get("detector_policy_sha256"):
        not_comparable.append("machine report detector policy mismatch")
    if memo.get("comparison_parameters_sha256") != bundle.get("comparison_parameters_sha256"):
        not_comparable.append("machine report comparison parameters mismatch")

    review_pair_ids = {
        pair_id
        for pair_id, pair in pair_by_id.items()
        if pair.get("evidence_class") in {item.value for item in REVIEW_REQUIRED_EVIDENCE_CLASSES}
    }
    expected_review_identities = {
        pair_id: bound_event_identity(bundle, pair_by_id[pair_id]) for pair_id in sorted(review_pair_ids)
    }
    replay = replay_review_events(review_event_log, expected_identities=expected_review_identities)
    review_log_bound = "review_event_log_path" in bundle or "review_event_log_sha256" in bundle
    if review_log_bound:
        if not isinstance(bundle.get("review_event_log_path"), str) or not _valid_sha256(
            bundle.get("review_event_log_sha256")
        ):
            not_comparable.append("bound review log identity is malformed")
        elif _path_identity(Path(str(bundle["review_event_log_path"]))) != _path_identity(review_event_log):
            not_comparable.append("bound review log path identity mismatch")
        elif replay.log_status == "missing":
            not_comparable.append("bound review log disappeared")
        elif review_event_log.is_file():
            try:
                if file_sha256(review_event_log) != bundle.get("review_event_log_sha256"):
                    not_comparable.append("bound review log SHA-256 mismatch")
            except OSError as error:
                not_comparable.append(f"bound review log cannot be read: {error}")
    if replay.log_status == "missing" and not review_log_bound:
        if review_pair_ids:
            pending.append("missing_review_log: review-required candidates are pending")
        else:
            contract = bundle.get("review_log_contract")
            absence_allowed = (
                isinstance(contract, dict)
                and contract.get("schema_version") == "sprite_lab_review_log_contract_v2"
                and contract.get("absence_allowed_when_no_review_candidates") is True
            )
            if not absence_allowed:
                not_comparable.append("missing optional review log is not allowed by the versioned contract")
    elif replay.log_status in {"unreadable", "malformed"}:
        not_comparable.extend(replay.invalid_reasons)
    else:
        not_comparable.extend(replay.invalid_reasons)
    if replay.legacy_events:
        not_comparable.append("historical reviews are unbound legacy events without promotion authority")
    extra_reviews = replay.seen_pair_ids - review_pair_ids
    if extra_reviews:
        not_comparable.append(f"reviews exist for non-review-required pairs: {', '.join(sorted(extra_reviews))}")

    cleared_pairs: list[str] = []
    blocked_pairs: list[str] = []
    pending_pairs: list[str] = []
    for pair_id, pair in pair_by_id.items():
        try:
            evidence_class = parse_evidence_class(pair.get("evidence_class"))
        except ValueError:
            continue
        if evidence_class in HARD_EVIDENCE_CLASSES:
            hard_blocks.append(f"verified nontrivial exact decoded-RGBA match: {pair_id}")
            blocked_pairs.append(pair_id)
            continue
        if evidence_class not in REVIEW_REQUIRED_EVIDENCE_CLASSES:
            continue
        chain = replay.chains.get(pair_id)
        event = replay.current.get(pair_id)
        if event is None:
            if chain is None or chain.chain_status == "missing":
                pending.append(f"missing required authoritative review: {pair_id}")
            pending_pairs.append(pair_id)
            continue
        mismatches = _event_identity_reasons(event, pair, bundle)
        if mismatches:
            not_comparable.append(f"review identity mismatch for pair {pair_id}: {', '.join(mismatches)}")
            continue
        outcome = str(event["review_outcome"])
        if outcome in HARD_BLOCK_OUTCOMES:
            hard_blocks.append(f"human review found same sprite or memorization: {pair_id}")
            blocked_pairs.append(pair_id)
        elif outcome in PENDING_OUTCOMES:
            pending.append(f"review remains uncertain: {pair_id}")
            pending_pairs.append(pair_id)
        elif outcome in CLEARING_OUTCOMES:
            cleared_pairs.append(pair_id)

    promotion = machine.get("promotion") if isinstance(machine.get("promotion"), dict) else {}
    checks = promotion.get("checks") if isinstance(promotion.get("checks"), dict) else None
    unrelated_machine_failures: list[str] = []
    if checks is None:
        not_comparable.append("machine scoring report has no bound gate checks")
    else:
        unrelated_machine_failures = sorted(
            name for name, passed in checks.items() if name not in MEMORIZATION_CHECKS and passed is not True
        )
        hard_blocks.extend(f"unrelated machine-gate failure: {name}" for name in unrelated_machine_failures)

    not_comparable = sorted(set(not_comparable))
    hard_blocks = sorted(set(hard_blocks))
    pending = sorted(set(pending))
    identity_valid = not not_comparable
    review_complete = identity_valid and not pending and review_pair_ids == set(cleared_pairs) | set(blocked_pairs)
    if not identity_valid or hard_blocks:
        decision = "blocked"
    elif pending:
        decision = "manual_review_required"
    elif review_complete and recomputed_status in {
        MemorizationMachineStatus.PASS,
        MemorizationMachineStatus.MANUAL_REVIEW_REQUIRED,
    }:
        decision = "eligible"
    else:
        decision = "blocked"
    try:
        candidate_file_hash = file_sha256(candidate_evidence)
    except OSError:
        candidate_file_hash = None
    review_log_identity: dict[str, Any] = {
        "path": str(review_event_log.resolve()),
        "state": replay.log_status,
        "sha256": None,
    }
    if review_event_log.is_file():
        try:
            review_log_identity["sha256"] = file_sha256(review_event_log)
        except OSError:
            review_log_identity["state"] = "unreadable"
    result: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "classification": "not_comparable"
        if not identity_valid
        else ("comparable" if decision == "eligible" else "promotion_hold"),
        "eligible_for_promotion": decision == "eligible",
        "machine_gates_passed": not unrelated_machine_failures,
        "reported_machine_status": reported_status,
        "validated_machine_status": machine_outcome.status.value,
        "recomputed_machine_status": recomputed_status.value,
        "identity_valid": identity_valid,
        "review_set_complete": review_complete,
        "hard_block_reasons": hard_blocks,
        "pending_review_reasons": pending,
        "not_comparable_reasons": not_comparable,
        "cleared_pairs": sorted(set(cleared_pairs)),
        "blocked_pairs": sorted(set(blocked_pairs)),
        "pending_pairs": sorted(set(pending_pairs)),
        "review_chain_statuses": {pair_id: replay.chains[pair_id].chain_status for pair_id in sorted(replay.chains)},
        "review_log_status": replay.log_status,
        "legacy_review_event_count": len(replay.legacy_events),
        "candidate_evidence_sha256": bundle.get("candidate_evidence_sha256"),
        "input_bundle_sha256": canonical_sha256(
            {
                "candidate_evidence_file_sha256": candidate_file_hash,
                "review_event_log": review_log_identity,
                **actual_hashes,
            }
        ),
        "checkpoint_copies": 0,
        "promotion_actions": 0,
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Memorization promotion decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Classification: `{decision['classification']}`",
        f"- Eligible for promotion: `{str(decision['eligible_for_promotion']).lower()}`",
        f"- Reported machine status: `{decision['reported_machine_status']}`",
        f"- Recomputed machine status: `{decision['recomputed_machine_status']}`",
        f"- Identity valid: `{str(decision['identity_valid']).lower()}`",
        f"- Review set complete: `{str(decision['review_set_complete']).lower()}`",
        f"- Decision SHA-256: `{decision['decision_sha256']}`",
    ]
    for title, key in (
        ("Hard blocks", "hard_block_reasons"),
        ("Pending reviews", "pending_review_reasons"),
        ("Not comparable", "not_comparable_reasons"),
    ):
        lines.extend(("", f"## {title}", ""))
        values = decision[key]
        lines.extend(f"- {value}" for value in values) if values else lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_decision_artifacts(output_dir: Path, decision: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "promotion_decision.json"
    markdown_path = output_dir / "promotion_decision.md"
    json_path.write_bytes(json.dumps(decision, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    markdown_path.write_text(decision_markdown(decision), encoding="utf-8", newline="\n")
    return json_path, markdown_path
