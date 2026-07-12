"""Read-only, fail-closed checkpoint promotion decisions.

This module consumes detector output; it does not detect or score memorization.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.evaluation.memorization_review import (
    canonical_sha256,
    replay_review_events,
)

DECISION_SCHEMA_VERSION = "sprite_lab_promotion_decision_v1"
CANDIDATE_SCHEMA_VERSION = "sprite_lab_memorization_candidate_evidence_v1"
CLEARING_OUTCOMES = frozenset({"different_sprite", "common_generic_shape", "likely_false_positive"})
HARD_BLOCK_OUTCOMES = frozenset({"same_sprite_or_memorized"})
PENDING_OUTCOMES = frozenset({"uncertain"})
EXACT_RGBA_EVIDENCE_CLASS = "exact_decoded_rgba"
BLANK_COLLISION_EVIDENCE_CLASSES = frozenset({"blank_collision", "near_blank_collision"})


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_rgba_sha256(path: Path) -> str:
    """Hash canonical decoded pixels as width, height, then row-major RGBA bytes."""
    with Image.open(path) as image:
        image.load()
        rgba = image.convert("RGBA")
        dimensions = rgba.width.to_bytes(8, "big") + rgba.height.to_bytes(8, "big")
        return sha256(dimensions + rgba.tobytes()).hexdigest()


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


def _path_identity(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def _validate_bound_file(
    *,
    actual_path: Path,
    evidence: dict[str, Any],
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


def _candidate_pair_reasons(pair: Any, index: int) -> list[str]:
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
        "training_source_sprite_id",
        "training_row_or_index",
        "training_decoded_rgba_sha256",
        "evidence_class",
        "exact_rgba",
    }
    missing = sorted(required - pair.keys())
    reasons = [f"candidate pair {index} missing fields: {', '.join(missing)}"] if missing else []
    if not isinstance(pair.get("pair_id"), str) or not pair.get("pair_id"):
        reasons.append(f"candidate pair {index} has invalid pair_id")
    if not isinstance(pair.get("exact_rgba"), bool):
        reasons.append(f"candidate pair {index} exact_rgba must be boolean")
    evidence_class = pair.get("evidence_class")
    if not isinstance(evidence_class, str) or not evidence_class:
        reasons.append(f"candidate pair {index} has no explicit evidence_class")
    if pair.get("exact_rgba") and evidence_class not in (
        {EXACT_RGBA_EVIDENCE_CLASS} | BLANK_COLLISION_EVIDENCE_CLASSES
    ):
        reasons.append(f"candidate pair {index} silently downgrades an exact-RGBA match")
    return reasons


def _event_identity_reasons(event: dict[str, Any], pair: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    mappings = {
        "checkpoint_path": evidence.get("checkpoint_path"),
        "checkpoint_sha256": evidence.get("checkpoint_sha256"),
        "benchmark_manifest_path": evidence.get("benchmark_manifest_path"),
        "benchmark_manifest_sha256": evidence.get("benchmark_manifest_sha256"),
        "generated_report_path": evidence.get("generated_report_path"),
        "generated_report_sha256": evidence.get("generated_report_sha256"),
        "generated_sample_id": pair.get("generated_sample_id"),
        "prompt_id": pair.get("prompt_id"),
        "seed": pair.get("seed"),
        "generated_png_sha256": pair.get("generated_png_sha256"),
        "generated_decoded_rgba_sha256": pair.get("generated_decoded_rgba_sha256"),
        "training_dataset_identity": evidence.get("training_dataset_identity"),
        "training_manifest_path": evidence.get("training_manifest_path"),
        "training_manifest_sha256": evidence.get("training_manifest_sha256"),
        "training_source_sprite_id": pair.get("training_source_sprite_id"),
        "training_row_or_index": pair.get("training_row_or_index"),
        "training_decoded_rgba_sha256": pair.get("training_decoded_rgba_sha256"),
        "detector_policy_version": evidence.get("detector_policy_version"),
        "comparison_method": evidence.get("comparison_method"),
        "comparison_parameters_sha256": evidence.get("comparison_parameters_sha256"),
        "candidate_evidence_sha256": canonical_sha256(pair),
    }
    if "noise_seed" in pair:
        mappings["noise_seed"] = pair.get("noise_seed")
    if evidence.get("generated_manifest_path") is not None:
        mappings["generated_manifest_sha256"] = evidence.get("generated_manifest_sha256")
    return [field for field, expected in mappings.items() if event.get(field) != expected]


def decide_promotion(
    *,
    checkpoint: Path,
    benchmark_manifest: Path,
    machine_report: Path,
    generated_report: Path,
    generated_manifest: Path | None,
    training_dataset_identity: str,
    training_manifest: Path,
    candidate_evidence: Path,
    review_event_log: Path,
    detector_policy_version: str | None = None,
) -> dict[str, Any]:
    """Produce a deterministic fail-closed decision without mutating any input."""
    not_comparable: list[str] = []
    hard_blocks: list[str] = []
    pending: list[str] = []
    evidence = _load_object(candidate_evidence, "candidate evidence", not_comparable)
    if evidence.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        not_comparable.append("candidate evidence has an unsupported schema")

    actual_hashes: dict[str, str | None] = {}
    bindings = (
        (checkpoint, "checkpoint_path", "checkpoint_sha256", "checkpoint"),
        (benchmark_manifest, "benchmark_manifest_path", "benchmark_manifest_sha256", "benchmark manifest"),
        (machine_report, "machine_report_path", "machine_report_sha256", "machine scoring report"),
        (generated_report, "generated_report_path", "generated_report_sha256", "generated report"),
        (training_manifest, "training_manifest_path", "training_manifest_sha256", "training manifest"),
    )
    for path, path_field, hash_field, label in bindings:
        actual_hashes[hash_field] = _validate_bound_file(
            actual_path=path,
            evidence=evidence,
            path_field=path_field,
            hash_field=hash_field,
            label=label,
            reasons=not_comparable,
        )
    if generated_manifest is not None:
        actual_hashes["generated_manifest_sha256"] = _validate_bound_file(
            actual_path=generated_manifest,
            evidence=evidence,
            path_field="generated_manifest_path",
            hash_field="generated_manifest_sha256",
            label="generated manifest",
            reasons=not_comparable,
        )
    elif evidence.get("generated_manifest_path") is not None:
        not_comparable.append("bound generated manifest input is missing")
    if evidence.get("training_dataset_identity") != training_dataset_identity:
        not_comparable.append("training dataset identity mismatch")

    policy = evidence.get("detector_policy_version")
    if not isinstance(policy, str) or not policy:
        not_comparable.append("candidate evidence has no detector policy version")
    if detector_policy_version is not None and policy != detector_policy_version:
        not_comparable.append("detector policy version is incompatible")
    if not isinstance(evidence.get("comparison_method"), str) or not evidence.get("comparison_method"):
        not_comparable.append("candidate evidence has no comparison method")
    parameters_hash = evidence.get("comparison_parameters_sha256")
    if (
        not isinstance(parameters_hash, str)
        or len(parameters_hash) != 64
        or any(character not in "0123456789abcdef" for character in parameters_hash)
    ):
        not_comparable.append("candidate evidence has an invalid comparison-parameters hash")

    replay = replay_review_events(review_event_log)
    not_comparable.extend(replay.invalid_reasons)
    if replay.legacy_events:
        not_comparable.append("historical reviews are unbound legacy events without promotion authority")

    raw_pairs = evidence.get("pairs")
    pairs = raw_pairs if isinstance(raw_pairs, list) else []
    if not isinstance(raw_pairs, list):
        not_comparable.append("candidate evidence pairs must be a list")
    pair_by_id: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs):
        not_comparable.extend(_candidate_pair_reasons(pair, index))
        if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
            continue
        pair_id = str(pair["pair_id"])
        if pair_id in pair_by_id:
            not_comparable.append(f"duplicate candidate pair: {pair_id}")
            continue
        pair_by_id[pair_id] = pair
        generated_path = pair.get("generated_png_path")
        if isinstance(generated_path, str):
            image_path = Path(generated_path)
            try:
                if file_sha256(image_path) != pair.get("generated_png_sha256"):
                    not_comparable.append(f"generated image file hash mismatch for pair {pair_id}")
                if decoded_rgba_sha256(image_path) != pair.get("generated_decoded_rgba_sha256"):
                    not_comparable.append(f"generated decoded-RGBA hash mismatch for pair {pair_id}")
            except (OSError, ValueError) as error:
                not_comparable.append(f"generated image cannot be validated for pair {pair_id}: {error}")
        training_path = pair.get("training_image_path")
        if training_path is not None:
            try:
                if decoded_rgba_sha256(Path(str(training_path))) != pair.get("training_decoded_rgba_sha256"):
                    not_comparable.append(f"training decoded-RGBA hash mismatch for pair {pair_id}")
            except (OSError, ValueError) as error:
                not_comparable.append(f"training image cannot be validated for pair {pair_id}: {error}")

    current_pair_ids = set(pair_by_id)
    extra_reviews = replay.seen_pair_ids - current_pair_ids
    if extra_reviews:
        not_comparable.append(f"review pairs absent from current candidate set: {', '.join(sorted(extra_reviews))}")

    cleared_pairs: list[str] = []
    blocked_pairs: list[str] = []
    pending_pairs: list[str] = []
    for pair_id, pair in sorted(pair_by_id.items()):
        if pair.get("exact_rgba") and pair.get("evidence_class") == EXACT_RGBA_EVIDENCE_CLASS:
            hard_blocks.append(f"verified nontrivial exact decoded-RGBA match: {pair_id}")
            blocked_pairs.append(pair_id)
        event = replay.current.get(pair_id)
        if event is None:
            pending.append(f"missing required authoritative review: {pair_id}")
            pending_pairs.append(pair_id)
            continue
        mismatches = _event_identity_reasons(event, pair, evidence)
        if mismatches:
            not_comparable.append(f"review identity mismatch for pair {pair_id}: {', '.join(mismatches)}")
            continue
        outcome = str(event["review_outcome"])
        if outcome in HARD_BLOCK_OUTCOMES:
            hard_blocks.append(f"human review found same sprite or memorization: {pair_id}")
            if pair_id not in blocked_pairs:
                blocked_pairs.append(pair_id)
        elif outcome in PENDING_OUTCOMES:
            pending.append(f"review remains uncertain: {pair_id}")
            if pair_id not in pending_pairs:
                pending_pairs.append(pair_id)
        elif outcome in CLEARING_OUTCOMES:
            cleared_pairs.append(pair_id)
        else:  # The parser already rejects this, retained as defense in depth.
            not_comparable.append(f"unknown review outcome for pair {pair_id}")

    machine = _load_object(machine_report, "machine scoring report", not_comparable)
    benchmark = _load_object(benchmark_manifest, "benchmark manifest", not_comparable)
    cases = benchmark.get("cases")
    seeds = benchmark.get("seeds")
    machine_summary = machine.get("summary")
    if isinstance(cases, list) and isinstance(seeds, list) and cases and seeds and isinstance(machine_summary, dict):
        expected_sample_count = len(cases) * len(seeds)
        actual_sample_count = machine_summary.get("sample_count")
        if actual_sample_count != expected_sample_count:
            not_comparable.append(
                "incomplete full-suite evidence: "
                f"machine report contains {actual_sample_count!r} of {expected_sample_count} expected samples"
            )
    promotion = machine.get("promotion") if isinstance(machine.get("promotion"), dict) else machine
    machine_gate_value = promotion.get("pass") if isinstance(promotion, dict) else None
    machine_gates_passed = machine_gate_value is True
    if not isinstance(machine_gate_value, bool):
        not_comparable.append("machine scoring report has no boolean promotion pass result")
    elif not machine_gate_value:
        hard_blocks.append("existing machine-gate failure")

    not_comparable = sorted(set(not_comparable))
    hard_blocks = sorted(set(hard_blocks))
    pending = sorted(set(pending))
    identity_valid = not not_comparable
    review_complete = identity_valid and not pending and len(replay.current) == len(pair_by_id)
    if not identity_valid or hard_blocks:
        decision = "blocked"
    elif pending:
        decision = "manual_review_required"
    elif machine_gates_passed and review_complete:
        decision = "eligible"
    else:
        decision = "blocked"
    eligible = decision == "eligible"

    input_bundle = {
        "candidate_evidence_sha256": file_sha256(candidate_evidence) if candidate_evidence.is_file() else None,
        "review_event_log_sha256": file_sha256(review_event_log) if review_event_log.is_file() else None,
        **actual_hashes,
    }
    result: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "classification": "not_comparable" if not identity_valid else ("comparable" if eligible else "promotion_hold"),
        "eligible_for_promotion": eligible,
        "machine_gates_passed": machine_gates_passed,
        "identity_valid": identity_valid,
        "review_set_complete": review_complete,
        "hard_block_reasons": hard_blocks,
        "pending_review_reasons": pending,
        "not_comparable_reasons": not_comparable,
        "cleared_pairs": sorted(set(cleared_pairs)),
        "blocked_pairs": sorted(set(blocked_pairs)),
        "pending_pairs": sorted(set(pending_pairs)),
        "legacy_review_event_count": len(replay.legacy_events),
        "input_bundle_sha256": canonical_sha256(input_bundle),
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def decision_markdown(decision: dict[str, Any]) -> str:
    """Render a stable human-readable companion to the JSON decision."""
    lines = [
        "# Memorization promotion decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Classification: `{decision['classification']}`",
        f"- Eligible for promotion: `{str(decision['eligible_for_promotion']).lower()}`",
        f"- Machine gates passed: `{str(decision['machine_gates_passed']).lower()}`",
        f"- Identity valid: `{str(decision['identity_valid']).lower()}`",
        f"- Review set complete: `{str(decision['review_set_complete']).lower()}`",
        f"- Input bundle SHA-256: `{decision['input_bundle_sha256']}`",
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
    """Create a new output directory and write only decision artifacts."""
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "promotion_decision.json"
    markdown_path = output_dir / "promotion_decision.md"
    json_path.write_bytes(json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    markdown_path.write_text(decision_markdown(decision), encoding="utf-8", newline="\n")
    return json_path, markdown_path
