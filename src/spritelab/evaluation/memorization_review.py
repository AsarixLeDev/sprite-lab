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

import numpy as np
from PIL import Image

from spritelab.evaluation.memorization import reconstruct_rgba
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
        "generated_sample_id",
        "prompt_id",
        "seed",
        "generated_png_sha256",
        "generated_decoded_rgba_sha256",
        "training_dataset_identity",
        "training_manifest_path",
        "training_manifest_sha256",
        "training_source_sprite_id",
        "training_row_or_index",
        "training_decoded_rgba_sha256",
        "detector_policy_version",
        "comparison_method",
        "comparison_parameters_sha256",
        "candidate_evidence_sha256",
    }
)


@dataclass(frozen=True)
class ReviewReplay:
    """Fail-closed replay result for an append-only review-event log."""

    current: dict[str, dict[str, Any]]
    legacy_events: tuple[dict[str, Any], ...]
    invalid_reasons: tuple[str, ...]
    seen_pair_ids: frozenset[str]


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
    try:
        timestamp = datetime.fromisoformat(str(event.get("created_at_utc", "")).replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError
    except ValueError:
        reasons.append(f"line {line_number}: created_at_utc must be an aware UTC timestamp")
    for field in sorted(
        name for name in BOUND_REVIEW_REQUIRED_FIELDS if name.endswith("_sha256") and name != "previous_event_sha256"
    ):
        if not _valid_sha256(event.get(field)):
            reasons.append(f"line {line_number}: invalid {field}")
    previous = event.get("previous_event_sha256")
    if previous is not None and not _valid_sha256(previous):
        reasons.append(f"line {line_number}: invalid previous_event_sha256")
    if "event_sha256" in event and event.get("event_sha256") != review_event_sha256(event):
        reasons.append(f"line {line_number}: invalid event_sha256")
    return reasons


def replay_review_events(results_path: Path) -> ReviewReplay:
    """Replay v2 chains without allowing malformed, legacy, or competing rows to win.

    Historical v1 rows are returned for display with ``promotion_authority=false``
    and ``identity_status=unbound_legacy``. They are never current decisions.
    """
    if not results_path.is_file():
        return ReviewReplay({}, (), ("review event log is missing",), frozenset())
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return ReviewReplay({}, (), (f"review event log cannot be read: {error}",), frozenset())
    events_by_pair: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    legacy: list[dict[str, Any]] = []
    invalid: list[str] = []
    seen_event_ids: set[str] = set()
    seen_pairs: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            invalid.append(f"line {line_number}: malformed JSON: {error.msg}")
            continue
        if not isinstance(raw, dict):
            invalid.append(f"line {line_number}: review event must be an object")
            continue
        pair_id = raw.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            seen_pairs.add(pair_id)
        if raw.get("schema_version") == SCHEMA_VERSION:
            legacy.append({**raw, "promotion_authority": False, "identity_status": "unbound_legacy"})
            continue
        row_reasons = _validate_bound_event(raw, line_number)
        event_id = raw.get("event_id")
        if isinstance(event_id, str) and event_id in seen_event_ids:
            row_reasons.append(f"line {line_number}: duplicate event_id")
        if isinstance(event_id, str):
            seen_event_ids.add(event_id)
        if row_reasons:
            invalid.extend(row_reasons)
            continue
        events_by_pair.setdefault(str(pair_id), []).append((line_number, dict(raw)))

    current: dict[str, dict[str, Any]] = {}
    for pair_id, numbered_events in sorted(events_by_pair.items()):
        by_revision: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for item in numbered_events:
            by_revision.setdefault(int(item[1]["revision"]), []).append(item)
        pair_reasons: list[str] = []
        for revision, competing in sorted(by_revision.items()):
            if len(competing) > 1:
                pair_reasons.append(f"pair {pair_id}: competing events for revision {revision}")
        revisions = sorted(by_revision)
        if revisions and revisions != list(range(1, revisions[-1] + 1)):
            pair_reasons.append(f"pair {pair_id}: revision gap")
        preceding_hash: str | None = None
        if not pair_reasons:
            for revision in revisions:
                event = by_revision[revision][0][1]
                previous = event.get("previous_event_sha256")
                if revision == 1:
                    if previous is not None:
                        pair_reasons.append(f"pair {pair_id}: revision 1 must use null genesis hash")
                        break
                elif previous != preceding_hash:
                    pair_reasons.append(f"pair {pair_id}: invalid previous-event hash at revision {revision}")
                    break
                preceding_hash = review_event_sha256(event)
        if pair_reasons:
            invalid.extend(pair_reasons)
            continue
        if revisions:
            event = dict(by_revision[revisions[-1]][0][1])
            event["event_sha256"] = review_event_sha256(event)
            current[pair_id] = event
    return ReviewReplay(current, tuple(legacy), tuple(invalid), frozenset(seen_pairs))


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
