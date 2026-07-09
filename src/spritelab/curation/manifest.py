"""JSONL curation decisions for SpriteBundle datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = ("accepted", "rejected", "quarantine", "needs_fix")
ALLOWED_REASONS = (
    "bad_alpha",
    "bad_palette",
    "bad_roles",
    "duplicate",
    "copyright_risky",
    "too_noisy",
    "too_empty",
    "wrong_category",
    "low_readability",
    "bad_silhouette",
    "bad_metadata",
    "bad_source",
    "not_pixel_art",
    "wrong_size",
    "other",
)
CURATION_FILENAME = "curation.jsonl"

_SPRITE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CurationDecision:
    """One immutable human curation decision event."""

    sprite_id: str
    status: str
    tags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    notes: str = ""
    timestamp: str = ""
    reviewer: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        sprite_id = _validate_sprite_id(self.sprite_id)
        status = _validate_status(self.status)
        tags = _normalize_tokens(self.tags, field_name="tags")
        reasons = _normalize_tokens(self.reasons, field_name="reasons")
        unknown_reasons = [reason for reason in reasons if reason not in ALLOWED_REASONS]
        if unknown_reasons:
            joined = ", ".join(unknown_reasons)
            raise ValueError(f"unknown curation reason value(s): {joined}")

        notes = self.notes if isinstance(self.notes, str) else str(self.notes)
        timestamp = self.timestamp or _utc_timestamp()
        reviewer = self.reviewer if self.reviewer is None or isinstance(self.reviewer, str) else str(self.reviewer)
        source_path = (
            self.source_path if self.source_path is None or isinstance(self.source_path, str) else str(self.source_path)
        )

        object.__setattr__(self, "sprite_id", sprite_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "source_path", source_path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for one JSONL line."""

        data = asdict(self)
        data["tags"] = list(self.tags)
        data["reasons"] = list(self.reasons)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CurationDecision:
        """Build a decision from JSON data, tolerating missing optional fields."""

        if not isinstance(data, Mapping):
            raise ValueError("curation decision line must be a JSON object.")
        return cls(
            sprite_id=str(data["sprite_id"]),
            status=str(data["status"]),
            tags=_coerce_sequence(data.get("tags", ()), field_name="tags"),
            reasons=_coerce_sequence(data.get("reasons", ()), field_name="reasons"),
            notes=data.get("notes", ""),
            timestamp=str(data.get("timestamp", "")),
            reviewer=data.get("reviewer"),
            source_path=data.get("source_path"),
        )


@dataclass(frozen=True)
class CurationSummary:
    """Summary counts for curation events and latest decisions."""

    total_event_count: int
    latest_unique_sprite_count: int
    count_by_status: dict[str, int]
    count_by_reason: dict[str, int]
    count_by_tag: dict[str, int]
    accepted_count: int
    rejected_count: int
    quarantine_count: int
    needs_fix_count: int


@dataclass(frozen=True)
class CurationValidationResult:
    """Comparison between latest curation decisions and discovered bundles."""

    unknown_curated_sprite_ids: tuple[str, ...]
    uncurated_bundle_ids: tuple[str, ...]
    collision_issues: tuple[str, ...]
    accepted_bundle_ids: tuple[str, ...]
    rejected_bundle_ids: tuple[str, ...]
    quarantine_bundle_ids: tuple[str, ...]
    needs_fix_bundle_ids: tuple[str, ...]


class _BundleIdMap(dict[str, Path]):
    def __init__(self, *args: Any, collisions: Mapping[str, Sequence[Path]] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.collisions: dict[str, tuple[Path, ...]] = {key: tuple(paths) for key, paths in (collisions or {}).items()}


def load_curation_events(path: str | Path) -> list[CurationDecision]:
    """Load all decision events from a JSONL curation file."""

    input_path = Path(path)
    if not input_path.exists():
        return []

    decisions: list[CurationDecision] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in curation file on line {line_number}: {exc}") from exc
        try:
            decisions.append(CurationDecision.from_dict(data))
        except Exception as exc:
            raise ValueError(f"Invalid curation decision on line {line_number}: {exc}") from exc
    return decisions


def append_curation_decision(path: str | Path, decision: CurationDecision) -> None:
    """Append one decision event to a UTF-8 JSONL curation file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(decision.to_dict(), sort_keys=True) + "\n")


def write_curation_events(path: str | Path, decisions: Sequence[CurationDecision]) -> None:
    """Rewrite a curation JSONL file from decision events."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(decision.to_dict(), sort_keys=True) for decision in decisions]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_latest_curation(path: str | Path) -> dict[str, CurationDecision]:
    """Load latest curation state, where later events override earlier events."""

    latest: dict[str, CurationDecision] = {}
    for decision in load_curation_events(path):
        latest[decision.sprite_id] = decision
    return latest


def summarize_curation(
    decisions: Mapping[str, CurationDecision] | Sequence[CurationDecision],
) -> CurationSummary:
    """Summarize latest curation status, reasons, and tags."""

    if isinstance(decisions, Mapping):
        total_event_count = len(decisions)
        latest = dict(decisions)
    else:
        total_event_count = len(decisions)
        latest = {decision.sprite_id: decision for decision in decisions}

    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    for decision in latest.values():
        status_counter.update([decision.status])
        reason_counter.update(decision.reasons)
        tag_counter.update(decision.tags)

    return CurationSummary(
        total_event_count=total_event_count,
        latest_unique_sprite_count=len(latest),
        count_by_status=dict(sorted(status_counter.items())),
        count_by_reason=dict(sorted(reason_counter.items())),
        count_by_tag=dict(sorted(tag_counter.items())),
        accepted_count=status_counter.get("accepted", 0),
        rejected_count=status_counter.get("rejected", 0),
        quarantine_count=status_counter.get("quarantine", 0),
        needs_fix_count=status_counter.get("needs_fix", 0),
    )


def format_curation_summary(summary: CurationSummary) -> str:
    """Render a concise human-readable curation summary."""

    lines = [
        "Curation summary",
        "----------------",
        f"Unique sprites: {summary.latest_unique_sprite_count}",
        f"Events: {summary.total_event_count}",
        "",
        "By status:",
    ]
    for status in ALLOWED_STATUSES:
        lines.append(f"  {status}: {summary.count_by_status.get(status, 0)}")
    lines.extend(["", "Top reasons:"])
    lines.extend(_counter_lines(summary.count_by_reason))
    lines.extend(["", "Top tags:"])
    lines.extend(_counter_lines(summary.count_by_tag))
    return "\n".join(lines)


def discover_bundle_ids(bundle_root: str | Path) -> dict[str, Path]:
    """Discover bundle directories keyed by metadata ID or directory name."""

    root = Path(bundle_root)
    if not root.exists():
        raise FileNotFoundError(f"bundle root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"bundle root is not a directory: {root}")

    bundle_dirs = _bundle_directories(root)
    grouped: dict[str, list[Path]] = {}
    for bundle_dir in bundle_dirs:
        sprite_id = _bundle_id_from_metadata(bundle_dir)
        grouped.setdefault(sprite_id, []).append(bundle_dir)

    unique = {sprite_id: paths[0] for sprite_id, paths in sorted(grouped.items())}
    collisions = {sprite_id: paths for sprite_id, paths in grouped.items() if len(paths) > 1}
    return _BundleIdMap(unique, collisions=collisions)


def validate_curation_against_bundles(
    curation: Mapping[str, CurationDecision],
    bundle_ids: Mapping[str, Path],
) -> CurationValidationResult:
    """Compare latest curation decisions against discovered bundle IDs."""

    curated_ids = set(curation)
    discovered_ids = set(bundle_ids)
    unknown = tuple(sorted(curated_ids - discovered_ids))
    uncurated = tuple(sorted(discovered_ids - curated_ids))
    collisions = tuple(_collision_messages(getattr(bundle_ids, "collisions", {})))

    def ids_for_status(status: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                sprite_id
                for sprite_id, decision in curation.items()
                if sprite_id in discovered_ids and decision.status == status
            )
        )

    return CurationValidationResult(
        unknown_curated_sprite_ids=unknown,
        uncurated_bundle_ids=uncurated,
        collision_issues=collisions,
        accepted_bundle_ids=ids_for_status("accepted"),
        rejected_bundle_ids=ids_for_status("rejected"),
        quarantine_bundle_ids=ids_for_status("quarantine"),
        needs_fix_bundle_ids=ids_for_status("needs_fix"),
    )


def _validate_sprite_id(sprite_id: object) -> str:
    if not isinstance(sprite_id, str) or not sprite_id.strip() or not _SPRITE_ID_RE.fullmatch(sprite_id):
        raise ValueError("sprite_id must be a non-empty filesystem-safe identifier.")
    return sprite_id


def _validate_status(status: object) -> str:
    if not isinstance(status, str):
        raise ValueError("status must be a string.")
    normalized = status.strip().lower()
    if normalized not in ALLOWED_STATUSES:
        allowed = ", ".join(ALLOWED_STATUSES)
        raise ValueError(f"status must be one of: {allowed}.")
    return normalized


def _normalize_tokens(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_token(value)
        if not token:
            raise ValueError(f"{field_name} cannot contain empty values.")
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    return tuple(normalized)


def _normalize_token(value: object) -> str:
    if not isinstance(value, str):
        value = str(value)
    token = _WHITESPACE_RE.sub("_", value.strip().lower())
    return token.strip("_")


def _coerce_sequence(value: object, *, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    raise ValueError(f"{field_name} must be a sequence.")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _counter_lines(counts: Mapping[str, int], limit: int = 10) -> list[str]:
    if not counts:
        return ["  none"]
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [f"  {key}: {value}" for key, value in ordered]


def _bundle_directories(root: Path) -> list[Path]:
    if _is_bundle_dir(root):
        return [root]

    search_root = root / "bundles" if (root / "bundles").is_dir() else root
    return sorted(
        {path.parent for path in search_root.rglob("bundle.npz") if (path.parent / "metadata.json").exists()},
        key=lambda path: str(path).lower(),
    )


def _is_bundle_dir(path: Path) -> bool:
    return (path / "bundle.npz").exists() and (path / "metadata.json").exists()


def _bundle_id_from_metadata(bundle_dir: Path) -> str:
    metadata_path = bundle_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return bundle_dir.name
    sprite_id = metadata.get("id")
    return str(sprite_id) if sprite_id else bundle_dir.name


def _collision_messages(collisions: Mapping[str, Sequence[Path]]) -> list[str]:
    messages: list[str] = []
    for sprite_id, paths in sorted(collisions.items()):
        joined = ", ".join(str(path) for path in paths)
        messages.append(f"bundle id {sprite_id} maps to multiple paths: {joined}")
    return messages


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage SpriteBundle curation JSONL files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Print curation summary.")
    summary.add_argument("--curation", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="Validate curation decisions against bundles.")
    validate.add_argument("--bundles", required=True, type=Path)
    validate.add_argument("--curation", required=True, type=Path)

    decide = subparsers.add_parser("decide", help="Append one curation decision.")
    decide.add_argument("--curation", required=True, type=Path)
    decide.add_argument("--sprite-id", required=True)
    decide.add_argument("--status", required=True, choices=ALLOWED_STATUSES)
    decide.add_argument("--tag", action="append", default=[])
    decide.add_argument("--reason", action="append", default=[])
    decide.add_argument("--notes", default="")
    decide.add_argument("--reviewer")
    decide.add_argument("--source-path")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "summary":
        summary = summarize_curation(load_curation_events(args.curation))
        print(format_curation_summary(summary))
        return

    if args.command == "validate":
        latest = load_latest_curation(args.curation)
        bundle_ids = discover_bundle_ids(args.bundles)
        result = validate_curation_against_bundles(latest, bundle_ids)
        print(f"Bundle IDs: {len(bundle_ids)}")
        print(f"Curated IDs: {len(latest)}")
        print(f"Unknown curated IDs: {len(result.unknown_curated_sprite_ids)}")
        print(f"Uncurated bundle IDs: {len(result.uncurated_bundle_ids)}")
        print(f"Collision issues: {len(result.collision_issues)}")
        return

    if args.command == "decide":
        decision = CurationDecision(
            sprite_id=args.sprite_id,
            status=args.status,
            tags=tuple(args.tag),
            reasons=tuple(args.reason),
            notes=args.notes,
            reviewer=args.reviewer,
            source_path=args.source_path,
        )
        append_curation_decision(args.curation, decision)
        print(f"Appended {decision.status} decision for {decision.sprite_id} to {args.curation}")
        return

    raise ValueError(f"unknown curation command: {args.command}")


if __name__ == "__main__":
    main()
