"""Versioned CLI surface for raw Dataset-v5 audits and provider gates."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.dataset_v5.audits import (
    audit_label_batch,
    audit_label_health,
    audit_random_sample,
    verify_label_drift,
    verify_no_name_leakage,
)
from spritelab.dataset_v5.sol import SOL_MODEL_UNAVAILABLE, SolConfig, SolModelUnavailable, SolProvider

RAW_COMMANDS = frozenset(
    {
        "audit-label-batch",
        "audit-label-health",
        "audit-random-sample",
        "inventory-raw-forensics",
        "inventory-raw-sources",
        "rebuild-from-raw",
        "sol-canary",
        "verify-frozen-rebuild",
        "verify-label-drift",
        "verify-no-name-leakage",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.dataset_v5")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("audit-label-batch")
    batch.add_argument("--records", type=Path, required=True)
    batch.add_argument("--batch-id", required=True)
    batch.add_argument("--provenance", type=Path, required=True)
    batch.add_argument("--relation-manifest", type=Path, required=True)
    batch.add_argument("--split-map", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)

    health = sub.add_parser("audit-label-health")
    health.add_argument("--reports-dir", type=Path, required=True)
    health.add_argument("--output", type=Path, required=True)

    leakage = sub.add_parser("verify-no-name-leakage")
    leakage.add_argument("--requests", type=Path, required=True)
    leakage.add_argument("--provenance", type=Path, required=True)
    leakage.add_argument("--output", type=Path, required=True)

    drift = sub.add_parser("verify-label-drift")
    drift.add_argument("--current", type=Path, required=True)
    drift.add_argument("--baseline", type=Path, required=True)
    drift.add_argument("--output", type=Path, required=True)

    sample = sub.add_parser("audit-random-sample")
    sample.add_argument("--records", type=Path, required=True)
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--salt", default="raw-v5-audit")
    sample.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("verify-frozen-rebuild")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    inventory = sub.add_parser("inventory-raw-sources")
    inventory.add_argument("--source-root", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    forensics = sub.add_parser("inventory-raw-forensics")
    forensics.add_argument("--source-root", type=Path, required=True)
    forensics.add_argument("--output", type=Path, required=True)

    rebuild = sub.add_parser("rebuild-from-raw")
    rebuild.add_argument("--source-root", type=Path, required=True)
    rebuild.add_argument("--inventory", type=Path, required=True)
    rebuild.add_argument("--plan", type=Path, required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    rebuild.add_argument("--verification-output", type=Path, required=True)

    canary = sub.add_parser("sol-canary")
    canary.add_argument("--cohort", type=Path, required=True)
    canary.add_argument("--image-root", type=Path, required=True)
    canary.add_argument("--projected-record-count", type=int, required=True)
    canary.add_argument("--taxonomy", type=Path)
    canary.add_argument("--input-cost-per-million", type=float)
    canary.add_argument("--output-cost-per-million", type=float)
    canary.add_argument("--pricing-identity", default="")
    canary.add_argument("--unmetered", action="store_true")
    canary.add_argument("--explicit-bulk-cost-authorization", action="store_true")
    canary.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "sol-canary":
        _validate_sol_canary_arguments(args)
    try:
        if args.command == "audit-label-batch":
            provenance = _index_by_record_id(_read_jsonl(args.provenance), source=args.provenance)
            split_by_record = _read_split_map(args.split_map)
            result = audit_label_batch(
                _read_jsonl(args.records),
                batch_id=args.batch_id,
                forbidden_metadata_by_id=provenance,
                relation_manifest=_read_jsonl(args.relation_manifest),
                split_by_record=split_by_record,
            )
            _exclusive_json(args.output, result)
        elif args.command == "audit-label-health":
            reports = [_read_json(path) for path in sorted(args.reports_dir.glob("*.json"))]
            result = audit_label_health(reports)
            _exclusive_json(args.output, result)
        elif args.command == "verify-no-name-leakage":
            provenance = {str(row.get("record_id") or ""): row for row in _read_jsonl(args.provenance)}
            result = verify_no_name_leakage(_read_jsonl(args.requests), provenance)
            _exclusive_json(args.output, result)
        elif args.command == "verify-label-drift":
            result = verify_label_drift(_read_jsonl(args.current), _read_jsonl(args.baseline))
            _exclusive_json(args.output, result)
        elif args.command == "audit-random-sample":
            records = audit_random_sample(_read_jsonl(args.records), count=args.count, salt=args.salt)
            _exclusive_jsonl(args.output, records)
            result = {
                "mode": "deterministic_preparation_only",
                "prepared_record_count": len(records),
                "provider_calls": 0,
            }
        elif args.command == "verify-frozen-rebuild":
            from spritelab.dataset_v5.raw_freeze import verify_frozen_rebuild

            result = verify_frozen_rebuild(args.dataset)
            _exclusive_json(args.output, result)
        elif args.command == "inventory-raw-sources":
            from spritelab.dataset_v5.raw_inventory import write_raw_inventory

            result = write_raw_inventory(args.source_root, args.output)
        elif args.command == "inventory-raw-forensics":
            from spritelab.dataset_v5.raw_forensics import (
                audit_raw_source_inventory,
                write_raw_forensic_inventory,
            )

            inventory = audit_raw_source_inventory(args.source_root)
            paths = write_raw_forensic_inventory(inventory, args.output)
            result = {
                "artifacts": {name: str(path) for name, path in sorted(paths.items())},
                "summary": inventory.summary,
            }
        elif args.command == "rebuild-from-raw":
            from spritelab.dataset_v5.raw_extraction import build_twice_and_verify

            if args.output.exists() or args.verification_output.exists():
                raise FileExistsError("raw rebuild output roots must both be fresh")
            specs = _load_extraction_plan(args.source_root, args.inventory, args.plan)
            result = build_twice_and_verify(specs, args.output, args.verification_output)
        else:
            # Preflight happens before cohort parsing or any provider call.  This
            # is the required behavior on machines without exact Sol identity.
            provider = SolProvider(SolConfig.from_env())
            provider.preflight()
            from spritelab.dataset_v5.canary import SolPricing, run_sol_canary

            pricing = _sol_pricing(args, SolPricing)
            cohort = _load_canary_cohort(args.cohort, args.image_root, args.taxonomy)
            result = run_sol_canary(
                cohort,
                provider=provider,
                projected_record_count=args.projected_record_count,
                pricing=pricing,
                metered=not args.unmetered,
                explicit_bulk_cost_authorization=args.explicit_bulk_cost_authorization,
            )
            _exclusive_json(args.output, result)
    except SolModelUnavailable as exc:
        if args.command == "sol-canary" and not args.output.exists():
            from spritelab.dataset_v5.canary import unavailable_canary_report

            partial_report = getattr(exc, "partial_report", None)
            report = (
                dict(partial_report)
                if isinstance(partial_report, dict)
                else unavailable_canary_report(_public_sol_environment())
            )
            _exclusive_json(args.output, report)
        print(SOL_MODEL_UNAVAILABLE)
        return 78
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if isinstance(result, dict) and result.get("ok") is False:
        return 2
    if isinstance(result, dict) and result.get("status") == "blocked_non_authoritative":
        return 2
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _index_by_record_id(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"provenance row has no record_id: {source}")
        if record_id in result:
            raise ValueError(f"duplicate provenance record_id {record_id!r}: {source}")
        result[record_id] = row
    return result


def _read_split_map(path: Path) -> dict[str, str]:
    document = _read_json(path)
    value = document.get("split_by_record", document)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"split map must be a non-empty JSON object: {path}")
    result: dict[str, str] = {}
    for record_id, split in value.items():
        if not isinstance(record_id, str) or not record_id or not isinstance(split, str) or not split:
            raise ValueError(f"split map contains an invalid record/split binding: {path}")
        result[record_id] = split
    return result


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _exclusive_json(path: Path, value: Any) -> None:
    _prepare_output(path)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _exclusive_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _prepare_output(path)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _load_extraction_plan(source_root: Path, inventory: Path, plan: Path) -> list[Any]:
    """Bind a provenance inventory to an explicit member/crop/padding ledger."""

    from spritelab.dataset_v5.raw_extraction import (
        ExtractionTransform,
        RawExtractionSpec,
        TransparentPadding,
    )
    from spritelab.dataset_v5.raw_inventory import RawSourceRecord, file_sha256

    inventory_path = inventory / "raw_source_inventory.jsonl" if inventory.is_dir() else inventory
    inventory_rows = _read_jsonl(inventory_path)
    by_source_row: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        key = _required_string(row, "source_row_sha256")
        if key in by_source_row:
            raise ValueError(f"duplicate source_row_sha256 in inventory: {key}")
        by_source_row[key] = row

    root = source_root.resolve()
    specs: list[Any] = []
    for line_number, row in enumerate(_read_jsonl(plan), start=1):
        allowed = {
            "archive_member_path",
            "crop_coordinates",
            "padding",
            "schema_version",
            "source_row_sha256",
        }
        extras = sorted(set(row) - allowed)
        if extras:
            raise ValueError(f"unsupported extraction-plan fields at line {line_number}: {extras}")
        if row.get("schema_version") not in (None, "sprite_lab_raw_extraction_plan_v1"):
            raise ValueError(f"unsupported extraction-plan schema at line {line_number}")
        source_key = _required_string(row, "source_row_sha256")
        source_row = by_source_row.get(source_key)
        if source_row is None:
            raise ValueError(f"extraction plan references an unknown inventory row: {source_key}")
        archive_path = _path_below(root, _required_string(source_row, "archive_path"))
        if file_sha256(archive_path) != _required_string(source_row, "archive_sha256"):
            raise ValueError(f"source archive changed after inventory: {source_key}")
        source = RawSourceRecord(
            acquisition_run=_required_string(source_row, "acquisition_run"),
            source_id=_required_string(source_row, "source_id"),
            source_name=_required_string(source_row, "source_name"),
            source_type=_required_string(source_row, "source_type"),
            source_url=str(source_row.get("source_url") or ""),
            download_url=str(source_row.get("download_url") or ""),
            distribution_platform=_required_string(source_row, "distribution_platform"),
            creator_or_publisher=_required_string(source_row, "creator_or_publisher"),
            original_filename=_required_string(source_row, "original_filename"),
            manifest_path=_required_string(source_row, "manifest_path"),
            manifest_sha256=_required_string(source_row, "manifest_sha256"),
            source_row_sha256=source_key,
            archive_path=_required_string(source_row, "archive_path"),
            archive_sha256=_required_string(source_row, "archive_sha256"),
            archive_size_bytes=_required_int(source_row, "archive_size_bytes"),
            expected_archive_sha256=_optional_string(source_row.get("expected_archive_sha256")),
            resolution_method=_required_string(source_row, "resolution_method"),
            provenance_status=_required_string(source_row, "provenance_status"),
            license=_required_mapping(source_row, "license"),
            source_record=_required_mapping(source_row, "source_record"),
            resolved_archive_path=archive_path,
        )
        member = row.get("archive_member_path")
        if member is not None and (not isinstance(member, str) or not member):
            raise ValueError(f"archive_member_path must be a non-empty string or null at line {line_number}")
        crop = _crop_coordinates(row.get("crop_coordinates"), line_number=line_number)
        padding = _transparent_padding(row.get("padding"), line_number=line_number, cls=TransparentPadding)
        specs.append(
            RawExtractionSpec(
                source=source,
                archive_member_path=member,
                transform=ExtractionTransform(crop_coordinates=crop, padding=padding),
            )
        )
    if not specs:
        raise ValueError("extraction plan must contain at least one explicit record")
    return specs


def _load_canary_cohort(cohort_path: Path, image_root: Path, taxonomy_path: Path | None) -> list[Any]:
    from spritelab.dataset_v5.blind import DEFAULT_TAXONOMY, BlindInput

    taxonomy: Any = DEFAULT_TAXONOMY if taxonomy_path is None else _read_json(taxonomy_path)
    root = image_root.resolve()
    records: list[Any] = []
    for line_number, row in enumerate(_read_jsonl(cohort_path), start=1):
        extras = sorted(set(row) - {"canary_tags", "image_path", "local_proposal", "record_id"})
        if extras:
            raise ValueError(f"unsupported canary cohort fields at line {line_number}: {extras}")
        image_path = _path_below(root, _required_string(row, "image_path"))
        with Image.open(image_path) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError(f"canary image must contain one frame: {image_path}")
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
        tags = row.get("canary_tags")
        if not isinstance(tags, list) or not all(isinstance(value, str) and value for value in tags):
            raise ValueError(f"canary_tags must be non-empty strings at line {line_number}")
        local = row.get("local_proposal")
        if local is not None and not isinstance(local, dict):
            raise ValueError(f"local_proposal must be an object or null at line {line_number}")
        blind_input = BlindInput.from_rgba(_required_string(row, "record_id"), rgba, taxonomy=taxonomy)
        records.append((blind_input, {"canary_tags": tags, "local_proposal": local}))
    return records


def _sol_pricing(args: argparse.Namespace, pricing_class: Any) -> Any:
    values = (args.input_cost_per_million, args.output_cost_per_million)
    if (values[0] is None) != (values[1] is None):
        raise ValueError("both Sol input and output prices must be supplied together")
    if values[0] is None:
        return None
    if values[0] < 0 or values[1] < 0:
        raise ValueError("Sol prices must be non-negative")
    if not args.pricing_identity:
        raise ValueError("--pricing-identity is required when pricing is configured")
    return pricing_class(
        input_per_million=values[0],
        output_per_million=values[1],
        pricing_identity=args.pricing_identity,
    )


def _validate_sol_canary_arguments(args: argparse.Namespace) -> None:
    count = args.projected_record_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("--projected-record-count must be a positive integer")
    prices = (args.input_cost_per_million, args.output_cost_per_million)
    if args.unmetered and (any(value is not None for value in prices) or args.pricing_identity):
        raise ValueError("--unmetered cannot be combined with metered pricing")
    if (prices[0] is None) != (prices[1] is None):
        raise ValueError("both Sol input and output prices must be supplied together")
    if prices[0] is None and args.pricing_identity:
        raise ValueError("--pricing-identity requires configured Sol prices")


def _path_below(root: Path, value: str) -> Path:
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes the declared root: {value}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid {field}")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional string must be null or non-empty")
    return value


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"missing or invalid {field}")
    return value


def _required_mapping(row: dict[str, Any], field: str) -> dict[str, Any]:
    value = row.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid {field}")
    return value


def _crop_coordinates(value: Any, *, line_number: int) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"crop_coordinates must be four integers at line {line_number}")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"crop_coordinates must be four integers at line {line_number}")
    return tuple(value)  # type: ignore[return-value]


def _transparent_padding(value: Any, *, line_number: int, cls: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"bottom", "left", "right", "top"}:
        raise ValueError(f"padding must contain bottom/left/right/top at line {line_number}")
    return cls(**value)


def _public_sol_environment() -> dict[str, Any]:
    raw_url = os.environ.get("SPRITELAB_SOL_BASE_URL", "").strip()
    parsed = urllib.parse.urlsplit(raw_url)
    endpoint = ""
    try:
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
            endpoint = urllib.parse.urlunsplit((parsed.scheme.casefold(), netloc.casefold(), parsed.path, "", ""))
    except ValueError:
        endpoint = ""
    return {
        "backend": os.environ.get("SPRITELAB_SOL_BACKEND", ""),
        "configured_model_identifier": os.environ.get("SPRITELAB_SOL_MODEL", ""),
        "endpoint_identity": endpoint,
        "provider_schema_version": "sol_provider_transport_v1",
    }
