"""Human- and machine-readable harvest reports."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.pipeline import HarvestedSprite
from spritelab.harvest.sources import SourceRecord, source_warnings


def build_harvest_report(
    sources: Sequence[SourceRecord],
    harvested: Sequence[HarvestedSprite],
) -> str:
    """Return a markdown harvest report."""

    data = build_harvest_report_data(sources, harvested)
    summary = data["summary"]
    lines = [
        "# Harvest Report",
        "",
        "## Summary",
        f"- Sources: {summary['sources']}",
        f"- Candidates: {summary['candidates']}",
        f"- Imported: {summary['imported']}",
        f"- Valid: {summary['valid']}",
        f"- Rejected: {summary['rejected']}",
        f"- Quarantine: {summary['quarantine']}",
        f"- Accepted: {summary['accepted']}",
        "",
        "## Sources and licenses",
        "| source_id | name | type | license | author | confirmed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source in sources:
        lines.append(
            f"| {source.source_id} | {source.source_name} | {source.source_type} "
            f"| {source.license.license} | {source.author or '-'} "
            f"| {'yes' if source.license.user_confirmed else 'no'} |"
        )

    lines += ["", "## Categories"]
    for category, count in sorted(data["categories"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {category}: {count}")

    lines += ["", "## Tags (top 20)"]
    for tag, count in data["top_tags"]:
        lines.append(f"- {tag}: {count}")

    lines += ["", "## Palette sizes"]
    for size, count in sorted(data["palette_sizes"].items()):
        lines.append(f"- {size} colors: {count}")

    lines += ["", "## Image sizes before processing"]
    for size, count in sorted(data["image_sizes"].items()):
        lines.append(f"- {size}: {count}")

    lines += ["", "## Warnings"]
    if data["warnings"]:
        lines += [f"- {warning}" for warning in data["warnings"]]
    else:
        lines.append("- none")

    lines += ["", "## Prefill quality"]
    if data["prefill_quality"]:
        for bucket, count in sorted(data["prefill_quality"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {bucket}: {count}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Recommendations",
        "- Review quarantined items before export.",
        "- Confirm licenses for any source marked unconfirmed.",
        "- Keep attribution metadata for CC-BY/OGA-BY sources.",
        "",
    ]
    return "\n".join(lines)


def build_harvest_report_data(
    sources: Sequence[SourceRecord],
    harvested: Sequence[HarvestedSprite],
) -> dict[str, Any]:
    """Return the JSON-serializable report payload."""

    statuses = Counter(sprite.final_item.status for sprite in harvested)
    categories = Counter(sprite.final_item.category for sprite in harvested)
    tags = Counter(tag for sprite in harvested for tag in sprite.final_item.tags)
    palette_sizes = Counter(
        sprite.final_item.palette_size for sprite in harvested if sprite.final_item.palette_size is not None
    )
    image_sizes = Counter(f"{sprite.candidate.width}x{sprite.candidate.height}" for sprite in harvested)

    warnings: list[str] = []
    for source in sources:
        warnings.extend(source_warnings(source))
    qwen_failures = [
        f"{sprite.final_item.sprite_id}: qwen failed: {sprite.auto_metadata['qwen_error']}"
        for sprite in harvested
        if "qwen_error" in sprite.auto_metadata
    ]
    invalid = [
        f"{sprite.final_item.sprite_id}: invalid image: {'; '.join(sprite.imported.errors)}"
        for sprite in harvested
        if sprite.imported.errors
    ]
    warnings.extend(qwen_failures)
    warnings.extend(invalid[:50])
    prefill_quality = Counter()
    for sprite in harvested:
        quality = sprite.auto_metadata.get("prefill_quality")
        if not isinstance(quality, dict):
            continue
        prefill_quality[str(quality.get("bucket") or "unknown")] += 1
        for flag in quality.get("flags") or ():
            prefill_quality[str(flag)] += 1

    return {
        "summary": {
            "sources": len(sources),
            "candidates": len({sprite.candidate.candidate_id for sprite in harvested}),
            "imported": len(harvested),
            "valid": sum(1 for sprite in harvested if sprite.imported.bundle is not None),
            "rejected": statuses.get("rejected", 0),
            "quarantine": statuses.get("quarantine", 0),
            "accepted": statuses.get("accepted", 0),
        },
        "sources": [
            {
                "source_id": source.source_id,
                "source_name": source.source_name,
                "source_type": source.source_type,
                "license": source.license.license,
                "author": source.author,
                "user_confirmed_license": source.license.user_confirmed,
                "source_url": source.source_url,
            }
            for source in sources
        ],
        "categories": dict(categories),
        "top_tags": tags.most_common(20),
        "palette_sizes": {str(size): count for size, count in palette_sizes.items()},
        "image_sizes": dict(image_sizes),
        "prefill_quality": dict(prefill_quality),
        "warnings": warnings,
    }


def write_harvest_reports(
    run_dir: str | Path,
    sources: Sequence[SourceRecord],
    harvested: Sequence[HarvestedSprite],
) -> tuple[Path, Path]:
    """Write harvest_report.md and harvest_report.json into the run directory."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = run_dir / "harvest_report.md"
    json_path = run_dir / "harvest_report.json"
    markdown_path.write_text(build_harvest_report(sources, harvested), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_harvest_report_data(sources, harvested), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return markdown_path, json_path
