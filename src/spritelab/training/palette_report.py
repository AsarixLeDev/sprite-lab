"""Palette-slot semantics reports for curated SpriteBundle datasets."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from spritelab.codec.io import load_bundle
from spritelab.codec.roles import ROLE_DEEP_SHADOW, ROLE_NAMES, ROLE_OUTLINE, role_name
from spritelab.curation.manifest import discover_bundle_ids, load_latest_curation


@dataclass(frozen=True)
class PaletteSlotStats:
    slot_id: int
    sprite_count: int
    pixel_count: int
    mean_rgb: tuple[float, float, float] | None
    mean_luminance: float | None
    mean_chroma: float | None
    mean_hue_degrees: float | None
    dominant_role_id: int | None
    dominant_role_name: str | None
    role_counts: dict[str, int]
    role_entropy: float | None
    mean_usage_fraction: float | None


@dataclass(frozen=True)
class PaletteSemanticsReport:
    bundle_count: int
    accepted_count: int
    palette_size_counts: dict[int, int]
    slot_stats: list[PaletteSlotStats]
    warnings: list[str]


def build_palette_semantics_report(bundle_paths: Sequence[Path]) -> PaletteSemanticsReport:
    """Build a deterministic palette-slot semantics report from accepted bundles."""

    paths = [Path(path) for path in bundle_paths]
    slot_rgb: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    slot_luma: dict[int, list[float]] = defaultdict(list)
    slot_chroma: dict[int, list[float]] = defaultdict(list)
    slot_hue_vectors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    slot_usage: dict[int, list[float]] = defaultdict(list)
    slot_pixel_counts: Counter[int] = Counter()
    slot_sprite_counts: Counter[int] = Counter()
    slot_role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    palette_size_counts: Counter[int] = Counter()
    role_map_count = 0

    for path in paths:
        bundle = load_bundle(path)
        palette = np.asarray(bundle.palette)
        index_map = np.asarray(bundle.index_map)
        alpha = np.asarray(bundle.alpha)
        opaque_count = int(np.count_nonzero(alpha == 1))
        visible_count = int(palette.shape[0] - 1)
        palette_size_counts.update([visible_count])
        if bundle.role_map is not None:
            role_map_count += 1
        role_map = np.asarray(bundle.role_map) if bundle.role_map is not None else None

        for slot_id in range(1, int(palette.shape[0])):
            slot_sprite_counts.update([slot_id])
            slot_mask = index_map == slot_id
            pixel_count = int(np.count_nonzero(slot_mask))
            slot_pixel_counts.update({slot_id: pixel_count})
            usage_fraction = pixel_count / opaque_count if opaque_count > 0 else 0.0
            slot_usage[slot_id].append(usage_fraction)

            rgb = tuple(float(channel) / 255.0 for channel in palette[slot_id])
            slot_rgb[slot_id].append(rgb)
            luma = _luminance(rgb)
            chroma = _chroma(rgb)
            slot_luma[slot_id].append(luma)
            slot_chroma[slot_id].append(chroma)
            hue = _hue_degrees(rgb, chroma)
            if hue is not None:
                radians = math.radians(hue)
                slot_hue_vectors[slot_id].append((math.cos(radians), math.sin(radians)))

            if role_map is not None and pixel_count:
                roles = role_map[slot_mask]
                for role_id in np.asarray(roles).flatten():
                    slot_role_counts[slot_id].update([role_name(int(role_id))])

    stats = [
        _slot_stats(
            slot_id=slot_id,
            sprite_count=slot_sprite_counts[slot_id],
            pixel_count=slot_pixel_counts[slot_id],
            rgb_values=slot_rgb[slot_id],
            luma_values=slot_luma[slot_id],
            chroma_values=slot_chroma[slot_id],
            hue_vectors=slot_hue_vectors[slot_id],
            usage_values=slot_usage[slot_id],
            role_counts=slot_role_counts[slot_id],
        )
        for slot_id in sorted(slot_sprite_counts)
    ]

    warnings = _warnings(
        bundle_count=len(paths),
        role_map_count=role_map_count,
        palette_size_counts=palette_size_counts,
        slot_stats=stats,
    )
    return PaletteSemanticsReport(
        bundle_count=len(paths),
        accepted_count=len(paths),
        palette_size_counts=dict(sorted(palette_size_counts.items())),
        slot_stats=stats,
        warnings=warnings,
    )


def build_palette_semantics_report_from_curation(
    bundle_root: Path,
    curation_path: Path,
) -> PaletteSemanticsReport:
    """Build a palette report from latest accepted curation decisions."""

    bundle_ids = discover_bundle_ids(bundle_root)
    latest = load_latest_curation(curation_path)
    accepted_paths = [
        bundle_ids[sprite_id]
        for sprite_id, decision in sorted(latest.items())
        if decision.status == "accepted" and sprite_id in bundle_ids
    ]
    return build_palette_semantics_report(accepted_paths)


def write_palette_semantics_report_json(report: PaletteSemanticsReport, path: Path) -> None:
    """Write a palette semantics report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_palette_semantics_report_markdown(report: PaletteSemanticsReport, path: Path) -> None:
    """Write a palette semantics report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_palette_semantics_report_markdown(report) + "\n", encoding="utf-8")


def format_palette_semantics_report_markdown(report: PaletteSemanticsReport) -> str:
    """Render a palette semantics report as Markdown."""

    lines = [
        "# Palette Slot Semantics Report",
        "",
        "## Overview",
        "",
        f"- Accepted bundles: {report.accepted_count}",
        f"- Bundle count: {report.bundle_count}",
        f"- Palette size distribution: {_format_distribution(report.palette_size_counts)}",
        "",
        "## Slot statistics",
        "",
        "| Slot | Sprites | Pixels | Mean RGB | Luma | Chroma | Dominant role | Role entropy | Usage |",
        "|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for stat in report.slot_stats:
        lines.append(
            "| "
            f"{stat.slot_id} | {stat.sprite_count} | {stat.pixel_count} | {_format_rgb(stat.mean_rgb)} | "
            f"{_fmt_float(stat.mean_luminance)} | {_fmt_float(stat.mean_chroma)} | "
            f"{stat.dominant_role_name or 'none'} | {_fmt_float(stat.role_entropy)} | "
            f"{_fmt_float(stat.mean_usage_fraction)} |"
        )
    lines.extend(["", "## Warnings", ""])
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- None")
    return "\n".join(lines)


def _slot_stats(
    *,
    slot_id: int,
    sprite_count: int,
    pixel_count: int,
    rgb_values: list[tuple[float, float, float]],
    luma_values: list[float],
    chroma_values: list[float],
    hue_vectors: list[tuple[float, float]],
    usage_values: list[float],
    role_counts: Counter[str],
) -> PaletteSlotStats:
    mean_rgb = _mean_rgb(rgb_values)
    dominant_name = role_counts.most_common(1)[0][0] if role_counts else None
    dominant_role_id = _role_id_for_name(dominant_name) if dominant_name else None
    return PaletteSlotStats(
        slot_id=slot_id,
        sprite_count=sprite_count,
        pixel_count=pixel_count,
        mean_rgb=mean_rgb,
        mean_luminance=_mean(luma_values),
        mean_chroma=_mean(chroma_values),
        mean_hue_degrees=_mean_hue(hue_vectors),
        dominant_role_id=dominant_role_id,
        dominant_role_name=dominant_name,
        role_counts=dict(sorted(role_counts.items())),
        role_entropy=_entropy(role_counts) if role_counts else None,
        mean_usage_fraction=_mean(usage_values),
    )


def _warnings(
    *,
    bundle_count: int,
    role_map_count: int,
    palette_size_counts: Counter[int],
    slot_stats: list[PaletteSlotStats],
) -> list[str]:
    warnings: list[str] = []
    if bundle_count == 0:
        warnings.append("No accepted bundles available.")
        return warnings
    if role_map_count < (bundle_count / 2):
        warnings.append("Most accepted bundles have no role_map; role semantics may be incomplete.")
    if palette_size_counts:
        palette_sizes = list(palette_size_counts.elements())
        if max(palette_sizes) - min(palette_sizes) > 16:
            warnings.append("Palette sizes vary widely.")
        if sum(count for size, count in palette_size_counts.items() if size > 24) > bundle_count / 2:
            warnings.append("Many accepted sprites have very large palettes.")
    for stat in slot_stats:
        if stat.role_entropy is not None and stat.pixel_count >= 10 and stat.role_entropy > 1.5:
            warnings.append(f"Slot {stat.slot_id} has high role entropy.")
    slot_one = next((stat for stat in slot_stats if stat.slot_id == 1), None)
    if (
        slot_one is not None
        and slot_one.dominant_role_id not in {ROLE_OUTLINE, ROLE_DEEP_SHADOW, None}
        and slot_one.pixel_count >= 10
    ):
        warnings.append("Slot 1 is not dominated by OUTLINE or DEEP_SHADOW.")
    return warnings


def _luminance(rgb: tuple[float, float, float]) -> float:
    red, green, blue = rgb
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _chroma(rgb: tuple[float, float, float]) -> float:
    return max(rgb) - min(rgb)


def _hue_degrees(rgb: tuple[float, float, float], chroma: float) -> float | None:
    if chroma < 1e-6:
        return None
    hue, _saturation, _value = colorsys.rgb_to_hsv(*rgb)
    return hue * 360.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_rgb(values: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    if not values:
        return None
    return tuple(float(sum(rgb[index] for rgb in values) / len(values)) for index in range(3))


def _mean_hue(vectors: list[tuple[float, float]]) -> float | None:
    if not vectors:
        return None
    mean_x = sum(vector[0] for vector in vectors) / len(vectors)
    mean_y = sum(vector[1] for vector in vectors) / len(vectors)
    if abs(mean_x) < 1e-9 and abs(mean_y) < 1e-9:
        return None
    return (math.degrees(math.atan2(mean_y, mean_x)) + 360.0) % 360.0


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return float(value)


def _role_id_for_name(name: str | None) -> int | None:
    if name is None:
        return None
    for role_id, role_name_value in ROLE_NAMES.items():
        if role_name_value == name:
            return int(role_id)
    return None


def _format_distribution(values: dict[int, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{size}: {count}" for size, count in sorted(values.items()))


def _format_rgb(value: tuple[float, float, float] | None) -> str:
    if value is None:
        return "none"
    return "(" + ", ".join(f"{component:.3f}" for component in value) + ")"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a palette-slot semantics report from accepted bundles.")
    parser.add_argument("--bundles", required=True, type=Path)
    parser.add_argument("--curation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = build_palette_semantics_report_from_curation(args.bundles, args.curation)
    write_palette_semantics_report_markdown(report, args.out)
    if args.json_path is not None:
        write_palette_semantics_report_json(report, args.json_path)
    print(f"Accepted bundles: {report.accepted_count}")
    print(f"Warnings: {len(report.warnings)}")
    print(f"Markdown: {args.out}")
    if args.json_path is not None:
        print(f"JSON: {args.json_path}")


if __name__ == "__main__":
    main()
