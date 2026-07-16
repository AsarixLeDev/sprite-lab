"""Role-aligned real-ramp transplant augmentation for indexed sprites.

Unlike :mod:`palette_swap`, this module never synthesizes an anchor ramp.  It
borrows hue/chroma from visible palette entries in another real training sprite
and retains the recipient's luminance, geometry, alpha, index map, and role map.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.oklab import oklab_array_to_rgb_u8, rgb_u8_array_to_oklab
from spritelab.codec.roles import ROLE_OUTLINE, ROLE_TRANSPARENT, role_name
from spritelab.training.framing_metrics import jsonable
from spritelab.training.palette_swap import (
    FAMILY_ANCHORS,
    _entry_roles,
    _update_prompt_fields,
    nearest_family,
    sample_seed,
)
from spritelab.training.rgba import npz_row_to_rgba

DEFAULT_EXCLUDED_FAMILIES: tuple[str, ...] = ("gold", "brown")
MIN_SLOT_COVERAGE = 0.01


def parse_excluded_families(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_EXCLUDED_FAMILIES
    values = value.split(",") if isinstance(value, str) else value
    return tuple(dict.fromkeys(str(item).strip().lower() for item in values if str(item).strip() in FAMILY_ANCHORS))


@dataclass(frozen=True)
class RoleRampTransplantConfig:
    enabled: bool = False
    prob: float = 0.0
    keep_original_prob: float = 0.5
    exclude_families: tuple[str, ...] = DEFAULT_EXCLUDED_FAMILIES
    require_trusted_role_map: bool = True
    debug_samples: int = 0
    max_resample_attempts: int = 8
    require_fill_target_match: bool = True
    min_primary_fill_coverage: float = 0.03
    seed: int = 0

    @classmethod
    def from_training_config(cls, config: Any) -> RoleRampTransplantConfig:
        prob = float(getattr(config, "role_ramp_transplant_prob", 0.0))
        return cls(
            enabled=prob > 0.0,
            prob=prob,
            keep_original_prob=float(getattr(config, "role_ramp_transplant_keep_original_prob", 0.5)),
            exclude_families=parse_excluded_families(
                getattr(config, "role_ramp_transplant_exclude_families", DEFAULT_EXCLUDED_FAMILIES)
            ),
            require_trusted_role_map=bool(getattr(config, "role_ramp_transplant_require_trusted_role_map", True)),
            debug_samples=int(getattr(config, "role_ramp_transplant_debug_samples", 0)),
            max_resample_attempts=int(getattr(config, "role_ramp_transplant_max_resample_attempts", 8)),
            require_fill_target_match=bool(getattr(config, "role_ramp_transplant_require_fill_target_match", True)),
            min_primary_fill_coverage=float(getattr(config, "role_ramp_transplant_min_primary_fill_coverage", 0.03)),
            seed=int(getattr(config, "seed", 0)),
        )

    def active(self) -> bool:
        return bool(self.enabled) and float(self.prob) > 0.0

    def report_dict(self) -> dict[str, Any]:
        return {
            "role_ramp_transplant": bool(self.enabled),
            "role_ramp_transplant_prob": float(self.prob),
            "role_ramp_transplant_keep_original_prob": float(self.keep_original_prob),
            "role_ramp_transplant_exclude_families": list(self.exclude_families),
            "role_ramp_transplant_require_trusted_role_map": bool(self.require_trusted_role_map),
            "role_ramp_transplant_debug_samples": int(self.debug_samples),
            "role_ramp_transplant_max_resample_attempts": int(self.max_resample_attempts),
            "role_ramp_transplant_require_fill_target_match": bool(self.require_fill_target_match),
            "role_ramp_transplant_min_primary_fill_coverage": float(self.min_primary_fill_coverage),
        }


@dataclass(frozen=True)
class RampSlot:
    palette_index: int
    role_bucket: str
    coverage: int
    coverage_fraction: float
    lab: tuple[float, float, float]


@dataclass(frozen=True)
class RampDonor:
    sprite_id: str
    color_family: str
    slots: tuple[RampSlot, ...]
    role_map_trusted: bool
    category: str = ""
    primary_fill_family: str = ""

    @property
    def role_buckets(self) -> frozenset[str]:
        return frozenset(slot.role_bucket for slot in self.slots)


@dataclass(frozen=True)
class RoleRampLibrary:
    donors: tuple[RampDonor, ...]

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({donor.color_family for donor in self.donors}))

    def report_dict(self) -> dict[str, Any]:
        counts = Counter(donor.color_family for donor in self.donors)
        return {"donor_count": len(self.donors), "donor_counts_by_family": dict(sorted(counts.items()))}


@dataclass
class RoleRampTransplantResult:
    applied: bool
    palette_rgb: np.ndarray
    record: Mapping[str, Any]
    caption: str
    triggered: bool = False
    kept_original: bool = False
    source_color_family: str = ""
    target_color_family: str = ""
    donor_sprite_id: str = ""
    role_map_trusted: bool = False
    ineligibility_reason: str | None = None
    role_mapping: list[dict[str, Any]] = field(default_factory=list)
    recolored_palette_indices: list[int] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    primary_fill_slots: list[int] = field(default_factory=list)
    primary_fill_coverage: float = 0.0
    post_transplant_primary_fill_family: str = ""
    resample_attempts: int = 0
    role_coverage_success: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "role_ramp_transplant_applied": bool(self.applied),
            "role_ramp_transplant_triggered": bool(self.triggered),
            "role_ramp_transplant_kept_original": bool(self.kept_original),
            "role_ramp_transplant_source_family": self.source_color_family,
            "role_ramp_transplant_target_family": self.target_color_family,
            "role_ramp_transplant_donor_sprite_id": self.donor_sprite_id,
            "role_ramp_transplant_role_map_trusted": bool(self.role_map_trusted),
            "role_ramp_transplant_ineligibility_reason": self.ineligibility_reason,
            "role_ramp_transplant_role_mapping": copy.deepcopy(self.role_mapping),
            "role_ramp_transplant_recolored_palette_indices": list(self.recolored_palette_indices),
            "role_ramp_transplant_safety": copy.deepcopy(self.safety),
            "role_ramp_transplant_primary_fill_slots": list(self.primary_fill_slots),
            "role_ramp_transplant_primary_fill_coverage": float(self.primary_fill_coverage),
            "role_ramp_transplant_post_transplant_primary_fill_family": self.post_transplant_primary_fill_family,
            "role_ramp_transplant_resample_attempts": int(self.resample_attempts),
            "role_ramp_transplant_role_coverage_success": bool(self.role_coverage_success),
        }


def build_role_ramp_library(
    dataset_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    require_trusted_role_map: bool = True,
    exclude_families: Sequence[str] = DEFAULT_EXCLUDED_FAMILIES,
    npz_cache: dict[str, dict[str, np.ndarray]] | None = None,
    min_slot_coverage: float = MIN_SLOT_COVERAGE,
) -> RoleRampLibrary:
    """Build a real-palette donor library; slot zero is never admitted."""
    root = Path(dataset_dir)
    cache = npz_cache if npz_cache is not None else {}
    excluded = set(parse_excluded_families(exclude_families))
    donors: list[RampDonor] = []
    for record in records:
        arrays = _arrays_for_record(root, record, cache)
        if arrays is None:
            continue
        row = int(record.get("npz_row", -1))
        if row < 0 or row >= int(arrays["alpha"].shape[0]):
            continue
        alpha = np.asarray(arrays["alpha"][row], dtype=np.float32)
        index = np.asarray(arrays["index_map"][row], dtype=np.int64)
        roles = np.asarray(arrays["role_map"][row], dtype=np.int64)
        palette = _palette_float(np.asarray(arrays["palette"][row]))
        mask = np.asarray(arrays["palette_mask"][row], dtype=bool)
        slots, trusted = _extract_slots(index, alpha, roles, palette, mask, min_slot_coverage)
        if require_trusted_role_map and not trusted:
            continue
        if len(slots) < 2:
            continue
        primary_slots = primary_fill_slots(slots, min_coverage=min_slot_coverage)
        family = _dominant_primary_fill_family(primary_slots, palette)
        if not family or family in excluded:
            continue
        donors.append(
            RampDonor(
                sprite_id=str(record.get("sprite_id") or ""),
                color_family=family,
                slots=tuple(slots),
                role_map_trusted=trusted,
                category=str(record.get("category") or ""),
                primary_fill_family=family,
            )
        )
    return RoleRampLibrary(tuple(sorted(donors, key=lambda donor: donor.sprite_id)))


def apply_role_ramp_transplant(
    *,
    index_map: np.ndarray,
    alpha: np.ndarray,
    role_map: np.ndarray | None,
    palette_rgb: np.ndarray,
    palette_mask: np.ndarray,
    record: Mapping[str, Any],
    caption: str,
    sprite_id: str,
    library: RoleRampLibrary,
    config: RoleRampTransplantConfig,
) -> RoleRampTransplantResult:
    """Transplant a role-aligned real donor ramp while preserving geometry."""
    palette = np.array(palette_rgb, dtype=np.float32, copy=True)
    result = RoleRampTransplantResult(False, palette, record, caption)
    if not config.active():
        return result
    rng = np.random.default_rng(sample_seed(config.seed, sprite_id))
    result.triggered = bool(rng.random() < float(config.prob))
    if not result.triggered:
        return result

    index = np.asarray(index_map)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim == 3 and alpha_arr.shape[0] == 1:
        alpha_arr = alpha_arr[0]
    visible = alpha_arr > 0.0
    mask = np.asarray(palette_mask, dtype=bool)
    role_arr = None if role_map is None else np.asarray(role_map)
    entry_roles, coverage, trusted = _entry_roles(index, visible, role_arr, mask.shape[0])
    result.role_map_trusted = trusted
    if config.require_trusted_role_map and not trusted:
        result.ineligibility_reason = "untrusted_role_map"
        return result
    recipient_slots = _recipient_slots(entry_roles, coverage, palette, mask)
    if not recipient_slots:
        result.ineligibility_reason = "no_visible_recipient_slots"
        return result
    primary_slots = primary_fill_slots(recipient_slots, min_coverage=config.min_primary_fill_coverage)
    if not primary_slots:
        result.ineligibility_reason = "no_primary_fill_slots"
        return result
    result.primary_fill_slots = [slot.palette_index for slot in primary_slots]
    result.primary_fill_coverage = float(sum(slot.coverage_fraction for slot in primary_slots))
    result.source_color_family = _dominant_primary_fill_family(primary_slots, palette)
    if not result.source_color_family:
        result.ineligibility_reason = "no_primary_fill_family"
        return result
    if float(config.keep_original_prob) > 0.0 and bool(
        rng.random() < float(np.clip(config.keep_original_prob, 0.0, 1.0))
    ):
        result.kept_original = True
        return result
    candidate_families = [
        family
        for family in library.families()
        if family != result.source_color_family and family not in set(config.exclude_families)
    ]
    if not candidate_families:
        result.ineligibility_reason = "no_target_family_donor"
        return result
    required_roles = {slot.role_bucket for slot in recipient_slots}
    compatible_any = False
    original_palette = np.array(palette, dtype=np.float32, copy=True)
    for attempt in range(max(1, int(config.max_resample_attempts))):
        result.resample_attempts = attempt + 1
        target_family = candidate_families[int(rng.integers(0, len(candidate_families)))]
        donors = [
            donor
            for donor in library.donors
            if donor.color_family == target_family
            and donor.primary_fill_family == target_family
            and required_roles.issubset(donor.role_buckets)
            and (not config.require_trusted_role_map or donor.role_map_trusted)
        ]
        if not donors:
            continue
        compatible_any = True
        result.role_coverage_success = True
        donor = donors[int(rng.integers(0, len(donors)))]
        candidate_palette = np.array(original_palette, dtype=np.float32, copy=True)
        role_mapping = _transplant_palette(candidate_palette, recipient_slots, donor)
        if not role_mapping:
            continue
        post_family = _dominant_primary_fill_family(primary_slots, candidate_palette)
        if config.require_fill_target_match and post_family != target_family:
            continue
        result.palette_rgb = candidate_palette
        result.applied = True
        result.target_color_family = target_family
        result.donor_sprite_id = donor.sprite_id
        result.role_mapping = role_mapping
        result.recolored_palette_indices = sorted(item["recipient_palette_index"] for item in role_mapping)
        result.post_transplant_primary_fill_family = post_family
        updated, updated_caption, _kind, _dropped = _update_prompt_fields(record, caption, target_family)
        result.record = updated
        result.caption = updated_caption
        result.safety = _safety_checks(
            index, alpha_arr, role_arr, mask, palette_rgb, candidate_palette, updated, target_family, role_mapping
        )
        return result
    result.ineligibility_reason = (
        "post_transplant_fill_family_mismatch" if compatible_any else "no_role_compatible_donor"
    )
    return result


def _transplant_palette(
    palette: np.ndarray, recipient_slots: Sequence[RampSlot], donor: RampDonor
) -> list[dict[str, Any]]:
    by_role_recipient: dict[str, list[RampSlot]] = defaultdict(list)
    by_role_donor: dict[str, list[RampSlot]] = defaultdict(list)
    for slot in recipient_slots:
        by_role_recipient[slot.role_bucket].append(slot)
    for slot in donor.slots:
        by_role_donor[slot.role_bucket].append(slot)
    mapping: list[dict[str, Any]] = []
    for bucket, recipients in sorted(by_role_recipient.items()):
        donor_slots = sorted(by_role_donor.get(bucket, []), key=lambda slot: slot.lab[0])
        if not donor_slots:
            continue
        recipients = sorted(recipients, key=lambda slot: slot.lab[0])
        for rank, recipient in enumerate(recipients):
            donor_rank = round(rank * (len(donor_slots) - 1) / max(1, len(recipients) - 1))
            donor_slot = donor_slots[int(donor_rank)]
            # Preserve recipient luminance; only use real donor hue/chroma.
            lab = np.asarray([recipient.lab[0], donor_slot.lab[1], donor_slot.lab[2]], dtype=np.float64)
            palette[recipient.palette_index, :3] = oklab_array_to_rgb_u8(lab).astype(np.float32) / 255.0
            mapping.append(
                {
                    "recipient_palette_index": int(recipient.palette_index),
                    "donor_palette_index": int(donor_slot.palette_index),
                    "role_bucket": bucket,
                    "recipient_luminance": float(recipient.lab[0]),
                    "donor_luminance": float(donor_slot.lab[0]),
                }
            )
    return mapping


def _extract_slots(
    index: np.ndarray,
    alpha: np.ndarray,
    role_map: np.ndarray | None,
    palette: np.ndarray,
    mask: np.ndarray,
    min_coverage: float,
) -> tuple[list[RampSlot], bool]:
    visible = np.asarray(alpha) > 0.0
    roles, coverage, trusted = _entry_roles(np.asarray(index), visible, role_map, mask.shape[0])
    total = max(1, int(np.count_nonzero(visible)))
    slots: list[RampSlot] = []
    for palette_index, count in sorted(coverage.items()):
        if palette_index == 0 or palette_index >= len(mask) or not bool(mask[palette_index]):
            continue
        if count / float(total) < float(min_coverage):
            continue
        role = int(roles.get(palette_index, ROLE_TRANSPARENT))
        if role == ROLE_TRANSPARENT:
            continue
        lab = rgb_u8_array_to_oklab(_to_u8(palette[palette_index, :3]))
        slots.append(
            RampSlot(palette_index, role_name(role), int(count), count / float(total), tuple(float(v) for v in lab))
        )
    return slots, trusted


def _recipient_slots(
    entry_roles: Mapping[int, int], coverage: Mapping[int, int], palette: np.ndarray, mask: np.ndarray
) -> list[RampSlot]:
    total = max(1, sum(coverage.values()))
    slots: list[RampSlot] = []
    for index, count in sorted(coverage.items()):
        if index == 0 or index >= len(mask) or not bool(mask[index]):
            continue
        role = int(entry_roles.get(index, ROLE_TRANSPARENT))
        if role == ROLE_TRANSPARENT:
            continue
        lab = rgb_u8_array_to_oklab(_to_u8(palette[index, :3]))
        slots.append(RampSlot(index, role_name(role), int(count), count / float(total), tuple(float(v) for v in lab)))
    return slots


_PRIMARY_FILL_ALLOW = ("fill", "midtone", "body", "main", "accent")
_PRIMARY_FILL_BLOCK = ("outline", "shadow", "highlight", "light", "specular", "border", "transparent")


def is_primary_fill_slot(slot: RampSlot) -> bool:
    """Whether a trusted role bucket represents a substantial object fill."""
    role = str(slot.role_bucket).strip().lower()
    if any(token in role for token in _PRIMARY_FILL_BLOCK):
        return False
    return any(token in role for token in _PRIMARY_FILL_ALLOW)


def primary_fill_slots(slots: Sequence[RampSlot], *, min_coverage: float = 0.03) -> list[RampSlot]:
    """Choose primary object-fill slots, with a conservative coverage fallback."""
    meaningful = [slot for slot in slots if slot.coverage_fraction >= float(min_coverage)]
    preferred = [slot for slot in meaningful if is_primary_fill_slot(slot)]
    if preferred:
        return preferred
    # Only fall back to non-light, non-outline body detail when no named fill role
    # exists. This prevents a large white highlight from defining the prompt color.
    fallback = [
        slot for slot in meaningful if not any(token in slot.role_bucket.lower() for token in _PRIMARY_FILL_BLOCK)
    ]
    return fallback


def _dominant_primary_fill_family(slots: Sequence[RampSlot], palette: np.ndarray) -> str:
    if not slots:
        return ""
    dominant = max(slots, key=lambda slot: slot.coverage)
    return nearest_family(palette[dominant.palette_index, :3])


def _dominant_family(slots: Sequence[RampSlot], palette: np.ndarray) -> str:
    usable = [slot for slot in slots if slot.role_bucket != role_name(ROLE_OUTLINE)] or list(slots)
    if not usable:
        return ""
    weights = np.asarray([slot.coverage for slot in usable], dtype=np.float64)
    colors = np.asarray([palette[slot.palette_index, :3] for slot in usable], dtype=np.float64)
    return nearest_family(np.average(colors, axis=0, weights=weights))


def _arrays_for_record(
    root: Path, record: Mapping[str, Any], cache: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray] | None:
    npz_file = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
    if npz_file not in cache:
        path = root / npz_file
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as loaded:
            cache[npz_file] = {key: np.asarray(loaded[key]) for key in loaded.files}
    return cache.get(npz_file)


def _palette_float(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    return value / 255.0 if float(np.nanmax(value, initial=0.0)) > 1.5 else value


def _to_u8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _safety_checks(
    index: np.ndarray,
    alpha: np.ndarray,
    role_map: np.ndarray | None,
    mask: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    record: Mapping[str, Any],
    target: str,
    mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    visible = alpha > 0.0
    visible_slots_before = {int(v) for v in np.unique(index[visible])}
    visible_slots_after = {int(v) for v in np.unique(index[visible])}
    non_outline = visible if role_map is None else visible & (np.asarray(role_map) != ROLE_OUTLINE)
    near_black = (
        np.max(after[np.asarray(index)[non_outline], :3], axis=1) < 0.12 if np.any(non_outline) else np.array([])
    )
    recolored_families = [nearest_family(after[int(row["recipient_palette_index"]), :3]) for row in mapping]
    return {
        "alpha_unchanged_exact": True,
        "index_map_unchanged_exact": True,
        "role_map_unchanged_exact": True,
        "visible_slot_count_unchanged": len(visible_slots_before) == len(visible_slots_after),
        "prompt_color_matches_target": target in set(record.get("colors") or []),
        "dominant_recolored_slots_match_target": bool(recolored_families) and recolored_families.count(target) >= 1,
        "near_black_nonoutline_fraction": float(np.mean(near_black)) if near_black.size else 0.0,
        "rare_color_warning": False,
        "recolored_slot_count": len(mapping),
    }


@dataclass(frozen=True)
class RoleRampTransplantReviewConfig:
    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    max_samples: int = 128
    seed: int = 20260706
    role_ramp_transplant_prob: float = 1.0
    role_ramp_transplant_keep_original_prob: float = 0.0
    role_ramp_transplant_exclude_families: str = "gold,brown"
    role_ramp_transplant_require_trusted_role_map: bool = True
    role_ramp_transplant_max_resample_attempts: int = 8
    role_ramp_transplant_require_fill_target_match: bool = True
    role_ramp_transplant_min_primary_fill_coverage: float = 0.03


@dataclass(frozen=True)
class RoleRampTransplantAuditConfig:
    """Data-only audit configuration matching the training augmentation flags."""

    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    max_samples: int = 2048
    seed: int = 20260706
    role_ramp_transplant_prob: float = 0.3
    role_ramp_transplant_keep_original_prob: float = 0.5
    role_ramp_transplant_exclude_families: str = "gold,brown,gray,white,black"
    role_ramp_transplant_require_trusted_role_map: bool = True
    role_ramp_transplant_max_resample_attempts: int = 8
    role_ramp_transplant_require_fill_target_match: bool = True
    role_ramp_transplant_min_primary_fill_coverage: float = 0.03


def review_role_ramp_transplant(config: RoleRampTransplantReviewConfig) -> dict[str, Any]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(config.training_manifest)
    selected = records[: max(0, int(config.max_samples))]
    transplant = RoleRampTransplantConfig(
        enabled=True,
        prob=float(config.role_ramp_transplant_prob),
        keep_original_prob=float(config.role_ramp_transplant_keep_original_prob),
        exclude_families=parse_excluded_families(config.role_ramp_transplant_exclude_families),
        require_trusted_role_map=bool(config.role_ramp_transplant_require_trusted_role_map),
        max_resample_attempts=int(config.role_ramp_transplant_max_resample_attempts),
        require_fill_target_match=bool(config.role_ramp_transplant_require_fill_target_match),
        min_primary_fill_coverage=float(config.role_ramp_transplant_min_primary_fill_coverage),
        seed=int(config.seed),
    )
    cache: dict[str, dict[str, np.ndarray]] = {}
    library = build_role_ramp_library(
        config.dataset_dir,
        records,
        require_trusted_role_map=transplant.require_trusted_role_map,
        exclude_families=transplant.exclude_families,
        npz_cache=cache,
    )
    decisions: list[dict[str, Any]] = []
    previews: list[tuple[np.ndarray, np.ndarray]] = []
    for record in selected:
        arrays = _arrays_for_record(Path(config.dataset_dir), record, cache)
        if arrays is None:
            continue
        row = int(record.get("npz_row", -1))
        if row < 0 or row >= int(arrays["alpha"].shape[0]):
            continue
        alpha = np.asarray(arrays["alpha"][row], dtype=np.float32)
        index = np.asarray(arrays["index_map"][row], dtype=np.int64)
        roles = np.asarray(arrays["role_map"][row], dtype=np.int64)
        palette = _palette_float(np.asarray(arrays["palette"][row]))
        mask = np.asarray(arrays["palette_mask"][row], dtype=bool)
        result = apply_role_ramp_transplant(
            index_map=index,
            alpha=alpha,
            role_map=roles,
            palette_rgb=palette,
            palette_mask=mask,
            record=record,
            caption=str(record.get("caption") or ""),
            sprite_id=str(record.get("sprite_id") or ""),
            library=library,
            config=transplant,
        )
        before = npz_row_to_rgba(index_map=index, alpha=alpha, palette=palette, palette_mask=mask)
        after = npz_row_to_rgba(index_map=index, alpha=alpha, palette=result.palette_rgb, palette_mask=mask)
        previews.append((before, after))
        decisions.append(
            {
                "sprite_id": str(record.get("sprite_id") or ""),
                "old_caption": str(record.get("caption") or ""),
                "new_caption": result.caption,
                "old_colors": record.get("colors"),
                "new_colors": result.record.get("colors"),
                **result.metadata(),
            }
        )
    _write_preview_contact_sheet(out / "preview_contact_sheet.png", previews)
    (out / "transplant_decisions.jsonl").write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in decisions), encoding="utf-8"
    )
    applied = sum(1 for row in decisions if row.get("role_ramp_transplant_applied"))
    summary = {
        "schema_version": 1,
        "config": {**transplant.report_dict(), "max_samples": int(config.max_samples)},
        "library": library.report_dict(),
        "sample_count": len(decisions),
        "applied_count": applied,
        "applied_rate": applied / float(len(decisions)) if decisions else 0.0,
        "safety_failures": sum(
            1
            for row in decisions
            if row.get("role_ramp_transplant_applied")
            and not all(
                bool(v)
                for k, v in (row.get("role_ramp_transplant_safety") or {}).items()
                if k.endswith("_exact") or k.endswith("_unchanged")
            )
        ),
        "preview_contact_sheet": str(out / "preview_contact_sheet.png"),
        "decisions_jsonl": str(out / "transplant_decisions.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def audit_role_ramp_transplant(config: RoleRampTransplantAuditConfig) -> dict[str, Any]:
    """Audit real transplant draws without changing a dataset or training a model."""
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(config.training_manifest)
    selected = records[: max(0, int(config.max_samples))]
    transplant = RoleRampTransplantConfig(
        enabled=True,
        prob=float(config.role_ramp_transplant_prob),
        keep_original_prob=float(config.role_ramp_transplant_keep_original_prob),
        exclude_families=parse_excluded_families(config.role_ramp_transplant_exclude_families),
        require_trusted_role_map=bool(config.role_ramp_transplant_require_trusted_role_map),
        max_resample_attempts=int(config.role_ramp_transplant_max_resample_attempts),
        require_fill_target_match=bool(config.role_ramp_transplant_require_fill_target_match),
        min_primary_fill_coverage=float(config.role_ramp_transplant_min_primary_fill_coverage),
        seed=int(config.seed),
    )
    cache: dict[str, dict[str, np.ndarray]] = {}
    library = build_role_ramp_library(
        config.dataset_dir,
        records,
        require_trusted_role_map=transplant.require_trusted_role_map,
        exclude_families=transplant.exclude_families,
        npz_cache=cache,
    )
    donor_by_id = {donor.sprite_id: donor for donor in library.donors}
    rows: list[dict[str, Any]] = []
    previews: list[tuple[np.ndarray, np.ndarray]] = []
    for record in selected:
        arrays = _arrays_for_record(Path(config.dataset_dir), record, cache)
        if arrays is None:
            rows.append(_missing_audit_row(record, "missing_npz"))
            continue
        row_index = int(record.get("npz_row", -1))
        if row_index < 0 or row_index >= int(arrays["alpha"].shape[0]):
            rows.append(_missing_audit_row(record, "invalid_npz_row"))
            continue
        alpha = np.asarray(arrays["alpha"][row_index], dtype=np.float32)
        index = np.asarray(arrays["index_map"][row_index], dtype=np.int64)
        roles = np.asarray(arrays["role_map"][row_index], dtype=np.int64)
        palette = _palette_float(np.asarray(arrays["palette"][row_index]))
        mask = np.asarray(arrays["palette_mask"][row_index], dtype=bool)
        result = apply_role_ramp_transplant(
            index_map=index,
            alpha=alpha,
            role_map=roles,
            palette_rgb=palette,
            palette_mask=mask,
            record=record,
            caption=str(record.get("caption") or ""),
            sprite_id=str(record.get("sprite_id") or ""),
            library=library,
            config=transplant,
        )
        recipient_slots, recipient_trusted = _extract_slots(index, alpha, roles, palette, mask, 0.0)
        donor = donor_by_id.get(result.donor_sprite_id)
        decision = _audit_decision_row(
            record=record,
            alpha=alpha,
            index=index,
            roles=roles,
            palette_before=palette,
            palette_after=result.palette_rgb,
            mask=mask,
            recipient_slots=recipient_slots,
            recipient_trusted=recipient_trusted,
            donor=donor,
            result=result,
            excluded=set(transplant.exclude_families),
        )
        rows.append(decision)
        previews.append(
            (
                npz_row_to_rgba(index_map=index, alpha=alpha, palette=palette, palette_mask=mask),
                npz_row_to_rgba(index_map=index, alpha=alpha, palette=result.palette_rgb, palette_mask=mask),
            )
        )
    summary = summarize_role_ramp_audit(rows, transplant)
    summary.update(
        {
            "schema_version": 1,
            "config": {**transplant.report_dict(), "max_samples": int(config.max_samples)},
            "library": library.report_dict(),
            "contact_sheet": str(out / "contact_sheet.png"),
            "decisions_jsonl": str(out / "transplant_decisions.jsonl"),
            "family_confusion_csv": str(out / "family_confusion.csv"),
            "role_coverage_csv": str(out / "role_coverage.csv"),
        }
    )
    _write_preview_contact_sheet(out / "contact_sheet.png", previews)
    (out / "transplant_decisions.jsonl").write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    _write_family_confusion_csv(out / "family_confusion.csv", rows)
    _write_role_coverage_csv(out / "role_coverage.csv", rows)
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _missing_audit_row(record: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "applied": False,
        "skip_reason": reason,
        "recipient_sprite_id": str(record.get("sprite_id") or ""),
        "recipient_object": str(record.get("object_name") or record.get("base_object") or ""),
        "recipient_category": str(record.get("category") or ""),
        "role_coverage_success": False,
    }


def _audit_decision_row(
    *,
    record: Mapping[str, Any],
    alpha: np.ndarray,
    index: np.ndarray,
    roles: np.ndarray,
    palette_before: np.ndarray,
    palette_after: np.ndarray,
    mask: np.ndarray,
    recipient_slots: Sequence[RampSlot],
    recipient_trusted: bool,
    donor: RampDonor | None,
    result: RoleRampTransplantResult,
    excluded: set[str],
) -> dict[str, Any]:
    primary_slots = primary_fill_slots(recipient_slots)
    recipient_family = _dominant_primary_fill_family(primary_slots, palette_before)
    role_buckets_recipient = sorted({slot.role_bucket for slot in recipient_slots})
    donor_slots = () if donor is None else donor.slots
    donor_visible_families = _slot_families(
        donor_slots, palette_after if donor is None else _donor_palette_colors(donor, donor_slots)
    )
    # For donors, color values are carried in RampSlot OKLab; reconstruct only for reporting.
    donor_fill_family = "" if donor is None else donor.primary_fill_family
    role_buckets_donor = [] if donor is None else sorted(donor.role_buckets)
    role_buckets_matched = sorted(set(role_buckets_recipient) & set(role_buckets_donor))
    dominant_slot = _dominant_fill_slot(primary_slots)
    old_family = "" if dominant_slot is None else nearest_family(palette_before[dominant_slot.palette_index, :3])
    new_family = "" if dominant_slot is None else nearest_family(palette_after[dominant_slot.palette_index, :3])
    target = str(result.target_color_family or "")
    post_primary_family = str(result.post_transplant_primary_fill_family or "")
    prompt_primary = str(result.record.get("primary_color") or "")
    prompt_colors = _as_color_list(result.record.get("colors"))
    semantic_colors = _nested_colors(result.record, "target_semantics")
    structured_colors = _nested_colors(result.record, "conditioning")
    all_visible_new = _visible_palette_families(index, alpha, palette_after, mask)
    role_coverage_success = bool(result.role_coverage_success)
    donor_family = "" if donor is None else donor.color_family
    donor_family_violation = donor_family in excluded
    row = {
        "applied": bool(result.applied),
        "skip_reason": _audit_skip_reason(result),
        "recipient_sprite_id": str(record.get("sprite_id") or ""),
        "recipient_object": str(record.get("object_name") or record.get("base_object") or ""),
        "recipient_category": str(record.get("category") or ""),
        "recipient_original_family": recipient_family,
        "target_prompt_family": target,
        "donor_sprite_id": "" if donor is None else donor.sprite_id,
        "donor_category": "" if donor is None else donor.category,
        "donor_family": donor_family,
        "donor_fill_family": donor_fill_family,
        "donor_visible_families": donor_visible_families,
        "excluded_family_violation": donor_family_violation,
        "excluded_visible_family_present": bool(set(donor_visible_families) & excluded),
        "role_buckets_recipient": role_buckets_recipient,
        "role_buckets_donor": role_buckets_donor,
        "role_buckets_matched": role_buckets_matched,
        "role_coverage_success": role_coverage_success,
        "recipient_role_map_trusted": bool(recipient_trusted),
        "primary_fill_slots": list(result.primary_fill_slots) or [slot.palette_index for slot in primary_slots],
        "primary_fill_coverage": float(result.primary_fill_coverage)
        or float(sum(slot.coverage_fraction for slot in primary_slots)),
        "dominant_fill_slot": None if dominant_slot is None else int(dominant_slot.palette_index),
        "dominant_fill_old_family": old_family,
        "dominant_fill_new_family": new_family,
        "dominant_fill_target_match": bool(result.applied and target and new_family == target),
        "post_transplant_primary_fill_family": post_primary_family,
        "post_transplant_primary_fill_target_match": bool(result.applied and target and post_primary_family == target),
        "all_visible_new_families": all_visible_new,
        "prompt_primary_color_after": prompt_primary,
        "prompt_colors_after": prompt_colors,
        "caption_after": result.caption,
        "semantic_colors_after": semantic_colors,
        "structured_colors_after": structured_colors,
        "prompt_pixel_family_match": bool(result.applied and prompt_primary and post_primary_family == prompt_primary),
        "resample_attempts": int(result.resample_attempts),
        **_palette_lc_stats(recipient_slots, palette_before, palette_after),
    }
    return row


def _audit_skip_reason(result: RoleRampTransplantResult) -> str | None:
    if result.applied:
        return None
    if result.kept_original:
        return "kept_original"
    if not result.triggered:
        return "not_triggered"
    return result.ineligibility_reason or "not_applied"


def _donor_palette_colors(donor: RampDonor, slots: Sequence[RampSlot]) -> np.ndarray:
    if not slots:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([oklab_array_to_rgb_u8(np.asarray(slot.lab)).astype(np.float32) / 255.0 for slot in slots])


def _slot_families(slots: Sequence[RampSlot], colors: np.ndarray) -> list[str]:
    if not slots:
        return []
    return sorted({nearest_family(colors[i, :3]) for i in range(min(len(slots), len(colors)))})


def _slot_dominant_family(slots: Sequence[RampSlot], families: Sequence[str]) -> str:
    if not slots or not families:
        return ""
    candidates = [
        (slot.coverage, family)
        for slot, family in zip(slots, families, strict=False)
        if slot.role_bucket != role_name(ROLE_OUTLINE)
    ]
    return max(candidates or [(slot.coverage, family) for slot, family in zip(slots, families, strict=False)])[1]


def _dominant_fill_slot(slots: Sequence[RampSlot]) -> RampSlot | None:
    non_outline = [slot for slot in slots if slot.role_bucket != role_name(ROLE_OUTLINE)]
    return max(non_outline or list(slots), key=lambda slot: slot.coverage, default=None)


def _visible_palette_families(index: np.ndarray, alpha: np.ndarray, palette: np.ndarray, mask: np.ndarray) -> list[str]:
    visible_indices = {int(v) for v in np.unique(index[np.asarray(alpha) > 0.0])}
    return sorted(nearest_family(palette[i, :3]) for i in visible_indices if i != 0 and i < len(mask) and bool(mask[i]))


def _as_color_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value] if isinstance(value, Sequence) else []


def _nested_colors(record: Mapping[str, Any], key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        return []
    if key == "conditioning":
        value = value.get("semantic_v3") if isinstance(value.get("semantic_v3"), Mapping) else value
    attributes = value.get("attributes") if isinstance(value.get("attributes"), Mapping) else value
    return _as_color_list(attributes.get("colors"))


def _palette_lc_stats(slots: Sequence[RampSlot], before: np.ndarray, after: np.ndarray) -> dict[str, float | None]:
    usable = [slot for slot in slots if slot.role_bucket != role_name(ROLE_OUTLINE)] or list(slots)
    if not usable:
        return dict.fromkeys(
            (
                "mean_chroma_before",
                "mean_chroma_after",
                "mean_chroma_delta",
                "mean_lightness_before",
                "mean_lightness_after",
            )
        )
    weights = np.asarray([slot.coverage for slot in usable], dtype=np.float64)
    before_lab = np.asarray([rgb_u8_array_to_oklab(_to_u8(before[slot.palette_index, :3])) for slot in usable])
    after_lab = np.asarray([rgb_u8_array_to_oklab(_to_u8(after[slot.palette_index, :3])) for slot in usable])
    before_chroma = np.hypot(before_lab[:, 1], before_lab[:, 2])
    after_chroma = np.hypot(after_lab[:, 1], after_lab[:, 2])
    return {
        "mean_chroma_before": float(np.average(before_chroma, weights=weights)),
        "mean_chroma_after": float(np.average(after_chroma, weights=weights)),
        "mean_chroma_delta": float(np.average(after_chroma - before_chroma, weights=weights)),
        "mean_lightness_before": float(np.average(before_lab[:, 0], weights=weights)),
        "mean_lightness_after": float(np.average(after_lab[:, 0], weights=weights)),
    }


def summarize_role_ramp_audit(rows: Sequence[Mapping[str, Any]], config: RoleRampTransplantConfig) -> dict[str, Any]:
    """Summarize decision rows and attach explicit no-go threshold flags."""
    attempted = len(rows)
    applied_rows = [row for row in rows if row.get("applied")]
    coverage_rows = [
        row for row in rows if row.get("skip_reason") not in {"not_triggered", "kept_original", "no_primary_fill_slots"}
    ]

    def counts(key: str, source: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(key) or "") for row in source if row.get(key)).items()))

    def rate(key: str, values: Sequence[Mapping[str, Any]] = applied_rows) -> float | None:
        known = [bool(row.get(key)) for row in values if row.get(key) is not None]
        return sum(known) / float(len(known)) if known else None

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in applied_rows if isinstance(row.get(key), (int, float))]
        return float(np.mean(values)) if values else None

    applied_rate = len(applied_rows) / float(attempted) if attempted else 0.0
    excluded_count = sum(1 for row in rows if row.get("excluded_family_violation"))
    coverage_rate = rate("role_coverage_success", coverage_rows)
    dominant_rate = rate("dominant_fill_target_match")
    post_primary_rate = rate("post_transplant_primary_fill_target_match")
    prompt_pixel_rate = rate("prompt_pixel_family_match")
    thresholds = {
        "excluded_family_violation": excluded_count > 0,
        "post_transplant_primary_fill_target_match": post_primary_rate is None or post_primary_rate < 0.85,
        "prompt_pixel_family_match": prompt_pixel_rate is None or prompt_pixel_rate < 0.85,
        "role_coverage": coverage_rate is None or coverage_rate < 0.95,
        "applied_rate": bool(config.prob < 1.0 and applied_rate < 0.10),
    }
    return {
        "attempted_count": attempted,
        "applied_count": len(applied_rows),
        "applied_rate": applied_rate,
        "skip_reasons": counts("skip_reason", rows),
        "target_prompt_family_counts": counts("target_prompt_family", applied_rows),
        "actual_dominant_fill_family_counts": counts("dominant_fill_new_family", applied_rows),
        "donor_family_counts": counts("donor_family", applied_rows),
        "recipient_original_family_counts": counts("recipient_original_family", rows),
        "excluded_family_violation_count": excluded_count,
        "excluded_visible_family_present_count": sum(
            1 for row in applied_rows if row.get("excluded_visible_family_present")
        ),
        "dominant_fill_target_match_rate": dominant_rate,
        "post_transplant_primary_fill_target_match_rate": post_primary_rate,
        "prompt_pixel_family_match_rate": prompt_pixel_rate,
        "role_coverage_success_rate": coverage_rate,
        "mean_resample_attempts": mean("resample_attempts"),
        "effective_applied_rate": applied_rate,
        "mean_chroma_before": mean("mean_chroma_before"),
        "mean_chroma_after": mean("mean_chroma_after"),
        "mean_chroma_delta": mean("mean_chroma_delta"),
        "mean_lightness_before": mean("mean_lightness_before"),
        "mean_lightness_after": mean("mean_lightness_after"),
        "failure_thresholds": thresholds,
        "failed": any(thresholds.values()),
    }


def _write_family_confusion_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(
        (str(row.get("target_prompt_family") or ""), str(row.get("dominant_fill_new_family") or ""))
        for row in rows
        if row.get("applied")
    )
    lines = ["target_prompt_family,dominant_fill_new_family,count"]
    lines.extend(f"{target},{actual},{count}" for (target, actual), count in sorted(counts.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_role_coverage_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "recipient_sprite_id,role_buckets_recipient,role_buckets_donor,role_buckets_matched,role_coverage_success,skip_reason"
    ]
    for row in rows:
        values = [
            str(row.get("recipient_sprite_id") or ""),
            "|".join(row.get("role_buckets_recipient") or []),
            "|".join(row.get("role_buckets_donor") or []),
            "|".join(row.get("role_buckets_matched") or []),
            str(bool(row.get("role_coverage_success"))).lower(),
            str(row.get("skip_reason") or ""),
        ]
        lines.append(",".join(value.replace(",", ";") for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview_contact_sheet(path: Path, previews: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    cell = 64
    width = max(cell, min(8, max(1, len(previews))) * cell)
    rows = max(1, (len(previews) + 7) // 8)
    sheet = Image.new("RGBA", (width, rows * cell), (48, 48, 48, 255))
    for i, (before, after) in enumerate(previews):
        pair = Image.new("RGBA", (64, 32), (48, 48, 48, 255))
        pair.paste(Image.fromarray(np.transpose(np.clip(before * 255.0, 0, 255).astype(np.uint8), (1, 2, 0))), (0, 0))
        pair.paste(Image.fromarray(np.transpose(np.clip(after * 255.0, 0, 255).astype(np.uint8), (1, 2, 0))), (32, 0))
        x, y = (i % 8) * cell, (i // 8) * cell
        sheet.paste(pair, (x, y))
    sheet.save(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows
