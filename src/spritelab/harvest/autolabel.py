"""Deterministic rule-based auto-labeling plus optional Qwen batch prefill."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spritelab.dataset_maker.model import DatasetMakerItem, normalize_tag
from spritelab.dataset_maker.prefill import (
    MetadataSuggestion,
    PrefillConfig,
    PrefillRequest,
    apply_suggestion_to_item,
    create_prefill_backend,
    suggestion_to_json_dict,
)
from spritelab.harvest.filename_rules import filename_suggestion_to_dict, parse_filename_metadata
from spritelab.harvest.prefill_fusion import fuse_prefill_suggestions

if TYPE_CHECKING:
    from spritelab.harvest.pipeline import HarvestedSprite

# keyword -> (category, extra tags)
_KEYWORD_RULES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (("mushroom", "fungi", "fungus", "shroom"), "plant", ("mushroom", "organic")),
    (("potion", "vial", "bottle", "flask"), "item_icon", ("vial", "liquid", "potion")),
    (("crystal", "gem", "shard"), "item_icon", ("crystal", "gem")),
    (("leaf", "herb", "flower", "root", "plant"), "plant", ("plant",)),
    (("powder", "dust", "ore", "ingot", "nugget"), "material", ("material",)),
    (("sword", "axe", "pickaxe", "hammer", "bow", "dagger"), "weapon", ("weapon",)),
    (("gear", "cog", "machine", "pipe", "valve"), "tool", ("machine_part",)),
    (("heart", "buff", "effect", "status"), "effect_icon", ("effect",)),
    (("ui", "icon", "button", "cursor", "menu"), "ui_icon", ("ui",)),
)


@dataclass(frozen=True)
class AutoLabelSuggestion:
    category: str
    tags: tuple[str, ...]
    object_name: str = ""
    notes: str = ""
    confidence: float = 0.0
    source: str = "rules"


def suggest_metadata_from_path(
    relative_path: str,
    source_name: str = "",
) -> AutoLabelSuggestion:
    """Suggest category/tags deterministically from path and pack name tokens."""

    tokens = _tokens(relative_path) + _tokens(source_name)
    token_set = set(tokens)
    category = "unknown"
    tags: list[str] = []
    matched = False
    for keywords, rule_category, rule_tags in _KEYWORD_RULES:
        hits = token_set & set(keywords)
        if not hits:
            continue
        if not matched:
            category = rule_category
            matched = True
        tags.extend(sorted(hits))
        tags.extend(rule_tags)

    object_name = " ".join(token for token in _tokens(Path(relative_path).stem) if not token.isdigit())
    seen: set[str] = set()
    unique_tags = tuple(tag for tag in tags if not (tag in seen or seen.add(tag)))
    return AutoLabelSuggestion(
        category=category,
        tags=unique_tags,
        object_name=object_name,
        confidence=0.4 if matched else 0.1,
        source="rules",
    )


def merge_auto_labels(
    base_item: DatasetMakerItem,
    suggestions: Sequence[AutoLabelSuggestion],
    *,
    overwrite_category_if_unknown: bool = True,
) -> DatasetMakerItem:
    """Merge tag/category suggestions; never touch license/author/source/status."""

    category = base_item.category
    tags = list(base_item.tags)
    for suggestion in suggestions:
        if (
            suggestion.category != "unknown"
            and category == "unknown"
            and overwrite_category_if_unknown
        ):
            category = suggestion.category
        for tag in suggestion.tags:
            normalized = normalize_tag(tag)
            if normalized and normalized not in tags:
                tags.append(normalized)

    return DatasetMakerItem(
        sprite_id=base_item.sprite_id,
        source_path=base_item.source_path,
        status=base_item.status,
        category=category,
        tags=tuple(tags),
        notes=base_item.notes,
        source_name=base_item.source_name,
        license=base_item.license,
        author=base_item.author,
        split=base_item.split,
        quality_issues=base_item.quality_issues,
        palette_size=base_item.palette_size,
        has_role_map=base_item.has_role_map,
    )


@dataclass(frozen=True)
class QwenBatchPrefillConfig:
    enabled: bool = False
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"
    runpod_token: str = ""
    timeout_seconds: float = 60.0
    cache_dir: Path = Path(".prefill_cache")
    max_items: int | None = None
    workers: int = 1
    continue_on_error: bool = True
    backend: str = "openai_compatible"
    # Blind-first: the hint measurably biased the model into copying it.
    include_filename_hint: bool = False
    adjudicate: bool = True
    adjudication_threshold: float = 0.6
    retry_attempts: int = 2
    retry_on_warning_only: bool = True
    min_qwen_confidence: float = 0.55
    fusion_policy: str = "weighted"
    structured_output: str = "auto"
    votes: int = 3
    vote_mode: str = "adaptive"
    vote_temperature: float = 0.5
    vlm_role: str = "labeler"
    propagate_dups: bool = True
    propagate_near_dups: bool = False
    near_dup_threshold: int = 2


def batch_prefill_with_qwen(
    harvested: Sequence["HarvestedSprite"],
    config: QwenBatchPrefillConfig,
    *,
    backend: Any | None = None,
) -> list["HarvestedSprite"]:
    """Run cached Qwen prefill over harvested sprites, merging safe fields only.

    License/author/source/status/split are never modified; the cache makes
    reruns resume where they left off. Pass ``backend`` to inject a test double.
    """

    if not config.enabled and backend is None:
        return list(harvested)

    if backend is None:
        backend = create_prefill_backend(
            PrefillConfig(
                enabled=True,
                backend=config.backend,
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
                runpod_token=config.runpod_token,
                timeout_seconds=config.timeout_seconds,
                cache_dir=config.cache_dir,
                include_filename_hint=config.include_filename_hint,
                retry_attempts=config.retry_attempts,
                retry_on_warning_only=config.retry_on_warning_only,
                min_qwen_confidence=config.min_qwen_confidence,
                fusion_policy=config.fusion_policy,
                structured_output=config.structured_output,
                votes=config.votes,
                vote_mode=config.vote_mode,
                vote_temperature=config.vote_temperature,
                vlm_role=config.vlm_role,
            )
        )

    worker_count = max(1, int(config.workers or 1))
    selected_indices = _qwen_prefill_indices(harvested, max_items=config.max_items)
    groups = _prefill_groups(harvested, selected_indices, config)
    results = list(harvested)

    # Phase A: one VLM labeling per group representative.
    _apply_indexed(
        results,
        [group.representative_index for group in groups],
        lambda index: _prefill_harvested_sprite(results[index], backend, config),
        config,
        worker_count,
        description="qwen prefill",
    )

    # Phase B: members reuse the representative's Qwen answer; filename fusion
    # and adjudication still run per member because filenames differ.
    member_tasks = [
        (member_index, group)
        for group in groups
        for member_index in group.member_indices
        if member_index != group.representative_index
    ]
    if member_tasks:
        by_member = {member_index: group for member_index, group in member_tasks}
        _apply_indexed(
            results,
            [member_index for member_index, _ in member_tasks],
            lambda index: _propagate_prefill(
                results[index],
                results[by_member[index].representative_index],
                backend,
                config,
                near=by_member[index].kind == "near",
            ),
            config,
            worker_count,
            description="prefill propagation",
        )
    return results


def _prefill_groups(
    harvested: Sequence["HarvestedSprite"],
    selected_indices: Sequence[int],
    config: QwenBatchPrefillConfig,
) -> list[Any]:
    from spritelab.harvest.prefill_dedupe import PrefillGroup, group_sprites_for_prefill

    if config.propagate_dups or config.propagate_near_dups:
        return group_sprites_for_prefill(
            harvested,
            selected_indices,
            exact_duplicates=config.propagate_dups,
            near_duplicates=config.propagate_near_dups,
            near_dup_threshold=config.near_dup_threshold,
        )
    return [PrefillGroup(index, (index,), "single") for index in selected_indices]


def _apply_indexed(
    results: list["HarvestedSprite"],
    indices: Sequence[int],
    task: Any,
    config: QwenBatchPrefillConfig,
    worker_count: int,
    *,
    description: str,
) -> None:
    def run_one(index: int) -> None:
        try:
            results[index] = task(index)
        except Exception as exc:
            if not config.continue_on_error:
                raise
            results[index] = replace(
                results[index],
                auto_metadata={
                    **results[index].auto_metadata,
                    "qwen_error": str(exc),
                },
            )

    if worker_count == 1 or len(indices) <= 1:
        for index in _progress(list(indices), description):
            run_one(index)
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_one, index) for index in indices]
        for future in _progress(as_completed(futures), description):
            future.result()


def _prefill_harvested_sprite(
    sprite: "HarvestedSprite",
    backend: Any,
    config: QwenBatchPrefillConfig,
) -> "HarvestedSprite":
    from spritelab.codec.reconstruct import reconstruct_rgba

    if sprite.imported.bundle is None:
        return sprite
    bundle = sprite.imported.bundle
    rgba = reconstruct_rgba(bundle)
    image_facts = image_facts_from_bundle(bundle)
    filename_suggestion = parse_filename_metadata(
        sprite.final_item.sprite_id,
        filename=Path(sprite.candidate.relative_path).name,
    )
    filename_dict = filename_suggestion_to_dict(filename_suggestion)
    # Blind-first: the request only carries the filename hint when explicitly
    # enabled; identical images then share one cache entry across filenames.
    request = PrefillRequest(
        sprite_id=sprite.final_item.sprite_id,
        image=rgba,
        existing_category=sprite.final_item.category,
        existing_tags=sprite.final_item.tags,
        source_path=str(sprite.final_item.source_path),
        filename_suggestion=filename_dict if config.include_filename_hint else None,
        image_facts=image_facts,
    )
    suggestion = backend.suggest(request)
    qwen_dict = suggestion_to_json_dict(suggestion)
    return _finalize_prefill(
        sprite,
        qwen_dict,
        image_facts,
        request,
        filename_suggestion,
        filename_dict,
        backend,
        config,
    )


def _propagate_prefill(
    sprite: "HarvestedSprite",
    representative: "HarvestedSprite",
    backend: Any,
    config: QwenBatchPrefillConfig,
    *,
    near: bool,
) -> "HarvestedSprite":
    """Reuse the group representative's Qwen answer for a duplicate sprite."""

    from spritelab.codec.reconstruct import reconstruct_rgba

    if sprite.imported.bundle is None:
        return sprite
    rep_qwen = representative.auto_metadata.get("qwen_suggestion")
    if not isinstance(rep_qwen, dict) or not rep_qwen:
        # Representative failed; leave the member for a later rerun.
        return sprite
    qwen_dict = dict(rep_qwen)
    if near and qwen_dict.get("confidence") is not None:
        try:
            qwen_dict["confidence"] = round(float(qwen_dict["confidence"]) * 0.9, 4)
        except (TypeError, ValueError):
            pass

    bundle = sprite.imported.bundle
    image_facts = image_facts_from_bundle(bundle)
    filename_suggestion = parse_filename_metadata(
        sprite.final_item.sprite_id,
        filename=Path(sprite.candidate.relative_path).name,
    )
    filename_dict = filename_suggestion_to_dict(filename_suggestion)
    request = PrefillRequest(
        sprite_id=sprite.final_item.sprite_id,
        image=reconstruct_rgba(bundle),
        existing_category=sprite.final_item.category,
        existing_tags=sprite.final_item.tags,
        source_path=str(sprite.final_item.source_path),
        filename_suggestion=None,
        image_facts=image_facts,
    )
    extra_metadata: dict[str, Any] = {
        "prefill_propagated_from": representative.final_item.sprite_id,
    }
    if near:
        extra_metadata["prefill_propagated_near_dup"] = True
    else:
        extra_metadata["prefill_propagated_exact_dup"] = True
    return _finalize_prefill(
        sprite,
        qwen_dict,
        image_facts,
        request,
        filename_suggestion,
        filename_dict,
        backend,
        config,
        extra_metadata=extra_metadata,
    )


def _finalize_prefill(
    sprite: "HarvestedSprite",
    qwen_dict: dict[str, Any],
    image_facts: dict[str, Any],
    request: "PrefillRequest",
    filename_suggestion: Any,
    filename_dict: dict[str, Any],
    backend: Any,
    config: QwenBatchPrefillConfig,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> "HarvestedSprite":
    adjudication_dict = None
    if config.adjudicate and not config.include_filename_hint:
        adjudication_dict = _maybe_adjudicate(backend, request, filename_suggestion, qwen_dict, config)

    fused = fuse_prefill_suggestions(
        filename_suggestion,
        qwen_dict,
        min_qwen_confidence=config.min_qwen_confidence,
        fusion_policy=config.fusion_policy,
        adjudication=adjudication_dict,
    )
    # Colors come from the palette, never from the model.
    fused.fused_suggestion["dominant_colors"] = list(image_facts.get("dominant_colors") or ())
    if extra_metadata:
        propagation_flag = "propagated_near_dup" if extra_metadata.get("prefill_propagated_near_dup") else "propagated_exact_dup"
        fused.prefill_quality["flags"] = sorted({*fused.prefill_quality.get("flags", ()), propagation_flag})
    updated_item = apply_suggestion_to_item(
        sprite.final_item,
        _metadata_suggestion_from_dict(fused.fused_suggestion),
    )
    # Hard guard: Qwen never changes provenance or approval fields.
    updated_item = DatasetMakerItem(
        sprite_id=updated_item.sprite_id,
        source_path=sprite.final_item.source_path,
        status=sprite.final_item.status,
        category=updated_item.category,
        tags=updated_item.tags,
        notes=updated_item.notes,
        source_name=sprite.final_item.source_name,
        license=sprite.final_item.license,
        author=sprite.final_item.author,
        split=sprite.final_item.split,
        quality_issues=sprite.final_item.quality_issues,
        palette_size=sprite.final_item.palette_size,
        has_role_map=sprite.final_item.has_role_map,
    )
    return replace(
        sprite,
        final_item=updated_item,
        auto_metadata={
            **sprite.auto_metadata,
            "filename_suggestion": filename_dict,
            "qwen_suggestion": qwen_dict,
            "fused_suggestion": fused.fused_suggestion,
            "prefill_quality": fused.prefill_quality,
            "image_facts": image_facts,
            **({"adjudication": adjudication_dict} if adjudication_dict else {}),
            **(extra_metadata or {}),
        },
    )


def _maybe_adjudicate(
    backend: Any,
    request: "PrefillRequest",
    filename_suggestion: Any,
    qwen_dict: dict[str, Any],
    config: QwenBatchPrefillConfig,
) -> dict[str, Any] | None:
    """Run the forced-choice call only for real, worthwhile conflicts."""

    from spritelab.dataset_maker.prefill import adjudication_to_dict
    from spritelab.harvest.filename_rules import metadata_suggestions_differ

    if filename_suggestion.confidence < config.adjudication_threshold:
        return None
    conflicts = metadata_suggestions_differ(filename_suggestion, qwen_dict)
    if not conflicts or "missing_qwen_suggestion" in conflicts:
        return None
    if not qwen_dict.get("object_name") and not qwen_dict.get("tags"):
        return None
    adjudicate = getattr(backend, "adjudicate", None)
    if adjudicate is None:
        return None
    result = adjudicate(
        request,
        qwen_dict,
        filename_suggestion_to_dict(filename_suggestion),
    )
    if result is None:
        return None
    return adjudication_to_dict(result)


def image_facts_from_bundle(bundle: Any) -> dict[str, Any]:
    """Deterministic prompt facts: content size, palette size, dominant colors."""

    import numpy as np

    from spritelab.codec.color_names import dominant_colors_from_bundle

    alpha = np.asarray(bundle.alpha)
    ys, xs = np.nonzero(alpha)
    if xs.size:
        content_width = int(xs.max() - xs.min() + 1)
        content_height = int(ys.max() - ys.min() + 1)
    else:
        content_height, content_width = (int(dim) for dim in alpha.shape)
    return {
        "content_width": content_width,
        "content_height": content_height,
        "opaque_palette_size": max(0, int(np.asarray(bundle.palette).shape[0]) - 1),
        "dominant_colors": list(dominant_colors_from_bundle(bundle)),
    }


def _metadata_suggestion_from_dict(data: dict[str, Any]) -> MetadataSuggestion:
    return MetadataSuggestion(
        category=str(data.get("category", "unknown")),
        object_name=str(data.get("object_name", "")),
        tags=tuple(str(tag) for tag in data.get("tags") or ()),
        materials=tuple(str(value) for value in data.get("materials") or ()),
        mood=tuple(str(value) for value in data.get("mood") or ()),
        dominant_colors=tuple(str(value) for value in data.get("dominant_colors") or ()),
        short_description=str(data.get("short_description", "")),
        suggested_sprite_id=str(data.get("suggested_sprite_id", "")),
        confidence=data.get("confidence"),
    )


def _qwen_prefill_indices(
    harvested: Sequence["HarvestedSprite"],
    *,
    max_items: int | None,
) -> list[int]:
    selected: list[int] = []
    for item_index, sprite in enumerate(harvested):
        if max_items is not None and len(selected) >= max_items:
            break
        if sprite.imported.preview_image is None or sprite.imported.bundle is None:
            continue
        selected.append(item_index)
    return selected


def _tokens(value: str) -> list[str]:
    text = str(value).lower()
    for separator in ("/", "\\", "-", ".", " "):
        text = text.replace(separator, "_")
    return [token for token in text.split("_") if token]


def _progress(items: Sequence[Any], description: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=description)
    except ImportError:
        return items
