"""Tests for duplicate grouping and prefill propagation."""

from __future__ import annotations

import threading

from _harvest_testdata import make_source, make_sprite_png
from spritelab.dataset_maker.prefill import MetadataSuggestion
from spritelab.harvest.autolabel import QwenBatchPrefillConfig, batch_prefill_with_qwen
from spritelab.harvest.pipeline import HarvestImportOptions, harvest_source_to_imported_sprites
from spritelab.harvest.prefill_dedupe import group_sprites_for_prefill


class _CountingBackend:
    def __init__(self):
        self.calls = 0
        self.sprite_ids = []
        self._lock = threading.Lock()

    def suggest(self, request):
        with self._lock:
            self.calls += 1
            self.sprite_ids.append(request.sprite_id)
        return MetadataSuggestion(
            category="plant",
            object_name="mushroom",
            tags=("mushroom",),
            confidence=0.9,
        )


def _harvest_with_duplicates(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "mushroom_a.png")
    make_sprite_png(root / "mushroom_b.png")  # identical pixels, other name
    make_sprite_png(root / "vial_one.png", color=(60, 100, 220, 255))
    source = make_source(local_root_path=str(root))
    return harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")


def test_group_sprites_exact_duplicates(tmp_path):
    harvested = _harvest_with_duplicates(tmp_path)
    groups = group_sprites_for_prefill(harvested, range(len(harvested)))

    sizes = sorted(len(group.member_indices) for group in groups)
    assert sizes == [1, 2]
    duplicate_group = next(group for group in groups if len(group.member_indices) == 2)
    assert duplicate_group.kind == "exact"
    # Representative is the lowest sprite_id, deterministically.
    rep_id = harvested[duplicate_group.representative_index].final_item.sprite_id
    member_ids = [harvested[index].final_item.sprite_id for index in duplicate_group.member_indices]
    assert rep_id == min(member_ids)


def test_group_sprites_deterministic(tmp_path):
    harvested = _harvest_with_duplicates(tmp_path)
    first = group_sprites_for_prefill(harvested, range(len(harvested)))
    second = group_sprites_for_prefill(harvested, range(len(harvested)))
    assert first == second


def test_batch_prefill_labels_each_unique_image_once(tmp_path):
    harvested = _harvest_with_duplicates(tmp_path)
    backend = _CountingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(enabled=True, adjudicate=False),
        backend=backend,
    )

    assert backend.calls == 2  # two unique images, three sprites
    assert all("qwen_suggestion" in sprite.auto_metadata for sprite in updated)
    propagated = [sprite for sprite in updated if "prefill_propagated_from" in sprite.auto_metadata]
    assert len(propagated) == 1
    source_ids = {sprite.final_item.sprite_id for sprite in updated}
    assert propagated[0].auto_metadata["prefill_propagated_from"] in source_ids
    assert propagated[0].auto_metadata["prefill_propagated_exact_dup"] is True
    assert "propagated_exact_dup" in propagated[0].auto_metadata["prefill_quality"]["flags"]
    # The propagated sprite still gets its own filename fusion.
    assert "fused_suggestion" in propagated[0].auto_metadata


def test_batch_prefill_propagation_disabled(tmp_path):
    harvested = _harvest_with_duplicates(tmp_path)
    backend = _CountingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(enabled=True, adjudicate=False, propagate_dups=False),
        backend=backend,
    )

    assert backend.calls == len(harvested)
    assert not any("prefill_propagated_from" in sprite.auto_metadata for sprite in updated)


def test_exact_propagation_disabled_even_when_near_enabled(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "copy_a.png")
    make_sprite_png(root / "copy_b.png")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    backend = _CountingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(
            enabled=True,
            adjudicate=False,
            propagate_dups=False,
            propagate_near_dups=True,
            near_dup_threshold=2,
        ),
        backend=backend,
    )

    assert backend.calls == len(harvested)
    assert not any("prefill_propagated_from" in sprite.auto_metadata for sprite in updated)


def test_near_duplicate_grouping(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "slime_a.png")
    make_sprite_png(root / "slime_b.png")  # identical
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")

    groups = group_sprites_for_prefill(harvested, range(len(harvested)), near_duplicates=True, near_dup_threshold=2)
    assert len(groups) == 1
    assert groups[0].kind == "exact"


def test_near_duplicate_grouping_merges_non_exact_images(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "gem_a.png", color=(200, 40, 40, 255))
    make_sprite_png(root / "gem_b.png", color=(201, 40, 40, 255))
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")

    groups = group_sprites_for_prefill(harvested, range(len(harvested)), near_duplicates=True, near_dup_threshold=2)

    assert len(groups) == 1
    assert groups[0].kind == "near"


def test_near_duplicate_prefill_propagates_and_scales_confidence(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "gem_a.png", color=(200, 40, 40, 255))
    make_sprite_png(root / "gem_b.png", color=(201, 40, 40, 255))
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    backend = _CountingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(
            enabled=True,
            adjudicate=False,
            propagate_near_dups=True,
            near_dup_threshold=2,
        ),
        backend=backend,
    )

    assert backend.calls == 1
    propagated = [sprite for sprite in updated if sprite.auto_metadata.get("prefill_propagated_near_dup")]
    assert len(propagated) == 1
    assert propagated[0].auto_metadata["prefill_propagated_from"] in {sprite.final_item.sprite_id for sprite in updated}
    assert propagated[0].auto_metadata["qwen_suggestion"]["confidence"] == 0.81
    assert "propagated_near_dup" in propagated[0].auto_metadata["prefill_quality"]["flags"]
