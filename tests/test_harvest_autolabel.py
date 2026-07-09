"""Tests for spritelab.harvest.autolabel."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from _harvest_testdata import make_source, make_sprite_png
from spritelab.dataset_maker.model import DatasetMakerItem
from spritelab.dataset_maker.prefill import MetadataSuggestion
from spritelab.harvest.autolabel import (
    AutoLabelSuggestion,
    QwenBatchPrefillConfig,
    batch_prefill_with_qwen,
    merge_auto_labels,
    suggest_metadata_from_path,
)
from spritelab.harvest.pipeline import HarvestImportOptions, harvest_source_to_imported_sprites


def test_mushroom_path_suggests_mushroom():
    suggestion = suggest_metadata_from_path("plants/purple_mushroom.png")
    assert "mushroom" in suggestion.tags
    assert suggestion.category == "plant"


def test_potion_path_suggests_vial():
    suggestion = suggest_metadata_from_path("items/health_potion_vial.png")
    assert suggestion.category == "item_icon"
    assert "vial" in suggestion.tags
    assert "potion" in suggestion.tags


def test_crystal_suggests_gem():
    suggestion = suggest_metadata_from_path("gems/blue_crystal_shard.png")
    assert "crystal" in suggestion.tags
    assert "gem" in suggestion.tags


def test_gear_suggests_machine_part():
    suggestion = suggest_metadata_from_path("factory/rusty_gear.png")
    assert "machine_part" in suggestion.tags
    assert suggestion.category == "tool"


def _item(category="unknown"):
    return DatasetMakerItem(
        sprite_id="sample",
        source_path=Path("sample.png"),
        status="accepted",
        category=category,
        tags=("existing",),
        license="cc0",
        author="Someone",
    )


def test_merge_does_not_overwrite_known_category():
    suggestion = AutoLabelSuggestion(category="plant", tags=("mushroom",))
    merged = merge_auto_labels(_item(category="weapon"), [suggestion])
    assert merged.category == "weapon"
    assert "mushroom" in merged.tags
    assert merged.license == "cc0"
    assert merged.author == "Someone"

    merged_unknown = merge_auto_labels(_item(category="unknown"), [suggestion])
    assert merged_unknown.category == "plant"


class _FakeBackend:
    def __init__(self, fail_for: set[str] = frozenset()):
        self.fail_for = set(fail_for)
        self.calls = 0
        self.requests = []
        self._lock = threading.Lock()

    def suggest(self, request):
        with self._lock:
            self.calls += 1
            self.requests.append(request)
        if request.sprite_id in self.fail_for:
            raise RuntimeError("boom")
        return MetadataSuggestion(category="item_icon", tags=("qwen_tag",), confidence=0.9)


class _SlowTrackingBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0

    def suggest(self, request):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().suggest(request)
        finally:
            with self._lock:
                self.active -= 1


def _harvest(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "mushroom_one.png")
    make_sprite_png(root / "vial_two.png", color=(60, 100, 220, 255))
    source = make_source(local_root_path=str(root))
    return harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")


def test_qwen_batch_merges_suggestions(tmp_path):
    harvested = _harvest(tmp_path)
    backend = _FakeBackend()
    updated = batch_prefill_with_qwen(harvested, QwenBatchPrefillConfig(enabled=True), backend=backend)
    assert backend.calls == len(harvested)
    for before, after in zip(harvested, updated):
        assert "qwen_tag" in after.final_item.tags
        assert after.final_item.license == before.final_item.license
        assert after.final_item.status == before.final_item.status
        assert "qwen_suggestion" in after.auto_metadata
        assert "filename_suggestion" in after.auto_metadata
        assert "fused_suggestion" in after.auto_metadata
        assert "prefill_quality" in after.auto_metadata
    # Blind-first by default: the labeling request carries no filename hint.
    assert all(request.filename_suggestion is None for request in backend.requests)
    assert all(request.image_facts for request in backend.requests)


def test_qwen_batch_includes_filename_hint_when_enabled(tmp_path):
    harvested = _harvest(tmp_path)
    backend = _FakeBackend()
    batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(enabled=True, include_filename_hint=True),
        backend=backend,
    )
    assert all(request.filename_suggestion for request in backend.requests)


class _AdjudicatingBackend(_FakeBackend):
    """Blind answer that conflicts with strong filename rules, then adjudicates."""

    def __init__(self, choice: str = "b"):
        super().__init__()
        self.choice = choice
        self.adjudications = []

    def suggest(self, request):
        with self._lock:
            self.calls += 1
            self.requests.append(request)
        return MetadataSuggestion(
            category="plant",
            object_name="mushroom",
            tags=("mushroom",),
            confidence=0.8,
        )

    def adjudicate(self, request, candidate_a, candidate_b):
        from spritelab.dataset_maker.prefill import AdjudicationResult

        with self._lock:
            self.adjudications.append((request.sprite_id, dict(candidate_a), dict(candidate_b)))
        return AdjudicationResult(choice=self.choice, reason="axe blade visible")


def test_qwen_batch_adjudicates_conflicts(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "W_Axe014.png")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    backend = _AdjudicatingBackend(choice="b")

    updated = batch_prefill_with_qwen(harvested, QwenBatchPrefillConfig(enabled=True), backend=backend)

    assert backend.adjudications, "conflict with a strong filename rule should trigger adjudication"
    sprite = updated[0]
    assert sprite.auto_metadata["adjudication"]["choice"] == "b"
    assert sprite.auto_metadata["prefill_quality"]["agreement"] == "adjudicated_filename"
    assert sprite.auto_metadata["fused_suggestion"]["category"] == "weapon"


def test_qwen_batch_no_adjudication_when_disabled(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "W_Axe014.png")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    backend = _AdjudicatingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(enabled=True, adjudicate=False),
        backend=backend,
    )

    assert not backend.adjudications
    assert "adjudication" not in updated[0].auto_metadata


def test_qwen_batch_workers_prefill_concurrently_and_preserve_order(tmp_path):
    harvested = _harvest(tmp_path)
    backend = _SlowTrackingBackend()

    updated = batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(enabled=True, workers=2),
        backend=backend,
    )

    assert backend.calls == len(harvested)
    assert backend.max_active >= 2
    assert [sprite.final_item.sprite_id for sprite in updated] == [sprite.final_item.sprite_id for sprite in harvested]
    assert all("qwen_suggestion" in sprite.auto_metadata for sprite in updated)


def test_qwen_failures_continue(tmp_path):
    harvested = _harvest(tmp_path)
    failing_id = harvested[0].final_item.sprite_id
    backend = _FakeBackend(fail_for={failing_id})
    updated = batch_prefill_with_qwen(
        harvested, QwenBatchPrefillConfig(enabled=True, continue_on_error=True), backend=backend
    )
    assert len(updated) == len(harvested)
    assert "qwen_error" in updated[0].auto_metadata
    assert "qwen_suggestion" in updated[1].auto_metadata


def test_qwen_config_passes_ollama_and_runpod_token(tmp_path, monkeypatch):
    harvested = _harvest(tmp_path)
    captured = {}

    def fake_create_prefill_backend(config):
        captured["backend"] = config.backend
        captured["runpod_token"] = config.runpod_token
        return _FakeBackend()

    monkeypatch.setattr("spritelab.harvest.autolabel.create_prefill_backend", fake_create_prefill_backend)
    batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(
            enabled=True,
            backend="ollama",
            runpod_token="runpod-secret",
            max_items=1,
        ),
    )

    assert captured == {"backend": "ollama", "runpod_token": "runpod-secret"}


def test_qwen_config_passes_retry_and_fusion_options(tmp_path, monkeypatch):
    harvested = _harvest(tmp_path)
    captured = {}

    def fake_create_prefill_backend(config):
        captured["include_filename_hint"] = config.include_filename_hint
        captured["retry_attempts"] = config.retry_attempts
        captured["retry_on_warning_only"] = config.retry_on_warning_only
        captured["min_qwen_confidence"] = config.min_qwen_confidence
        captured["fusion_policy"] = config.fusion_policy
        return _FakeBackend()

    monkeypatch.setattr("spritelab.harvest.autolabel.create_prefill_backend", fake_create_prefill_backend)
    batch_prefill_with_qwen(
        harvested,
        QwenBatchPrefillConfig(
            enabled=True,
            include_filename_hint=False,
            retry_attempts=3,
            retry_on_warning_only=False,
            min_qwen_confidence=0.7,
            fusion_policy="weighted",
            max_items=1,
        ),
    )

    assert captured == {
        "include_filename_hint": False,
        "retry_attempts": 3,
        "retry_on_warning_only": False,
        "min_qwen_confidence": 0.7,
        "fusion_policy": "weighted",
    }
