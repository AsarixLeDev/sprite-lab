"""Tests for deterministic filename/Qwen prefill fusion."""

from __future__ import annotations

from spritelab.harvest.filename_rules import parse_filename_metadata
from spritelab.harvest.prefill_fusion import (
    QUALITY_FILENAME_QWEN_CONFLICT,
    QUALITY_FUSED_AUTOMATICALLY,
    QUALITY_LOW_CONFIDENCE,
    QUALITY_NEEDS_REVIEW,
    QUALITY_WARNING_ONLY,
    fuse_prefill_suggestions,
)


def test_fusion_uses_strong_filename_when_qwen_is_ambiguous() -> None:
    filename = parse_filename_metadata("sprite", filename="I_C_Banana.png")
    qwen = {
        "category": "unknown",
        "object_name": "ambiguous_object",
        "tags": ["ambiguous"],
        "confidence": 0.31,
        "warnings": ["Object is ambiguous at 32x32."],
    }

    result = fuse_prefill_suggestions(filename, qwen)

    assert result.fused_suggestion["object_name"] == "banana"
    assert result.fused_suggestion["category"] == "item_icon"
    assert QUALITY_LOW_CONFIDENCE in result.prefill_quality["flags"]


def test_fusion_uses_qwen_when_filename_is_weak() -> None:
    filename = parse_filename_metadata("sprite", filename="MysteryBlob.png")
    qwen = {
        "category": "plant",
        "object_name": "mushroom",
        "tags": ["mushroom", "brown"],
        "confidence": 0.86,
        "visual_evidence": ["cap", "stem"],
    }

    result = fuse_prefill_suggestions(filename, qwen)

    assert result.fused_suggestion["object_name"] == "mushroom"
    assert result.fused_suggestion["category"] == "plant"
    assert result.prefill_quality["agreement"] == "conflict"


def test_fusion_adjudication_choice_a_prefers_qwen() -> None:
    filename = parse_filename_metadata("sprite", filename="W_Axe014.png")
    qwen = {"category": "plant", "object_name": "mushroom", "tags": ["mushroom"], "confidence": 0.8}

    result = fuse_prefill_suggestions(
        filename,
        qwen,
        adjudication={"choice": "a", "reason": "The image shows a mushroom cap."},
    )

    assert result.fused_suggestion["category"] == "plant"
    assert result.fused_suggestion["object_name"] == "mushroom"
    assert result.fused_suggestion["source"] == "qwen_adjudicated"
    assert result.prefill_quality["agreement"] == "adjudicated_qwen"
    assert result.prefill_quality["needs_review"] is False


def test_fusion_adjudication_choice_b_prefers_filename() -> None:
    filename = parse_filename_metadata("sprite", filename="W_Axe014.png")
    qwen = {"category": "plant", "object_name": "mushroom", "tags": ["mushroom"], "confidence": 0.8}

    result = fuse_prefill_suggestions(
        filename,
        qwen,
        adjudication={"choice": "b", "reason": "The image shows an axe blade."},
    )

    assert result.fused_suggestion["category"] == "weapon"
    assert result.fused_suggestion["object_name"] == "axe"
    assert result.fused_suggestion["source"] == "filename_adjudicated"


def test_fusion_adjudication_both_wrong_uses_correction_and_reviews() -> None:
    filename = parse_filename_metadata("sprite", filename="W_Axe014.png")
    qwen = {"category": "plant", "object_name": "mushroom", "tags": ["mushroom"], "confidence": 0.8}

    result = fuse_prefill_suggestions(
        filename,
        qwen,
        adjudication={
            "choice": "both_wrong",
            "corrected_category": "tool",
            "corrected_object_name": "pickaxe",
            "reason": "The head has two points.",
        },
    )

    assert result.fused_suggestion["category"] == "tool"
    assert result.fused_suggestion["object_name"] == "pickaxe"
    assert result.fused_suggestion["source"] == "adjudicator_corrected"
    assert result.prefill_quality["needs_review"] is True
    assert result.prefill_quality["adjudication"]["choice"] == "both_wrong"


def test_fusion_marks_strong_conflict_as_needs_review() -> None:
    filename = parse_filename_metadata("sprite", filename="W_Axe014.png")
    qwen = {
        "category": "plant",
        "object_name": "mushroom",
        "tags": ["mushroom"],
        "confidence": 0.9,
    }

    result = fuse_prefill_suggestions(filename, qwen)

    assert result.prefill_quality["bucket"] == QUALITY_NEEDS_REVIEW
    assert QUALITY_FILENAME_QWEN_CONFLICT in result.prefill_quality["flags"]
    assert "axe" in result.fused_suggestion["tags"]
    assert "mushroom" in result.fused_suggestion["tags"]


def test_fusion_warning_only_keeps_filename_and_flags_warning() -> None:
    filename = parse_filename_metadata("sprite", filename="Ac_Necklace01.png")
    qwen = {"warnings": ["prefill request timed out after 60 seconds"]}

    result = fuse_prefill_suggestions(filename, qwen)

    assert result.fused_suggestion["object_name"] == "necklace"
    assert result.prefill_quality["bucket"] == "request_failure"
    assert QUALITY_WARNING_ONLY in result.prefill_quality["flags"]


def test_fusion_agreement_auto_fuses_and_merges_tags() -> None:
    filename = parse_filename_metadata("sprite", filename="I_C_Banana.png")
    qwen = {
        "category": "item_icon",
        "object_name": "banana",
        "tags": ["banana", "yellow"],
        "dominant_colors": ["yellow"],
        "confidence": 0.91,
        "filename_agreement": "agree",
    }

    result = fuse_prefill_suggestions(filename, qwen)

    assert result.prefill_quality["bucket"] == QUALITY_FUSED_AUTOMATICALLY
    assert result.fused_suggestion["tags"] == ["banana", "yellow", "fruit", "food", "consumable"]
