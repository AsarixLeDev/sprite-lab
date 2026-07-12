from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.label_v2_audit import summarize_label_v2_upstream
from spritelab.harvest.label_v2_pipeline import _compact_visual_audit_codes
from spritelab.harvest.visual_facts import VisualFacts


def test_upstream_audit_summary_covers_statuses_and_compact_codes(tmp_path: Path) -> None:
    imported = [
        {"sprite_id": "accepted", "status": "accepted"},
        {"sprite_id": "quarantined", "status": "quarantine"},
        {"sprite_id": "review", "status": "needs_review"},
        {"sprite_id": "rejected", "status": "rejected", "errors": ["invalid_bundle"]},
    ]
    predictions = [
        {
            "sprite_id": row["sprite_id"],
            "source_profile": {"name": "generic_unknown"},
            "label_quality": {
                "bucket": "needs_review" if row["sprite_id"] == "review" else "auto_filename_trusted",
                "audit_codes": ["shape_hint_contradiction"] if row["sprite_id"] == "review" else [],
            },
        }
        for row in imported
    ]
    (tmp_path / "imported.jsonl").write_text("\n".join(json.dumps(row) for row in imported) + "\n", encoding="utf-8")
    (tmp_path / "label_v2_suggestions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8"
    )

    summary = summarize_label_v2_upstream(tmp_path)

    assert summary["status_counts"] == {"accepted": 1, "quarantine": 1, "rejected": 1, "review": 1}
    assert summary["confidence_tier_counts"]["T0"] == 3
    assert summary["confidence_tier_counts"]["T4"] == 1
    assert summary["audit_code_histogram"] == {"shape_hint_contradiction": 1}
    assert summary["review_or_rejection_reason_counts"] == {"invalid_bundle": 1}


def test_upstream_visual_audits_persist_compact_codes_only() -> None:
    facts = VisualFacts(
        content_bbox=(0, 0, 2, 2),
        content_width=2,
        content_height=2,
        opaque_pixel_count=4,
        alpha_hard=True,
        palette_size=1,
        dominant_colors=("blue",),
        aspect_hint="square",
        shape_hints=("small_content", "tall"),
    )
    suggestion = LabelSuggestion("item_icon", "round_orb", tags=("solid",))
    assert _compact_visual_audit_codes(suggestion, facts) == [
        "role_inference_contradiction",
        "shape_hint_contradiction",
    ]
