from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.dataset_v5.blind import deterministic_pixel_facts
from spritelab.dataset_v5.codex_blind import (
    CodexBlindError,
    ForbiddenMetadataError,
    _reconcile_field,
    audit_blind_value,
    blind_prompt,
    checkpoint_stop,
    claim_shards,
    deterministic_shards,
    evaluate_health_check,
    export_supervision_candidates,
    finalize_campaign,
    freeze_pass,
    ingest_compact_labels,
    output_schema,
    prepare_pass,
    reconcile_campaign,
    reconcile_source_metadata,
    render_blind_image,
    resume_status,
    stage_campaign,
    validate_batch,
)
from spritelab.dataset_v5.identity import canonical_rgba_bytes, decoded_rgba_sha256


def _rgba() -> np.ndarray:
    value = np.zeros((4, 6, 4), dtype=np.uint8)
    value[1:3, 1:5] = [130, 80, 40, 255]
    return value


def _record_id(seed: str = "record") -> str:
    return "rec_" + hashlib.sha256(seed.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _raw_fixture(root: Path, *, filename: str = "misleading_sword.png") -> tuple[str, str]:
    root.mkdir()
    (root / "raw_audit_blobs").mkdir()
    rgba = _rgba()
    image_hash = decoded_rgba_sha256(rgba)
    record_id = _record_id()
    blob = canonical_rgba_bytes(rgba)
    blob_path = root / "raw_audit_blobs" / f"{image_hash}.rgba"
    blob_path.write_bytes(blob)
    _write_jsonl(
        root / "blob_manifest.jsonl",
        [
            {
                "blob_file_sha256": hashlib.sha256(blob).hexdigest(),
                "blob_id": image_hash,
                "blob_path": f"raw_audit_blobs/{image_hash}.rgba",
                "height": 4,
                "width": 6,
            }
        ],
    )
    _write_jsonl(
        root / "extraction_manifest.jsonl",
        [
            {
                "blob_id": image_hash,
                "crop_coordinates": [0, 0, 6, 4],
                "decode_status": "verified_from_original",
                "deterministic_pixel_facts": deterministic_pixel_facts(rgba),
                "forensic_inclusion_decision": "quarantine",
                "height": 4,
                "interpolation_policy": "none",
                "record_id": record_id,
                "width": 6,
            },
            {
                "blob_id": None,
                "crop_coordinates": None,
                "decode_status": "not_decodable",
                "record_id": None,
            },
        ],
    )
    _write_jsonl(
        root / "provenance_manifest.jsonl",
        [
            {
                "creator_or_publishers": ["Example Creator"],
                "original_filename": filename,
                "packs": ["Named Pack"],
                "provenance_status": "blocked",
                "record_id": record_id,
            }
        ],
    )
    _write_jsonl(
        root / "suitability_manifest.jsonl",
        [{"audit_status": "reject", "record_id": record_id}],
    )
    return record_id, image_hash


def _compact_label(
    record_id: str,
    *,
    canonical_object: str | None = "sword",
    category: str | None = "weapon",
    certainty: str = "high",
) -> dict[str, object]:
    return {
        "canonical_object": canonical_object,
        "category": category,
        "certainty": certainty,
        "color_evidence": "The visible pixels are mainly brown and gray with a black edge.",
        "description": "A narrow gray blade-like form extends from a brown handle.",
        "domain": "equipment_icon",
        "evidence": "A narrow pointed form and short handle are visibly separated.",
        "material_evidence": "Bright gray highlights create a metallic visual cue without proving a material.",
        "needs_individual_inspection": certainty == "low",
        "outline_colors": ["black"],
        "primary_colors": ["gray", "brown"],
        "quality_flags": [],
        "record_id": record_id,
        "role": "combat_weapon",
        "secondary_colors": [],
        "surface_alias": "gray blade with brown handle",
        "visual_form": "narrow pointed blade-like form with a short handle",
        "visual_material_cue": "metallic",
    }


def test_codex_blind_prompt_contains_no_forbidden_metadata_keys() -> None:
    assert audit_blind_value({"instructions": blind_prompt(), "schema": output_schema()})["ok"] is True
    approved_collision = hashlib.sha256(b"source").hexdigest()
    assert audit_blind_value({"instructions": blind_prompt()}, forbidden_fingerprints=[approved_collision])["ok"]
    secret = hashlib.sha256(b"secretpackxyz").hexdigest()
    with pytest.raises(ForbiddenMetadataError):
        audit_blind_value({"note": "secretpackxyz"}, forbidden_fingerprints=[secret])
    with pytest.raises(ForbiddenMetadataError):
        audit_blind_value({"original_filename": "hidden.png"})


def test_codex_blind_render_is_fixed_nearest_neighbor_checkerboard() -> None:
    rendered = render_blind_image(_rgba())
    assert rendered.size == (128, 128)
    assert rendered.mode == "RGB"
    colors = set(rendered.get_flattened_data())
    assert (176, 176, 176) in colors
    assert (208, 208, 208) in colors
    assert (130, 80, 40) in colors


def test_codex_blind_staging_binds_record_image_and_preserves_status(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, image_hash = _raw_fixture(raw)
    output = tmp_path / "campaign"
    result = stage_campaign(raw, output, model_display="Codex displayed model", session_id="thread-test")

    assert result["readable_records"] == 1
    assert result["decode_failures"] == 1
    blind = json.loads((output / "blind_manifest.jsonl").read_text())
    assert blind["record_id"] == record_id
    assert blind["image_sha256"] == image_hash
    assert (output / "blind_images" / f"{record_id}.png").is_file()
    assert (output / "original_blobs" / f"{record_id}.rgba").is_file()
    eligibility = json.loads((output / "eligibility_manifest.jsonl").read_text())
    assert eligibility["provenance_status"] == "blocked"
    assert eligibility["suitability_status"] == "reject"
    serialized_blind = (output / "blind_manifest.jsonl").read_text() + (output / "blind_prompt.md").read_text()
    assert "misleading_sword.png" not in serialized_blind
    assert "Named Pack" not in serialized_blind


def test_same_image_under_different_source_names_produces_identical_blind_payload(tmp_path: Path) -> None:
    rows = []
    for index, filename in enumerate(("sword.png", "plant.png")):
        raw = tmp_path / f"raw_{index}"
        _raw_fixture(raw, filename=filename)
        output = tmp_path / f"campaign_{index}"
        stage_campaign(raw, output, model_display="Codex", session_id=f"thread-{index}")
        rows.append(json.loads((output / "blind_manifest.jsonl").read_text()))
    assert rows[0] == rows[1]


def test_codex_blind_opaque_image_names_have_no_semantic_words(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread")
    names = [path.name for path in (output / "blind_images").iterdir()]
    assert names == [f"{record_id}.png"]
    assert all(word not in names[0] for word in ("sword", "plant", "pack", "creator"))


def test_codex_blind_deterministic_sharding_and_pass_b_shuffle() -> None:
    rows = [{"record_id": _record_id(f"record-{index}"), "image_sha256": f"{index:064x}"} for index in range(57)]
    a_first = deterministic_shards(rows, "A")
    a_second = deterministic_shards(list(reversed(rows)), "A")
    b_first = deterministic_shards(rows, "B")
    assert a_first == a_second
    assert [len(shard) for shard in a_first] == [25, 25, 7]
    assert [row["record_id"] for shard in a_first for row in shard] != [
        row["record_id"] for shard in b_first for row in shard
    ]
    assert len({row["record_id"] for shard in a_first for row in shard}) == 57


def test_codex_blind_pass_inputs_are_isolated_and_use_different_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread-a")
    prepare_pass(output, "A")
    marker = "pass-a-secret-output"
    (output / "pass_a" / "shard_0000" / "untrusted_prior.txt").write_text(marker)
    prepare_pass(output, "B")

    a = json.loads((output / "pass_a" / "shard_0000" / "batch_payload.json").read_text())
    b_text = (output / "pass_b" / "shard_0000" / "batch_payload.json").read_text()
    b = json.loads(b_text)
    assert a["field_order"] != b["field_order"]
    assert marker not in b_text
    assert "pass_a" not in b_text
    sheet_a = output / "contact_sheets_pass_a" / "shard_0000_sheet_00.png"
    sheet_b = output / "contact_sheets_pass_b" / "shard_0000_sheet_00.png"
    assert sheet_a.read_bytes() != sheet_b.read_bytes()


def test_codex_blind_duplicate_shard_ownership_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread")
    prepare_pass(output, "A")
    claim_shards(output, "A", "agent-one", ["shard_0000"])
    with pytest.raises(CodexBlindError, match="duplicate shard ownership"):
        claim_shards(output, "A", "agent-two", ["shard_0000"])


def test_codex_blind_compact_ingest_validates_schema_and_is_resumable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread")
    prepare_pass(output, "A")
    shard = output / "pass_a" / "shard_0000"
    compact = _compact_label(record_id)
    _write_jsonl(shard / "compact_labels.jsonl", [compact])
    result = ingest_compact_labels(
        output,
        "A",
        "shard_0000",
        model_display="Codex displayed model",
        session_id="thread-a",
    )
    assert result["label_count"] == 1
    label = json.loads((shard / "labels.jsonl").read_text())
    assert label["fields"]["explicit_material"]["value"] is None
    assert label["fields"]["explicit_material"]["state"] == "unsupported"
    assert label["labeler"]["session_id"] == "thread-a"
    checkpoint = json.loads((shard / "checkpoint.json").read_text())
    assert checkpoint["jsonl_sha256"] == hashlib.sha256((shard / "labels.jsonl").read_bytes()).hexdigest()
    with pytest.raises(CodexBlindError, match="refusing to overwrite"):
        ingest_compact_labels(
            output,
            "A",
            "shard_0000",
            model_display="Codex",
            session_id="thread-a",
        )


def test_codex_blind_batch_hash_tamper_stops_before_inspection(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread")
    prepare_pass(output, "A")
    payload_path = output / "pass_a" / "shard_0000" / "batch_payload.json"
    payload = json.loads(payload_path.read_text())
    payload["records"][0]["image_sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CodexBlindError, match="record/image hash binding"):
        validate_batch(output, "A", "shard_0000")


def test_codex_blind_reconciliation_preserves_category_and_flags_object_conflict(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="stage-thread")
    prepare_pass(output, "A")
    prepare_pass(output, "B")
    _write_jsonl(output / "pass_a" / "shard_0000" / "compact_labels.jsonl", [_compact_label(record_id)])
    _write_jsonl(
        output / "pass_b" / "shard_0000" / "compact_labels.jsonl",
        [_compact_label(record_id, canonical_object="dagger")],
    )
    ingest_compact_labels(output, "A", "shard_0000", model_display="Codex", session_id="pass-a-thread")
    ingest_compact_labels(output, "B", "shard_0000", model_display="Codex", session_id="pass-b-thread")
    freeze_pass(output, "A")
    freeze_pass(output, "B")

    report = reconcile_campaign(output)
    reconciled = json.loads((output / "reconciled_labels.jsonl").read_text())
    assert report["reconciled_record_count"] == 1
    assert reconciled["fields"]["category"]["status"] == "codex_consistent"
    assert reconciled["fields"]["category"]["value"] == "weapon"
    assert reconciled["fields"]["canonical_object"]["status"] == "codex_conflicted"
    assert reconciled["fields"]["canonical_object"]["value"] is None
    assert reconciled["critical_status"] == "codex_conflicted"


def test_codex_blind_abstention_and_exact_material_conflict_are_preserved() -> None:
    abstained = _reconcile_field(
        "canonical_object",
        {"value": None, "state": "model_abstained", "confidence": "low", "visual_evidence": "ambiguous"},
        {"value": "sword", "state": "known", "confidence": "low", "visual_evidence": "weak silhouette"},
    )
    assert abstained["status"] == "codex_abstained"
    assert abstained["value"] is None
    material = _reconcile_field(
        "explicit_material",
        {"value": "iron", "state": "known", "confidence": "low", "visual_evidence": "gray color"},
        {"value": "bronze", "state": "known", "confidence": "low", "visual_evidence": "brown color"},
    )
    assert material["status"] == "codex_conflicted"
    assert material["value"] is None


def test_codex_blind_invalid_json_is_not_silently_repaired(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="thread")
    prepare_pass(output, "A")
    shard = output / "pass_a" / "shard_0000"
    (shard / "compact_labels.jsonl").write_text('{"record_id":', encoding="utf-8")
    with pytest.raises(CodexBlindError, match="invalid JSONL"):
        ingest_compact_labels(output, "A", "shard_0000", model_display="Codex", session_id="fresh-turn")
    assert not (shard / "labels.jsonl").exists()
    assert not (shard / "checkpoint.json").exists()


def test_codex_blind_health_gate_stops_on_critical_disagreement(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw)
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="stage-thread")
    prepare_pass(output, "A")
    prepare_pass(output, "B")
    _write_jsonl(output / "pass_a" / "shard_0000" / "compact_labels.jsonl", [_compact_label(record_id)])
    _write_jsonl(
        output / "pass_b" / "shard_0000" / "compact_labels.jsonl",
        [_compact_label(record_id, canonical_object="dagger", category="tool")],
    )
    ingest_compact_labels(output, "A", "shard_0000", model_display="Codex", session_id="pass-a-thread")
    ingest_compact_labels(output, "B", "shard_0000", model_display="Codex", session_id="audit-fresh-thread")
    audit_dir = output / "health_checks" / "check_000100"
    audit_dir.mkdir()
    b_batch = json.loads((output / "pass_b" / "shard_0000" / "batch_payload.json").read_text())
    (audit_dir / "batch_payload.json").write_text(
        json.dumps(
            {
                "pre_inspection_audit": {"forbidden_metadata_leakage": 0},
                "records": b_batch["records"],
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "audit_labels.jsonl").write_bytes((output / "pass_b" / "shard_0000" / "labels.jsonl").read_bytes())
    report = evaluate_health_check(output, 100)
    assert report["passed"] is False
    assert report["critical_field_disagreement_rate"] > 0.05
    assert "critical_field_disagreement" in report["gate_failures"]
    assert json.loads((output / "progress.json").read_text())["campaign_status"] == "stopped_health_gate"


def test_codex_blind_source_metadata_requires_freeze_and_never_replaces_visual_label(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw, filename="semantic_sword.png")
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="stage-thread")
    with pytest.raises(CodexBlindError, match="before Pass A and Pass B are frozen"):
        reconcile_source_metadata(output, raw)
    prepare_pass(output, "A")
    prepare_pass(output, "B")
    for pass_id, session_id in (("A", "pass-a-thread"), ("B", "pass-b-thread")):
        _write_jsonl(
            output / f"pass_{pass_id.lower()}" / "shard_0000" / "compact_labels.jsonl",
            [_compact_label(record_id)],
        )
        ingest_compact_labels(output, pass_id, "shard_0000", model_display="Codex", session_id=session_id)
        freeze_pass(output, pass_id)
    reconcile_campaign(output)
    report = reconcile_source_metadata(output, raw)
    row = json.loads((output / "source_reconciliation.jsonl").read_text())
    assert report["state_counts"]["source_metadata_unverifiable"] == 1
    assert row["semantic_filename_taint"] is True
    assert row["visual_label_replaced"] is False
    visual = json.loads((output / "reconciled_labels.jsonl").read_text())
    assert visual["fields"]["canonical_object"]["value"] == "sword"


def test_codex_blind_supervision_never_becomes_strong_and_reports_are_resumable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    record_id, _ = _raw_fixture(raw)
    (raw / "rebuild_report.json").write_text(
        json.dumps({"freeze": {"production_frozen": False}, "raw_source_gate_passed": False}),
        encoding="utf-8",
    )
    output = tmp_path / "campaign"
    stage_campaign(raw, output, model_display="Codex", session_id="stage-thread")
    prepare_pass(output, "A")
    prepare_pass(output, "B")
    for pass_id, session_id in (("A", "pass-a-thread"), ("B", "pass-b-thread")):
        _write_jsonl(
            output / f"pass_{pass_id.lower()}" / "shard_0000" / "compact_labels.jsonl",
            [_compact_label(record_id)],
        )
        ingest_compact_labels(output, pass_id, "shard_0000", model_display="Codex", session_id=session_id)
        freeze_pass(output, pass_id)
    reconcile_campaign(output)
    reconcile_source_metadata(output, raw)
    supervision = export_supervision_candidates(output)
    candidates = (output / "supervision_candidates.jsonl").read_text()
    assert supervision["candidate_count"] == 1
    assert "supervised_weak" in candidates
    assert "supervised_strong" not in candidates
    assert "deterministic_field_supervision_only" in candidates
    stop = checkpoint_stop(output, "active_context_unreliable")
    assert stop["usage_limit_stop"] is False
    resume = resume_status(output)
    assert resume["next_record_id"] is None
    assert "python -m spritelab.dataset_v5.codex_blind resume" in resume["resume_command"]
    report = finalize_campaign(output, raw)
    assert report["labels_are_calibrated_truth"] is False
    assert report["consistent_critical_record_count"] == 1
    for name in (
        "campaign_report.md",
        "campaign_report.json",
        "label_distribution_report.json",
        "artifact_hashes.json",
    ):
        assert (output / name).is_file()
    artifact_hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert "blind_images" not in "\n".join(artifact_hashes["artifacts"])
    assert "reconciled_labels.jsonl" in artifact_hashes["artifacts"]
