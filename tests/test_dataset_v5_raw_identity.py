from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.dataset_v5.blind import (
    BlindInput,
    audit_blind_payload,
    blind_cache_key,
    build_blind_request,
    requests_equal_except_id,
)
from spritelab.dataset_v5.identity import (
    RecordBinding,
    decoded_rgba_sha256,
    is_opaque_geometry_id,
    is_opaque_record_id,
    make_geometry_family_id,
    make_record_id,
    relation_membership_key,
)
from spritelab.dataset_v5.taint import detect_filename_taint, reconcile_metadata_taint


def _sprite() -> np.ndarray:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, 1:7] = [125, 82, 47, 255]
    rgba[1, 3:5] = [220, 190, 125, 255]
    rgba[6, 2:6] = [55, 35, 20, 255]
    return rgba


def _binding(rgba: np.ndarray) -> RecordBinding:
    return RecordBinding(
        source_archive_sha256="a" * 64,
        archive_member_path="immutable/member/0001.png",
        extraction_operation="decode_member_rgba",
        crop_coordinates=None,
        decoded_rgba_sha256=decoded_rgba_sha256(rgba),
    )


def test_opaque_ids_contain_no_semantic_tokens() -> None:
    rgba = _sprite()
    record_id = make_record_id(_binding(rgba))
    geometry_id = make_geometry_family_id(rgba[:, :, 3])
    assert is_opaque_record_id(record_id)
    assert is_opaque_geometry_id(geometry_id)
    for forbidden in ("helmet", "mineral", "weapon", "armor", "tool", "pack"):
        assert forbidden not in record_id
        assert forbidden not in geometry_id


def test_same_pixels_under_misleading_names_have_identical_blind_requests(tmp_path: Path) -> None:
    rgba = _sprite()
    source = tmp_path / "source.png"
    Image.fromarray(rgba, mode="RGBA").save(source)
    names = ("helmet.png", "mineral.png", "sword.png", "unknown_0001.png")
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(source.read_bytes())
        paths.append(path)

    record_id = make_record_id(_binding(rgba))
    requests = []
    image_hashes = []
    geometry_ids = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            decoded = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        image_hashes.append(decoded_rgba_sha256(decoded))
        geometry_ids.append(make_geometry_family_id(decoded[:, :, 3]))
        blind_input = BlindInput.from_rgba(record_id, decoded)
        request = build_blind_request(
            blind_input,
            model="gpt-5.6-sol",
            request_id=f"req_{index}",
            pass_kind="adjudication",
        )
        audit_blind_payload(request, forbidden_metadata={"original_source_filename": path.name})
        requests.append(request)

    assert len(set(image_hashes)) == 1
    assert len(set(geometry_ids)) == 1
    assert all(requests_equal_except_id(requests[0], request) for request in requests[1:])
    keys = {blind_cache_key(request, endpoint_identity="provider/sol", provider="example") for request in requests}
    assert len(keys) == 1
    serialized = json.dumps(requests[0], sort_keys=True)
    assert all(name not in serialized for name in names)


def test_semantic_local_rename_does_not_change_membership_or_relation_grouping() -> None:
    rgba = _sprite()
    record_ids = [make_record_id(_binding(rgba)) for _ in ("helmet.png", "mineral.png", "sword.png")]
    assert len(set(record_ids)) == 1
    assert relation_membership_key(record_ids) == relation_membership_key([record_ids[0]])


def test_source_filename_is_provenance_only_and_later_flagged_as_tainted() -> None:
    rgba = _sprite()
    record_id = make_record_id(_binding(rgba))
    request = build_blind_request(
        BlindInput.from_rgba(record_id, rgba),
        model="gpt-5.6-sol",
        request_id="req_provenance_boundary",
        pass_kind="adjudication",
    )
    filename = "misleading_helmet.png"
    audit_blind_payload(request, forbidden_metadata={"original_source_filename": filename})
    provenance = {"record_id": record_id, "original_source_filename": filename}
    assert provenance["original_source_filename"] == filename
    taint = detect_filename_taint(filename)
    assert taint["status"] == "tainted_metadata"
    reconciliation = reconcile_metadata_taint(taint, {"canonical_object": "mineral", "category": "mineral"})
    assert reconciliation["blind_label_unchanged"] is True
    assert reconciliation["metadata_conflict"] is True


def test_consistency_request_uses_shuffled_fields_and_no_first_answer() -> None:
    rgba = _sprite()
    record_id = make_record_id(_binding(rgba))
    request = build_blind_request(
        BlindInput.from_rgba(record_id, rgba),
        model="gpt-5.6-sol",
        request_id="req_consistency",
        pass_kind="consistency",
    )
    serialized = json.dumps(request, sort_keys=True)
    assert '"pass_kind": "consistency"' in serialized
    assert "first_answer" not in serialized
    assert "prior_answer" not in serialized
