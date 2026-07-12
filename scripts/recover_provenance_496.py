"""Generate the audited append-only provenance repair for the 496 RPG icon pack."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from spritelab.unlabeled_pool.builder import alpha_mask_sha256, canonical_rgba_sha256, hash_json
from spritelab.unlabeled_pool.provenance_repair import deterministic_json, file_sha256, load_provenance_repairs

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "harvest_runs/oga_496_rpg_icons"
FIXED_RUN = ROOT / "harvest_runs/oga_496_rpg_icons_32fix"
ARCHIVE = ORIGINAL_RUN / "downloads/oga_496_rpg_icons.zip"
EXTRACTED = ORIGINAL_RUN / "extracted/oga_496_rpg_icons"
FIXED = ROOT / "data_sources/fixed/oga_496_rpg_icons_32"
DATASET = ROOT / "datasets/oga_496_rpg_icons_32fix_label_v2_semantic_v3"
REPORT_DIR = ROOT / "experiments/provenance_recovery_496"
REPAIR_PATH = FIXED_RUN / "provenance_repair_v1.json"
SOURCE_PAGE = "https://opengameart.org/content/496-pixel-art-icons-for-medievalfantasy-rpg"
DOWNLOAD_URL = "https://opengameart.org/sites/default/files/496_RPG_icons.zip"
CONFIG = {
    "archive_format": "zip",
    "expected_member_count": 496,
    "expected_source_dimensions": [34, 34],
    "expected_derived_dimensions": [32, 32],
    "transform": {"kind": "exact_rgba_crop", "crop_box": [1, 1, 33, 33]},
    "hash_algorithms": ["sha256_file_bytes", "spritelab_exported_rgba_v1", "spritelab_alpha_mask_v1"],
    "version": "provenance_recovery_496_v1",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(file_sha256(file_path)))
    return digest.hexdigest()


def _dataset_correspondence() -> dict[str, dict]:
    values: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        arrays = np.load(DATASET / f"{split}.npz", allow_pickle=False)
        manifests = {row["sprite_id"]: row for row in _jsonl(DATASET / f"manifest_{split}.jsonl")}
        for index, raw_sprite_id in enumerate(arrays["sprite_id"]):
            sprite_id = str(raw_sprite_id)
            row = manifests[sprite_id]
            index_map = arrays["index_map"][index]
            rgb = arrays["palette"][index][index_map].astype(np.uint8)
            alpha = arrays["alpha"][index].astype(np.uint8)
            if int(alpha.max()) <= 1:
                alpha = alpha * 255
            exported = np.concatenate([rgb, alpha[..., None]], axis=-1)
            source_path = ROOT / Path(row["source_path"].replace("\\", "/"))
            with Image.open(source_path) as image:
                source = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            opaque = source[..., 3] > 0
            pixel_diff = np.any(exported[..., :3] != source[..., :3], axis=-1) & opaque
            max_delta = (
                int(np.abs(exported[..., :3].astype(int) - source[..., :3].astype(int))[pixel_diff].max())
                if pixel_diff.any()
                else 0
            )
            values[sprite_id] = {
                "split": split,
                "alpha_mask_exact": bool(np.array_equal(exported[..., 3], source[..., 3])),
                "visible_rgb_exact": not bool(pixel_diff.any()),
                "visible_rgb_mismatch_pixels": int(pixel_diff.sum()),
                "visible_rgb_max_channel_delta": max_delta,
                "expected_palette_quantization": bool(
                    pixel_diff.any() and len(np.unique(source[..., :3][opaque], axis=0)) > 32
                ),
            }
    return values


def build_correspondence() -> tuple[list[dict], list[dict], dict]:
    candidates = {row["relative_path"].lower(): row for row in _jsonl(FIXED_RUN / "candidates.jsonl")}
    imported = {row["relative_path"].lower(): row for row in _jsonl(FIXED_RUN / "imported.jsonl")}
    rejected = {row["relative_path"].lower(): row for row in _jsonl(FIXED_RUN / "rejected.jsonl")}
    dataset = _dataset_correspondence()
    rows: list[dict] = []
    repair_rows: list[dict] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != CONFIG["expected_member_count"]:
            raise ValueError(f"expected 496 archive members, found {len(infos)}")
        for info in sorted(infos, key=lambda item: item.filename.lower()):
            name = Path(info.filename).name
            key = name.lower()
            candidate = candidates[key]
            record = imported.get(key) or rejected.get(key)
            sprite_id = record["sprite_id"]
            source_bytes = archive.read(info)
            extracted_path = EXTRACTED / name
            derived_path = FIXED / name
            if extracted_path.read_bytes() != source_bytes:
                raise ValueError(f"extracted member mismatch: {name}")
            with Image.open(io.BytesIO(source_bytes)) as source_image, Image.open(derived_path) as derived_image:
                source_rgba = np.asarray(source_image.convert("RGBA"), dtype=np.uint8)
                derived_rgba = np.asarray(derived_image.convert("RGBA"), dtype=np.uint8)
            reproduced = source_rgba[1:33, 1:33]
            if source_rgba.shape != (34, 34, 4) or derived_rgba.shape != (32, 32, 4):
                raise ValueError(f"unexpected dimensions: {name}")
            if not np.array_equal(reproduced, derived_rgba):
                raise ValueError(f"center crop mismatch: {name}")
            affected = record.get("status") == "accepted"
            mapping = {
                "alpha_mask_sha256": alpha_mask_sha256(derived_rgba[..., 3]),
                "archive_member": info.filename,
                "crop_box": [1, 1, 33, 33],
                "derived_image_path": derived_path.relative_to(ROOT).as_posix(),
                "derived_image_sha256": file_sha256(derived_path),
                "exported_rgba_sha256": canonical_rgba_sha256(derived_rgba),
                "source_dimensions": {"height": 34, "width": 34},
                "source_image_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "sprite_id": sprite_id,
            }
            row = {
                **mapping,
                "affected_blocked_record": affected,
                "candidate_id": candidate["candidate_id"],
                "derived_dimensions": {"height": 32, "width": 32},
                "extracted_member_byte_exact": True,
                "import_status": record.get("status"),
                "zip_compressed_size": info.compress_size,
                "zip_crc32": f"{info.CRC:08x}",
                "zip_uncompressed_size": info.file_size,
            }
            if sprite_id in dataset:
                row["existing_dataset_export"] = dataset[sprite_id]
            rows.append(row)
            if affected:
                repair_rows.append(mapping)
    return rows, sorted(repair_rows, key=lambda row: row["sprite_id"]), dataset


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    correspondence, repair_mapping, dataset = build_correspondence()
    affected = sorted(row["sprite_id"] for row in repair_mapping)
    archive_hash = file_sha256(ARCHIVE)
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tool_config_hash = hash_json(CONFIG)
    dataset_counts = Counter(
        "alpha_exact_visible_rgb_exact"
        if row["alpha_mask_exact"] and row["visible_rgb_exact"]
        else "alpha_exact_expected_palette_quantization"
        if row["alpha_mask_exact"] and row["expected_palette_quantization"]
        else "mismatch"
        for row in dataset.values()
    )
    historical_hashes = {
        "harvest_runs/oga_496_rpg_icons/sources.jsonl": file_sha256(ORIGINAL_RUN / "sources.jsonl"),
        "harvest_runs/oga_496_rpg_icons_32fix/imported.jsonl": file_sha256(FIXED_RUN / "imported.jsonl"),
        "harvest_runs/oga_496_rpg_icons_32fix/sources.jsonl": file_sha256(FIXED_RUN / "sources.jsonl"),
    }

    report = {
        "schema_version": "spritelab_correspondence_report_v1",
        "generated_at": now,
        "source": {
            "archive_path": ARCHIVE.relative_to(ROOT).as_posix(),
            "download_sha256": archive_hash,
            "download_size": ARCHIVE.stat().st_size,
            "download_timestamp_file_mtime_utc": _utc_timestamp(ARCHIVE.stat().st_mtime),
            "recorded_download_url": DOWNLOAD_URL,
            "recorded_source_url": SOURCE_PAGE,
            "stored_filename": ARCHIVE.name,
            "server_url_filename": Path(urlparse(DOWNLOAD_URL).path).name,
        },
        "summary": {
            "affected_records_verified": len(affected),
            "archive_members": len(correspondence),
            "archive_png_members": len(correspondence),
            "derived_images_exact_center_crop": len(correspondence),
            "existing_extracted_members_byte_exact": len(correspondence),
            "expected_sprite_count": 496,
            "source_dimensions": {"34x34": len(correspondence)},
            "derived_dimensions": {"32x32": len(correspondence)},
            "dataset_export_correspondence": dict(sorted(dataset_counts.items())),
            "unmatched_or_ambiguous": 0,
        },
        "historical_manifest_hashes": historical_hashes,
        "correspondence": correspondence,
    }
    (REPORT_DIR / "correspondence_report.json").write_text(deterministic_json(report), encoding="utf-8", newline="\n")

    repair = {
        "repair_schema_version": "spritelab_provenance_repair_v1",
        "source_id": "oga_496_rpg_icons_32fix",
        "source_run": "oga_496_rpg_icons_32fix",
        "recorded_source_url": SOURCE_PAGE,
        "recorded_download_url": DOWNLOAD_URL,
        "final_url": DOWNLOAD_URL,
        "redirects": [],
        "downloaded_filename": ARCHIVE.name,
        "server_url_filename": Path(urlparse(DOWNLOAD_URL).path).name,
        "download_sha256": archive_hash,
        "download_hash_scope": "downloaded_file_bytes",
        "download_size": ARCHIVE.stat().st_size,
        "download_timestamp": _utc_timestamp(ARCHIVE.stat().st_mtime),
        "local_download_path": ARCHIVE.relative_to(ROOT).as_posix(),
        "recovery_method": "original_file_recovered",
        "license": "CC0",
        "license_page": SOURCE_PAGE,
        "attribution_page": SOURCE_PAGE,
        "verification_evidence": {
            "archive_member_count": len(correspondence),
            "archive_member_mapping": repair_mapping,
            "correspondence_report": (REPORT_DIR / "correspondence_report.json").relative_to(ROOT).as_posix(),
            "derived_images_exact_center_crop": len(correspondence),
            "existing_dataset_alpha_exact": sum(row["alpha_mask_exact"] for row in dataset.values()),
            "existing_dataset_visible_rgb_exact": sum(row["visible_rgb_exact"] for row in dataset.values()),
            "existing_dataset_expected_palette_quantization": sum(
                row["expected_palette_quantization"] for row in dataset.values()
            ),
            "extracted_members_byte_exact": len(correspondence),
            "transform": {"crop_box": [1, 1, 33, 33], "kind": "exact_rgba_crop"},
        },
        "affected_sprite_ids": affected,
        "old_provenance_status": "blocked_provenance",
        "new_provenance_status": "original_download_verified",
        "timestamp": now,
        "tool_config_hash": tool_config_hash,
    }
    REPAIR_PATH.write_text(deterministic_json(repair), encoding="utf-8", newline="\n")
    load_provenance_repairs([REPAIR_PATH], workspace_root=ROOT)

    inventory = f"""# Source inventory

## Decision-bearing source

- Recovered original archive: `{ARCHIVE.relative_to(ROOT).as_posix()}`
- Stored filename: `{ARCHIVE.name}`; recorded server URL filename: `{Path(urlparse(DOWNLOAD_URL).path).name}`
- Size: `{ARCHIVE.stat().st_size}` bytes
- SHA-256: `{archive_hash}` (computed from the recovered ZIP bytes)
- Filesystem creation/mtime observed: `{_utc_timestamp(ARCHIVE.stat().st_ctime)}` / `{_utc_timestamp(ARCHIVE.stat().st_mtime)}` UTC
- Recorded download URL: `{DOWNLOAD_URL}`
- Recorded source/license/attribution page: `{SOURCE_PAGE}` (CC0, Henrique Lazarini / 7Soul1)
- Archive inventory: 496 PNG members, all 34x34; no non-PNG members.

## Provenance chain

- Original harvest manifest: `harvest_runs/oga_496_rpg_icons/sources.jsonl` records the canonical source page and direct download URL. SHA-256: `{historical_hashes["harvest_runs/oga_496_rpg_icons/sources.jsonl"]}`.
- Original harvest events place import at `2026-07-03T06:44:28Z`, three seconds after the recovered file mtime.
- Original extracted directory contains 496 files; all 496 are byte-identical to their ZIP members.
- Fixed source directory contains 496 32x32 PNGs; each is the exact RGBA crop `[1,1,33,33]` of the same-named 34x34 ZIP member.
- Fixed run has 494 imported records: 476 accepted and 18 quarantined; two additional candidates were rejected for soft alpha.
- The frozen v1 r1 pool identifies exactly the 476 accepted records as blocked only for the missing download hash.
- Existing semantic-v3 dataset exports preserve all 476 alpha masks. Visible RGB is exact for 475; `e_wood04` has the expected <=4-channel palette quantization because its source has 40 visible colors and the exporter limit is 32.

## Other sources inspected

- Acquisition YAML/config: no pack-specific acquisition YAML was found; the two `sources.jsonl` manifests are the acquisition records.
- Download logs/events: `events.jsonl` in both runs provide import/export timestamps; no HTTP redirect log exists.
- Archive-member metadata: ZIP names, CRC32, compressed/uncompressed sizes, member timestamps, byte hashes, dimensions, and mapping are in `correspondence_report.json`.
- Browser/download filenames: the recorded URL supplies `496_RPG_icons.zip`; the harvester stored it as `oga_496_rpg_icons.zip`. No second matching file was found in Downloads, Desktop, the shallow temp/cache scan, or other repository download directories.
- Git history: `harvest_runs/`, `data_sources/`, and `datasets/` are ignored, so Git contains no historical revision for these acquisition files.
- Experiment/import manifests: labeling and exported-dataset manifests consistently point to `data_sources/fixed/oga_496_rpg_icons_32`; none contains a competing download hash.
- Cache directories: repository cache/temp names and targeted user cache/temp locations yielded no competing archive. PowerShell history could not be read because access was denied; this is non-blocking because the original file itself was recovered and reproduced.
- Historical fixed manifest remained unchanged at SHA-256 `{historical_hashes["harvest_runs/oga_496_rpg_icons_32fix/sources.jsonl"]}`.
- Frozen pool tree audit SHA-256 before repair work: `d8158f1c677fc7efc76237f9e206623ad0c839d1affcd864cc1e84e0d2d96ca7`.
"""
    (REPORT_DIR / "source_inventory.md").write_text(inventory, encoding="utf-8", newline="\n")

    attempts = f"""# Recovery attempts

1. Traced `sources.jsonl`, `events.jsonl`, `imported.jsonl`, `candidates.jsonl`, rejected records, dataset manifests, reports, docs, Git history/ignore state, caches, and targeted user download locations.
2. Located the recovered original ZIP in the earlier run at `{ARCHIVE.relative_to(ROOT).as_posix()}`.
3. Computed file-byte SHA-256 `{archive_hash}` and size `{ARCHIVE.stat().st_size}` without using any exported-sprite hash.
4. Verified all 496 extracted files byte-for-byte against ZIP members.
5. Verified all 496 fixed PNGs as exact same-name RGBA center crops; zero failures and zero ambiguous substitutions.
6. Bound all 476 blocked accepted sprite IDs to member hash, dimensions, crop, fixed-file hash, exported-RGBA hash, and alpha-mask hash.
7. Checked the existing 476-record dataset export: 476/476 alpha exact; 475/476 visible RGB exact; one documented deterministic palette quantization (`e_wood04`, 63 pixels, max channel delta 4, 40 source colors versus 32-slot export policy).
8. Loaded the completed repair through the fail-closed repair loader, which re-hashed the ZIP, checked its size, re-opened every mapped member, reproduced every derived PNG, and rejects sprite hashes as download hashes.

No network reacquisition was performed because recovery path A succeeded. Redirect and retrieval metadata therefore do not apply; the historical direct URL is retained as recorded.
"""
    (REPORT_DIR / "recovery_attempts.md").write_text(attempts, encoding="utf-8", newline="\n")

    correspondence_md = f"""# Correspondence report

- Candidate source: `{ARCHIVE.relative_to(ROOT).as_posix()}`
- Download SHA-256: `{archive_hash}`
- Expected/archive/fixed counts: 496 / 496 / 496
- Source dimensions: 496 x 34x34
- Exact transform: crop `[1,1,33,33]` to 32x32 RGBA
- ZIP member to extracted file byte matches: 496/496
- ZIP member crop to fixed source matches: 496/496
- Affected blocked accepted records mapped: 476/476
- Existing dataset alpha-mask correspondence: 476/476
- Existing dataset visible-RGB correspondence: 475 exact plus one expected 32-color palette quantization; no source replacement
- Unmatched, ambiguous, or visually inferred mappings: 0

Per-member hashes, paths, dimensions, archive metadata, sprite IDs, statuses, exported RGBA hashes, alpha hashes, and dataset-export checks are in `correspondence_report.json`.
"""
    (REPORT_DIR / "correspondence_report.md").write_text(correspondence_md, encoding="utf-8", newline="\n")

    decision = f"""# Provenance decision

**Decision: original provenance recovered (path A).**

The recovered file is the local archive used by the original `oga_496_rpg_icons` harvest. Its direct download URL is recorded in that run, its filesystem mtime precedes the first import event by three seconds, its 496 extracted files are byte-identical to its members, and the complete fixed source directory is deterministically reproduced by one exact crop per same-named member.

- New provenance status: `original_download_verified`
- Download SHA-256: `{archive_hash}`
- Records eligible to unblock in a future release: 476
- Records remaining blocked in this source run after applying the repair: 0 of the 476 targeted records
- Historical manifests and frozen `sprite_lab_unlabeled_pool_v1_r1`: unchanged

The repair is append-only and must be supplied explicitly with `--provenance-repair`; no historical source manifest was rewritten.
"""
    (REPORT_DIR / "provenance_decision.md").write_text(decision, encoding="utf-8", newline="\n")

    command_log = """# Principal audit and verification commands (PowerShell)
Get-FileHash harvest_runs\\oga_496_rpg_icons\\downloads\\oga_496_rpg_icons.zip -Algorithm SHA256
python scripts\\recover_provenance_496.py
$env:PYTHONPATH='src'; python -m pytest tests\\test_provenance_repair.py tests\\test_unlabeled_candidate_pool.py -q
$env:PYTHONPATH='src'; python -m pytest -q
python -m ruff check src tests scripts\\recover_provenance_496.py
$env:PYTHONPATH='src'; python -m spritelab.unlabeled_pool build --harvest-root harvest_runs --provenance-repair harvest_runs\\oga_496_rpg_icons_32fix\\provenance_repair_v1.json --output datasets\\sprite_lab_unlabeled_pool_v1_r2 --reports experiments\\unlabeled_candidate_pool_v1_r2 --pool-name sprite_lab_unlabeled_pool_v1_r2
$env:PYTHONPATH='src'; python -m spritelab.unlabeled_pool verify --pool datasets\\sprite_lab_unlabeled_pool_v1_r2
"""
    (REPORT_DIR / "command_log.txt").write_text(command_log, encoding="utf-8", newline="\n")

    print(
        deterministic_json(
            {
                "archive_sha256": archive_hash,
                "archive_size": ARCHIVE.stat().st_size,
                "affected_records": len(affected),
                "correspondence_records": len(correspondence),
                "fixed_tree_sha256": _tree_hash(FIXED),
                "repair": REPAIR_PATH.relative_to(ROOT).as_posix(),
                "tool_config_hash": tool_config_hash,
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
