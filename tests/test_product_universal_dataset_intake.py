from __future__ import annotations

import io
import json
import os
import time
import zipfile
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from spritelab.product_core import ProjectContext
from spritelab.product_features.dataset import cli as dataset_cli
from spritelab.product_features.dataset import intake as dataset_intake
from spritelab.product_features.dataset import sidecar as dataset_sidecar
from spritelab.product_features.dataset import web as dataset_web
from spritelab.product_features.dataset.intake import (
    DatasetInputError,
    DatasetIntakeService,
    _require_source_entry_confined,
    build_dataset,
    discover_source_packs,
    inspect_dataset_folder,
)
from spritelab.product_features.dataset.plugin import build_plugin, create_plugin
from spritelab.product_features.dataset.review import DatasetReviewStore, ReviewDecisionError
from spritelab.product_features.dataset.sidecar import (
    PackMetadataError,
    apply_metadata_file,
    load_grouping,
    load_pack_metadata,
    merge_grouping_roots,
    metadata_file_template,
    save_grouping,
    save_pack_metadata,
    sidecar_is_applicable,
    validate_pack_metadata,
)
from spritelab.product_web.app import create_app
from spritelab.v3 import cli as v3_cli
from test_product_dataset_helpers import make_configured, make_png, save_same_rgba_with_metadata, tree_hashes


def _context(project: Path, *, output: Path | None = None) -> ProjectContext:
    runs = project / "runs" / "v3"
    runs.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"dataset": {}}
    if output is not None:
        config["dataset"]["output_root"] = str(output)
    return ProjectContext(project, config=config, runs_directory=runs)


def _metadata(*, source_type: str = "my_original_work", license_identifier: str = "cc0") -> dict[str, Any]:
    original = source_type == "my_original_work"
    return {
        "creator_or_rights_holder": "Synthetic Artist",
        "pack_title": "Synthetic Pack",
        "source_type": source_type,
        "source_page_url": None if original else "https://example.test/synthetic-pack",
        "original_work_declaration": original,
        "license_identifier": license_identifier,
        "license_url": "https://example.test/license" if license_identifier != "unknown" else None,
        "license_evidence_file": None,
        "attribution_text": "Synthetic Artist" if license_identifier in {"cc_by", "cc_by_sa"} else None,
        "permission_confirmed": license_identifier in {"custom", "private_permission"},
    }


def _items(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "items.jsonl").read_text(encoding="utf-8").splitlines()]


def _csrf(client: TestClient) -> dict[str, str]:
    return {"x-csrf-token": client.app.state.spritelab_csrf_token}


def _await_dataset_build(client: TestClient, response: Any) -> dict[str, Any]:
    assert response.status_code == 202
    status_url = str(response.json()["status_url"])
    for _ in range(200):
        status_response = client.get(status_url)
        assert status_response.status_code == 200
        job = status_response.json()
        if job["status"] == "complete":
            assert isinstance(job["result"], dict)
            return job["result"]
        if job["status"] == "failed":
            pytest.fail(str(job.get("message") or "Dataset build failed."))
        time.sleep(0.01)
    pytest.fail("Dataset build did not complete before the test timeout.")


def _ambiguous_sheet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (64, 16), (0, 0, 0, 0))
    for start, width in ((2, 6), (18, 8), (34, 5), (50, 8)):
        for y in range(4, 12):
            for x in range(start, start + width):
                image.putpixel((x, y), (80 + start, 210, 100, 255))
    image.save(path)
    return path


def test_root_images_nested_packs_and_case_insensitive_evidence(tmp_path: Path) -> None:
    root = tmp_path / "folder with spaces Ω"
    root.mkdir()
    (root / "source.txt").write_text(
        "Name: Root Pack\nCreator: Root Artist\nhttps://example.test/root\n", encoding="utf-8"
    )
    (root / "LiCeNsE.TxT").write_text("CC0\n", encoding="utf-8")
    make_png(root / "root.png")
    for index, name in enumerate(("pack_a", "pack_b")):
        pack = root / name
        pack.mkdir()
        (pack / "source.yaml").write_text(
            yaml.safe_dump({"name": name, "creator": f"{name} artist", "url": f"https://example.test/{name}"}),
            encoding="utf-8",
        )
        (pack / "license.yaml").write_text("license: MIT\n", encoding="utf-8")
        make_png(pack / "images" / f"{name}.png", color=(80, 110 + index * 30, 220, 255))
    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    assert inspection["image_count"] == 3
    assert {pack["relative_root"] for pack in inspection["packs"]} == {".", "pack_a", "pack_b"}
    assert build_dataset(root, output_root=tmp_path / "out").data["counts"]["accepted"] == 3


def test_png_symlink_cannot_escape_the_approved_source(tmp_path: Path) -> None:
    outside = make_png(tmp_path / "outside" / "secret.png", color=(17, 93, 211, 255))
    root = tmp_path / "approved"
    root.mkdir()
    link = root / "escaped.png"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows test session.")
    outside_before = outside.read_bytes()

    with pytest.raises(DatasetInputError, match=r"escape|outside|approved|symbolic|link"):
        discover_source_packs(root, context=_context(tmp_path / "project"))

    assert outside.read_bytes() == outside_before


def test_evidence_symlink_cannot_escape_the_approved_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "source.txt"
    outside.parent.mkdir()
    outside.write_text("Name: Secret pack\nCreator: Outside Artist\nhttps://example.test/outside\n", encoding="utf-8")
    root = tmp_path / "approved"
    make_png(root / "sprite.png")
    (root / "LICENSE").write_text("CC0\n", encoding="utf-8")
    try:
        os.symlink(outside, root / "source.txt")
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows test session.")

    with pytest.raises(DatasetInputError, match=r"escape|outside|approved|evidence|link"):
        inspect_dataset_folder(root, context=_context(tmp_path / "project"))


def test_archive_symlink_cannot_escape_the_approved_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "pack.zip"
    outside.parent.mkdir()
    outside.write_bytes(b"synthetic outside archive")
    root = tmp_path / "approved"
    make_png(root / "pack" / "sprite.png")
    try:
        os.symlink(outside, root / "pack.zip")
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows test session.")

    with pytest.raises(DatasetInputError, match=r"escape|outside|approved|archive|link"):
        discover_source_packs(root, context=_context(tmp_path / "project"))


def test_png_symlink_cannot_launder_pixels_across_source_packs(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    pack_a = make_configured(root / "licensed-a")
    pack_b = make_configured(root / "independent-b", license_text="MIT")
    make_png(pack_a / "owned.png")
    target = make_png(pack_b / "sprite.png", color=(17, 93, 211, 255))
    try:
        os.symlink(target, pack_a / "borrowed.png")
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows test session.")

    with pytest.raises(DatasetInputError, match=r"symbolic-link|provenance|Linked entries"):
        discover_source_packs(root, context=_context(tmp_path / "project"))


def test_source_confinement_rejects_reparse_seam_without_platform_symlink_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "approved"
    candidate = make_png(root / "sprite.png")
    original_is_symlink = Path.is_symlink

    def synthetic_reparse(path: Path) -> bool:
        return path == candidate or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_reparse)

    with pytest.raises(DatasetInputError, match=r"symbolic-link|provenance|Linked entries"):
        _require_source_entry_confined(candidate, root.resolve())


def test_source_discovery_fails_closed_when_walk_reports_an_unreadable_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "approved"
    root.mkdir()

    def unreadable_walk(*args: Any, **kwargs: Any) -> list[tuple[str, list[str], list[str]]]:
        kwargs["onerror"](PermissionError(13, "permission denied", str(root / "private")))
        return []

    monkeypatch.setattr(dataset_intake.os, "walk", unreadable_walk)

    with pytest.raises(DatasetInputError, match=r"unreadable|private"):
        discover_source_packs(root, context=_context(tmp_path / "project"))


def test_oga_and_kenney_presets_prefill_platform_but_never_license(tmp_path: Path) -> None:
    root = tmp_path / "packs"
    for name, url in (
        ("oga", "https://opengameart.org/content/synthetic"),
        ("kenney", "https://kenney.nl/assets/demo"),
    ):
        pack = root / name
        pack.mkdir(parents=True)
        (pack / "source.txt").write_text(f"Name: Synthetic {name}\nCreator: Artist\n{url}\n", encoding="utf-8")
        make_png(pack / "sprite.png", color=(80, 140 if name == "oga" else 160, 220, 255))
    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    presets = {pack["relative_root"]: pack["prefill"] for pack in inspection["packs"]}
    assert presets["oga"]["source_type"] == "opengameart"
    assert presets["kenney"]["source_type"] == "kenney"
    assert all(prefill["license_assumed_from_platform"] is False for prefill in presets.values())
    assert all("license_identifier" not in prefill for prefill in presets.values())


def test_ambiguous_inferred_boundary_requires_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "input"
    make_png(root / "bundle" / "source_one" / "a.png")
    make_png(root / "bundle" / "source_two" / "b.png", color=(30, 170, 230, 255))
    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    assert inspection["grouping_confirmation_required"] is True
    assert inspection["packs"][0]["relative_root"] == "bundle"
    assert inspection["packs"][0]["proposed_children"] == ["bundle/source_one", "bundle/source_two"]


def test_archive_boundary_is_hashed_but_never_unpacked(tmp_path: Path) -> None:
    root = tmp_path / "input"
    make_png(root / "downloaded_pack" / "visible.png")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("nested/hidden.png", b"not decoded or unpacked")
        archive.writestr("nested/again.zip", b"nested archive remains untouched")
    archive_path = root / "downloaded_pack.zip"
    archive_path.write_bytes(payload.getvalue())
    before = tree_hashes(root)
    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    pack = inspection["packs"][0]
    assert pack["boundary_evidence"] == "original_archive_boundary"
    assert pack["archive"]["sha256"]
    assert inspection["image_count"] == 1
    assert tree_hashes(root) == before


def test_original_work_sidecar_is_project_scoped_and_bound_to_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "original assets"
    make_png(root / "my sprite.png")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    before = tree_hashes(root)
    record = save_pack_metadata(
        project, source_root, packs[0], _metadata(), covered_byte_hashes=[next(iter(before.values()))]
    )
    assert tree_hashes(root) == before
    assert record["binding"]["canonical_source_path"] == str(root.resolve())
    assert record["binding"]["covered_file_identities"] == {"my sprite.png": next(iter(before.values()))}
    result = build_dataset(root, output_root=tmp_path / "out", context=context)
    assert result.data["counts"]["accepted"] == 1
    assert not (root / "source.yaml").exists()
    assert len(load_pack_metadata(project)) == 1


def test_tampered_sidecar_declaration_cannot_lift_legal_quarantine(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "download"
    make_png(root / "sprite.png")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        _metadata(source_type="other_downloaded", license_identifier="unknown"),
        covered_byte_hashes=list(tree_hashes(root).values()),
    )
    assert build_dataset(root, output_root=tmp_path / "before", context=context).data["counts"]["quarantined"] == 1
    sidecar_path = next((project / "datasets" / "source_metadata").glob("pack_*.json"))
    tampered = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tampered.update(
        {
            "license_identifier": "cc0",
            "license_url": "https://example.test/fabricated-license",
        }
    )
    sidecar_path.write_text(json.dumps(tampered), encoding="utf-8")

    loaded = load_pack_metadata(project)[packs[0].pack_id]
    assert sidecar_is_applicable(loaded, packs[0], source_root) is False
    result = build_dataset(root, output_root=tmp_path / "after", context=context)
    assert result.data["counts"]["accepted"] == 0
    assert result.data["counts"]["quarantined"] == 1


@pytest.mark.parametrize("project_location", ["same", "nested"])
def test_project_side_metadata_write_cannot_overlap_the_approved_source(tmp_path: Path, project_location: str) -> None:
    root = tmp_path / "approved-source"
    make_png(root / "sprite.png")
    project = root if project_location == "same" else root / ".spritelab-project"
    project.mkdir(parents=True, exist_ok=True)
    context = ProjectContext(project, config={}, runs_directory=tmp_path / "outside-runs")
    source_root, _paths, packs = discover_source_packs(root, context=context)
    before = tree_hashes(root)

    with pytest.raises(PackMetadataError, match=r"outside|overlap|source|input"):
        save_pack_metadata(
            project,
            source_root,
            packs[0],
            _metadata(),
            covered_byte_hashes=[before["sprite.png"]],
        )

    assert tree_hashes(root) == before
    assert not (project / "datasets" / "source_metadata").exists()


def test_project_side_grouping_write_cannot_overlap_the_approved_source(tmp_path: Path) -> None:
    root = tmp_path / "approved-source"
    make_png(root / "sprite.png")
    before = tree_hashes(root)

    with pytest.raises(PackMetadataError, match=r"outside|overlap|source|input"):
        save_grouping(root, root, ["."])

    assert tree_hashes(root) == before
    assert not (root / "datasets" / "source_metadata").exists()


def test_metadata_store_creation_rejects_linked_project_child_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    datasets = project / "datasets"
    datasets.mkdir(parents=True)
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_text("must remain unchanged", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path == datasets or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)

    with pytest.raises(OSError, match=r"linked|reparse"):
        save_grouping(project, root, ["."])

    assert outside.read_text(encoding="utf-8") == "must remain unchanged"
    assert not (datasets / "source_metadata").exists()


def test_concurrent_grouping_merges_preserve_every_confirmed_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    root.mkdir()
    barrier = Barrier(2)

    def merge(value: str) -> None:
        barrier.wait()
        merge_grouping_roots(project, root, [value])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(merge, value) for value in ("pack-a", "pack-b")]
        for future in futures:
            future.result()

    assert load_grouping(project, root)["confirmed_pack_roots"] == ["pack-a", "pack-b"]


def test_pack_metadata_save_rejects_grouping_change_before_locked_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "input"
    make_png(root / "bundle" / "source_one" / "a.png")
    make_png(root / "bundle" / "source_two" / "b.png", color=(30, 170, 230, 255))
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    assert len(packs) == 1
    original_prepare = dataset_sidecar._prepare_pack_metadata_record
    grouping_changed = False

    def change_grouping_before_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal grouping_changed
        record = original_prepare(*args, **kwargs)
        if not grouping_changed:
            grouping_changed = True
            save_grouping(project, source_root, ["bundle/source_one", "bundle/source_two"])
        return record

    monkeypatch.setattr(dataset_sidecar, "_prepare_pack_metadata_record", change_grouping_before_commit)

    with pytest.raises(PackMetadataError, match=r"grouping|membership|inspect"):
        save_pack_metadata(
            project,
            source_root,
            packs[0],
            _metadata(),
            covered_byte_hashes=list(tree_hashes(root).values()),
        )

    assert load_pack_metadata(project) == {}


def test_grouping_reader_rejects_linked_canonical_file_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    root.mkdir()
    save_grouping(project, root, ["."])
    grouping = dataset_sidecar.grouping_path(project, root)
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_text("must remain unchanged", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path == grouping or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)

    with pytest.raises(PackMetadataError, match=r"unreadable|corrupt"):
        load_grouping(project, root)

    assert outside.read_text(encoding="utf-8") == "must remain unchanged"


def test_grouping_reader_fails_closed_on_corrupt_canonical_record(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    root.mkdir()
    save_grouping(project, root, ["."])
    dataset_sidecar.grouping_path(project, root).write_text("{not-json", encoding="utf-8")

    with pytest.raises(PackMetadataError, match=r"unreadable|corrupt"):
        load_grouping(project, root)


def test_cli_rejects_project_run_and_output_writes_inside_source_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "approved-source"
    make_png(root / "sprite.png")
    (root / "spritelab.yaml").write_text(
        "project:\n  name: overlap-test\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    before = tree_hashes(root)
    args = Namespace(
        folder=str(root),
        output=root / "dataset-output",
        metadata_file=None,
        no_review=True,
        allow_hosted=False,
        provider_factory=None,
        json=False,
        no_color=False,
        quiet=False,
        debug=False,
    )

    result = dataset_cli._handle_build(args, [])

    assert result.status.value == "BLOCKED"
    assert result.data["build_started"] is False
    assert tree_hashes(root) == before
    assert not (root / "runs").exists()
    assert not (root / "dataset-output").exists()


def test_web_rejects_project_run_and_output_writes_inside_source_before_mutation(tmp_path: Path) -> None:
    root = tmp_path / "approved-source"
    make_png(root / "sprite.png")
    context = ProjectContext(root, config={"dataset": {}}, runs_directory=root / "runs" / "v3")
    client = TestClient(create_app(context, plugins=(create_plugin(folder_chooser=lambda: root),)))
    before = tree_hashes(root)
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={}).json()

    response = client.post(
        "/dataset/api/build",
        headers=_csrf(client),
        json={"approval_id": chosen["approval"]["approval_id"], "confirm_hosted": False},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "dataset_write_boundary_overlap"
    assert tree_hashes(root) == before
    assert not (root / "runs").exists()
    assert not (root / "datasets").exists()


def test_build_rejects_input_nested_beneath_managed_output_before_destructive_cleanup(tmp_path: Path) -> None:
    output = tmp_path / "dataset-output"
    root = make_configured(output / "raw_extraction" / "selected-source")
    make_png(root / "sprite.png")
    before = tree_hashes(root)

    result = build_dataset(root, output_root=output)

    assert result.status.value == "BLOCKED"
    assert "must not contain one another" in result.message
    assert tree_hashes(root) == before
    assert root.is_dir()


@pytest.mark.parametrize(
    ("source_type", "original_declaration"),
    [
        ("opengameart", True),
        ("kenney", True),
        ("other_downloaded", True),
        ("custom_private", True),
        ("my_original_work", False),
    ],
)
def test_original_work_declaration_must_match_source_type(
    tmp_path: Path, source_type: str, original_declaration: bool
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    fields = _metadata(source_type=source_type)
    fields["original_work_declaration"] = original_declaration
    fields["license_url"] = None

    with pytest.raises(PackMetadataError, match=r"original_work_declaration|original work|source_type"):
        save_pack_metadata(
            project,
            source_root,
            packs[0],
            fields,
            covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
        )

    assert load_pack_metadata(project) == {}


@pytest.mark.parametrize(
    ("source_type", "license_identifier", "field", "value"),
    [
        ("my_original_work", "cc0", "original_work_declaration", "false"),
        ("other_downloaded", "custom", "permission_confirmed", "false"),
        ("other_downloaded", "private_permission", "permission_confirmed", 1),
    ],
)
def test_declaration_flags_require_actual_json_booleans(
    tmp_path: Path,
    source_type: str,
    license_identifier: str,
    field: str,
    value: object,
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    fields = _metadata(source_type=source_type, license_identifier=license_identifier)
    fields[field] = value

    with pytest.raises(PackMetadataError, match=rf"{field}.*JSON boolean"):
        save_pack_metadata(
            project,
            source_root,
            packs[0],
            fields,
            covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
        )

    assert load_pack_metadata(project) == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("creator_or_rights_holder", ["Synthetic Artist"]),
        ("pack_title", {"title": "Synthetic Pack"}),
        ("source_page_url", 42),
        ("license_url", ["https://example.test/license"]),
        ("notes", {"unsafe": "object"}),
    ],
)
def test_declaration_text_fields_require_actual_json_strings(field: str, value: object) -> None:
    fields = _metadata(source_type="other_downloaded")
    fields[field] = value

    with pytest.raises(PackMetadataError, match=rf"{field}.*JSON string"):
        validate_pack_metadata(fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_page_url", "file:///private/source"),
        ("source_page_url", "/relative/source"),
        ("license_url", "https://"),
        ("direct_download_url", "ftp://example.test/archive.zip"),
    ],
)
def test_declaration_urls_require_valid_http_or_https(field: str, value: str) -> None:
    fields = _metadata(source_type="other_downloaded")
    fields[field] = value

    with pytest.raises(PackMetadataError, match=rf"{field}.*HTTP\(S\)"):
        validate_pack_metadata(fields)


def test_metadata_template_rejects_same_layout_in_a_different_source_root(tmp_path: Path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    make_png(source_a / "sprite.png", color=(220, 80, 60, 255))
    make_png(source_b / "sprite.png", color=(40, 160, 230, 255))
    root_a, _paths_a, packs_a = discover_source_packs(source_a, context=_context(tmp_path / "project-a"))
    root_b, _paths_b, packs_b = discover_source_packs(source_b, context=_context(tmp_path / "project-b"))
    payload = metadata_file_template(root_a, packs_a)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PackMetadataError, match=r"source|pack|identity|binding"):
        apply_metadata_file(tmp_path / "project-b", root_b, packs_b, metadata_path)
    assert load_pack_metadata(tmp_path / "project-b") == {}
    assert root_a != root_b and packs_a[0].pack_id != packs_b[0].pack_id


def test_metadata_template_rejects_source_bytes_changed_after_generation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png", color=(220, 80, 60, 255))
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    make_png(root / "sprite.png", color=(40, 160, 230, 255))
    current_root, _current_paths, current_packs = discover_source_packs(root, context=_context(project))

    with pytest.raises(PackMetadataError, match=r"source|changed|identity|binding"):
        apply_metadata_file(project, current_root, current_packs, metadata_path)
    assert load_pack_metadata(project) == {}


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_metadata_template_rejects_source_membership_changed_after_generation(tmp_path: Path, mutation: str) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "a.png")
    make_png(root / "b.png", color=(40, 160, 230, 255))
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    if mutation == "add":
        make_png(root / "c.png", color=(120, 90, 210, 255))
    else:
        (root / "b.png").unlink()
    current_root, _paths, current_packs = discover_source_packs(root, context=_context(project))

    with pytest.raises(PackMetadataError, match=r"source|changed|identity|binding"):
        apply_metadata_file(project, current_root, current_packs, metadata_path)
    assert load_pack_metadata(project) == {}


def test_metadata_batch_rediscovers_and_rejects_a_new_png_when_caller_packs_are_stale(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    source_root, _paths, stale_packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, stale_packs)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    make_png(root / "added-after-template.png", color=(40, 160, 230, 255))

    with pytest.raises(PackMetadataError, match=r"membership|changed|binding|source"):
        apply_metadata_file(project, source_root, stale_packs, metadata_path)

    assert load_pack_metadata(project) == {}


def test_metadata_batch_rechecks_membership_after_initial_discovery_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_discovery = dataset_sidecar._discover_source_pngs
    calls = 0

    def add_png_after_first_discovery(input_root: Path) -> list[Path]:
        nonlocal calls
        paths = original_discovery(input_root)
        calls += 1
        if calls == 1:
            make_png(root / "added-during-apply.png", color=(40, 160, 230, 255))
        return paths

    monkeypatch.setattr(dataset_sidecar, "_discover_source_pngs", add_png_after_first_discovery)

    with pytest.raises(PackMetadataError, match=r"membership|identities|changed"):
        apply_metadata_file(project, source_root, packs, metadata_path)

    assert load_pack_metadata(project) == {}


def test_metadata_batch_rejects_newer_persisted_grouping_without_lost_update(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "a" / "sprite.png")
    make_png(root / "b" / "sprite.png", color=(40, 160, 230, 255))
    save_grouping(project, root, ["a"])
    source_root, _paths, stale_packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, stale_packs)
    assert payload["confirmed_pack_roots"] == ["a"]
    for row in payload["packs"]:
        row.update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    merge_grouping_roots(project, root, ["b"])

    with pytest.raises(PackMetadataError, match=r"grouping|regenerate|changed"):
        apply_metadata_file(project, source_root, stale_packs, metadata_path)

    assert load_grouping(project, root)["confirmed_pack_roots"] == ["a", "b"]
    assert load_pack_metadata(project) == {}


def test_metadata_batch_requires_exactly_one_row_for_every_current_pack(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    for name, color in (("a", (220, 80, 60, 255)), ("b", (40, 160, 230, 255))):
        pack = make_configured(root / name)
        make_png(pack / "sprite.png", color=color)
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"] = payload["packs"][:1]
    payload["packs"][0].update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PackMetadataError, match=r"exactly one|every current|missing pack"):
        apply_metadata_file(project, source_root, packs, metadata_path)

    assert load_pack_metadata(project) == {}


def test_empty_metadata_batch_is_rejected_without_creating_a_journal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "empty-source"
    root.mkdir()
    payload = {
        "schema_version": dataset_sidecar.PACK_METADATA_BATCH_SCHEMA,
        "canonical_input_root": str(root.resolve()),
        "confirmed_pack_roots": [],
        "packs": [],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PackMetadataError, match=r"at least one|PNG|source pack"):
        apply_metadata_file(project, root, [], metadata_path)

    assert not (project / "datasets" / "source_metadata").exists()


def test_locked_transaction_helper_treats_empty_entries_as_a_no_op(tmp_path: Path) -> None:
    store = tmp_path / "project" / "datasets" / "source_metadata"
    with dataset_sidecar._metadata_store_guard(store, create=True):
        dataset_sidecar._write_json_transaction_locked(store, ())

    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_template_rejects_a_forged_source_pack_path_outside_input(tmp_path: Path) -> None:
    root = tmp_path / "source"
    outside = make_png(tmp_path / "outside.png", color=(17, 93, 211, 255))
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(tmp_path / "project"))
    packs[0].image_relative_paths = ["../outside.png"]
    before = outside.read_bytes()

    with pytest.raises(PackMetadataError, match=r"confined|relative|pack|source"):
        metadata_file_template(source_root, packs)

    assert outside.read_bytes() == before


def test_metadata_template_rejects_evidence_changed_after_generation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    (root / "source.txt").write_text(
        "Name: Synthetic pack\nCreator: Synthetic Artist\nhttps://example.test/source\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text("CC0\n", encoding="utf-8")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    current_root, _paths, current_packs = discover_source_packs(root, context=_context(project))

    with pytest.raises(PackMetadataError, match=r"source|changed|identity|binding|evidence"):
        apply_metadata_file(project, current_root, current_packs, metadata_path)
    assert load_pack_metadata(project) == {}


def test_metadata_template_rejects_archive_changed_after_generation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "pack" / "sprite.png")
    with zipfile.ZipFile(root / "pack.zip", "w") as archive:
        archive.writestr("identity.txt", "original archive")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    assert packs[0].archive is not None
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with zipfile.ZipFile(root / "pack.zip", "w") as archive:
        archive.writestr("identity.txt", "changed archive")
    current_root, _paths, current_packs = discover_source_packs(root, context=_context(project))

    with pytest.raises(PackMetadataError, match=r"source|changed|identity|binding|archive"):
        apply_metadata_file(project, current_root, current_packs, metadata_path)

    assert load_pack_metadata(project) == {}


def test_metadata_batch_validation_is_atomic_across_packs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    for name, color in (("a", (220, 80, 60, 255)), ("b", (40, 160, 230, 255))):
        pack = root / name
        make_png(pack / "sprite.png", color=color)
        (pack / "source.txt").write_text(
            f"Name: Pack {name}\nCreator: Synthetic Artist\nhttps://example.test/{name}\n", encoding="utf-8"
        )
        (pack / "LICENSE").write_text("CC0\n", encoding="utf-8")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    for row in payload["packs"]:
        row.update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    make_png(root / "b" / "sprite.png", color=(180, 40, 190, 255))
    current_root, _paths, current_packs = discover_source_packs(root, context=_context(project))

    with pytest.raises(PackMetadataError, match=r"source|changed|identity|binding"):
        apply_metadata_file(project, current_root, current_packs, metadata_path)

    assert load_pack_metadata(project) == {}


def test_metadata_batch_rolls_back_if_a_sidecar_commit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    for name, color in (("a", (220, 80, 60, 255)), ("b", (40, 160, 230, 255))):
        pack = root / name
        make_png(pack / "sprite.png", color=color)
        (pack / "source.txt").write_text(
            f"Name: Pack {name}\nCreator: Synthetic Artist\nhttps://example.test/{name}\n", encoding="utf-8"
        )
        (pack / "LICENSE").write_text("CC0\n", encoding="utf-8")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    for row in payload["packs"]:
        row.update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_install = dataset_sidecar._install_transaction_payload
    sidecar_installs = 0

    def fail_second_sidecar(payload: Path, target: Path, **kwargs: Any) -> None:
        nonlocal sidecar_installs
        if target.name.startswith("pack_"):
            sidecar_installs += 1
            if sidecar_installs == 2:
                raise OSError("synthetic second-sidecar commit failure")
        original_install(payload, target, **kwargs)

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", fail_second_sidecar)

    with pytest.raises(PackMetadataError, match="committed atomically"):
        apply_metadata_file(project, source_root, packs, metadata_path)

    store = project / "datasets" / "source_metadata"
    assert load_pack_metadata(project) == {}
    assert not list(store.glob("pack_*.json"))
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_batch_reports_success_if_only_committed_transaction_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        {**_metadata(), "pack_title": "Before transaction"},
        covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
    )
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update({**_metadata(), "pack_title": "Committed transaction"})
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_cleanup = dataset_sidecar._cleanup_transaction_directory

    def fail_transaction_cleanup(_transaction: Path) -> None:
        raise OSError("synthetic post-commit transaction cleanup failure")

    monkeypatch.setattr(dataset_sidecar, "_cleanup_transaction_directory", fail_transaction_cleanup)

    applied = apply_metadata_file(project, source_root, packs, metadata_path)

    assert applied["applied_pack_ids"] == [packs[0].pack_id]
    store = project / "datasets" / "source_metadata"
    sidecar = next(store.glob("pack_*.json"))
    assert json.loads(sidecar.read_text(encoding="utf-8"))["pack_title"] == "Committed transaction"
    assert list((store / ".transactions").glob("txn_*"))

    monkeypatch.setattr(dataset_sidecar, "_cleanup_transaction_directory", original_cleanup)
    assert load_pack_metadata(project)[packs[0].pack_id]["pack_title"] == "Committed transaction"
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_batch_recovers_the_complete_old_generation_after_process_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = tmp_path / "source"
    for name, color in (("a", (220, 80, 60, 255)), ("b", (40, 160, 230, 255))):
        pack_root = make_configured(root / name)
        make_png(pack_root / "sprite.png", color=color)
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    for pack in packs:
        save_pack_metadata(
            project,
            source_root,
            pack,
            {**_metadata(source_type="other_downloaded"), "pack_title": f"Old {pack.relative_root}"},
            covered_byte_hashes=[tree_hashes(root)[pack.image_relative_paths[0]]],
        )
    payload = metadata_file_template(source_root, packs)
    for row in payload["packs"]:
        row.update(
            {
                **_metadata(source_type="other_downloaded"),
                "pack_title": f"New {row['pack_relative_root']}",
            }
        )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_install = dataset_sidecar._install_transaction_payload
    installed = 0

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_first_install(payload_path: Path, target: Path, **kwargs: Any) -> None:
        nonlocal installed
        original_install(payload_path, target, **kwargs)
        if target.name.startswith("pack_"):
            installed += 1
            if installed == 1:
                raise SimulatedProcessCrash

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", crash_after_first_install)

    with pytest.raises(SimulatedProcessCrash):
        apply_metadata_file(project, source_root, packs, metadata_path)

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", original_install)
    recovered = load_pack_metadata(project)

    assert {record["pack_title"] for record in recovered.values()} == {"Old a", "Old b"}
    store = project / "datasets" / "source_metadata"
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_recovery_removes_partial_non_authoritative_install_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_install = dataset_sidecar._install_transaction_payload

    class SimulatedPartialWriteCrash(BaseException):
        pass

    def leave_partial_temporary(payload_path: Path, target: Path, **kwargs: Any) -> None:
        transaction_id = str(kwargs["transaction_id"])
        index = int(kwargs["index"])
        temporary = target.with_name(f".{target.name}.{transaction_id}.{index:04d}.installing")
        dataset_sidecar._write_durable_file(temporary, payload_path.read_bytes()[:7])
        raise SimulatedPartialWriteCrash

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", leave_partial_temporary)

    with pytest.raises(SimulatedPartialWriteCrash):
        apply_metadata_file(project, source_root, packs, metadata_path)

    store = project / "datasets" / "source_metadata"
    assert list(store.glob("*.installing"))
    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", original_install)
    original_replace = dataset_sidecar._durable_replace
    recovery_moves: list[tuple[Path, Path]] = []

    def track_recovery_move(source: Path, target: Path) -> None:
        recovery_moves.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(dataset_sidecar, "_durable_replace", track_recovery_move)

    assert load_pack_metadata(project) == {}
    assert any(
        source.name.endswith(".installing") and target.name.startswith("discarded_installing_")
        for source, target in recovery_moves
    )
    assert not list(store.glob("*.installing"))
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_recovery_durably_moves_uncommitted_new_target_into_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_install = dataset_sidecar._install_transaction_payload

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_install(payload_path: Path, target: Path, **kwargs: Any) -> None:
        original_install(payload_path, target, **kwargs)
        raise SimulatedProcessCrash

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", crash_after_install)
    with pytest.raises(SimulatedProcessCrash):
        apply_metadata_file(project, source_root, packs, metadata_path)

    store = project / "datasets" / "source_metadata"
    assert list(store.glob("pack_*.json"))
    assert list((store / ".transactions").glob("txn_*"))
    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", original_install)
    original_replace = dataset_sidecar._durable_replace
    recovery_moves: list[tuple[Path, Path]] = []

    def track_recovery_move(source: Path, target: Path) -> None:
        recovery_moves.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(dataset_sidecar, "_durable_replace", track_recovery_move)

    assert load_pack_metadata(project) == {}
    assert any(
        source.name.startswith("pack_") and target.name.startswith("discarded_target_")
        for source, target in recovery_moves
    )
    assert not list(store.glob("pack_*.json"))
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_recovery_retries_store_durability_before_journal_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata(source_type="other_downloaded"))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_install = dataset_sidecar._install_transaction_payload

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_install(payload_path: Path, target: Path, **kwargs: Any) -> None:
        original_install(payload_path, target, **kwargs)
        raise SimulatedProcessCrash

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", crash_after_install)
    with pytest.raises(SimulatedProcessCrash):
        apply_metadata_file(project, source_root, packs, metadata_path)

    monkeypatch.setattr(dataset_sidecar, "_install_transaction_payload", original_install)
    store = project / "datasets" / "source_metadata"
    original_fsync_directory = dataset_sidecar._fsync_directory
    failed_store_barrier = False

    def fail_first_store_barrier(path: Path) -> None:
        nonlocal failed_store_barrier
        if path == store and not failed_store_barrier:
            failed_store_barrier = True
            raise OSError("synthetic source-directory durability failure")
        original_fsync_directory(path)

    monkeypatch.setattr(dataset_sidecar, "_fsync_directory", fail_first_store_barrier)
    with pytest.raises(OSError, match=r"durability|failure"):
        load_pack_metadata(project)

    assert failed_store_barrier is True
    assert not list(store.glob("pack_*.json"))
    assert list((store / ".transactions").glob("txn_*"))
    monkeypatch.setattr(dataset_sidecar, "_fsync_directory", original_fsync_directory)
    original_required_barrier = dataset_sidecar._require_directory_durable
    required_barriers: list[Path] = []

    def track_required_barrier(path: Path) -> None:
        required_barriers.append(path)
        original_required_barrier(path)

    monkeypatch.setattr(dataset_sidecar, "_require_directory_durable", track_required_barrier)

    assert load_pack_metadata(project) == {}
    assert store in required_barriers
    assert not list((store / ".transactions").glob("txn_*"))


def test_metadata_transaction_rejects_lexical_target_symlink_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "project" / "datasets" / "source_metadata"
    target = store / "pack_0123456789abcdef01234567.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"old": true}\n', encoding="utf-8")
    before = target.read_bytes()
    original_is_symlink = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)

    with pytest.raises(OSError, match=r"linked|reparse"):
        dataset_sidecar._write_json_transaction(((target, {"new": True}),))

    assert target.read_bytes() == before


def test_metadata_transaction_reverifies_targets_after_source_callback(tmp_path: Path) -> None:
    store = tmp_path / "project" / "datasets" / "source_metadata"
    store.mkdir(parents=True)
    target = store / "pack_0123456789abcdef01234567.json"
    target.write_text('{"generation": "old"}\n', encoding="utf-8")
    before = target.read_bytes()

    def mutate_installed_target() -> None:
        target.write_bytes(b"intruder")

    with pytest.raises(OSError, match=r"changed|validation|commit"):
        dataset_sidecar._write_json_transaction(
            ((target, {"generation": "new"}),),
            validate_before_commit=mutate_installed_target,
        )

    assert target.read_bytes() == b"intruder"
    assert target.read_bytes() != before
    assert list((store / ".transactions").glob("txn_*"))
    with pytest.raises(OSError, match=r"conflict|recovery"):
        load_pack_metadata(tmp_path / "project")


def test_metadata_recovery_rejects_linked_transactions_directory_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    store = project / "datasets" / "source_metadata"
    transactions = store / ".transactions"
    transactions.mkdir(parents=True)
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_text("must remain unchanged", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path == transactions or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)

    with pytest.raises(OSError, match=r"linked|reparse"):
        load_pack_metadata(project)

    assert outside.read_text(encoding="utf-8") == "must remain unchanged"
    assert transactions.is_dir()


def test_metadata_lock_rejects_hard_link_without_mutating_outside_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = project / "datasets" / "source_metadata"
    store.mkdir(parents=True)
    outside = tmp_path / "outside-lock-target.bin"
    outside.write_bytes(b"")
    try:
        os.link(outside, store / ".metadata.lock")
    except OSError:
        pytest.skip("Hard links are unavailable in this test session.")

    with pytest.raises(OSError, match=r"hard link|multiple"):
        load_pack_metadata(project)

    assert outside.read_bytes() == b""


def test_durable_metadata_writer_never_truncates_a_preplanted_hard_link(tmp_path: Path) -> None:
    outside = tmp_path / "outside-payload.bin"
    outside.write_bytes(b"outside must remain unchanged")
    target = tmp_path / "preplanted.installing"
    try:
        os.link(outside, target)
    except OSError:
        pytest.skip("Hard links are unavailable in this test session.")

    with pytest.raises(FileExistsError):
        dataset_sidecar._write_durable_file(target, b"replacement")

    assert outside.read_bytes() == b"outside must remain unchanged"


def test_pack_metadata_reader_rejects_alias_filename_and_corrupt_record(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    source_root, _paths, packs = discover_source_packs(root, context=_context(project))
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        _metadata(source_type="other_downloaded"),
        covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
    )
    store = project / "datasets" / "source_metadata"
    canonical = next(store.glob("pack_*.json"))
    alias = store / "pack_alias.json"
    canonical.replace(alias)

    with pytest.raises(PackMetadataError, match=r"does not match|identity"):
        load_pack_metadata(project)

    alias.replace(canonical)
    canonical.write_text("{not-json", encoding="utf-8")
    with pytest.raises(PackMetadataError, match=r"unreadable|corrupt"):
        load_pack_metadata(project)


@pytest.mark.parametrize("marker", ["PREPARED", "COMMITTED"])
def test_metadata_batch_recovers_when_process_crashes_before_atomic_marker_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    project = tmp_path / "project"
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        {**_metadata(source_type="other_downloaded"), "pack_title": "Old generation"},
        covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
    )
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update({**_metadata(source_type="other_downloaded"), "pack_title": "New generation"})
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    original_replace = dataset_sidecar._durable_replace

    class SimulatedMarkerCrash(BaseException):
        pass

    def crash_before_marker_install(source: Path, target: Path) -> None:
        if target.name == marker:
            raise SimulatedMarkerCrash
        original_replace(source, target)

    monkeypatch.setattr(dataset_sidecar, "_durable_replace", crash_before_marker_install)

    with pytest.raises(SimulatedMarkerCrash):
        apply_metadata_file(project, source_root, packs, metadata_path)

    monkeypatch.setattr(dataset_sidecar, "_durable_replace", original_replace)
    recovered = load_pack_metadata(project)

    assert recovered[packs[0].pack_id]["pack_title"] == "Old generation"
    store = project / "datasets" / "source_metadata"
    assert not list((store / ".transactions").glob("txn_*"))


def test_changed_file_invalidates_only_affected_pack(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "mixed"
    context = _context(project)
    for name, color in (("a", (240, 80, 60, 255)), ("b", (60, 160, 230, 255))):
        make_png(root / name / "sprite.png", color=color)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    for pack in packs:
        save_pack_metadata(
            project,
            source_root,
            pack,
            {**_metadata(), "pack_title": pack.relative_root},
            covered_byte_hashes=[tree_hashes(root)[pack.image_relative_paths[0]]],
        )
    assert build_dataset(root, output_root=tmp_path / "out", context=context).data["counts"]["accepted"] == 2
    make_png(root / "a" / "sprite.png", color=(10, 220, 120, 255))
    result = build_dataset(root, output_root=tmp_path / "out", context=context)
    assert result.data["counts"]["accepted"] == 1
    assert result.data["counts"]["quarantined"] == 1
    assert {
        item["pack_relative_root"] for item in _items(tmp_path / "out") if item["current_disposition"] == "accepted"
    } == {"b"}


def test_changed_source_evidence_invalidates_only_its_pack_sidecar(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "mixed"
    context = _context(project)
    for name, color in (("a", (240, 80, 60, 255)), ("b", (60, 160, 230, 255))):
        pack_root = root / name
        make_png(pack_root / "sprite.png", color=color)
        (pack_root / "source.txt").write_text(f"https://example.test/{name}\n", encoding="utf-8")
        (pack_root / "LICENSE").write_text("CC0\n", encoding="utf-8")
    source_root, _paths, packs = discover_source_packs(root, context=context)
    for pack in packs:
        save_pack_metadata(
            project,
            source_root,
            pack,
            {**_metadata(source_type="other_downloaded"), "pack_title": pack.relative_root},
            covered_byte_hashes=[tree_hashes(root)[pack.image_relative_paths[0]]],
        )
    assert build_dataset(root, output_root=tmp_path / "out", context=context).data["counts"]["accepted"] == 2
    (root / "a" / "source.txt").write_text("https://example.test/a-v2\n", encoding="utf-8")
    result = build_dataset(root, output_root=tmp_path / "out", context=context)
    assert result.data["counts"]["accepted"] == 1
    assert result.data["counts"]["missing_creator"] == 1
    inspection = inspect_dataset_folder(root, context=context)
    states = {pack["relative_root"]: pack["sidecar_stale"] for pack in inspection["packs"]}
    assert states == {"a": True, "b": False}


def test_changed_inherited_license_evidence_invalidates_covered_pack(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "shared-license"
    pack_root = root / "pack"
    make_png(pack_root / "sprite.png")
    (pack_root / "source.txt").write_text("https://example.test/pack\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC0\n", encoding="utf-8")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    assert {row["relative_path"] for row in packs[0].evidence_files} == {
        "LICENSE",
        "pack/source.txt",
    }
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        {
            **_metadata(source_type="other_downloaded"),
            "license_url": None,
            "license_evidence_file": "LICENSE",
        },
        covered_byte_hashes=[tree_hashes(root)["pack/sprite.png"]],
    )
    assert inspect_dataset_folder(root, context=context)["packs"][0]["sidecar_applied"] is True
    (root / "LICENSE").write_text("Unknown\n", encoding="utf-8")
    inspection = inspect_dataset_folder(root, context=context)
    assert inspection["packs"][0]["sidecar_stale"] is True
    result = build_dataset(root, output_root=tmp_path / "out", context=context)
    assert result.data["counts"]["accepted"] == 0
    assert result.data["counts"]["quarantined"] == 1


def test_build_blocks_if_evidence_changes_after_initial_pack_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    output = tmp_path / "out"
    original_effective_evidence = dataset_intake._effective_evidence
    changed = False

    def change_evidence_after_read(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal changed
        evidence = original_effective_evidence(*args, **kwargs)
        if not changed:
            (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
            changed = True
        return evidence

    monkeypatch.setattr(dataset_intake, "_effective_evidence", change_evidence_after_read)

    result = build_dataset(root, output_root=output)

    assert result.status.value == "BLOCKED"
    assert "changed during import" in result.message
    assert not (output / "raw_extraction").exists()
    assert not (output / "items.jsonl").exists()
    assert not (output / "review_queue.json").exists()


def test_build_blocks_if_evidence_changes_during_raw_extraction_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    output = tmp_path / "out"
    original_build_raw_extraction = dataset_intake.build_raw_extraction

    def mutate_after_extraction(*args: Any, **kwargs: Any) -> Any:
        result = original_build_raw_extraction(*args, **kwargs)
        (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
        return result

    monkeypatch.setattr(dataset_intake, "build_raw_extraction", mutate_after_extraction)

    result = build_dataset(root, output_root=output)

    assert result.status.value == "BLOCKED"
    assert "changed during import" in result.message
    assert not (output / "raw_extraction").exists()
    assert not (output / ".raw_extraction.next").exists()
    assert not (output / "items.jsonl").exists()


def test_build_rolls_back_raw_extraction_if_evidence_changes_during_directory_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_configured(tmp_path / "source")
    make_png(root / "sprite.png")
    output = tmp_path / "out"
    original_replace = Path.replace

    def mutate_after_raw_publish(source: Path, target: Path) -> Path:
        result = original_replace(source, target)
        if source.name == ".raw_extraction.next" and Path(target).name == "raw_extraction":
            (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "replace", mutate_after_raw_publish)

    result = build_dataset(root, output_root=output)

    assert result.status.value == "BLOCKED"
    assert "changed during import" in result.message
    assert not (output / "raw_extraction").exists()
    assert not (output / ".raw_extraction.previous").exists()
    assert not (output / "items.jsonl").exists()


def test_conflicting_license_evidence_requires_explicit_pack_resolution(tmp_path: Path) -> None:
    root = tmp_path / "conflict"
    make_png(root / "sprite.png")
    (root / "source.txt").write_text(
        "Name: Conflict pack\nCreator: Synthetic Artist\nhttps://example.test/conflict\n", encoding="utf-8"
    )
    (root / "license.yaml").write_text("license: MIT\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC-BY-SA-4.0\n", encoding="utf-8")

    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))

    assert inspection["wizard_required"] is True
    assert "conflicting_license_evidence" in inspection["packs"][0]["missing_fields"]
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["accepted"] == 0
    assert result.data["counts"]["quarantined"] == 1
    assert result.data["counts"]["missing_information"] == 1
    assert result.data["packs"]["packs"][0]["information_complete"] is False
    assert result.data["packs"]["packs"][0]["images_missing_information"] == 1
    assert "conflicting_license_evidence" in _items(tmp_path / "out")[0]["reasons"]


def test_unrecognized_dedicated_license_conflicts_with_mit(tmp_path: Path) -> None:
    root = tmp_path / "conflict"
    make_png(root / "sprite.png")
    (root / "source.txt").write_text(
        "Name: Conflict pack\nCreator: Synthetic Artist\nhttps://example.test/conflict\n", encoding="utf-8"
    )
    (root / "license.yaml").write_text("license: MIT\n", encoding="utf-8")
    (root / "LICENSE").write_text(
        "Use is permitted only with prior written authorization from Example Corp.\n", encoding="utf-8"
    )

    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    licenses = {record["license"] for record in inspection["packs"][0]["license"]["evidence_records"]}
    result = build_dataset(root, output_root=tmp_path / "out")

    assert licenses == {"mit", "unknown"}
    assert "conflicting_license_evidence" in inspection["packs"][0]["missing_fields"]
    assert result.data["counts"]["accepted"] == 0
    assert result.data["counts"]["quarantined"] == 1


def test_equivalent_license_aliases_do_not_create_a_false_conflict(tmp_path: Path) -> None:
    root = tmp_path / "aliases"
    make_png(root / "sprite.png")
    (root / "source.txt").write_text(
        "Name: Alias pack\nCreator: Synthetic Artist\nhttps://example.test/aliases\n", encoding="utf-8"
    )
    (root / "license.yaml").write_text("license: cc_by_sa\n", encoding="utf-8")
    (root / "LICENSE").write_text("https://creativecommons.org/licenses/by-sa/4.0/\n", encoding="utf-8")

    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))

    assert "conflicting_license_evidence" not in inspection["packs"][0]["missing_fields"]
    assert build_dataset(root, output_root=tmp_path / "out").data["counts"]["accepted"] == 1


def test_conflicting_source_evidence_quarantines_only_the_affected_pack(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    conflicted = root / "a-conflicted"
    valid = root / "b-valid"
    make_png(conflicted / "sprite.png")
    make_png(valid / "sprite.png", color=(40, 160, 230, 255))
    (conflicted / "source.yaml").write_text(
        "name: Conflict pack\ncreator: Artist A\nurl: https://example.test/a\n", encoding="utf-8"
    )
    (conflicted / "source.txt").write_text(
        "Name: Conflict pack\nCreator: Artist B\nhttps://example.test/b\n", encoding="utf-8"
    )
    (conflicted / "LICENSE").write_text("CC0\n", encoding="utf-8")
    (valid / "source.txt").write_text("Name: Valid pack\nCreator: Artist C\nhttps://example.test/c\n", encoding="utf-8")
    (valid / "LICENSE").write_text("CC0\n", encoding="utf-8")

    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    by_root = {pack["relative_root"]: pack for pack in inspection["packs"]}
    assert "conflicting_source_evidence" in by_root["a-conflicted"]["missing_fields"]
    assert by_root["b-valid"]["missing_fields"] == []
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["accepted"] == 1
    assert result.data["counts"]["quarantined"] == 1
    assert result.data["counts"]["missing_information"] == 1
    summaries = {pack["relative_root"]: pack for pack in result.data["packs"]["packs"]}
    assert summaries["a-conflicted"]["information_complete"] is False
    assert summaries["a-conflicted"]["images_missing_information"] == 1
    assert summaries["b-valid"]["information_complete"] is True


@pytest.mark.parametrize(
    ("first_url", "second_url", "conflict"),
    [
        (
            "HTTPS://Example.Test/Assets/Hero.PNG?Token=AbC",
            "https://example.test/Assets/Hero.PNG?Token=AbC",
            False,
        ),
        ("https://example.test/Assets/Hero.PNG", "https://example.test/assets/Hero.PNG", True),
        ("https://example.test/asset?Token=AbC", "https://example.test/asset?Token=abc", True),
    ],
)
def test_source_url_conflicts_casefold_only_scheme_and_host(
    tmp_path: Path, first_url: str, second_url: str, conflict: bool
) -> None:
    root = tmp_path / "source"
    make_png(root / "sprite.png")
    (root / "source.yaml").write_text(
        yaml.safe_dump({"name": "Pack", "creator": "Artist", "url": first_url}), encoding="utf-8"
    )
    (root / "source.txt").write_text(f"Name: Pack\nCreator: Artist\n{second_url}\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC0\n", encoding="utf-8")

    inspection = inspect_dataset_folder(root, context=_context(tmp_path / "project"))
    reasons = set(inspection["packs"][0]["missing_fields"])

    assert ("conflicting_source_evidence" in reasons) is conflict


def test_bound_sidecar_resolves_conflict_until_evidence_changes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "conflict"
    make_png(root / "sprite.png")
    (root / "source.txt").write_text(
        "Name: Conflict pack\nCreator: Synthetic Artist\nhttps://example.test/conflict\n", encoding="utf-8"
    )
    (root / "license.yaml").write_text("license: MIT\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC-BY-SA-4.0\n", encoding="utf-8")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        _metadata(source_type="other_downloaded", license_identifier="cc0"),
        covered_byte_hashes=[tree_hashes(root)["sprite.png"]],
    )
    assert build_dataset(root, output_root=tmp_path / "before", context=context).data["counts"]["accepted"] == 1
    (root / "LICENSE").write_text("Public domain\n", encoding="utf-8")

    inspection = inspect_dataset_folder(root, context=context)

    assert inspection["packs"][0]["sidecar_stale"] is True
    assert "conflicting_license_evidence" in inspection["packs"][0]["missing_fields"]
    result = build_dataset(root, output_root=tmp_path / "after", context=context)
    assert result.data["counts"]["accepted"] == 0
    assert result.data["counts"]["quarantined"] == 1


def test_valid_duplicate_pack_remains_eligible_when_earlier_pack_is_unknown(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    first = make_png(root / "a_unknown" / "sprite.png")
    second = root / "b_valid" / "sprite.png"
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())
    (root / "a_unknown" / "source.txt").write_text(
        "Name: Unknown pack\nCreator: A\nhttps://example.test/a\n", encoding="utf-8"
    )
    (root / "a_unknown" / "LICENSE").write_text("Unknown\n", encoding="utf-8")
    (root / "b_valid" / "source.txt").write_text(
        "Name: Valid pack\nCreator: B\nhttps://example.test/b\n", encoding="utf-8"
    )
    (root / "b_valid" / "LICENSE").write_text("CC0\n", encoding="utf-8")
    result = build_dataset(root, output_root=tmp_path / "out")
    items = _items(tmp_path / "out")
    valid = next(item for item in items if item["pack_relative_root"] == "b_valid")
    unknown = next(item for item in items if item["pack_relative_root"] == "a_unknown")
    assert result.data["counts"]["accepted"] == 1
    assert valid["current_disposition"] == "accepted"
    assert unknown["duplicate_of"] == valid["item_id"]
    assert valid["duplicate_associations"][0]["pack_id"] == unknown["pack_id"]


def test_unknown_license_quarantines_and_cannot_be_overridden(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "download"
    make_png(root / "sprite.png")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    save_pack_metadata(
        project,
        source_root,
        packs[0],
        _metadata(source_type="other_downloaded", license_identifier="unknown"),
        covered_byte_hashes=list(tree_hashes(root).values()),
    )
    output = tmp_path / "out"
    result = build_dataset(root, output_root=output, context=context)
    assert result.data["counts"]["quarantined"] == 1
    item = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))["items"][0]
    assert "unverified_license" in item["reasons"]
    with pytest.raises(ReviewDecisionError, match="source/license evidence"):
        DatasetReviewStore(output).apply(item["item_id"], "keep")


def test_missing_creator_quarantines_only_the_incomplete_pack(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    incomplete = root / "a_incomplete"
    valid = root / "b_valid"
    make_png(incomplete / "sprite.png")
    make_png(valid / "sprite.png", color=(40, 170, 230, 255))
    (incomplete / "source.txt").write_text("Name: Incomplete pack\nhttps://example.test/incomplete\n", encoding="utf-8")
    (incomplete / "LICENSE").write_text("CC0\n", encoding="utf-8")
    (valid / "source.txt").write_text(
        "Name: Valid pack\nCreator: Valid Artist\nhttps://example.test/valid\n", encoding="utf-8"
    )
    (valid / "LICENSE").write_text("CC0\n", encoding="utf-8")
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["accepted"] == 1
    assert result.data["counts"]["missing_creator"] == 1
    assert "1 images are waiting for source or license information" in result.message


def test_metadata_file_automation_and_input_immutability(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = tmp_path / "input"
    make_png(root / "sprite.png")
    context = _context(project)
    source_root, _paths, packs = discover_source_packs(root, context=context)
    metadata_path = tmp_path / "metadata.json"
    payload = metadata_file_template(source_root, packs)
    payload["packs"][0].update(_metadata())
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    before = tree_hashes(root)
    applied = apply_metadata_file(project, source_root, packs, metadata_path)
    assert applied["applied_pack_ids"] == [packs[0].pack_id]
    assert build_dataset(root, output_root=tmp_path / "out", context=context).data["counts"]["accepted"] == 1
    assert tree_hashes(root) == before


def test_noninteractive_cli_quarantines_writes_json_template_and_durable_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spritelab.yaml").write_text(
        "project:\n  name: intake-test\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    root = tmp_path / "assets Ω"
    make_png(root / "sprite.png")
    monkeypatch.setattr(dataset_cli, "launch_metadata_wizard", lambda *_args: pytest.fail("browser forbidden"))
    monkeypatch.setattr(dataset_cli, "launch_review_interface", lambda *_args: pytest.fail("browser forbidden"))
    with pytest.raises(SystemExit) as caught:
        v3_cli.main(["dataset", "build", str(root), "--output", str(tmp_path / "out")], plugins=(build_plugin(),))
    payload = json.loads(capsys.readouterr().out)
    assert int(caught.value.code) == 4
    product = payload["data"]["product_result"]
    assert product["data"]["browser_opened"] is False
    template = Path(product["data"]["metadata_template"])
    assert template.is_file()
    assert "--metadata-file" in product["data"]["next_command"]
    run = next(path.parent for path in tmp_path.rglob("command.json"))
    assert all((run / name).is_file() for name in ("state.json", "events.jsonl", "command.json"))


def test_noninteractive_metadata_template_includes_every_pack_required_by_batch_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spritelab.yaml").write_text(
        "project:\n  name: complete-batch-template\n  schema_version: 3\npaths:\n  runs: runs/v3\n",
        encoding="utf-8",
    )
    root = tmp_path / "assets"
    ready = make_configured(root / "ready")
    make_png(ready / "sprite.png")
    make_png(root / "missing" / "sprite.png", color=(40, 160, 230, 255))
    monkeypatch.setattr(dataset_cli, "launch_metadata_wizard", lambda *_args: pytest.fail("browser forbidden"))
    monkeypatch.setattr(dataset_cli, "launch_review_interface", lambda *_args: pytest.fail("browser forbidden"))

    with pytest.raises(SystemExit) as caught:
        v3_cli.main(
            ["dataset", "build", str(root), "--output", str(tmp_path / "out")],
            plugins=(build_plugin(),),
        )

    product = json.loads(capsys.readouterr().out)["data"]["product_result"]
    template = json.loads(Path(product["data"]["metadata_template"]).read_text(encoding="utf-8"))
    _source_root, _paths, current_packs = discover_source_packs(root, context=_context(tmp_path))
    assert int(caught.value.code) in {0, 4}
    assert {row["pack_id"] for row in template["packs"]} == {pack.pack_id for pack in current_packs}


def test_web_wizard_prefill_save_and_build(tmp_path: Path) -> None:
    project = tmp_path / "project"
    selected = tmp_path / "selected"
    make_png(selected / "sprite.png")
    before = tree_hashes(selected)
    context = _context(project)
    client = TestClient(create_app(context, plugins=(create_plugin(folder_chooser=lambda: selected),)))
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={}).json()
    approval_id = chosen["approval"]["approval_id"]
    inspection = client.post("/dataset/api/inspect", headers=_csrf(client), json={"approval_id": approval_id}).json()
    assert inspection["wizard_required"] is True
    page = client.get(f"/dataset/metadata?approval_id={approval_id}")
    assert "My original work" in page.text and "cannot verify ownership" in page.text
    pack_id = inspection["packs"][0]["pack_id"]
    saved = client.post(
        "/dataset/api/metadata/save",
        headers=_csrf(client),
        json={"approval_id": approval_id, "pack_id": pack_id, "metadata": _metadata()},
    )
    assert saved.status_code == 200
    assert saved.json()["input_folder_written"] is False
    assert tree_hashes(selected) == before
    built = client.post(
        "/dataset/api/build",
        headers=_csrf(client),
        json={"approval_id": approval_id, "confirm_hosted": False},
    )
    result = _await_dataset_build(client, built)
    assert result["data"]["counts"]["accepted"] == 1


def test_dataset_browser_apis_never_disclose_canonical_local_paths(tmp_path: Path) -> None:
    project = tmp_path / "private-project"
    selected = tmp_path / "private-source" / "selected"
    make_png(selected / "sprite.png")
    context = _context(project)
    client = TestClient(create_app(context, plugins=(create_plugin(folder_chooser=lambda: selected),)))
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={})
    approval_id = chosen.json()["approval"]["approval_id"]
    selected_path = str(selected.resolve())
    project_path = str(project.resolve())

    inspection = client.post("/dataset/api/inspect", headers=_csrf(client), json={"approval_id": approval_id})
    metadata_inspection = client.post(
        "/dataset/api/metadata/inspect", headers=_csrf(client), json={"approval_id": approval_id}
    )
    page = client.get(f"/dataset/metadata?approval_id={approval_id}")
    pack_id = inspection.json()["packs"][0]["pack_id"]
    saved = client.post(
        "/dataset/api/metadata/save",
        headers=_csrf(client),
        json={"approval_id": approval_id, "pack_id": pack_id, "metadata": _metadata()},
    )
    built = client.post(
        "/dataset/api/build",
        headers=_csrf(client),
        json={"approval_id": approval_id, "confirm_hosted": False},
    )
    result = _await_dataset_build(client, built)
    review_page = client.get("/dataset/review")
    review_data = client.get("/dataset/api/review/data")

    for response in (chosen, inspection, metadata_inspection, page, saved, built, review_page, review_data):
        assert selected_path not in response.text
        assert project_path not in response.text
    serialized_result = json.dumps(result)
    assert selected_path not in serialized_result
    assert project_path not in serialized_result
    assert "input_root" not in inspection.json()
    assert "input_root" not in metadata_inspection.json()
    assert "input_root" not in saved.json()["inspection"]
    assert "input_root" not in result["data"]
    assert "output_root" not in result["data"]
    assert result["data"]["approval_id"] == approval_id
    assert result["data"]["dataset_id"].startswith("dataset-")
    assert "input_root" not in review_data.json()
    assert "output_root" not in review_data.json()
    assert "append_only_log" not in review_data.json()
    assert all("source_path" not in item for item in review_data.json()["items"])


def test_dataset_review_redacts_nested_decode_errors_and_source_paths(tmp_path: Path) -> None:
    project = tmp_path / "private-project"
    selected = make_configured(tmp_path / "private-source" / "selected")
    (selected / "broken.png").write_bytes(b"not a PNG")
    client = TestClient(create_app(_context(project), plugins=(create_plugin(folder_chooser=lambda: selected),)))
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={}).json()
    approval_id = chosen["approval"]["approval_id"]

    built = client.post(
        "/dataset/api/build",
        headers=_csrf(client),
        json={"approval_id": approval_id, "confirm_hosted": False},
    )
    result = _await_dataset_build(client, built)
    review_page = client.get("/dataset/review")
    review_data = client.get("/dataset/api/review/data")

    assert review_data.status_code == 200
    assert review_data.json()["items"]
    assert all("source_path" not in item for item in review_data.json()["items"])
    assert any(item.get("decode_error") for item in review_data.json()["items"])
    for response in (built, review_page, review_data):
        serialized = response.text.replace("\\\\", "\\")
        assert str(selected.resolve()) not in serialized
        assert selected.resolve().as_posix() not in serialized
        assert str(project.resolve()) not in serialized
    serialized_result = json.dumps(result).replace("\\\\", "\\")
    assert str(selected.resolve()) not in serialized_result
    assert selected.resolve().as_posix() not in serialized_result
    assert str(project.resolve()) not in serialized_result


def test_dataset_metadata_error_responses_redact_known_and_unexpected_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "private-project"
    selected = tmp_path / "private-source" / "selected"
    make_png(selected / "sprite.png")
    client = TestClient(create_app(_context(project), plugins=(create_plugin(folder_chooser=lambda: selected),)))
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={}).json()
    approval_id = chosen["approval"]["approval_id"]
    inspected = client.post("/dataset/api/inspect", headers=_csrf(client), json={"approval_id": approval_id})
    pack_id = inspected.json()["packs"][0]["pack_id"]
    unexpected = Path("C:/unexpected-private-location/secret.txt")
    unexpected_posix = "/workspace/unexpected-private-location/secret.txt"

    def fail_save(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError(
            f"could not write {selected.resolve()} or {project.resolve()} or {unexpected} or {unexpected_posix}"
        )

    monkeypatch.setattr(dataset_web, "save_pack_metadata", fail_save)
    saved = client.post(
        "/dataset/api/metadata/save",
        headers=_csrf(client),
        json={"approval_id": approval_id, "pack_id": pack_id, "metadata": _metadata()},
    )

    def fail_inspection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise DatasetInputError(f"could not inspect {selected.resolve()} or {unexpected}")

    monkeypatch.setattr(dataset_web, "inspect_dataset_folder", fail_inspection)
    page = client.get(f"/dataset/metadata?approval_id={approval_id}")

    assert saved.status_code == 422
    assert page.status_code == 409
    for response in (saved, page):
        serialized = response.text.replace("\\\\", "\\")
        assert str(selected.resolve()) not in serialized
        assert selected.resolve().as_posix() not in serialized
        assert str(project.resolve()) not in serialized
        assert project.resolve().as_posix() not in serialized
        assert str(unexpected) not in serialized
        assert unexpected.as_posix() not in serialized
        assert unexpected_posix not in serialized


def test_memorization_review_api_exposes_urls_instead_of_local_artifact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "private-project"
    generated = tmp_path / "private-evaluation" / "generated.png"
    training = tmp_path / "private-dataset" / "training.png"
    candidate = tmp_path / "private-evaluation" / "candidate.json"
    review = {
        "schema_version": "spritelab.product.memorization-display.v2",
        "review_message": "Review required",
        "items": [
            {
                "pair_id": "pair_123",
                "display_state": "Review required",
                "generated_image": str(generated.resolve()),
                "training_comparison_image": str(training.resolve()),
                "candidate_bundle_path": str(candidate.resolve()),
            }
        ],
    }
    monkeypatch.setattr(dataset_web, "discover_memorization_review", lambda _context: review)
    client = TestClient(create_app(_context(project), plugins=(create_plugin(),)))

    response = client.get("/review/memorization/data")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["generated_image_url"] == "/review/memorization/pair_123/image/generated"
    assert item["training_comparison_image_url"] == "/review/memorization/pair_123/image/training"
    assert "generated_image" not in item
    assert "training_comparison_image" not in item
    assert "candidate_bundle_path" not in item
    for path in (project, generated, training, candidate):
        assert str(path.resolve()) not in response.text.replace("\\\\", "\\")


def test_web_wizard_requires_explicit_license_selection_for_platform_prefills(tmp_path: Path) -> None:
    project = tmp_path / "project"
    selected = tmp_path / "selected"
    make_png(selected / "sprite.png")
    (selected / "source.txt").write_text(
        "Name: OGA-style pack\nCreator: Synthetic Artist\nhttps://opengameart.org/content/synthetic\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(_context(project), plugins=(create_plugin(folder_chooser=lambda: selected),)))
    chosen = client.post("/dataset/api/folders/choose", headers=_csrf(client), json={}).json()
    page = client.get(f"/dataset/metadata?approval_id={chosen['approval']['approval_id']}")
    license_select = page.text.split('<select name="license_identifier" required>', 1)[1].split("</select>", 1)[0]

    assert '<option value=""' in license_select
    assert license_select.index('value=""') < license_select.index('value="cc0"')


def test_duplicate_categories_and_near_duplicates_are_reported_separately(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    first = make_png(root / "a.png")
    (root / "b.png").write_bytes(first.read_bytes())
    save_same_rgba_with_metadata(first, root / "c.png")
    make_png(root / "near.png", color=(238, 82, 62, 255))
    result = build_dataset(root, output_root=tmp_path / "out")
    counts = result.data["counts"]
    assert counts["byte_duplicates"] == 1
    assert counts["decoded_pixel_duplicates"] == 1
    assert counts["exact_duplicates_removed"] == 2
    assert counts["possible_near_duplicates"] == 2
    items = _items(tmp_path / "out")
    assert next(item for item in items if item["relative_path"] == "near.png")["current_disposition"] == "accepted"
    canonical = next(item for item in items if item["relative_path"] == "a.png")
    assert len(canonical["duplicate_associations"]) == 2


def test_ambiguous_sheet_has_prefilled_review_and_keep_proposal(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    _ambiguous_sheet(root / "sheet.png")
    output = tmp_path / "out"
    result = build_dataset(root, output_root=output)
    assert result.data["counts"]["needs_sheet_review"] == 1
    item = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))["items"][0]
    assert item["current_disposition"] == "requires_special_extraction"
    assert item["sheet_plan"]["unambiguous"] is False
    context = _context(tmp_path / "project", output=output)
    client = TestClient(create_app(context, plugins=(build_plugin(),)))
    page = client.get("/dataset/review")
    assert "Keep proposal" in page.text and "Adjust grid" in page.text and "Exclude sheet" in page.text
    preview = client.get(f"/dataset/review/sheets/{item['item_id']}/preview/proposal")
    assert preview.status_code == 200 and preview.headers["content-type"].startswith("image/png")
    decision = client.post(
        f"/dataset/api/review/sheets/{item['item_id']}/decision",
        headers=_csrf(client),
        json={"action": "keep_proposal"},
    )
    assert decision.status_code == 200
    assert decision.json()["counts"]["extracted_from_sheets"] == 4
    assert decision.json()["counts"]["needs_sheet_review"] == 0


def test_sheet_decision_is_invalidated_when_source_identity_changes(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    sheet = _ambiguous_sheet(root / "sheet.png")
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    item = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))["items"][0]
    client = TestClient(create_app(_context(tmp_path / "project", output=output), plugins=(build_plugin(),)))
    decision = client.post(
        f"/dataset/api/review/sheets/{item['item_id']}/decision",
        headers=_csrf(client),
        json={"action": "keep_proposal"},
    )
    assert decision.json()["counts"]["extracted_from_sheets"] == 4
    with Image.open(sheet) as opened:
        changed = opened.convert("RGBA")
    changed.putpixel((2, 4), (211, 77, 39, 255))
    changed.save(sheet)

    rebuilt = build_dataset(root, output_root=output)

    assert rebuilt.data["counts"]["extracted_from_sheets"] == 0
    assert rebuilt.data["counts"]["needs_sheet_review"] == 1
    current = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))["items"][0]
    assert "sheet_decision" not in current


def test_provider_is_not_contacted_when_every_image_is_quarantined(tmp_path: Path) -> None:
    class HostedProvider:
        provider_id = "hosted.forbidden"
        title = "Hosted"

        def probe(self, _context: ProjectContext) -> Any:
            raise AssertionError("provider must not be contacted for intake-only quarantine")

        def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("provider must not receive images")

    root = tmp_path / "input"
    make_png(root / "sprite.png")
    result = DatasetIntakeService(HostedProvider()).build(
        root,
        output_root=tmp_path / "out",
        context=_context(tmp_path / "project"),
    )
    assert result.data["counts"]["quarantined"] == 1
    assert result.data["semantic"]["provider_status"] == "configured_not_needed"
