from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spritelab.product_core import ProjectContext
from spritelab.product_features.dataset.intake import build_dataset
from spritelab.product_features.dataset.managed import ManagedDatasetError, validate_managed_dataset_output
from test_product_dataset_helpers import make_configured, make_png


def _managed_dataset(tmp_path: Path, *, images: int = 1) -> tuple[Path, Path, ProjectContext]:
    project = tmp_path / "project"
    source = make_configured(tmp_path / "source")
    for index in range(images):
        make_png(
            source / f"sprite-{index}.png",
            color=(40 + index * 30, 100 + index * 20, 180 - index * 20, 255),
        )
    output = project / "datasets" / "managed"
    context = ProjectContext(
        project,
        config={"dataset": {"output_root": str(output)}},
        runs_directory=project / "runs" / "v3",
    )
    build_dataset(source, output_root=output, context=context)
    return project, output, context


def test_managed_dataset_accepts_only_a_complete_identity_bound_project_output(tmp_path: Path) -> None:
    _project, output, context = _managed_dataset(tmp_path)

    assert (
        validate_managed_dataset_output(
            output,
            context=context,
            require_datasets_root=True,
        )
        == output.resolve()
    )


def test_managed_dataset_rejects_an_output_outside_the_project_without_mutation(tmp_path: Path) -> None:
    project, _output, context = _managed_dataset(tmp_path)
    outside_source = make_configured(tmp_path / "outside-source")
    make_png(outside_source / "outside.png")
    outside = tmp_path / "outside-dataset"
    build_dataset(outside_source, output_root=outside)
    before = {path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()}

    with pytest.raises(ManagedDatasetError, match="inside the project datasets directory"):
        validate_managed_dataset_output(outside, context=context, require_datasets_root=True)

    assert project.is_dir()
    assert {path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()} == before


def test_managed_dataset_rejects_a_linked_output_seam(tmp_path: Path) -> None:
    project, output, context = _managed_dataset(tmp_path)
    linked = project / "datasets" / "linked"
    try:
        os.symlink(output, linked, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable in this test session")

    with pytest.raises(ManagedDatasetError, match="inside the project datasets directory"):
        validate_managed_dataset_output(linked, context=context, require_datasets_root=True)


def test_managed_dataset_rejects_a_hard_linked_mutable_document_without_touching_its_peer(
    tmp_path: Path,
) -> None:
    _project, output, context = _managed_dataset(tmp_path)
    outside = tmp_path / "outside-review-log.jsonl"
    sentinel = b'{"outside":"must remain byte-identical"}\n'
    outside.write_bytes(sentinel)
    try:
        os.link(outside, output / "review_log.jsonl")
    except OSError:
        pytest.skip("hard links are unavailable in this test session")

    with pytest.raises(ManagedDatasetError, match="hard-link count"):
        validate_managed_dataset_output(output, context=context)

    assert outside.read_bytes() == sentinel


def test_managed_dataset_rejects_a_mismatched_output_identity(tmp_path: Path) -> None:
    _project, output, context = _managed_dataset(tmp_path)
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["data"]["output_root"] = str(output.parent / "different")
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ManagedDatasetError, match="output identity"):
        validate_managed_dataset_output(output, context=context)


def test_managed_dataset_rejects_case_colliding_relative_paths(tmp_path: Path) -> None:
    _project, output, context = _managed_dataset(tmp_path, images=2)
    items_path = output / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
    items[1]["relative_path"] = items[0]["relative_path"].upper()
    items_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")

    with pytest.raises(ManagedDatasetError, match="case or Unicode-normalization path collision"):
        validate_managed_dataset_output(output, context=context)
