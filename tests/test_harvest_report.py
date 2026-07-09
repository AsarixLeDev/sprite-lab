"""Tests for spritelab.harvest.report."""

from __future__ import annotations

import json
from dataclasses import replace

from _harvest_testdata import make_source, make_sprite_png
from spritelab.harvest.pipeline import HarvestImportOptions, harvest_source_to_imported_sprites
from spritelab.harvest.report import build_harvest_report, build_harvest_report_data, write_harvest_reports


def _harvest(tmp_path, license_name="cc_by"):
    root = tmp_path / "pngs"
    make_sprite_png(root / "mushroom.png")
    make_sprite_png(root / "vial.png")
    source = make_source("report_source", license_name=license_name, local_root_path=str(root), author="")
    return source, harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )


def test_report_contains_sources_and_licenses(tmp_path):
    source, harvested = _harvest(tmp_path)
    report = build_harvest_report([source], harvested)
    assert "## Sources and licenses" in report
    assert "report_source" in report
    assert "cc_by" in report


def test_report_includes_counts(tmp_path):
    source, harvested = _harvest(tmp_path)
    report = build_harvest_report([source], harvested)
    assert "- Imported: 2" in report
    assert "## Categories" in report


def test_report_includes_warnings(tmp_path):
    source, harvested = _harvest(tmp_path)  # cc_by with no author -> warnings
    report = build_harvest_report([source], harvested)
    assert "## Warnings" in report
    assert "attribution" in report


def test_json_report_serializable(tmp_path):
    source, harvested = _harvest(tmp_path)
    data = build_harvest_report_data([source], harvested)
    text = json.dumps(data)
    assert "report_source" in text

    md_path, json_path = write_harvest_reports(tmp_path / "run", [source], harvested)
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["imported"] == 2


def test_report_includes_prefill_quality_counts(tmp_path):
    source, harvested = _harvest(tmp_path)
    updated = [
        replace(
            harvested[0],
            auto_metadata={
                **harvested[0].auto_metadata,
                "prefill_quality": {
                    "bucket": "needs_review",
                    "flags": ["filename_qwen_conflict"],
                },
            },
        ),
        harvested[1],
    ]

    data = build_harvest_report_data([source], updated)
    report = build_harvest_report([source], updated)

    assert data["prefill_quality"]["needs_review"] == 1
    assert data["prefill_quality"]["filename_qwen_conflict"] == 1
    assert "## Prefill quality" in report
