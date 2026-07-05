"""Tests for spritelab.harvest.sources."""

from __future__ import annotations

from spritelab.harvest.sources import (
    SourceLicense,
    SourceRecord,
    is_license_allowed_for_training,
    license_requires_attribution,
    normalize_license_name,
    normalize_source_id,
    source_record_from_dict,
    source_record_to_dict,
    source_warnings,
)


def test_normalizes_license_names():
    assert normalize_license_name("CC0") == "cc0"
    assert normalize_license_name("CC-BY") == "cc_by"
    assert normalize_license_name("Public Domain") == "public_domain"
    assert normalize_license_name("Apache-2.0") == "apache_2"
    assert normalize_license_name("something weird") == "unknown"
    assert normalize_license_name("custom") == "custom_unreviewed"


def test_allowed_licenses_pass():
    for name in ("cc0", "public_domain", "own_work", "cc_by", "oga_by", "cc_by_sa", "wtfpl", "mit", "apache_2", "bsd"):
        assert is_license_allowed_for_training(name), name


def test_unsafe_licenses_fail_by_default():
    for name in ("unknown", "noncommercial", "no_derivatives", "all_rights_reserved", "custom_unreviewed"):
        assert not is_license_allowed_for_training(name), name


def test_source_record_roundtrip():
    record = SourceRecord(
        source_id="My Pack!",
        source_name="My Pack",
        source_type="manual_zip",
        source_url="https://example.com/pack",
        author="Someone",
        license=SourceLicense(license="cc_by", user_confirmed=True),
        notes="hello",
    )
    restored = source_record_from_dict(source_record_to_dict(record))
    assert restored == record


def test_source_id_normalization():
    assert normalize_source_id("Kenney Generic Items!") == "kenney_generic_items"
    assert SourceRecord(source_id="A B/C", source_name="x", source_type="manual_zip").source_id == "a_b_c"


def test_attribution_required_detected():
    assert license_requires_attribution("cc_by")
    assert license_requires_attribution("oga_by")
    assert not license_requires_attribution("cc0")
    record = SourceRecord(
        source_id="s",
        source_name="s",
        source_type="manual_zip",
        license=SourceLicense(license="cc_by"),
    )
    assert record.license.attribution_required
    warnings = source_warnings(record)
    assert any("attribution" in warning for warning in warnings)
    assert any("author is missing" in warning for warning in warnings)
