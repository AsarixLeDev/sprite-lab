"""Smart source-prefill contracts shared by the Harvest CLI and web UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from _harvest_testdata import make_sprite_png
from spritelab.harvest.cli import main as harvest_main
from spritelab.harvest.source_prefill import (
    CC0_LICENSE_URL,
    available_source_presets,
    build_source_prefill,
)
from spritelab.product_core import ProjectContext
from spritelab.product_features.harvest import create_plugin
from spritelab.product_web import create_app

OGA_SOURCE_URL = "https://opengameart.org/content/behrs-2500-pixel-battle-axes-32x32-archive"
OGA_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
OGA_DIRECT_URL = "https://opengameart.org/sites/default/files/battleaxes_01.zip"


def _oga_source_html(*, extra_license_link: str = "", extra_file_link: str = "") -> bytes:
    return (
        '<div class="node node-art view-mode-full clearfix">'
        '<div class="field field-name-title"><h1>Behr\'s 2500+ Pixel Battle Axes 32x32 Archive</h1></div>'
        '<div class="field field-name-author-submitter">Submitted by Behrtron</div>'
        '<div class="field field-name-field-art-licenses">'
        f'<a href="{OGA_LICENSE_URL}">CC0 1.0 public domain</a>{extra_license_link}</div>'
        '<div class="field field-name-body">Totally free in the public domain.</div>'
        '<div class="field field-name-field-art-files">'
        f'<a href="{OGA_DIRECT_URL}">battleaxes_01.zip</a>{extra_file_link}</div>'
        "</div>"
    ).encode()


def test_known_source_profiles_are_useful_without_assuming_pack_specific_licenses() -> None:
    labels = {value["label"] for value in available_source_presets()}
    assert {"OpenGameArt", "Kenney", "itch.io"} <= labels

    kenney = build_source_prefill("https://kenney.nl/assets/new-platformer-pack")
    assert kenney.preset_id == "kenney"
    assert kenney.source_id == "kenney.new-platformer-pack"
    assert kenney.title == "New Platformer Pack"
    assert kenney.creator == "Kenney"
    assert kenney.license_id == "cc0-1.0"
    assert kenney.license_evidence_url == CC0_LICENSE_URL
    assert kenney.terms_evidence_url == "https://kenney.nl/terms-of-service"
    assert kenney.direct_download_url == ""
    assert kenney.review_fields == ("direct_download_url",)

    opengameart = build_source_prefill("https://opengameart.org/content/pixel-art-platformer-2")
    assert opengameart.preset_id == "opengameart"
    assert opengameart.source_id == "oga.pixel-art-platformer-2"
    assert opengameart.title == "Pixel Art Platformer"
    assert opengameart.license_id == ""
    assert opengameart.license_evidence_url == ""
    assert "license_id" in opengameart.review_fields

    itch = build_source_prefill("https://grafxkid.itch.io/mini-pixel-pack-3")
    assert itch.preset_id == "itchio"
    assert itch.source_id == "itchio.grafxkid.mini-pixel-pack"
    assert itch.creator == "Grafxkid"
    assert itch.license_id == ""
    assert itch.terms_evidence_url == "https://itch.io/docs/legal/terms"


def test_retained_opengameart_evidence_prefills_exact_bound_fields() -> None:
    prefill = build_source_prefill(OGA_SOURCE_URL, retained_source_bytes=_oga_source_html())

    assert prefill.title == "Behr's 2500+ Pixel Battle Axes 32x32 Archive"
    assert prefill.creator == "Behrtron"
    assert prefill.license_id == "cc0-1.0"
    assert prefill.license_evidence_url == OGA_LICENSE_URL
    assert prefill.direct_download_url == OGA_DIRECT_URL
    assert prefill.attribution_text == "Behrtron"
    assert prefill.review_fields == ("terms_evidence_url",)


def test_retained_opengameart_evidence_does_not_guess_between_file_links() -> None:
    extra = '<a href="https://opengameart.org/sites/default/files/alternate.zip">alternate.zip</a>'
    prefill = build_source_prefill(
        OGA_SOURCE_URL,
        retained_source_bytes=_oga_source_html(extra_file_link=extra),
    )

    assert prefill.direct_download_url == ""
    assert "direct_download_url" in prefill.review_fields


def test_retained_opengameart_evidence_does_not_guess_between_license_links() -> None:
    extra = '<a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>'
    prefill = build_source_prefill(
        OGA_SOURCE_URL,
        retained_source_bytes=_oga_source_html(extra_license_link=extra),
    )

    assert prefill.license_id == ""
    assert prefill.license_evidence_url == ""
    assert {"license_id", "license_evidence_url"} <= set(prefill.review_fields)


@pytest.mark.parametrize(
    "url",
    (
        "http://kenney.nl/assets/pack",
        "https://kenney.nl/assets/pack?token=private",
        "https://kenney.nl/assets/pack#download",
        "https://127.0.0.1/assets/pack",
        "https://user:secret@kenney.nl/assets/pack",
    ),
)
def test_prefill_rejects_non_public_or_private_url_shapes(url: str) -> None:
    with pytest.raises(ValueError):
        build_source_prefill(url)


def test_explicit_preset_rejects_a_spoofed_or_mismatched_host() -> None:
    spoof = build_source_prefill("https://notkenney.nl/assets/new-platformer-pack")
    assert spoof.preset_id == "generic"
    with pytest.raises(ValueError, match="does not match"):
        build_source_prefill("https://notkenney.nl/assets/new-platformer-pack", preset_id="kenney")


def test_cli_prefill_and_import_share_the_same_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    harvest_main(
        [
            "source-prefill",
            "https://kenney.nl/assets/new-platformer-pack",
            "--format",
            "json",
        ]
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["source_id"] == "kenney.new-platformer-pack"
    assert preview["license_evidence_url"] == CC0_LICENSE_URL

    image_root = tmp_path / "pack"
    make_sprite_png(image_root / "sprite.png")
    run_root = tmp_path / "harvest_runs"
    harvest_main(
        [
            "import-dir",
            "--dir",
            str(image_root),
            "--run-name",
            "smart_kenney",
            "--run-root",
            str(run_root),
            "--source-url",
            "https://kenney.nl/assets/new-platformer-pack",
            "--user-confirmed-license",
        ]
    )
    source = json.loads((run_root / "smart_kenney" / "sources.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert source["source_id"] == preview["source_id"]
    assert source["source_name"] == preview["title"]
    assert source["source_type"] == "local_directory"
    assert source["author"] == "Kenney"
    assert source["license"]["license"] == "cc0"
    assert source["license"]["license_url"] == CC0_LICENSE_URL
    assert source["license"]["user_confirmed"] is True


def test_web_smart_prefill_is_csrf_protected_network_free_and_pathless(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    app = create_app(ProjectContext(project), plugins=(create_plugin(sources=()),))
    client = TestClient(app)

    page = client.get("/harvest")
    assert page.status_code == 200
    assert "Paste one pack page" in page.text
    assert "OpenGameArt, Kenney, itch.io" in page.text
    assert not (project / "harvest_runs").exists()

    payload = {"source_page": "https://kenney.nl/assets/new-platformer-pack", "preset": "auto"}
    denied = client.post("/harvest/api/source-prefill", json=payload)
    assert denied.status_code == 403

    response = client.post(
        "/harvest/api/source-prefill",
        json=payload,
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert response.status_code == 200
    assert response.json()["prefill"]["source_id"] == "kenney.new-platformer-pack"
    assert response.json()["prefill"]["direct_download_url"] == ""
    assert not (project / "harvest_runs").exists()

    rejected = client.post(
        "/harvest/api/source-prefill",
        json={"source_page": payload["source_page"], "preset": "auto", "output_path": "C:/private"},
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == "browser_path_not_allowed"

    javascript = client.get("/harvest/static/harvest.js").text
    assert 'request("/harvest/api/source-prefill"' in javascript
    assert 'direct_download_url: "#probe-direct-url"' in javascript
    assert "Use detected OpenGameArt fields" in javascript
    assert ".innerHTML" not in javascript
