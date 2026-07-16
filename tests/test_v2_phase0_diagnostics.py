from __future__ import annotations

from pathlib import Path


def test_docs_mention_v2_phase0_diagnostics() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "v2_phase0_diagnostics.md"
    text = docs_path.read_text(encoding="utf-8")
    assert "Phase 0" in text
    assert "--factored-cfg" in text
    assert "--cfg-base-scale" in text
    assert "--cfg-color-scale" in text
    assert "--null-fields" in text
    assert "ci95" in text
    assert "near_copy_rate" in text
    assert "v1.1" in text


def test_docs_mention_v1_1_factored_cfg_tradeoff() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "v1_1_factored_cfg.md"
    text = docs_path.read_text(encoding="utf-8")
    assert "v1" in text and "v1.1" in text
    assert "v1_1" in text
    assert "phase1_v1_1" in text
    assert "2.5" in text
    assert "3.0" in text
    assert "+0.03" in text
    assert "-0.019" in text
    assert "official" in text.lower()
    assert "optional" in text.lower()
    assert "No retraining" in text or "no retraining" in text.lower()
    assert "--export-preset v1.1" in text
    assert "build-v1-gallery" in text


def test_v1_default_docs_link_to_v1_1() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "v1_default.md"
    text = docs_path.read_text(encoding="utf-8")
    assert "v1.1" in text
    assert "v1_1_factored_cfg.md" in text
