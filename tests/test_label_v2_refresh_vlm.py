import json
import threading
import time

from _harvest_testdata import make_sprite_png
from spritelab.dataset_maker.prefill import MetadataSuggestion
from spritelab.harvest.cli import main


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def suggest(self, request):
        self.calls += 1
        return MetadataSuggestion(
            category="item_icon",
            object_name="backend_apple",
            tags=("backend",),
            confidence=0.9,
            source_consistency="consistent",
        )


class _SlowTrackingBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.sprite_ids: list[str] = []
        self._lock = threading.Lock()

    def suggest(self, request):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.sprite_ids.append(request.sprite_id)
        try:
            time.sleep(0.05)
            return MetadataSuggestion(
                category="item_icon",
                object_name=f"backend_{request.sprite_id}",
                tags=("backend",),
                confidence=0.9,
                source_consistency="consistent",
            )
        finally:
            with self._lock:
                self.active -= 1


def _write_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    png = make_sprite_png(run / "apple.png")
    imported = {
        "sprite_id": "apple",
        "status": "accepted",
        "source_id": "oga_cc0_food_ocal",
        "source_name": "Food",
        "relative_path": "apple.png",
        "final_png_path": str(png),
        "category": "unknown",
        "tags": [],
    }
    (run / "imported.jsonl").write_text(json.dumps(imported) + "\n", encoding="utf-8")
    (run / "qwen_suggestions.jsonl").write_text(
        json.dumps(
            {
                "sprite_id": "apple",
                "category": "item_icon",
                "object_name": "existing_apple",
                "tags": ["existing"],
                "confidence": 0.85,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def _write_multi_run(tmp_path, count: int = 4):
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for index in range(count):
        filename = f"apple_{index}.png"
        png = make_sprite_png(run / filename, color=(200, 40 + index, 40, 255))
        rows.append(
            {
                "sprite_id": f"apple_{index}",
                "status": "accepted",
                "source_id": "oga_cc0_food_ocal",
                "source_name": "Food",
                "relative_path": filename,
                "final_png_path": str(png),
                "category": "unknown",
                "tags": [],
            }
        )
    (run / "imported.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run


def test_label_v2_reuses_existing_vlm_without_refresh(tmp_path, monkeypatch, capsys) -> None:
    run = _write_run(tmp_path)
    backend = _CountingBackend()
    monkeypatch.setattr("spritelab.harvest.label_v2_pipeline.create_vlm_backend_from_args", lambda parsed: backend)

    main(["label-v2", "--run", str(run), "--backend", "rule_based"])

    assert backend.calls == 0
    stdout = capsys.readouterr().out
    assert "vlm_reused_existing: 1" in stdout
    assert "vlm_backend_called: 0" in stdout
    row = json.loads((run / "label_v2_suggestions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["vlm_descriptor"]["object_name"] == "existing_apple"
    assert row["vlm_status"] == "reused_existing"


def test_label_v2_refresh_vlm_ignores_existing_and_calls_backend(tmp_path, monkeypatch, capsys) -> None:
    run = _write_run(tmp_path)
    backend = _CountingBackend()
    monkeypatch.setattr("spritelab.harvest.label_v2_pipeline.create_vlm_backend_from_args", lambda parsed: backend)

    main(["label-v2", "--run", str(run), "--backend", "rule_based", "--refresh-vlm"])

    assert backend.calls == 1
    stdout = capsys.readouterr().out
    assert "vlm_reused_existing: 0" in stdout
    assert "vlm_backend_called: 1" in stdout
    row = json.loads((run / "label_v2_suggestions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["vlm_descriptor"]["object_name"] == "backend_apple"
    assert row["vlm_status"] == "backend_called"

    summary = json.loads((run / "label_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["vlm_stats"]["vlm_backend_called"] == 1
    report = (run / "label_v2_report.md").read_text(encoding="utf-8")
    assert "## VLM" in report
    assert "- vlm_backend_called: 1" in report


def test_label_v2_workers_call_backend_concurrently_and_preserve_order(tmp_path, monkeypatch, capsys) -> None:
    run = _write_multi_run(tmp_path, count=4)
    backend = _SlowTrackingBackend()
    monkeypatch.setattr("spritelab.harvest.label_v2_pipeline.create_vlm_backend_from_args", lambda parsed: backend)

    main(
        [
            "label-v2",
            "--run",
            str(run),
            "--backend",
            "rule_based",
            "--refresh-vlm",
            "--workers",
            "3",
            "--no-propagate-dups",
        ]
    )

    assert backend.calls == 4
    assert backend.max_active >= 2
    stdout = capsys.readouterr().out
    assert "Workers: 3" in stdout
    rows = [json.loads(line) for line in (run / "label_v2_suggestions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["sprite_id"] for row in rows] == [f"apple_{index}" for index in range(4)]
    assert [row["vlm_status"] for row in rows] == ["backend_called"] * 4
