from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from spritelab.harvest.label_v3.stage_cache_v3 import StageCache
from spritelab.harvest.label_v3.vlm_diagnostics import write_failure_diagnostic
from spritelab.harvest.label_v3.vlm_orchestration import (
    VlmBackendResponse,
    VlmRequestMetrics,
    VlmUnavailable,
    build_vlm_stage_cache_key,
    parse_stage_output,
    run_vlm_cascade,
)
from spritelab.harvest.label_v3.vlm_runtime import OpenAICompatibleV3Backend, VlmRuntimeConfig


def _key(stage: str, *, exact: str, geometry: str) -> str:
    return build_vlm_stage_cache_key(
        stage,
        exact_rgba_hash=exact,
        geometry_hash=geometry,
        image_view="views-v1",
        preprocessing_hash="prep-v1",
        model_identity="provider:model",
        prompt_version="prompt-v1",
        prompt_hash="prompt-hash",
        taxonomy_hash="taxonomy-v1",
        context_hash="context-v1",
    )


def test_stage_c_key_uses_exact_pixels_not_only_geometry() -> None:
    assert _key("stage_c_constrained_classification", exact="red", geometry="same-alpha") != _key(
        "stage_c_constrained_classification", exact="blue", geometry="same-alpha"
    )


def test_stage_b_explicitly_reuses_geometry() -> None:
    assert _key("stage_b_morphology", exact="red", geometry="same-alpha") == _key(
        "stage_b_morphology", exact="blue", geometry="same-alpha"
    )


def test_stage_cache_single_flight_and_atomic_write(tmp_path) -> None:
    cache = StageCache(tmp_path / "cache")
    calls = 0
    lock = threading.Lock()

    def compute():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get_or_compute("abcdef123456", compute), range(8)))
    assert calls == 1
    assert all(value == {"ok": True} for value, _ in results)
    assert sum(hit for _, hit in results) == 7
    assert cache.get("abcdef123456") == {"ok": True}
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_failed_result_is_not_cached_as_success(tmp_path) -> None:
    cache = StageCache(tmp_path / "cache")
    with pytest.raises(RuntimeError):
        cache.get_or_compute("failed-key", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert cache.get("failed-key") is None
    assert cache.get_or_compute("failed-key", lambda: {"ok": True}) == ({"ok": True}, False)


def test_stage_e_contract_rejects_classification_schema() -> None:
    assert parse_stage_output("stage_e_consistency", {"top_1": "gem"}) is None
    assert (
        parse_stage_output(
            "stage_e_consistency",
            {"consistency_result": "consistent", "confirmed_fields": ["color"], "conflicts": []},
        )
        is not None
    )


def test_failed_stage_metrics_reconcile_exactly() -> None:
    class Backend:
        model_identity = "mock:model"

        def infer(self, *, stage_id, **_kwargs):
            if stage_id == "stage_e_consistency":
                raise VlmUnavailable(
                    "timeout",
                    VlmRequestMetrics(http_attempts=2, retries=1, timeouts=2),
                )
            values = {
                "stage_a_blind_descriptor": {"literal_description": "a gem"},
                "stage_b_morphology": {"silhouette_family": "compact"},
                "stage_c_constrained_classification": {"canonical_object": "gem"},
                "stage_d_open_set_verify": {"verification_result": "supported"},
            }
            return VlmBackendResponse(values[stage_id], VlmRequestMetrics(http_attempts=1))

    result = run_vlm_cascade("gem", backend=Backend(), image_hash="rgba", geometry_hash="alpha")
    stages = (result.stage_a, result.stage_b, result.stage_c, result.stage_d, result.stage_e)
    totals = {
        name: sum(getattr(stage.metrics, name) for stage in stages) for name in VlmRequestMetrics.__dataclass_fields__
    }
    assert totals == {
        "logical_stage_requests": 5,
        "successful_stage_outputs": 4,
        "cache_hits": 0,
        "http_attempts": 6,
        "retries": 1,
        "timeouts": 2,
        "transport_failures": 0,
        "json_parse_failures": 0,
        "schema_validation_failures": 0,
        "fallbacks": 1,
        "abstentions_caused_by_backend_failure": 1,
    }


def test_diagnostic_is_bounded_sanitized_and_excludes_headers(tmp_path) -> None:
    secret = "rpa_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    content = f"Authorization: Bearer {secret} " + "x" * 5000 + " data:image/png;base64,AAAAABBBBB"
    path = write_failure_diagnostic(
        tmp_path,
        enabled=True,
        provider="provider",
        model="model",
        stage="stage_c",
        content=content,
        status_code=422,
        content_type="text/plain",
        exception=ValueError("bad"),
        prompt_hash="prompt",
        model_hash="modelhash",
        cache_hash="cache",
        excerpt_chars=80,
    )
    artifact_text = path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text)
    assert secret not in artifact_text and "Bearer" not in artifact_text and "base64,AAAAA" not in artifact_text
    assert "headers" not in artifact and "request" not in artifact and "image" not in artifact
    assert len(artifact["first_excerpt"]) <= 160 and len(artifact["last_excerpt"]) <= 160
    assert artifact["response_length"] == len(content.encode())


class _Response:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode()
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


@pytest.mark.parametrize("backend_name", ["openai_compatible", "ollama"])
def test_mocked_openai_and_ollama_smoke(monkeypatch, tmp_path, backend_name) -> None:
    payload = (
        {"choices": [{"message": {"content": json.dumps({"literal_description": "gem"})}}]}
        if backend_name == "openai_compatible"
        else {"message": {"content": json.dumps({"literal_description": "gem"})}}
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(payload))
    config = VlmRuntimeConfig(
        backend=backend_name,
        model="mock",
        base_url="http://127.0.0.1:9",
        retries=0,
        failure_diagnostics_dir=str(tmp_path),
    )
    backend = OpenAICompatibleV3Backend(
        config,
        dict.fromkeys(("checkerboard", "nearest_neighbor", "tight_foreground_crop"), "data:image/png;base64,AA=="),
    )
    response = backend.infer(stage_id="stage_a_blind_descriptor", image_ref="", prompt="p", prompt_hash="h")
    assert response.data == {"literal_description": "gem"}
    assert response.metrics.http_attempts == 1
