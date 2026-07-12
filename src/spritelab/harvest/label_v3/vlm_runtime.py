"""Runtime adapters for the conservative v3 VLM prefill cascade.

This module deliberately has no dependency on the legacy v2 label contract.
It reuses only the local OpenAI-compatible transport convention so a Qwen
server can be used without a cloud API.  The backend receives prepared visual
views and never receives record metadata from the blind stages.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v3.sha256_utils import sha256_short
from spritelab.harvest.label_v3.vlm_diagnostics import write_failure_diagnostic
from spritelab.harvest.label_v3.vlm_orchestration import (
    VlmBackendResponse,
    VlmRequestMetrics,
    VlmUnavailable,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmRuntimeConfig:
    backend: str = "none"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"
    structured_output: str = "auto"
    prompt_version: str = "vlm_prefill_v3_2"
    disable_thinking: bool = True
    timeout_seconds: float = 60.0
    retries: int = 1
    concurrency: int = 1
    retry_backoff_seconds: float = 1.0
    cache_dir: str = ""
    enrichment_model: str = ""
    enrichment_enabled: bool = False
    failure_diagnostics_enabled: bool = True
    failure_diagnostics_dir: str = "vlm_failure_diagnostics"

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", self.backend.strip().lower())
        object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/"))
        object.__setattr__(self, "retries", max(0, int(self.retries)))
        # RunPod Serverless safety ceiling; local callers may still choose a
        # lower value.  The cap is applied before any request is created.
        object.__setattr__(self, "concurrency", min(5, max(1, int(self.concurrency))))
        object.__setattr__(self, "retry_backoff_seconds", max(0.0, float(self.retry_backoff_seconds)))


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (232, 232, 232, 255))
    px = image.load()
    tile = max(2, min(size) // 8)
    for y in range(size[1]):
        for x in range(size[0]):
            value = 232 if ((x // tile) + (y // tile)) % 2 == 0 else 184
            px[x, y] = (value, value, value, 255)
    return image


def _crop_with_padding(image: Image.Image, padding: int = 2) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return image.copy()
    left, top, right, bottom = bbox
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_v3_views(path: str | Path) -> tuple[dict[str, str], str, str, str]:
    """Return unsmoothed visual views, preprocessing hash, and image hash."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    image_hash = sha256_short(rgba.tobytes().hex() + f"|{rgba.size}|RGBA", length=24)
    alpha = rgba.getchannel("A")
    geometry_hash = sha256_short(alpha.tobytes().hex() + f"|{rgba.size}", length=24)
    crop = _crop_with_padding(rgba)
    enlarged = crop.resize((crop.width * 16, crop.height * 16), Image.Resampling.NEAREST)
    checker = Image.alpha_composite(_checkerboard(enlarged.size), enlarged).convert("RGB")
    views = {
        "native_alpha": image_to_data_url(rgba),
        "checkerboard": image_to_data_url(checker),
        "nearest_neighbor": image_to_data_url(enlarged),
        "tight_foreground_crop": image_to_data_url(_crop_with_padding(rgba)),
    }
    preprocess_hash = sha256_short("v3_checkerboard_nearest16_tightpad2", length=16)
    return views, preprocess_hash, image_hash, geometry_hash


def deterministic_morphology(path: str | Path) -> dict[str, Any]:
    """Small deterministic morphology facts used alongside the VLM response."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return {
            "silhouette_family": "empty",
            "aspect_ratio": 0.0,
            "major_components": [],
            "relationships": [],
            "symmetry": "unknown",
            "multipart_or_fragment": "fragment",
            "complete_object": False,
            "broad_visual_family": "unknown",
            "warnings": ["empty_alpha"],
        }
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ratio = round(width / max(1, height), 3)
    family = "elongated" if ratio >= 1.6 or ratio <= 0.625 else "compact"
    return {
        "silhouette_family": family,
        "aspect_ratio": ratio,
        "major_components": [],
        "relationships": [],
        "symmetry": "unknown",
        "multipart_or_fragment": "complete_or_unknown",
        "complete_object": True,
        "broad_visual_family": family,
        "warnings": [],
        "deterministic": True,
    }


class OpenAICompatibleV3Backend:
    """OpenAI-compatible structured VLM transport for local Qwen/vLLM servers."""

    def __init__(self, config: VlmRuntimeConfig, views: Mapping[str, str]) -> None:
        self.config = config
        self.views = dict(views)
        endpoint_id = sha256_short(config.base_url.lower(), length=12)
        # Never include api_key in cache identity or diagnostic output.
        self.model_identity = f"{config.backend}:{endpoint_id}:{config.model}"
        self._structured_unsupported = False

    def _diagnose(
        self,
        *,
        stage_id: str,
        content: bytes | str,
        exception: BaseException,
        prompt_hash: str,
        cache_hash: str = "",
        status_code: int | None = None,
        content_type: str = "",
    ) -> None:
        write_failure_diagnostic(
            root=self.config.failure_diagnostics_dir,
            enabled=self.config.failure_diagnostics_enabled,
            provider=self.config.backend,
            model=self.config.model,
            stage=stage_id,
            content=content,
            status_code=status_code,
            content_type=content_type,
            exception=exception,
            prompt_hash=prompt_hash,
            model_hash=sha256_short(self.config.model, length=16),
            cache_hash=cache_hash,
        )

    def record_schema_failure(self, *, stage_id: str, raw: Any, prompt_hash: str, cache_hash: str) -> None:
        self._diagnose(
            stage_id=stage_id,
            content=json.dumps(raw, sort_keys=True, default=str),
            exception=ValueError("schema_validation_failure"),
            prompt_hash=prompt_hash,
            cache_hash=cache_hash,
            content_type="application/json",
        )

    def infer(
        self,
        *,
        stage_id: str,
        image_ref: str,
        prompt: str,
        prompt_hash: str,
        candidates: tuple[str, ...] | None = None,
    ) -> VlmBackendResponse:
        del image_ref, candidates
        if self.config.backend not in {"openai_compatible", "openai", "qwen", "ollama"}:
            raise VlmUnavailable(f"unsupported_v3_backend:{self.config.backend}")
        if self.config.backend == "ollama":
            return self._infer_ollama(stage_id, prompt, prompt_hash)
        selected_names = {
            "stage_a_blind_descriptor": ("checkerboard", "nearest_neighbor", "tight_foreground_crop"),
            "stage_b_morphology": ("checkerboard", "tight_foreground_crop"),
            "stage_c_constrained_classification": ("checkerboard", "tight_foreground_crop"),
            "stage_d_open_set_verify": ("checkerboard", "tight_foreground_crop"),
            "stage_e_consistency": ("checkerboard", "tight_foreground_crop"),
        }.get(stage_id, ("checkerboard",))
        selected = [self.views[name] for name in selected_names]
        max_tokens = {
            "stage_a_blind_descriptor": 320,
            "stage_b_morphology": 180,
            "stage_c_constrained_classification": 240,
            "stage_d_open_set_verify": 120,
            "stage_e_consistency": 160,
        }.get(stage_id, 240)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *({"type": "image_url", "image_url": {"url": view}} for view in selected),
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if self.config.disable_thinking:
            # vLLM/Qwen3 accepts this request-level chat-template override.
            # Non-thinking mode is a better fit for short, schema-bound prefill.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.config.structured_output != "off" and not self._structured_unsupported:
            payload["response_format"] = {"type": "json_object"}
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        attempts = retries = timeouts = transport_failures = json_failures = schema_failures = fallbacks = 0
        last_error = ""
        compatibility_retry_available = True
        attempt = 0
        while attempt <= self.config.retries:
            attempts += 1
            if attempts > 1:
                retries += 1
            response_bytes = b""
            content_type = ""
            try:
                body = json.dumps(payload).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    response_bytes = response.read()
                    response_headers = getattr(response, "headers", None)
                    content_type = response_headers.get("Content-Type", "") if response_headers else ""
                try:
                    response_data = json.loads(response_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    json_failures += 1
                    self._diagnose(
                        stage_id=stage_id,
                        content=response_bytes,
                        exception=exc,
                        prompt_hash=prompt_hash,
                        content_type=content_type,
                    )
                    raise
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                logger.debug("VLM response: stage=%s provider=%s", stage_id, self.config.backend)
                try:
                    parsed = json.loads(str(content))
                except json.JSONDecodeError as exc:
                    json_failures += 1
                    self._diagnose(
                        stage_id=stage_id,
                        content=str(content),
                        exception=exc,
                        prompt_hash=prompt_hash,
                        content_type="application/json",
                    )
                    raise
                return VlmBackendResponse(
                    parsed,
                    VlmRequestMetrics(
                        http_attempts=attempts,
                        retries=retries,
                        timeouts=timeouts,
                        transport_failures=transport_failures,
                        json_parse_failures=json_failures,
                        schema_validation_failures=schema_failures,
                        fallbacks=fallbacks,
                    ),
                )
            except urllib.error.HTTPError as exc:
                response_bytes = exc.read()
                detail = response_bytes.decode("utf-8", errors="replace")
                content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
                transport_failures += 1
                self._diagnose(
                    stage_id=stage_id,
                    content=response_bytes,
                    exception=exc,
                    prompt_hash=prompt_hash,
                    status_code=exc.code,
                    content_type=content_type,
                )
                if (
                    exc.code == 400
                    and self.config.structured_output == "auto"
                    and "response_format" in detail.lower()
                    and compatibility_retry_available
                ):
                    self._structured_unsupported = True
                    payload.pop("response_format", None)
                    compatibility_retry_available = False
                    fallbacks += 1
                    logger.info("VLM compatibility fallback: response_format unsupported; retrying plain JSON mode")
                    continue
                last_error = f"HTTP {exc.code}: {detail[:180]}"
                if exc.code == 429 and attempt < self.config.retries:
                    logger.warning(
                        "VLM rate limited: retry %d/%d after %.1fs",
                        attempt + 1,
                        self.config.retries,
                        self.config.retry_backoff_seconds * (2**attempt),
                    )
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                elif attempt < self.config.retries:
                    logger.warning("VLM HTTP %d: retry %d/%d", exc.code, attempt + 1, self.config.retries)
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, (TimeoutError,)) or isinstance(getattr(exc, "reason", None), TimeoutError):
                    timeouts += 1
                elif isinstance(exc, (KeyError, IndexError)):
                    schema_failures += 1
                    self._diagnose(
                        stage_id=stage_id,
                        content=response_bytes,
                        exception=exc,
                        prompt_hash=prompt_hash,
                        content_type=content_type,
                    )
                elif not isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
                    transport_failures += 1
                    self._diagnose(
                        stage_id=stage_id,
                        content=response_bytes,
                        exception=exc,
                        prompt_hash=prompt_hash,
                        content_type=content_type,
                    )
                if attempt < self.config.retries:
                    logger.warning(
                        "VLM request failure (%s): retry %d/%d", type(exc).__name__, attempt + 1, self.config.retries
                    )
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
            attempt += 1
        logger.warning(
            "VLM exhausted retries: provider=%s stage=%s; returning safe unavailable result",
            self.config.backend,
            stage_id,
        )
        raise VlmUnavailable(
            last_error or "v3_vlm_request_failed",
            VlmRequestMetrics(
                http_attempts=attempts,
                retries=retries,
                timeouts=timeouts,
                transport_failures=transport_failures,
                json_parse_failures=json_failures,
                schema_validation_failures=schema_failures,
                fallbacks=fallbacks,
            ),
        )

    def _infer_ollama(self, stage_id: str, prompt: str, prompt_hash: str) -> VlmBackendResponse:
        """Minimal native Ollama `/api/chat` adapter for Windows installations."""
        endpoint = self.config.base_url.rstrip("/") + "/api/chat"
        images = [
            view.split(",", 1)[1]
            for view in (
                self.views["checkerboard"],
                self.views["nearest_neighbor"],
                self.views["tight_foreground_crop"],
            )
        ]
        max_tokens = {
            "stage_a_blind_descriptor": 320,
            "stage_b_morphology": 180,
            "stage_c_constrained_classification": 240,
            "stage_d_open_set_verify": 120,
            "stage_e_consistency": 160,
        }.get(stage_id, 240)
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "think": not self.config.disable_thinking,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        last_error = ""
        attempts = retries = timeouts = transport_failures = json_failures = 0
        for attempt in range(self.config.retries + 1):
            attempts += 1
            if attempt:
                retries += 1
            response_bytes = b""
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    response_bytes = response.read()
                data = json.loads(response_bytes.decode("utf-8"))
                logger.debug("VLM response: native Ollama")
                parsed = json.loads(str(data.get("message", {}).get("content", "")))
                return VlmBackendResponse(
                    parsed,
                    VlmRequestMetrics(
                        http_attempts=attempts,
                        retries=retries,
                        timeouts=timeouts,
                        transport_failures=transport_failures,
                        json_parse_failures=json_failures,
                    ),
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, json.JSONDecodeError):
                    json_failures += 1
                elif isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                    timeouts += 1
                else:
                    transport_failures += 1
                status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                if isinstance(exc, urllib.error.HTTPError):
                    response_bytes = exc.read()
                self._diagnose(
                    stage_id=stage_id,
                    content=response_bytes,
                    exception=exc,
                    prompt_hash=prompt_hash,
                    status_code=status,
                    content_type="application/json",
                )
                if attempt < self.config.retries:
                    logger.warning(
                        "Ollama VLM request failure (%s): retry %d/%d",
                        type(exc).__name__,
                        attempt + 1,
                        self.config.retries,
                    )
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        logger.warning("Ollama VLM exhausted retries; returning safe unavailable result")
        raise VlmUnavailable(
            last_error or "ollama_v3_vlm_request_failed",
            VlmRequestMetrics(
                http_attempts=attempts,
                retries=retries,
                timeouts=timeouts,
                transport_failures=transport_failures,
                json_parse_failures=json_failures,
            ),
        )


def create_v3_backend(config: VlmRuntimeConfig, views: Mapping[str, str]) -> OpenAICompatibleV3Backend | None:
    if config.backend.strip().lower() in {"", "none"}:
        return None
    return OpenAICompatibleV3Backend(config, views)


def make_text_enricher(config: VlmRuntimeConfig):
    """Create a small text-only derived-description callable, or ``None``.

    The returned callable only accepts already accepted facts and a literal
    descriptor.  It does not receive source paths, candidates, or raw VLM
    classification output.
    """
    if not config.enrichment_enabled or not config.enrichment_model:
        return None
    if config.backend not in {"openai_compatible", "openai", "qwen", "ollama"}:
        logger.warning("LLM enrichment disabled: unsupported backend=%s", config.backend or "none")
        return None
    connected = False

    def enrich(facts: Mapping[str, str], literal_description: str) -> str:
        nonlocal connected
        prompt = (
            "Write exactly one concise factual sprite description from these accepted facts only. "
            "Do not add lore, properties, materials, specificity, background, action, style, or quality claims. "
            f"Facts: {json.dumps(dict(facts), sort_keys=True)}. Literal visual descriptor (not a new fact): {literal_description!r}. "
            'Return JSON only: {"enriched_description": string}.'
        )
        if config.backend == "ollama":
            result = _ollama_text_enrich(config, prompt)
            if result and not connected:
                connected = True
                logger.info("LLM connected: native Ollama enrichment response received")
            return result
        endpoint = config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": config.enrichment_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 120,
        }
        if config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        for attempt in range(config.retries + 1):
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = json.loads(str(content))
                result = str(parsed.get("enriched_description", ""))
                if result and not connected:
                    connected = True
                    logger.info("LLM connected: OpenAI-compatible enrichment response received")
                return result
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                ValueError,
                KeyError,
                IndexError,
            ):
                if attempt < config.retries:
                    logger.warning("LLM enrichment request failed: retry %d/%d", attempt + 1, config.retries)
                    time.sleep(config.retry_backoff_seconds * (2**attempt))
        logger.warning("LLM enrichment exhausted retries; using canonical description")
        return ""

    return enrich


def _ollama_text_enrich(config: VlmRuntimeConfig, prompt: str) -> str:
    endpoint = config.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": config.enrichment_model,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "think": not config.disable_thinking,
        "options": {"temperature": 0, "num_predict": 96},
    }
    for attempt in range(config.retries + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            return str(json.loads(str(data.get("message", {}).get("content", ""))).get("enriched_description", ""))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError):
            if attempt < config.retries:
                logger.warning("Ollama LLM enrichment request failed: retry %d/%d", attempt + 1, config.retries)
                time.sleep(config.retry_backoff_seconds * (2**attempt))
    logger.warning("Ollama LLM enrichment exhausted retries; using canonical description")
    return ""
