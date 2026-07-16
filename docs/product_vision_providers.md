# Vision-provider hub

Sprite Lab keeps vision labeling simple for normal use while preserving explicit controls for advanced setups. The
settings page is provided by `spritelab.product_features.providers.plugin.build_plugin()` at `/settings/vision`.
Integration owns final plugin and command registration; this feature does not edit the central registry or web shell.

## Normal setup

Choose one of these modes:

- **Automatic**: prefer an available local provider, then consider hosted providers allowed by the privacy policy.
- **Local**: consider local providers only.
- **Hosted**: consider hosted providers only.

Advanced choices are Local Ollama, vLLM / OpenAI-compatible, Hosted endpoint, or Custom plugin. Model selection can be
left empty for auto-detection when an endpoint exposes exactly one model. The connection-test action performs model-list
and model-detail health checks only. It rejects image input; labeling is a separate, privacy-gated action.

## Configuration

Persist references and endpoint metadata, never a credential:

```yaml
providers:
  vision:
    type: auto
    endpoint: http://127.0.0.1:8000/v1
    location: local
    model: my-installed-vision-model
    credential_env: SPRITELAB_VISION_API_KEY
    privacy_policy: ask_before_hosted
    timeout: 30
    batch_size: 4
    maximum_retries: 2
    capabilities:
      vision: true
      structured_output: true
      multiple_images: true
      batching: true
      maximum_image_count: 8
      maximum_payload_size: 20971520
```

For a hosted endpoint, set `type: hosted`, an explicit `https://` endpoint, and `location: hosted`. A non-loopback
OpenAI-compatible URL is rejected unless it is explicitly declared hosted. URLs are parsed as URLs rather than
filesystem paths, including on Windows.

Automatic mode also recognizes these non-secret environment settings when the project does not specify an endpoint:

- `SPRITELAB_VISION_ENDPOINT`
- `SPRITELAB_VISION_ADAPTER`
- `SPRITELAB_VISION_LOCATION`
- `SPRITELAB_VISION_MODEL`
- `SPRITELAB_VISION_CREDENTIAL_ENV` (the name of another environment variable)

`SPRITELAB_VISION_CREDENTIAL_ENV` is a reference. The referenced variable holds the runtime secret. Sprite Lab does not
persist it and does not include it in identities, provider metadata, error messages, or settings responses.

## Safe automatic discovery

Discovery is deterministic and inexpensive. It does not send images, make inference requests, scan ports, or inspect
arbitrary network ranges. Candidates are ordered as follows:

1. explicitly configured local endpoint;
2. local Ollama at its documented default endpoint;
3. installed local plugin providers;
4. explicitly configured hosted endpoint;
5. installed hosted plugin providers.

Within a tier, provider IDs give stable ordering. Discovery uses only:

- Ollama `GET /api/tags` to list models;
- Ollama `POST /api/show` for an explicitly selected model's declared `vision` capability;
- OpenAI-compatible `GET /models` beneath the configured API base URL;
- installed `spritelab.vision_providers` entry points.

The possible states are `available`, `unavailable`, `misconfigured`, `authentication_required`, and
`unsupported_model`. Endpoint response bodies and credentials are never copied into user-facing errors.

The Ollama endpoint and capability shapes follow the official [Ollama API introduction](https://docs.ollama.com/api/introduction),
[model list](https://docs.ollama.com/api/tags), and [model details](https://docs.ollama.com/api-reference/show-model-details)
documentation. The native image request and JSON-schema `format` follow the official
[vision](https://docs.ollama.com/capabilities/vision) and
[structured output](https://docs.ollama.com/capabilities/structured-outputs) documentation.

The generic adapter follows vLLM's documented [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
and [multimodal Chat Completions input](https://docs.vllm.ai/en/stable/features/multimodal_inputs/). The API base URL,
authentication reference, and model name always come from configuration or model discovery; Sprite Lab does not guess
hosted URLs or model names.

## Adapter matrix

| Adapter | Location | Discovery | Image request | Status |
|---|---|---|---|---|
| Ollama native | Local | `/api/tags`, selected-model `/api/show` | `/api/chat`, base64 images, JSON-schema `format` | Implemented and tested |
| OpenAI-compatible | Local or hosted | configured-base `/models` | configured-base `/chat/completions`, image data URLs, JSON-schema response format | Implemented and tested |
| Deterministic mock | Local/test | in memory | in memory | Implemented; no network |
| Entry-point plugin | Plugin-declared | delegated probe | delegated contract | Implemented and validated |
| RunPod-specific native | Hosted | none | none | Unavailable scaffold; use OpenAI-compatible |

Current official [RunPod vLLM documentation](https://docs.runpod.io/serverless/vllm/openai-compatibility) says its workers
implement OpenAI API compatibility. A separate RunPod payload or authentication adapter is therefore unnecessary and
would duplicate a supported contract. Selecting `type: runpod` returns a tested unavailable state directing the user to
the generic adapter; it does not fabricate a URL, header, payload, or model.

## Provider contract

Every provider exposes:

```text
provider_id
display_name
provider_kind
privacy_class
probe()
list_models()
validate_model()
capabilities()
estimate_request()
label_batch()
cancel()
health()
```

Capabilities are per adapter or provider declaration and include `vision`, `structured_output`, `multiple_images`,
`batching`, `maximum_image_count`, `maximum_payload_size`, `timeout_support`, `cancellation_support`, and
`local_or_hosted`. The hub checks required capabilities before image transfer. It does not infer that every model behind
an OpenAI-compatible endpoint is a vision model; advanced configurations can explicitly disable unsupported features,
and runtime rejections are normalized.

Third-party packages register an entry point in the `spritelab.vision_providers` group. The loaded object or zero-argument
factory must satisfy the full contract. Contract or import failures become an unavailable provider record with a redacted
exception type; they do not abort discovery of other providers.

## Normalized labels

Every successful image result must contain exactly:

```json
{
  "state": "labeled | abstained | needs_review",
  "domain": "string or null",
  "category": "string or null",
  "canonical_object": "string or null",
  "role": "string or null",
  "description": "string or null",
  "confidence": 0.0,
  "abstention_reasons": [],
  "provider_metadata": {}
}
```

Confidence must be numeric and between zero and one. An abstained label must retain at least one non-empty reason. Types,
empty strings, missing fields, extra fields, unknown image IDs, duplicates, and malformed JSON are not silently coerced.
The affected result becomes `provider_invalid_output`. Valid sibling results from the same completed response are kept.
This keeps conservative labeling and audit behavior intact.

## Privacy policies

| Policy | Local image request | Hosted image request |
|---|---:|---:|
| `local_only` | allowed | blocked before provider model validation or labeling |
| `allow_hosted` | allowed | allowed |
| `hosted_only` | blocked | allowed |
| `ask_before_hosted` | allowed | one confirmation before the first hosted batch in a hub run |

Declining confirmation sends no image. Once accepted, the hub does not prompt again for later hosted batches in the same
run. Provider discovery and health checks contain no image and remain safe under every policy.

## Reliability and resume behavior

- Batches are bounded by configured size, provider image count, and provider payload bytes.
- Timeouts are passed to the transport; cancellation is cooperative and discards a response received after cancellation.
- Retryable timeouts, rate limits, and server/unavailable errors use bounded exponential backoff. `Retry-After` seconds are
  honored only up to the configured bound.
- Attempts are finite. Exhausted retries become `provider_retry_exhausted`; with retries disabled, the original normalized
  error (for example `provider_timeout`) is retained.
- Partial success is committed per image. Only retryable failed images are retried.
- Successful prior image results can be supplied on resume and are skipped.
- Request identity hashes provider, model, prompt hash, ordered image IDs/content hashes, and stable request options.
- Response identity hashes the request identity and validated normalized payload. Mismatched plugin identities are rejected.

Built-in adapters use an injectable HTTP transport. Tests provide fake transports and explicitly fail any unexpected
request, so the test suite never contacts a real provider.

## Secrets and logging

Only environment-variable credential references are supported. An OS credential-store integration was not added because
the current base dependencies do not include a maintained credential-store library. Sensitive header names and query
parameters are redacted by shared helpers. Authorization values and referenced credentials are removed from diagnostic
text. Provider errors intentionally report normalized state rather than raw response bodies.

## Cost display

Cost is `unknown` by default. An estimate is calculated only when `pricing.per_request` and/or `pricing.per_image` is
explicitly configured, with an optional currency. The hub does not scrape or hardcode provider prices and does not treat
untrusted response metadata as pricing.

## CLI extension

After the integration layer registers the plugin, these commands are available:

```text
python -m spritelab v3 providers
python -m spritelab v3 providers detect
python -m spritelab v3 providers test
```

The first lists configured adapters without network access. `detect` and `test` run the documented health probes without
images. Final command registration remains an integration responsibility.
