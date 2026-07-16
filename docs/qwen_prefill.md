# Qwen Metadata Auto-fill

## Purpose

Qwen metadata auto-fill suggests editable metadata fields for imported Dataset Maker sprites. Deterministic Sprite Lab code still validates PNGs and bundles, and the human user approves final metadata before export.

## What it can fill

- category
- object_name
- tags
- materials
- mood
- short description
- suggested sprite ID

Dominant colors are not model-filled; Sprite Lab computes them deterministically
from the imported bundle palette.

## What it must not fill

- license
- author
- source
- accept/reject status
- train/val/test split
- hard validation

## Recommended local server options

Use an OpenAI-compatible local vision-language model server such as vLLM, SGLang, llama.cpp server, LM Studio, or an Ollama-compatible endpoint.
On Windows, Ollama is the simplest local path if you have a supported vision model available locally.

## Example vLLM command

```bash
vllm serve "Qwen/Qwen3-VL-8B-Instruct"
```

## Example Ollama on Windows

Install Ollama for Windows, then in PowerShell:

```powershell
ollama pull qwen2.5vl:7b
ollama serve
```

Then use the native Ollama backend:

```powershell
python -m spritelab dataset-maker-prefill `
  --png path\to\sprite.png `
  --backend ollama `
  --model qwen2.5vl:7b `
  --base-url http://127.0.0.1:11434
```

The Ollama backend calls `http://127.0.0.1:11434/api/chat` and sends the sprite as a base64 PNG image.

## Example RunPod token

For a RunPod endpoint that exposes an OpenAI-compatible `/v1/chat/completions` API, pass the token as bearer auth:

```powershell
python -m spritelab dataset-maker-prefill `
  --png path\to\sprite.png `
  --backend openai_compatible `
  --model Qwen/Qwen3-VL-8B-Instruct `
  --base-url https://<your-runpod-endpoint>/v1 `
  --runpod-token $env:RUNPOD_API_KEY
```

If `--runpod-token` is omitted, Sprite Lab also checks `RUNPOD_API_KEY` and `RUNPOD_TOKEN`.

## Example llama.cpp / GGUF note

Use a Qwen3-VL-8B-Instruct GGUF quantization through a local OpenAI-compatible server if VRAM is limited.

## GUI usage

1. Import PNGs.
2. Enable auto-fill.
3. Enter the local server base URL.
4. Prefill the current sprite or all visible/filter-matched sprites.
5. Review suggestions.
6. Apply suggestions.
7. Export the dataset.

For bulk prefill, set **Bulk workers** above `1` to send multiple requests at the same time. On Windows with Ollama, start with `1` or `2`; use higher values only if your local model server has enough VRAM/CPU headroom.

## Troubleshooting GUI feedback

The prefill panel writes a Markdown report after each click. If nothing appears to change, check the report first:

- `Auto-fill is disabled`: turn on the Enable auto-fill checkbox.
- `Backend is set to none`: choose `rule_based`, `ollama`, or `openai_compatible`.
- `Selected sprites: 0`: the bulk scope and filters did not match any sprites.
- `could not connect`: start the local OpenAI-compatible Qwen server or fix the base URL.
- `Suggestions with metadata: 0`: the backend responded, but did not return usable metadata fields.

Failed connection attempts are not cached, so you can start the local model server and click prefill again.

The harvest batch command also supports concurrent workers:

```powershell
python -m spritelab harvest qwen_prefill `
  --run harvest_runs\pack_run `
  --backend ollama `
  --model qwen2.5vl:7b `
  --base-url http://127.0.0.1:11434 `
  --workers 2
```

The harvest command is blind-first by default: Qwen labels the image without a
filename hint, while deterministic filename rules are computed separately for
fusion and optional adjudication. It retries invalid JSON, empty/warning-only
responses, and responses below the minimum Qwen confidence.
The main reliability knobs are:

```powershell
python -m spritelab harvest qwen_prefill `
  --run harvest_runs\pack_run `
  --backend openai_compatible `
  --model Qwen/Qwen3-VL-8B-Instruct `
  --base-url http://127.0.0.1:8000/v1 `
  --retry-attempts 2 `
  --min-qwen-confidence 0.55 `
  --fusion-policy weighted
```

Use `--filename-hint` only if you explicitly want the filename-rule hint embedded
in the model prompt. Use `--no-retry-warning-only` only when you want the first
warning response recorded without another model call.

Exact duplicate propagation is enabled by default so identical decoded sprites
are labeled once and copied to duplicate rows. Optional near-duplicate
propagation compares perceptual hashes:

```powershell
python -m spritelab harvest qwen_prefill `
  --run harvest_runs\pack_run `
  --propagate-near-dups `
  --near-dup-threshold 2
```

Use `--no-propagate-dups` to disable exact duplicate propagation. Propagated rows
carry `prefill_propagated_from`; near-duplicate propagation also scales Qwen
confidence down by 10% and adds the `propagated_near_dup` quality flag.

## Label v2 fusion and review

The retired `filename-prefill`, `fuse-prefill`, and `prefill-review` commands
are replaced by source-aware label v2. It preserves the Qwen response as
evidence, treats trustworthy source filenames as the primary authority, and
marks unresolved conflicts for review.

To combine an existing Qwen prefill run with label v2:

```powershell
python -m spritelab harvest fuse-prefill-v2 `
  --run harvest_runs\pack_run `
  --out harvest_runs\pack_run\label_v2_suggestions.jsonl
```

For deterministic labels without a VLM request:

```powershell
python -m spritelab harvest label-v2 `
  --run harvest_runs\pack_run `
  --no-vlm
```

Review uncertain suggestions in the assisted golden GUI:

```powershell
python -m spritelab harvest assisted-golden `
  --run harvest_runs\pack_run `
  --host 127.0.0.1 `
  --port 7861
```

## CLI examples

Single PNG:

```bash
python -m spritelab dataset-maker-prefill \
  --png path/to/sprite.png \
  --backend openai_compatible \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1
```

Directory JSONL:

```bash
python -m spritelab dataset-maker-prefill \
  --png-dir raw_pngs \
  --out prefill_suggestions.jsonl \
  --backend openai_compatible \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --cache-dir .prefill_cache \
  --workers 4
```

## Important warning

Auto-filled metadata must be reviewed before training. The model can be wrong, ambiguous, or overconfident, and it never decides license, author, source, split, or accept/reject status.
