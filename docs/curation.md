# SpriteBundle curation

Human curation decisions are stored as JSON Lines in `curation.jsonl`. Each line
is one immutable decision event. Later events for the same `sprite_id` override
earlier events when computing the latest curation state.

Example line:

```json
{"sprite_id":"copper_vial_001","status":"accepted","tags":["item_icon","vial","copper"],"reasons":[],"notes":"Good silhouette.","timestamp":"2026-07-02T12:34:56Z","reviewer":null,"source_path":null}
```

Allowed statuses:

- `accepted`
- `rejected`
- `quarantine`
- `needs_fix`

Allowed reasons:

- `bad_alpha`
- `bad_palette`
- `bad_roles`
- `duplicate`
- `copyright_risky`
- `too_noisy`
- `too_empty`
- `wrong_category`
- `low_readability`
- `bad_silhouette`
- `bad_metadata`
- `bad_source`
- `not_pixel_art`
- `wrong_size`
- `other`

Tags are free-form but normalized to lowercase underscore tokens.

Raw bundle directories are never mutated by curation commands or the browser.
Only `curation.jsonl` is appended or rewritten. Future training export should
consume only latest decisions with status `accepted`.

## CLI examples

Append one decision:

```bash
python -m spritelab curation decide \
  --curation outputs/items/curation.jsonl \
  --sprite-id copper_vial_001 \
  --status accepted \
  --tag item_icon \
  --tag copper \
  --notes "Good silhouette."
```

Summarize decisions:

```bash
python -m spritelab curation summary --curation outputs/items/curation.jsonl
```

Validate decisions against bundle directories:

```bash
python -m spritelab curation validate \
  --bundles outputs/items/bundles \
  --curation outputs/items/curation.jsonl
```

Launch the optional browser:

```bash
python -m spritelab curation browser \
  --bundles outputs/items/bundles \
  --curation outputs/items/curation.jsonl
```

The browser requires the optional UI dependency:

```bash
pip install "sprite-lab[ui]"
```
