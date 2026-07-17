# Folder-based dataset intake and exception review

The Dataset-v3 product feature lets a normal computer user build an image dataset from a folder:

```powershell
python -m spritelab v3 dataset build "path/to/my-sprites"
```

The feature never edits, moves, renames, or deletes files in the input folder. Its result is written outside the input folder. It launches neither training nor a vision provider on its own.

## Smallest supported folder

```text
dataset_folder/
  images/
    sprite.png
  source.txt
  LICENSE
```

PNG files may instead be directly inside `dataset_folder`. Discovery is recursive, so Unicode names, spaces, nested folders, and `originals/` work without special handling.

`source.txt` can contain a pack name, creator, source URL, and provenance notes. `LICENSE` can be recognizable license text, an SPDX-style identifier, a license URL, or an explicit public-domain statement. Sprite Lab records the supplied evidence; it does not invent absent fields. A supplied but unrecognized license is quarantined for legal review rather than treated as permission.

For multiple sources, place evidence beside each pack:

```text
dataset_folder/
  pack_a/
    images/
    source.txt
    LICENSE
  pack_b/
    images/
    source.yaml
    license.yaml
```

Pack detection gives priority to explicit `source.yaml`/`license.yaml`, nearest evidence, an original sibling ZIP boundary, an explicit user grouping, and finally conservative directory inference. Independently evidenced subfolders remain separate. Ambiguous inferred boundaries are shown once in the wizard for confirmation.

The evidence names `LICENSE`, `LICENSE.txt`, `COPYING`, `README`, `README.txt`, `source.txt`, `source.yaml`, `metadata.yaml`, `credits.txt`, and `attribution.txt` are recognized case-insensitively. Original ZIPs are hashed as boundary evidence but are never modified or recursively unpacked by this workflow.

## Universal metadata wizard

An interactive build inspects every detected pack and opens one short, prefilled local-web form per incomplete pack. It offers OpenGameArt, Kenney, Other downloaded source, My original work, and Custom/private agreement presets. Platform recognition may prefill a platform or explicit URL, creator, and title; it never assumes a license, claims ownership, copies another pack's license, scrapes, or downloads.

Every complete pack records:

- creator or rights holder and pack title;
- source type;
- source page URL, or an explicit original-work declaration;
- license identifier and a license URL or evidence file;
- attribution text when the license requires it;
- optional direct-download URL, version, acquisition date, and notes.

`Unknown` is a valid declaration but keeps that pack quarantined. My original work requires an explicit rights declaration plus a chosen license or usage policy. Sprite Lab records the declaration but cannot verify ownership or provide legal advice.

Normalized metadata is stored under the project at `datasets/source_metadata/`. Each sidecar binds both the declaration and the canonical source path, pack identity, complete evidence inventory, archive identity, and every covered PNG identity. Sprite Lab rejects project-side metadata storage that would overlap the selected input. The imported folder is not changed. Export metadata is a separate explicit wizard action and never overwrites existing `source.yaml` or `LICENSE.txt` files.

For automation, use the documented JSON batch schema printed into `metadata-required.json`:

```powershell
python -m spritelab v3 dataset build "path/to/assets" --metadata-file "path/to/project/metadata-required.json"
```

The batch file has this JSON shape (the noninteractive command writes a prefilled copy):

```json
{
  "schema_version": "spritelab.dataset.pack_metadata_batch.v2",
  "canonical_input_root": "path/to/assets",
  "confirmed_pack_roots": [],
  "packs": [
    {
      "pack_relative_root": ".",
      "pack_id": "pack_0123456789abcdef01234567",
      "source_binding": {
        "source_binding_schema": "spritelab.dataset.pack_source_binding.v2",
        "canonical_source_path": "path/to/assets",
        "input_root": "path/to/assets",
        "pack_relative_root": ".",
        "pack_identity": "pack_0123456789abcdef01234567",
        "pack_boundary_evidence": "conservative_directory_root_inference",
        "pack_boundary_status": "confirmed",
        "evidence_file_hashes": {},
        "evidence_file_identities": {},
        "archive_sha256": null,
        "archive_identity": null,
        "covered_file_count": 1,
        "covered_file_identities": {"sprite.png": "0000000000000000000000000000000000000000000000000000000000000000"},
        "covered_files_digest": "0000000000000000000000000000000000000000000000000000000000000000"
      },
      "creator_or_rights_holder": "Artist name",
      "pack_title": "Pack title",
      "source_type": "other_downloaded",
      "source_page_url": "https://example.test/pack",
      "original_work_declaration": false,
      "license_identifier": "cc_by",
      "license_url": "https://creativecommons.org/licenses/by/4.0/",
      "license_evidence_file": "LICENSE.txt",
      "attribution_text": "Artist name - Pack title",
      "permission_confirmed": false,
      "direct_download_url": "",
      "version": "",
      "acquisition_date": "",
      "notes": ""
    }
  ]
}
```

The command generates the canonical root, pack ID, and source-binding fields. Edit only the declaration fields. The batch must contain exactly one row for every current pack. Applying it to another folder, or after any covered PNG, evidence file, archive, pack boundary, or identity changes, is rejected atomically; regenerate the template instead. Declaration text must be JSON strings, declaration flags must be JSON booleans, and URL fields accept only valid HTTP(S) URLs.

Project-side metadata commits use a store-level interprocess lock and a durable transaction journal. If Sprite Lab or the computer stops during a multi-pack commit, the next metadata reader or writer recovers the whole old generation before the commit marker, or the whole new generation after it; it does not accept a mixed batch.

When standard input is noninteractive, the command never opens a browser or guesses declarations. It preprocesses the folder, quarantines incomplete packs, emits structured JSON, writes the batch template, and includes the exact next command.

## Optional files

- `source.yaml` or `source.yml`: structured source name, creator/author/publisher, URL, and notes.
- `license.yaml` or `license.yml`: structured `license`, `spdx`, `identifier`, `url`, or `public_domain` evidence.
- `labels.csv` or `labels.jsonl`: user-supplied semantic labels keyed by `path`, `relative_path`, `filename`, `image`, or `file`.
- `groups.csv`: user-supplied group information using the same path keys.
- `README` or `README.txt`: supporting source/license evidence and pack notes.
- `credits.txt` or `attribution.txt`: supporting attribution evidence.
- `originals/`: ordinary recursively discovered PNG files.
- `include.txt` and `exclude.txt`: one slash-normalized glob per line. Nonincluded or excluded PNGs receive `policy_excluded`; they do not silently disappear from the processed count.

No JSONL manifest is required.

## Current dispositions

Every discovered PNG has exactly one current disposition:

- `accepted`: passes technical decoding, source evidence, license allowlisting, duplicate handling, and the current suitability backend.
- `rejected`: controlled technical or policy exclusion, such as unreadable content, blank pixels, a duplicate, invalid alpha, or `policy_excluded`.
- `uncertain`: the suitability backend found a soft technical concern. It remains excluded unless a reviewer rescues it.
- `quarantined`: source or usable license evidence is missing or unverified. Review cannot fabricate this evidence.
- `requires_special_extraction`: an animation, multi-frame file, or ambiguous sprite sheet needs an explicit extraction operation before it can become one image record.
- `sheet_split`: the source sheet was safely retained while deterministic derived crops were processed.

The feature reuses the existing semantics-independent suitability audit. Accepted pixels are materialized with the existing Dataset-v5 content-bound identity and raw extraction backend. Controlled user-facing reasons include `unreadable`, `blank`, `duplicate`, `unsupported_animation`, `unresolved_sheet`, `invalid_alpha`, `missing_source`, `missing_license`, `unverified_license`, `unusual_dimensions`, and `policy_excluded`. Other current suitability reasons are preserved in normalized lowercase form.

Unknown semantic identity is not a rejection reason.

## Sprite sheets and duplicates

The existing extraction backend splits a sheet automatically only when its separator/grid plan is unambiguous and satisfies the extraction identity contract. Each derived record stores the source identity, crop rectangle, frame index, output decoded identity, and extraction-policy version. The source sheet is never changed.

Ambiguous sheets stay `requires_special_extraction`. The prefilled review shows the source, crop overlay, and previews, with Keep proposal, Adjust grid, and Exclude sheet actions. Multi-frame/animated files remain special extraction rather than being guessed.

Byte-identical and decoded-RGBA-identical duplicates are counted separately and exact copies are excluded. The canonical record retains every duplicate's source, license, and pack association. Canonical selection favors an eligible pack, so incomplete evidence in one pack cannot suppress a valid independently licensed copy. Average-hash near duplicates are reported and optionally reviewable but remain accepted; they are never silently deleted.

## Missing evidence

Missing creator, title, source, license, or boundary confirmation never causes a traceback. Only the affected pack is quarantined and excluded from training; complete independent packs continue. Image review cannot override a legal/provenance quarantine. Changed evidence, archives, added files, changed files, or deleted files invalidate only the affected sidecar and require revalidation.

## Review by exception

After an interactive build with exclusions, the CLI offers:

```text
Review excluded images now? [Y/n]
```

That prompt is shown only when both input and output are interactive terminals. Noninteractive builds and reviews never open a browser automatically. Review can always be started later:

```powershell
python -m spritelab v3 review
```

The local feature router serves `/dataset/review`. By default it displays rejected, uncertain, and special-extraction items. Quarantined legal exceptions and semantic exceptions are available through reason filters. Cards are prefilled with the current decision, show thumbnails, reasons, and source/license evidence, and classify the reason as technical or legal.

Routine review needs no typing:

- `K` keeps/rescues a suitability false positive.
- `E` excludes it.
- Left/right arrows move between visible cards.
- Buttons provide next/previous, reason filters, contact-sheet mode, and batch confirmation of current exclusions.

The review log is JSONL and append-only. A suitability false positive can be rescued and the accepted raw extraction is rebuilt. Review cannot rescue missing/unverified legal evidence, exact duplicates, unreadable files, blank files, animations, or unresolved sheets because a click cannot manufacture the missing evidence or pixels.

## Semantic behavior

A vision model is optional. Without a configured `VisionProvider`, technical preprocessing completes, the accepted image-only dataset is ready, semantic records are marked pending, and the conditioned dataset remains unavailable.

An integration layer can inject an object implementing the shared `VisionProvider` protocol into `DatasetIntakeService`. Intake first runs the provider health probe. It then sends one structured `dataset.semantic.propose` action for current accepted items. Returned values remain `provider_proposal_not_human_truth`. Valid proposals at or above `0.8` confidence are automatic prefills when conflict-free and healthy. Abstentions, omissions, lower confidence, and conflicts are excluded from semantic supervision by default and enter the optional human-rescue queue. Provider health failures are reported once rather than creating per-image human work, and never discard the accepted image-only dataset.

User-supplied rows in `labels.csv` or `labels.jsonl` are marked `human_supplied`; provider proposals never receive that status.

## Results and resume safety

The output directory contains:

- `result.json`: stable machine-readable `ProductResult` with all requested counts.
- `report_data.json` and `status_cards.json`: static report data and plugin cards.
- `review_queue.json` and append-only `review_log.jsonl`.
- `items.jsonl`: the current item records (generated internally; users do not author it).
- `preprocessing_state.json`: resumable source identities and completed per-file stages.
- `pack_detection.json`: detected pack boundaries and evidence bindings.
- `raw_extraction/`: accepted Dataset-v5 raw extraction artifacts.

Each resume hashes source bytes and evidence again. Unchanged completed items reuse preprocessing. Changed files are reprocessed. Deleted paths are omitted from current eligibility and listed in resumability data. Inputs are revalidated once more before accepted records are materialized.

The simple result reports processed PNGs, automatically accepted images, extracted crops, exact duplicates, technical rejections, sheets needing review, and images missing source/license information. Machine output additionally separates byte duplicates, decoded-pixel duplicates, possible near duplicates, uncertain images, quarantined images, semantic status, and image-only eligibility.

## Plugin registration

The exact feature export is:

```python
from spritelab.product_features.dataset.plugin import build_plugin

plugin = build_plugin()  # ProductPlugin
```

`build_plugin() -> ProductPlugin` owns the `dataset` and `review` CLI installers and the `/dataset/review` web router. An integration layer supplies the returned plugin to the instance-scoped product registry. The feature does not modify the central plugin registry or web shell.
