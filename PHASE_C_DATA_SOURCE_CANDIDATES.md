# Phase C data-source candidates

Research snapshot: 2026-07-18.

This is a link-only acquisition plan. The links below are **unprobed**: no
Harvest probe was run, no attachment or archive was downloaded, no redirect or
response was trusted, and no asset bytes were inspected. Page statements and
advertised counts are leads, not intake evidence. A source becomes usable only
after the certified Harvest backend records and validates the complete receipt
described at the end of this document.

“CC0”, “CC-BY”, dimensions, cost, and yields below report what the linked
publisher page says. They are not new legal conclusions and do not override the
license text bound by a future receipt. “Strict-Harvest-ready” means only that
the queue has an author/publisher page, a supported direct ZIP candidate,
explicit page-level cost evidence, a selectable license, and a bounded intended
use. It does **not** mean acquired, certified, eligible, or approved for
training.

Canonical license reference used by the source pages:
[CC0 1.0 Universal deed](https://creativecommons.org/publicdomain/zero/1.0/).
CC0 does not supply trademark, patent, privacy, publicity, warranty, or
endorsement rights, so those boundaries still need review.

## Strict eight-receipt queue

Dataset-v5 accepts at most eight selected managed dataset receipts. The queue
therefore treats every attachment as a separate source and never combines
shards under one receipt.

| Receipt | Publisher page | Unprobed direct ZIP | Page evidence: license and cost | Advertised dimensions and nominal yield | Required caveat |
|---|---|---|---|---|---|
| 1 — OPP2017 Jungle and temple | [OpenGameArt page](https://opengameart.org/content/opp2017-jungle-and-temple-set) | [opp1_jungle_tiles.zip](https://opengameart.org/sites/default/files/opp1_jungle_tiles.zip) | The page offers CC0 among several selectable licenses, says the work is public domain, and explicitly says it is free to use, modify, and sell. Use CC0 only if the future receipt binds that exact choice and the attachment evidence agrees. Cost candidate: `free`. | Page says 32×32 and “over 500 tiles.” | The page describes grid tiles, not necessarily more than 500 independent exact-32 PNG files; some platforms extend beyond one 32×32 cell. Strict dimension inventory and any approved frame/crop derivation determine the eligible yield. |
| 2 — OPP2017 Sprites, characters, objects, effects | [OpenGameArt page](https://opengameart.org/content/opp2017-sprites-characters-objects-effects) | [opp2_sprites.zip](https://opengameart.org/sites/default/files/opp2_sprites.zip) | Same OPP public-domain/free-to-use-modify-sell statement and selectable CC0 evidence as above. Cost candidate: `free`. | Page advertises “over 100 sprites and animations”; it is part of the 32×32 OPP family. | Animations, sheets, icons, and objects can have different file/canvas shapes. Do not equate an advertised sprite or animation with one exact-32 eligible image. |
| 3 — OPP2017 Cave and mine cart | [OpenGameArt page](https://opengameart.org/content/opp2017-cave-and-mine-cart) | [opp3_cave_tiles.zip](https://opengameart.org/sites/default/files/opp3_cave_tiles.zip) | The author says everything is public domain, CC0 is among the selectable licenses, and the page says it is free to use, modify, and sell. Cost candidate: `free`. | Page says 32×32 and “over 400 tiles.” | Multi-cell/platform pieces and sheets may not be exact-32 files. Bind the attachment’s embedded license/readme and count only strict inventory results. |
| 4 — OPP2017 Village and room | [OpenGameArt page](https://opengameart.org/content/opp2017-village-and-room) | [opp4_village_tiles.zip](https://opengameart.org/sites/default/files/opp4_village_tiles.zip) | Same OPP page-level public-domain/selectable-CC0 and explicit free-use evidence. Cost candidate: `free`. | Page says 32×32 and “over 400 tiles.” | Houses, rooms, clouds, and platform pieces may span cells or be sheets. Advertised count is not an eligible-file count. |
| 5 — OPP2017 Castle tiles | [OpenGameArt page](https://opengameart.org/content/opp2017-castle-tiles) | [opp5_castle_tiles.zip](https://opengameart.org/sites/default/files/opp5_castle_tiles.zip) | Same OPP page-level public-domain/selectable-CC0 and explicit free-use evidence. Cost candidate: `free`. | Page says 32×32 and “over 400 tiles.” | Floors, walls, waterfalls, stairs, and other platform pieces can span cells. Require exact dimensions, deduplication, and source-grounded labels. |
| 6 — Behr battle axes shard 01 | [OpenGameArt page](https://opengameart.org/content/behrs-2500-pixel-battle-axes-32x32-archive) | [battleaxes_01.zip](https://opengameart.org/sites/default/files/battleaxes_01.zip) | Page lists CC0, says “totally free in the public domain,” and identifies Public Domain in its attribution notice. Cost candidate: `free`. | The five-shard page advertises 2,500 individual 32×32 axes; nominally about 500 per shard, but no per-shard count is claimed. | High near-duplicate risk: page discussion notes repeated recognizable shapes and color/detail variants. Near-duplicate clustering and diversity caps must run before selection. |
| 7 — Behr battle axes shard 02 | [OpenGameArt page](https://opengameart.org/content/behrs-2500-pixel-battle-axes-32x32-archive) | [battleaxes_02.zip](https://opengameart.org/sites/default/files/battleaxes_02.zip) | Same CC0/public-domain/explicit-free page evidence. Cost candidate: `free`. | Same five-shard 2,500 exact-32 advertisement; per-shard yield unknown until inventory. | Separate source, probe, hash, receipt, import, and preview. Do not infer disjointness or eligible yield from the filename. Same near-duplicate risk. |
| 8 — Behr battle axes shard 03 | [OpenGameArt page](https://opengameart.org/content/behrs-2500-pixel-battle-axes-32x32-archive) | [battleaxes_03.zip](https://opengameart.org/sites/default/files/battleaxes_03.zip) | Same CC0/public-domain/explicit-free page evidence. Cost candidate: `free`. | Same five-shard 2,500 exact-32 advertisement; per-shard yield unknown until inventory. | Separate source, probe, hash, receipt, import, and preview. Same near-duplicate risk. Stop early if shards 01–02 already overconcentrate the taxonomy. |

The queue intentionally leaves battle-axe shards 04 and 05 outside the first
eight receipts. It prioritizes five thematic OPP packs for breadth, then uses
only as many disjoint weapon shards as the eight-receipt limit permits. Preview
the conditioned dataset after every import; do not continue merely to fill all
eight slots.

## Hold candidates

These are not in the strict queue. They need a policy, provenance, format, or
technical issue resolved before acquisition.

### Dungeon Crawl Stone Soup

- [Dungeon Crawl 32×32 tiles supplemental](https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental)
  advertises more than 3,000 supplemental individual 32×32 PNGs and a full pack
  of more than 6,000, lists CC0, and says the collaborator list is in the ZIP.
  Candidate full attachment URL, still unprobed:
  [Dungeon Crawl Stone Soup Full.zip](https://opengameart.org/sites/default/files/Dungeon%20Crawl%20Stone%20Soup%20Full.zip).
- This remains **HOLD**, despite excellent size and labeling potential. Its
  “use freely” wording does not satisfy the current acquisition parser’s
  explicit zero-cost evidence grammar. The license and price gates are separate:
  a CC0 statement is not a price statement. Do not weaken or bypass the policy;
  either record qualifying explicit zero-cost evidence or make a separately
  reviewed policy change.
- Prefer the supplemental-only attachment over the combined attachment if it is
  eventually admitted alongside the original pack, so the source identities and
  contributor evidence remain distinct. Never count the page’s 3,000/6,000
  claims as eligible yield before strict inventory and deduplication.

### High-priority canaries and exact-32 leads

- [2d Isometric Pixel art Cave tiles](https://opengameart.org/content/2d-isometric-pixel-art-cave-tiles)
  by Kipperfalcon: the current page lists CC0, says “ALL FOR FREE,” advertises
  101 tiles plus 69 props at 32×32/64×64/128×128, and exposes a small ZIP. The
  discussion records that an earlier README conflict was corrected. **HOLD for
  certified probe** because dimensions are mixed and the attachment/README must
  be bound; useful as a small canary once Harvest is certified.
- [Tiny Top Down Pack](https://opengameart.org/content/tiny-top-down-pack):
  author-posted CC0/free lead advertising 100 exact 32×32 tiles. Earlier
  research saw an unsupported RAR, while the page checked for this snapshot
  exposes `sbs_-_tiny_top_down_pack.zip`. **HOLD until the certified probe
  resolves and binds the current attachment**; do not rely on a stale attachment
  URL or format claim.
- [Pixel Texture Pack](https://opengameart.org/content/pixel-texture-pack):
  60+ 32×32 textures, with CC-BY 4.0 among the page choices and attribution to
  Jestan. Good voxel/environment diversity; attribution and exact license choice
  must be bound.
- [32FPS Textures](https://opengameart.org/content/32fps-textures): 150
  wall/floor/foliage textures tagged 32×32; page offers CC-BY 4.0, CC-BY 3.0,
  and OGA-BY. Use only one explicitly selected and receipt-bound license, with
  required credit.
- [Avesh: Texture Assets 32x](https://opengameart.org/content/avesh-texture-assets-32x):
  author says all textures are 32×32 voxel-terrain/structure textures; page
  lists CC-BY 4.0 and requests credit to Ailia. Exact count remains unknown.
- [N64 Texture Pack](https://opengameart.org/content/n64-texture-pack): CC0 and
  individually stored low-resolution textures, mostly 32×32. It explicitly has
  32×16 and other exceptions, so only the exact-32 subset can enter.
- [Prototype Textures (32px)](https://opengameart.org/content/prototype-textures-32x32px):
  CC0/public-domain prototyping textures with individual 32×32 files plus an
  atlas. Exclude the atlas and accept only strict exact-32 inventory entries.
- [Treasure chests 32x32](https://opengameart.org/content/treasure-chests-32x32):
  author-posted CC-BY 4.0, advertising 120 exact-32 chests. Bind the stated
  Bonsaiheldin/page attribution.
- [Spaceships 32x32](https://opengameart.org/content/spaceships-32x32):
  author-posted CC-BY 4.0, advertising five individual exact-32 ships plus a
  combined image. Exclude the combined image.

Other conservative leads from the earlier source-page review include Eldiran’s
32×32 RPG characters, DezrasDragons’ Fantasy RPG kit, OwlishMedia’s 32×32
tiles, Ganamoda’s sci-fi/forest pack, Enyph Games’ dungeon tiles,
AndHeGames’ creature sheets, and LordNeo’s floor tiles. They remain names for a
future page-evidence pass, not approved sources; no URL, license, count, or
dimension claim is inferred here without re-opening the primary page.

### Native exact-32 project-source leads

- [Curated Dungeon Crawl Stone Soup tiles repository](https://github.com/crawl/tiles)
  and its [November 2015 packaged snapshot](https://github.com/crawl/tiles/tree/master/releases/Nov-2015):
  the repository README says approved artists signed off to CC0, with separate
  [artist approvals](https://github.com/crawl/tiles/blob/master/ARTISTS.md) and
  a mandatory [unknown-license exclusion list](https://github.com/crawl/tiles/blob/master/TILES_UNDER_UNKNOWN_LICENSE.md).
  It offers more than 3,000 native 32×32 orthogonal tiles spanning terrain,
  monsters, items, effects, avatars, and UI. **HOLD:** use only this curated
  export, never assume the modern game repository has the same license; exclude
  every named unknown-license tile, treat unused/UI content cautiously, and
  allow sheet slicing only through an explicit receipt-bound recipe.
- [Alex’s 32x32 Dungeon Pack](https://alexs-assets.itch.io/32x32-dungeon-pack):
  page metadata reports CC0 and the pack advertises 36 transparent individual
  32×32 dungeon PNGs plus a sheet. **HOLD:** the download uses a dynamic itch
  handoff and the future probe must bind explicit cost evidence and the final
  archive; ingest the individual directory or the sheet-derived results, never
  both.
- [Hexany’s Monster Menagerie](https://hexany-ives.itch.io/hexanys-monster-menagerie):
  page reports CC0; version 0.3 advertises 64 static transparent 32×32 1-bit
  creatures and an individual Tiles directory. **HOLD:** bind version 0.3,
  dynamic download, price evidence, page/license bytes, and archive hash; do not
  mix older version archives that duplicate the same creatures.
- [Cavalier Sprite Pack](https://github.com/vllsystems/cavalier-sprite-pack):
  repository [license](https://github.com/vllsystems/cavalier-sprite-pack/blob/main/LICENSE)
  reports CC0 and the README advertises transparent 32×32/64×64 item PNGs across
  equipment, household, food, plants, tools, and weapons. **HOLD:** exact yield
  is unknown; select only `Transparent/32x32` after inventory and deduplicate
  background and 64×64 copies. Explicit zero-cost evidence still has to pass the
  acquisition policy.

### Normalization or future-Harvest project backlog

- [OpenDuelyst](https://github.com/open-duelyst/duelyst), pinned research tag
  `v1.97.13`: repository license reports CC0 and the project has more than 600
  animated units plus many icons/effects. Frames are variable-sized, animated,
  sheet/XML-described, and duplicated, so it is not an exact-static-32 source.
  It needs a separately designed normalization pipeline and must preserve the
  trademark/no-endorsement boundary.
- [Superpowers Asset Packs](https://github.com/sparklinlabs/superpowers-asset-packs):
  upstream repository reports CC0 and 1,200+ mixed files, but sizes and formats
  range across sprites, sheets, GIFs, previews, audio, and 3D. **Future
  normalization only.** Do not substitute a current marketplace repack whose
  terms add no-AI/no-redistribution restrictions.
- [Icon Machine](https://github.com/BrianMacIntosh/icon-machine): generated
  icons are reported as CC0 while the generator source is GPL-3.0. This is a
  future synthetic-data track, not a static corpus; every output would need the
  generator commit, seed, settings, and aggressive near-duplicate evidence.
- [Josh Moody Game Assets](https://gameassets.joshmoody.org/) and its
  [repository](https://github.com/joshmoody24/game-assets): site reports CC0 and
  repository uses the Unlicense, with potentially useful city/UI sprites.
  Exact dimensions and grid structure were not verified; snapshot and reconcile
  both license sources before any technical normalization.

## Minecraft 32×32 search result

No actual Minecraft Java resource pack reviewed so far clears all of these at
once: exact-32 technical suitability, original-source provenance, a license
that permits the intended redistribution/commercial use, and evidence that the
texture copyright—not merely repository code or page metadata—is covered.
Minecraft branding or visual similarity is not needed for the model; original
voxel-style textures above are the safer route.

### Rejected or held Minecraft packs

| Pack | Status | Evidence-based reason |
|---|---|---|
| [Faithful 32x](https://www.faithfulpack.net/license) | **REJECT** | Custom texture license requires credit and its license file, prohibits monetizing content containing the work, and reserves discretionary withdrawal. It does not permit the intended unrestricted commercial dataset/model use. |
| [Textureless](https://github.com/Null-MC/Textureless) | **HOLD / do not acquire** | Repository labels itself CC0, but also says colors/patterns are based on original vanilla Minecraft textures. A repository license does not by itself prove clean rights to upstream-derived texture expression; provenance and derivative boundaries are unresolved. It also contains PBR/material assets beyond simple exact-32 color PNGs. |
| [CotCotPack](https://www.curseforge.com/minecraft/texture-packs/cotcotpack) | **REJECT** | Page calls it a Faithful edit, credits multiple outside packs, and labels the license only as “Creative Commons 4.0” without a precise variant in the captured page evidence. Inherited Faithful restrictions and mixed provenance are not cleared. |
| [Improved Create 32x](https://www.curseforge.com/minecraft/texture-packs/improved-create-32x) | **REJECT** | Although the listing says MIT, the description says some textures are based on or taken from Faithful projects and it targets another mod’s art. The listing-level license cannot clear all upstream texture rights. |
| [Ozocraft Remix](https://modrinth.com/resourcepack/ozocraft-remix) | **HOLD / do not acquire** | Fan continuation/fork with permission, but current research did not establish a complete, machine-bindable chain covering every original and added texture for dataset redistribution and training. |
| [Lithos 32x](https://www.curseforge.com/minecraft/texture-packs/lithos-core-32x-1-6-1-11-complete) | **REJECT** | Publisher states all rights reserved for most work and “No Redistribution,” with third-party textures used by permission. Personal/server/video exceptions do not cover dataset publication or model training. |
| [MultiPixel](https://modrinth.com/resourcepack/multipixel) | **REJECT** | Listing shows CC-BY 4.0, but its own copyright statement forbids distributing modified packs, integration, second release, and commercial use; it is also a repost authorized by another original author and based on Minecraft textures. The conflicting/restrictive terms fail closed. |

### Safer original voxel-texture alternatives

Use the author-posted OpenGameArt leads listed above—especially Pixel Texture
Pack, 32FPS Textures, Avesh, N64 Texture Pack, and Prototype Textures—instead of
Minecraft-derived packs. They target voxel, FPS, or retro 3D surfaces without
claiming to reproduce Minecraft’s proprietary base art. CC-BY sources still
require exact attribution/license compliance, and CC0 sources still require
provenance, cost, dimension, and archive verification.

## Explicit non-Minecraft rejects

- **496 pixel-art medieval/fantasy icons / older 7Soul repack — REJECT.** The
  older third-party page says incompatible derivatives were removed, but the
  current author storefront is paid and marks the larger pack
  Attribution-NoDerivatives. Exact historical author evidence has not resolved
  that conflict; do not acquire the repack.
- **Nominally CC0 2024 dungeon page with a no-redistribution caveat — REJECT.**
  A page badge cannot override conflicting publisher restrictions.
- **Wyrmsun 900-item archive — REJECT for now.** The uploader identifies it as a
  redistribution; upstream credit, license, provenance, and dimensions would
  need to bind every selected item.
- **Super Epic single PNG/sheet — REJECT as a bulk source.** Harvest accepts a
  downloaded PNG only when that file itself is exactly 32×32; it does not
  silently split a larger sheet. It could be reconsidered only if the linked
  file itself passes exact dimensions.
- **Kenney Pixel Pack — HOLD.** CC0 is promising, but archive inspection and
  exact-32 yield are unknown; it is a normalization/future-Harvest source, not a
  strict current receipt.
- **RPG UI Icons — HOLD.** The archive mixes 16×16 and 32×32 assets. It needs
  strict dimension filtering and an exact eligible count.
- **32x32 Blocks and More — HOLD for diversity review.** Page is CC0 and ships
  individual files, but its discussion identifies the style as 8×8 blocks
  scaled to 32×32. It may add little effective-resolution diversity.
- **Open Pixel Project combined archive — do not use in the first queue.** Use
  the five creator-posted component ZIPs so source identity, theme, receipt, and
  rollback remain separate.
- **Galaxy Pixel Pack — REJECT for training.** Although the page also reports
  CC0, the author states that the work is not for generative-AI use. Do not use
  a general license badge to erase a specific creator restriction; it is also a
  changeable 2026 work in progress.
- **Screaming Brain Tiny Platformer — REJECT for training in the current
  plan.** It advertises 1,414 individual 32×32 tiles and CC0/public-domain terms,
  but the author also gives a no-AI statement. It additionally uses RAR,
  magenta-key backgrounds, and many procedural/recolor/adjacency variants.
- **32rogues — REJECT.** The author’s actual terms expressly prohibit
  generative AI/ML, NFTs, and redistribution; third-party CC0 descriptions do
  not override them.
- **Frogatto, OpenHV, and Universal LPC wholesale archives — REJECT.** They are
  mixed-license collections involving combinations of attribution,
  share-alike, noncommercial, OGA, or GPL terms. Only a separately manifested
  per-file source with independently proven compatible rights could be
  reconsidered.
- **Modern Dungeon Crawl Stone Soup game repository — REJECT as a corpus.** It
  contains exceptions and unknown-license legacy material. Only the curated
  `crawl/tiles` export and its explicit exclusions are a candidate.

## Evidence required during eventual certified probes

For each attachment, the managed source/probe/import receipt must record at
least the following. Missing or inconsistent evidence fails closed.

1. Stable candidate/source ID, publisher-page URL, requested attachment URL,
   UTC probe time, backend capability/certificate identity, and exact code and
   runtime identity.
2. Every redirect hop and the final URL; scheme, host, resolved addresses,
   peer/connection evidence required by the network policy, HTTP status,
   relevant response headers, declared MIME type, detected file type, and
   content-disposition filename.
3. Bounded streamed byte count, configured download/time/redirect limits,
   cancellation/deadline outcome, and SHA-256 of the exact downloaded bytes.
4. Publisher/uploader/creator identity, captured page-evidence identity,
   selected SPDX-like license identifier, license/deed URL, exact applicable
   license/readme bytes and hashes, explicit acquisition-price evidence, and
   all required attribution text/links. License and zero-cost evidence are
   separate fields.
5. Archive format and limits; member count; normalized member paths; compressed
   and expanded sizes/ratio; nested-archive status; and rejection of absolute,
   traversal, duplicate/colliding, link, reparse-point, device, encrypted, or
   otherwise unsupported members.
6. Per-member cryptographic hash, decoded format/mode, width, height, frame
   count, alpha/transparency facts, decode errors, and exact reason for every
   accepted, held, or rejected file. Only decoded 32×32 single-frame candidates
   enter the exact-size lane unless a separately approved, receipt-bound
   derivation recipe exists.
7. Exact duplicate and perceptual/near-duplicate group IDs, cross-source
   collisions, conditioning output hashes, eligible/excluded/quarantined
   counts, and diversity caps. Advertised page counts must remain separate from
   raw archive, decodable, exact-size, deduplicated, conditioned, reviewed, and
   selected counts.
8. Source-grounded label evidence: original relative path/name, pack/theme,
   creator-supplied category, proposed taxonomy label, confidence, review
   status, and any ambiguity. A generated label must never be presented as a
   publisher fact.
9. Managed staging/output identities, immutable publication/terminal-commit
   evidence, receipt identity, import receipt, Dataset-v5 selection receipt,
   post-publication rehash, and rollback/recovery result.

No link in this file authorizes a network action. Begin only after a current,
independent Harvest PASS/capability certificate reload-validates against the
final committed code identity, and preview the conditioned dataset after every
receipt before deciding whether more acquisition is needed.
