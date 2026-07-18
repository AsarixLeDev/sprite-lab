# Product Harvest security and integration contract

The Product Harvest feature is a passive inventory and certified-acquisition
surface. The default plugin has no source catalog or acquisition backend and is
therefore unavailable for acquisition. Opening the page, status probing,
listing sources, and inventorying runs never construct a backend, contact a
network, or create `harvest_runs`.

## Inventory and reuse gate

Inventory scans only immediate repository-local children of `harvest_runs`.
It recognizes:

- managed `harvest-<id>` runs with v2 request/state evidence; and
- managed `probe-<id>` catalog-probe runs with bounded durable evidence; and
- read-only legacy run names containing immediate `sources.jsonl`,
  `candidates.jsonl`, or `imported.jsonl` files.

Each recognized managed name is parsed only as its exact run class through a
held child-directory anchor; corrupt managed evidence is unsafe and can never
fall back to the legacy parser. Legacy JSONL is bounded, hashed, counted, and
never traversed or changed. Every new start or retry requires a current
inventory identity and an arithmetic `reuse_exhausted` or `deficit_confirmed`
record with a positive deficit. A changed inventory invalidates that
authorization.

## Catalog and evidence

`HarvestSource` is administrator-owned; browsers select only its opaque
`source_id`. A source requires:

- explicit non-Unknown title, creator, attribution, source page, license
  evidence URL and text;
- CC0-1.0 or explicit public-domain status and zero-cost policy;
- exact public DNS download hosts, an HTTPS private acquisition reference, and
  an expected response SHA-256; and
- a `CatalogEvidenceBinding` covering verifier/code identity, retrieval times,
  requested and final URL identities, successful statuses, content hashes, and
  a deterministic attestation identity; and
- a v2 `CatalogAutomationTermsBinding` covering the exact retained terms URL,
  final URL/status, content hash, verification/expiry window, interpretation
  mode, matched declaration, limitation flag, tri-state decision, and decision
  identity.

Evidence is rejected when absent, changed, future-dated, expired, or valid for
more than 30 days. The verifier ID is fixed and its code identity is recomputed
from the live no-follow-read catalog validation modules; an arbitrary
hash-shaped verifier self-attestation is not authoritative. Private acquisition
URLs and query strings never enter web responses or durable evidence.
An otherwise fully valid append-only record whose verifier code identity is no
longer current remains immutable but is omitted from the active trusted source
set. Malformed, expired, policy-invalid, identity-invalid, unsafe-linked, or
conflicting records still fail the whole catalog closed.
Trusted-catalog v1 records lack automation-terms evidence and fail closed; they
are never treated as approved compatibility records. Every ordinary acquisition
revalidates the current source, license, and terms binding before backend
construction, including after a live catalog reload.

## Bounded source onboarding

The CLI and web app share a deterministic URL-only smart-prefill layer. One
public HTTPS pack-page URL produces a reviewable title, stable source ID,
platform defaults, and known evidence links for OpenGameArt, Kenney, itch.io,
or a generic creator page. Kenney supplies CC0 and linked site-terms defaults;
itch.io licenses remain deliberately blank because they vary by pack. In the
web app, checking bounded network authorization lets OpenGameArt smart prefill
read only the canonical detail page after its robots policy. The retained-page
parser then fills the exact visible title and submitter plus an unambiguous
supported license and single file link before any probe starts. It does not
fetch asset bytes, infer among multiple license/file choices, check any
authorization box, create `harvest_runs`, or weaken the later verifier.
Applying another draft clears the prior direct link so evidence from two packs
cannot be accidentally combined. After an
explicitly authorized OpenGameArt probe retained a structurally valid page but
rejected submitted field bindings, the web form offers an evidence-backed
recovery prefill. That retry draft uses the exact visible title and submitter,
fills only an unambiguous supported license and file link, and never changes
authorization checkboxes or relaxes final verification. The web form
explains when its probe action is disabled because current independent backend
capability evidence is missing, invalid, or could not be verified, and directs
the operator to configure or renew the repository Harvest certificate before
reloading.

An explicit catalog probe fetches only bounded public HTTPS evidence through
pinned public DNS. Before every distinct source, terms, license, or direct-link
path, it fetches and evaluates that origin's `robots.txt` using the fixed
`spritelab-harvest/0.1` user agent. Robots permission is retained separately and
is never treated as terms permission.

The retained creator page must visibly bind the exact title, creator, direct
download, selected license (or its exact evidence link), and an explicit
zero-cost declaration inside one structural source-pack block. Split-card
evidence is not composable. Paid indicators, license negation, all-rights-
reserved language, historical or negated gratis wording, or other restrictive
terms fail closed on both the pack block and retained license page. Explicit
counted offers such as `100 FREE tiles`, subject-bound declarations such as
`assets are available for free`, `archive ... totally free`, and pack-count
statements ending in `ALL FOR FREE` qualify only as complete visible block or
sentence declarations. Prefixes, suffixes, partial-quantity qualifiers, free
documentation/previews/shipping, and license-only wording such as
`royalty-free`, `free to use`, or `use freely` do not establish the pack's
acquisition price. Inert parsing ignores script,
style, template,
canvas, SVG, and other non-visible declarations. Likely governing Terms, ToS,
acceptable-use, or automation-policy links are detected from normalized anchor
labels and URL paths, while license/legal-code links are excluded. A submitted
terms URL must equal the single independently detected governing link. No
candidate permits only a source-page scan; multiple candidates fail as
ambiguous.

Terms interpretation is honest tri-state evidence:

- `BLOCK` for a visible explicit automation prohibition; the probe cannot be
  promoted;
- `ALLOW` only for one exact supported affirmative declaration; or
- `NO_PROHIBITION_OBSERVED` when a complete retained scan is silent, with
  `limited_evidence=true` and no claim of affirmative permission.

Prohibition matching scans the normalized whole visible document in addition
to closed block segments, so direct body/table text, inline-split phrases, and
unclosed blocks cannot disappear. Only exact closed text segments can produce
`ALLOW`; robots decisions never influence this terms classification.

The direct response is downloaded once into a single-link quarantine file only
to bind its SHA-256. A probe never opens, decodes, extracts, discovers,
classifies, imports, or publishes those bytes. Durable request, robots/pages,
raw receipt, events, lease, result, and terminal commit precede a separate
explicit promotion. The operator must first load the retained evidence and the
promotion request must echo its exact verification identity and source-pack
evidence-text SHA-256 under a separate zero-cost-review authorization. The
immutable promotion receipt binds those reviewed values. Promotion then
reopens and rehashes every retained page and raw file, reruns
robots/provenance/license/terms verification, rechecks current capability
evidence, and publishes one immutable no-replace source record below
`artifacts/harvest/trusted_catalog.d`. Record filenames are private SHA-256
identities of source IDs; the passive loader validates each record's exact
self-identity, source identity, inode/link contract, and then returns a
deterministically sorted merged catalog. The former single
`artifacts/harvest/trusted_catalog.json` file remains strict passive read-only
compatibility input and is never replaced by promotion. Conflicting source IDs
across legacy and append-only records fail closed. Each catalog source
atomically embeds the reviewed verification identity,
source-pack text hash, and probe-receipt identity in its v3 evidence
attestation. A crash before the separate promotion receipt is published cannot
leave an unreviewed trusted source; exact replay recovers that receipt.
Catalog publication never replaces an existing record; staged-inode
substitution, an unexpected hard link, or a target that appears concurrently
fails closed without changing prior trusted sources. POSIX named-fd
publication retains one validator-bound stage alias when anonymous staging is
unavailable; Windows moves the exact held handle and leaves a single link.
Promotion inventories inactive old-verifier records for source-ID conflicts,
source-count limits, and aggregate byte limits, while allowing a differently
identified current-verifier source to be appended. The inactive record is
never rewritten or silently re-attested.
Idempotent replay does not overwrite a conflicting source ID. A live-view
refresh callback may be retried without turning a durable successful promotion
into a failed response.

One monotonic deadline covers the entire probe. Every robots, source, terms,
license, and raw-quarantine request receives only the smaller of its stage cap
and the remaining whole-probe budget. Deadline expiry and cancellation both
publish durable terminal evidence and never publish a result or receipt.

## Certified backend seam

The repository contains a hardened acquisition adapter, but it is activated only
when fixed repository-local audit-report and certificate artifacts independently
record PASS and bind the exact current implementation. The loader does not
generate, refresh, or infer PASS. Capability v4 binds a no-follow-read
transitive first-party module inventory plus Python, Pillow, OpenSSL, NumPy,
and PyYAML exact runtime inventories. Installed distributions are bound through
the conditioned runtime's public full-owned-file inventory, including
unrecorded bytecode, native, and supplemental package files. Python evidence
compactly binds the exact executable, full standard-library tree (including
bytecode), native-extension tree, loaded interpreter libraries, TLS modules,
and loaded OpenSSL libraries without exposing local paths. It also binds the
runtime-selected conditioned Dataset import callback ID, its full production
code-inventory identity, and a separate runtime identity covering the exact
dependency inventory and isolated worker executable, launch policy, and
dependency roots. Certificate/report v5
evidence must reproduce those values; any code, callback, worker, or runtime
drift disables acquisition. One validation captures the module, runtime, and
callback identities in a single coherent snapshot and reuses that loader-sealed
snapshot during immediate service construction and passive source rendering;
it does not repeat the same full-tree hashes at each layer. The report contains
an exact, pathless, bounded
per-gate map: every original gate and the direct-image, retained-anchor,
whole-operation deadline, durable import-control, same-pack license/cost,
exact-pixel usability, and non-self-attestation gates must each record `PASS`.
An aggregate PASS cannot substitute for a missing or failed gate. Production
construction requires loader-sealed current backend and conditioned-callback
bindings plus live-reloadable independent evidence, and accepts only the exact
conditioned callback class bound to the project root. Arbitrary injected
backends are available only through the explicit test seam.

General product startup, navigation, and passive inventory do not perform this
runtime-wide attestation. The Harvest page renders from the trusted catalog and
local inventory first, then validates and caches one certified service when its
source API is requested. Every acquisition, source-probe, promotion, handoff,
and Dataset-import boundary still uses that certified service; mutating actions
reload current repository evidence into a fresh full snapshot before
authorization or publication.

Refresh an unchanged implementation's expiry from a clean tracked worktree:

```powershell
python -m spritelab harvest certificate refresh
```

If audited source code changed and the operator explicitly chooses to carry the
existing PASS gate decisions forward, both the report and certificate must be
rebound with the explicit waiver:

```powershell
python -m spritelab harvest certificate refresh --rebind-current-implementation --confirm-carry-forward-pass
```

The command preserves unique recovery copies of both prior artifacts. A
changed-code rebind requires restarting Sprite Lab before a probe or acquisition.

A certified adapter must provide both:

1. immutable `CertifiedBackendCapabilities` affirming HTTP success, HTTPS,
   DNS-resolution/private-network blocking, every redirect, response MIME and
   expected response hash, per-file hashes, file/count/byte/depth/name limits,
   archive expansion limits, duration, and cooperative cancellation; and
2. `AcquisitionBackend.acquire(source, destination, limits, *,
   cancel_requested, progress) -> AcquisitionResult`.

`AcquisitionResult.receipt` must bind the capability identity, final and
redirect URLs, status, MIME, expected/actual response SHA-256, response bytes,
duration, archive counts/expanded bytes, and every output file's portable
relative path, size, SHA-256, MIME, usable/quarantine decision, and controlled
taxonomy. The service independently validates the receipt and re-scans files.
It rejects links, reparse points, hard links, mounts/device crossings,
case/Unicode collisions, non-NFC or reserved names, MIME mismatches, TOCTOU
changes, excessive depth/count/bytes, and receipt disagreement.

The hardened adapter holds directory handles from before network access through
publication. DNS and response readers are bounded, every redirect is rechecked,
the raw response is copied to an immutable private snapshot, and the exact raw
descriptor and snapshot are rehashed with deadline/cancellation probes before
publication. Extraction, PNG discovery, final artifact hashing, receipt writes,
and publication use handle-relative no-follow operations. A platform without a
safe anonymous snapshot primitive retains a uniquely named mode-0400 snapshot
as hash-bound recovery evidence under `downloads/`; it is reported in the
acquisition receipt and is never treated as a Dataset candidate.

A direct static PNG, GIF, or WebP response is also accepted. Its original
response bytes remain hash-bound under `downloads/`; a single-frame,
MIME/magic-matched decode publishes `direct-image.png`. PNG bytes remain
unchanged, while GIF/WebP are deterministically converted to RGBA PNG. The v1
derivation record binds raw and output counts/hashes, decoded RGBA hash,
format, MIME, dimensions, frame count, recipe, and whether derivation occurred.
Animated or multi-frame direct responses fail closed.

Only exact 32x32, static, single-frame, technically usable PNG files are usable.
Archive members whose `.png` names do not contain the PNG file signature are
ignored before extraction and never enter the artifact receipt. Files with a
real PNG signature that are animated, non-32x32, corrupt, fully transparent, or
constant-RGBA are retained with explicit quarantine reasons. Usable uniqueness
is computed from exact decoded RGBA pixels, not encoded bytes, so differently
encoded duplicates count once and later copies are quarantined as
`duplicate_exact_pixels`.

## Durable run and API

An explicit mutation uses one bounded project-wide cross-process
`.harvest.lock`. Any live ordinary acquisition or catalog probe blocks every
other managed Harvest start across service instances. Idempotency binds source catalog, evidence,
backend capability, limits, reuse evidence, and retry identity. Request and
authorization receipts precede backend construction; retries always use a new
exclusive run directory. State is atomic, logs are append-only and bounded,
and backend exception text is never persisted. Retained output-root and
inode-bound run-directory anchors span creation, worker publication, and
multi-record job/handoff/evidence/cancel reads, so a rename/symlink ABA cannot
redirect a transaction or action. A short renewable worker lease makes active ownership durable across
service instances. Progress has a hard per-stage event budget independent of
elapsed callback time, with terminal capacity reserved. COMPLETE is importable
only when a terminal commit binds the final state, terminal event, and handoff.
Browser start, retry, and import actions retain only an opaque idempotency key
in session storage. The key survives reloads and response loss, contains no
path, URL, or payload, and is cleared only after success or a definitive client
error.

One monotonic deadline covers backend construction, acquisition, extraction,
validation, live configuration reload, handoff build/publication, and COMPLETE
commit. Only the remaining budget reaches the backend. Cancellation is both
in-memory and durable and is polled on a bounded cadence, including during
construction, archive work, final artifact rehashes, and commit boundaries. Failed operations retain
only explicit, identity-bound transaction/recovery residues; they do not run
unbounded recursive cleanup or report a committed result as failed merely
because warning delivery failed.

The web endpoints are:

- `GET /harvest/api/inventory` and `GET /harvest/api/sources`;
- `POST /harvest/api/source-prefill` for network-free, non-durable source drafts;
- `POST /harvest/api/jobs`;
- `GET /harvest/api/jobs/<run-id>` and `/evidence`;
- `POST /harvest/api/jobs/<run-id>/cancel` and `/retry`;
- `GET /harvest/api/jobs/<run-id>/handoff`; and
- `POST /harvest/api/jobs/<run-id>/import` when a callback is configured;
- `POST /harvest/api/probes` and `GET /harvest/api/probes/<probe-id>`;
- `GET /harvest/api/probes/<probe-id>/evidence`; and
- `POST /harvest/api/probes/<probe-id>/cancel`, `/retry`, and `/promote`.

All mutations use the real product shell's `X-CSRF-Token`. Payloads have exact
field allowlists and never accept a path, URL, destination, or output root.
The two start controls show an immediate indeterminate progress bar while their
request is accepted, then bind that bar to the returned acquisition or source
probe and update its durable stage and counters during normal polling.

## Dataset handoff v2

Completed runs publish `spritelab.harvest.dataset-handoff.v2`. The handoff
contains only a managed run reference and portable relative paths. It binds:

- current signed source/license evidence and catalog identity;
- backend capability, code/downloader, limit, response, redirect, and
  acquisition-receipt identities;
- archive versus direct-static-image response kind and any exact direct-image
  derivation/provenance record;
- artifact-manifest and artifact-set identities;
- per-file expected/actual SHA-256, size, MIME, usability, quarantine reason,
  and taxonomy; and
- aggregate usable/quarantine/file/byte/taxonomy counts.

Every handoff read and every idempotent completed-run reuse rehashes all files.
Changed files, catalog evidence, backend identity, or limits fail closed.

The certified acquisition path always requires the catalog's exact expected
response SHA-256. The bounded onboarding workflow is the only expected-hash
bootstrap: its quarantine phase cannot extract or import, and only a later
explicit, fully reverified catalog promotion binds the observed hash.

The conditioned Dataset integration implements `DatasetImportCallback` and
receives only a server-owned `DatasetImportRequest`. The first import persists
a stable request identity and its original callback idempotency key; later
browser sessions with fresh keys recover or replay that same durable request.
Per-attempt RUNNING/CANCELLING/FAILED/CANCELLED/COMPLETE state and cancellation
receipts make interruption and retry observable. The callback receives the
whole-import monotonic deadline plus a durable cancellation probe. Cancellation
and the deadline are rechecked immediately after every mutation-lock
acquisition and before every durable publication, so a lock wait that outlasts
either can never publish. Terminal finalization is a compare-and-set on the
request identity, attempt number, and current RUNNING/CANCELLING status, so a
late attempt can never modify a newer attempt's durable outcome. Import
COMPLETE becomes visible only through an atomic
`dataset_import_terminal_commit.json` that binds the receipt identity, the
final state identity, the attempt, and the request identity; a receipt without
that commit reports INTERRUPTED, and the next explicit import idempotently
re-finalizes it without re-invoking the callback. Probe READY is likewise
visible only through a terminal commit that also binds the published READY
state identity. Job, handoff, and evidence responses expose only privacy-safe
import status, opaque Dataset reference, and counts; they never expose the
artifact directory.
