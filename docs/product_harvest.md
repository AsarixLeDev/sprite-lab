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
- read-only legacy run names containing immediate `sources.jsonl`,
  `candidates.jsonl`, or `imported.jsonl` files.

Legacy JSONL is bounded, hashed, counted, and never traversed or changed. Every
new start or retry requires a current inventory identity and an arithmetic
`reuse_exhausted` or `deficit_confirmed` record with a positive deficit. A
changed inventory invalidates that authorization.

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
Trusted-catalog v1 records lack automation-terms evidence and fail closed; they
are never treated as approved compatibility records. Every ordinary acquisition
revalidates the current source, license, and terms binding before backend
construction, including after a live catalog reload.

## Bounded source onboarding

An explicit catalog probe fetches only bounded public HTTPS evidence through
pinned public DNS. Before every distinct source, terms, license, or direct-link
path, it fetches and evaluates that origin's `robots.txt` using the fixed
`spritelab-harvest/0.1` user agent. Robots permission is retained separately and
is never treated as terms permission.

The retained creator page must visibly bind the exact title and creator, link
the submitted license evidence and exact direct download, and use an initially
accepted zero-cost license. Inert parsing ignores script, style, template,
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
explicit promotion. Promotion reopens and rehashes every retained page and raw
file, reruns robots/provenance/license/terms verification, rechecks current
capability evidence, and transactionally publishes the sorted trusted catalog.
Catalog-target changes detected immediately before replacement fail closed;
idempotent replay does not overwrite a conflicting source ID. A live-view
refresh callback may be retried without turning a durable successful promotion
into a failed response.

## Certified backend seam

The repository contains a hardened archive adapter, but it is activated only
when fixed repository-local audit-report and certificate artifacts independently
record PASS and bind the exact current implementation. The loader does not
generate, refresh, or infer PASS. Capability v3 binds a no-follow-read
transitive first-party module inventory plus Python, Pillow, OpenSSL, NumPy,
and PyYAML runtime versions. It also binds the runtime-selected conditioned
Dataset import callback ID, its full production code-inventory identity, and a
separate runtime identity covering the exact dependency inventory and isolated
worker executable, launch policy, and dependency roots. Certificate/report v4
evidence must reproduce those values; any code, callback, worker, or runtime
drift disables acquisition.

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

## Durable run and API

An explicit mutation uses one bounded project-wide cross-process
`.harvest.lock`. Any live ordinary acquisition or catalog probe blocks every
other managed Harvest start across service instances. Idempotency binds source catalog, evidence,
backend capability, limits, reuse evidence, and retry identity. Request and
authorization receipts precede backend construction; retries always use a new
exclusive run directory. State is atomic, logs are append-only and bounded,
and backend exception text is never persisted. One inode-bound run-directory
anchor spans acquisition receipt, manifest, handoff, state, event, and terminal
commit publication, so a rename/symlink ABA cannot redirect any transaction
write. A short renewable worker lease makes active ownership durable across
service instances. Progress has a hard per-stage event budget independent of
elapsed callback time, with terminal capacity reserved. COMPLETE is importable
only when a terminal commit binds the final state, terminal event, and handoff.

Cancellation is both in-memory and durable and is polled on a bounded cadence,
including during archive and final artifact rehashes. Failed operations retain
only explicit, identity-bound transaction/recovery residues; they do not run
unbounded recursive cleanup or report a committed result as failed merely
because warning delivery failed.

The web endpoints are:

- `GET /harvest/api/inventory` and `GET /harvest/api/sources`;
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

## Dataset handoff v2

Completed runs publish `spritelab.harvest.dataset-handoff.v2`. The handoff
contains only a managed run reference and portable relative paths. It binds:

- current signed source/license evidence and catalog identity;
- backend capability, code/downloader, limit, response, redirect, and
  acquisition-receipt identities;
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

Dataset integration is intentionally not wired. A future trusted integration
implements `DatasetImportCallback`, receives a server-owned
`DatasetImportRequest`, and must honor the supplied idempotency key. The browser
receives only an opaque Dataset reference and counts; it never receives the
artifact directory.
