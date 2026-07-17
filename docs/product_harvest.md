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
  a deterministic attestation identity.

Evidence is rejected when absent, changed, future-dated, expired, or valid for
more than 30 days. Private acquisition URLs and query strings never enter web
responses or durable evidence.

## Certified backend seam

No operational downloader is configured by this feature. A separately audited
adapter must provide both:

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

## Durable run and API

An explicit mutation uses a bounded cross-process `.harvest.lock`. A same-source
active run blocks another start. Idempotency binds source catalog, evidence,
backend capability, limits, reuse evidence, and retry identity. Request and
authorization receipts precede backend construction; retries always use a new
exclusive run directory. State is atomic, logs are append-only and bounded,
and backend exception text is never persisted.

The web endpoints are:

- `GET /harvest/api/inventory` and `GET /harvest/api/sources`;
- `POST /harvest/api/jobs`;
- `GET /harvest/api/jobs/<run-id>` and `/evidence`;
- `POST /harvest/api/jobs/<run-id>/cancel` and `/retry`;
- `GET /harvest/api/jobs/<run-id>/handoff`; and
- `POST /harvest/api/jobs/<run-id>/import` when a callback is configured.

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

Dataset integration is intentionally not wired. A future trusted integration
implements `DatasetImportCallback`, receives a server-owned
`DatasetImportRequest`, and must honor the supplied idempotency key. The browser
receives only an opaque Dataset reference and counts; it never receives the
artifact directory.
