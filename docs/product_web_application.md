# Sprite Lab v3 local web application

Sprite Lab v3 is a server-rendered local product built with FastAPI, Jinja2, Uvicorn, and plain browser assets. It has no Node build, CDN, or runtime internet dependency. The shell owns product-wide navigation, responsive layout, run/event presentation, accessibility, and web security; feature plugins continue to own their routes and feature behavior.

## Launch contract

The normal interactive command is:

```text
python -m spritelab v3
```

The shortcut expands to the stable local launch below: loopback-only on port
`8765`, opening the browser in an interactive desktop session.

```text
python -m spritelab v3 app --host 127.0.0.1 --port 8765
```

Use the explicit app command when different launch options are needed:

```text
python -m spritelab v3 app
```

Without overrides, the explicit app command uses the project UI configuration: it binds `127.0.0.1` and selects an available port when `ui.port` is `auto`. In an interactive desktop session it opens the default browser. Automation and terminal-only sessions can suppress that behavior:

```text
python -m spritelab v3 app --no-open
```

`--host` and `--port` override the project UI configuration. A non-loopback `--host` is rejected unless an authentication token is supplied using the runtime-only `SPRITELAB_WEB_TOKEN` environment variable or `--auth-token`. A non-loopback host from persisted configuration alone is insufficient because the host must be explicit at launch.

On Windows the listening address is exclusive. Starting another Sprite Lab
process on the same host and port fails instead of distributing requests across
two app versions.

The command does not start training, generation, a provider, or any feature backend. Existing commands such as `v3 status`, `v3 train`, and `v3 eval` keep their foundation behavior.

## Product navigation

The stable primary areas are Home, Dataset, Training, Evaluation, Playground, Runs, and Settings. Product plugin navigation entries are merged by route and sorted by their declared order. Entries that identify developer, internal, or audit surfaces are deliberately excluded from normal navigation. Technical details are available from a small footer link and contain only sanitized implementation facts.

The header presents the selected project and current run. At narrow widths the sidebar becomes an off-canvas navigation region controlled by a labeled button. The mobile layout does not require a separate route or template.

## Plugin mounting contract

The integration layer passes plugin instances directly to:

```python
from spritelab.product_web import create_app

app = create_app(project_context, plugins=plugins, settings=web_settings)
```

There is no global or central plugin registry. For each `ProductPlugin`, the shell:

1. calls `capability_probe(ProjectContext)` without launching a backend;
2. aggregates capabilities whose status is ready, running, or complete;
3. checks `required_backend_capabilities`;
4. mounts `web_router_factory(ProjectContext) -> APIRouter` only when all requirements are available;
5. merges non-developer `WebNavigationItem` entries;
6. reads `status_provider(ProjectContext) -> ProductResult` for product status cards and actions;
7. mounts package-owned static directories at `/plugins/{plugin_id}/static`;
8. records package-owned template directories in `app.state.spritelab_plugin_templates` for integration use.

Plugins do not edit a shell template or shared registry. A plugin route has precedence over the shell fallback for its navigation path. When a route factory is absent or a required capability is missing, the path still returns HTTP 200 with a normal unavailable state. For example, `/training` shows “Training is not available yet” and “No training backend is registered.” instead of a 404.

Every plugin mutation is covered by the shell CSRF middleware. Plugin-generated forms or JavaScript obtain the request token from `request.app.state.spritelab_csrf_token` and submit it as `X-CSRF-Token`. A plugin may serve its own complete template response; it is not required to inherit a shell template.

### Status cards and actions

Every plugin contributes its main `ProductResult` as a status card. A provider may also place a list of mappings in `ProductResult.data["status_cards"]`; supported fields are `title`, `status`, `message`, and a local `path`. `ProductResult.action` and `ProductResult.data["actions"]` appear as links to plugin-owned routes. The shell displays actions but does not implement the feature operation.

## Home status behavior

Home resolves Dataset, Training, and Evaluation from plugin navigation IDs, route paths, or feature names. It shows only supplied counts, so a dataset count appears only when `ProductResult.data["usable_images"]` exists. Missing metrics are not invented. The recommended next step follows this product sequence:

1. prepare the dataset;
2. start training;
3. set up evaluation;
4. open Playground.

An active run takes priority and recommends opening that run. Expected blockers remain ordinary cards or banners; commit hashes, source branches, audit IDs, raw schemas, private paths, and raw tracebacks do not appear.

## ProductEvent and SSE behavior

The shell reconstructs a run exclusively from `ProductEvent` records in the configured `events.jsonl`; it never scrapes logs to infer progress. `EventRepository` validates run IDs, confines reads to the configured runs directory, ignores invalid or oversized event lines, and parses records with `ProductEvent.from_dict`.

`GET /api/runs/{run_id}/events` is an SSE stream. Each valid record receives its durable one-based line number as the SSE `id`. Browsers reconnect automatically, send `Last-Event-ID`, and receive only subsequent events. A page refresh starts from the durable stream and reconstructs the current state. A terminal stream sends a final snapshot and closes. `?once=true` is available for deterministic local tests.

The snapshot contains current stage, status, exact counters, progress only when a positive total exists, elapsed time, ETA supplied by metrics or derived from actual elapsed progress, scalar metrics, recent messages, artifact display names, stage timeline, resume availability, and report availability. Missing totals and chart points remain missing. Completed runs are reconstructed from their durable events after a server restart.

Logs have a separate SSE endpoint at `GET /api/runs/{run_id}/logs`. Log text is never used as product state. HTML templates rely on Jinja autoescaping; browser updates use `textContent`; credential-like assignments and bearer values are redacted; configured private roots are replaced with `<project>`.

## Runs and reports

- `/runs` lists recent valid run state directories.
- `/runs/{run_id}` shows stage timeline, progress, counters, elapsed time, ETA, metrics, recent messages, artifact names, action visibility, log preview, report availability, and resume availability.
- `/runs/{run_id}/logs` shows a safe streaming log surface.
- `/runs/{run_id}/report` downloads a fixed-schema, pathless public run snapshot as inert JSON for a validated run ID. Raw feature report bytes are never streamed through this route, and the response uses `nosniff`, `no-store`, and a route-specific sandbox policy.

There is no arbitrary path or directory browser. Artifact references are presented as portable display names rather than absolute paths.

## Offline charts

`spritelab.product_web.components` exports reusable line chart, bar chart, distribution, metric card, image gallery, and run timeline helpers. Charts use semantic HTML and inline SVG or CSS, resize with their containers, inherit dark-theme colors, include a textual table alternative, and show “No data available” when empty. They filter non-finite values and never interpolate missing points.

## Security model

Loopback is the default boundary. A non-loopback launch requires an explicit host and a runtime authentication token. The token is never persisted by the shell, printed in a response, or included in technical details. Browser authentication uses an HTTP-only, strict same-site cookie established by a dedicated token form; API clients can use `Authorization: Bearer`.

All POST, PUT, PATCH, and DELETE routes require `X-CSRF-Token`. Responses apply a same-origin Content Security Policy, frame denial, MIME sniffing protection, no-referrer, and no-store headers. A persistent warning banner identifies non-loopback sessions.

The app does not expose environment variables or enumerate arbitrary files. Unexpected errors receive an `ERR-...` reference and a safe message. A raw traceback is logged locally for diagnosis but never returned to the normal product response.

## Accessibility and presentation

The shell includes semantic headings and regions, a skip link, keyboard-operable controls, visible focus, form labels, text status labels, screen-reader live regions, native accessible dialogs, dark and light themes, forced-color support, and reduced-motion handling. Status never relies on color alone. Empty, loading, disconnected, blocker, expected-unavailable, and unexpected-error states use distinct product copy.

The responsive breakpoint at 760px moves navigation off canvas and preserves a compact header. Browser verification at a 390px viewport found no horizontal page overflow. Dialog verification confirmed focus moved into the open dialog and the accessible label resolved from its heading.

## Verification

The focused tests use FastAPI’s in-process client and mocks; they start no browser, provider, trainer, generator, or external network service. They cover loopback launch, mocked browser opening, `--no-open`, plugin mounting/assets/navigation/status/actions, unavailable capabilities, home guidance, event replay/reconnect/completion, progress, charts, themes, narrow layout, escaping, CSRF, authentication, secret redaction, safe errors, Windows paths, and offline assets.
