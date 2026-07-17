(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  if (!$('[data-harvest-root]')) return;
  const csrf = $('meta[name="spritelab-csrf"]')?.content ||
    decodeURIComponent(document.cookie.split("; ").find((value) => value.startsWith("spritelab_csrf="))?.split("=")[1] || "");
  const sourceSelect = $("#harvest-source");
  const summary = $("#harvest-summary");
  const runsNode = $("#harvest-runs");
  const detail = $("#harvest-detail");
  const inFlight = new Set();
  let state = {inventory: {runs: [], legacy_runs: []}, catalog: {sources: []}};
  try { state = JSON.parse($("#harvest-initial-state")?.textContent || "{}"); } catch (_error) { /* safe defaults */ }

  const idempotency = (prefix) => `${prefix}-${crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36)}`;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };
  const setBusy = (control, busy) => {
    if (!control) return;
    control.disabled = busy;
    control.setAttribute("aria-busy", String(busy));
  };
  async function request(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf, ...(options.headers || {})},
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      await response.text();
      throw new Error("Unexpected Harvest response. Reload the page and try again.");
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "Harvest request failed.");
    return payload;
  }
  async function once(key, control, action) {
    if (inFlight.has(key)) return;
    inFlight.add(key); setBusy(control, true);
    try { await action(); } finally { inFlight.delete(key); setBusy(control, false); }
  }
  function reuseEvidence() {
    const assessed = Number.parseInt($("#harvest-assessed").value, 10);
    const required = Number.parseInt($("#harvest-required").value, 10);
    const decision = $("#harvest-reuse-decision").value;
    return {
      decision,
      evidence_code: decision === "reuse_exhausted" ? "no_reusable_items" : "target_deficit",
      inventory_identity: state.inventory.inventory_identity,
      assessed_usable_items: assessed,
      required_usable_items: required,
      deficit_items: Math.max(required - assessed, 0),
    };
  }
  function authorization(prefix) {
    return {
      idempotency_key: idempotency(prefix), explicit_action: true,
      authorize_zero_cost: $("#harvest-zero-cost").checked,
      authorize_permissive_license: $("#harvest-license").checked,
      authorize_existing_inventory_reviewed: $("#harvest-inventory-reviewed").checked,
      reuse_evidence: reuseEvidence(),
    };
  }
  function renderSources(catalog) {
    state.catalog = catalog;
    sourceSelect.replaceChildren();
    for (const source of catalog.sources || []) {
      const option = element("option", `${source.title} · ${source.license.identifier}`);
      option.value = source.source_id; sourceSelect.append(option);
    }
    sourceSelect.disabled = !sourceSelect.options.length || !catalog.backend_configured;
    $("#harvest-limits").textContent = JSON.stringify(catalog.limits || {}, null, 2);
    renderSourceEvidence();
  }
  function renderSourceEvidence() {
    const source = (state.catalog.sources || []).find((item) => item.source_id === sourceSelect.value);
    const node = $("#harvest-source-evidence"); node.replaceChildren();
    if (!source) { node.append(element("p", "No certified catalog source is configured.")); return; }
    for (const [label, value] of [
      ["Creator", source.creator], ["License", source.license.identifier],
      ["Attribution", source.license.attribution_text], ["Expected response SHA-256", source.expected_response_sha256],
      ["Evidence expires", source.evidence_binding.expires_at], ["Catalog identity", source.catalog_identity],
    ]) { const row = element("p"); row.append(element("strong", `${label}: `), element("span", value)); node.append(row); }
  }
  function renderInventory(inventory, jobs = null) {
    state.inventory = inventory;
    $("#harvest-assessed").value = String(inventory.known_usable_items || 0);
    summary.textContent = `${inventory.run_count} managed, ${inventory.legacy_run_count} legacy read-only; ${inventory.known_usable_items} known usable; ${inventory.unsafe_entries} unsafe ignored.`;
    const legacy = $("#harvest-legacy"); legacy.replaceChildren();
    for (const run of inventory.legacy_runs || []) {
      const card = element("article", undefined, "harvest-evidence");
      card.append(element("strong", run.legacy_id), element("p", `${run.source_records} sources · ${run.candidate_records} candidates · ${run.imported_records} imported`), element("code", run.legacy_identity));
      legacy.append(card);
    }
    renderRuns(jobs || inventory.runs || []);
  }
  function action(label, callback, disabled = false) {
    const button = element("button", label, "button secondary"); button.type = "button"; button.disabled = disabled;
    button.addEventListener("click", () => once(`${label}-${button.dataset.run || "global"}`, button, callback).catch((error) => { summary.textContent = error.message; }));
    return button;
  }
  function renderRuns(runs) {
    runsNode.replaceChildren();
    for (const run of runs) {
      const card = element("article", undefined, "harvest-run");
      const heading = element("div"); heading.append(element("strong", run.source_id), element("code", run.run_id), element("span", run.status, "status-pill"));
      const metrics = element("p", `${run.usable_count || 0} usable · ${run.quarantined_count || 0} quarantined`);
      const provenance = element("p", run.provenance ? `${run.provenance.creator} · ${run.provenance.license.identifier}` : "Load Evidence for full provenance.");
      const progress = document.createElement("progress");
      progress.max = Math.max(run.total || 1, 1); progress.value = Math.min(run.current || 0, progress.max);
      progress.setAttribute("aria-label", `${run.stage || run.status} progress`);
      const operational = document.createElement("details");
      operational.append(element("summary", "Taxonomy, limits, and durable logs"));
      const operationalText = element("pre");
      operationalText.textContent = JSON.stringify({taxonomy_counts: run.taxonomy_counts || {}, limits: run.limits || {}, logs: run.events || []}, null, 2);
      operational.append(operationalText);
      const controls = element("div", undefined, "harvest-actions");
      const evidence = action("Evidence", async () => { detail.textContent = JSON.stringify(await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/evidence`), null, 2); }); evidence.dataset.run = run.run_id; controls.append(evidence);
      if (["QUEUED", "RUNNING", "CANCELLING"].includes(run.status)) { const cancel = action("Cancel", async () => { await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/cancel`, {method: "POST", body: JSON.stringify({explicit_action: true})}); await refresh(); }); cancel.dataset.run = run.run_id; controls.append(cancel); }
      if (["FAILED", "CANCELLED", "INTERRUPTED"].includes(run.status)) { const retry = action("Retry", async () => { await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/retry`, {method: "POST", body: JSON.stringify(authorization("retry"))}); await refresh(); }); retry.dataset.run = run.run_id; controls.append(retry); }
      if (run.handoff_ready) { const handoff = action("Dataset handoff", async () => { const value = await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/handoff`); detail.textContent = JSON.stringify(value, null, 2); if (value.dataset_import_available) { const importButton = action("Import into Dataset", async () => { const receipt = await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/import`, {method: "POST", body: JSON.stringify({explicit_action: true, idempotency_key: idempotency("dataset-import")})}); detail.textContent = JSON.stringify(receipt, null, 2); }); importButton.dataset.run = run.run_id; controls.append(importButton); } }); handoff.dataset.run = run.run_id; controls.append(handoff); }
      card.append(heading, metrics, provenance, progress, operational, controls); runsNode.append(card);
    }
  }
  async function refresh() {
    const inventory = await request("/harvest/api/inventory");
    const jobs = await Promise.all((inventory.runs || []).slice(0, 100).map(async (run) => {
      try { return await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}`); } catch (_error) { return run; }
    }));
    renderInventory(inventory, jobs);
    $("#harvest-poll-state").textContent = inventory.runs.some((run) => ["QUEUED", "RUNNING", "CANCELLING"].includes(run.status)) ? "Polling active jobs" : "No active jobs";
  }
  $("#harvest-start-form").addEventListener("submit", (event) => {
    event.preventDefault(); const button = $("#harvest-start");
    once("start", button, async () => {
      const payload = {source_id: sourceSelect.value, ...authorization("start")};
      await request("/harvest/api/jobs", {method: "POST", body: JSON.stringify(payload)}); await refresh();
    }).catch((error) => { summary.textContent = error.message; });
  });
  $("#harvest-refresh").addEventListener("click", (event) => once("refresh", event.currentTarget, refresh).catch((error) => { summary.textContent = error.message; }));
  sourceSelect.addEventListener("change", renderSourceEvidence);
  renderSources(state.catalog || {sources: []}); renderInventory(state.inventory || {runs: [], legacy_runs: []});
  Promise.all([request("/harvest/api/sources"), request("/harvest/api/inventory")]).then(([catalog, inventory]) => { renderSources(catalog); renderInventory(inventory); }).catch((error) => { summary.textContent = error.message; });
  window.setInterval(() => { if ((state.inventory.runs || []).some((run) => ["QUEUED", "RUNNING", "CANCELLING"].includes(run.status))) refresh().catch((error) => { summary.textContent = error.message; }); }, 1500);
})();
