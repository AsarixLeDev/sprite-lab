(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  if (!$('[data-harvest-root]')) return;
  const csrf = $('meta[name="spritelab-csrf"]')?.content ||
    decodeURIComponent(document.cookie.split("; ").find((value) => value.startsWith("spritelab_csrf="))?.split("=")[1] || "");
  const sourceSelect = $("#harvest-source");
  const summary = $("#harvest-summary");
  const runsNode = $("#harvest-runs");
  const probesNode = $("#harvest-probes");
  const probeSummary = $("#harvest-probe-summary");
  const probeAvailability = $("#harvest-probe-availability");
  const prefillSummary = $("#harvest-prefill-summary");
  const prefillPreset = $("#probe-prefill-preset");
  const detail = $("#harvest-detail");
  const inFlight = new Set();
  const pendingKeys = new Map();
  const idempotencyStoragePrefix = "spritelab.harvest.pending-idempotency.v1:";
  let state = {inventory: {runs: [], probe_runs: [], legacy_runs: []}, catalog: {sources: []}, source_prefill_presets: []};
  let activeJobs = [];
  try { state = JSON.parse($("#harvest-initial-state")?.textContent || "{}"); } catch (_error) { /* safe defaults */ }

  const newIdempotency = (prefix) => `${prefix}-${crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`}`;
  const idempotencyScope = (scope) => {
    if (!/^[A-Za-z0-9._:-]{1,100}$/.test(scope)) throw new Error("Harvest action identity is invalid.");
    return `${idempotencyStoragePrefix}${scope}`;
  };
  function pendingIdempotency(scope, prefix) {
    const storageKey = idempotencyScope(scope);
    let value = pendingKeys.get(storageKey);
    if (!value) {
      try { value = window.sessionStorage.getItem(storageKey); } catch (_error) { /* in-memory fallback */ }
    }
    if (!value || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(value)) {
      value = newIdempotency(prefix);
      pendingKeys.set(storageKey, value);
      try { window.sessionStorage.setItem(storageKey, value); } catch (_error) { /* in-memory fallback */ }
    }
    return value;
  }
  function clearIdempotency(scope) {
    const storageKey = idempotencyScope(scope);
    pendingKeys.delete(storageKey);
    try { window.sessionStorage.removeItem(storageKey); } catch (_error) { /* in-memory fallback */ }
  }
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
    if (!response.ok) {
      const error = new Error(payload.message || "Harvest request failed.");
      error.definitive = response.status >= 400 && response.status < 500 && ![408, 425, 429].includes(response.status);
      throw error;
    }
    return payload;
  }
  async function idempotentRequest(scope, prefix, url, payload) {
    const idempotencyKey = pendingIdempotency(scope, prefix);
    try {
      const result = await request(url, {
        method: "POST",
        body: JSON.stringify({...payload, idempotency_key: idempotencyKey}),
      });
      clearIdempotency(scope);
      return result;
    } catch (error) {
      if (error.definitive) clearIdempotency(scope);
      throw error;
    }
  }
  async function once(key, control, action) {
    if (inFlight.has(key)) return;
    inFlight.add(key); setBusy(control, true);
    try { await action(); } finally { inFlight.delete(key); setBusy(control, false); }
  }
  const prefillFields = {
    source_id: "#probe-source-id", title: "#probe-title", creator: "#probe-creator",
    source_page: "#probe-source-page", license_id: "#probe-license-id",
    license_evidence_url: "#probe-license-url", terms_evidence_url: "#probe-terms-url",
    direct_download_url: "#probe-direct-url", attribution_text: "#probe-attribution",
    taxonomy_hints: "#probe-taxonomy",
  };
  const reviewLabels = {
    creator: "creator", license_id: "license", license_evidence_url: "license evidence",
    terms_evidence_url: "site terms", direct_download_url: "exact download link",
    attribution_text: "attribution",
  };
  function applySourcePrefill(prefill) {
    for (const [field, selector] of Object.entries(prefillFields)) {
      const control = $(selector);
      if (!control) continue;
      const raw = prefill[field];
      control.value = Array.isArray(raw) ? raw.join(", ") : String(raw || "");
    }
    const licenseEvidence = $("#probe-license-url");
    delete licenseEvidence.dataset.smartLicense;
    if (prefill.license_id === "cc0-1.0" && prefill.license_evidence_url === "https://creativecommons.org/publicdomain/zero/1.0/") {
      licenseEvidence.dataset.smartLicense = "cc0-1.0";
    }
    const review = (prefill.review_fields || []).map((field) => reviewLabels[field] || field).join(", ");
    prefillSummary.textContent = `${prefill.preset_label} draft ready. ${prefill.guidance} Review: ${review || "all fields"}.`;
  }
  function renderPrefillPresets() {
    if (!prefillPreset) return;
    const existing = new Set([...prefillPreset.options].map((option) => option.value));
    for (const preset of state.source_prefill_presets || []) {
      if (!preset?.preset_id || existing.has(preset.preset_id)) continue;
      const option = element("option", preset.label);
      option.value = preset.preset_id;
      option.title = preset.description || "";
      prefillPreset.append(option);
      existing.add(preset.preset_id);
    }
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
  function authorization() {
    return {
      explicit_action: true,
      authorize_zero_cost: $("#harvest-zero-cost").checked,
      authorize_permissive_license: $("#harvest-license").checked,
      authorize_existing_inventory_reviewed: $("#harvest-inventory-reviewed").checked,
      reuse_evidence: reuseEvidence(),
    };
  }
  function probePayload() {
    const taxonomy = $("#probe-taxonomy").value.split(",").map((value) => value.trim()).filter(Boolean);
    return {
      source_id: $("#probe-source-id").value.trim(), title: $("#probe-title").value.trim(),
      creator: $("#probe-creator").value.trim(), source_page: $("#probe-source-page").value.trim(),
      license_id: $("#probe-license-id").value,
      license_evidence_url: $("#probe-license-url").value.trim(),
      terms_evidence_url: $("#probe-terms-url").value.trim() || null,
      direct_download_url: $("#probe-direct-url").value.trim(),
      attribution_text: $("#probe-attribution").value.trim(), taxonomy_hints: taxonomy,
      inventory_identity: state.inventory.inventory_identity,
      backend_capability_evidence_identity: state.catalog.backend_capability_evidence?.evidence_identity || "",
      explicit_action: true,
      authorize_network: $("#probe-network").checked,
      authorize_hash_probe: $("#probe-hash").checked,
      authorize_zero_cost: $("#probe-zero-cost").checked,
      authorize_permissive_license: $("#probe-license-auth").checked,
    };
  }
  function renderSources(catalog, availabilityChecked = true) {
    state.catalog = catalog;
    sourceSelect.replaceChildren();
    for (const source of catalog.sources || []) {
      const option = element("option", `${source.title} · ${source.license.identifier}`);
      option.value = source.source_id; sourceSelect.append(option);
    }
    sourceSelect.disabled = !sourceSelect.options.length || !catalog.backend_configured;
    const probeButton = $("#harvest-probe-start");
    const probeAvailable = availabilityChecked && Boolean(catalog.backend_capability_evidence?.evidence_identity);
    if (probeButton) {
      probeButton.disabled = !probeAvailable;
      probeButton.title = probeAvailable
        ? ""
        : availabilityChecked
          ? "Source probing requires current independent backend capability evidence."
          : "Checking whether the certified source-probe backend is available.";
    }
    if (probeAvailability) {
      probeAvailability.hidden = probeAvailable;
      probeAvailability.textContent = probeAvailable
        ? ""
        : availabilityChecked
          ? "Source probing is unavailable because current independent backend capability evidence is missing or invalid. Configure or renew the repository Harvest certificate, then reload this page."
          : "Checking whether the certified source-probe backend is available…";
    }
    $("#harvest-limits").textContent = JSON.stringify(catalog.limits || {}, null, 2);
    renderSourceEvidence();
  }
  function renderSourceEvidence() {
    const source = (state.catalog.sources || []).find((item) => item.source_id === sourceSelect.value);
    const node = $("#harvest-source-evidence"); node.replaceChildren();
    if (!source) { node.append(element("p", "No certified catalog source is configured.")); return; }
    const terms = source.evidence_binding.automation_terms;
    const termsScope = terms.limited_evidence
      ? "Limited evidence: no prohibition observed; not affirmative permission"
      : "Explicit declaration retained";
    for (const [label, value] of [
      ["Creator", source.creator], ["License", source.license.identifier],
      ["Attribution", source.license.attribution_text], ["Expected response SHA-256", source.expected_response_sha256],
      ["Automation terms decision", terms.decision], ["Automation terms mode", terms.mode],
      ["Automation terms scope", termsScope], ["Automation terms expire", terms.expires_at],
      ["Evidence expires", source.evidence_binding.expires_at], ["Catalog identity", source.catalog_identity],
    ]) { const row = element("p"); row.append(element("strong", `${label}: `), element("span", value)); node.append(row); }
  }
  function renderInventory(inventory, jobs = null) {
    state.inventory = inventory;
    activeJobs = jobs || inventory.runs || [];
    $("#harvest-assessed").value = String(inventory.known_usable_items || 0);
    summary.textContent = `${inventory.run_count} acquisitions, ${inventory.probe_run_count || 0} source probes, ${inventory.legacy_run_count} legacy read-only; ${inventory.known_usable_items} known usable; ${inventory.unsafe_entries} unsafe ignored.`;
    const legacy = $("#harvest-legacy"); legacy.replaceChildren();
    for (const run of inventory.legacy_runs || []) {
      const card = element("article", undefined, "harvest-evidence");
      card.append(element("strong", run.legacy_id), element("p", `${run.source_records} sources · ${run.candidate_records} candidates · ${run.imported_records} imported`), element("code", run.legacy_identity));
      legacy.append(card);
    }
    renderRuns(jobs || inventory.runs || []);
    renderProbes(inventory.probe_runs || []);
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
      operationalText.textContent = JSON.stringify({taxonomy_counts: run.taxonomy_counts || {}, limits: run.limits || {}, dataset_import: run.dataset_import || {}, logs: run.events || []}, null, 2);
      operational.append(operationalText);
      const controls = element("div", undefined, "harvest-actions");
      const evidence = action("Evidence", async () => { detail.textContent = JSON.stringify(await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/evidence`), null, 2); }); evidence.dataset.run = run.run_id; controls.append(evidence);
      if (["QUEUED", "RUNNING", "CANCELLING"].includes(run.status)) { const cancel = action("Cancel", async () => { await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/cancel`, {method: "POST", body: JSON.stringify({explicit_action: true})}); await refresh(); }); cancel.dataset.run = run.run_id; controls.append(cancel); }
      if (["RUNNING", "CANCELLING"].includes(run.dataset_import?.status)) { const cancelImport = action("Cancel Dataset import", async () => { await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/cancel`, {method: "POST", body: JSON.stringify({explicit_action: true})}); await refresh(); }); cancelImport.dataset.run = run.run_id; controls.append(cancelImport); }
      if (["FAILED", "CANCELLED", "INTERRUPTED"].includes(run.status)) { const retry = action("Retry", async () => { await idempotentRequest(`retry:${run.run_id}`, "retry", `/harvest/api/jobs/${encodeURIComponent(run.run_id)}/retry`, authorization()); await refresh(); }); retry.dataset.run = run.run_id; controls.append(retry); }
      if (run.handoff_ready) { const handoff = action("Dataset handoff", async () => { const value = await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}/handoff`); detail.textContent = JSON.stringify(value, null, 2); if (value.dataset_import_available && !value.dataset_import?.completed) { const importButton = action("Import into Dataset", async () => { const importRequest = idempotentRequest(`dataset-import:${run.run_id}`, "dataset-import", `/harvest/api/jobs/${encodeURIComponent(run.run_id)}/import`, {explicit_action: true}); window.setTimeout(() => { refresh().catch((error) => { summary.textContent = error.message; }); }, 100); const receipt = await importRequest; detail.textContent = JSON.stringify(receipt, null, 2); await refresh(); }); importButton.dataset.run = run.run_id; controls.append(importButton); } }); handoff.dataset.run = run.run_id; controls.append(handoff); }
      card.append(heading, metrics, provenance, progress, operational, controls); runsNode.append(card);
    }
  }
  function renderProbes(probes) {
    if (!probesNode) return;
    probesNode.replaceChildren();
    for (const probe of probes) {
      const card = element("article", undefined, "harvest-run harvest-probe-card");
      const heading = element("div");
      heading.append(element("strong", probe.title || probe.source_id), element("code", probe.probe_id), element("span", probe.status, "status-pill"));
      const progress = document.createElement("progress");
      progress.max = Math.max(probe.total || 1, 1); progress.value = Math.min(probe.current || 0, progress.max);
      progress.setAttribute("aria-label", `${probe.stage || probe.status} probe progress`);
      const status = element("p", probe.message || `${probe.stage || probe.status}`);
      const controls = element("div", undefined, "harvest-actions");
      let reviewedEvidence = null;
      const evidence = action("Probe evidence", async () => {
        reviewedEvidence = await request(`/harvest/api/probes/${encodeURIComponent(probe.probe_id)}/evidence`);
        detail.textContent = JSON.stringify(reviewedEvidence, null, 2);
      }); evidence.dataset.run = probe.probe_id; controls.append(evidence);
      if (["QUEUED", "RUNNING", "CANCELLING"].includes(probe.status)) {
        const cancel = action("Cancel probe", async () => {
          await request(`/harvest/api/probes/${encodeURIComponent(probe.probe_id)}/cancel`, {method: "POST", body: JSON.stringify({explicit_action: true})}); await refresh();
        }); cancel.dataset.run = probe.probe_id; controls.append(cancel);
      }
      if (["FAILED", "CANCELLED", "INTERRUPTED"].includes(probe.status)) {
        const retry = action("Retry with form", async () => {
          const form = $("#harvest-probe-form"); if (!form.reportValidity()) return;
          await idempotentRequest(`probe-retry:${probe.probe_id}`, "probe-retry", `/harvest/api/probes/${encodeURIComponent(probe.probe_id)}/retry`, probePayload()); await refresh();
        }); retry.dataset.run = probe.probe_id; controls.append(retry);
      }
      if (probe.status === "READY") {
        const promotion = element("label", undefined, "harvest-check harvest-promotion-check");
        const checkbox = document.createElement("input"); checkbox.type = "checkbox";
        promotion.append(checkbox, document.createTextNode(" I reviewed the displayed retained price evidence and separately authorize trusted-catalog promotion."));
        const promote = action("Promote trusted source", async () => {
          const evidenceReceipt = reviewedEvidence?.receipt;
          if (!checkbox.checked || !evidenceReceipt?.verification_identity || !evidenceReceipt?.source_pack_evidence_sha256) { probeSummary.textContent = "Load and review Probe evidence, then check the separate promotion authorization."; return; }
          const receipt = await request(`/harvest/api/probes/${encodeURIComponent(probe.probe_id)}/promote`, {method: "POST", body: JSON.stringify({explicit_action: true, authorize_catalog_promotion: true, authorize_zero_cost_evidence_review: true, reviewed_verification_identity: evidenceReceipt.verification_identity, reviewed_source_pack_evidence_sha256: evidenceReceipt.source_pack_evidence_sha256})});
          detail.textContent = JSON.stringify(receipt, null, 2);
          const [catalog] = await Promise.all([request("/harvest/api/sources"), refresh()]); renderSources(catalog);
        }); promote.dataset.run = probe.probe_id;
        controls.append(promotion, promote);
      }
      card.append(heading, status, progress, controls); probesNode.append(card);
    }
    if (!probes.length) probesNode.append(element("p", "No durable source probes yet."));
  }
  async function refresh() {
    const inventory = await request("/harvest/api/inventory");
    const jobs = await Promise.all((inventory.runs || []).slice(0, 100).map(async (run) => {
      try { return await request(`/harvest/api/jobs/${encodeURIComponent(run.run_id)}`); } catch (_error) { return run; }
    }));
    const probes = await Promise.all((inventory.probe_runs || []).slice(0, 100).map(async (probe) => {
      try { return await request(`/harvest/api/probes/${encodeURIComponent(probe.probe_id)}`); } catch (_error) { return probe; }
    }));
    inventory.probe_runs = probes;
    renderInventory(inventory, jobs);
    const active = [...jobs, ...(inventory.probe_runs || [])].some((run) => ["QUEUED", "RUNNING", "CANCELLING"].includes(run.status) || ["RUNNING", "CANCELLING"].includes(run.dataset_import?.status));
    $("#harvest-poll-state").textContent = active ? "Polling active work" : "No active work";
  }
  $("#harvest-start-form").addEventListener("submit", (event) => {
    event.preventDefault(); const button = $("#harvest-start");
    once("start", button, async () => {
      const payload = {source_id: sourceSelect.value, ...authorization()};
      await idempotentRequest(`start:${sourceSelect.value}`, "start", "/harvest/api/jobs", payload); await refresh();
    }).catch((error) => { summary.textContent = error.message; });
  });
  $("#harvest-refresh").addEventListener("click", (event) => once("refresh", event.currentTarget, refresh).catch((error) => { summary.textContent = error.message; }));
  $("#harvest-smart-prefill")?.addEventListener("click", (event) => {
    const sourcePage = $("#probe-source-page");
    if (!sourcePage.reportValidity()) return;
    once("source-prefill", event.currentTarget, async () => {
      const value = await request("/harvest/api/source-prefill", {
        method: "POST",
        body: JSON.stringify({source_page: sourcePage.value.trim(), preset: prefillPreset.value}),
      });
      applySourcePrefill(value.prefill);
    }).catch((error) => { prefillSummary.textContent = error.message; });
  });
  $("#probe-license-id")?.addEventListener("change", (event) => {
    const evidence = $("#probe-license-url");
    const cc0 = "https://creativecommons.org/publicdomain/zero/1.0/";
    if (event.currentTarget.value === "cc0-1.0" && !evidence.value.trim()) {
      evidence.value = cc0;
      evidence.dataset.smartLicense = "cc0-1.0";
    } else if (event.currentTarget.value !== "cc0-1.0" && evidence.dataset.smartLicense === "cc0-1.0" && evidence.value === cc0) {
      evidence.value = "";
      delete evidence.dataset.smartLicense;
    }
  });
  $("#harvest-probe-form")?.addEventListener("submit", (event) => {
    event.preventDefault(); const form = event.currentTarget; const button = $("#harvest-probe-start");
    if (!form.reportValidity()) return;
    once("probe-start", button, async () => {
      const sourceId = $("#probe-source-id").value.trim();
      const value = await idempotentRequest(`probe-start:${sourceId}`, "probe-start", "/harvest/api/probes", probePayload());
      probeSummary.textContent = `Probe ${value.probe.probe_id} recorded. Progress is durable and safe to reload.`; await refresh();
    }).catch((error) => { probeSummary.textContent = error.message; });
  });
  sourceSelect.addEventListener("change", renderSourceEvidence);
  renderPrefillPresets();
  renderSources(state.catalog || {sources: []}, false); renderInventory(state.inventory || {runs: [], legacy_runs: []});
  request("/harvest/api/sources").then(renderSources).catch((error) => {
    if (probeAvailability) {
      probeAvailability.hidden = false;
      probeAvailability.textContent = `Source probing remains unavailable because backend capability evidence could not be verified: ${error.message}`;
    }
  });
  refresh().catch((error) => { summary.textContent = error.message; });
  window.setInterval(() => { if ([...activeJobs, ...(state.inventory.probe_runs || [])].some((run) => ["QUEUED", "RUNNING", "CANCELLING"].includes(run.status) || ["RUNNING", "CANCELLING"].includes(run.dataset_import?.status))) refresh().catch((error) => { summary.textContent = error.message; }); }, 1500);
})();
