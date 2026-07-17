(() => {
  const state = { items: [], index: 0, taxonomy: [], queue: null, jobTimer: null };
  const byId = (id) => document.getElementById(id);
  const text = (id, value) => { const node = byId(id); if (node) node.textContent = value; };
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";

  async function json(url, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    const headers = { ...(options.headers || {}) };
    if (!['GET', 'HEAD'].includes(method)) headers['X-CSRF-Token'] = csrf;
    const response = await fetch(url, { ...options, headers });
    const body = await response.json();
    if (!response.ok && body.error_code === "csrf_validation_failed") {
      text("labeling-action-status", "Sprite Lab restarted. Reloading this page with a fresh action tokenâ€¦");
      text("review-status", "Sprite Lab restarted. Reloading this page with a fresh action tokenâ€¦");
      window.setTimeout(() => window.location.reload(), 250);
    }
    if (!response.ok) throw new Error(body.message || body.detail || "The action could not be completed.");
    return body;
  }

  function current() { return state.items[state.index]; }

  function setActionsBusy(busy) {
    document.querySelectorAll("[data-labeling-action]").forEach(button => {
      button.disabled = busy && button.dataset.labelingAction === "run_automatic_labeling";
      button.setAttribute("aria-busy", String(button.disabled));
    });
  }

  function renderJob(job) {
    const panel = byId("labeling-job");
    if (!panel || !job) return;
    panel.hidden = false;
    panel.dataset.status = job.status;
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    const progressBar = byId("labeling-job-progress");
    progressBar.value = progress;
    progressBar.setAttribute("value", String(progress));
    progressBar.setAttribute("aria-valuenow", String(progress));
    progressBar.textContent = `${progress}%`;
    text("labeling-job-percent", `${progress}%`);
    text("labeling-job-stage", job.stage || "Working");
    text("labeling-job-message", job.message || "Hierarchical labeling is running.");
    const logs = byId("labeling-job-logs");
    logs.replaceChildren();
    for (const entry of job.logs || []) {
      const item = document.createElement("li");
      const time = document.createElement("time");
      time.dateTime = entry.timestamp;
      time.textContent = new Date(entry.timestamp).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const message = document.createElement("span");
      message.textContent = entry.message;
      item.append(time, message);
      logs.append(item);
    }
    const terminal = ["complete", "failed"].includes(job.status);
    setActionsBusy(!terminal);
    if (terminal) {
      if (state.jobTimer) window.clearTimeout(state.jobTimer);
      state.jobTimer = null;
      text("labeling-action-status", job.message);
      if (job.status === "complete") initializeReview();
    }
  }

  async function monitorJob(jobId) {
    try {
      const job = await json(`/labeling/api/jobs/${encodeURIComponent(jobId)}`);
      renderJob(job);
      if (!["complete", "failed"].includes(job.status)) {
        state.jobTimer = window.setTimeout(() => monitorJob(jobId), 500);
      }
    } catch (error) {
      setActionsBusy(false);
      text("labeling-action-status", error.message);
    }
  }

  function render() {
    const item = current();
    byId("review-empty").hidden = Boolean(item);
    byId("review-workspace").hidden = !item;
    text("review-position", item ? `${state.index + 1} of ${state.items.length} low-certainty · ${state.queue?.auto_prefilled || 0} prefilled` : "No items");
    if (!item) return;
    byId("review-image").src = item.image_url;
    text("suggested-path", item.suggested_path.length ? item.suggested_path.join(" / ") : "No safe suggestion — abstention is available");
    text("semantic-confidence", Number.isFinite(Number(item.confidence)) ? `Low certainty · ${Math.round(Number(item.confidence) * 100)}% confidence` : "Low certainty · provider abstained");
    text("visual-description", item.visual_description || "No strict visual description is available.");
    text("metadata-evidence", JSON.stringify(item.metadata_evidence, null, 2));
    text("conflict-indicators", item.conflicts.length ? `Conflict: ${item.conflicts.join("; ")}` : "");
    const select = byId("taxonomy-node"); select.replaceChildren();
    for (const node of state.taxonomy) { const option = document.createElement("option"); option.value = node.node_id; option.textContent = `${"· ".repeat(node.depth || 0)}${node.display_name}`; select.append(option); }
    if (item.suggested_path.length) select.value = item.suggested_path[item.suggested_path.length - 1];
    const neighbors = byId("retrieved-neighbors"); neighbors.replaceChildren();
    if (!item.retrieved_verified_neighbors.length) { const li = document.createElement("li"); li.textContent = "No verified neighbors available."; neighbors.append(li); }
    const strip = byId("render-strip"); strip.replaceChildren();
    for (const view of item.render_views) { const button = document.createElement("button"); button.className = "button secondary"; button.type = "button"; button.textContent = view.render_type.replaceAll("_", " "); button.addEventListener("click", () => { byId("review-image").src = view.url; }); strip.append(button); }
  }

  async function submitReview(action) {
    const item = current(); if (!item) return;
    const reviewer = byId("reviewer-identity").value.trim();
    if (!reviewer) { text("review-status", "Enter a reviewer identity before saving."); byId("reviewer-identity").focus(); return; }
    localStorage.setItem("spritelab-reviewer", reviewer);
    const selected = byId("taxonomy-node").value || null;
    let selectedNode = selected;
    if (action === "accept_suggested_path") selectedNode = item.suggested_path.at(-1) || selected;
    if (action === "choose_parent" && item.suggested_path.length > 1) selectedNode = item.suggested_path.at(-2);
    const abstentions = [];
    if (byId("flag-render").checked) abstentions.push("render_problem");
    if (byId("flag-pack").checked) abstentions.push("pack_context_issue");
    text("review-status", "Saving verified review event…");
    try {
      await json(`/labeling/api/review/${encodeURIComponent(item.record_identity)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, selected_node: selectedNode, reviewer_identity: reviewer, partition: "reference", explicit_abstentions: abstentions, review_notes: byId("review-caption").value.trim() || null, exclude_semantic_supervision: byId("exclude-semantic").checked, submission_token: crypto.randomUUID() }) });
      text("review-status", "Review saved to the append-only truth log."); state.items.splice(state.index, 1); if (state.index >= state.items.length) state.index = 0; render();
    } catch (error) { text("review-status", error.message); }
  }

  async function initializeReview() {
    try {
      const [queue, taxonomy] = await Promise.all([json("/labeling/api/queue"), json("/labeling/api/taxonomy")]);
      state.items = queue.items; state.taxonomy = taxonomy.nodes; state.queue = queue;
      if (!state.items.length) text("review-empty", queue.message);
      render();
    } catch (error) { byId("review-empty").hidden = false; text("review-empty", error.message); }
  }

  async function initialize() {
    await initializeReview();
    try {
      const currentJob = await json("/labeling/api/jobs/current");
      if (currentJob.job) {
        renderJob(currentJob.job);
        if (!["complete", "failed"].includes(currentJob.job.status)) monitorJob(currentJob.job.job_id);
      }
    } catch (error) {
      text("labeling-action-status", error.message);
    }
  }

  document.querySelectorAll("[data-review-action]").forEach((button) => button.addEventListener("click", () => submitReview(button.dataset.reviewAction)));
  document.querySelectorAll("[data-labeling-action]").forEach((button) => button.addEventListener("click", async () => {
    const action = button.dataset.labelingAction;
    if (action === "open_semantic_review") {
      byId("semantic-review").scrollIntoView({ behavior: "smooth" });
      return;
    }
    setActionsBusy(true);
    text("labeling-action-status", "Starting hierarchical labeling in the background…");
    try {
      const result = await json(`/labeling/api/actions/${action}`, { method: "POST" });
      renderJob(result);
      monitorJob(result.job_id);
    } catch (error) {
      setActionsBusy(false);
      text("labeling-action-status", error.message);
    }
  }));
  document.addEventListener("keydown", (event) => { if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return; const action = ({ a: "accept_suggested_path", p: "choose_parent", x: "choose_alternative", u: "abstain", g: "flag_taxonomy_gap" })[event.key.toLowerCase()]; if (action) { event.preventDefault(); submitReview(action); } });
  byId("reviewer-identity").value = localStorage.getItem("spritelab-reviewer") || "";
  initialize();
})();
