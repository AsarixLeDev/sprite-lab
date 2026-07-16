(() => {
  const state = { items: [], index: 0, taxonomy: [] };
  const byId = (id) => document.getElementById(id);
  const text = (id, value) => { const node = byId(id); if (node) node.textContent = value; };

  async function json(url, options) {
    const response = await fetch(url, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || body.detail || "The action could not be completed.");
    return body;
  }

  function current() { return state.items[state.index]; }

  function render() {
    const item = current();
    byId("review-empty").hidden = Boolean(item);
    byId("review-workspace").hidden = !item;
    text("review-position", item ? `${state.index + 1} of ${state.items.length}` : "No items");
    if (!item) return;
    byId("review-image").src = item.image_url;
    text("suggested-path", item.suggested_path.length ? item.suggested_path.join(" / ") : "No safe suggestion — abstention is available");
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

  async function initialize() {
    try {
      const [queue, taxonomy] = await Promise.all([json("/labeling/api/queue"), json("/labeling/api/taxonomy")]);
      state.items = queue.items; state.taxonomy = taxonomy.nodes; render();
    } catch (error) { byId("review-empty").hidden = false; text("review-empty", error.message); }
  }

  document.querySelectorAll("[data-review-action]").forEach((button) => button.addEventListener("click", () => submitReview(button.dataset.reviewAction)));
  document.querySelectorAll("[data-labeling-action]").forEach((button) => button.addEventListener("click", async () => { try { const result = await json(`/labeling/api/actions/${button.dataset.labelingAction}`, { method: "POST" }); text("labeling-action-status", result.next_step); if (button.dataset.labelingAction === "open_semantic_review") byId("semantic-review").scrollIntoView({ behavior: "smooth" }); } catch (error) { text("labeling-action-status", error.message); } }));
  document.addEventListener("keydown", (event) => { if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return; const action = ({ a: "accept_suggested_path", p: "choose_parent", x: "choose_alternative", u: "abstain", g: "flag_taxonomy_gap" })[event.key.toLowerCase()]; if (action) { event.preventDefault(); submitReview(action); } });
  byId("reviewer-identity").value = localStorage.getItem("spritelab-reviewer") || "";
  initialize();
})();
