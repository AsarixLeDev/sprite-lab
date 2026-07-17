(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let state = {};
  try { state = JSON.parse($("evaluation-initial-state")?.textContent || "{}"); } catch (_error) { state = {}; }
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content ||
    decodeURIComponent(document.cookie.split("; ").find((value) => value.startsWith("spritelab_csrf="))?.split("=")[1] || "");
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (character) =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[character]);
  const toast = (message) => {
    const node = $("toast");
    if (!node) return;
    node.textContent = message;
    node.classList.add("show");
    window.setTimeout(() => node.classList.remove("show"), 3200);
  };
  const busy = (control, value) => {
    if (!control) return;
    control.disabled = value;
    control.setAttribute("aria-busy", String(value));
  };
  const jsonRequest = async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: {"Content-Type":"application/json", "X-CSRF-Token":csrf, ...(options.headers || {})},
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      await response.text();
      throw new Error("Sprite Lab received an unexpected response. Reload the page and try again.");
    }
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || "The evaluation request could not be completed.");
    return body;
  };
  const productionCheckpoints = state.checkpoints?.eligible || [];
  let exploratoryCheckpoints = state.exploratory_checkpoints?.eligible || [];
  const fillCheckpoints = () => {
    const evaluationSelect = $("eval-checkpoint");
    if (evaluationSelect) {
      const selected = evaluationSelect.value;
      evaluationSelect.replaceChildren();
      if (!productionCheckpoints.length) evaluationSelect.add(new Option("No eligible production checkpoint", ""));
      productionCheckpoints.forEach((item) => evaluationSelect.add(new Option(
        `${item.friendly_run_name} · step ${item.checkpoint_step ?? "—"} · ${item.weights.toUpperCase()}`,
        item.checkpoint_id,
      )));
      if ([...evaluationSelect.options].some((option) => option.value === selected)) evaluationSelect.value = selected;
    }
    const playgroundSelect = $("play-checkpoint");
    if (playgroundSelect) {
      const selected = playgroundSelect.value || state.playground_defaults?.checkpoint_id || "";
      playgroundSelect.replaceChildren();
      const choices = [
        ...productionCheckpoints.map((item) => ({...item, catalogLabel:"production"})),
        ...exploratoryCheckpoints.map((item) => ({...item, catalogLabel:"EXPLORATORY ONLY"})),
      ];
      if (!choices.length) playgroundSelect.add(new Option("No Playground checkpoint", ""));
      choices.forEach((item) => playgroundSelect.add(new Option(
        `[${item.catalogLabel}] ${item.friendly_run_name} · step ${item.checkpoint_step ?? "—"} · ${item.weights.toUpperCase()}`,
        item.checkpoint_id,
      )));
      if ([...playgroundSelect.options].some((option) => option.value === selected)) playgroundSelect.value = selected;
    }
  };
  const renderStages = (stages = []) => {
    const timeline = $("stage-timeline");
    if (!timeline) return;
    timeline.innerHTML = stages.map((stage) =>
      `<li class="${esc(String(stage.status).toLowerCase())}"><strong>${esc(stage.title)}</strong><span>${esc(stage.message || stage.status)}</span></li>`
    ).join("");
    $("progress-count").textContent = `${stages.filter((stage) => stage.status === "COMPLETE").length} / ${stages.length || 10} complete`;
  };
  const chartMarkup = (chart) => {
    if (chart.status === "NO_DATA") {
      return `<article class="chart"><h3>${esc(chart.title)}</h3><div class="no-data">${esc(chart.no_data_message)}</div></article>`;
    }
    const series = chart.series || [];
    const max = Math.max(...series.map((item) => Number(item.value) || 0), 1);
    const summary = `${chart.title}: ${series.length} measured value${series.length === 1 ? "" : "s"}.`;
    return `<article class="chart"><h3>${esc(chart.title)}</h3><p class="chart-summary">${esc(summary)}</p>` +
      `<div class="bars" aria-hidden="true">${series.map((item) => `<i class="bar" style="height:${Math.max(2, Number(item.value) / max * 100)}%" data-label="${esc(item.label)}: ${esc(item.value)}"></i>`).join("")}</div>` +
      `<details class="chart-table"><summary>Text data for ${esc(chart.title)}</summary><div class="table-wrap"><table><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>${series.map((item) => `<tr><th scope="row">${esc(item.label)}</th><td>${esc(item.value)}</td></tr>`).join("")}</tbody></table></div></details></article>`;
  };
  const renderDashboard = (dashboard = {}) => {
    if (dashboard.stale) toast(dashboard.message || "Evaluation artifacts are stale and not comparable.");
    $("metric-cards").innerHTML = (dashboard.metric_cards || []).map((card) =>
      `<article class="metric-card ${card.status === "NO_DATA" ? "no-data" : ""}"><span>${esc(card.title)}</span><strong>${card.status === "NO_DATA" ? "No data" : esc(`${card.value}${card.unit || ""}`)}</strong></article>`
    ).join("");
    $("charts").innerHTML = (dashboard.charts || []).map(chartMarkup).join("");
    $("category-results").innerHTML = (dashboard.per_category || []).map((row) =>
      `<tr><td>${esc(row.name)}</td><td>${esc(row.sample_count)}</td><td>${row.structural_validity_rate == null ? "—" : `${(row.structural_validity_rate * 100).toFixed(1)}%`}</td><td>${row.conditional_adherence == null ? "—" : `${(row.conditional_adherence * 100).toFixed(1)}%`}</td></tr>`
    ).join("");
    renderGallery(dashboard.gallery || [], "sample-gallery");
  };
  const renderGallery = (samples, target) => {
    const node = $(target);
    if (!node) return;
    if (!samples.length) {
      node.innerHTML = `<div class="no-data ${target === "play-results" ? "dark-no-data" : ""}">No generated sample data is available.</div>`;
      return;
    }
    node.innerHTML = samples.map((sample) => {
      const label = `Open sample ${sample.prompt || "Untitled sample"}, seed ${sample.seed ?? "unknown"}`;
      return `<button type="button" class="sample-card" aria-label="${esc(label)}" data-sample="${encodeURIComponent(JSON.stringify(sample))}"><span class="sample-art"><code>${esc(sample.output_hash?.slice(0, 12) || sample.image_reference || "sample")}</code></span><span class="sample-meta"><strong>${esc(sample.prompt || "Untitled sample")}</strong><span>Seed ${esc(sample.seed)} · ${esc(String(sample.weights || "").toUpperCase())}</span></span></button>`;
    }).join("");
    node.querySelectorAll(".sample-card").forEach((card) => card.addEventListener("click", () => {
      const sample = JSON.parse(decodeURIComponent(card.dataset.sample));
      $("sample-preview").innerHTML = `<h2>${esc(sample.prompt || "Sample")}</h2><pre>${esc(JSON.stringify(sample, null, 2))}</pre>`;
      $("sample-dialog").showModal();
    }));
  };
  const evalPayload = (dryRun) => ({
    checkpoint_id: $("eval-checkpoint").value,
    weights: document.querySelector('input[name="eval-weights"]:checked').value,
    dry_run: dryRun,
    explicit_action: !dryRun,
  });
  const renderPlaygroundRun = (run) => {
    if (!run) return;
    renderGallery(run.results || [], "play-results");
    const progress = run.progress || {};
    const integrity = run.integrity_reasons?.length ? ` ${run.integrity_reasons.join(" ")}` : "";
    $("play-run-status").textContent = `${run.status || "NOT_STARTED"} | ${progress.current || 0} / ${progress.total ?? "?"} images.${integrity}`;
    const request = run.request || {};
    if (request.prompt) $("play-prompt").value = request.prompt;
    if (request.checkpoint_id) $("play-checkpoint").value = request.checkpoint_id;
    if (request.weights) $("play-weights").value = request.weights;
    for (const [id, value] of [["play-seed", request.seed], ["play-steps", request.sampling_steps], ["play-guidance", request.guidance], ["play-count", request.image_count]]) {
      if (value != null) $(id).value = value;
    }
  };
  const runEvaluation = async (dryRun, control) => {
    busy(control, true);
    try {
      const result = await jsonRequest("/evaluation/api/run", {method:"POST", body:JSON.stringify(evalPayload(dryRun))});
      const data = result.data?.product_result?.data || result.data || {};
      renderStages(data.stages);
      if (data.dashboard) renderDashboard(data.dashboard);
      const memory = data.memorization;
      if (memory?.review_required_count) {
        $("review-callout").hidden = false;
        $("review-message").textContent = memory.review_message;
      }
      toast(result.message);
    } catch (error) { toast(error.message); } finally { busy(control, false); }
  };
  const generate = async (control) => {
    busy(control, true);
    const payload = {prompt:$("play-prompt").value, checkpoint_id:$("play-checkpoint").value,
      weights:$("play-weights").value, seed:Number($("play-seed").value), sampling_steps:Number($("play-steps").value),
      guidance:Number($("play-guidance").value), image_count:Number($("play-count").value), explicit_action:true,
      confirm_billable:$("billable-confirm").checked};
    try {
      const result = await jsonRequest("/evaluation/api/playground/generate", {method:"POST", body:JSON.stringify(payload)});
      renderPlaygroundRun(result); toast("Exploratory generation complete and durably recorded.");
    } catch (error) { toast(error.message); } finally { busy(control, false); }
  };
  const refreshPresets = async () => {
    try {
      const result = await jsonRequest("/evaluation/api/playground/presets");
      const select = $("saved-presets"); select.innerHTML = '<option value="">Saved presets</option>';
      result.presets.forEach((preset) => select.add(new Option(preset.name, preset.name)));
    } catch (_error) { /* Empty preset state is valid. */ }
  };
  const smokePublications = state.smoke_publications?.eligible || [];
  const smokePlans = [...(state.smoke_plans?.eligible || [])];
  const fillSmokePublications = () => {
    const select = $("smoke-conditioned-job");
    if (!select) return;
    select.replaceChildren();
    if (!smokePublications.length) select.add(new Option("No eligible pre-activation publication", ""));
    smokePublications.forEach((item) => select.add(new Option(item.label, item.conditioned_job_id)));
  };
  const smokeIdentityPayload = () => ({
    conditioned_job_id: $("smoke-conditioned-job").value,
    smoke_id:$("smoke-id").value,
    plan_identity:$("smoke-plan-identity").value,
  });
  const freshSmokeNonce = () => {
    if (window.crypto?.randomUUID) return window.crypto.randomUUID();
    return `nonce-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };
  const prepareSmoke = async (control) => {
    busy(control, true);
    const nonce = $("smoke-preparation-nonce");
    if (!nonce.value.trim()) nonce.value = freshSmokeNonce();
    try {
      const plan = await jsonRequest("/evaluation/api/playground/smokes/prepare", {
        method:"POST",
        body:JSON.stringify({conditioned_job_id:$("smoke-conditioned-job").value, preparation_nonce:nonce.value.trim(), explicit_action:true}),
      });
      $("smoke-id").value = plan.smoke_id;
      $("smoke-plan-identity").value = plan.plan_identity;
      $("smoke-plan-output").textContent = JSON.stringify(plan, null, 2);
      smokePlans.unshift({...plan, conditioned_job_id:$("smoke-conditioned-job").value});
      $("smoke-registration-status").textContent = "Plan prepared. Use the Run CPU and Run CUDA web actions; commands are displayed only for transparency.";
      toast("Exploratory smoke plan prepared. Production training was not started.");
      await refreshSmokeStatus();
    } catch (error) {
      $("smoke-registration-status").textContent = error.message;
      toast(error.message);
    } finally { busy(control, false); }
  };
  let smokePoll = null;
  const renderSmokeDevice = (device, value) => {
    const node = $(`smoke-${device}-status`);
    if (!node) return;
    node.textContent = `${device.toUpperCase()} · ${value.status} · ${value.current || 0} / ${value.total || 2}\n${(value.logs || []).join("\n")}`;
  };
  const refreshSmokeStatus = async () => {
    const identity = smokeIdentityPayload();
    if (!identity.conditioned_job_id || !identity.smoke_id || !identity.plan_identity) return;
    const query = new URLSearchParams({conditioned_job_id:identity.conditioned_job_id, plan_identity:identity.plan_identity});
    const status = await jsonRequest(`/evaluation/api/playground/smokes/${encodeURIComponent(identity.smoke_id)}/status?${query}`);
    renderSmokeDevice("cpu", status.devices.cpu);
    renderSmokeDevice("cuda", status.devices.cuda);
    $("run-cpu-smoke").disabled = status.devices.cpu.status !== "NOT_STARTED";
    $("run-cuda-smoke").disabled = status.devices.cpu.status !== "COMPLETE" || status.devices.cuda.status !== "NOT_STARTED";
    $("register-smoke").disabled = !status.registration_ready;
    const active = Object.values(status.devices).some((value) => ["STARTING", "RUNNING"].includes(value.status));
    const terminalFailure = Object.values(status.devices).some((value) => ["FAILED", "INTERRUPTED"].includes(value.status));
    if (terminalFailure) $("smoke-registration-status").textContent = "A device run failed or was interrupted. Use a fresh retry nonce to prepare a new bundle; this bundle cannot resume.";
    if (active && !smokePoll) smokePoll = window.setInterval(() => refreshSmokeStatus().catch((error) => toast(error.message)), 1500);
    if (!active && smokePoll) { window.clearInterval(smokePoll); smokePoll = null; }
  };
  const runSmoke = async (device, control) => {
    busy(control, true);
    try {
      const result = await jsonRequest(`/evaluation/api/playground/smokes/run-${device}`, {
        method:"POST",
        body:JSON.stringify({...smokeIdentityPayload(), explicit_action:true}),
      });
      renderSmokeDevice(device, result);
      $("smoke-registration-status").textContent = `${device.toUpperCase()} smoke started by Sprite Lab.`;
      await refreshSmokeStatus();
    } catch (error) {
      $("smoke-registration-status").textContent = error.message;
      toast(error.message);
    } finally { busy(control, false); }
  };
  const registerSmoke = async (control) => {
    busy(control, true);
    try {
      const result = await jsonRequest("/evaluation/api/playground/smokes/register", {
        method:"POST",
        body:JSON.stringify({
          ...smokeIdentityPayload(),
          explicit_action:true,
        }),
      });
      const catalog = await jsonRequest("/evaluation/api/playground/exploratory-checkpoints");
      exploratoryCheckpoints = catalog.eligible || [];
      fillCheckpoints();
      $("smoke-registration-status").textContent = result.message;
      toast("Exploratory checkpoint registered for Playground only.");
    } catch (error) {
      $("smoke-registration-status").textContent = error.message;
      toast(error.message);
    } finally { busy(control, false); }
  };
  if ($("smoke-preparation-nonce") && !$("smoke-preparation-nonce").value) {
    $("smoke-preparation-nonce").value = freshSmokeNonce();
  }
  fillSmokePublications();
  const restoreSmokePlan = () => {
    const plan = smokePlans.find((item) => item.conditioned_job_id === $("smoke-conditioned-job").value);
    if (!plan) return;
    $("smoke-id").value = plan.smoke_id;
    $("smoke-plan-identity").value = plan.plan_identity;
    $("smoke-plan-output").textContent = JSON.stringify(plan, null, 2);
    refreshSmokeStatus().catch((error) => toast(error.message));
  };
  restoreSmokePlan();
  fillCheckpoints();
  const durable = state.durable_run || {};
  renderStages(durable.stages?.length ? durable.stages : (state.promotion ? [state.promotion] : []));
  renderDashboard(durable.dashboard || {metric_cards:[], charts:[], per_category:[], gallery:[]});
  renderPlaygroundRun(state.playground_run);
  refreshPresets();
  jsonRequest("/evaluation/api/plan").then((result) => {
    if (!durable.run_id) renderStages(result.stages);
  }).catch((error) => toast(error.message));
  $("start-evaluation")?.addEventListener("click", (event) => runEvaluation(false, event.currentTarget));
  $("dry-run")?.addEventListener("click", (event) => runEvaluation(true, event.currentTarget));
  $("generate")?.addEventListener("click", (event) => generate(event.currentTarget));
  $("prepare-smoke")?.addEventListener("click", (event) => prepareSmoke(event.currentTarget));
  $("smoke-conditioned-job")?.addEventListener("change", restoreSmokePlan);
  $("fresh-smoke-bundle")?.addEventListener("click", () => {
    $("smoke-preparation-nonce").value = freshSmokeNonce();
    $("smoke-id").value = "";
    $("smoke-plan-identity").value = "";
    $("smoke-plan-output").textContent = "Fresh retry nonce ready. Prepare a new immutable bundle.";
    renderSmokeDevice("cpu", {status:"NOT_STARTED", current:0, total:2, logs:[]});
    renderSmokeDevice("cuda", {status:"NOT_STARTED", current:0, total:2, logs:[]});
    $("register-smoke").disabled = true;
    $("run-cpu-smoke").disabled = true;
    $("run-cuda-smoke").disabled = true;
  });
  $("run-cpu-smoke")?.addEventListener("click", (event) => runSmoke("cpu", event.currentTarget));
  $("run-cuda-smoke")?.addEventListener("click", (event) => runSmoke("cuda", event.currentTarget));
  $("register-smoke")?.addEventListener("click", (event) => registerSmoke(event.currentTarget));
  $("load-technical")?.addEventListener("click", async (event) => {
    busy(event.currentTarget, true);
    try { $("technical-output").textContent = JSON.stringify(await jsonRequest("/evaluation/api/technical/checkpoints?acknowledge=true"), null, 2); }
    catch (error) { toast(error.message); } finally { busy(event.currentTarget, false); }
  });
  $("save-preset")?.addEventListener("click", async () => {
    const name = window.prompt("Preset name"); if (!name) return;
    const request = {prompt:$("play-prompt").value, checkpoint_id:$("play-checkpoint").value, weights:$("play-weights").value,
      seed:Number($("play-seed").value), sampling_steps:Number($("play-steps").value), guidance:Number($("play-guidance").value), image_count:Number($("play-count").value)};
    try { await jsonRequest("/evaluation/api/playground/presets", {method:"POST", body:JSON.stringify({name, request})}); await refreshPresets(); toast("Prompt preset saved."); }
    catch (error) { toast(error.message); }
  });
  $("rerun-preset")?.addEventListener("click", async () => {
    const name = $("saved-presets").value; if (!name) { toast("Choose a saved preset."); return; }
    try { const result = await jsonRequest(`/evaluation/api/playground/presets/${encodeURIComponent(name)}/rerun`, {method:"POST", body:JSON.stringify({explicit_action:true, confirm_billable:$("billable-confirm").checked})}); renderPlaygroundRun(result); }
    catch (error) { toast(error.message); }
  });
  $("gallery-prompt")?.addEventListener("input", async () => {
    const params = new URLSearchParams({prompt:$("gallery-prompt").value, category:$("gallery-category").value, sort_metric:$("gallery-sort").value});
    if ($("gallery-seed").value) params.set("seed", $("gallery-seed").value);
    try { const result = await jsonRequest(`/evaluation/api/gallery?${params}`); renderGallery(result.samples, "sample-gallery"); }
    catch (error) { toast(error.message); }
  });
  $("close-dialog")?.addEventListener("click", () => $("sample-dialog").close());
  $("comparison-mode")?.addEventListener("click", () => toast("Select two eligible evaluation reports to compare."));
})();
