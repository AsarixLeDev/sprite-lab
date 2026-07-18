(() => {
  "use strict";

  const root = document.querySelector("[data-conditioned-v5-root]");
  if (!root) return;

  const get = (id) => document.getElementById(id);
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  const busy = new Set();
  let state = { managed_intakes: [], jobs: [], config_sha256: null };
  let selectedJob = "";
  let selectedJobData = null;
  let selectedPublication = null;
  let trainingAuditOptions = [];
  let trainingAuditOptionsError = "";
  let trainingAuditResult = null;
  let previewResult = null;
  let pollTimer = null;
  let pollInFlight = false;

  try {
    state = JSON.parse(get("conditioned-v5-initial")?.textContent || "{}");
  } catch (_error) {
    // The first inventory refresh supplies the authoritative state.
  }

  const setStatus = (value) => {
    const node = get("cv5-status");
    if (node) node.textContent = value;
  };

  const request = async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
        ...(options.headers || {}),
      },
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.message || "Dataset-v5 request failed.");
    }
    return body;
  };

  const authorizationId = (prefix) =>
    `${prefix}-${globalThis.crypto?.randomUUID?.() || Date.now().toString(36)}`;

  const selectedReferences = () =>
    [...document.querySelectorAll("[data-intake]:checked")].map((node) => node.value);

  const setPolling = (enabled) => {
    const indicator = get("cv5-poll");
    if (indicator) indicator.textContent = enabled ? "Polling" : "Idle";
    if (!enabled && pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (enabled && pollTimer === null) {
      pollTimer = setInterval(() => {
        if (pollInFlight || busy.size > 0 || !selectedJob) return;
        pollInFlight = true;
        void loadJob(selectedJob, { startPolling: true })
          .catch((error) => setStatus(error instanceof Error ? error.message : "Job polling failed."))
          .finally(() => {
            pollInFlight = false;
          });
      }, 1500);
    }
  };

  const resetJobAuthorizations = () => {
    const freeze = get("cv5-authorize");
    if (freeze) freeze.checked = false;
    const id = get("cv5-activation-auth");
    if (id) id.value = "";
    for (const checkboxId of ["cv5-authorize-dataset", "cv5-authorize-training"]) {
      const checkbox = get(checkboxId);
      if (checkbox) checkbox.checked = false;
    }
  };

  const selectJobIdentity = (jobId) => {
    if (jobId === selectedJob) return;
    selectedJob = jobId;
    selectedJobData = null;
    selectedPublication = null;
    trainingAuditOptions = [];
    trainingAuditOptionsError = "";
    trainingAuditResult = null;
    renderTrainingAuditOptions();
    renderTrainingAuditResult();
    resetJobAuthorizations();
  };

  function updateControls() {
    const anyBusy = busy.size > 0;
    const build = get("cv5-build");
    const publish = get("cv5-publish");
    const trainingAudit = get("cv5-training-audit");
    const activate = get("cv5-activate");
    const auditReady = Boolean(
      selectedJobData?.candidate && selectedJobData?.status === "NEEDS_REVIEW" && !selectedPublication,
    );
    const evidenceReady = Boolean(
      selectedJobData?.candidate &&
        selectedJobData?.evidence?.label_audit &&
        selectedJobData?.evidence?.dataset_validation,
    );
    const activationReady = Boolean(
      selectedJob &&
        selectedJobData?.candidate &&
        selectedPublication &&
        selectedPublication.configuration_activated === false &&
        get("cv5-config-sha")?.value?.match(/^[0-9a-f]{64}$/) &&
        get("cv5-activation-auth")?.value?.trim() &&
        get("cv5-authorize-dataset")?.checked &&
        get("cv5-authorize-training")?.checked,
    );
    const publicationReadyForAudit = Boolean(
      selectedJob &&
        selectedJobData?.status === "COMPLETE" &&
        selectedJobData?.candidate &&
        selectedPublication &&
        selectedPublication.configuration_activated === false,
    );
    const selectedSmokeId = get("cv5-training-smoke")?.value || "";
    const registeredSmokeReady = trainingAuditOptions.some((item) => item.smoke_id === selectedSmokeId);
    const trainingAuditReady = Boolean(
      publicationReadyForAudit &&
        registeredSmokeReady &&
        !trainingAuditResult &&
        state.config_sha256?.match(/^[0-9a-f]{64}$/),
    );
    if (build) build.disabled = anyBusy || previewResult?.ready_to_build !== true;
    if (publish) {
      publish.disabled = anyBusy || !evidenceReady || Boolean(selectedPublication) || !get("cv5-authorize")?.checked;
    }
    if (activate) activate.disabled = anyBusy || !activationReady;
    if (trainingAudit) trainingAudit.disabled = anyBusy || !trainingAuditReady;
    const trainingAuditControls = get("cv5-training-audit-controls");
    if (trainingAuditControls) {
      trainingAuditControls.hidden = !publicationReadyForAudit || trainingAuditOptions.length === 0;
    }
    const trainingAuditPrerequisite = get("cv5-training-audit-prerequisite");
    if (trainingAuditPrerequisite) {
      if (!selectedPublication) {
        trainingAuditPrerequisite.textContent = "Publish the exact freeze and campaign first.";
      } else if (selectedPublication.configuration_activated !== false) {
        trainingAuditPrerequisite.textContent = "This publication is already activated; audit must precede activation.";
      } else if (trainingAuditOptions.length === 0) {
        trainingAuditPrerequisite.textContent =
          trainingAuditOptionsError || "Complete and register this publication's CPU/CUDA smoke in Playground first.";
      } else if (trainingAuditResult) {
        trainingAuditPrerequisite.textContent =
          `The server-managed audit returned ${trainingAuditResult.verdict || "UNKNOWN"}. It did not activate or start training.`;
      } else {
        trainingAuditPrerequisite.textContent =
          "A registered exploratory smoke is bound to this exact freeze and campaign. The audit does not activate or start training.";
      }
    }
    for (const id of ["cv5-refresh", "cv5-preview"]) {
      const button = get(id);
      if (button) button.disabled = anyBusy;
    }
    for (const id of ["cv5-audit-label", "cv5-audit-validation"]) {
      const button = get(id);
      if (button) button.disabled = anyBusy || !auditReady;
    }
  }

  function renderTrainingAuditOptions() {
    const select = get("cv5-training-smoke");
    if (!select) return;
    const previous = select.value;
    select.replaceChildren(
      ...trainingAuditOptions.map((item) => {
        const option = document.createElement("option");
        option.value = item.smoke_id;
        option.textContent = `${item.registration_id} · ${item.smoke_id}`;
        return option;
      }),
    );
    if (trainingAuditOptions.some((item) => item.smoke_id === previous)) select.value = previous;
  }

  function renderTrainingAuditResult() {
    const output = get("cv5-training-audit-result");
    if (!output) return;
    if (!trainingAuditResult) {
      output.textContent = "No training infrastructure audit run.";
      return;
    }
    output.textContent = JSON.stringify(
      {
        schema_version: trainingAuditResult.schema_version,
        verdict: trainingAuditResult.verdict,
        smoke_id: trainingAuditResult.smoke_id,
        operation_identity: trainingAuditResult.operation_identity,
        config_unchanged: trainingAuditResult.config_unchanged,
        configuration_activated: trainingAuditResult.configuration_activated,
        training_started: trainingAuditResult.training_started,
        paths_exposed: trainingAuditResult.paths_exposed,
      },
      null,
      2,
    );
  }

  async function loadTrainingAuditOptions(jobId, job) {
    trainingAuditOptions = [];
    trainingAuditOptionsError = "";
    const publication = job?.publication;
    if (!publication || publication.configuration_activated !== false) {
      renderTrainingAuditOptions();
      return;
    }
    try {
      const result = await request(
        `/dataset-v5/api/jobs/${encodeURIComponent(jobId)}/training-audit-options`,
      );
      if (selectedJob !== jobId) return;
      trainingAuditOptions = Array.isArray(result.eligible) ? result.eligible : [];
    } catch (error) {
      if (selectedJob !== jobId) return;
      trainingAuditOptionsError =
        error instanceof Error ? error.message : "Registered smoke options are unavailable.";
    }
    renderTrainingAuditOptions();
  }

  async function runExclusive(key, action) {
    if (busy.size > 0) return;
    busy.add(key);
    updateControls();
    try {
      await action();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Dataset-v5 request failed.");
    } finally {
      busy.delete(key);
      updateControls();
    }
  }

  function renderInventory(preservedReferences = new Set()) {
    const intakes = get("cv5-intakes");
    if (intakes) {
      intakes.replaceChildren(
        ...(state.managed_intakes || []).map((item) => {
          const label = document.createElement("label");
          label.className = "cv5-card";
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.value = item.dataset_reference;
          checkbox.dataset.intake = "";
          checkbox.checked = preservedReferences.has(item.dataset_reference);
          const body = document.createElement("span");
          const heading = document.createElement("strong");
          heading.textContent = item.dataset_reference;
          const summary = document.createElement("p");
          summary.textContent = `${item.accepted_count} accepted · ${item.quarantined_count} quarantined · ${item.status}`;
          body.append(heading, summary);
          label.append(checkbox, body);
          return label;
        }),
      );
    }

    const jobs = get("cv5-jobs");
    if (jobs) {
      jobs.replaceChildren(
        ...(state.jobs || []).map((item) => {
          const article = document.createElement("article");
          article.className = "cv5-job";
          const head = document.createElement("div");
          head.className = "cv5-job-head";
          const button = document.createElement("button");
          button.type = "button";
          button.className = "button secondary";
          button.dataset.jobId = item.job_id;
          button.textContent = item.job_id;
          const badge = document.createElement("span");
          badge.className = "status-pill";
          badge.textContent = item.status;
          head.append(button, badge);
          const summary = document.createElement("p");
          summary.textContent = `${item.stage} · ${item.current || 0}/${item.total || 0} · ${item.message || ""}`;
          article.append(head, summary);
          return article;
        }),
      );
    }
  }

  async function loadJob(jobId, { startPolling = false } = {}) {
    selectJobIdentity(jobId);
    const job = await request(`/dataset-v5/api/jobs/${encodeURIComponent(jobId)}`);
    selectedJobData = job;
    selectedPublication = job.publication || null;
    const logs = get("cv5-logs");
    if (logs) {
      logs.textContent = JSON.stringify(
        {
          status: job.status,
          stage: job.stage,
          counts: job.candidate,
          events: job.events,
          evidence: job.evidence,
        },
        null,
        2,
      );
    }
    const publication = get("cv5-publication");
    if (publication) {
      publication.textContent = selectedPublication
        ? JSON.stringify(selectedPublication, null, 2)
        : "Nothing published.";
    }
    const activationAuthorization = get("cv5-activation-auth");
    if (selectedPublication && activationAuthorization && !activationAuthorization.value) {
      activationAuthorization.value = authorizationId("activation-auth");
    }
    await loadTrainingAuditOptions(jobId, job);
    setPolling(startPolling && ["RUNNING", "CANCELLING"].includes(job.status));
    updateControls();
    return job;
  }

  async function refreshInventory() {
    const preserved = new Set(selectedReferences());
    state = await request("/dataset-v5/api/inventory");
    const config = get("cv5-config-sha");
    if (config) config.value = state.config_sha256 || "";
    renderInventory(preserved);
    if (selectedJob) {
      const stillPresent = (state.jobs || []).some((job) => job.job_id === selectedJob);
      if (stillPresent) await loadJob(selectedJob, { startPolling: true });
      else selectJobIdentity("");
    }
    updateControls();
  }

  get("cv5-intakes")?.addEventListener("change", () => {
    previewResult = null;
    updateControls();
  });

  get("cv5-jobs")?.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest("button[data-job-id]");
    if (!button) return;
    void runExclusive("select", async () => {
      await loadJob(button.dataset.jobId || "", { startPolling: true });
    });
  });

  get("cv5-refresh")?.addEventListener("click", () => {
    void runExclusive("refresh", refreshInventory);
  });

  get("cv5-preview")?.addEventListener("click", () => {
    void runExclusive("preview", async () => {
      const result = await request("/dataset-v5/api/preview", {
        method: "POST",
        body: JSON.stringify({ dataset_references: selectedReferences() }),
      });
      previewResult = result;
      const output = get("cv5-preview-output");
      if (output) output.textContent = JSON.stringify(result, null, 2);
      setStatus(result.ready_to_build ? "Preview is ready to build." : result.blockers.join(" "));
    });
  });

  get("cv5-build")?.addEventListener("click", () => {
    void runExclusive("build", async () => {
      const result = await request("/dataset-v5/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          dataset_references: selectedReferences(),
          idempotency_key: authorizationId("conditioned-build"),
          explicit_action: true,
        }),
      });
      selectJobIdentity(result.job.job_id);
      previewResult = null;
      await refreshInventory();
      setStatus("Candidate build started.");
    });
  });

  const runIndependentAudit = async (kind) => {
    if (!selectedJob) throw new Error("Select a completed candidate job first.");
    setStatus(`Running independent ${kind.replace("_", " ")} on the server...`);
    await request(`/dataset-v5/api/jobs/${encodeURIComponent(selectedJob)}/evidence`, {
      method: "POST",
      body: JSON.stringify({ kind, explicit_action: true }),
    });
    await loadJob(selectedJob);
    setStatus(`${kind.replace("_", " ")} completed and attached.`);
  };

  get("cv5-audit-label")?.addEventListener("click", () => {
    void runExclusive("audit-label", () => runIndependentAudit("label_audit"));
  });
  get("cv5-audit-validation")?.addEventListener("click", () => {
    void runExclusive("audit-validation", () => runIndependentAudit("dataset_validation"));
  });

  get("cv5-authorize")?.addEventListener("change", updateControls);
  get("cv5-training-smoke")?.addEventListener("change", updateControls);
  for (const id of ["cv5-authorize-dataset", "cv5-authorize-training", "cv5-activation-auth"]) {
    get(id)?.addEventListener("input", updateControls);
    get(id)?.addEventListener("change", updateControls);
  }

  get("cv5-publish")?.addEventListener("click", () => {
    void runExclusive("publish", async () => {
      if (!selectedJob) throw new Error("Select a completed candidate job first.");
      const job = await loadJob(selectedJob);
      const result = await request(`/dataset-v5/api/jobs/${encodeURIComponent(selectedJob)}/publish`, {
        method: "POST",
        body: JSON.stringify({
          candidate_identity: job.candidate.candidate_identity,
          label_audit_sha256: job.evidence.label_audit.sha256,
          dataset_validation_sha256: job.evidence.dataset_validation.sha256,
          authorization_id: authorizationId("freeze-auth"),
          explicit_action: true,
          authorize_one_time_freeze: true,
        }),
      });
      selectedJobData = result;
      selectedPublication = result.publication;
      const freeze = get("cv5-authorize");
      if (freeze) freeze.checked = false;
      await refreshInventory();
      setStatus("Freeze and campaign published. Configuration and training remain unchanged.");
    });
  });

  get("cv5-training-audit")?.addEventListener("click", () => {
    void runExclusive("training-audit", async () => {
      if (!selectedJob) throw new Error("Select a published conditioned job first.");
      await refreshInventory();
      const job = selectedJobData;
      const publication = job?.publication;
      const smokeId = get("cv5-training-smoke")?.value || "";
      if (!job?.candidate || !publication || publication.configuration_activated !== false) {
        throw new Error("Select an inactive published conditioned job first.");
      }
      if (!trainingAuditOptions.some((item) => item.smoke_id === smokeId)) {
        throw new Error("Select a registered smoke bound to this publication.");
      }
      setStatus("Running the server-managed 18-gate training audit. Configuration and training remain unchanged...");
      trainingAuditResult = await request(
        `/dataset-v5/api/jobs/${encodeURIComponent(selectedJob)}/training-audit`,
        {
          method: "POST",
          body: JSON.stringify({
            candidate_identity: job.candidate.candidate_identity,
            publication_identity_sha256: publication.publication_identity_sha256,
            activation_manifest_sha256: publication.activation_manifest_sha256,
            campaign_config_sha256: publication.campaign_config_sha256,
            campaign_identity_sha256: publication.campaign_identity_sha256,
            expected_config_sha256: state.config_sha256 || "",
            smoke_id: smokeId,
            operation_nonce: authorizationId("training-audit"),
            explicit_action: true,
          }),
        },
      );
      renderTrainingAuditResult();
      setStatus(
        `Training infrastructure audit verdict: ${trainingAuditResult.verdict || "UNKNOWN"}. Configuration was not activated and training was not started.`,
      );
    });
  });

  get("cv5-activate")?.addEventListener("click", () => {
    void runExclusive("activate", async () => {
      if (!selectedJob) throw new Error("Select a published conditioned job first.");
      await refreshInventory();
      const job = selectedJobData;
      if (!job) throw new Error("Select a published conditioned job first.");
      const publication = job.publication;
      if (!publication) throw new Error("Select a published conditioned job first.");
      const result = await request(`/dataset-v5/api/jobs/${encodeURIComponent(selectedJob)}/activate`, {
        method: "POST",
        body: JSON.stringify({
          candidate_identity: job.candidate.candidate_identity,
          publication_identity_sha256: publication.publication_identity_sha256,
          activation_manifest_sha256: publication.activation_manifest_sha256,
          campaign_config_sha256: publication.campaign_config_sha256,
          campaign_identity_sha256: publication.campaign_identity_sha256,
          expected_config_sha256: get("cv5-config-sha")?.value || "",
          activation_authorization_id: get("cv5-activation-auth")?.value?.trim() || "",
          explicit_action: true,
          authorize_dataset_freeze: get("cv5-authorize-dataset")?.checked === true,
          authorize_training: get("cv5-authorize-training")?.checked === true,
        }),
      });
      selectedJobData = result;
      selectedPublication = result.publication;
      const config = get("cv5-config-sha");
      if (config) config.value = result.activated_config_sha256 || "";
      resetJobAuthorizations();
      setStatus("Audited Dataset-v5 and three-seed campaign activated. Training was not started.");
    });
  });

  renderInventory();
  updateControls();
  void runExclusive("refresh", refreshInventory);
})();
