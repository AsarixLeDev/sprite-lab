(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content ||
    decodeURIComponent(document.cookie.split("; ").find((value) => value.startsWith("spritelab_csrf="))?.split("=")[1] || "");
  let initial = {};
  try { initial = JSON.parse($("evaluation-initial-state")?.textContent || "{}"); } catch (_error) { initial = {}; }
  const request = async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf},
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || "The exploratory smoke action was refused safely.");
    return body;
  };
  const identity = () => ({
    conditioned_job_id: $("smoke-conditioned-job").value,
    smoke_id: $("smoke-id").value,
    plan_identity: $("smoke-plan-identity").value,
  });
  const freshNonce = () => window.crypto?.randomUUID?.() || `nonce-${Date.now()}`;
  const jobs = initial.smoke_publications?.eligible || [];
  const plans = [...(initial.smoke_plans?.eligible || [])];
  if (!jobs.length) $("smoke-conditioned-job").add(new Option("No eligible pre-activation publication", ""));
  jobs.forEach((item) => $("smoke-conditioned-job").add(new Option(item.label, item.conditioned_job_id)));
  $("smoke-preparation-nonce").value = freshNonce();
  let poll = null;
  const render = (device, value) => {
    $(`smoke-${device}-status`).textContent = `${device.toUpperCase()} · ${value.status} · ${value.current || 0} / ${value.total || 2}\n${(value.logs || []).join("\n")}`;
  };
  const status = async () => {
    const value = identity();
    if (!value.conditioned_job_id || !value.smoke_id || !value.plan_identity) return;
    const query = new URLSearchParams({conditioned_job_id:value.conditioned_job_id, plan_identity:value.plan_identity});
    const result = await request(`/evaluation/api/playground/smokes/${encodeURIComponent(value.smoke_id)}/status?${query}`);
    render("cpu", result.devices.cpu);
    render("cuda", result.devices.cuda);
    $("run-cpu-smoke").disabled = result.devices.cpu.status !== "NOT_STARTED";
    $("run-cuda-smoke").disabled = result.devices.cpu.status !== "COMPLETE" || result.devices.cuda.status !== "NOT_STARTED";
    $("register-smoke").disabled = !result.registration_ready;
    const active = Object.values(result.devices).some((item) => ["STARTING", "RUNNING"].includes(item.status));
    if (active && !poll) poll = window.setInterval(() => status().catch(() => {}), 1500);
    if (!active && poll) { window.clearInterval(poll); poll = null; }
  };
  $("prepare-smoke").addEventListener("click", async () => {
    try {
      const plan = await request("/evaluation/api/playground/smokes/prepare", {
        method:"POST",
        body:JSON.stringify({conditioned_job_id:$("smoke-conditioned-job").value, preparation_nonce:$("smoke-preparation-nonce").value, explicit_action:true}),
      });
      $("smoke-id").value = plan.smoke_id;
      $("smoke-plan-identity").value = plan.plan_identity;
      $("smoke-plan-output").textContent = JSON.stringify(plan, null, 2);
      plans.unshift({...plan, conditioned_job_id:$("smoke-conditioned-job").value});
      $("smoke-registration-status").textContent = "Plan prepared. Use both server-run web actions; commands are transparency only.";
      await status();
    } catch (error) {
      $("smoke-registration-status").textContent = error.message;
    }
  });
  const run = async (device) => {
    try {
      const result = await request(`/evaluation/api/playground/smokes/run-${device}`, {method:"POST", body:JSON.stringify({...identity(), explicit_action:true})});
      render(device, result);
      await status();
    } catch (error) { $("smoke-registration-status").textContent = error.message; }
  };
  $("run-cpu-smoke").addEventListener("click", () => run("cpu"));
  $("run-cuda-smoke").addEventListener("click", () => run("cuda"));
  $("fresh-smoke-bundle").addEventListener("click", () => {
    $("smoke-preparation-nonce").value = freshNonce();
    $("smoke-id").value = "";
    $("smoke-plan-identity").value = "";
    $("run-cpu-smoke").disabled = true;
    $("run-cuda-smoke").disabled = true;
    $("register-smoke").disabled = true;
  });
  const restore = () => {
    const plan = plans.find((item) => item.conditioned_job_id === $("smoke-conditioned-job").value);
    if (!plan) return;
    $("smoke-id").value = plan.smoke_id;
    $("smoke-plan-identity").value = plan.plan_identity;
    $("smoke-plan-output").textContent = JSON.stringify(plan, null, 2);
    status().catch(() => {});
  };
  $("smoke-conditioned-job").addEventListener("change", restore);
  restore();
  $("register-smoke").addEventListener("click", async () => {
    try {
      const result = await request("/evaluation/api/playground/smokes/register", {
        method:"POST",
        body:JSON.stringify({...identity(), explicit_action:true}),
      });
      $("smoke-registration-status").textContent = result.message;
    } catch (error) {
      $("smoke-registration-status").textContent = error.message;
    }
  });
})();
