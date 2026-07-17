(() => {
  "use strict";
  const form = document.getElementById("hierarchical-labeling-settings");
  const summary = document.getElementById("labeling-settings-summary");
  if (!form || !summary) return;
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  const button = id => document.getElementById(id);
  const state = button("hierarchical-state");
  const busy = (node, value) => {
    if (!node) return;
    node.disabled = value;
    node.setAttribute("aria-busy", String(value));
  };
  const reflectEnabled = () => {
    const enabled = button("hierarchical-enabled").checked;
    form.dataset.enabled = String(enabled);
    if (state) state.lastChild.textContent = enabled ? "Enabled" : "Disabled";
  };
  const request = async (method) => {
    const options = {method, headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf}};
    if (method === "POST") {
      options.body = JSON.stringify({
        hierarchical_enabled: button("hierarchical-enabled").checked,
        hierarchical_profile: button("hierarchical-profile").value,
        reference_cohort_size: Number(button("reference-cohort-size").value),
      });
    }
    const response = await fetch("/settings/api/labeling", options);
    const body = await response.json();
    if (!response.ok) {
      if (body.error_code === "csrf_validation_failed") {
        throw new Error("This page expired. Reload it before saving settings.");
      }
      throw new Error(body.message || "The settings could not be saved.");
    }
    return body;
  };
  button("save-labeling-settings")?.addEventListener("click", async event => {
    busy(event.currentTarget, true);
    summary.textContent = "Saving project settings…";
    try { summary.textContent = (await request("POST")).message; }
    catch (error) { summary.textContent = error.message; }
    finally { busy(event.currentTarget, false); }
  });
  button("hierarchical-enabled")?.addEventListener("change", reflectEnabled);
  button("reset-labeling-settings")?.addEventListener("click", async event => {
    busy(event.currentTarget, true);
    try { summary.textContent = (await request("DELETE")).message; }
    catch (error) { summary.textContent = error.message; }
    finally { busy(event.currentTarget, false); }
  });
  reflectEnabled();
})();
