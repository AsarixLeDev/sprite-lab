(() => {
  "use strict";
  const form = document.getElementById("vision-provider-settings");
  const summary = document.getElementById("summary");
  if (!form || !summary) return;
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  const button = (id) => document.getElementById(id);
  const setBusy = (node, busy) => { if (node) { node.disabled = busy; node.setAttribute("aria-busy", String(busy)); } };
  const formData = () => {
    const data = Object.fromEntries(new FormData(form));
    data.timeout = Number(data.timeout);
    data.batch_size = Number(data.batch_size);
    for (const key of ["endpoint", "model", "credential_env", "location"]) if (!data[key]) delete data[key];
    return data;
  };
  const request = async (url, options = {}) => {
    const response = await fetch(url, { ...options, headers: {"Content-Type":"application/json", "X-CSRF-Token":csrf, ...(options.headers || {})} });
    const contentType = response.headers.get("content-type") || "";
    let body;
    if (contentType.includes("application/json")) body = await response.json();
    else { await response.text(); throw new Error("Sprite Lab received an unexpected non-JSON response. Reload and try again."); }
    if (!response.ok) throw new Error(body.message || "The provider action could not be completed.");
    return body;
  };
  const show = (value) => { summary.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2); };
  const updateMode = () => {
    const mode = form.querySelector('input[name="type"]:checked')?.value || "auto";
    const automatic = mode === "auto";
    document.querySelector("[data-advanced-provider-fields]").hidden = automatic;
    document.querySelector("[data-hosted-warning]").hidden = mode !== "hosted";
  };
  form.querySelectorAll('input[name="type"]').forEach((node) => node.addEventListener("change", updateMode));
  button("save")?.addEventListener("click", async (event) => {
    setBusy(event.currentTarget, true); show("Saving settings without contacting the provider…");
    try { const result = await request("/settings/vision/api/settings", {method:"POST", body:JSON.stringify(formData())}); show(result.message); }
    catch (error) { show(error.message); }
    finally { setBusy(event.currentTarget, false); }
  });
  button("detect")?.addEventListener("click", async (event) => {
    setBusy(event.currentTarget, true); show("Detecting providers after your explicit request…");
    try { show(await request("/settings/vision/api/detect", {method:"POST", body:"{}"})); }
    catch (error) { show(error.message); }
    finally { setBusy(event.currentTarget, false); }
  });
  button("test")?.addEventListener("click", async (event) => {
    setBusy(event.currentTarget, true); show("Running one health-only connection test. No image is sent…");
    try { show(await request("/settings/vision/api/test", {method:"POST", body:JSON.stringify(formData())})); }
    catch (error) { show(error.message); }
    finally { setBusy(event.currentTarget, false); }
  });
  button("refresh-models")?.addEventListener("click", async (event) => {
    setBusy(event.currentTarget, true); show("Refreshing the configured provider model list…");
    try { const result=await request("/settings/vision/api/models/refresh", {method:"POST", body:JSON.stringify(formData())}); document.getElementById("models").textContent=result.models.length ? result.models.map(model=>model.display_name || model.model_id).join("\n") : "No models were reported."; show("Model list refreshed by explicit action."); }
    catch (error) { show(error.message); }
    finally { setBusy(event.currentTarget, false); }
  });
  button("clear")?.addEventListener("click", async (event) => {
    setBusy(event.currentTarget, true);
    try { const result=await request("/settings/vision/api/settings", {method:"DELETE", body:"{}"}); show(result.message); }
    catch (error) { show(error.message); }
    finally { setBusy(event.currentTarget, false); }
  });
  updateMode();
})();
