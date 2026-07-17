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
  const enableProviderActions = ({ models = false } = {}) => {
    for (const id of ["test", "clear"]) {
      const node = button(id); if (node) { node.disabled = false; node.removeAttribute("title"); }
    }
    const refresh = button("refresh-models");
    if (refresh) {
      refresh.disabled = !models;
      if (models) refresh.removeAttribute("title");
      else refresh.title = "Test the provider connection first";
    }
  };
  const updateMode = () => {
    const mode = form.querySelector('input[name="type"]:checked')?.value || "auto";
    const automatic = mode === "auto";
    document.querySelector("[data-advanced-provider-fields]").hidden = automatic;
    document.querySelector("[data-hosted-warning]").hidden = mode !== "hosted";
    document.querySelector("[data-ollama-guidance]").hidden = mode !== "ollama";
  };
  form.querySelectorAll('input[name="type"]').forEach((node) => node.addEventListener("change", updateMode));
  button("save")?.addEventListener("click", async (event) => {
    const actionButton = event.currentTarget;
    setBusy(actionButton, true); show("Saving settings without contacting the provider…");
    try { const result = await request("/settings/vision/api/settings", {method:"POST", body:JSON.stringify(formData())}); enableProviderActions(); show(`${result.message} Test the connection before starting automatic labeling.`); }
    catch (error) { show(error.message); }
    finally { setBusy(actionButton, false); }
  });
  button("detect")?.addEventListener("click", async (event) => {
    const actionButton = event.currentTarget;
    setBusy(actionButton, true); show("Detecting providers after your explicit request…");
    try { const result=await request("/settings/vision/api/detect", {method:"POST", body:"{}"}); const available=result.providers.filter(item=>item.probe?.state==="available"); if(available.length) enableProviderActions({models:true}); show(available.length ? `Found ${available.length} available provider${available.length===1?"":"s"}. Save the mode you want to use, then test the connection.` : "No available vision provider was detected. Start the provider locally or configure an explicit endpoint."); }
    catch (error) { show(error.message); }
    finally { setBusy(actionButton, false); }
  });
  button("test")?.addEventListener("click", async (event) => {
    const actionButton = event.currentTarget;
    setBusy(actionButton, true); show("Running one health-only connection test. No image is sent…");
    try { const result=await request("/settings/vision/api/test", {method:"POST", body:JSON.stringify(formData())}); if(result.available) enableProviderActions({models:true}); show(result.available ? `${result.display_name} is available. You can refresh models, then start labeling.` : `${result.display_name || "The provider"} is not ready: ${result.message || "check the endpoint and selected model"}`); }
    catch (error) { show(error.message); }
    finally { setBusy(actionButton, false); }
  });
  button("refresh-models")?.addEventListener("click", async (event) => {
    const actionButton = event.currentTarget;
    setBusy(actionButton, true); show("Refreshing the configured provider model list…");
    try { const result=await request("/settings/vision/api/models/refresh", {method:"POST", body:JSON.stringify(formData())}); document.getElementById("models").textContent=result.models.length ? result.models.map(model=>model.display_name || model.model_id).join("\n") : "No models were reported."; show("Model list refreshed by explicit action."); }
    catch (error) { show(error.message); }
    finally { setBusy(actionButton, false); }
  });
  button("clear")?.addEventListener("click", async (event) => {
    const actionButton = event.currentTarget;
    setBusy(actionButton, true);
    try { const result=await request("/settings/vision/api/settings", {method:"DELETE", body:"{}"}); show(result.message); for(const id of ["test","refresh-models","clear"]){const node=button(id);if(node)node.disabled=true;} }
    catch (error) { show(error.message); }
    finally { setBusy(actionButton, false); }
  });
  updateMode();
})();
