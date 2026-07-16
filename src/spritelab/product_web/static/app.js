(() => {
  "use strict";
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("spritelab-theme");
  const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = storedTheme || (preferredDark ? "dark" : "light");

  const toastRegion = document.querySelector("[data-toast-region]");
  const toast = (message, tone = "neutral") => {
    if (!toastRegion) return;
    const item = document.createElement("div");
    item.className = `toast toast-${tone}`;
    item.textContent = message;
    toastRegion.appendChild(item);
    window.setTimeout(() => item.remove(), 5000);
  };

  document.querySelector("[data-theme-toggle]")?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("spritelab-theme", root.dataset.theme);
    toast(`${root.dataset.theme[0].toUpperCase()}${root.dataset.theme.slice(1)} theme enabled`);
  });

  const sidebar = document.querySelector(".sidebar");
  const sidebarToggle = document.querySelector(".sidebar-toggle");
  sidebarToggle?.addEventListener("click", () => {
    const open = sidebar?.classList.toggle("is-open") || false;
    sidebarToggle.setAttribute("aria-expanded", String(open));
  });

  document.querySelectorAll("[data-open-dialog]").forEach((button) => {
    button.addEventListener("click", () => document.getElementById(button.dataset.openDialog)?.showModal());
  });
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog")?.close());
  });
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });

  const connection = document.querySelector("[data-connection-state]");
  const setConnection = (connected) => {
    if (connection) connection.hidden = connected;
    document.querySelectorAll('[data-live="connection"]').forEach((node) => {
      node.textContent = connected ? "Live" : "Reconnecting…";
    });
  };
  const liveRoot = document.querySelector("[data-run-id]");
  if (liveRoot) {
    const runId = encodeURIComponent(liveRoot.dataset.runId);
    const source = new EventSource(`/api/runs/${runId}/events`);
    source.addEventListener("open", () => setConnection(true));
    source.addEventListener("error", () => setConnection(false));
    source.addEventListener("product", (event) => {
      const data = JSON.parse(event.data);
      const setText = (name, value) => document.querySelectorAll(`[data-live="${name}"]`).forEach((node) => { node.textContent = value; });
      setText("stage", String(data.stage || "Current stage").replaceAll("-", " "));
      setText("status", String(data.status || "RUNNING").replaceAll("_", " "));
      setText("message", data.message || "");
      setText("counter", data.total == null ? String(data.current) : `${data.current} / ${data.total}`);
      const progress = document.querySelector("progress");
      if (progress) {
        progress.value = Number(data.current || 0);
        if (data.total != null) progress.max = Number(data.total);
      }
      const list = document.querySelector("[data-live-messages]");
      if (list && data.message) {
        const item = document.createElement("li");
        item.textContent = data.message;
        list.appendChild(item);
        while (list.children.length > 8) list.firstElementChild?.remove();
      }
    });
    source.addEventListener("snapshot", (event) => {
      const data = JSON.parse(event.data);
      if (data.terminal) {
        setConnection(true);
        source.close();
      }
    });
  }

  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  document.querySelectorAll("[data-async-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const response = await fetch(button.dataset.asyncAction, { method: "POST", headers: { "X-CSRF-Token": csrf } });
        const payload = await response.json();
        toast(payload.detail || (response.ok ? "Action accepted" : "Action could not be completed"), response.ok ? "success" : "warning");
      } catch (_error) {
        toast("Disconnected. The action was not sent.", "warning");
      } finally {
        button.disabled = false;
      }
    });
  });

  const logPanel = document.querySelector("[data-log-run]");
  if (logPanel) {
    const output = logPanel.querySelector("[data-log-output]");
    const initialLines = output?.textContent.split("\n").filter(Boolean).length || 0;
    const source = new EventSource(`/api/runs/${encodeURIComponent(logPanel.dataset.logRun)}/logs?after=${initialLines}`);
    source.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      if (output) output.textContent += `${output.textContent.endsWith("\n") ? "" : "\n"}${data.line}`;
    });
  }
})();
