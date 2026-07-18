from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/spritelab/product_features/training/templates/training.html"
JAVASCRIPT = ROOT / "src/spritelab/product_features/training/static/training.js"
NODE = shutil.which("node")


class _ControlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id is not None:
            self.elements[element_id] = (tag, attributes)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def test_training_template_exposes_one_accessible_fresh_cloud_authorization() -> None:
    parser = _ControlParser()
    parser.feed(TEMPLATE.read_text(encoding="utf-8"))

    cloud_tag, cloud = parser.elements["cloud-confirmation"]
    confirm_tag, confirm = parser.elements["confirm-cloud"]
    form_tag, form = parser.elements["compute-settings"]
    start_tag, start = parser.elements["start"]
    resume_tag, resume = parser.elements["resume"]
    warning_tag, warning = parser.elements["training-resource-warning"]
    rendered_text = " ".join(parser.text).casefold()

    assert cloud_tag == "div"
    assert "hidden" in cloud
    assert confirm_tag == "input"
    assert confirm["type"] == "checkbox"
    assert confirm["aria-describedby"] == "cloud-confirmation-note"
    assert form_tag == "form"
    assert form["data-configuration-version"] == "{{ compute_configuration_version }}"
    assert form["data-backend-identity"] == "{{ compute_backend_identity }}"
    assert start_tag == "button"
    assert {"blockers", "cloud-confirmation-note"} <= set(start["aria-describedby"].split())
    assert resume_tag == "button"
    assert {"resume-reason", "cloud-confirmation-note"} <= set(resume["aria-describedby"].split())
    assert "starting or resuming this cloud backend can incur" in rendered_text
    assert "fresh confirmation is required for each cloud start or resume" in rendered_text
    assert "bound to the exact saved configuration version and backend" in rendered_text
    assert warning_tag == "p" and warning.get("role") == "alert" and "hidden" in warning
    assert parser.elements["training-seed-outcomes"][0] == "ul"
    assert parser.elements["training-job-outcomes"][0] == "ul"


_DOM_HARNESS = r"""
const assert = require("assert");

class FakeNode {
  constructor(id = "") {
    this.id = id;
    this.attributes = {};
    this.children = [];
    this.dataset = {};
    this.disabled = false;
    this.checked = false;
    this.hidden = false;
    this.textContent = "";
    this.value = "";
    this.listeners = new Map();
  }
  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }
  async fire(type, target = this) {
    const event = {currentTarget: this, target, preventDefault() {}};
    for (const listener of this.listeners.get(type) || []) await listener(event);
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  querySelector(selector) { return element(`${this.id}:${selector}`); }
}

const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, new FakeNode(id));
  return elements.get(id);
};

element("dashboard").dataset.trainingRunId = "run-1";
element("training-initial-dashboard").textContent = "null";
element("compute-type").value = "ssh";
element("run-profile").value = "recommended";
element("compute-settings").dataset.configurationVersion = "7";
element("compute-settings").dataset.backendIdentity = "ssh";
element("other-backend").value = "provider-x";

const panels = ["local", "ssh", "runpod", "other"].map(name => {
  const panel = new FakeNode(`panel-${name}`);
  panel.dataset.computePanel = name;
  return panel;
});

global.document = {
  getElementById: element,
  querySelector(selector) {
    if (selector === 'meta[name="spritelab-csrf"]') return {content: "csrf"};
    return element(`query:${selector}`);
  },
  querySelectorAll(selector) {
    return selector === "[data-compute-panel]" ? panels : [];
  },
  createElement(tag) { return new FakeNode(tag); },
  createElementNS(_namespace, tag) { return new FakeNode(tag); },
};

const requests = [];
let failNextLaunch = false;
let savedVersion = 7;
let holdNextSave = false;
let releaseHeldSave = null;
let durableStatus = "RUNNING";
let durableResume = false;
global.FormData = class {
  constructor() {
    const backend = element("compute-type").value;
    this.rows = [["type", backend], ["run_profile", element("run-profile").value], ["preview_interval", "500"]];
    if (backend === "other") this.rows.push(["backend_id", element("other-backend").value]);
  }
  [Symbol.iterator]() { return this.rows[Symbol.iterator](); }
};
const jsonResponse = (body, ok = true) => ({
  ok,
  headers: {get: name => name.toLowerCase() === "content-type" ? "application/json" : ""},
  async json() { return body; },
  async text() { return ""; },
});

global.fetch = async (url, options = {}) => {
  const request = {url: String(url), options};
  requests.push(request);
  if (request.url === "/training/api/preparation") {
    return jsonResponse({status: "not_started", current: 0, total: 0, logs: []});
  }
  if (request.url.startsWith("/training/api/state?")) {
    return jsonResponse({
      status: "READY",
      blockers: [],
      data: {
        ready: true,
        dataset: {images: 2400, status: "READY"},
        model_label: "Recommended baseline",
        compute: element("compute-type").value,
        estimate: {duration_seconds: 60},
        availability_state: "READY",
      },
    });
  }
  if (request.url === "/training/api/settings" && request.options.method === "POST") {
    const body = JSON.parse(request.options.body);
    if (holdNextSave) {
      holdNextSave = false;
      await new Promise(resolve => { releaseHeldSave = resolve; });
    }
    savedVersion += 1;
    return jsonResponse({
      status: "saved",
      message: "saved",
      configuration_version: savedVersion,
      backend_identity: body.type === "other" ? body.backend_id : body.type,
    });
  }
  if (request.url === "/training/api/cloud-challenge") {
    const body = JSON.parse(request.options.body);
    return jsonResponse({status: "READY", data: {challenge: `challenge-${body.action}-${requests.length}`}});
  }
  if (request.url === "/training/api/runs/run-1" && !request.options.method) {
    return jsonResponse({status: durableStatus, data: {status: durableStatus, resume_available: durableResume, event_cursor: 4}});
  }
  if (request.url === "/training/api/start") {
    if (failNextLaunch) {
      failNextLaunch = false;
      return jsonResponse({message: "Synthetic launch refusal."}, false);
    }
    return jsonResponse({run: {run_id: "run-1"}, data: {}});
  }
  if (/\/training\/api\/runs\/[^/]+\/(pause|resume|cancel)$/.test(request.url)) {
    const action = request.url.split("/").at(-1);
    if (action === "resume" && failNextLaunch) {
      failNextLaunch = false;
      return jsonResponse({message: "Synthetic resume refusal."}, false);
    }
    durableStatus = action === "pause" ? "PAUSED" : action === "cancel" ? "CANCELLED" : "RUNNING";
    durableResume = action === "pause";
    return jsonResponse({
      message: `${action} complete`,
      data: {},
    });
  }
  throw new Error(`Unexpected request: ${request.url}`);
};

const eventSourceUrls = [];
global.EventSource = class {
  constructor(url) { eventSourceUrls.push(String(url)); }
  addEventListener() {}
  close() {}
};
global.setInterval = () => 1;
global.clearInterval = () => {};
"""


_BEHAVIOR_ASSERTIONS = r"""
const flush = async () => {
  await Promise.resolve();
  await new Promise(resolve => setImmediate(resolve));
};
const actionRequests = name => requests.filter(
  request => request.url === `/training/api/${name}` || request.url.endsWith(`/${name}`),
);
const payload = request => JSON.parse(request.options.body);

(async () => {
  await flush();
  await flush();

  const backend = element("compute-type");
  const profile = element("run-profile");
  const form = element("compute-settings");
  const confirmation = element("confirm-cloud");
  const confirmationPanel = element("cloud-confirmation");
  const start = element("start");
  const resume = element("resume");

  assert.strictEqual(confirmationPanel.hidden, false, "SSH must show cloud authorization");
  assert.strictEqual(start.disabled, true, "SSH Start must wait for confirmation");
  assert.strictEqual(requests.filter(request => request.url === "/training/api/runs/run-1").length, 1);
  assert.deepStrictEqual(eventSourceUrls, [], "passive page load must not start live replay or backend refresh");

  const guardedStartCount = actionRequests("start").length;
  await start.fire("click");
  assert.strictEqual(actionRequests("start").length, guardedStartCount);
  assert.match(element("blockers").textContent, /cloud-cost authorization/i);

  confirmation.checked = true;
  await confirmation.fire("change");
  assert.strictEqual(start.disabled, false);
  await start.fire("click");
  assert.strictEqual(payload(actionRequests("start").at(-1)).confirm_cloud, true);
  assert.strictEqual(payload(actionRequests("start").at(-1)).compute_configuration_version, 7);
  assert.strictEqual(payload(actionRequests("start").at(-1)).backend_identity, "ssh");
  assert.match(payload(actionRequests("start").at(-1)).cloud_challenge, /^challenge-start-/);
  assert.strictEqual(confirmation.checked, false, "successful Start must consume confirmation");
  assert.strictEqual(start.disabled, true);
  assert.strictEqual(eventSourceUrls.at(-1), "/api/runs/run-1/events?after=4");

  backend.value = "other";
  await backend.fire("change");
  await flush();
  assert.strictEqual(confirmationPanel.hidden, false, "provider plugins fail closed as cloud");
  confirmation.checked = true;
  await confirmation.fire("change");
  assert.strictEqual(start.disabled, true, "dirty provider settings must block Start");
  const dirtyStartCount = actionRequests("start").length;
  await start.fire("click");
  assert.strictEqual(actionRequests("start").length, dirtyStartCount);
  await element("save-compute").fire("click");
  await flush();
  confirmation.checked = true;
  await confirmation.fire("change");
  failNextLaunch = true;
  await start.fire("click");
  assert.strictEqual(payload(actionRequests("start").at(-1)).confirm_cloud, true);
  assert.strictEqual(confirmation.checked, false, "failed Start must consume confirmation");

  confirmation.checked = true;
  await confirmation.fire("change");
  profile.value = "quality";
  await profile.fire("change");
  assert.strictEqual(confirmation.checked, false, "profile changes invalidate confirmation");
  assert.strictEqual(start.disabled, true);
  await element("save-compute").fire("click");
  await flush();

  confirmation.checked = true;
  await confirmation.fire("change");
  await form.fire("input", element("ssh-host"));
  assert.strictEqual(confirmation.checked, false, "compute edits invalidate confirmation");
  assert.strictEqual(start.disabled, true, "any dirty compute field must block Start");

  backend.value = "local";
  await backend.fire("change");
  await flush();
  assert.strictEqual(confirmationPanel.hidden, true);
  assert.strictEqual(confirmation.disabled, true);
  assert.strictEqual(start.disabled, true, "dirty local settings must still block Start");
  holdNextSave = true;
  const pendingSave = element("save-compute").fire("click");
  await flush();
  profile.value = "fast_preview";
  await profile.fire("change");
  assert.strictEqual(typeof releaseHeldSave, "function");
  releaseHeldSave();
  await pendingSave;
  await flush();
  assert.strictEqual(start.disabled, true, "edits made while Save is in flight must remain dirty");
  assert.match(element("compute-status").textContent, /form changed while saving/i);
  await element("save-compute").fire("click");
  await flush();
  assert.strictEqual(start.disabled, false, "saved local Start requires no cloud confirmation");
  await start.fire("click");
  assert.strictEqual(payload(actionRequests("start").at(-1)).confirm_cloud, false);
  assert.strictEqual(payload(actionRequests("start").at(-1)).backend_identity, "local");

  backend.value = "runpod";
  await backend.fire("change");
  await flush();
  assert.strictEqual(start.disabled, true);
  assert.strictEqual(confirmationPanel.hidden, true);
  assert.strictEqual(element("save-compute").disabled, true);
  const unavailableStartCount = actionRequests("start").length;
  await start.fire("click");
  assert.strictEqual(actionRequests("start").length, unavailableStartCount);

  backend.value = "ssh";
  await backend.fire("change");
  await flush();
  await element("save-compute").fire("click");
  await flush();
  element("pause").disabled = false;
  const dashboardLoadsBeforePause = requests.filter(request => request.url === "/training/api/runs/run-1").length;
  await element("pause").fire("click");
  await flush();
  assert.deepStrictEqual(payload(actionRequests("pause").at(-1)), {});
  assert.strictEqual(requests.filter(request => request.url === "/training/api/runs/run-1").length, dashboardLoadsBeforePause + 1);
  assert.strictEqual(element('query:[data-metric="status"]').textContent, "PAUSED");
  assert.strictEqual(resume.disabled, true, "cloud Resume must wait for confirmation");

  confirmation.checked = true;
  await confirmation.fire("change");
  assert.strictEqual(resume.disabled, false);
  const pauseCount = actionRequests("pause").length;
  await element("pause").fire("click");
  await flush();
  assert.deepStrictEqual(payload(actionRequests("pause").at(-1)), {});
  assert.strictEqual(actionRequests("pause").length, pauseCount + 1);
  assert.strictEqual(confirmation.checked, true, "Pause must not consume cloud authority");

  await resume.fire("click");
  await flush();
  assert.strictEqual(payload(actionRequests("resume").at(-1)).confirm_cloud, true);
  assert.strictEqual(payload(actionRequests("resume").at(-1)).backend_identity, "ssh");
  assert.match(payload(actionRequests("resume").at(-1)).cloud_challenge, /^challenge-resume-/);
  assert.strictEqual(element('query:[data-metric="status"]').textContent, "RUNNING");
  assert.strictEqual(confirmation.checked, false, "successful Resume must consume confirmation");

  confirmation.checked = true;
  await confirmation.fire("change");
  await element("pause").fire("click");
  await flush();
  confirmation.checked = true;
  await confirmation.fire("change");
  failNextLaunch = true;
  await resume.fire("click");
  await flush();
  assert.strictEqual(payload(actionRequests("resume").at(-1)).confirm_cloud, true);
  assert.strictEqual(confirmation.checked, false, "failed Resume must consume confirmation");

  confirmation.checked = true;
  await confirmation.fire("change");
  await element("cancel").fire("click");
  await flush();
  assert.deepStrictEqual(payload(actionRequests("cancel").at(-1)), {});
  assert.strictEqual(element('query:[data-metric="status"]').textContent, "CANCELLED");
  assert.strictEqual(confirmation.checked, true, "Cancel must not consume cloud authority");

  console.log("training cloud launch controls: ok");
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_cloud_start_and_resume_controls_fail_closed_in_javascript() -> None:
    javascript = JAVASCRIPT.read_text(encoding="utf-8")
    script = "\n".join((_DOM_HARNESS, javascript, _BEHAVIOR_ASSERTIONS))

    completed = subprocess.run(
        [NODE],
        input=script.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stdout.decode("utf-8", errors="replace") + completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert b"training cloud launch controls: ok" in completed.stdout
