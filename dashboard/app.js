"use strict";

const $ = (id) => document.getElementById(id);

// Persisted UI settings (deliberately NOT the arm switch: every page load
// starts disarmed so a stale tab can never silently drive the real rover).
const STORE_KEYS = { "rover-ip": "dt_rover_ip", "scan-degrees": "dt_degrees", "goal-distance": "dt_goal_distance" };

let paramSpec = null;
let lastStatus = null;

/* ---------- helpers ---------- */

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* non-JSON body */ }
  if (!response.ok) {
    throw new Error((body && body.error) ? body.error : `HTTP ${response.status}`);
  }
  return body;
}

function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

function toast(message, kind = "info", ms = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function withButton(button, fn) {
  button.disabled = true;
  try { await fn(); } finally { button.disabled = false; }
}

function fmtNum(value, digits = 2) {
  return (typeof value === "number" && isFinite(value)) ? value.toFixed(digits) : "--";
}

function setLog(el, lines) {
  const text = (lines || []).join("\n");
  if (el.textContent === text) return;
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 6;
  el.textContent = text;
  if (atBottom) el.scrollTop = el.scrollHeight;
}

function showFrame(imgId, blobUrl) {
  const img = $(imgId);
  const prev = img.dataset.blobUrl;
  img.src = blobUrl;
  img.dataset.blobUrl = blobUrl;
  img.closest(".feed-frame").classList.add("has-frame");
  if (prev) URL.revokeObjectURL(prev);
}

function roverConfig() {
  return { rover_ip: $("rover-ip").value.trim(), arm: $("arm-toggle").checked };
}

/* ---------- status poll ---------- */

function pill(label, cls) {
  return `<span class="pill ${cls}"><span class="dot"></span>${label}</span>`;
}

function renderHeader(status) {
  const twin = status.jobs.twin;
  const scan = status.jobs.scan360;
  const goal = status.jobs.drive_goal;
  const parts = [
    twin.running ? pill("twin running", "ok") : pill("twin stopped", ""),
    scan.running ? pill("360 scanning", "ok") : "",
    goal && goal.running ? pill("validated drive", "ok") : "",
    status.tracking_normal ? pill("phone tracking", "ok") : pill("no phone tracking", "warn"),
    $("arm-toggle").checked ? pill("ARMED", "danger") : "",
  ];
  $("header-status").innerHTML = parts.filter(Boolean).join("");
}

function stat(label, value, title) {
  return `<div class="stat" ${title ? `title="${title}"` : ""}><div class="k">${label}</div><div class="v">${value}</div></div>`;
}

function renderStats(status) {
  const pose = status.pose;
  const poseAge = pose && typeof pose.timestamp_s === "number"
    ? (status.server_time_s - pose.timestamp_s) : null;
  const session = status.session_id
    ? status.session_id.replace(/^(live-|session-)/, "") : "--";
  $("stats-row").innerHTML = [
    stat("x (m)", pose ? fmtNum(pose.x) : "--"),
    stat("y (m)", pose ? fmtNum(pose.y) : "--"),
    stat("yaw (deg)", pose ? fmtNum(pose.yaw_deg, 1) : "--"),
    stat("pose age (s)", poseAge !== null ? fmtNum(poseAge, 1) : "--"),
    stat("tiles", status.tile_count ?? "--"),
    stat("map rev", status.manifest_rev ?? "--"),
    stat("session", session, status.map_dir || ""),
  ].join("");
}

function renderButtons(status) {
  $("twin-start").disabled = status.jobs.twin.running;
  $("twin-stop").disabled = !status.jobs.twin.running;
  $("scan-start").disabled = status.jobs.scan360.running;
  $("scan-stop").disabled = !status.jobs.scan360.running;
  const goal = status.jobs.drive_goal;
  if (goal) {
    $("goal-start").disabled = goal.running || status.jobs.scan360.running;
    $("goal-stop").disabled = !goal.running;
  }
}

async function pollStatus() {
  try {
    const status = await api("/api/status");
    lastStatus = status;
    renderHeader(status);
    renderStats(status);
    renderButtons(status);
    setLog($("twin-log"), status.jobs.twin.log_tail);
    setLog($("scan-log"), status.jobs.scan360.log_tail);
    if (status.jobs.drive_goal) setLog($("goal-log"), status.jobs.drive_goal.log_tail);
  } catch (err) {
    $("header-status").innerHTML = pill(`dashboard unreachable: ${err.message}`, "warn");
  } finally {
    setTimeout(pollStatus, 2000);
  }
}

/* ---------- camera feeds ---------- */

async function pollPhoneCamera() {
  const ageEl = $("phone-frame-age");
  let delay = 700;
  try {
    const response = await fetch("/api/camera", { cache: "no-store" });
    if (response.ok) {
      const source = response.headers.get("X-Frame-Source");
      const stamp = Number(response.headers.get("X-Frame-Timestamp"));
      $("phone-feed-label").textContent =
        source === "depth" ? "iPhone camera (depth preview)" : "iPhone camera";
      if (isFinite(stamp) && stamp > 0) {
        const age = Date.now() / 1000 - stamp;
        ageEl.textContent = `${age.toFixed(1)}s old`;
        ageEl.className = "frame-age " + (age < 3 ? "fresh" : "stale");
      }
      showFrame("phone-frame", URL.createObjectURL(await response.blob()));
    } else {
      ageEl.textContent = "no frame yet";
      ageEl.className = "frame-age stale";
      delay = 2500;
    }
  } catch (_) {
    ageEl.textContent = "offline";
    ageEl.className = "frame-age stale";
    delay = 4000;
  } finally {
    setTimeout(pollPhoneCamera, delay);
  }
}

async function pollChaseCamera() {
  const ageEl = $("chase-frame-age");
  let delay = 1500;
  try {
    const obs = await api("/api/observation");
    const frame =
      (obs.vision_chase && obs.vision_chase.status === "ok" && obs.vision_chase.rgb_base64) ? obs.vision_chase :
      (obs.vision && obs.vision.status === "ok" && obs.vision.rgb_base64) ? obs.vision : null;
    if (frame) {
      const bytes = Uint8Array.from(atob(frame.rgb_base64), (c) => c.charCodeAt(0));
      showFrame("chase-frame", URL.createObjectURL(new Blob([bytes], { type: "image/png" })));
      ageEl.textContent = "live";
      ageEl.className = "frame-age fresh";
    }
  } catch (_) {
    ageEl.textContent = lastStatus && lastStatus.jobs.twin.running ? "waiting" : "";
    ageEl.className = "frame-age stale";
    delay = 4000;
  } finally {
    setTimeout(pollChaseCamera, delay);
  }
}

/* ---------- parameters ---------- */

function paramGroups(spec) {
  const groups = new Map();
  for (const [name, def] of Object.entries(spec)) {
    const group = def.group || "Other";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([name, def]);
  }
  return groups;
}

function renderParamsForm(spec, values) {
  const form = $("params-form");
  form.innerHTML = "";
  for (const [group, entries] of paramGroups(spec)) {
    const section = document.createElement("div");
    section.className = "param-group";
    section.innerHTML = `<h3>${group}</h3><div class="fields"></div>`;
    const fields = section.querySelector(".fields");
    for (const [name, def] of entries) {
      const value = values[name];
      const label = document.createElement("label");
      const title = def.help ? ` title="${def.help.replace(/"/g, "&quot;")}"` : "";
      const text = def.label || name;
      if (def.type === "bool") {
        label.className = "checkbox";
        label.innerHTML = `<input type="checkbox" data-param="${name}" ${value ? "checked" : ""}${title}/> ${text}`;
      } else if (def.type === "choice") {
        const options = def.choices.map((choice) =>
          `<option value="${choice}" ${String(choice) === String(value) ? "selected" : ""}>${choice}</option>`).join("");
        label.innerHTML = `<span${title}>${text}</span><select data-param="${name}">${options}</select>`;
      } else {
        const inputType = (def.type === "int" || def.type === "float") ? "number" : "text";
        const step = def.type === "float" ? ` step="any"` : def.type === "int" ? ` step="1"` : "";
        label.innerHTML = `<span${title}>${text}</span><input type="${inputType}" data-param="${name}" value="${value}"${step}/>`;
      }
      fields.appendChild(label);
    }
    form.appendChild(section);
  }
}

function collectParamOverrides() {
  const overrides = {};
  document.querySelectorAll("#params-form [data-param]").forEach((el) => {
    const name = el.dataset.param;
    overrides[name] = paramSpec[name].type === "bool" ? el.checked : el.value;
  });
  return overrides;
}

/* ---------- actions ---------- */

function wireActions() {
  $("twin-start").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/jobs/twin/start", { params: collectParamOverrides() });
      toast("Live twin launching (params saved)", "ok");
    } catch (err) { toast(`Launch failed: ${err.message}`, "err"); }
  }));

  $("twin-stop").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/jobs/twin/stop");
      toast("Stopping the live twin…", "ok");
    } catch (err) { toast(err.message, "err"); }
  }));

  $("scan-start").addEventListener("click", (event) => withButton(event.target, async () => {
    const config = roverConfig();
    try {
      await postJson("/api/jobs/scan360/start", {
        rover_ip: config.rover_ip,
        arm: config.arm,
        degrees: Number($("scan-degrees").value) || 360,
      });
      toast(config.arm ? "360 scan started (ARMED with odometry stop)" : "360 scan started (rehearsal)", "ok");
    } catch (err) { toast(`360 scan failed: ${err.message}`, "err"); }
  }));

  $("scan-stop").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/jobs/scan360/stop");
      toast("Stopping the 360 scan…", "ok");
    } catch (err) { toast(err.message, "err"); }
  }));

  $("goal-start").addEventListener("click", (event) => withButton(event.target, async () => {
    const config = roverConfig();
    try {
      await postJson("/api/jobs/drive_goal/start", {
        rover_ip: config.rover_ip,
        arm: config.arm,
        distance_m: Number($("goal-distance").value) || 1.0,
      });
      toast(config.arm
        ? "Validated drive started (ARMED) — approved commands go to the rover"
        : "Validated drive started (rehearsal) — twin only, nothing sent", "ok");
    } catch (err) { toast(`Validated drive failed: ${err.message}`, "err"); }
  }));

  $("goal-stop").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/jobs/drive_goal/stop");
      toast("Stopping the validated drive…", "ok");
    } catch (err) { toast(err.message, "err"); }
  }));

  $("tracking-start").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/tracking/start");
      toast("Tracking resumed — twin follows the rover (map not rewritten)", "ok");
    } catch (err) { toast(`Tracking failed: ${err.message}`, "err"); }
  }));

  $("tracking-stop").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/tracking/stop");
      toast("Tracking paused — phone scan stopped", "ok");
    } catch (err) { toast(err.message, "err"); }
  }));

  $("params-save").addEventListener("click", (event) => withButton(event.target, async () => {
    try {
      await postJson("/api/params", collectParamOverrides());
      toast("Parameters saved", "ok");
    } catch (err) { toast(`Save failed: ${err.message}`, "err"); }
  }));

  $("params-reset").addEventListener("click", () => {
    const defaults = Object.fromEntries(
      Object.entries(paramSpec).map(([name, def]) => [name, def.default]));
    renderParamsForm(paramSpec, defaults);
    toast("Defaults restored — press Save to apply", "info");
  });

  $("arm-toggle").addEventListener("change", () => {
    const armed = $("arm-toggle").checked;
    document.body.classList.toggle("armed", armed);
    $("arm-label").textContent = armed
      ? "ARMED — drive and scan commands move the real rover"
      : "Rehearsal mode — nothing is sent to the rover";
    if (lastStatus) renderHeader(lastStatus);
  });

  ["phone-frame", "chase-frame"].forEach((id) => {
    $(id).addEventListener("click", () => {
      const img = $(id);
      if (img.requestFullscreen) img.requestFullscreen();
    });
  });

  for (const [id, key] of Object.entries(STORE_KEYS)) {
    const saved = localStorage.getItem(key);
    if (saved !== null) $(id).value = saved;
    $(id).addEventListener("change", () => localStorage.setItem(key, $(id).value));
  }
}

/* ---------- boot ---------- */

async function main() {
  wireActions();
  try {
    const [spec, values] = await Promise.all([api("/api/param_spec"), api("/api/params")]);
    paramSpec = spec;
    renderParamsForm(spec, values);
  } catch (err) {
    toast(`Could not load parameters: ${err.message}`, "err");
  }
  pollStatus();
  pollPhoneCamera();
  pollChaseCamera();
}

main();
