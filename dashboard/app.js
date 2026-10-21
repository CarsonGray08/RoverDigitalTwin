"use strict";

const $ = (id) => document.getElementById(id);
const STORE = { "rover-ip": "rt_rover_ip", "level": "rt_level", "drive-command": "rt_command", "drive-amount": "rt_amount" };
const MAP_REFRESH_MS = 2000;          // at most this often, and only when the map changed
const POSE_STALE_S = 1.0;             // the drive tool's own limit

let rover = null;                     // /health's rover: footprint and units
let grid = null;                      // the last /map, drawn into an offscreen canvas
let mapVersion = null, mapFetchedAt = 0, mapLoading = false;
let status = null;

/* ---------- helpers ---------- */

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* not JSON */ }
  if (!response.ok) throw new Error(body && body.error ? body.error : `HTTP ${response.status}`);
  return body;
}

const post = (path, payload) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}),
});

function toast(message, kind = "info", ms = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function withButton(button, fn) {
  button.disabled = true;
  try { await fn(); } catch (err) { toast(err.message, "err"); } finally { button.disabled = false; }
}

const fmt = (v, digits = 2) => (typeof v === "number" && isFinite(v)) ? v.toFixed(digits) : "--";

function setLog(el, lines) {
  const text = (lines || []).join("\n");
  if (el.textContent === text) return;
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 6;
  el.textContent = text;
  if (atBottom) el.scrollTop = el.scrollHeight;
}

const armed = () => $("arm-toggle").checked;
const motionBody = (extra) => ({ arm: armed(), rover_ip: $("rover-ip").value.trim(), level: Number($("level").value), ...extra });

/* ---------- map ---------- */

// Drawn as seen from above with the start heading up: twin +x is up, +y is left.
function toPixel(x, y) {
  return [grid.ny - 1 - (y - grid.oy) / grid.cell, grid.nx - 1 - (x - grid.ox) / grid.cell];
}

function buildGrid(m) {
  // Only the part that has been seen, with a margin.
  let i0 = Infinity, i1 = -1, j0 = Infinity, j1 = -1;
  m.state.forEach((row, i) => row.forEach((state, j) => {
    if (state) { i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j); }
  }));
  if (i1 < 0) return null;
  const pad = 10;
  i0 = Math.max(0, i0 - pad); j0 = Math.max(0, j0 - pad);
  i1 = Math.min(m.shape[0] - 1, i1 + pad); j1 = Math.min(m.shape[1] - 1, j1 + pad);
  const nx = i1 - i0 + 1, ny = j1 - j0 + 1;
  const canvas = document.createElement("canvas");
  canvas.width = ny; canvas.height = nx;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(ny, nx);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const k = ((nx - 1 - i) * ny + (ny - 1 - j)) * 4;
      const state = m.state[i0 + i][j0 + j], h = m.height_m[i0 + i][j0 + j];
      let c = [22, 26, 33];                                   // not seen
      if (m.obstacle[i0 + i][j0 + j]) c = [255, 93, 93];
      else if (state === 2) c = [92, 118, 168];
      else if (state === 1) { const v = Math.max(60, Math.min(210, 150 + h * 900)); c = [v, v, v]; }
      image.data.set([c[0], c[1], c[2], 255], k);
    }
  }
  ctx.putImageData(image, 0, 0);
  return { canvas, nx, ny, cell: m.cell_m, ox: m.origin_m[0] + i0 * m.cell_m, oy: m.origin_m[1] + j0 * m.cell_m };
}

async function refreshMap(version) {
  if (mapLoading || (version === mapVersion && grid)) return;
  if (Date.now() - mapFetchedAt < MAP_REFRESH_MS) return;
  mapLoading = true;
  try {
    const m = await api("/api/map");
    mapFetchedAt = Date.now();
    if (m.shape[0] > 0) { grid = buildGrid(m) || grid; mapVersion = version; }
  } catch (_) { /* the twin is down or busy; the next poll tries again */ }
  finally { mapLoading = false; }
}

function drawMap(live) {
  const canvas = $("map");
  $("map-empty").style.display = grid ? "none" : "flex";
  if (!grid) return;
  const scale = Math.min(canvas.parentElement.clientWidth / grid.ny, 0.7 * window.innerHeight / grid.nx);
  canvas.width = Math.round(grid.ny * scale);
  canvas.height = Math.round(grid.nx * scale);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(grid.canvas, 0, 0, canvas.width, canvas.height);
  const at = (x, y) => toPixel(x, y).map((p) => p * scale);
  const [sx, sy] = at(0, 0);                                  // where the map was anchored
  ctx.strokeStyle = "#f2b84b"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(sx - 6, sy); ctx.lineTo(sx + 6, sy); ctx.moveTo(sx, sy - 6); ctx.lineTo(sx, sy + 6); ctx.stroke();
  if (!live || !rover) return;
  const [hl, hw] = rover.footprint_half_m;
  const yaw = live.yaw_deg * Math.PI / 180, c = Math.cos(yaw), s = Math.sin(yaw);
  const corner = (u, v) => at(live.x + c * u - s * v, live.y + s * u + c * v);
  ctx.strokeStyle = ctx.fillStyle = "#46d17a"; ctx.lineWidth = 2;
  ctx.beginPath();
  [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].forEach(([u, v], n) => { const p = corner(u, v); n ? ctx.lineTo(...p) : ctx.moveTo(...p); });
  ctx.closePath(); ctx.stroke();
  const [cx, cy] = at(live.x, live.y), [fx, fy] = corner(hl + 0.15, 0);
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(fx, fy); ctx.stroke();
}

/* ---------- status ---------- */

const pill = (label, cls = "") => `<span class="pill ${cls}"><span class="dot"></span>${label}</span>`;
const stat = (k, v) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;

function render(s) {
  const obs = s.observation, live = obs && obs.live_pose;
  const fresh = live && live.age_s <= POSE_STALE_S;
  $("header-status").innerHTML = [
    s.twin_job.running ? pill("twin running", "ok") : pill(obs ? "twin up" : "twin stopped", obs ? "ok" : ""),
    obs ? (obs.mapper_connected ? pill("mapper connected", "ok") : pill("no mapper", "warn")) : "",
    obs ? (fresh ? pill("live pose", "ok") : pill(live ? "pose stale" : "no pose yet", "warn")) : "",
    s.motion_job.running ? pill(armed() ? "rover moving" : "rehearsing", "ok") : "",
    armed() ? pill("ARMED", "danger") : "",
  ].join("");
  $("stats-row").innerHTML = [
    stat("x (m)", live ? fmt(live.x) : "--"),
    stat("y (m)", live ? fmt(live.y) : "--"),
    stat("yaw (deg)", live ? fmt(live.yaw_deg, 1) : "--"),
    stat("pose age (s)", live ? fmt(live.age_s, 1) : "--"),
    stat("map rev", live ? live.rev : "--"),
    stat("tiles", obs ? obs.terrain_tiles : "--"),
  ].join("");
  $("twin-start").disabled = s.twin_job.running;
  $("twin-stop").disabled = !s.twin_job.running;
  $("scan-start").disabled = $("drive-start").disabled = s.motion_job.running || !obs;
  setLog($("twin-log"), s.twin_job.log);
  setLog($("motion-log"), s.motion_job.log);
  if (obs) refreshMap(live ? live.rev : obs.terrain_tiles);
  drawMap(live);
}

async function poll() {
  try {
    status = await api("/api/status");
    if (status.observation && !rover) rover = (await api("/api/health")).rover;
    if (!$("rover-ip").value && status.rover_ip) $("rover-ip").value = status.rover_ip;
    render(status);
  } catch (err) {
    $("header-status").innerHTML = pill(`dashboard unreachable: ${err.message}`, "warn");
  } finally {
    setTimeout(poll, 500);
  }
}

/* ---------- actions ---------- */

function wire() {
  $("twin-start").addEventListener("click", (e) => withButton(e.target, async () => {
    await post("/api/twin/start"); toast("Starting mapper and twin", "ok");
  }));
  $("twin-stop").addEventListener("click", (e) => withButton(e.target, async () => {
    await post("/api/twin/stop"); toast("Stopping mapper and twin", "ok");
  }));
  $("scan-start").addEventListener("click", (e) => withButton(e.target, async () => {
    await post("/api/motion/start", motionBody({ kind: "scan" }));
    toast(armed() ? "360 scan: the rover turns" : "360 scan rehearsed only", "ok");
  }));
  $("drive-start").addEventListener("click", (e) => withButton(e.target, async () => {
    await post("/api/motion/start", motionBody({ kind: "drive", command: $("drive-command").value,
                                                 amount: Number($("drive-amount").value) }));
    toast(armed() ? "Driving: rehearsed first, stopped on the live pose" : "Rehearsing only", "ok");
  }));
  $("motion-stop").addEventListener("click", (e) => withButton(e.target, async () => {
    await post("/api/motion/stop", { rover_ip: $("rover-ip").value.trim() }); toast("STOP sent", "ok");
  }));
  $("arm-toggle").addEventListener("change", () => {
    document.body.classList.toggle("armed", armed());
    $("arm-label").textContent = armed()
      ? "ARMED: scan and drive move the real rover"
      : "Rehearsal mode: nothing is sent to the rover";
    if (status) render(status);
  });
  for (const [id, key] of Object.entries(STORE)) {      // not the arm switch: every page load starts disarmed
    const saved = localStorage.getItem(key);
    if (saved !== null) $(id).value = saved;
    $(id).addEventListener("change", () => localStorage.setItem(key, $(id).value));
  }
  window.addEventListener("resize", () => status && drawMap(status.observation && status.observation.live_pose));
}

wire();
poll();
