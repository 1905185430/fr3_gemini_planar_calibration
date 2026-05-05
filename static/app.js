const videoFeed = document.getElementById("videoFeed");

function round2(value) {
  return Number(value.toFixed(2));
}

function formatValue(value) {
  if (typeof value === "number") return round2(value);
  if (Array.isArray(value)) return value.map(formatValue);
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = formatValue(v);
    return out;
  }
  return value;
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

async function apiPost(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return res.json();
}

function fmt(value) {
  if (value === null || value === undefined) return "-";
  const formatted = formatValue(value);
  if (Array.isArray(formatted)) return JSON.stringify(formatted);
  if (typeof formatted === "object") return JSON.stringify(formatted, null, 2);
  if (typeof formatted === "boolean") return formatted ? "是" : "否";
  if (typeof formatted === "number") return formatted.toFixed(2);
  if (formatted === "SAMPLE") return "采样模式";
  if (formatted === "VERIFY") return "验证模式";
  return String(value);
}

function renderState(state) {
  document.getElementById("modeText").textContent = state.verify_mode ? "验证模式" : "采样模式";
  document.getElementById("statusText").textContent = state.last_status || "-";
  document.getElementById("sampleCount").textContent = String(state.sample_count || 0);
  document.getElementById("selectedPixel").textContent = fmt(state.selected_pixel);
  document.getElementById("moveEnabled").textContent = state.move_enabled ? "是" : "否";
  document.getElementById("poseRefresh").textContent =
    state.pose_age_s === null || state.pose_age_s === undefined
      ? "-"
      : `${state.pose_age_s.toFixed(2)} 秒前`;
  document.getElementById("verifyRpy").textContent = fmt(state.verify_rpy_deg);
  document.getElementById("tcpPose").textContent = fmt(state.current_tcp_pose);
  document.getElementById("predictionText").textContent = fmt(state.prediction_data);
  document.getElementById("errorText").textContent = fmt(state.result_data?.error_stats);
  document.getElementById("btnMoveToggle").textContent = state.move_enabled ? "禁止运动" : "允许运动";
}

async function refreshState() {
  try {
    const state = await apiGet("/api/state");
    renderState(state);
  } catch (err) {
    document.getElementById("statusText").textContent = err.message;
  }
}

videoFeed.addEventListener("click", async (event) => {
  const rect = videoFeed.getBoundingClientRect();
  const scaleX = videoFeed.naturalWidth / rect.width;
  const scaleY = videoFeed.naturalHeight / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  try {
    const state = await apiPost("/api/click", { x, y });
    renderState(state);
  } catch (err) {
    document.getElementById("statusText").textContent = err.message;
  }
});

document.getElementById("btnSave").onclick = async () => renderState(await apiPost("/api/sample/save"));
document.getElementById("btnUndo").onclick = async () => renderState(await apiPost("/api/sample/undo"));
document.getElementById("btnCompute").onclick = async () => renderState(await apiPost("/api/calibration/compute"));
document.getElementById("btnVerify").onclick = async () => renderState(await apiPost("/api/mode/verify", { enabled: true }));
document.getElementById("btnSampleMode").onclick = async () => renderState(await apiPost("/api/mode/verify", { enabled: false }));
document.getElementById("btnMoveToggle").onclick = async () => {
  const enable = document.getElementById("moveEnabled").textContent !== "是";
  renderState(await apiPost("/api/move/enable", { enabled: enable }));
};
document.getElementById("btnGo").onclick = async () => renderState(await apiPost("/api/move/go"));
document.getElementById("btnAlign").onclick = async () => renderState(await apiPost("/api/robot/align"));

setInterval(refreshState, 500);
refreshState();
