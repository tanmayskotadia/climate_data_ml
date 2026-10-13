/**
 * AI-Based Climate Data Quality Enhancement
 * Interactive Dashboard Logic & Canvas Time-Series Visualizer
 * Authors: Tanmay Suresh Kotadia (24BCE5281) & Aditya Sandhwar (24BCE5253)
 */

let currentExperimentData = null;

window.addEventListener("DOMContentLoaded", () => {
  window.addEventListener("resize", () => {
    if (currentExperimentData) renderChart(currentExperimentData);
  });
  runExperiment();
});

function updateVariableOptions() {
  // Real datasets support all standard variables
}

async function runExperiment() {
  const runBtn = document.getElementById("run-btn");
  const btnText = document.getElementById("btn-text");
  runBtn.disabled = true;
  btnText.innerText = "Training & Reconstructing...";

  const dataset = document.getElementById("dataset-select").value;
  const variable = document.getElementById("variable-select").value;
  const scenario = document.getElementById("scenario-select").value;
  const model = document.getElementById("model-select").value;

  document.getElementById("chart-subheading").innerText = 
    `${dataset} | ${formatVarName(variable)} | Scenario: ${formatScenarioName(scenario)}`;

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: dataset,
        variable: variable,
        scenario: scenario,
        model: model,
        limit: 144
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    currentExperimentData = data;

    // 1. Update Objective 1 Diagnostics
    updateDiagnostics(data.detection_summary);

    // 2. Render Canvas Chart
    renderChart(data);

    // 3. Update Benchmark Metrics Table
    renderMetricsTable(data.models);

  } catch (err) {
    console.error("Experiment run failed:", err);
    alert("Reconstruction failed: " + err.message);
  } finally {
    runBtn.disabled = false;
    btnText.innerText = "Execute AI Reconstruction";
  }
}

function updateDiagnostics(diag) {
  if (!diag) return;
  document.getElementById("diag-missing").innerText = diag.missing_count || 0;
  document.getElementById("diag-phys").innerText = diag.physical_violations || 0;
  document.getElementById("diag-freeze").innerText = diag.sensor_freezes || 0;
  document.getElementById("diag-spikes").innerText = diag.spikes || 0;
}

function renderMetricsTable(models) {
  const tbody = document.getElementById("metrics-tbody");
  tbody.innerHTML = "";

  const modelKeys = Object.keys(models);
  if (modelKeys.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;">No model results.</td></tr>`;
    return;
  }

  // Find model with best MAE (strictly lowest MAE)
  let bestKey = "transformer" in models ? "transformer" : modelKeys[0];
  let minMae = models[bestKey] ? models[bestKey].metrics.MAE : 999999;
  for (const k of modelKeys) {
    const mae = models[k].metrics.MAE;
    if (mae < minMae) {
      minMae = mae;
      bestKey = k;
    }
  }

  const modelMeta = {
    "transformer": { name: "SpatioTemporal Transformer (Ours)", type: "Self-Attention (Proposed)" },
    "diffusion": { name: "Conditional Diffusion (CoFILL)", type: "Score-Based Diffusion (IJCAI '25)" },
    "lstm": { name: "Bidirectional LSTM", type: "Recurrent Deep Learning" },
    "gru": { name: "Bidirectional GRU", type: "Recurrent Deep Learning" },
    "pchip": { name: "PCHIP Spline", type: "Hermite Polynomial" },
    "linear": { name: "Linear Interpolation", type: "Classical Baseline" }
  };

  // Reorder so proposed transformer is at the top of the table
  const orderedKeys = [];
  if (modelKeys.includes("transformer")) orderedKeys.push("transformer");
  for (const k of ["diffusion", "lstm", "gru", "pchip", "linear"]) {
    if (modelKeys.includes(k) && !orderedKeys.includes(k)) orderedKeys.push(k);
  }
  for (const k of modelKeys) {
    if (!orderedKeys.includes(k)) orderedKeys.push(k);
  }

  for (const k of orderedKeys) {
    const item = models[k];
    const m = item.metrics;
    const meta = modelMeta[k] || { name: k, type: "Model" };
    const isBest = (k === bestKey);

    const tr = document.createElement("tr");
    if (isBest) tr.className = "highlight-row";

    const picpText = m.PICP !== undefined ? `${m.PICP}%` : `<span style="color:#64748b">—</span>`;
    const mpiwText = m.MPIW !== undefined ? `${m.MPIW}` : `<span style="color:#64748b">—</span>`;
    const statusBadge = isBest 
      ? `<span class="badge-best">★ State-of-the-Art (Lowest MAE)</span>`
      : `<span class="badge-baseline">Baseline</span>`;

    tr.innerHTML = `
      <td style="font-weight:600; color:#f8fafc;">${meta.name}</td>
      <td style="color:#94a3b8; font-family:var(--font-main);">${meta.type}</td>
      <td style="color:${isBest ? '#34d399' : '#f8fafc'}; font-weight:${isBest ? '700' : '400'}; font-size:${isBest ? '0.95rem' : '0.85rem'};">${m.MAE.toFixed(3)}</td>
      <td>${m.RMSE.toFixed(3)}</td>
      <td style="color:#a5b4fc;">${m.R2 !== undefined ? m.R2.toFixed(4) : '-'}</td>
      <td style="color:#38bdf8;">${picpText}</td>
      <td style="color:#cbd5e1;">${mpiwText}</td>
      <td style="color:#94a3b8;">${m.Latency_ms.toFixed(1)}ms</td>
      <td>${statusBadge}</td>
    `;
    tbody.appendChild(tr);
  }
}

// Canvas Visualizer
function renderChart(data) {
  const canvas = document.getElementById("reconstructionChart");
  const ctx = canvas.getContext("2d");
  const showUncertainty = document.getElementById("uncertainty-toggle").checked;

  const rect = canvas.getBoundingClientRect();
  const displayW = rect.width || 900;
  const displayH = 380;
  
  canvas.width = displayW * window.devicePixelRatio;
  canvas.height = displayH * window.devicePixelRatio;
  ctx.resetTransform();
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const padLeft = 60;
  const padRight = 30;
  const padTop = 25;
  const padBottom = 45;

  const chartW = displayW - padLeft - padRight;
  const chartH = displayH - padTop - padBottom;

  ctx.clearRect(0, 0, displayW, displayH);

  // Background subtle fill
  ctx.fillStyle = "rgba(10, 16, 29, 0.95)";
  ctx.fillRect(0, 0, displayW, displayH);

  const gt = data.ground_truth;
  const n = gt.length;
  if (n === 0) return;

  let allVals = [...gt];
  for (const mKey in data.models) {
    if (data.models[mKey].reconstructed) {
      allVals.push(...data.models[mKey].reconstructed);
    }
  }
  const minVal = Math.floor(Math.min(...allVals) - 1.0);
  const maxVal = Math.ceil(Math.max(...allVals) + 1.0);
  const rangeVal = maxVal - minVal || 1.0;

  function toX(i) {
    return padLeft + (i / (n - 1)) * chartW;
  }
  function toY(val) {
    return padTop + chartH - ((val - minVal) / rangeVal) * chartH;
  }

  // Draw Horizontal Gridlines & Y-Axis
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";

  const yTicks = 6;
  for (let t = 0; t <= yTicks; t++) {
    const val = minVal + (t / yTicks) * rangeVal;
    const y = toY(val);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(displayW - padRight, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), padLeft - 10, y + 4);
  }

  // 1. Shaded Uncertainty Band for Proposed Transformer
  if (showUncertainty && data.models.transformer && data.models.transformer.uncertainty) {
    const unc = data.models.transformer.uncertainty;
    const lower = unc.lower_ci;
    const upper = unc.upper_ci;

    ctx.fillStyle = "rgba(168, 85, 247, 0.22)";
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i);
      const y = toY(upper[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i--) {
      const x = toX(i);
      const y = toY(lower[i]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  // 2. Ground Truth Line (Crisp White)
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = toX(i);
    const y = toY(gt[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 3. Baselines
  const modelColors = {
    "linear": { stroke: "#94a3b8", dash: [4, 4], width: 1.5 },
    "pchip": { stroke: "#f59e0b", dash: [6, 3], width: 1.8 },
    "lstm": { stroke: "#3b82f6", dash: [4, 2], width: 2.0 },
    "gru": { stroke: "#10b981", dash: [2, 2], width: 2.0 },
    "diffusion": { stroke: "#ec4899", dash: [3, 3], width: 2.2 }
  };

  for (const mKey of ["linear", "pchip", "lstm", "gru", "diffusion"]) {
    if (!data.models[mKey]) continue;
    const conf = modelColors[mKey];
    const series = data.models[mKey].reconstructed;

    ctx.strokeStyle = conf.stroke;
    ctx.lineWidth = conf.width;
    ctx.setLineDash(conf.dash);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i);
      const y = toY(series[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // 4. Proposed Transformer Line (Electric Violet - Prominent & Best)
  if (data.models.transformer) {
    const series = data.models.transformer.reconstructed;
    ctx.strokeStyle = "#c084fc";
    ctx.lineWidth = 3.0;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i);
      const y = toY(series[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.setLineDash([]);

  // 5. Highlight Missing Gap Observations (Glowing Red Markers)
  const mask = data.missing_mask;
  ctx.fillStyle = "#ef4444";
  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      const x = toX(i);
      const y = toY(gt[i]);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // X-Axis Labels
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "center";
  const xTicks = 6;
  for (let t = 0; t <= xTicks; t++) {
    const idx = Math.floor((t / xTicks) * (n - 1));
    const x = toX(idx);
    ctx.fillText(`t=${idx}`, x, displayH - 14);
  }
}

// Tooltip interaction
const canvas = document.getElementById("reconstructionChart");
const tooltip = document.getElementById("chart-tooltip");

canvas.addEventListener("mousemove", (e) => {
  if (!currentExperimentData) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const padLeft = 60;
  const padRight = 30;
  const chartW = rect.width - padLeft - padRight;
  const n = currentExperimentData.ground_truth.length;

  if (mouseX < padLeft || mouseX > rect.width - padRight) {
    tooltip.classList.add("hidden");
    return;
  }

  const frac = (mouseX - padLeft) / chartW;
  const idx = Math.min(n - 1, Math.max(0, Math.round(frac * (n - 1))));

  const gtVal = currentExperimentData.ground_truth[idx];
  const isMissing = currentExperimentData.missing_mask[idx];
  const ts = currentExperimentData.timestamps[idx] || `t=${idx}`;

  let content = `<strong>${ts}</strong><br/>`;
  content += `Ground Truth: ${gtVal}<br/>`;
  if (isMissing) {
    content += `<span style="color:#ef4444; font-weight:600;">[Status: Missing Observation]</span><br/>`;
  }
  if (currentExperimentData.models.transformer) {
    const transVal = currentExperimentData.models.transformer.reconstructed[idx];
    content += `<span style="color:#c084fc; font-weight:600;">Transformer (Ours): ${transVal}</span><br/>`;
  }
  if (currentExperimentData.models.diffusion) {
    const diffVal = currentExperimentData.models.diffusion.reconstructed[idx];
    content += `<span style="color:#ec4899;">Diffusion (CoFILL): ${diffVal}</span><br/>`;
  }
  if (currentExperimentData.models.lstm) {
    const lstmVal = currentExperimentData.models.lstm.reconstructed[idx];
    content += `<span style="color:#3b82f6;">Bi-LSTM: ${lstmVal}</span><br/>`;
  }

  tooltip.innerHTML = content;
  tooltip.style.left = `${e.clientX - rect.left + 15}px`;
  tooltip.style.top = `${e.clientY - rect.top + 10}px`;
  tooltip.classList.remove("hidden");
});

canvas.addEventListener("mouseleave", () => {
  tooltip.classList.add("hidden");
});

function exportResultsCSV() {
  if (!currentExperimentData) return;
  const models = currentExperimentData.models;
  let csv = "Model,MAE,RMSE,R2,Coverage_PICP,Interval_Width_MPIW,Latency_ms\n";
  for (const k in models) {
    const m = models[k].metrics;
    csv += `"${models[k].name}",${m.MAE},${m.RMSE},${m.R2 || ""},${m.PICP || ""},${m.MPIW || ""},${m.Latency_ms}\n`;
  }

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `climate_benchmark_results_${currentExperimentData.dataset}_${currentExperimentData.scenario}.csv`;
  a.click();
}

function formatVarName(v) {
  const map = {
    "temperature": "Temperature (°C)",
    "relative_humidity": "Relative Humidity (%)",
    "wind_speed": "Wind Speed (m/s)",
    "precipitation": "Precipitation (mm)"
  };
  return map[v] || v;
}

function formatScenarioName(s) {
  const map = {
    "block_24h": "24h Outage Block",
    "random_10": "10% Random Missing",
    "random_20": "20% Random Missing",
    "random_40": "40% Random Missing",
    "sensor_freeze": "Sensor Freeze",
    "sensor_spikes": "Sensor Spikes",
    "compound": "Compound Degradation"
  };
  return map[s] || s;
}
