/* Log Sentinel - Client Dashboard JavaScript */

let totalLogs = 0;
let anomalyLogs = 0;
let isSimulating = true;
let eventSource = null;
let anomalyChart = null;

// Initialize Chart.js Anomaly Timeline Graph
function initChart() {
  const ctx = document.getElementById('anomaly-chart').getContext('2d');
  anomalyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Anomaly Decision Score',
        data: [],
        borderColor: '#00f2fe',
        borderWidth: 2,
        pointBackgroundColor: [],
        pointRadius: 4,
        tension: 0.3,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { size: 10 } },
          suggestedMin: -0.15,
          suggestedMax: 0.15
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#00f2fe',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      }
    }
  });
}

// Connect to Server-Sent Events (SSE) Stream
function connectStream() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource('/api/stream');

  eventSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      processLogStreamEntry(data);
    } catch (e) {
      console.error("Failed to parse SSE payload:", e);
    }
  };

  eventSource.onerror = function() {
    console.warn("SSE connection interrupted. Reconnecting...");
  };
}

// Process streamed log line entry from SSE
function processLogStreamEntry(data) {
  totalLogs++;
  const isAnomaly = data.prediction === -1;

  if (isAnomaly) {
    anomalyLogs++;
  }

  // Update KPI counters
  document.getElementById('kpi-total').innerText = totalLogs.toLocaleString();
  document.getElementById('kpi-anomalies').innerText = anomalyLogs.toLocaleString();
  const rate = totalLogs > 0 ? ((anomalyLogs / totalLogs) * 100).toFixed(2) : '0.00';
  document.getElementById('kpi-rate').innerText = `${rate}%`;

  // Append entry to terminal window
  const terminal = document.getElementById('terminal-window');
  const entryDiv = document.createElement('div');
  const levelClass = isAnomaly ? 'level-ANOMALY' : `level-${data.parsed.log_level}`;
  const badgeClass = isAnomaly ? 'badge-anomaly' : `badge-${data.parsed.log_level.toLowerCase()}`;
  const badgeLabel = isAnomaly ? '🚨 ANOMALY' : data.parsed.log_level;

  entryDiv.className = `log-entry ${levelClass}`;
  entryDiv.innerHTML = `
    <span class="log-ts">[${data.parsed.timestamp.split(' ')[1] || data.parsed.timestamp}]</span>
    <span class="badge ${badgeClass}">${badgeLabel}</span>
    <span class="log-msg">${escapeHtml(data.raw_line)}</span>
  `;

  terminal.appendChild(entryDiv);

  // Auto-scroll terminal if near bottom
  if (terminal.scrollHeight - terminal.scrollTop < 600) {
    terminal.scrollTop = terminal.scrollHeight;
  }

  // Limit terminal history to 300 entries
  if (terminal.children.length > 300) {
    terminal.removeChild(terminal.firstChild);
  }

  // Update Chart Timeline
  updateChart(data.parsed.timestamp.split(' ')[1] || data.parsed.timestamp, data.score, isAnomaly);
}

// Push data point to Chart.js timeline
function updateChart(timestamp, score, isAnomaly) {
  if (!anomalyChart) return;

  const labels = anomalyChart.data.labels;
  const dataset = anomalyChart.data.datasets[0];

  labels.push(timestamp);
  dataset.data.push(score);
  dataset.pointBackgroundColor.push(isAnomaly ? '#ff3366' : '#00f2fe');

  if (labels.length > 25) {
    labels.shift();
    dataset.data.shift();
    dataset.pointBackgroundColor.shift();
  }

  anomalyChart.update('none'); // Update smoothly without full animation reset
}

// Toggle Live Stream Simulator
async function toggleSimulator() {
  try {
    const res = await fetch('/api/simulator/toggle', { method: 'POST' });
    const data = await res.json();
    isSimulating = data.running;

    const btn = document.getElementById('toggle-sim-btn');
    const badge = document.getElementById('status-badge');
    const badgeText = document.getElementById('status-text');

    if (isSimulating) {
      btn.innerHTML = '<span>⏸️ Pause Stream</span>';
      badge.className = 'status-badge';
      badgeText.innerText = 'Streaming Active';
    } else {
      btn.innerHTML = '<span>▶️ Resume Stream</span>';
      badge.className = 'status-badge stopped';
      badgeText.innerText = 'Stream Paused';
    }
  } catch (e) {
    console.error("Failed to toggle simulator:", e);
  }
}

// Evaluate Anomaly Sandbox Custom Entry
async function evaluateSandbox() {
  const input = document.getElementById('sandbox-input').value.trim();
  if (!input) return;

  const resultBox = document.getElementById('sandbox-result');
  resultBox.style.display = 'block';
  resultBox.className = 'result-box';
  resultBox.innerHTML = '<span style="color: var(--text-muted);">Running ML inference pipeline...</span>';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log_line: input })
    });

    const data = await res.json();
    const isAnomaly = data.prediction === -1;

    resultBox.className = `result-box ${isAnomaly ? 'is-anomaly' : 'is-normal'}`;
    resultBox.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="font-weight: 700; font-size: 1rem;">
          ${isAnomaly ? '🚨 ANOMALY DETECTED' : '✅ NORMAL LOG ENTRY'}
        </span>
        <span class="badge ${isAnomaly ? 'badge-anomaly' : 'badge-info'}">Score: ${data.score.toFixed(4)}</span>
      </div>
      <div style="font-size: 0.825rem; color: var(--text-muted); display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
        <div>Status Code: <strong style="color: var(--text-main);">${data.parsed.status_code}</strong></div>
        <div>Log Level: <strong style="color: var(--text-main);">${data.parsed.log_level}</strong></div>
        <div>IP Address: <strong style="color: var(--text-main);">${data.parsed.ip}</strong></div>
        <div>Bytes Sent: <strong style="color: var(--text-main);">${data.parsed.bytes_sent}</strong></div>
      </div>
    `;
  } catch (e) {
    resultBox.innerHTML = `<span style="color: var(--danger-red);">Inference Error: ${e.message}</span>`;
  }
}

// Modal Handlers
function openModal(id) {
  document.getElementById(id).classList.add('active');
}

function closeModal(id) {
  document.getElementById(id).classList.remove('active');
}

// Retrain Model Form Submit
async function handleRetrain(e) {
  e.preventDefault();
  const submitBtn = document.getElementById('retrain-submit-btn');
  submitBtn.disabled = true;
  submitBtn.innerText = 'Training Model...';

  const contamination = parseFloat(document.getElementById('train-contamination').value);
  const n_estimators = parseInt(document.getElementById('train-trees').value);
  const num_samples = parseInt(document.getElementById('train-samples').value);

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contamination, n_estimators, num_samples })
    });
    const data = await res.json();
    alert(`Model retrained successfully!\n\nNormal entries: ${data.summary.normal_entries}\nAnomalies: ${data.summary.detected_anomalies} (${(data.summary.detected_anomalies/num_samples*100).toFixed(1)}%)`);
    closeModal('retrain-modal');
  } catch (err) {
    alert(`Training failed: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerText = '🚀 Fit & Save Model';
  }
}

// Slack Config Save & Test
async function saveSlackConfig() {
  const webhook_url = document.getElementById('slack-webhook').value.trim();
  const cooldown_seconds = parseInt(document.getElementById('slack-cooldown').value);

  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slack: { webhook_url, cooldown_seconds } })
    });
    alert("Slack settings saved!");
    closeModal('slack-modal');
  } catch (err) {
    alert("Failed to save configuration.");
  }
}

async function testSlackWebhook() {
  try {
    const res = await fetch('/api/slack/test', { method: 'POST' });
    const data = await res.json();
    alert(data.message || (data.success ? "Test alert sent!" : "Slack test failed."));
  } catch (err) {
    alert("Slack test request failed.");
  }
}

function clearTerminal() {
  document.getElementById('terminal-window').innerHTML = '';
}

function filterLogs() {
  const query = document.getElementById('log-search').value.toLowerCase();
  const entries = document.querySelectorAll('.log-entry');
  entries.forEach(entry => {
    const text = entry.innerText.toLowerCase();
    entry.style.display = text.includes(query) ? 'flex' : 'none';
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Auto init on page load
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  connectStream();
});
