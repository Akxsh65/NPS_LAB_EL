import { DEFENSES, DEFENSE_ORDER, ARCHITECTURE, CHANNEL_ABLATION } from "./data.js";

let charts = {};

export function initCharts() {
  initAccuracyChart();
  initParetoChart();
  initArchitectureChart();
  initAblationChart();
}

function initAccuracyChart() {
  const ctx = document.getElementById("chart-accuracy");
  if (!ctx || typeof Chart === "undefined") return;

  const labels = DEFENSE_ORDER.map((k) => DEFENSES[k].short);
  const acc = DEFENSE_ORDER.map((k) => DEFENSES[k].accuracy);
  const f1 = DEFENSE_ORDER.map((k) => DEFENSES[k].macroF1);
  const colors = DEFENSE_ORDER.map((k) => DEFENSES[k].color);

  charts.accuracy = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Accuracy %",
          data: acc,
          backgroundColor: colors.map((c) => c + "cc"),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: "Macro F1 %",
          data: f1,
          backgroundColor: colors.map((c) => c + "55"),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    },
    options: chartOptions("Privacy vs. defense setting"),
  });
}

function initParetoChart() {
  const ctx = document.getElementById("chart-pareto");
  if (!ctx || typeof Chart === "undefined") return;

  const points = DEFENSE_ORDER.filter((k) => k !== "baseline").map((k) => {
    const d = DEFENSES[k];
    return {
      x: d.latMs > 0 ? d.latMs : d.bwPct + 0.5,
      y: d.accuracy,
      label: d.short,
      color: d.color,
    };
  });

  charts.pareto = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Defense settings",
          data: points,
          pointBackgroundColor: points.map((p) => p.color),
          pointRadius: 10,
        },
        {
          label: "Baseline",
          data: [{ x: 0.5, y: DEFENSES.baseline.accuracy }],
          pointBackgroundColor: "#94a3b8",
          pointRadius: 12,
          pointStyle: "star",
        },
        {
          label: "Chance",
          data: [{ x: 0.5, y: 1.56 }],
          pointBackgroundColor: "#ef4444",
          pointRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
        tooltip: {
          callbacks: {
            label(ctx) {
              const p = ctx.raw;
              if (ctx.datasetIndex === 1) return `Baseline: ${p.y}%`;
              if (ctx.datasetIndex === 2) return "Chance: 1.56%";
              return `${points[ctx.dataIndex].label}: ${p.y.toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Cost (latency ms or BW %)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.1)" },
        },
        y: {
          title: { display: true, text: "Accuracy %", color: "#94a3b8" },
          min: 0,
          max: 85,
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.1)" },
        },
      },
    },
  });
}

function initArchitectureChart() {
  const ctx = document.getElementById("chart-architecture");
  if (!ctx || typeof Chart === "undefined") return;

  charts.arch = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ARCHITECTURE.map((a) => a.setting.replace(/_/g, " ")),
      datasets: [
        {
          label: "Transformer",
          data: ARCHITECTURE.map((a) => a.transformer),
          backgroundColor: "rgba(34,211,238,0.75)",
          borderRadius: 6,
        },
        {
          label: "CNN-BiLSTM",
          data: ARCHITECTURE.map((a) => a.bilstm),
          backgroundColor: "rgba(167,139,250,0.75)",
          borderRadius: 6,
        },
      ],
    },
    options: chartOptions("Architecture robustness"),
  });
}

function initAblationChart() {
  const ctx = document.getElementById("chart-ablation");
  if (!ctx || typeof Chart === "undefined") return;

  charts.ablation = new Chart(ctx, {
    type: "bar",
    data: {
      labels: CHANNEL_ABLATION.map((c) => c.channels),
      datasets: [
        {
          label: "Baseline",
          data: CHANNEL_ABLATION.map((c) => c.baseline),
          backgroundColor: "rgba(148,163,184,0.7)",
          borderRadius: 4,
        },
        {
          label: "jitter_low",
          data: CHANNEL_ABLATION.map((c) => c.jitterLow),
          backgroundColor: "rgba(34,211,238,0.7)",
          borderRadius: 4,
        },
      ],
    },
    options: { indexAxis: "y", ...chartOptions("Channel ablation") },
  });
}

function chartOptions(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#cbd5e1", boxWidth: 12 } },
      title: { display: true, text: title, color: "#e2e8f0", font: { size: 14 } },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", maxRotation: 45 },
        grid: { color: "rgba(148,163,184,0.08)" },
      },
      y: {
        min: 0,
        max: 85,
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(148,163,184,0.08)" },
      },
    },
  };
}
