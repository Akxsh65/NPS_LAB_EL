/** Embedded experiment results — sourced from phase4 CSVs */

export const PROJECT = {
  title: "QUIC Metadata Privacy",
  subtitle: "Deep Learning Traffic Classification vs. Deterministic Obfuscation",
  dataset: "CESNET-QUIC22-XS",
  testFlows: 49305,
  classes: 64,
  trainWeek: "W-2022-44",
  testWeek: "W-2022-45",
  chanceAccuracy: 1.56,
};

export const PIPELINE = [
  {
    phase: 1,
    title: "Data Engineering",
    desc: "Download QUIC backbone flows, normalize IPT / direction / size into (3×30) tensors with temporal train/test split.",
    icon: "📦",
  },
  {
    phase: 2,
    title: "Train Attackers",
    desc: "Train frozen classifiers: masked Transformer (d=160) and CNN-BiLSTM on clean metadata fingerprints.",
    icon: "🧠",
  },
  {
    phase: 3,
    title: "Obfuscate Metadata",
    desc: "Apply deterministic defenses on test flows — jitter timing, pad sizes to 128 B or MTU — log bandwidth/latency cost.",
    icon: "🛡️",
  },
  {
    phase: 4,
    title: "Evaluate & Compare",
    desc: "Measure privacy (macro F1 drop) vs. cost, Pareto frontiers, bootstrap CIs, and architecture robustness.",
    icon: "📊",
  },
];

export const DEFENSES = {
  baseline: {
    id: "baseline",
    label: "No defense",
    short: "Clean",
    padding: "none",
    jitterScale: 0,
    jitterKey: null,
    color: "#94a3b8",
    icon: "○",
    mechanism:
      "Raw QUIC metadata passes through unchanged. Packet sizes, directions, and inter-packet times reveal application fingerprints.",
    bwPct: 0,
    latMs: 0,
    accuracy: 77.77,
    macroF1: 74.41,
    privacyNote: "Attacker sees the true timing/size pattern.",
  },
  jitter_low: {
    id: "jitter_low",
    label: "Jitter — Low",
    short: "Jitter Low",
    padding: "none",
    jitterScale: 1.0,
    jitterKey: "low",
    color: "#22d3ee",
    icon: "⏱",
    mechanism:
      "Adds one-sided Laplace delay (scale=1 ms) to inter-packet times for packets 1–29. Index 0 is never jittered (CESNET convention). Zero bytes added.",
    bwPct: 0,
    latMs: 11.0,
    accuracy: 76.84,
    macroF1: 72.72,
    privacyNote: "Best deployable point: tiny privacy loss, 11 ms latency, 0% bandwidth.",
  },
  jitter_medium: {
    id: "jitter_medium",
    label: "Jitter — Medium",
    short: "Jitter Med",
    padding: "none",
    jitterScale: 5.0,
    jitterKey: "medium",
    color: "#38bdf8",
    icon: "⏱",
    mechanism:
      "Laplace jitter with scale=5 ms. Stronger timing noise disrupts sequential models more than Transformers.",
    bwPct: 0,
    latMs: 55.2,
    accuracy: 68.83,
    macroF1: 63.01,
    privacyNote: "Moderate privacy gain; ~55 ms added latency per flow.",
  },
  jitter_high: {
    id: "jitter_high",
    label: "Jitter — High",
    short: "Jitter High",
    padding: "none",
    jitterScale: 20.0,
    jitterKey: "high",
    color: "#6366f1",
    icon: "⏱",
    mechanism:
      "Laplace scale=20 ms — aggressively smears timing fingerprints. Still 0% bandwidth overhead.",
    bwPct: 0,
    latMs: 220.8,
    accuracy: 54.68,
    macroF1: 47.66,
    privacyNote: "Strong privacy but 221 ms latency — poor for interactive QUIC.",
  },
  linear128: {
    id: "linear128",
    label: "Linear-128 Padding",
    short: "Linear128",
    padding: "linear128",
    jitterScale: 0,
    jitterKey: null,
    color: "#a78bfa",
    icon: "▭",
    mechanism:
      "Round each packet size up to the next 128-byte block (max 1500 B). Quantizes size fingerprints.",
    bwPct: 17.4,
    latMs: 0,
    accuracy: 59.06,
    macroF1: 62.01,
    privacyNote: "+17.4% bytes sent; no added delay.",
  },
  linear128_jitter_medium: {
    id: "linear128_jitter_medium",
    label: "Linear128 + Jitter Med",
    short: "L128+JMed",
    padding: "linear128",
    jitterScale: 5.0,
    jitterKey: "medium",
    color: "#c084fc",
    icon: "▭⏱",
    mechanism: "Combines 128 B size quantization with 5 ms Laplace jitter — dual-channel obfuscation.",
    bwPct: 17.4,
    latMs: 55.2,
    accuracy: 52.30,
    macroF1: 49.99,
    privacyNote: "On formal bandwidth Pareto frontier; high combined cost.",
  },
  mtu: {
    id: "mtu",
    label: "MTU Padding",
    short: "MTU",
    padding: "mtu",
    jitterScale: 0,
    jitterKey: null,
    color: "#f472b6",
    icon: "█",
    mechanism:
      "Pad every active packet to 1500 B (MTU). Destroys size fingerprints entirely but massively inflates traffic.",
    bwPct: 274.0,
    latMs: 0,
    accuracy: 2.14,
    macroF1: 0.30,
    privacyNote: "Near-random classification (~2% acc) at ~4× bandwidth — impractical on backbone links.",
  },
  mtu_jitter_medium: {
    id: "mtu_jitter_medium",
    label: "MTU + Jitter Med",
    short: "MTU+JMed",
    padding: "mtu",
    jitterScale: 5.0,
    jitterKey: "medium",
    color: "#fb7185",
    icon: "█⏱",
    mechanism: "MTU padding plus medium jitter — maximum obfuscation, extreme bandwidth cost.",
    bwPct: 274.0,
    latMs: 55.2,
    accuracy: 2.97,
    macroF1: 0.37,
    privacyNote: "Still ~3% accuracy; not viable for production backbone deployment.",
  },
};

export const DEFENSE_ORDER = [
  "baseline",
  "jitter_low",
  "jitter_medium",
  "jitter_high",
  "linear128",
  "linear128_jitter_medium",
  "mtu",
  "mtu_jitter_medium",
];

export const ARCHITECTURE = [
  { setting: "baseline", transformer: 77.77, bilstm: 72.75 },
  { setting: "jitter_low", transformer: 76.84, bilstm: 70.98 },
  { setting: "jitter_medium", transformer: 68.83, bilstm: 57.93 },
  { setting: "jitter_high", transformer: 54.68, bilstm: 35.96 },
  { setting: "linear128", transformer: 59.06, bilstm: 66.81 },
  { setting: "mtu", transformer: 2.14, bilstm: 2.88 },
];

export const STATS = {
  pairedJitterLow: {
    accDropPp: 0.93,
    accDropCi: [0.74, 1.12],
    mcnemarP: 5.8e-22,
  },
};

export const CHANNEL_ABLATION = [
  { channels: "All (IPT + DIR + SIZE)", baseline: 77.77, jitterLow: 76.84 },
  { channels: "IPT only", baseline: 0.17, jitterLow: 0.17 },
  { channels: "Direction only", baseline: 3.10, jitterLow: 3.10 },
  { channels: "Size only", baseline: 0.17, jitterLow: 0.17 },
  { channels: "IPT + Direction", baseline: 3.69, jitterLow: 3.72 },
];

export const FIGURES = {
  macroF1: "../phase4/results/macro_f1_comparison_bars.png",
  paretoLat: "../phase4/results/pareto_latency_accuracy_practical.png",
  paretoBw: "../phase4/results/pareto_bw_accuracy_practical.png",
  confusionBase: "../phase4/results/confusion_baseline.png",
  confusionJlow: "../phase4/results/confusion_obfuscated_jitter_low.png",
  dualMetric: "../phase4/results/dual_metric_top5_bars.png",
};

/** Synthetic demo flows for the interactive simulator (realistic patterns, not from dataset) */
export const DEMO_APPS = {
  video: {
    name: "Video Streaming",
    emoji: "🎬",
    sizes: [1200, 1400, 1380, 1420, 1350, 1400, 1450, 1300, 1380, 1420, 900, 850, 1200, 1400, 1380, 1420, 1350, 1400, 1450, 1300],
    ipts: [0, 12, 8, 9, 10, 8, 11, 9, 8, 10, 45, 15, 12, 8, 9, 10, 8, 11, 9, 8],
  },
  web: {
    name: "Web Browsing",
    emoji: "🌐",
    sizes: [517, 89, 1420, 312, 128, 89, 64, 1200, 89, 517, 1420, 89, 312, 128, 64, 89, 1200, 517, 89, 1420],
    ipts: [0, 85, 120, 45, 30, 200, 15, 90, 180, 50, 110, 95, 40, 25, 300, 60, 75, 100, 150, 80],
  },
};
