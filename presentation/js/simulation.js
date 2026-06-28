/**
 * Client-side obfuscation simulator mirroring phase3/obfuscator.py logic.
 */

const MTU = 1500;
const LINEAR_BLOCK = 128;
const SEQ_LEN = 30;

export function padToLength(arr, len = SEQ_LEN) {
  const out = [...arr];
  while (out.length < len) out.push(0);
  return out.slice(0, len);
}

export function activeMask(sizes) {
  return sizes.map((s) => s > 0);
}

export function padSizesLinear(sizes, mask) {
  return sizes.map((s, i) => {
    if (!mask[i] || s <= 0) return s;
    const padded = Math.ceil(s / LINEAR_BLOCK) * LINEAR_BLOCK;
    return Math.min(padded, MTU);
  });
}

export function padSizesMtu(sizes, mask) {
  return sizes.map((s, i) => (mask[i] && s > 0 ? MTU : s));
}

/** One-sided Laplace sample (loc=0) */
export function laplaceSample(scale, rng = Math.random) {
  const u = rng() - 0.5;
  return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

export function addJitter(ipts, mask, scale, seed = 42) {
  let state = seed;
  const rng = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };

  const out = [...ipts];
  for (let i = 0; i < out.length; i++) {
    if (!mask[i] || i === 0 || scale <= 0) continue;
    const noise = laplaceSample(scale, rng);
    out[i] = out[i] + Math.max(0, noise);
  }
  return out;
}

export function applyDefense(flow, defense) {
  const sizes = padToLength(flow.sizes);
  const ipts = padToLength(flow.ipts);
  const dirs = padToLength(flow.dirs ?? sizes.map((s, i) => (s > 0 ? (i % 2 === 0 ? 1 : -1) : 0)));
  const mask = activeMask(sizes);

  let newSizes = [...sizes];
  let newIpts = [...ipts];

  if (defense.padding === "linear128") {
    newSizes = padSizesLinear(newSizes, mask);
  } else if (defense.padding === "mtu") {
    newSizes = padSizesMtu(newSizes, mask);
  }

  if (defense.jitterScale > 0) {
    newIpts = addJitter(newIpts, mask, defense.jitterScale, 42);
  }

  const origBytes = sizes.reduce((a, s, i) => a + (mask[i] ? s : 0), 0);
  const newBytes = newSizes.reduce((a, s, i) => a + (mask[i] ? s : 0), 0);
  const bwOverhead = origBytes > 0 ? ((newBytes - origBytes) / origBytes) * 100 : 0;

  let latOverhead = 0;
  for (let i = 1; i < mask.length; i++) {
    if (mask[i]) latOverhead += Math.max(0, newIpts[i] - ipts[i]);
  }

  return {
    sizes: newSizes,
    ipts: newIpts,
    dirs,
    mask,
    origSizes: sizes,
    origIpts: ipts,
    bwOverhead,
    latOverhead,
  };
}

/** Pattern similarity 0–1 (lower = more obfuscated) */
export function fingerprintSimilarity(origSizes, newSizes, mask) {
  let diff = 0;
  let count = 0;
  for (let i = 0; i < origSizes.length; i++) {
    if (!mask[i]) continue;
    diff += Math.abs(origSizes[i] - newSizes[i]) / Math.max(origSizes[i], 1);
    count++;
  }
  if (count === 0) return 1;
  return Math.max(0, 1 - diff / count);
}

export function mockClassifier(defense, correctApp, wrongApp) {
  const acc = defense.accuracy / 100;
  const correct = Math.random() < acc;
  const topConf = correct ? 0.55 + Math.random() * 0.35 : 0.15 + Math.random() * 0.25;
  const secondConf = (1 - topConf) * (0.4 + Math.random() * 0.3);

  return {
    correct,
    predictions: correct
      ? [
          { app: correctApp, conf: topConf },
          { app: wrongApp, conf: secondConf },
          { app: "Other QUIC app", conf: 1 - topConf - secondConf },
        ]
      : [
          { app: wrongApp, conf: topConf },
          { app: correctApp, conf: secondConf },
          { app: "Other QUIC app", conf: 1 - topConf - secondConf },
        ],
    reportedAccuracy: defense.accuracy,
    macroF1: defense.macroF1,
  };
}

export function renderPacketTimeline(container, flow, label) {
  if (!container) return;
  container.innerHTML = "";
  const maxSize = MTU;
  const activeIpts = flow.ipts.filter((_, i) => flow.mask[i]);
  const maxIpt = Math.max(...activeIpts, 1);

  flow.sizes.forEach((size, i) => {
    if (!flow.mask[i]) return;

    const slot = document.createElement("div");
    slot.className = "packet-slot";
    slot.style.setProperty("--ipt", `${(flow.ipts[i] / maxIpt) * 100}%`);

    const bar = document.createElement("div");
    bar.className = "packet-bar";
    bar.style.height = `${Math.max(8, (size / maxSize) * 100)}%`;
    bar.title = `Pkt ${i}: ${Math.round(size)} B, IPT ${flow.ipts[i].toFixed(1)} ms`;

    const dir = document.createElement("span");
    dir.className = `packet-dir ${flow.dirs[i] >= 0 ? "up" : "down"}`;
    dir.textContent = flow.dirs[i] >= 0 ? "↑" : "↓";

    const idx = document.createElement("span");
    idx.className = "packet-idx";
    idx.textContent = i;

    slot.appendChild(bar);
    slot.appendChild(dir);
    slot.appendChild(idx);
    container.appendChild(slot);
  });

  const parent = container.parentElement;
  parent.querySelector(".timeline-caption")?.remove();
  const caption = document.createElement("p");
  caption.className = "timeline-caption";
  caption.textContent = label;
  container.after(caption);
}

export function renderClassifierPanel(container, result, trueApp) {
  if (!container) return;
  container.innerHTML = `
    <div class="classifier-header">
      <span class="classifier-label">Attacker prediction</span>
      <span class="classifier-badge ${result.correct ? "success" : "fail"}">
        ${result.correct ? "Correct guess" : "Misclassified"}
      </span>
    </div>
    <ul class="prediction-list">
      ${result.predictions
        .map(
          (p, i) => `
        <li class="prediction-row ${i === 0 ? "top" : ""}">
          <span>${p.app}</span>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:${(p.conf * 100).toFixed(0)}%"></div>
          </div>
          <span class="conf-pct">${(p.conf * 100).toFixed(0)}%</span>
        </li>`
        )
        .join("")}
    </ul>
    <div class="classifier-metrics">
      <div><strong>True app:</strong> ${trueApp}</div>
      <div><strong>Population accuracy:</strong> ${result.reportedAccuracy.toFixed(1)}%</div>
      <div><strong>Macro F1:</strong> ${result.macroF1.toFixed(1)}%</div>
    </div>
  `;
}
