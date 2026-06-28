import { PROJECT, PIPELINE, DEFENSES, DEFENSE_ORDER, DEMO_APPS, FIGURES } from "./data.js";
import {
  applyDefense,
  fingerprintSimilarity,
  mockClassifier,
  renderPacketTimeline,
  renderClassifierPanel,
  padToLength,
} from "./simulation.js";
import { initCharts } from "./charts.js";

const state = { appKey: "video", defenseKey: "jitter_low" };

function $(sel) {
  return document.querySelector(sel);
}

function initNav() {
  const links = document.querySelectorAll(".nav-link");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((l) =>
            l.classList.toggle("active", l.getAttribute("href") === `#${e.target.id}`)
          );
        }
      });
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );
  document.querySelectorAll("section[id]").forEach((s) => observer.observe(s));

  links.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelector(link.getAttribute("href"))?.scrollIntoView({ behavior: "smooth" });
    });
  });
}

function renderHeroStats() {
  const el = $("#hero-stats");
  if (!el) return;
  el.innerHTML = `
    <div class="stat-card"><span class="stat-value">${PROJECT.testFlows.toLocaleString()}</span><span class="stat-label">Test flows</span></div>
    <div class="stat-card"><span class="stat-value">${PROJECT.classes}</span><span class="stat-label">App classes</span></div>
    <div class="stat-card highlight"><span class="stat-value">77.8%</span><span class="stat-label">Attack accuracy (clean)</span></div>
    <div class="stat-card highlight"><span class="stat-value">76.8%</span><span class="stat-label">After jitter_low ★</span></div>`;
}

function renderPipeline() {
  const el = $("#pipeline-steps");
  if (!el) return;
  el.innerHTML = PIPELINE.map(
    (p, i) => `
    <article class="pipeline-card" style="--delay:${i * 0.12}s">
      <div class="pipeline-num">Phase ${p.phase}</div>
      <div class="pipeline-icon">${p.icon}</div>
      <h3>${p.title}</h3>
      <p>${p.desc}</p>
    </article>`
  ).join("");
}

function renderDefenseCards() {
  const el = $("#defense-grid");
  if (!el) return;
  el.innerHTML = DEFENSE_ORDER.map((key) => {
    const d = DEFENSES[key];
    return `
    <article class="defense-card ${d.recommended ? "recommended" : ""}" data-defense="${key}">
      <div class="defense-card-head">
        <span class="defense-icon">${d.icon}</span>
        <h3>${d.label}${d.recommended ? " ★" : ""}</h3>
      </div>
      <p class="defense-mechanism">${d.mechanism}</p>
      <div class="defense-metrics">
        <span>Acc <strong>${d.accuracy.toFixed(1)}%</strong></span>
        <span>F1 <strong>${d.macroF1.toFixed(1)}%</strong></span>
        <span>BW <strong>${d.bwPct.toFixed(1)}%</strong></span>
        <span>Lat <strong>${d.latMs.toFixed(0)} ms</strong></span>
      </div>
      <p class="defense-note">${d.privacyNote}</p>
    </article>`;
  }).join("");

  el.querySelectorAll(".defense-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.defenseKey = card.dataset.defense;
      el.querySelectorAll(".defense-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      document.querySelector(`.defense-tab[data-defense="${state.defenseKey}"]`)?.click();
      document.getElementById("simulation")?.scrollIntoView({ behavior: "smooth" });
    });
  });
}

function renderFigureGallery() {
  const el = $("#figure-gallery");
  if (!el) return;
  const items = [
    { src: FIGURES.macroF1, cap: "Macro F1 across all defenses" },
    { src: FIGURES.paretoLat, cap: "Latency vs. accuracy Pareto" },
    { src: FIGURES.paretoBw, cap: "Bandwidth vs. accuracy" },
    { src: FIGURES.dualMetric, cap: "Top-5 dual-metric ranking" },
    { src: FIGURES.confusionBase, cap: "Confusion matrix — baseline" },
    { src: FIGURES.confusionJlow, cap: "Confusion matrix — jitter_low" },
  ];
  el.innerHTML = items
    .map(
      (f) => `
    <figure class="gallery-item">
      <img src="${f.src}" alt="${f.cap}" loading="lazy" onerror="this.closest('figure').classList.add('missing')"/>
      <figcaption>${f.cap}</figcaption>
    </figure>`
    )
    .join("");
}

function initSimulation() {
  const tabs = $("#defense-tabs");
  const appSelect = $("#app-select");
  if (!tabs) return;

  DEFENSE_ORDER.forEach((key) => {
    const d = DEFENSES[key];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `defense-tab ${key === state.defenseKey ? "active" : ""} ${d.recommended ? "star" : ""}`;
    btn.dataset.defense = key;
    btn.innerHTML = `<span>${d.icon}</span>${d.short}`;
    btn.addEventListener("click", () => {
      state.defenseKey = key;
      tabs.querySelectorAll(".defense-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      runSimulation();
    });
    tabs.appendChild(btn);
  });

  appSelect?.addEventListener("change", (e) => {
    state.appKey = e.target.value;
    runSimulation();
  });

  $("#sim-run")?.addEventListener("click", runSimulation);
  runSimulation();
}

function runSimulation() {
  const app = DEMO_APPS[state.appKey];
  const defense = DEFENSES[state.defenseKey];
  const wrongKey = state.appKey === "video" ? "web" : "video";
  const wrongApp = DEMO_APPS[wrongKey];

  const flow = { sizes: padToLength(app.sizes), ipts: padToLength(app.ipts) };
  const obfuscated = applyDefense(flow, defense);
  const similarity = fingerprintSimilarity(obfuscated.origSizes, obfuscated.sizes, obfuscated.mask);

  renderPacketTimeline(
    $("#timeline-before"),
    { ...obfuscated, sizes: obfuscated.origSizes, ipts: obfuscated.origIpts },
    "Before obfuscation — metadata visible on the wire"
  );
  renderPacketTimeline($("#timeline-after"), obfuscated, `After ${defense.label}`);

  const fpMeter = $("#fingerprint-meter");
  if (fpMeter) {
    fpMeter.style.width = `${(similarity * 100).toFixed(0)}%`;
    fpMeter.parentElement.dataset.label = `${(similarity * 100).toFixed(0)}% size fingerprint retained`;
  }

  $("#sim-bw").textContent = `${obfuscated.bwOverhead.toFixed(1)}%`;
  $("#sim-lat").textContent = `${obfuscated.latOverhead.toFixed(1)} ms`;
  $("#sim-acc").textContent = `${defense.accuracy.toFixed(1)}%`;
  $("#sim-f1").textContent = `${defense.macroF1.toFixed(1)}%`;

  const mechanism = $("#sim-mechanism");
  if (mechanism) mechanism.textContent = defense.mechanism;

  const result = mockClassifier(
    defense,
    `${app.emoji} ${app.name}`,
    `${wrongApp.emoji} ${wrongApp.name}`
  );
  renderClassifierPanel($("#classifier-panel"), result, `${app.emoji} ${app.name}`);

  const attackFlow = $("#attack-flow-text");
  if (attackFlow) {
    attackFlow.innerHTML = `
      <strong>Attacker input:</strong> 30 packets × 3 channels (IPT, direction, size) — TLS payload hidden.<br>
      <strong>Defense:</strong> ${defense.label}. ${defense.privacyNote}`;
  }
}

function renderTakeaways() {
  const el = $("#takeaways");
  if (!el) return;
  el.innerHTML = `
    <li><strong>Metadata leaks apps</strong> on encrypted QUIC — 77.8% accuracy from timing + sizes alone.</li>
    <li><strong>jitter_low ★</strong> — 76.8% acc, 11 ms latency, 0% bandwidth (McNemar p ≈ 10⁻²¹).</li>
    <li><strong>MTU padding</strong> — ~2% accuracy but +274% bandwidth (impractical).</li>
    <li><strong>Transformer &gt; BiLSTM</strong> under jitter; both fail under MTU.</li>
    <li><strong>All 3 channels</strong> required — single-channel models ≈ chance.</li>`;
}

function initAttackDiagram() {
  const el = $("#attack-diagram");
  if (!el) return;
  el.innerHTML = `
    <div class="flow-node user">User<div class="sub">QUIC apps</div></div>
    <div class="flow-arrow encrypted">🔒 Encrypted</div>
    <div class="flow-node proxy">Privacy proxy<div class="sub">Obfuscation layer</div></div>
    <div class="flow-arrow meta">Metadata exposed</div>
    <div class="flow-node observer">Observer<div class="sub">ISP / backbone</div></div>
    <div class="flow-arrow meta">IPT · DIR · SIZE</div>
    <div class="flow-node model">DL classifier<div class="sub">Transformer</div></div>
    <div class="flow-arrow leak">App guess</div>
    <div class="flow-node result breach">Privacy loss<div class="sub">64 classes</div></div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  initNav();
  renderHeroStats();
  renderPipeline();
  renderDefenseCards();
  renderFigureGallery();
  renderTakeaways();
  initAttackDiagram();
  initSimulation();
  initCharts();
});
