/**
 * Bundled presentation app (no ES modules — works with file:// and all local servers).
 */
(function () {
  "use strict";

  /* ─── DATA (measured metrics from phase4/results/accuracy_results.csv) ─── */
  const PROJECT = {
    dataset: "CESNET-QUIC22-XS",
    testFlows: 49305,
    classes: 64,
    trainWeek: "W-2022-44",
    testWeek: "W-2022-45",
    chancePct: 1.56,
    attacker: "Masked Transformer (d=160)",
    resultsSource: "phase4/results/accuracy_results.csv",
  };

  const PIPELINE = [
    { phase: 1, title: "Data Engineering", desc: "Download QUIC backbone flows, normalize IPT / direction / size into (3×30) tensors with temporal train/test split.", icon: "📦" },
    { phase: 2, title: "Train Attackers", desc: "Train frozen classifiers: masked Transformer (d=160) and CNN-BiLSTM on clean metadata fingerprints.", icon: "🧠" },
    { phase: 3, title: "Obfuscate Metadata", desc: "Apply deterministic defenses on test flows — jitter timing, pad sizes to 128 B or MTU — log bandwidth/latency cost.", icon: "🛡️" },
    { phase: 4, title: "Evaluate & Compare", desc: "Measure privacy (macro F1 drop) vs. cost, Pareto frontiers, bootstrap CIs, and architecture robustness.", icon: "📊" },
  ];

  const DEFENSES = {
    baseline: { id: "baseline", label: "No defense", short: "Clean", padding: "none", jitterScale: 0, color: "#94a3b8", icon: "○", mechanism: "Raw QUIC metadata passes through unchanged. Packet sizes, directions, and inter-packet times reveal application fingerprints.", bwPct: 0, latMs: 0, accuracy: 77.77, macroF1: 74.41, privacyNote: "Attacker sees the true timing/size pattern." },
    jitter_low: { id: "jitter_low", label: "Jitter — Low", short: "Jitter Low", padding: "none", jitterScale: 1.0, color: "#22d3ee", icon: "⏱", mechanism: "Adds one-sided Laplace delay (scale=1 ms) to inter-packet times for packets 1–29. Index 0 is never jittered (CESNET convention). Zero bytes added.", bwPct: 0, latMs: 11.0, accuracy: 76.84, macroF1: 72.72, accDropPp: 0.93, privacyNote: "In our evaluation: −0.93 pp accuracy vs. clean, 11 ms mean latency overhead on the test manifest, 0% bandwidth." },
    jitter_medium: { id: "jitter_medium", label: "Jitter — Medium", short: "Jitter Med", padding: "none", jitterScale: 5.0, color: "#38bdf8", icon: "⏱", mechanism: "Laplace jitter with scale=5 ms. Stronger timing noise disrupts sequential models more than Transformers.", bwPct: 0, latMs: 55.2, accuracy: 68.83, macroF1: 63.01, privacyNote: "Moderate privacy gain; ~55 ms added latency per flow." },
    jitter_high: { id: "jitter_high", label: "Jitter — High", short: "Jitter High", padding: "none", jitterScale: 20.0, color: "#6366f1", icon: "⏱", mechanism: "Laplace scale=20 ms — aggressively smears timing fingerprints. Still 0% bandwidth overhead.", bwPct: 0, latMs: 220.8, accuracy: 54.68, macroF1: 47.66, privacyNote: "Strong privacy but 221 ms latency — poor for interactive QUIC." },
    linear128: { id: "linear128", label: "Linear-128 Padding", short: "Linear128", padding: "linear128", jitterScale: 0, color: "#a78bfa", icon: "▭", mechanism: "Round each packet size up to the next 128-byte block (max 1500 B). Quantizes size fingerprints.", bwPct: 17.4, latMs: 0, accuracy: 59.06, macroF1: 62.01, privacyNote: "+17.4% bytes sent; no added delay." },
    linear128_jitter_medium: { id: "linear128_jitter_medium", label: "Linear128 + Jitter Med", short: "L128+JMed", padding: "linear128", jitterScale: 5.0, color: "#c084fc", icon: "▭⏱", mechanism: "Combines 128 B size quantization with 5 ms Laplace jitter.", bwPct: 17.4, latMs: 55.2, accuracy: 52.30, macroF1: 49.99, privacyNote: "High combined cost; strong privacy." },
    mtu: { id: "mtu", label: "MTU Padding", short: "MTU", padding: "mtu", jitterScale: 0, color: "#f472b6", icon: "█", mechanism: "Pad every active packet to 1500 B (MTU). Destroys size fingerprints but massively inflates traffic.", bwPct: 274.0, latMs: 0, accuracy: 2.14, macroF1: 0.30, privacyNote: "Test-set mean accuracy 2.14% (near 1.56% chance); mean bandwidth overhead ~274% on the manifest." },
    mtu_jitter_medium: { id: "mtu_jitter_medium", label: "MTU + Jitter Med", short: "MTU+JMed", padding: "mtu", jitterScale: 5.0, color: "#fb7185", icon: "█⏱", mechanism: "MTU padding plus medium jitter — maximum obfuscation, extreme bandwidth cost.", bwPct: 274.0, latMs: 55.2, accuracy: 2.97, macroF1: 0.37, privacyNote: "Still ~3% accuracy; not viable for production." },
  };

  const DEFENSE_ORDER = ["baseline", "jitter_low", "jitter_medium", "jitter_high", "linear128", "linear128_jitter_medium", "mtu", "mtu_jitter_medium"];

  const ARCHITECTURE = [
    { setting: "baseline", transformer: 77.77, bilstm: 72.75 },
    { setting: "jitter_low", transformer: 76.84, bilstm: 70.98 },
    { setting: "jitter_medium", transformer: 68.83, bilstm: 57.93 },
    { setting: "jitter_high", transformer: 54.68, bilstm: 35.96 },
    { setting: "linear128", transformer: 59.06, bilstm: 66.81 },
    { setting: "linear128_jitter_medium", transformer: 52.30, bilstm: 50.57 },
    { setting: "mtu", transformer: 2.14, bilstm: 2.88 },
    { setting: "mtu_jitter_medium", transformer: 2.97, bilstm: 2.56 },
  ];

  function populationAccuracy(defenseKey, modelKey) {
    var row = null;
    for (var i = 0; i < ARCHITECTURE.length; i++) {
      if (ARCHITECTURE[i].setting === defenseKey) {
        row = ARCHITECTURE[i];
        break;
      }
    }
    if (row) return modelKey === "bilstm" ? row.bilstm : row.transformer;
    return DEFENSES[defenseKey].accuracy;
  }

  const CHANNEL_ABLATION = [
    { channels: "All (IPT + DIR + SIZE)", baseline: 77.77, jitterLow: 76.84 },
    { channels: "IPT only", baseline: 0.17, jitterLow: 0.17 },
    { channels: "Direction only", baseline: 3.10, jitterLow: 3.10 },
    { channels: "Size only", baseline: 0.17, jitterLow: 0.17 },
    { channels: "IPT + Direction", baseline: 3.69, jitterLow: 3.72 },
  ];

  const BASELINE_ACC = DEFENSES.baseline.accuracy;
  const API_BASE = "http://127.0.0.1:8765";
  var apiLive = false;
  var apiExtended = false;

  /* Real CESNET-QUIC22 test flows — generated by presentation/scripts/export_demo_flows.py */
  const DEMO_APPS = window.DEMO_FLOWS;
  if (!DEMO_APPS || !Object.keys(DEMO_APPS).length) {
    throw new Error("Missing demo flows. Run: python presentation/scripts/export_demo_flows.py");
  }

  const FLOW_CATALOG = window.TEST_FLOW_CATALOG || [];
  var CATALOG_ORDER = [];

  function shuffleArray(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
    return arr;
  }

  function initFlowCatalog() {
    if (!FLOW_CATALOG.length) {
      console.warn("TEST_FLOW_CATALOG missing — run export_flow_catalog.py");
      return;
    }
    CATALOG_ORDER = [];
    for (var i = 0; i < FLOW_CATALOG.length; i++) CATALOG_ORDER.push(i);
    shuffleArray(CATALOG_ORDER);
    state.catalogPos = 0;
  }

  function catalogEntryAt(pos) {
    return FLOW_CATALOG[CATALOG_ORDER[pos % CATALOG_ORDER.length]];
  }

  function appFromCatalogEntry(entry) {
    return {
      id: "catalog_" + entry.flowIndex,
      flowIndex: entry.flowIndex,
      classId: entry.classId,
      classIndex: null,
      name: "App " + entry.classId + " · test flow #" + entry.flowIndex,
      emoji: "📊",
      tag: "catalog",
      activePackets: null,
      testWeek: PROJECT.testWeek,
      fromCatalog: true,
      desc:
        "Full test-set flow #" + entry.flowIndex + " of " + PROJECT.testFlows.toLocaleString() +
        " (CESNET " + PROJECT.testWeek + ").",
      sizes: [],
      ipts: [],
      dirs: [],
    };
  }

  const APP_KEYS = Object.keys(DEMO_APPS);
  const state = {
    appKey: APP_KEYS[0],
    defenseKey: "jitter_low",
    modelKey: "transformer",
    dynamicFlow: null,
    sessionKey: "",
    sessionCorrect: 0,
    sessionTotal: 0,
    seenFlows: {},
    catalogPos: 0,
  };

  /* ─── SIMULATION (mirrors phase3/obfuscator.py) ─── */
  const MTU = 1500, LINEAR_BLOCK = 128, SEQ_LEN = 30;
  const PAD_DIR_EPS = 0.5, PAD_SIZE_EPS = 1e-6;

  function padToLength(arr, len) {
    len = len || SEQ_LEN;
    var out = arr.slice();
    while (out.length < len) out.push(0);
    return out.slice(0, len);
  }

  function activePacketMask(dirs, sizes) {
    return dirs.map(function (d, i) {
      return !(Math.abs(d) < PAD_DIR_EPS && sizes[i] < PAD_SIZE_EPS);
    });
  }

  function laplaceSample(scale, rng) {
    var u = rng() - 0.5;
    return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }

  function addJitter(ipts, mask, scale, seed) {
    var state = seed || 42;
    function rng() {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    }
    var out = ipts.slice();
    for (var i = 0; i < out.length; i++) {
      if (!mask[i] || i === 0 || scale <= 0) continue;
      out[i] = out[i] + Math.max(0, laplaceSample(scale, rng));
    }
    return out;
  }

  function applyDefense(flow, defense) {
    var sizes = padToLength(flow.sizes);
    var ipts = padToLength(flow.ipts);
    var dirs = padToLength(flow.dirs || sizes.map(function (s, i) { return s > 0 ? (i % 2 === 0 ? 1 : -1) : 0; }));
    var mask = activePacketMask(dirs, sizes);
    var newSizes = sizes.slice();
    var newIpts = ipts.slice();

    if (defense.padding === "linear128") {
      newSizes = newSizes.map(function (s, i) {
        if (!mask[i] || s <= 0) return s;
        return Math.min(Math.ceil(s / LINEAR_BLOCK) * LINEAR_BLOCK, MTU);
      });
    } else if (defense.padding === "mtu") {
      newSizes = newSizes.map(function (s, i) { return mask[i] && s > 0 ? MTU : s; });
    }
    if (defense.jitterScale > 0) newIpts = addJitter(newIpts, mask, defense.jitterScale, 42);

    var origBytes = 0, newBytes = 0, latOverhead = 0;
    for (var j = 0; j < mask.length; j++) {
      if (!mask[j]) continue;
      origBytes += sizes[j];
      newBytes += newSizes[j];
      if (j > 0) latOverhead += Math.max(0, newIpts[j] - ipts[j]);
    }
    return {
      sizes: newSizes, ipts: newIpts, dirs: dirs, mask: mask,
      origSizes: sizes, origIpts: ipts,
      bwOverhead: origBytes > 0 ? ((newBytes - origBytes) / origBytes) * 100 : 0,
      latOverhead: latOverhead,
    };
  }

  function sessionContextKey() {
    return state.modelKey + "|" + state.defenseKey;
  }

  function resetSessionTally() {
    state.sessionKey = sessionContextKey();
    state.sessionCorrect = 0;
    state.sessionTotal = 0;
    state.seenFlows = {};
    updateSessionTallyDisplay();
  }

  function recordSessionResult(flowIndex, correct, live) {
    if (!live) return;
    var key = sessionContextKey();
    if (key !== state.sessionKey) resetSessionTally();
    var flowKey = key + "|" + flowIndex;
    if (state.seenFlows[flowKey]) return;
    state.seenFlows[flowKey] = true;
    state.sessionTotal += 1;
    if (correct) state.sessionCorrect += 1;
    updateSessionTallyDisplay();
  }

  function updateSessionTallyDisplay() {
    var el = $("#session-accuracy");
    if (!el) return;
    if (!apiLive || state.sessionTotal === 0) {
      el.textContent = "";
      return;
    }
    var pct = (100 * state.sessionCorrect / state.sessionTotal).toFixed(0);
    var popAcc = populationAccuracy(state.defenseKey, state.modelKey);
    el.innerHTML =
      "Session: <strong>" + state.sessionCorrect + "/" + state.sessionTotal +
      " (" + pct + "%)</strong> vs test-set <strong>" + popAcc.toFixed(1) + "%</strong>";
  }

  function getActiveApp() {
    if (state.dynamicFlow) return state.dynamicFlow;
    return DEMO_APPS[state.appKey];
  }

  function updateFlowBadge(app) {
    var idxEl = $("#sim-flow-index");
    var totalEl = $("#sim-flow-total");
    var appEl = $("#sim-flow-app");
    if (idxEl) idxEl.textContent = String(app.flowIndex);
    if (totalEl) totalEl.textContent = PROJECT.testFlows.toLocaleString();
    if (appEl) {
      appEl.textContent = app.emoji + " App " + app.classId + (state.dynamicFlow ? " (loaded)" : "");
    }
  }

  function appFromFlowVis(vis, dynamic) {
    var clean = vis.clean;
    var tag = dynamic ? "random" : "loaded";
    return {
      id: "dynamic_" + vis.flow_index,
      flowIndex: vis.flow_index,
      classIndex: vis.class_index,
      classId: vis.class_id,
      name: "App " + vis.class_id + " · test flow #" + vis.flow_index,
      emoji: dynamic ? "🎲" : "🔢",
      tag: tag,
      activePackets: vis.active_packets,
      testWeek: PROJECT.testWeek,
      dynamic: true,
      desc:
        "Real CESNET-QUIC22 test flow #" + vis.flow_index + " (" + PROJECT.testWeek + "): " +
        vis.active_packets + " active packets, application ID " + vis.class_id + ".",
      sizes: clean.sizes,
      ipts: clean.ipts,
      dirs: clean.dirs,
    };
  }

  function packetsToOb(clean, obfuscated) {
    var mask = clean.mask;
    var origSizes = clean.sizes;
    var origIpts = clean.ipts;
    var sizes = obfuscated.sizes;
    var ipts = obfuscated.ipts;
    var dirs = clean.dirs;
    var origBytes = 0, newBytes = 0, latOverhead = 0;
    for (var j = 0; j < mask.length; j++) {
      if (!mask[j]) continue;
      origBytes += origSizes[j];
      newBytes += sizes[j];
      if (j > 0) latOverhead += Math.max(0, ipts[j] - origIpts[j]);
    }
    return {
      sizes: sizes,
      ipts: ipts,
      dirs: dirs,
      mask: mask,
      origSizes: origSizes,
      origIpts: origIpts,
      bwOverhead: origBytes > 0 ? ((newBytes - origBytes) / origBytes) * 100 : 0,
      latOverhead: latOverhead,
      fromApi: true,
    };
  }

  function apiPost(path, body) {
    return fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || path + " failed"); });
      return r.json();
    });
  }

  function fetchFlowVis(flowIndex, defenseId) {
    return apiPost("/flow_vis", { flow_index: flowIndex, defense: defenseId });
  }

  function fetchFlowInfo(flowIndex) {
    return apiPost("/flow_info", { flow_index: flowIndex });
  }

  function fetchRandomFlowMeta() {
    return fetch(API_BASE + "/random_flow", { method: "GET" }).then(function (r) {
      if (!r.ok) throw new Error("random_flow failed");
      return r.json();
    });
  }
  function pickWrongApp(app) {
    var currentKey = state.dynamicFlow ? null : state.appKey;
    var others = APP_KEYS.filter(function (k) { return k !== currentKey; });
    return DEMO_APPS[others[Math.floor(Math.random() * others.length)]];
  }

  function renderAppSelect() {
    var sel = $("#app-select");
    if (!sel) return;
    sel.innerHTML = APP_KEYS.map(function (key) {
      var a = DEMO_APPS[key];
      var hint = a.baselineCorrect === true ? " ✓" : a.baselineCorrect === false ? " ✗" : "";
      return '<option value="' + key + '">' + a.emoji + " #" + a.flowIndex + " · " + a.name + hint + "</option>";
    }).join("");
    if (!state.dynamicFlow) sel.value = state.appKey;
  }

  function mockClassifier(defense, correctApp, wrongApp) {
    var popAcc = populationAccuracy(state.defenseKey, state.modelKey);
    var correct = Math.random() < popAcc / 100;
    var topConf = correct ? 0.55 + Math.random() * 0.35 : 0.15 + Math.random() * 0.25;
    var secondConf = (1 - topConf) * (0.4 + Math.random() * 0.3);
    return {
      correct: correct,
      live: false,
      predictions: correct
        ? [{ app: correctApp, conf: topConf }, { app: wrongApp, conf: secondConf }, { app: "Other QUIC app", conf: 1 - topConf - secondConf }]
        : [{ app: wrongApp, conf: topConf }, { app: correctApp, conf: secondConf }, { app: "Other QUIC app", conf: 1 - topConf - secondConf }],
      reportedAccuracy: populationAccuracy(state.defenseKey, state.modelKey),
      macroF1: defense.macroF1,
    };
  }

  function apiResponseToResult(api) {
    var modelKey = api.model || state.modelKey;
    var defenseKey = api.defense || state.defenseKey;
    return {
      correct: api.correct,
      live: true,
      flowIndex: api.flow_index,
      predictions: (api.predictions || []).map(function (p) {
        return { app: p.label, conf: p.confidence };
      }),
      reportedAccuracy: populationAccuracy(defenseKey, modelKey),
      macroF1: api.macro_f1,
      predictedAppId: api.predicted_app_id,
      trueAppId: api.true_app_id,
    };
  }

  function checkInferenceApi() {
    return fetch(API_BASE + "/health", { method: "GET" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("health " + r.status)); })
      .then(function (h) {
        apiLive = h.status === "ok";
        apiExtended = apiLive && (h.api_version >= 2 || (h.features && h.features.indexOf("flow_vis") >= 0));
        return h;
      })
      .catch(function () {
        apiLive = false;
        apiExtended = false;
        return null;
      });
  }

  function loadDynamicFlow(flowIndex, isRandom) {
    var defense = DEFENSES[state.defenseKey];
    function applyVis(vis) {
      state.dynamicFlow = appFromFlowVis(vis, isRandom);
      var sel = $("#app-select");
      if (sel) sel.selectedIndex = -1;
      if (flowInput) flowInput.value = String(flowIndex);
      runSimulationWithOb(state.dynamicFlow, defense, packetsToOb(vis.clean, vis.obfuscated));
    }
    var flowInput = $("#flow-index-input");

    if (apiExtended) {
      return fetchFlowVis(flowIndex, defense.id).then(applyVis);
    }

    return fetchLivePrediction(flowIndex, defense.id, state.modelKey).then(function (pred) {
      var sel = $("#app-select");
      state.dynamicFlow = {
        id: "dynamic_" + flowIndex,
        flowIndex: flowIndex,
        classIndex: pred.trueAppId,
        classId: pred.true_app_id || pred.trueAppId,
        name: "App " + (pred.true_app_id || pred.trueAppId) + " · flow #" + flowIndex,
        emoji: isRandom ? "🎲" : "🔢",
        tag: isRandom ? "random" : "loaded",
        activePackets: null,
        testWeek: PROJECT.testWeek,
        dynamic: true,
        predictOnly: true,
        desc:
          "Test flow #" + flowIndex + " — classifier live; restart API (v2) for packet timelines.",
        sizes: [],
        ipts: [],
        dirs: [],
      };
      if (sel) sel.selectedIndex = -1;
      if (flowInput) flowInput.value = String(flowIndex);
      updateFlowBadge(state.dynamicFlow);
      var panel = $("#classifier-panel");
      renderClassifierPanel(panel, pred, state.dynamicFlow.emoji + " App " + state.dynamicFlow.classId);
      alert(
        "Flow #" + flowIndex + " loaded for inference only.\n\n" +
        "Restart api_server.py to enable packet timelines (Ctrl+C, then python presentation/api_server.py)."
      );
    });
  }

  function loadFlowByIndex(flowIndex) {
    if (!apiLive) {
      alert("Start api_server.py to load arbitrary test flows.");
      return;
    }
    loadDynamicFlow(flowIndex, false).catch(function (err) {
      console.warn(err);
      alert("Could not load flow #" + flowIndex + ": " + err.message);
    });
  }

  function loadRandomFlow() {
    if (!apiLive) {
      alert("Start api_server.py to sample random test flows.");
      return;
    }
    var pick;
    if (FLOW_CATALOG.length) {
      pick = Promise.resolve(catalogEntryAt(Math.floor(Math.random() * FLOW_CATALOG.length)).flowIndex);
    } else if (apiExtended) {
      pick = fetchRandomFlowMeta().then(function (meta) { return meta.flow_index; });
    } else {
      pick = Promise.resolve(Math.floor(Math.random() * PROJECT.testFlows));
    }
    pick
      .then(function (flowIndex) { return loadDynamicFlow(flowIndex, true); })
      .catch(function (err) {
        console.warn(err);
        alert("Random flow failed: " + err.message);
      });
  }

  function nextTestFlowFromCatalog() {
    if (!FLOW_CATALOG.length) {
      alert("Run: python presentation/scripts/export_flow_catalog.py");
      return;
    }
    if (!apiLive) {
      alert("Start api_server.py first.");
      return;
    }
    var entry = catalogEntryAt(state.catalogPos);
    state.catalogPos += 1;
    state.dynamicFlow = appFromCatalogEntry(entry);
    state.appKey = "";
    var sel = $("#app-select");
    if (sel) sel.selectedIndex = -1;
    runSimulation();
  }

  function showBatchResult(r, defense, viaLabel) {
    state.sessionKey = sessionContextKey();
    state.sessionCorrect = r.correct;
    state.sessionTotal = r.n;
    state.seenFlows = {};
    updateSessionTallyDisplay();
    var panel = $("#classifier-panel");
    if (panel) {
      panel.innerHTML =
        '<div class="classifier-header"><span class="classifier-label">Batch accuracy verification</span>' +
        '<span class="classifier-badge success">Live batch</span></div>' +
        '<p class="classifier-disclaimer">Random sample of <strong>' + r.n + '</strong> flows from the full ' +
        PROJECT.testFlows.toLocaleString() + '-flow test set (' + defense.label + ', ' +
        modelDisplayName(state.modelKey) + '). ' + (viaLabel || "") + '</p>' +
        '<div class="classifier-metrics">' +
        '<div><strong>Sample accuracy:</strong> ' + r.correct + '/' + r.n + ' = <strong>' +
        r.accuracy.toFixed(2) + '%</strong></div>' +
        '<div><strong>Phase 4 reported:</strong> ' + r.reported_accuracy.toFixed(2) + '%</div>' +
        '<div><strong>Difference:</strong> ' + (r.accuracy - r.reported_accuracy).toFixed(2) + ' pp</div>' +
        '</div>';
    }
  }

  function clientBatchVerify(count) {
    var defense = DEFENSES[state.defenseKey];
    var btn = $("#sim-batch-verify");
    var indices = [];
    for (var i = 0; i < count; i++) {
      if (FLOW_CATALOG.length) {
        indices.push(catalogEntryAt(Math.floor(Math.random() * FLOW_CATALOG.length)).flowIndex);
      } else {
        indices.push(Math.floor(Math.random() * PROJECT.testFlows));
      }
    }
    var correct = 0;
    var pos = 0;
    function step() {
      if (pos >= indices.length) {
        return Promise.resolve({
          n: count,
          correct: correct,
          accuracy: (100 * correct) / count,
          reported_accuracy: populationAccuracy(state.defenseKey, state.modelKey),
        });
      }
      var fi = indices[pos++];
      if (btn) btn.textContent = "Running " + pos + "/" + count + "…";
      return fetchLivePrediction(fi, defense.id, state.modelKey).then(function (r) {
        if (r.correct) correct += 1;
        return step();
      });
    }
    return step();
  }

  function runBatchAccuracyVerify(count) {
    if (!apiLive) {
      alert("Start api_server.py first.");
      return;
    }
    var defense = DEFENSES[state.defenseKey];
    var btn = $("#sim-batch-verify");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Running " + count + " flows…";
    }
    function finishBatch() {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Verify accuracy (" + count + " flows)";
      }
    }
    function runClientFallback(reason) {
      console.warn("batch_predict unavailable, using /predict fallback:", reason);
      return clientBatchVerify(count).then(function (r) {
        showBatchResult(r, defense, "(via individual /predict calls)");
        finishBatch();
      });
    }
    apiPost("/batch_predict", {
      defense: defense.id,
      model: state.modelKey,
      count: count,
      seed: Math.floor(Math.random() * 1000000),
    })
      .then(function (r) {
        showBatchResult(r, defense, "");
        finishBatch();
      })
      .catch(function (err) {
        runClientFallback(err.message).catch(function (e2) {
          console.warn(e2);
          alert("Batch verify failed: " + e2.message);
          finishBatch();
        });
      });
  }

  function fetchLivePrediction(flowIndex, defenseId, modelKey) {
    return apiPost("/predict", {
      flow_index: flowIndex,
      defense: defenseId,
      model: modelKey || state.modelKey,
    }).then(apiResponseToResult);
  }

  function modelDisplayName(key) {
    return key === "bilstm" ? "CNN-BiLSTM" : "Masked Transformer (d=160)";
  }

  function runClassifierInference(app, defense, wrongApp) {
    var panel = $("#classifier-panel");
    var trueLabel = app.emoji + " App " + app.classId;
    var modelName = modelDisplayName(state.modelKey);
    if (panel) {
      panel.innerHTML = '<p class="classifier-disclaimer">Running ' + modelName + " on flow #" +
        app.flowIndex + " (" + defense.short + ")…</p>";
    }
    if (apiLive && typeof app.flowIndex === "number") {
      fetchLivePrediction(app.flowIndex, defense.id, state.modelKey)
        .then(function (result) {
          result.modelKey = state.modelKey;
          recordSessionResult(app.flowIndex, result.correct, result.live);
          renderClassifierPanel(panel, result, trueLabel);
        })
        .catch(function (err) {
          console.warn("Live inference failed:", err);
          renderClassifierPanel(
            panel,
            mockClassifier(defense, app.emoji + " " + app.name, wrongApp.emoji + " " + wrongApp.name),
            trueLabel
          );
        });
    } else {
      renderClassifierPanel(
        panel,
        mockClassifier(defense, app.emoji + " " + app.name, wrongApp.emoji + " " + wrongApp.name),
        trueLabel
      );
    }
  }

  function renderPacketTimeline(container, flow, label) {
    if (!container) return;
    container.innerHTML = "";

    var slotIndex = 0;
    flow.sizes.forEach(function (size, i) {
      if (!flow.mask[i]) return;
      var slot = document.createElement("div");
      slot.className = "packet-slot";
      slot.style.animationDelay = slotIndex * 0.04 + "s";
      slotIndex++;

      var bar = document.createElement("div");
      bar.className = "packet-bar";
      bar.style.height = Math.max(8, (size / MTU) * 100) + "%";
      bar.title = "Pkt " + i + ": " + Math.round(size) + " B, IPT " + flow.ipts[i].toFixed(1) + " ms";

      var dir = document.createElement("span");
      dir.className = "packet-dir " + (flow.dirs[i] >= 0 ? "up" : "down");
      dir.textContent = flow.dirs[i] >= 0 ? "↑" : "↓";

      var idx = document.createElement("span");
      idx.className = "packet-idx";
      idx.textContent = i;

      slot.appendChild(bar);
      slot.appendChild(dir);
      slot.appendChild(idx);
      container.appendChild(slot);
    });

    if (slotIndex > 14) {
      container.style.minWidth = slotIndex * 18 + "px";
    } else {
      container.style.minWidth = "";
    }

    var wrap = container.closest(".timeline-wrap");
    if (wrap) {
      var old = wrap.querySelector(".timeline-caption");
      if (old) old.remove();
      var cap = document.createElement("p");
      cap.className = "timeline-caption";
      cap.textContent = label;
      wrap.appendChild(cap);
    }
  }

  /* ─── LIVE PACKET STREAM (flex gaps + index columns) ─── */
  var streamAnim = { raf: null, gen: 0, playing: false, events: [], defense: null };

  var GAP_MIN = 0.35;

  function stopStreamAnim() {
    streamAnim.gen++;
    if (streamAnim.raf) cancelAnimationFrame(streamAnim.raf);
    streamAnim.raf = null;
    streamAnim.playing = false;
    var btn = $("#stream-play");
    if (btn) {
      btn.classList.remove("playing");
      btn.textContent = "▶ Play stream";
    }
    var prog = $("#stream-progress");
    if (prog) prog.style.width = "0%";
  }

  function gapFlex(ipt) {
    return Math.max(ipt, GAP_MIN);
  }

  function buildStreamEvents(ob) {
    var events = [];
    var totalIptClean = 0;
    var totalIptObf = 0;
    for (var i = 0; i < ob.mask.length; i++) {
      if (!ob.mask[i]) continue;
      var iptClean = i > 0 ? ob.origIpts[i] : 0;
      var iptObf = i > 0 ? ob.ipts[i] : 0;
      totalIptClean += iptClean;
      totalIptObf += iptObf;
      events.push({
        i: i,
        iptClean: iptClean,
        iptObf: iptObf,
        origSize: ob.origSizes[i],
        newSize: ob.sizes[i],
        dir: ob.dirs[i],
        jitter: i > 0 ? Math.max(0, iptObf - iptClean) : 0,
        pad: Math.max(0, ob.sizes[i] - ob.origSizes[i]),
      });
    }
    events.totalIptClean = totalIptClean;
    events.totalIptObf = totalIptObf;
    return events;
  }

  function createStreamGap(ev, mode, defense) {
    if (ev.i === 0) return null;
    var gap = document.createElement("div");
    gap.className = "stream-gap-seg";
    gap.dataset.idx = String(ev.i);

    var base = document.createElement("div");
    base.className = "stream-gap-base";
    base.style.flex = gapFlex(ev.iptClean) + " 1 4px";
    gap.appendChild(base);

    var hasJitter = mode === "obf" && defense.jitterScale > 0 && ev.jitter > 0.05;
    if (hasJitter) {
      var jitterSeg = document.createElement("div");
      jitterSeg.className = "stream-gap-jitter";
      jitterSeg.style.flex = gapFlex(ev.jitter) + " 1 4px";
      jitterSeg.title = "+" + ev.jitter.toFixed(1) + " ms jitter before packet " + ev.i;
      var lbl = document.createElement("span");
      lbl.className = "stream-gap-label";
      lbl.textContent = "+" + formatMs(ev.jitter);
      jitterSeg.appendChild(lbl);
      gap.appendChild(jitterSeg);
      gap.style.flex = gapFlex(ev.iptObf) + " 1 6px";
    } else {
      gap.style.flex = gapFlex(mode === "obf" ? ev.iptObf : ev.iptClean) + " 1 6px";
    }

    return gap;
  }

  function formatMs(ms) {
    if (ms >= 100) return Math.round(ms) + " ms";
    if (ms >= 10) return ms.toFixed(1) + " ms";
    return ms.toFixed(2) + " ms";
  }

  function createStreamColumn(ev, mode, defense) {
    var col = document.createElement("div");
    col.className = "stream-col";
    col.dataset.idx = String(ev.i);

    var showPad = mode === "obf" && defense.padding !== "none" && ev.pad > 0.5;
    if (showPad) {
      var badge = document.createElement("span");
      badge.className = "stream-col-badge pad";
      badge.textContent = "+" + Math.round(ev.pad) + " B";
      if (defense.padding === "mtu") badge.textContent = "→ MTU";
      col.appendChild(badge);
    }

    var bar = document.createElement("div");
    bar.className = "stream-col-bar";
    var h = Math.max(6, (ev.origSize / MTU) * 56);
    bar.style.height = h + "px";
    bar.title = "Pkt " + ev.i + ": " + Math.round(ev.origSize) + " B";

    var dir = document.createElement("span");
    dir.className = "stream-col-dir " + (ev.dir >= 0 ? "up" : "down");
    dir.textContent = ev.dir >= 0 ? "↑" : "↓";

    var idx = document.createElement("span");
    idx.className = "stream-col-idx";
    idx.textContent = ev.i;

    col.appendChild(bar);
    col.appendChild(dir);
    col.appendChild(idx);

    if (mode === "obf" && showPad) {
      col.dataset.newHeight = String(Math.max(6, (ev.newSize / MTU) * 56));
    }

    return col;
  }

  function renderStreamLane(track, events, mode, defense) {
    track.innerHTML = "";
    track.className = "stream-track" + (mode === "obf" ? " obf" : "");
    if (events.length > 22) track.classList.add("very-dense");
    else if (events.length > 14) track.classList.add("dense");

    events.forEach(function (ev) {
      var gap = createStreamGap(ev, mode, defense);
      if (gap) track.appendChild(gap);
      track.appendChild(createStreamColumn(ev, mode, defense));
    });
  }

  function setupStreamVisualization(ob, defense) {
    stopStreamAnim();
    streamAnim.events = buildStreamEvents(ob);
    streamAnim.defense = defense;

    var trackClean = $("#track-clean");
    var trackObf = $("#track-obf");
    var bridgeSvg = $("#stream-bridge-svg");
    var effects = $("#stream-effects");
    var obfLabel = $("#stream-obf-label");
    var rulerInfo = $("#stream-ruler-info");
    var status = $("#stream-status");

    if (!trackClean || !trackObf) return;

    renderStreamLane(trackClean, streamAnim.events, "clean", defense);
    renderStreamLane(trackObf, streamAnim.events, "obf", defense);

    if (bridgeSvg) bridgeSvg.innerHTML = "";
    if (effects) effects.innerHTML = "";
    if (obfLabel) obfLabel.textContent = "After: " + defense.short;

    var n = streamAnim.events.length;
    var tc = streamAnim.events.totalIptClean || 0;
    var to = streamAnim.events.totalIptObf || 0;
    if (rulerInfo) {
      rulerInfo.textContent =
        n + " packets · clean IPT Σ " + tc.toFixed(0) + " ms" +
        (defense.jitterScale > 0 ? " → obf Σ " + to.toFixed(0) + " ms" : "");
    }
    if (status) status.textContent = n + " packets · press Play";

    buildEffectChips(effects, defense, ob);
  }

  function buildEffectChips(container, defense, ob) {
    if (!container) return;
    if (defense.padding === "none" && defense.jitterScale <= 0) {
      addEffectChip(container, "", "○ Baseline — lanes match (no jitter or padding)");
      return;
    }
    if (defense.jitterScale > 0) {
      var jCount = streamAnim.events.filter(function (e) { return e.jitter > 0.05; }).length;
      addEffectChip(container, "jitter", "⏱ Jitter: gold gaps = +" + defense.jitterScale + " ms scale (" + jCount + " packets shifted)");
    }
    if (defense.padding === "linear128") {
      var pCount = streamAnim.events.filter(function (e) { return e.pad > 0.5; }).length;
      addEffectChip(container, "pad", "▭ Linear-128: bars grow to 128 B blocks (" + pCount + " padded)");
    } else if (defense.padding === "mtu") {
      addEffectChip(container, "pad", "█ MTU: all active packets → 1500 B");
    }
  }

  function addEffectChip(container, kind, text) {
    if (!container) return;
    var chip = document.createElement("span");
    chip.className = "stream-effect-chip" + (kind ? " " + kind : "");
    chip.textContent = text;
    container.appendChild(chip);
  }

  function playStreamAnimation(ob, defense) {
    stopStreamAnim();
    var gen = streamAnim.gen;
    var events = streamAnim.events;
    if (!events.length) return;

    var trackClean = $("#track-clean");
    var trackObf = $("#track-obf");
    var status = $("#stream-status");
    var btn = $("#stream-play");
    var prog = $("#stream-progress");
    var chips = document.querySelectorAll(".stream-effect-chip");
    var duration = Math.min(10000, Math.max(2800, events.length * 320));
    var step = duration / (events.length * 2 + 2);

    streamAnim.playing = true;
    if (btn) { btn.classList.add("playing"); btn.textContent = "⏸ Playing…"; }
    if (status) status.textContent = "Revealing clean flow…";
    chips.forEach(function (c) { c.classList.remove("active", "done"); });

    var cleanCols = trackClean.querySelectorAll(".stream-col");
    var cleanGaps = trackClean.querySelectorAll(".stream-gap-seg");
    var obfCols = trackObf.querySelectorAll(".stream-col");
    var obfGaps = trackObf.querySelectorAll(".stream-gap-seg");

    function resetLane(cols, gaps, isObf) {
      cols.forEach(function (c) { c.classList.remove("visible"); });
      gaps.forEach(function (g) {
        g.classList.remove("visible");
        if (isObf) {
          g.querySelectorAll(".stream-gap-jitter").forEach(function (j) { j.classList.remove("show"); });
        }
      });
      if (isObf) {
        cols.forEach(function (c) {
          c.querySelectorAll(".stream-col-badge").forEach(function (b) { b.classList.remove("show"); });
          var bar = c.querySelector(".stream-col-bar");
          if (bar) {
            bar.classList.remove("padded");
            delete bar.dataset.grown;
            var origH = Math.max(6, (parseFloat(c.dataset.origSize || 0) / MTU) * 56);
            if (c.dataset.origSize) bar.style.height = origH + "px";
          }
        });
      }
    }

    cleanCols.forEach(function (c, idx) {
      if (events[idx]) c.dataset.origSize = String(events[idx].origSize);
    });
    obfCols.forEach(function (c, idx) {
      if (events[idx]) c.dataset.origSize = String(events[idx].origSize);
    });

    resetLane(cleanCols, cleanGaps, false);
    resetLane(obfCols, obfGaps, true);

    var start = performance.now();
    var totalSteps = events.length * 2;

    function revealClean(upto) {
      for (var i = 0; i < upto; i++) {
        if (cleanCols[i]) cleanCols[i].classList.add("visible");
        if (i > 0 && cleanGaps[i - 1]) cleanGaps[i - 1].classList.add("visible");
      }
    }

    function revealObf(upto) {
      for (var j = 0; j < upto; j++) {
        if (j > 0 && obfGaps[j - 1]) {
          obfGaps[j - 1].classList.add("visible");
          var jitterEl = obfGaps[j - 1].querySelector(".stream-gap-jitter");
          if (jitterEl) jitterEl.classList.add("show");
        }
        if (obfCols[j]) {
          obfCols[j].classList.add("visible");
          var bar = obfCols[j].querySelector(".stream-col-bar");
          var badge = obfCols[j].querySelector(".stream-col-badge");
          var newH = obfCols[j].dataset.newHeight;
          if (bar && newH && !bar.dataset.grown) {
            bar.dataset.grown = "1";
            setTimeout(function (b, h, bd) {
              if (gen !== streamAnim.gen) return;
              b.style.height = h + "px";
              b.classList.add("padded");
              if (bd) bd.classList.add("show");
            }, 200, bar, newH, badge);
          }
        }
      }
    }

    function frame(now) {
      if (gen !== streamAnim.gen) return;
      var elapsed = now - start;
      var p = Math.min(1, elapsed / duration);
      if (prog) prog.style.width = (p * 100).toFixed(1) + "%";

      var cleanCount = Math.min(events.length, Math.floor(p * totalSteps / 2) + (p > 0 ? 1 : 0));
      var obfStart = 0.45;
      var obfCount = 0;
      if (p >= obfStart) {
        obfCount = Math.min(events.length, Math.floor((p - obfStart) / (1 - obfStart) * events.length) + 1);
      }

      revealClean(cleanCount);
      if (p >= obfStart) {
        revealObf(obfCount);
        if (status) status.textContent = "Obfuscated: " + defense.short + " · " + obfCount + "/" + events.length;
      } else if (status) {
        status.textContent = "Clean flow · " + cleanCount + "/" + events.length + " packets";
      }

      if (p > 0.08 && p < 0.45) {
        chips.forEach(function (c, i) { if (p > 0.1 + i * 0.06) c.classList.add("active"); });
      }

      if (p >= 1) {
        revealClean(events.length);
        revealObf(events.length);
        chips.forEach(function (c) { c.classList.add("active", "done"); });
        streamAnim.playing = false;
        if (btn) { btn.classList.remove("playing"); btn.textContent = "↻ Replay stream"; }
        if (status) status.textContent = "Done · " + defense.short + " applied";
        return;
      }
      streamAnim.raf = requestAnimationFrame(frame);
    }

    streamAnim.raf = requestAnimationFrame(frame);
  }

  function initStreamControls() {
    var btn = $("#stream-play");
    if (!btn || btn.dataset.bound) return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", function () {
      if (streamAnim.playing) {
        stopStreamAnim();
        return;
      }
      if (state.streamOb && state.streamDefense) {
        playStreamAnimation(state.streamOb, state.streamDefense);
      }
    });
  }

  function renderClassifierPanel(container, result, trueApp) {
    if (!container) return;
    var rows = result.predictions.map(function (p, i) {
      return '<li class="prediction-row' + (i === 0 ? " top" : "") + '"><span>' + p.app + '</span><div class="conf-bar-wrap"><div class="conf-bar" style="width:' + (p.conf * 100).toFixed(0) + '%"></div></div><span class="conf-pct">' + (p.conf * 100).toFixed(0) + '%</span></li>';
    }).join("");
    var headerLabel = result.live
      ? "Live " + modelDisplayName(result.modelKey || state.modelKey) + " prediction"
      : "Example single-flow guess (simulated)";
    var badgeText = result.correct
      ? (result.live ? "Correct classification" : "Would guess correctly")
      : (result.live ? "Misclassified" : "Would misclassify");
    var disclaimer = result.live
      ? "Real forward pass on test flow <strong>#" + (result.flowIndex != null ? result.flowIndex : "…") + "</strong> after Phase 3 obfuscation. " +
        "The 12 curated examples are for packet visuals only. Use <strong>Next test flow</strong> (full 49k set) or <strong>Verify accuracy</strong>. " +
        "Population test-set accuracy: <strong>" + result.reportedAccuracy.toFixed(2) + "%</strong>."
      : 'Demo only: one random outcome sampled from the defense\'s <strong>measured test-set accuracy</strong> below — start <code>python presentation/api_server.py</code> for live inference.';
    container.innerHTML =
      '<div class="classifier-header">' +
      '<span class="classifier-label">' + headerLabel + '</span>' +
      '<span class="classifier-badge ' + (result.correct ? "success" : "fail") + '">' +
      badgeText + '</span></div>' +
      '<p class="classifier-disclaimer">' + disclaimer + '</p>' +
      '<ul class="prediction-list">' + rows + '</ul>' +
      '<div class="classifier-metrics">' +
      '<div><strong>True app (CESNET ID):</strong> ' + trueApp + '</div>' +
      (result.live && result.predictedAppId != null
        ? '<div><strong>Predicted app ID:</strong> ' + result.predictedAppId + '</div>'
        : '') +
      '<div><strong>Measured test-set accuracy (' + (result.live ? "this defense" : "population") + '):</strong> ' + result.reportedAccuracy.toFixed(2) + '%</div>' +
      '<div><strong>Measured macro F1:</strong> ' + result.macroF1.toFixed(2) + '%</div></div>';
  }

  /* ─── CHARTS ─── */
  function chartOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", boxWidth: 12 } },
        title: { display: true, text: title, color: "#e2e8f0", font: { size: 14 } },
      },
      scales: {
        x: { ticks: { color: "#94a3b8", maxRotation: 45 }, grid: { color: "rgba(148,163,184,0.08)" } },
        y: { min: 0, max: 85, ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
      },
    };
  }

  function initCharts() {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js not loaded — charts skipped (need internet for CDN).");
      return;
    }
    var labels = DEFENSE_ORDER.map(function (k) { return DEFENSES[k].short; });
    var colors = DEFENSE_ORDER.map(function (k) { return DEFENSES[k].color; });

    var accCtx = document.getElementById("chart-accuracy");
    if (accCtx) {
      new Chart(accCtx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            { label: "Accuracy %", data: DEFENSE_ORDER.map(function (k) { return DEFENSES[k].accuracy; }), backgroundColor: colors.map(function (c) { return c + "cc"; }), borderColor: colors, borderWidth: 1, borderRadius: 6 },
            { label: "Macro F1 %", data: DEFENSE_ORDER.map(function (k) { return DEFENSES[k].macroF1; }), backgroundColor: colors.map(function (c) { return c + "55"; }), borderColor: colors, borderWidth: 1, borderRadius: 6 },
          ],
        },
        options: chartOptions("Privacy vs. defense setting (chance = 1.56%)"),
      });
    }

    var paretoCtx = document.getElementById("chart-pareto");
    if (paretoCtx) {
      var points = DEFENSE_ORDER.filter(function (k) { return k !== "baseline"; }).map(function (k) {
        var d = DEFENSES[k];
        return { x: d.latMs > 0 ? d.latMs : d.bwPct + 0.5, y: d.accuracy, label: d.short, color: d.color };
      });
      new Chart(paretoCtx, {
        type: "scatter",
        data: {
          datasets: [
            { label: "Defenses", data: points, pointBackgroundColor: points.map(function (p) { return p.color; }), pointRadius: 10 },
            { label: "Baseline", data: [{ x: 0.5, y: DEFENSES.baseline.accuracy }], pointBackgroundColor: "#94a3b8", pointRadius: 12, pointStyle: "star" },
            { label: "Chance", data: [{ x: 0.5, y: 1.56 }], pointBackgroundColor: "#ef4444", pointRadius: 8 },
          ],
        },
        options: chartOptions("Cost vs. accuracy"),
      });
    }

    var archCtx = document.getElementById("chart-architecture");
    if (archCtx) {
      new Chart(archCtx, {
        type: "bar",
        data: {
          labels: ARCHITECTURE.map(function (a) { return a.setting.replace(/_/g, " "); }),
          datasets: [
            { label: "Transformer", data: ARCHITECTURE.map(function (a) { return a.transformer; }), backgroundColor: "rgba(34,211,238,0.75)", borderRadius: 6 },
            { label: "CNN-BiLSTM", data: ARCHITECTURE.map(function (a) { return a.bilstm; }), backgroundColor: "rgba(167,139,250,0.75)", borderRadius: 6 },
          ],
        },
        options: chartOptions("Architecture robustness"),
      });
    }

    var abCtx = document.getElementById("chart-ablation");
    if (abCtx) {
      new Chart(abCtx, {
        type: "bar",
        data: {
          labels: CHANNEL_ABLATION.map(function (c) { return c.channels; }),
          datasets: [
            { label: "Baseline", data: CHANNEL_ABLATION.map(function (c) { return c.baseline; }), backgroundColor: "rgba(148,163,184,0.7)", borderRadius: 4 },
            { label: "jitter_low", data: CHANNEL_ABLATION.map(function (c) { return c.jitterLow; }), backgroundColor: "rgba(34,211,238,0.7)", borderRadius: 4 },
          ],
        },
        options: Object.assign({ indexAxis: "y" }, chartOptions("Channel ablation")),
      });
    }
  }

  /* ─── UI ─── */
  function $(sel) { return document.querySelector(sel); }

  function showReady(apiHealth) {
    var badge = $("#js-status");
    if (!badge) return;
    if (apiHealth && apiHealth.status === "ok") {
      badge.textContent = apiExtended
        ? "Live model API v2 (" + apiHealth.device + ")"
        : "Live API v1 — restart api_server.py for random/load flows";
      badge.className = apiExtended ? "js-status ok" : "js-status warn";
      var models = apiHealth.models || [];
      document.querySelectorAll(".model-tab").forEach(function (btn) {
        var ok = models.indexOf(btn.dataset.model) >= 0;
        btn.disabled = !ok;
        btn.title = ok ? "" : "Checkpoint not loaded on server";
      });
    } else {
      badge.textContent = "Demo loaded — start api_server.py for live inference";
      badge.className = "js-status warn";
    }
    updateRunHint();
  }

  function updateRunHint() {
    var hint = $("#sim-run-hint");
    if (!hint) return;
    if (apiLive) {
      hint.innerHTML =
        "All <strong>" + PROJECT.testFlows.toLocaleString() + " test flows</strong> are in the catalog. " +
        "Use <strong>Next test flow</strong> to walk the full set, or <strong>Verify accuracy (200 flows)</strong> to match Phase 4. " +
        "Curated dropdown = packet-visual examples only.";
    } else {
      hint.textContent =
        "Without the API, Re-run randomizes a demo outcome from measured test-set accuracy.";
    }
  }

  function renderHeroStats() {
    var el = $("#hero-stats");
    if (!el) return;
    var tfm = ARCHITECTURE[0].transformer;
    var lstm = ARCHITECTURE[0].bilstm;
    el.innerHTML =
      '<div class="stat-card"><span class="stat-value">' + PROJECT.testFlows.toLocaleString() + '</span><span class="stat-label">Test flows</span></div>' +
      '<div class="stat-card"><span class="stat-value">' + PROJECT.classes + '</span><span class="stat-label">App classes</span></div>' +
      '<div class="stat-card highlight"><span class="stat-value">' + tfm.toFixed(1) + '%</span><span class="stat-label">Transformer accuracy (clean)</span></div>' +
      '<div class="stat-card highlight"><span class="stat-value">' + lstm.toFixed(1) + '%</span><span class="stat-label">BiLSTM accuracy (clean)</span></div>';
  }

  function renderPipeline() {
    var el = $("#pipeline-steps");
    if (!el) return;
    el.innerHTML = PIPELINE.map(function (p, i) {
      return '<article class="pipeline-card" style="--delay:' + (i * 0.12) + 's"><div class="pipeline-num">Phase ' + p.phase + '</div><div class="pipeline-icon">' + p.icon + '</div><h3>' + p.title + '</h3><p>' + p.desc + '</p></article>';
    }).join("");
  }

  function renderDefenseCards() {
    var el = $("#defense-grid");
    if (!el) return;
    el.innerHTML = DEFENSE_ORDER.map(function (key) {
      var d = DEFENSES[key];
      var drop = BASELINE_ACC - d.accuracy;
      return '<article class="defense-card' + (key === state.defenseKey ? " selected" : "") + '" data-defense="' + key + '">' +
        '<div class="defense-card-head"><span class="defense-icon">' + d.icon + '</span><h3>' + d.label + '</h3></div>' +
        '<p class="defense-mechanism">' + d.mechanism + '</p>' +
        '<div class="defense-metrics"><span>Acc <strong>' + d.accuracy.toFixed(1) + '%</strong></span>' +
        '<span>F1 <strong>' + d.macroF1.toFixed(1) + '%</strong></span>' +
        '<span>BW <strong>' + d.bwPct.toFixed(1) + '%</strong></span>' +
        '<span>Lat <strong>' + d.latMs.toFixed(0) + ' ms</strong></span></div>' +
        (drop > 0.5 ? '<span class="defense-drop">−' + drop.toFixed(1) + ' pp accuracy</span>' : '') +
        '<p class="defense-note">' + d.privacyNote + '</p></article>';
    }).join("");
    el.querySelectorAll(".defense-card").forEach(function (card) {
      card.addEventListener("click", function () {
        selectDefense(card.dataset.defense, true);
      });
    });
  }

  function selectDefense(key, scroll) {
    state.defenseKey = key;
    resetSessionTally();
    document.querySelectorAll(".defense-card").forEach(function (c) {
      c.classList.toggle("selected", c.dataset.defense === key);
    });
    document.querySelectorAll(".defense-tab").forEach(function (b) {
      b.classList.toggle("active", b.dataset.defense === key);
    });
    runSimulation();
    if (scroll) {
      var sim = document.getElementById("simulation");
      if (sim) sim.scrollIntoView({ behavior: "smooth" });
    }
  }

  function renderResultsHighlights() {
    var el = $("#results-highlights");
    if (!el) return;
    var picks = [
      { key: "baseline", tag: "Measured baseline", desc: "Clean test tensors — no obfuscation" },
      { key: "jitter_low", tag: "Low jitter", desc: "76.84% accuracy, 11 ms latency, 0% bandwidth" },
      { key: "jitter_high", tag: "Strong jitter", desc: "54.68% test accuracy, 220.8 ms mean latency" },
      { key: "mtu", tag: "Strongest measured privacy", desc: "2.14% test accuracy, ~274% bandwidth overhead" },
    ];
    el.innerHTML = picks.map(function (p) {
      var d = DEFENSES[p.key];
      return '<article class="highlight-card">' +
        '<span class="highlight-tag">' + p.tag + '</span>' +
        '<h4>' + d.label + '</h4>' +
        '<p class="highlight-desc">' + p.desc + '</p>' +
        '<div class="highlight-stats">' +
        '<span><em>Acc</em> ' + d.accuracy.toFixed(1) + '%</span>' +
        '<span><em>Lat</em> ' + d.latMs.toFixed(0) + ' ms</span>' +
        '<span><em>BW</em> ' + d.bwPct.toFixed(0) + '%</span></div>' +
        '<button type="button" class="highlight-btn" data-defense="' + p.key + '">Load in simulator</button></article>';
    }).join("");
    el.querySelectorAll(".highlight-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { selectDefense(btn.dataset.defense, true); });
    });
  }

  function renderResultsTable() {
    var tbody = document.querySelector("#results-table tbody");
    if (!tbody) return;
    tbody.innerHTML = DEFENSE_ORDER.map(function (key) {
      var d = DEFENSES[key];
      var drop = BASELINE_ACC - d.accuracy;
      var rowClass = key === "baseline" ? ' class="row-base"' : "";
      return "<tr" + rowClass + "><td>" + d.label + "</td>" +
        "<td>" + d.accuracy.toFixed(1) + "%</td>" +
        "<td>" + d.macroF1.toFixed(1) + "%</td>" +
        "<td>" + (drop > 0 ? "−" + drop.toFixed(1) : "—") + "</td>" +
        "<td>" + d.bwPct.toFixed(1) + "%</td>" +
        "<td>" + d.latMs.toFixed(0) + " ms</td></tr>";
    }).join("");
  }

  function renderTakeaways() {
    var el = $("#takeaways-grid");
    if (!el) return;
    var items = [
      { icon: "🔓", title: "Metadata leaks apps", text: "77.77% test accuracy on clean CESNET-QUIC22-XS flows (64 classes, W-2022-45 holdout)." },
      { icon: "⏱", title: "jitter_low measured", text: "76.84% accuracy, 72.72% macro F1, 11.0 ms mean latency overhead, 0% bandwidth (Phase 4 manifest)." },
      { icon: "💥", title: "MTU padding", text: "2.14% test accuracy but ~274% mean bandwidth overhead — impractical at backbone scale." },
      { icon: "🧠", title: "Transformer vs BiLSTM", text: "Measured on same test set: Transformer stronger under jitter; BiLSTM slightly higher on linear128 only." },
      { icon: "📡", title: "Three channels required", text: "Channel ablation: single-channel inputs near chance; full IPT+DIR+SIZE tensor needed." },
    ];
    el.innerHTML = items.map(function (it) {
      return '<article class="takeaway-card glass"><span class="takeaway-icon">' + it.icon + '</span><h4>' + it.title + '</h4><p>' + it.text + '</p></article>';
    }).join("");
  }

  function initAttackDiagram() {
    var el = $("#attack-diagram");
    if (!el) return;
    el.innerHTML =
      '<div class="flow-node user">User<div class="sub">QUIC apps</div></div><div class="flow-arrow encrypted">Encrypted</div><div class="flow-node proxy">Privacy proxy<div class="sub">Obfuscation</div></div><div class="flow-arrow meta">Metadata exposed</div><div class="flow-node observer">Observer<div class="sub">ISP / backbone</div></div><div class="flow-arrow meta">IPT · DIR · SIZE</div><div class="flow-node model">DL classifier<div class="sub">Transformer</div></div><div class="flow-arrow leak">App guess</div><div class="flow-node result breach">Privacy loss<div class="sub">64 classes</div></div>';
  }

  function nextExampleFlow() {
    state.dynamicFlow = null;
    var idx = APP_KEYS.indexOf(state.appKey);
    var nextKey = APP_KEYS[(idx + 1) % APP_KEYS.length];
    state.appKey = nextKey;
    var appSelect = $("#app-select");
    if (appSelect) appSelect.value = nextKey;
    runSimulation();
  }

  function runSimulationWithOb(app, defense, ob) {
    var wrongApp = pickWrongApp(app);
    updateFlowBadge(app);

    var afterLabel = "After defense (Python obfuscator.py)";
    if (!ob.fromApi) afterLabel = "Same fingerprint after: " + defense.label + " (client preview)";

    renderPacketTimeline(
      $("#timeline-before"),
      { sizes: ob.origSizes, ipts: ob.origIpts, dirs: ob.dirs, mask: ob.mask },
      "Test flow #" + app.flowIndex + " — clean metadata"
    );
    renderPacketTimeline($("#timeline-after"), ob, afterLabel);

    var bw = $("#sim-bw"); if (bw) bw.textContent = defense.bwPct.toFixed(1) + "%";
    var lat = $("#sim-lat"); if (lat) lat.textContent = defense.latMs.toFixed(1) + " ms";
    var acc = $("#sim-acc");
    if (acc) {
      acc.textContent = defense.accuracy.toFixed(1) + "%";
      acc.classList.add("metric-flash");
      setTimeout(function () { acc.classList.remove("metric-flash"); }, 600);
    }
    var f1 = $("#sim-f1"); if (f1) f1.textContent = defense.macroF1.toFixed(1) + "%";
    var delta = $("#sim-delta");
    if (delta) {
      var drop = BASELINE_ACC - defense.accuracy;
      if (drop < 0.05) {
        delta.textContent = "None";
        delta.className = "";
      } else {
        delta.textContent = "−" + drop.toFixed(1) + " pp acc";
        delta.className = drop > 20 ? "bad" : drop > 5 ? "mid" : "good";
      }
    }
    var mech = $("#sim-mechanism"); if (mech) mech.textContent = defense.mechanism;
    var ctx = $("#attack-flow-text");
    if (ctx) {
      var obfNote = ob.fromApi
        ? " <strong>Obfuscated packets</strong> from <code>phase3/obfuscator.py</code> via API."
        : " <strong>Obfuscation preview</strong> in browser (start API for Python-exact packets).";
      ctx.innerHTML =
        "<strong>Flow #" + app.flowIndex + ":</strong> " + app.desc + obfNote +
        " <strong>Metrics</strong> are test-set means on " + PROJECT.testFlows.toLocaleString() +
        " flows (" + PROJECT.testWeek + ").";
    }

    renderClassifierPanel(
      $("#classifier-panel"),
      { predictions: [], correct: false, live: false, reportedAccuracy: populationAccuracy(state.defenseKey, state.modelKey), macroF1: defense.macroF1 },
      app.emoji + " App " + app.classId
    );
    runClassifierInference(app, defense, wrongApp);

    state.streamOb = ob;
    state.streamDefense = defense;
    setupStreamVisualization(ob, defense);
    clearTimeout(state.streamAutoTimer);
    state.streamAutoTimer = setTimeout(function () {
      if (state.streamOb === ob && state.defenseKey === defense.id) {
        playStreamAnimation(ob, defense);
      }
    }, 350);
  }

  function runSimulation() {
    var app = getActiveApp();
    var defense = DEFENSES[state.defenseKey];
    var flow = { sizes: padToLength(app.sizes), ipts: padToLength(app.ipts), dirs: padToLength(app.dirs) };

    if (apiLive && typeof app.flowIndex === "number") {
      fetchFlowVis(app.flowIndex, defense.id)
        .then(function (vis) {
          if (state.dynamicFlow && state.dynamicFlow.flowIndex === vis.flow_index) {
            state.dynamicFlow.sizes = vis.clean.sizes;
            state.dynamicFlow.ipts = vis.clean.ipts;
            state.dynamicFlow.dirs = vis.clean.dirs;
          }
          runSimulationWithOb(getActiveApp(), defense, packetsToOb(vis.clean, vis.obfuscated));
        })
        .catch(function (err) {
          console.warn("flow_vis failed, using client obfuscation:", err);
          runSimulationWithOb(app, defense, applyDefense(flow, defense));
        });
      return;
    }

    runSimulationWithOb(app, defense, applyDefense(flow, defense));
  }

  function initSimulation() {
    var tabs = $("#defense-tabs");
    if (!tabs) return;
    tabs.innerHTML = "";
    DEFENSE_ORDER.forEach(function (key) {
      var d = DEFENSES[key];
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "defense-tab" + (key === state.defenseKey ? " active" : "");
      btn.dataset.defense = key;
      btn.innerHTML = "<span>" + d.icon + "</span>" + d.short;
      btn.addEventListener("click", function () {
        selectDefense(key, false);
      });
      tabs.appendChild(btn);
    });
    var appSelect = $("#app-select");
    renderAppSelect();
    if (appSelect) {
      appSelect.addEventListener("change", function (e) {
        state.dynamicFlow = null;
        state.appKey = e.target.value;
        runSimulation();
      });
    }
    document.querySelectorAll(".model-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.modelKey = btn.dataset.model;
        resetSessionTally();
        document.querySelectorAll(".model-tab").forEach(function (b) {
          b.classList.toggle("active", b === btn);
        });
        runSimulation();
      });
    });
    var runBtn = $("#sim-run");
    if (runBtn) runBtn.addEventListener("click", runSimulation);
    var nextBtn = $("#sim-next-flow");
    if (nextBtn) nextBtn.addEventListener("click", nextExampleFlow);
    var nextTestBtn = $("#sim-next-test-flow");
    if (nextTestBtn) nextTestBtn.addEventListener("click", nextTestFlowFromCatalog);
    var batchBtn = $("#sim-batch-verify");
    if (batchBtn) batchBtn.addEventListener("click", function () { runBatchAccuracyVerify(200); });
    var randomBtn = $("#sim-random-flow");
    if (randomBtn) randomBtn.addEventListener("click", loadRandomFlow);
    var loadBtn = $("#sim-load-flow");
    var flowInput = $("#flow-index-input");
    if (loadBtn && flowInput) {
      loadBtn.addEventListener("click", function () {
        var v = parseInt(flowInput.value, 10);
        if (isNaN(v) || v < 0 || v >= PROJECT.testFlows) {
          alert("Enter a flow index between 0 and " + (PROJECT.testFlows - 1));
          return;
        }
        loadFlowByIndex(v);
      });
    }
    initStreamControls();
    runSimulation();
  }

  function initNav() {
    var links = document.querySelectorAll(".nav-link, .mobile-nav a");
    links.forEach(function (link) {
      link.addEventListener("click", function (e) {
        e.preventDefault();
        var t = document.querySelector(link.getAttribute("href"));
        if (t) t.scrollIntoView({ behavior: "smooth" });
      });
    });
    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          links.forEach(function (l) {
            l.classList.toggle("active", l.getAttribute("href") === "#" + e.target.id);
          });
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      document.querySelectorAll("section[id]").forEach(function (s) { observer.observe(s); });
    }
  }

  function initPresentMode() {
    var btn = $("#present-mode");
    if (!btn) return;
    btn.addEventListener("click", function () {
      document.body.classList.toggle("present-mode");
      btn.textContent = document.body.classList.contains("present-mode") ? "Exit" : "Present";
    });
  }

  function initQuickCompare() {
    var btn = $("#demo-compare");
    if (!btn) return;
    var sequence = ["baseline", "jitter_low", "mtu"];
    var idx = 0;
    btn.addEventListener("click", function () {
      document.getElementById("simulation").scrollIntoView({ behavior: "smooth" });
      selectDefense(sequence[idx], false);
      idx = (idx + 1) % sequence.length;
      btn.textContent = "Next: " + DEFENSES[sequence[idx]].short + " →";
    });
  }

  function boot() {
    try {
      initNav();
      initPresentMode();
      initQuickCompare();
      renderHeroStats();
      renderPipeline();
      renderDefenseCards();
      renderResultsHighlights();
      renderResultsTable();
      renderTakeaways();
      initAttackDiagram();
      initFlowCatalog();
      initSimulation();
      initCharts();
      checkInferenceApi().then(showReady);
    } catch (err) {
      console.error(err);
      var banner = $("#js-error");
      if (banner) {
        banner.hidden = false;
        banner.textContent = "JavaScript error: " + err.message;
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
