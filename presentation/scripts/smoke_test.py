"""Smoke test: presentation static assets + inference API + demo flow alignment."""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8765"
WEB = "http://localhost:8080"
DEMO_JS = ROOT / "presentation" / "js" / "demo_flows.js"
FAILURES: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}", file=sys.stderr)


def get(url: str, timeout: float = 30) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def post_json(url: str, payload: dict, timeout: float = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def parse_demo_flows() -> list[dict]:
    text = DEMO_JS.read_text(encoding="utf-8")
    flows = []
    for block in re.finditer(
        r'"flowIndex":\s*(\d+).*?"classIndex":\s*(\d+).*?"classId":\s*(\d+)',
        text,
        re.DOTALL,
    ):
        flows.append(
            {
                "flowIndex": int(block.group(1)),
                "classIndex": int(block.group(2)),
                "classId": int(block.group(3)),
            }
        )
    return flows


def test_static() -> None:
    print("\n[1] Static presentation")
    try:
        status, body = get(f"{WEB}/presentation/")
        if status != 200:
            fail(f"GET /presentation/ -> {status}")
            return
        html = body.decode("utf-8", errors="replace")
        ok(f"index.html ({len(html)} bytes)")
        for path in (
            "/presentation/js/app.js",
            "/presentation/js/demo_flows.js",
            "/presentation/css/styles.css",
        ):
            s, b = get(f"{WEB}{path}")
            if s != 200 or len(b) < 100:
                fail(f"{path} -> {s}, {len(b)} bytes")
            else:
                ok(f"{path} ({len(b)} bytes)")
        if "window.DEMO_FLOWS" not in get(f"{WEB}/presentation/js/demo_flows.js")[1].decode():
            fail("demo_flows.js missing window.DEMO_FLOWS")
        else:
            ok("demo_flows.js defines DEMO_FLOWS")
    except urllib.error.URLError as exc:
        fail(f"static server not reachable at {WEB}: {exc}")


def test_api_health() -> dict | None:
    print("\n[2] Inference API /health")
    try:
        health = json.loads(get(f"{API}/health")[1].decode())
    except urllib.error.URLError as exc:
        fail(f"API not reachable at {API}: {exc}")
        return None
    if health.get("status") != "ok":
        fail(f"health status={health.get('status')}")
        return None
    ok(f"status=ok device={health.get('device')} flows={health.get('test_flows')}")
    models = health.get("models") or []
    if "transformer" not in models:
        fail(f"transformer not loaded: {models}")
    else:
        ok("transformer loaded")
    return health


def test_predict(flow: dict, defense: str) -> None:
    fi = flow["flowIndex"]
    try:
        r = post_json(f"{API}/predict", {"flow_index": fi, "defense": defense, "model": "transformer"})
    except Exception as exc:
        fail(f"predict flow={fi} defense={defense}: {exc}")
        return
    if not r.get("live"):
        fail(f"predict flow={fi}: live flag false")
    if r.get("true_app_id") != flow["classId"]:
        fail(
            f"flow #{fi}: true_app_id={r.get('true_app_id')} != demo classId={flow['classId']}"
        )
    else:
        ok(f"flow #{fi} ({defense}): true App {r['true_app_id']}, pred App {r['predicted_app_id']}, correct={r['correct']}")
    preds = r.get("predictions") or []
    if not preds or "confidence" not in preds[0]:
        fail(f"flow #{fi}: missing predictions")
    elif not (0 < preds[0]["confidence"] <= 1):
        fail(f"flow #{fi}: bad confidence {preds[0]['confidence']}")


def test_classifier_matrix(flows: list[dict]) -> None:
    print("\n[3] Classifier /predict (all demo flows)")
    for flow in flows:
        test_predict(flow, "baseline")
    print("\n[4] Python obfuscation /flow_vis")
    test_flow_vis(flows[0], "jitter_low")
    test_flow_vis(flows[0], "mtu")
    test_predict(flows[0], "jitter_low")
    r = post_json(f"{API}/predict", {"flow_index": flows[0]["flowIndex"], "defense": "baseline", "model": "bilstm"})
    ok(f"bilstm predict flow #{flows[0]['flowIndex']} correct={r.get('correct')}")


def test_flow_vis(flow: dict, defense: str) -> None:
    fi = flow["flowIndex"]
    try:
        vis = post_json(f"{API}/flow_vis", {"flow_index": fi, "defense": defense})
    except Exception as exc:
        fail(f"flow_vis flow={fi} defense={defense}: {exc}")
        return
    if vis.get("source") != "phase3/obfuscator.py":
        fail(f"flow_vis missing source tag")
    if vis.get("class_id") != flow["classId"]:
        fail(f"flow_vis class_id mismatch")
    else:
        ok(f"flow_vis flow #{fi} ({defense}): {len(vis['clean']['sizes'])} slots from Python")


def test_random_flow() -> None:
    print("\n[5] Random flow + load by index")
    try:
        meta = json.loads(get(f"{API}/random_flow")[1].decode())
        ok(f"random_flow -> #{meta['flow_index']} app {meta['class_id']}")
        info = post_json(f"{API}/flow_info", {"flow_index": 3714})
        if info.get("class_id") != 29:
            fail(f"flow_info class_id={info.get('class_id')}")
        else:
            ok("flow_info flow #3714 -> App 29")
    except Exception as exc:
        fail(f"random/flow_info: {exc}")


def test_cors_preflight() -> None:
    print("\n[6] CORS (browser fetch)")
    req = urllib.request.Request(f"{API}/health", method="OPTIONS")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            origin = resp.headers.get("Access-Control-Allow-Origin")
            if origin != "*":
                fail(f"CORS Allow-Origin={origin!r}")
            else:
                ok("OPTIONS /health returns Access-Control-Allow-Origin: *")
    except Exception as exc:
        fail(f"CORS preflight: {exc}")


def main() -> None:
    print("Presentation smoke test")
    if not DEMO_JS.is_file():
        fail(f"missing {DEMO_JS}")
        sys.exit(1)
    flows = parse_demo_flows()
    if len(flows) < 12:
        fail(f"expected 12 demo flows, parsed {len(flows)}")
    else:
        ok(f"parsed {len(flows)} demo flows from demo_flows.js")

    test_static()
    if test_api_health():
        test_classifier_matrix(flows)
        test_cors_preflight()
        test_random_flow()

    print("\n" + ("=" * 50))
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} issue(s)):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL SMOKE CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
