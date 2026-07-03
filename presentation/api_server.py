"""
Local inference API for the presentation UI.

Run from repo root:
  python presentation/api_server.py

Endpoints:
  GET  /health
  GET  /random_flow
  POST /predict      {"flow_index": 16420, "defense": "jitter_low", "model": "transformer"}
  POST /flow_vis     {"flow_index": 16420, "defense": "jitter_low"}
  POST /flow_info    {"flow_index": 16420}
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "phase1" / "artifacts"
PHASE2 = ROOT / "phase2" / "artifacts"
PHASE3 = ROOT / "phase3"
PHASE4 = ROOT / "phase4"

# phase1/config must win while loading obfuscator (phase4 also has config.py).
sys.path.insert(0, str(PHASE1.parent))
for p in (PHASE3, PHASE2):
    s = str(p)
    if s not in sys.path:
        sys.path.append(s)

from adaptive_policies import get_policy  # noqa: E402
from obfuscator import (  # noqa: E402
    active_packet_mask,
    denormalize_flow,
    load_ipt_scaler_from,
    obfuscate,
)
from settings import JITTER_SCALES, SEED  # noqa: E402

# obfuscator loads phase1 `config` into sys.modules; evaluate needs phase4 `config`.
sys.modules.pop("config", None)
sys.path.insert(0, str(PHASE4))
from evaluate import load_model  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765

TRANSFORMER_CKPT = PHASE2 / "transformer_production.pt"
TRANSFORMER_CFG = PHASE2 / "transformer_masked_config.json"
BILSTM_CKPT = PHASE2 / "cnn_bilstm_best.pt"
TEST_TENSORS = PHASE1 / "test_tensors.pt"
LABEL_ENCODER = PHASE1 / "label_encoder.pkl"

# Population metrics for UI footers (phase4 architecture_comparison.csv)
POPULATION_METRICS: dict[str, dict[str, dict[str, float]]] = {
    "transformer": {
        "baseline": {"accuracy": 77.77, "macro_f1": 74.41},
        "jitter_low": {"accuracy": 76.84, "macro_f1": 72.72},
        "jitter_medium": {"accuracy": 68.83, "macro_f1": 63.01},
        "jitter_high": {"accuracy": 54.68, "macro_f1": 47.66},
        "linear128": {"accuracy": 59.06, "macro_f1": 62.01},
        "linear128_jitter_medium": {"accuracy": 52.30, "macro_f1": 49.99},
        "mtu": {"accuracy": 2.14, "macro_f1": 0.30},
        "mtu_jitter_medium": {"accuracy": 2.97, "macro_f1": 0.37},
    },
    "bilstm": {
        "baseline": {"accuracy": 72.75, "macro_f1": 67.40},
        "jitter_low": {"accuracy": 70.98, "macro_f1": 65.10},
        "jitter_medium": {"accuracy": 57.93, "macro_f1": 50.37},
        "jitter_high": {"accuracy": 35.96, "macro_f1": 29.56},
        "linear128": {"accuracy": 66.81, "macro_f1": 63.14},
        "linear128_jitter_medium": {"accuracy": 50.57, "macro_f1": 43.88},
        "mtu": {"accuracy": 2.88, "macro_f1": 1.64},
        "mtu_jitter_medium": {"accuracy": 2.56, "macro_f1": 1.19},
    },
}


def population_metrics(defense_id: str, model_key: str) -> dict[str, float]:
    model_metrics = POPULATION_METRICS.get(model_key, POPULATION_METRICS["transformer"])
    return model_metrics.get(defense_id, model_metrics["baseline"])


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        if not TEST_TENSORS.is_file():
            raise FileNotFoundError(f"Missing {TEST_TENSORS}")
        data = torch.load(TEST_TENSORS, map_location="cpu", weights_only=False)
        self.X: torch.Tensor = data["X"].float()
        self.y: torch.Tensor = data["y"].long()
        self.n_flows = int(self.X.shape[0])
        print(f"Loaded test tensors: {self.n_flows} flows")

        self.label_encoder = joblib.load(LABEL_ENCODER)
        self.ipt_mean, self.ipt_std = load_ipt_scaler_from()

        self.models: dict[str, torch.nn.Module] = {}
        self._load_model("transformer", str(TRANSFORMER_CKPT), str(TRANSFORMER_CFG))
        if BILSTM_CKPT.is_file():
            self._load_model("bilstm", str(BILSTM_CKPT), None)
        else:
            print("BiLSTM checkpoint not found — /predict?model=bilstm will fail.")

    def _load_model(
        self, key: str, checkpoint: str, config: Optional[str]
    ) -> None:
        attack = "cnn_bilstm" if key == "bilstm" else "transformer"
        model, _, _ = load_model(
            checkpoint,
            self.device,
            config_path=config,
            attack_model=attack,
        )
        self.models[key] = model
        print(f"Loaded {key} from {checkpoint}")

    def _defense_params(self, defense_id: str) -> tuple[str, float]:
        if defense_id == "baseline":
            return "none", 0.0
        policy = get_policy(defense_id)
        jitter_scale = (
            0.0 if policy.jitter_key == "none" else JITTER_SCALES[policy.jitter_key]
        )
        return policy.padding_type, float(jitter_scale)

    def _prepare_flow(self, flow_index: int, defense_id: str) -> np.ndarray:
        if flow_index < 0 or flow_index >= self.n_flows:
            raise ValueError(f"flow_index must be in [0, {self.n_flows - 1}]")
        flow = self.X[flow_index].numpy()
        if defense_id == "baseline":
            return flow
        padding, jitter_scale = self._defense_params(defense_id)
        obf, _meta = obfuscate(
            flow,
            padding_type=padding,  # type: ignore[arg-type]
            jitter_scale=jitter_scale,
            flow_index=flow_index,
            seed=SEED,
        )
        return obf

    def _flow_meta(self, flow_index: int) -> dict[str, Any]:
        if flow_index < 0 or flow_index >= self.n_flows:
            raise ValueError(f"flow_index must be in [0, {self.n_flows - 1}]")
        class_index = int(self.y[flow_index].item())
        app_id = int(self.label_encoder.classes_[class_index])
        clean = self.X[flow_index].numpy()
        ipt, dir_seq, size = denormalize_flow(clean, self.ipt_mean, self.ipt_std)
        mask = active_packet_mask(dir_seq, size)
        return {
            "flow_index": flow_index,
            "class_index": class_index,
            "class_id": app_id,
            "active_packets": int(mask.sum()),
            "test_flows": self.n_flows,
        }

    def _packets_payload(
        self, ipt: np.ndarray, dir_seq: np.ndarray, size: np.ndarray, mask: np.ndarray
    ) -> dict[str, Any]:
        return {
            "sizes": [int(round(v)) for v in size.tolist()],
            "ipts": [round(float(v), 2) for v in ipt.tolist()],
            "dirs": [int(v) for v in dir_seq.tolist()],
            "mask": [bool(v) for v in mask.tolist()],
        }

    def flow_visualization(self, flow_index: int, defense_id: str) -> dict[str, Any]:
        meta = self._flow_meta(flow_index)
        clean = self.X[flow_index].numpy()
        ipt_c, dir_c, size_c = denormalize_flow(clean, self.ipt_mean, self.ipt_std)
        mask = active_packet_mask(dir_c, size_c)

        obf_norm = self._prepare_flow(flow_index, defense_id)
        ipt_o, dir_o, size_o = denormalize_flow(obf_norm, self.ipt_mean, self.ipt_std)

        return {
            **meta,
            "defense": defense_id,
            "source": "phase3/obfuscator.py",
            "clean": self._packets_payload(ipt_c, dir_c, size_c, mask),
            "obfuscated": self._packets_payload(ipt_o, dir_o, size_o, mask),
        }

    def random_flow_index(self) -> int:
        return random.randint(0, self.n_flows - 1)

    @torch.no_grad()
    def baseline_correct(self, flow_index: int, model_key: str = "transformer") -> bool:
        result = self.predict(flow_index, "baseline", model_key, top_k=1)
        return bool(result["correct"])

    @torch.no_grad()
    def predict(
        self,
        flow_index: int,
        defense_id: str = "baseline",
        model_key: str = "transformer",
        top_k: int = 5,
    ) -> dict[str, Any]:
        if model_key not in self.models:
            raise ValueError(f"Unknown model '{model_key}'. Loaded: {list(self.models)}")

        flow = self._prepare_flow(flow_index, defense_id)
        x = torch.from_numpy(flow).unsqueeze(0).to(self.device)
        logits = self.models[model_key](x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        true_idx = int(self.y[flow_index].item())
        true_app_id = int(self.label_encoder.classes_[true_idx])
        pred_idx = int(np.argmax(probs))

        k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:k]

        predictions = []
        for rank, cls_idx in enumerate(top_indices, start=1):
            app_id = int(self.label_encoder.classes_[int(cls_idx)])
            predictions.append(
                {
                    "rank": rank,
                    "class_index": int(cls_idx),
                    "app_id": app_id,
                    "label": f"App {app_id}",
                    "confidence": float(probs[int(cls_idx)]),
                }
            )

        pop = population_metrics(defense_id, model_key)
        return {
            "flow_index": flow_index,
            "defense": defense_id,
            "model": model_key,
            "true_class_index": true_idx,
            "true_app_id": true_app_id,
            "predicted_class_index": pred_idx,
            "predicted_app_id": int(self.label_encoder.classes_[pred_idx]),
            "correct": pred_idx == true_idx,
            "predictions": predictions,
            "reported_accuracy": pop["accuracy"],
            "macro_f1": pop["macro_f1"],
            "live": True,
        }

    @torch.no_grad()
    def batch_predict(
        self,
        defense_id: str = "baseline",
        model_key: str = "transformer",
        count: int = 100,
        seed: int = 42,
    ) -> dict[str, Any]:
        if model_key not in self.models:
            raise ValueError(f"Unknown model '{model_key}'")
        count = max(1, min(int(count), self.n_flows))
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_flows, size=count, replace=False)
        correct = 0
        for idx in indices:
            flow = self._prepare_flow(int(idx), defense_id)
            x = torch.from_numpy(flow).unsqueeze(0).to(self.device)
            logits = self.models[model_key](x)
            pred_idx = int(logits.argmax(dim=1).item())
            true_idx = int(self.y[int(idx)].item())
            if pred_idx == true_idx:
                correct += 1
        pop = population_metrics(defense_id, model_key)
        acc = 100.0 * correct / count
        return {
            "defense": defense_id,
            "model": model_key,
            "n": count,
            "correct": correct,
            "accuracy": acc,
            "reported_accuracy": pop["accuracy"],
            "macro_f1": pop["macro_f1"],
            "seed": seed,
            "live": True,
        }


SERVICE: Optional[InferenceService] = None


def get_service() -> InferenceService:
    global SERVICE
    if SERVICE is None:
        SERVICE = InferenceService()
    return SERVICE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} {fmt % args}")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            try:
                svc = get_service()
                self._json(
                    200,
                    {
                        "status": "ok",
                        "api_version": 2,
                        "device": str(svc.device),
                        "models": list(svc.models.keys()),
                        "test_flows": svc.n_flows,
                        "features": ["flow_vis", "random_flow", "flow_info", "batch_predict"],
                    },
                )
            except Exception as exc:
                self._json(500, {"status": "error", "detail": str(exc)})
            return
        if path == "/random_flow":
            try:
                svc = get_service()
                idx = svc.random_flow_index()
                meta = svc._flow_meta(idx)
                meta["baseline_correct_transformer"] = svc.baseline_correct(idx, "transformer")
                self._json(200, meta)
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        if path == "/predict":
            try:
                flow_index = int(data["flow_index"])
                defense = str(data.get("defense", "baseline"))
                model = str(data.get("model", "transformer"))
                if model == "cnn_bilstm":
                    model = "bilstm"
                result = get_service().predict(flow_index, defense, model)
                self._json(200, result)
            except KeyError as exc:
                self._json(400, {"error": f"missing field: {exc}"})
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})
            return

        if path == "/flow_vis":
            try:
                flow_index = int(data["flow_index"])
                defense = str(data.get("defense", "baseline"))
                result = get_service().flow_visualization(flow_index, defense)
                self._json(200, result)
            except KeyError as exc:
                self._json(400, {"error": f"missing field: {exc}"})
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})
            return

        if path == "/flow_info":
            try:
                flow_index = int(data["flow_index"])
                meta = get_service()._flow_meta(flow_index)
                self._json(200, meta)
            except KeyError as exc:
                self._json(400, {"error": f"missing field: {exc}"})
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})
            return

        if path == "/batch_predict":
            try:
                defense = str(data.get("defense", "baseline"))
                model = str(data.get("model", "transformer"))
                if model == "cnn_bilstm":
                    model = "bilstm"
                count = int(data.get("count", 100))
                seed = int(data.get("seed", 42))
                result = get_service().batch_predict(defense, model, count, seed)
                self._json(200, result)
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})
            return

        self._json(404, {"error": "not found"})


def main() -> None:
    print(f"Starting inference API on http://{HOST}:{PORT}")
    print("  GET  /health")
    print("  GET  /random_flow")
    print("  POST /predict")
    print("  POST /flow_vis")
    print("  POST /flow_info")
    print("  POST /batch_predict")
    get_service()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
