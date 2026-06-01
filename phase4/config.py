"""Phase 4 paths and evaluation defaults."""
import os

PHASE1_ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "phase1", "artifacts"))
PHASE2_ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "phase2", "artifacts"))
PHASE3_ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "phase3", "artifacts"))
PHASE4_RESULTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "results"))

TEST_TENSORS = os.path.join(PHASE1_ARTIFACTS, "test_tensors.pt")
LABEL_ENCODER = os.path.join(PHASE1_ARTIFACTS, "label_encoder.pkl")
MANIFEST = os.path.join(PHASE3_ARTIFACTS, "obfuscation_manifest.json")

# Prefer best-acc checkpoint from Phase 2 training
DEFAULT_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "transformer_best_acc.pt")
FALLBACK_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "transformer_best.pt")

BATCH_SIZE = 1024
NUM_WORKERS = 0  # safe on Windows
MODEL_NAME = "transformer"

os.makedirs(PHASE4_RESULTS, exist_ok=True)
