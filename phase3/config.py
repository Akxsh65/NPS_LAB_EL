"""Phase 3 paths and obfuscation defaults."""
import os

# Repo-relative paths (run scripts from phase3/)
PHASE1_ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "phase1", "artifacts"))
PHASE3_ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "artifacts"))

TEST_TENSORS = os.path.join(PHASE1_ARTIFACTS, "test_tensors.pt")
IPT_SCALER = os.path.join(PHASE1_ARTIFACTS, "ipt_scaler.pkl")

SEQ_LEN = 30
MTU = 1500.0
IPT_CLIP_MAX = 1_000_000.0
LOG_EPS = 1e-8

# Linear padding block size (bytes)
LINEAR_BLOCK = 128

# Laplace jitter scales (milliseconds) — tune for your experiments
JITTER_SCALES = {
    "low": 1.0,
    "medium": 5.0,
    "high": 20.0,
}

SEED = 42

os.makedirs(PHASE3_ARTIFACTS, exist_ok=True)
