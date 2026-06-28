"""Phase 3 paths and obfuscation defaults (named settings.py to avoid shadowing phase1/config.py)."""
import os

# Repo-relative paths (run scripts from phase3/)
PHASE1_ARTIFACTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "phase1", "artifacts")
)
PHASE3_ARTIFACTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "artifacts")
)

TEST_TENSORS = os.path.join(PHASE1_ARTIFACTS, "test_tensors.pt")
IPT_SCALER = os.path.join(PHASE1_ARTIFACTS, "ipt_scaler.pkl")

SEQ_LEN = 30
MTU = 1500.0
IPT_CLIP_MAX = 1_000_000.0
LOG_EPS = 1e-8

# Linear padding block size (bytes)
LINEAR_BLOCK = 128

# Laplace jitter scales (milliseconds) — one-sided (negative noise clipped to 0)
JITTER_SCALES = {
    "low": 1.0,
    "medium": 5.0,
    "high": 20.0,
}

SEED = 42
MANIFEST_VERSION = "2.0"

# Padding detection (Phase 1 contract: DIR=0 and SIZE=0 => no packet)
PAD_DIR_EPS = 0.5
PAD_SIZE_EPS = 1e-6

# Validation tolerances
ATOL_DIR_EXACT = 0.0
ATOL_SIZE_EXACT = 1e-5
ATOL_IPT_PAD = 1e-4
MAX_NAN_INF_FRACTION = 0.0

os.makedirs(PHASE3_ARTIFACTS, exist_ok=True)
