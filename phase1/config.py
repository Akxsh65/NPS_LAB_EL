# config.py
# ============================================================
# Central configuration for Phase 1 — CPU-only Windows setup
# All other modules import from here. Change values here only.
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
DATASET_ROOT  = "./data/cesnet_quic22"      # cesnet-datazoo download cache
ARTIFACTS_DIR = "./artifacts"               # saved scalers, encoders, tensors
LOGS_DIR      = "./logs"                    # validation + diagnostic logs

# ── Dataset config ───────────────────────────────────────────
TRAIN_PERIOD  = "W-2022-44"   # 32.6M flows, Oct 31 – Nov 6 2022
TEST_PERIOD   = "W-2022-45"   # 42.6M flows, held-out — DO NOT TOUCH until Phase 4
DATASET_SIZE  = "XS"           # ~25M sample subset, safe for CPU RAM
TRAIN_VAL_SPLIT = 0.9         # 90% train / 10% val carved from W-2022-44

# ── Feature engineering ──────────────────────────────────────
SEQ_LEN       = 30            # first N packets per flow
MTU           = 1500.0        # maximum transmission unit in bytes
IPT_CLIP_MAX  = 1_000_000.0   # clip extreme IPT outliers before log transform
LOG_EPS       = 1e-8          # numerical stability in Z-score denominator

# ── DataLoader ───────────────────────────────────────────────
BATCH_SIZE    = 512
NUM_WORKERS   = 0             # MUST be 0 on Windows (multiprocessing limitation)
PIN_MEMORY    = False         # CPU-only: pin_memory has no effect, keep False

# ── Minimum samples per class ─────────────────────────────────
MIN_CLASS_SAMPLES = 200       # classes below this are dropped before training

# ── Random seed ──────────────────────────────────────────────
SEED = 42

# ── Create directories if they don't exist ───────────────────
for _dir in [DATASET_ROOT, ARTIFACTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
