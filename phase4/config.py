"""Phase 4 paths and evaluation defaults."""
import os

PHASE1_ARTIFACTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "phase1", "artifacts")
)
PHASE2_ARTIFACTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "phase2", "artifacts")
)
PHASE3_ARTIFACTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "phase3", "artifacts")
)
PHASE4_RESULTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "results"))

TEST_TENSORS = os.path.join(PHASE1_ARTIFACTS, "test_tensors.pt")
LABEL_ENCODER = os.path.join(PHASE1_ARTIFACTS, "label_encoder.pkl")
MANIFEST = os.path.join(PHASE3_ARTIFACTS, "obfuscation_manifest.json")

# Production model (masked Transformer d_model=160)
DEFAULT_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "transformer_production.pt")
MASKED_D160_CHECKPOINT = os.path.join(
    PHASE2_ARTIFACTS,
    "refine/architecture/run_masked_d160/transformer_masked_best_acc.pt",
)
MASKED_D160_CONFIG = os.path.join(
    PHASE2_ARTIFACTS,
    "refine/architecture/run_masked_d160/transformer_masked_config.json",
)
DEFAULT_MODEL_CONFIG = MASKED_D160_CONFIG

FALLBACK_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "transformer_best_acc.pt")
FALLBACK_CONFIG = None  # resolved next to checkpoint if present

# CNN-BiLSTM attack baseline (Tier C architecture comparison)
CNN_BILSTM_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "cnn_bilstm_best.pt")
CNN_BILSTM_ALT_CHECKPOINT = os.path.join(PHASE2_ARTIFACTS, "cnn_bilstm", "cnn_bilstm_best.pt")
CNN_BILSTM_CONFIG = os.path.join(PHASE2_ARTIFACTS, "cnn_bilstm_config.json")
CNN_BILSTM_ALT_CONFIG = os.path.join(PHASE2_ARTIFACTS, "cnn_bilstm", "cnn_bilstm_config.json")

# Tensor channels (Phase 1): 0=IPT, 1=DIR, 2=SIZE
CHANNEL_IPT = 0
CHANNEL_DIR = 1
CHANNEL_SIZE = 2
CHANNEL_ABLATION_PRESETS = {
    "all": [0, 1, 2],
    "ipt_only": [0],
    "dir_only": [1],
    "size_only": [2],
    "ipt_dir": [0, 1],
}

PREDICTIONS_DIR = os.path.join(PHASE4_RESULTS, "predictions")

BATCH_SIZE = 1024
NUM_WORKERS = 4
MODEL_NAME = "transformer_masked"

# Clean test baseline from Phase 2 finalize (for summary annotation)
REFERENCE_TEST_ACCURACY = 0.7777
REFERENCE_TEST_MACRO_F1 = 0.7441

os.makedirs(PHASE4_RESULTS, exist_ok=True)
