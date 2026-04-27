# dataset_loader.py
# ============================================================
# Phase 1.1 — Downloads and loads CESNET-QUIC22 via datazoo.
# Fixed for cesnet-datazoo==0.2.0 API.
#
# Key API changes from older versions:
#   - CESNET_QUIC22 is now in cesnet_datazoo.datasets (not .datasets.cesnet_quic22)
#   - Dataset is instantiated with (path, size=) directly
#   - Config is passed via dataset.set_dataset_config_and_initialize(config)
#   - DatasetConfig takes dataset= as first arg, not dataset_path=
# ============================================================
    # Add this import at the top of dataset_loader.py
import cesnet_datazoo
import pandas as pd
from config import (
    DATASET_ROOT, TRAIN_PERIOD, TEST_PERIOD,
    DATASET_SIZE, TRAIN_VAL_SPLIT, MIN_CLASS_SAMPLES
)


def load_raw_dataframes():
    """
    Downloads (first run) or loads from cache (subsequent runs) the
    CESNET-QUIC22 dataset and returns the three split DataFrames.

    Returns
    -------
    train_df : pd.DataFrame
    val_df   : pd.DataFrame
    test_df  : pd.DataFrame
    """
    try:
        from cesnet_datazoo.datasets import CESNET_QUIC22
        from cesnet_datazoo.config import DatasetConfig, AppSelection
    except ImportError:
        raise ImportError(
            "cesnet-datazoo is not installed or wrong version.\n"
            "Run:  pip install cesnet-datazoo==0.2.0"
        )

    # ── Step 1: Instantiate dataset object ───────────────────
    print("[1/4] Initialising CESNET-QUIC22 dataset object ...")
    print(f"      Data root : {DATASET_ROOT}")
    print(f"      Size tier : {DATASET_SIZE}  (~25M samples)\n")

    dataset = CESNET_QUIC22(
        data_root=DATASET_ROOT,
        size=DATASET_SIZE
    )
# Add these two lines right before set_dataset_config_and_initialize
    cesnet_datazoo.datasets.cesnet_dataset.TRAIN_DATALOADER_WORKERS = 0
    cesnet_datazoo.datasets.cesnet_dataset.TEST_DATALOADER_WORKERS = 0
    # ── Step 2: Build config and attach to dataset ────────────
    print("[2/4] Configuring train/val/test splits ...")
    dataset_config = DatasetConfig(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=TRAIN_PERIOD,
        test_period_name=TEST_PERIOD,
        val_approach="split-from-train",
        train_val_split_fraction=0.1,
        train_size=200_000,
        val_known_size=20_000,
        test_known_size=50_000,
        train_workers=0,
        val_workers=0,
        test_workers=0,
    )

    # ── Step 3: Initialise — triggers download if needed ─────
    print("[3/4] Running set_dataset_config_and_initialize ...")
    print("      (First run will download ~19 GB — this is normal)\n")
    dataset.set_dataset_config_and_initialize(dataset_config)

    # ── Step 4: Extract DataFrames ───────────────────────────
    print("[4/4] Fetching DataFrames ...")
    train_df = dataset.get_train_df()
    val_df   = dataset.get_val_df()
    test_df  = dataset.get_test_df()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Train samples : {len(train_df):>12,}")
    print(f"  Val   samples : {len(val_df):>12,}")
    print(f"  Test  samples : {len(test_df):>12,}")
    print(f"  Unique apps   : {train_df['APP'].nunique():>12,}")
    print(f"{'='*50}\n")

    _verify_columns(train_df)

    return train_df, val_df, test_df


def _verify_columns(df: pd.DataFrame):
    """
    Checks that the expected PPI columns exist.
    Column names may vary slightly by datazoo version — prints
    all available columns if expected ones are missing so you
    can update config.py accordingly.
    """
    print("Available columns:", list(df.columns))

    # The library may use slightly different column name casing
    # depending on the version — we check for com mon variants
    col_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if "IPT" in col_upper:
            col_map["PPI_IPT"] = col
        elif "DIR" in col_upper:
            col_map["PPI_DIR"] = col
        elif "SIZE" in col_upper or "LEN" in col_upper:
            col_map["PPI_SIZE"] = col
        elif col_upper in ("APP", "LABEL", "APPLICATION"):
            col_map["APP"] = col

    missing = {"PPI_IPT", "PPI_DIR", "PPI_SIZE", "APP"} - set(col_map.keys())
    if missing:
        print(f"\n⚠  Could not auto-detect columns for: {missing}")
        print("   Paste the 'Available columns' line above into the chat")
        print("   and the exact column names will be confirmed.\n")
    else:
        # Rename to canonical names so downstream code is consistent
        rename = {v: k for k, v in col_map.items() if v != k}
        if rename:
            df.rename(columns=rename, inplace=True)
            print(f"  Columns renamed: {rename}")
        print("  ✓ PPI columns verified: PPI_IPT, PPI_DIR, PPI_SIZE, APP\n")


if __name__ == "__main__":
    train_df, val_df, test_df = load_raw_dataframes()
    print("\nSample row (train):")
    print(train_df.iloc[0])