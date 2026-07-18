"""
Step 0: Merge raw CICIDS2017 CICFlowMeter CSVs into CICIDS2017_standardised.csv

This script is the step BEFORE tab-preprocessing.py.
Run it once to produce preprocess_csv/CICIDS2017/CICIDS2017_standardised.csv.

Input:  raw/CICIDS2017/  (directory containing the day-by-day CICFlowMeter CSVs)
Output: preprocess_csv/CICIDS2017/CICIDS2017_standardised.csv

The raw CSVs are NOT included in the repo (they are 1.6 GB total).
Download from: https://www.unb.ca/cic/datasets/ids-2017.html
  -> "GeneratedLabelledFlows.zip" (CICFlowMeter-processed, ~1.6 GB)

We use only three day-files (the ones containing our 8 target classes):
  Monday-WorkingHours.pcap_ISCX.csv        -> Benign only
  Tuesday-WorkingHours.pcap_ISCX.csv       -> FTP-Patator, SSH-Patator
  Wednesday-workingHours.pcap_ISCX.csv     -> DoS variants, Heartbleed

Thursday / Friday files contain Web Attacks, Infiltration, Botnet, DDoS --
these classes are NOT in this paper and are excluded.

Usage:
    python scripts/prepare_cicids2017_legacy.py [--raw-dir <path>]

    --raw-dir: path to the folder containing the three day CSVs above
               default: raw/CICIDS2017
"""

import argparse
import os
import sys
import pandas as pd

# Class names in order (0-indexed) — must match configs/config.yaml
CLASSES = [
    "Benign",  # 0
    "DoS GoldenEye",  # 1
    "DoS Hulk",  # 2
    "DoS Slowhttptest",  # 3
    "DoS slowloris",  # 4
    "FTP-Patator",  # 5
    "Heartbleed",  # 6
    "SSH-Patator",  # 7
]

# Raw label strings as they appear in the UNB CICFlowMeter CSVs -> class index
LABEL_MAP = {
    "BENIGN": 0,
    "DoS GoldenEye": 1,
    "DoS Hulk": 2,
    "DoS Slowhttptest": 3,
    "DoS slowloris": 4,
    "FTP-Patator": 5,
    "Heartbleed": 6,
    "SSH-Patator": 7,
}

DAY_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

OUTPUT_DIR = "preprocess_csv/CICIDS2017"
OUTPUT_FILE = "CICIDS2017_standardised.csv"


def standardise(raw_dir: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    frames = []
    for fname in DAY_FILES:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        print(f"  Loading {fname} ...", end="", flush=True)
        df = pd.read_csv(fpath, low_memory=False)
        # Strip leading/trailing whitespace from column names (common UNB artefact)
        df.columns = df.columns.str.strip()
        print(f" {len(df):,} rows", flush=True)
        frames.append(df)

    if not frames:
        sys.exit("ERROR: no input files found. Check --raw-dir.")

    print("Concatenating ...", end="", flush=True)
    merged = pd.concat(frames, ignore_index=True)
    print(f" {len(merged):,} rows total")

    # Map string labels to integer class indices; drop unknown labels
    label_col = "Label"
    if label_col not in merged.columns:
        sys.exit(
            f"ERROR: column '{label_col}' not found. Columns: {list(merged.columns)}"
        )

    merged[label_col] = merged[label_col].str.strip()
    before = len(merged)
    merged[label_col] = merged[label_col].map(LABEL_MAP)
    merged = merged.dropna(subset=[label_col])
    merged[label_col] = merged[label_col].astype(int)
    after = len(merged)
    print(
        f"Label filtering: {before:,} -> {after:,} rows (dropped {before - after:,} unknown labels)"
    )

    print("Class distribution:")
    for idx, name in enumerate(CLASSES):
        count = (merged[label_col] == idx).sum()
        print(f"  {idx} {name}: {count:,}")

    print(f"Saving to {out_path} ...", end="", flush=True)
    merged.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f" {size_mb:.0f} MB")

    # Write catfeaturelist.npy — categorical column indices in the feature matrix.
    # These are the 13 columns that are flag counts or protocol fields (not continuous).
    # Indices validated against the actual CICIDS2017 schema; do not change without
    # re-verifying against the column order produced by this merge.
    import numpy as np

    cat_indices = [0, 31, 32, 33, 34, 43, 44, 45, 46, 47, 48, 49, 50]
    cat_path = os.path.join(OUTPUT_DIR, "catfeaturelist.npy")
    np.save(cat_path, cat_indices)
    print(f"Saved catfeaturelist.npy: {cat_indices}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="raw/CICIDS2017",
        help="Directory containing the three day-CSV files",
    )
    args = parser.parse_args()
    print(f"Raw CSV dir: {args.raw_dir}")
    standardise(args.raw_dir)
