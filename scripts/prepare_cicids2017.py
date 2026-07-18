#!/usr/bin/env python3
"""Build the exact CICIDS2017 arrays used by ReplayIDS.

The script owns the complete raw-CSV -> split-array transformation. It validates
the raw schema, class counts, feature names and split shapes, then writes both
the preprocessing workspace and the dataset directory consumed by ``main.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_indices(columns: list[str], names: list[str], group: str) -> list[int]:
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"Missing {group} feature columns: {missing}")
    return [columns.index(name) for name in names]


def class_counts(labels: pd.Series | np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def validate_existing(dataset_dir: Path, expected_shapes: dict) -> dict:
    report: dict[str, object] = {"dataset_dir": str(dataset_dir), "files": {}}
    for stem in (
        "train",
        "val",
        "test",
        "class_names",
        "catfeaturelist",
        "iatfeaturelist",
    ):
        path = dataset_dir / f"{stem}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path, allow_pickle=True)
        report["files"][path.name] = {
            "shape": list(array.shape),
            "sha256": sha256(path),
        }
    for stem, shape in expected_shapes.items():
        actual = report["files"][f"{stem}.npy"]["shape"]
        if list(shape) != actual:
            raise ValueError(f"{stem}.npy shape {actual}; expected {shape}")
    return report


def build(args: argparse.Namespace) -> dict:
    data_cfg = load_yaml(args.data_config)
    feature_cfg = load_yaml(args.feature_config)
    expected = data_cfg["expected"]
    raw_paths = [args.raw_dir / name for name in data_cfg["raw_files"]]
    missing = [str(path) for path in raw_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CICIDS2017 CSVs: " + ", ".join(missing)
        )

    frames = []
    raw_hashes = {}
    for path in raw_paths:
        print(f"Loading {path.name} ...")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.str.strip()
        frames.append(frame)
        raw_hashes[path.name] = sha256(path)

    merged = pd.concat(frames, ignore_index=True)
    label_column = data_cfg["label_column"]
    if label_column not in merged.columns:
        raise ValueError(f"Missing label column {label_column!r}")
    merged[label_column] = merged[label_column].astype(str).str.strip()
    merged[label_column] = merged[label_column].map(data_cfg["classes"])
    merged = merged.dropna(subset=[label_column]).copy()
    merged[label_column] = merged[label_column].astype(int)

    features = [column for column in merged.columns if column != label_column]
    if len(features) != int(expected["feature_count"]):
        raise ValueError(
            f"Found {len(features)} features; expected {expected['feature_count']}"
        )
    categorical = feature_indices(features, feature_cfg["categorical"], "categorical")
    iat = feature_indices(features, feature_cfg["backdoor_iat"], "backdoor IAT")

    observed_counts = class_counts(merged[label_column])
    expected_counts = {
        int(key): int(value) for key, value in expected["class_counts"].items()
    }
    if not args.allow_count_mismatch:
        if len(merged) != int(expected["merged_rows"]):
            raise ValueError(
                f"Merged row count {len(merged)}; expected {expected['merged_rows']}"
            )
        if observed_counts != expected_counts:
            raise ValueError(
                f"Class counts {observed_counts}; expected {expected_counts}"
            )

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.fillna(0, inplace=True)
    X = merged[features]
    y = merged[label_column]
    seed = int(data_cfg["split"]["seed"])
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    arrays = {
        "train": np.column_stack((X_train.to_numpy(), y_train.to_numpy())),
        "val": np.column_stack((X_val.to_numpy(), y_val.to_numpy())),
        "test": np.column_stack((X_test.to_numpy(), y_test.to_numpy())),
    }
    class_names = [
        name
        for name, _ in sorted(data_cfg["classes"].items(), key=lambda item: item[1])
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    for stem, array in arrays.items():
        np.save(args.output_dir / f"{stem}.npy", array)
    np.save(args.output_dir / "class_names.npy", np.asarray(class_names))
    np.save(args.output_dir / "catfeaturelist.npy", np.asarray(categorical, dtype=int))
    np.save(args.output_dir / "iatfeaturelist.npy", np.asarray(iat, dtype=int))

    produced = []
    for path in sorted(args.output_dir.glob("*.npy")):
        target = args.dataset_dir / path.name
        shutil.copy2(path, target)
        produced.append(target)

    report = {
        "dataset": data_cfg["dataset"],
        "seed": seed,
        "raw_sha256": raw_hashes,
        "merged_rows": len(merged),
        "features": features,
        "class_counts": observed_counts,
        "preprocessing": data_cfg["preprocessing"],
        "categorical_indices": categorical,
        "iat_indices": iat,
        "splits": {
            name: {
                "shape": list(array.shape),
                "class_counts": class_counts(array[:, -1]),
            }
            for name, array in arrays.items()
        },
        "artifacts": {path.name: sha256(path) for path in produced},
    }
    (args.dataset_dir / "data_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw/CICIDS2017")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "preprocess_csv/CICIDS2017"
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/CICIDS2017")
    parser.add_argument(
        "--data-config", type=Path, default=ROOT / "configs/data/cicids2017.yaml"
    )
    parser.add_argument(
        "--feature-config", type=Path, default=ROOT / "configs/data/features.yaml"
    )
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    if args.verify_only:
        report = validate_existing(
            args.dataset_dir, data_cfg["expected"]["split_shapes"]
        )
    else:
        report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
