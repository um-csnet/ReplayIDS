#!/usr/bin/env python3
"""Fail if an EAAI table source, manifest, or bundled result is missing."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    contract_path = ROOT / "configs/paper/eaai_expected.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    required = [
        ROOT / "configs/data/cicids2017.yaml",
        ROOT / "configs/data/features.yaml",
        ROOT / "configs/experiments/eaai-primary.yaml",
        ROOT / "configs/experiments/eaai-attacks.yaml",
        ROOT / "configs/experiments/eaai-cross-architecture.yaml",
        ROOT / "results/eaai-reported/primary/baselines.csv",
        ROOT / "results/eaai-reported/primary/er_results.csv",
        ROOT / "results/eaai-reported/primary/per_experience.csv",
        ROOT / "benchmarks/cross_architecture/results/summary_ci.csv",
        ROOT / "benchmarks/cross_architecture/results/summary_cii.csv",
        ROOT / "benchmarks/cross_architecture/results/icarl/summary_ci.csv",
        ROOT / "benchmarks/cross_architecture/results/icarl/summary_cii.csv",
    ]
    required.extend((ROOT / "results/eaai-reported/asr-logs").glob("backdoor_*.txt"))
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing EAAI reproduction sources: " + ", ".join(missing))
    table_keys = set(contract["tables"])
    expected_keys = {f"table_{number:02d}" for number in range(3, 15)}
    if table_keys != expected_keys:
        raise SystemExit(
            f"Paper map covers {sorted(table_keys)}; expected {sorted(expected_keys)}"
        )
    print(
        json.dumps(
            {"status": "ok", "tables": sorted(table_keys), "sources": len(required)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
