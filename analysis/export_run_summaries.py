#!/usr/bin/env python3
"""Convert completed primary run.json artifacts into paper-builder input CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/paper/eaai_expected.yaml"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_tag(strategy: str, scenario: str, memory: int) -> str:
    variant = "balanced" if strategy == "er_balanced" else "strat"
    return f"er_{variant}_{scenario.lower()}_m{memory}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rerun-primary")
    args = parser.parse_args()

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    oracles = contract["paper"]["oracles"]
    baselines: list[dict] = []
    replay: list[dict] = []
    per_experience: list[dict] = []
    sources = []

    for path in sorted(args.runs.glob("*/run.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        config = artifact["configuration"]
        strategy = config["strategy"]
        if strategy == "oracle" or strategy in {"label_flip", "backdoor"}:
            continue
        metrics = artifact.get("metrics", {})
        if artifact.get("return_code") != 0 or not metrics.get("complete"):
            raise ValueError(f"Primary run is incomplete: {path}")

        scenario = "CI" if int(config["scenario"]) == 1 else "CII"
        checkpoints = metrics["checkpoints"]
        if scenario == "CI":
            accuracy = checkpoints[-1]["accuracy"]
            macro_f1 = checkpoints[-1]["macro_f1"]
        else:
            accuracy = sum(item["accuracy"] for item in checkpoints) / len(checkpoints)
            macro_f1 = sum(item["macro_f1"] for item in checkpoints) / len(checkpoints)
        bwt_acc = metrics["bwt_accuracy"]
        bwt_f1 = metrics["bwt_macro_f1"]
        common = {
            "scenario": scenario,
            "acc": round(accuracy, 4),
            "f1": round(macro_f1, 4),
            "bwt_acc": bwt_acc,
            "bwt_f1": bwt_f1,
            "forgetting_acc": round(max(0.0, -bwt_acc), 4),
            "forgetting_f1": round(max(0.0, -bwt_f1), 4),
            "intrans_acc": round(oracles[scenario]["accuracy"] - accuracy, 4),
            "intrans_f1": round(oracles[scenario]["macro_f1"] - macro_f1, 4),
        }
        if strategy in {"naive", "ewc", "lwf", "icarl"}:
            label = {"naive": "Naive", "ewc": "EWC", "lwf": "LwF", "icarl": "iCaRL"}[
                strategy
            ]
            baselines.append({"strategy": label, **common})
            tag = label
        elif strategy in {"er_stratified", "er_balanced"}:
            memory = int(config["memory"])
            tag = run_tag(strategy, scenario, memory)
            replay.append({"run_tag": tag, "memory_budget": memory, **common})
        else:
            raise ValueError(f"Unsupported primary strategy {strategy!r} in {path}")
        per_experience.extend(
            {
                "run_tag": tag,
                "scenario": scenario,
                "experience": checkpoint["experience"],
                "acc": checkpoint["accuracy"],
                "f1": checkpoint["macro_f1"],
            }
            for checkpoint in checkpoints
        )
        sources.append(
            {"run_id": artifact["run_id"], "git_commit": artifact.get("git_commit")}
        )

    if not baselines and not replay:
        raise ValueError(f"No completed primary run.json files found below {args.runs}")
    args.output.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "scenario",
        "acc",
        "f1",
        "bwt_acc",
        "bwt_f1",
        "forgetting_acc",
        "forgetting_f1",
        "intrans_acc",
        "intrans_f1",
    ]
    write_csv(args.output / "baselines.csv", baselines, ["strategy", *metric_fields])
    write_csv(
        args.output / "er_results.csv",
        replay,
        ["run_tag", "memory_budget", *metric_fields],
    )
    write_csv(
        args.output / "per_experience.csv",
        per_experience,
        ["run_tag", "scenario", "experience", "acc", "f1"],
    )
    (args.output / "provenance.json").write_text(
        json.dumps({"schema_version": 1, "sources": sources}, indent=2),
        encoding="utf-8",
    )
    print(
        f"Exported {len(baselines)} baseline, {len(replay)} replay, and "
        f"{len(per_experience)} checkpoint rows to {args.output}"
    )


if __name__ == "__main__":
    main()
