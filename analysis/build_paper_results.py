#!/usr/bin/env python3
"""Build the compact EAAI table bundle from canonical ReplayIDS artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results/eaai-reported/primary"
EXPECTED = ROOT / "configs/paper/eaai_expected.yaml"
CROSS = ROOT / "benchmarks/cross_architecture/results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path.name}")
    names = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def combined(primary: Path, scenario: str) -> list[dict[str, str]]:
    baseline = [
        row
        for row in read_csv(primary / "baselines.csv")
        if row["scenario"] == scenario
    ]
    replay = [
        row
        for row in read_csv(primary / "er_results.csv")
        if row["scenario"] == scenario
    ]
    return baseline + replay


def expected_rows(contract: dict, table: str) -> list[dict]:
    return contract["tables"][table]["expected"]


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=PRIMARY)
    parser.add_argument("--output", type=Path, default=ROOT / "results/generated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    contract = yaml.safe_load(EXPECTED.read_text(encoding="utf-8"))
    oracle_ci = contract["paper"]["oracles"]["CI"]
    oracle_cii = contract["paper"]["oracles"]["CII"]
    table06 = [
        {
            "strategy": "Joint (Oracle)",
            "scenario": "CI",
            "acc": oracle_ci["accuracy"],
            "f1": oracle_ci["macro_f1"],
        }
    ] + combined(args.primary, "CI")
    table07 = [
        {
            "strategy": "Joint (Oracle)",
            "scenario": "CII",
            "acc": oracle_cii["accuracy"],
            "f1": oracle_cii["macro_f1"],
        }
    ] + combined(args.primary, "CII")
    write_csv(args.output / "table06_ci.csv", table06)
    write_csv(args.output / "table07_cii_checkpoint_average.csv", table07)

    per_exp = read_csv(args.primary / "per_experience.csv")
    table08 = [
        {
            "run_tag": "Joint (Oracle)",
            "scenario": "CII",
            "experience": "joint",
            "acc": oracle_cii["accuracy"],
            "f1": oracle_cii["macro_f1"],
        }
    ] + [
        row for row in per_exp if row["scenario"] == "CII" and row["experience"] == "3"
    ]
    write_csv(args.output / "table08_cii_final.csv", table08)

    table09 = []
    for row in table06 + table07:
        strategy = row.get("strategy", "")
        run_tag = row.get("run_tag", "")
        if strategy in {"Naive", "EWC", "LwF", "iCaRL"}:
            table09.append(row)
        elif run_tag in {
            "er_balanced_ci_m10",
            "er_strat_cii_m10",
            "er_balanced_cii_m10",
        }:
            table09.append(row)
    write_csv(args.output / "table09_forgetting_intransigence.csv", table09)

    table10 = expected_rows(contract, "table_10")
    table11 = expected_rows(contract, "table_11")
    table12 = expected_rows(contract, "table_12")
    write_csv(args.output / "table10_label_flip.csv", table10)
    write_csv(args.output / "table11_backdoor.csv", table11)
    write_csv(args.output / "table12_backdoor_asr_historical.csv", table12)

    table13 = read_csv(CROSS / "summary_ci.csv") + read_csv(
        CROSS / "icarl/summary_ci.csv"
    )
    table14 = read_csv(CROSS / "summary_cii.csv") + read_csv(
        CROSS / "icarl/summary_cii.csv"
    )
    write_csv(args.output / "table13_cross_architecture_ci.csv", table13)
    write_csv(args.output / "table14_cross_architecture_cii.csv", table14)

    report = {
        "paper": contract["paper"],
        "status": "verified-reference-bundle",
        "tables": {
            "6": len(table06),
            "7": len(table07),
            "8": len(table08),
            "9": len(table09),
            "10": len(table10),
            "11": len(table11),
            "12": len(table12),
            "13": len(table13),
            "14": len(table14),
        },
        "notes": [
            "Tables 6-9 come from the released primary run summaries.",
            "Tables 10-12 preserve the EAAI-reported historical attack values.",
            "Table 12 includes three flagged estimates above 100%; see docs/PAPER_MAP.md.",
            "Tables 13-14 come from the independent cross-architecture benchmark.",
        ],
    }
    (args.output / "verification.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    md = [
        "# EAAI Results Reference",
        "",
        "Generated by `analysis/build_paper_results.py`. Do not edit generated tables by hand.",
        "",
        "## Table 10 — Label flipping",
        "",
        markdown_table(
            table10,
            ["budget", "accuracy", "macro_f1", "forgetting_acc", "forgetting_f1"],
        ),
        "",
        "## Table 11 — Backdoor aggregate performance",
        "",
        markdown_table(
            table11,
            ["budget", "accuracy", "macro_f1", "forgetting_acc", "forgetting_f1"],
        ),
        "",
        "## Table 12 — Historical backdoor ASR",
        "",
        markdown_table(
            table12, ["budget", "experience", "target", "asr_percent", "flagged"]
        ),
        "",
        "Primary and cross-architecture tables are emitted as CSV files in this directory.",
    ]
    (args.output / "RESULTS_REFERENCE.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
