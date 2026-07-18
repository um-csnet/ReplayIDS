#!/usr/bin/env python3
"""Render deterministic SVG counterparts of EAAI Figures 4 and 7-10."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#176B87", "#64CCC5", "#DA7B29", "#8E5EA2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_label(row: dict[str, str]) -> str:
    if row.get("strategy"):
        return row["strategy"]
    tag = row.get("run_tag", "")
    label = tag.replace("er_balanced", "ER-Balanced").replace(
        "er_strat", "ER-Stratified"
    )
    return label.replace("_ci", "").replace("_cii", "").replace("_m", " b=")


def grouped_bars(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    output: Path,
) -> None:
    width = max(980, 105 + len(labels) * 74)
    height = 570
    left, right, top, bottom = 72, 24, 55, 165
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(labels)
    bar_w = min(19, group_w * 0.75 / len(series))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Arial,sans-serif;fill:#17252A}"
        ".title{font-size:20px;font-weight:700}.tick{font-size:12px}"
        ".label{font-size:11px}.legend{font-size:12px}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">'
        f"{html.escape(title)}</text>",
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#DCE6E8"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 9}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value:.1f}</text>'
        )
    for index, label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        for series_index, (_, values) in enumerate(series):
            value = max(0.0, min(1.0, values[index]))
            x = center + (series_index - (len(series) - 1) / 2) * bar_w
            y = top + plot_h * (1 - value)
            parts.append(
                f'<rect x="{x - bar_w * 0.44:.1f}" y="{y:.1f}" '
                f'width="{bar_w * 0.88:.1f}" height="{top + plot_h - y:.1f}" '
                f'fill="{COLORS[series_index]}"/>'
            )
        parts.append(
            f'<text class="label" transform="translate({center + 3:.1f},'
            f'{top + plot_h + 12}) rotate(55)" text-anchor="start">'
            f"{html.escape(label)}</text>"
        )
    legend_x = left
    for index, (name, _) in enumerate(series):
        x = legend_x + index * 155
        parts.extend(
            [
                f'<rect x="{x}" y="{height - 24}" width="13" height="13" '
                f'fill="{COLORS[index]}"/>',
                f'<text class="legend" x="{x + 19}" y="{height - 13}">'
                f"{html.escape(name)}</text>",
            ]
        )
    parts.append("</svg>\n")
    output.write_text("\n".join(parts), encoding="utf-8")


def scenario_distribution(output: Path) -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/data/cicids2017.yaml").read_text(encoding="utf-8")
    )
    counts = {
        int(key): value for key, value in contract["expected"]["class_counts"].items()
    }
    ci = [[0, 1], [2, 3], [4, 5], [6, 7]]
    cii = [[0, 1], [0, 2, 3], [0, 4, 5], [0, 6, 7]]
    labels, ci_values, cii_values = [], [], []
    for exp in range(4):
        labels.append(f"E{exp}")
        ci_values.append(float(sum(counts[item] for item in ci[exp])))
        cii_values.append(
            float(sum(counts[item] for item in cii[exp] if item != 0) + counts[0] / 4)
        )
    maximum = max(ci_values + cii_values)
    grouped_bars(
        "CICIDS2017 experience distribution (relative flow count)",
        labels,
        [
            ("CI", [value / maximum for value in ci_values]),
            ("CII", [value / maximum for value in cii_values]),
        ],
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results/generated")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/generated/figures"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_distribution(args.output_dir / "figure04_distribution.svg")

    for number, filename, title in (
        (7, "table06_ci.csv", "CI performance by continual-learning method"),
        (8, "table07_cii_checkpoint_average.csv", "CII checkpoint-average performance"),
    ):
        rows = read_csv(args.tables_dir / filename)
        grouped_bars(
            title,
            [method_label(row) for row in rows],
            [
                ("Accuracy", [float(row["acc"]) for row in rows]),
                ("Macro-F1", [float(row["f1"]) for row in rows]),
            ],
            args.output_dir / f"figure{number:02d}_results.svg",
        )

    backdoor = read_csv(args.tables_dir / "table11_backdoor.csv")
    grouped_bars(
        "Backdoor aggregate performance",
        [
            "Clean" if row["budget"] == "0" else f"p={row['budget']}%"
            for row in backdoor
        ],
        [
            ("Accuracy", [float(row["accuracy"]) for row in backdoor]),
            ("Macro-F1", [float(row["macro_f1"]) for row in backdoor]),
        ],
        args.output_dir / "figure09_backdoor.svg",
    )

    ci = read_csv(args.tables_dir / "table13_cross_architecture_ci.csv")
    cii = read_csv(args.tables_dir / "table14_cross_architecture_cii.csv")
    labels = [f"{row['model'].upper()}-{row['strategy']}" for row in ci]
    grouped_bars(
        "Cross-architecture CI/CII performance",
        labels,
        [
            ("CI accuracy", [float(row["overall_acc"]) for row in ci]),
            ("CI macro-F1", [float(row["macro_f1"]) for row in ci]),
            ("CII accuracy", [float(row["overall_acc"]) for row in cii]),
            ("CII macro-F1", [float(row["macro_f1"]) for row in cii]),
        ],
        args.output_dir / "figure10_cross_architecture.svg",
    )
    print(f"Wrote five deterministic SVG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
