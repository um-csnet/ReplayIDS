"""
Generate the three reference CSV files from experiment log files.

Usage:
    python export_results_csv.py [--logdir <path>] [--outdir <path>]

    --logdir: directory containing the run .log files
              default: /home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs
    --outdir: directory to write CSVs into
              default: v2/results (relative to repo root, resolved automatically)

Output files:
    baselines.csv       — Naive, EWC, LwF, iCaRL (CI and CII)
    er_results.csv      — 12 ER runs (balanced + stratified × CI/CII × b=1,5,10)
    per_experience.csv  — per-experience Acc/F1 for all 20 runs

Metric definitions:
    CI  Acc = overall_accuracy at final experience (Table 5)
    CII Acc = mean(overall_accuracy) across all 4 experiences (Table 6)
    Forgetting = max(0, -BWT)
    Intransigence = oracle_acc - method_acc  (can be negative)
"""

import csv
import os
import re
import sys
import argparse

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DEFAULT_OUT = os.path.join(_REPO, "results")

_p = argparse.ArgumentParser()
_p.add_argument(
    "--logdir",
    default="/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs",
)
_p.add_argument("--outdir", default=_DEFAULT_OUT)
_args = _p.parse_args()
LOGDIR = _args.logdir
OUTDIR = _args.outdir

# ---------------------------------------------------------------------------
# Oracle values (joint training, seed=42, RTX 3090)
# ---------------------------------------------------------------------------
ORACLE_CI_ACC = 0.9997
ORACLE_CI_F1 = 0.9979
ORACLE_CII_ACC = 0.9955
ORACLE_CII_F1 = 0.9795


# ---------------------------------------------------------------------------
# Log parsing (same regex as extract_er_metrics.py)
# ---------------------------------------------------------------------------
def parse_log(path):
    if not os.path.exists(path):
        return None
    content = open(path, errors="replace").read()
    exp_matches = re.findall(
        r"Completed experience \d+ - Overall accuracy: ([\d.]+), Macro F1: ([\d.]+)",
        content,
    )
    if not exp_matches:
        return None
    bwt_acc_m = re.search(r"Overall BWT \(Accuracy\): (-?[\d.]+)", content)
    bwt_f1_m = re.search(r"Overall BWT \(F1-Score\): (-?[\d.]+)", content)
    if not bwt_acc_m or not bwt_f1_m:
        return None
    return dict(
        overall_accs=[float(m[0]) for m in exp_matches],
        overall_f1s=[float(m[1]) for m in exp_matches],
        bwt_acc=float(bwt_acc_m.group(1)),
        bwt_f1=float(bwt_f1_m.group(1)),
        n_exp=len(exp_matches),
    )


def ci_metrics(d):
    acc = d["overall_accs"][-1]
    f1 = d["overall_f1s"][-1]
    forg_acc = max(0.0, -d["bwt_acc"])
    forg_f1 = max(0.0, -d["bwt_f1"])
    return dict(
        acc=acc,
        f1=f1,
        bwt_acc=d["bwt_acc"],
        bwt_f1=d["bwt_f1"],
        forg_acc=forg_acc,
        forg_f1=forg_f1,
        intr_acc=round(ORACLE_CI_ACC - acc, 4),
        intr_f1=round(ORACLE_CI_F1 - f1, 4),
    )


def cii_metrics(d):
    acc = sum(d["overall_accs"]) / d["n_exp"]
    f1 = sum(d["overall_f1s"]) / d["n_exp"]
    forg_acc = max(0.0, -d["bwt_acc"])
    forg_f1 = max(0.0, -d["bwt_f1"])
    return dict(
        acc=acc,
        f1=f1,
        bwt_acc=d["bwt_acc"],
        bwt_f1=d["bwt_f1"],
        forg_acc=forg_acc,
        forg_f1=forg_f1,
        intr_acc=round(ORACLE_CII_ACC - acc, 4),
        intr_f1=round(ORACLE_CII_F1 - f1, 4),
    )


def r4(v):
    return round(v, 4)


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------
BASELINES = [
    # (strategy_label, scenario, log_tag)
    ("Naive", "CI", "valid_naive_ci"),
    ("Naive", "CII", "tabT_naive_cii"),
    ("EWC", "CI", "tabT_ewc_ci"),
    ("EWC", "CII", "tabT_ewc_cii"),
    ("LwF", "CI", "tabT_lwf_ci"),
    ("LwF", "CII", "tabT_lwf_cii"),
    ("iCaRL", "CI", "tabT_icarl_ci"),
    ("iCaRL", "CII", "tabT_icarl_cii"),
]

ER_RUNS = [
    # (run_tag, scenario, memory_budget)
    ("er_balanced_ci_m1", "CI", 1),
    ("er_balanced_ci_m5", "CI", 5),
    ("er_balanced_ci_m10", "CI", 10),
    ("er_strat_ci_m1", "CI", 1),
    ("er_strat_ci_m5", "CI", 5),
    ("er_strat_ci_m10", "CI", 10),
    ("er_balanced_cii_m1", "CII", 1),
    ("er_balanced_cii_m5", "CII", 5),
    ("er_balanced_cii_m10", "CII", 10),
    ("er_strat_cii_m1", "CII", 1),
    ("er_strat_cii_m5", "CII", 5),
    ("er_strat_cii_m10", "CII", 10),
]

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)
errors = []

# --- baselines.csv ---
baseline_rows = []
for strategy, scenario, tag in BASELINES:
    path = os.path.join(LOGDIR, f"{tag}.log")
    data = parse_log(path)
    if data is None:
        errors.append(f"MISSING/INCOMPLETE: {tag}.log")
        continue
    m = ci_metrics(data) if scenario == "CI" else cii_metrics(data)
    baseline_rows.append(
        [
            strategy,
            scenario,
            r4(m["acc"]),
            r4(m["f1"]),
            r4(m["bwt_acc"]),
            r4(m["bwt_f1"]),
            r4(m["forg_acc"]),
            r4(m["forg_f1"]),
            r4(m["intr_acc"]),
            r4(m["intr_f1"]),
        ]
    )

bpath = os.path.join(OUTDIR, "baselines.csv")
with open(bpath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "strategy",
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
    )
    w.writerows(baseline_rows)
print(f"Wrote {len(baseline_rows)} rows → {bpath}")

# --- er_results.csv and per_experience.csv ---
er_rows = []
exp_rows = []

all_runs = [(tag, "CI" if "_ci_" in tag else "CII", b) for tag, _, b in ER_RUNS]

for tag, scenario, mem in ER_RUNS:
    path = os.path.join(LOGDIR, f"{tag}.log")
    data = parse_log(path)
    if data is None:
        errors.append(f"MISSING/INCOMPLETE: {tag}.log")
        continue
    m = ci_metrics(data) if scenario == "CI" else cii_metrics(data)
    er_rows.append(
        [
            tag,
            scenario,
            mem,
            r4(m["acc"]),
            r4(m["f1"]),
            r4(m["bwt_acc"]),
            r4(m["bwt_f1"]),
            r4(m["forg_acc"]),
            r4(m["forg_f1"]),
            r4(m["intr_acc"]),
            r4(m["intr_f1"]),
        ]
    )
    for exp_idx, (acc, f1) in enumerate(zip(data["overall_accs"], data["overall_f1s"])):
        exp_rows.append([tag, scenario, exp_idx, r4(acc), r4(f1)])

erpath = os.path.join(OUTDIR, "er_results.csv")
with open(erpath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "run_tag",
            "scenario",
            "memory_budget",
            "acc",
            "f1",
            "bwt_acc",
            "bwt_f1",
            "forgetting_acc",
            "forgetting_f1",
            "intrans_acc",
            "intrans_f1",
        ]
    )
    w.writerows(er_rows)
print(f"Wrote {len(er_rows)} rows → {erpath}")

# Also add baseline per-experience rows
for strategy, scenario, tag in BASELINES:
    path = os.path.join(LOGDIR, f"{tag}.log")
    data = parse_log(path)
    if data is None:
        continue
    for exp_idx, (acc, f1) in enumerate(zip(data["overall_accs"], data["overall_f1s"])):
        exp_rows.append([strategy, scenario, exp_idx, r4(acc), r4(f1)])

# Sort: baselines first (by insertion order above), then ER runs
# (already in order since we appended baselines after ER)
# Reorder: baselines then ER
baseline_exp = [r for r in exp_rows if not r[0].startswith("er_")]
er_exp = [r for r in exp_rows if r[0].startswith("er_")]
exp_rows_ordered = baseline_exp + er_exp

ppath = os.path.join(OUTDIR, "per_experience.csv")
with open(ppath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_tag", "scenario", "experience", "acc", "f1"])
    w.writerows(exp_rows_ordered)
print(f"Wrote {len(exp_rows_ordered)} rows → {ppath}")

if errors:
    print("\nWARNINGS:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("\nAll CSVs generated from log files. No missing runs.")
