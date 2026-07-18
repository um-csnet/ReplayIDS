"""
Extract ER run metrics from log files and compute table values for the manuscript.

Usage:
    python extract_er_metrics.py [--logdir <path>]

    --logdir: directory containing the run .log files
              default: /home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs

Prints a table of Acc, F1, Forgetting (Acc/F1), Intransigence (Acc/F1) for each run.
Reports INCOMPLETE for runs that haven't finished.

Metric definitions:
    CI  (Table 5): Acc = overall_accuracy at final experience
    CII (Table 6): Acc = mean(overall_accuracy across all experiences)
    Forgetting    = max(0, -BWT)  [positive BWT = forward transfer -> 0 forgetting]
    Intransigence = oracle_acc - method_acc  [can be negative if method beats oracle]
"""

import re
import sys
import os
import argparse

_p = argparse.ArgumentParser()
_p.add_argument(
    "--logdir",
    default="/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs",
)
_args = _p.parse_args()
LOGDIR = _args.logdir

# Oracle values from joint training (Azizi et al., TabTransformer pipeline)
ORACLE_CI_ACC = 0.9997
ORACLE_CI_F1 = 0.9979
ORACLE_CII_ACC = 0.9955
ORACLE_CII_F1 = 0.9795


def parse_log(path):
    """Return dict with per-experience accs/F1s and BWT values, or None if incomplete."""
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
    """Table 5 (CI) — final-experience accuracy."""
    acc = d["overall_accs"][-1]
    f1 = d["overall_f1s"][-1]
    forg_acc = max(0.0, -d["bwt_acc"])
    forg_f1 = max(0.0, -d["bwt_f1"])
    return dict(
        acc=acc,
        f1=f1,
        forg_acc=forg_acc,
        forg_f1=forg_f1,
        intr_acc=ORACLE_CI_ACC - acc,
        intr_f1=ORACLE_CI_F1 - f1,
    )


def cii_metrics(d):
    """Table 6 (CII) — mean accuracy across all experiences."""
    acc = sum(d["overall_accs"]) / d["n_exp"]
    f1 = sum(d["overall_f1s"]) / d["n_exp"]
    forg_acc = max(0.0, -d["bwt_acc"])
    forg_f1 = max(0.0, -d["bwt_f1"])
    return dict(
        acc=acc,
        f1=f1,
        forg_acc=forg_acc,
        forg_f1=forg_f1,
        intr_acc=ORACLE_CII_ACC - acc,
        intr_f1=ORACLE_CII_F1 - f1,
    )


def fmt(v):
    return f"{v:.4f}"


RUNS = [
    # (log_tag, scenario, mem, method_label)
    ("er_balanced_ci_m1", "ci", 1, "ER-Balanced"),
    ("er_balanced_ci_m5", "ci", 5, "ER-Balanced"),
    ("er_balanced_ci_m10", "ci", 10, "ER-Balanced"),
    ("er_strat_ci_m1", "ci", 1, "ER-Strat"),
    ("er_strat_ci_m5", "ci", 5, "ER-Strat"),
    ("er_strat_ci_m10", "ci", 10, "ER-Strat"),
    ("er_strat_cii_m1", "cii", 1, "ER-Strat"),
    ("er_strat_cii_m5", "cii", 5, "ER-Strat"),
    ("er_strat_cii_m10", "cii", 10, "ER-Strat"),
    ("er_balanced_cii_m1", "cii", 1, "ER-Balanced"),
    ("er_balanced_cii_m5", "cii", 5, "ER-Balanced"),
    ("er_balanced_cii_m10", "cii", 10, "ER-Balanced"),
]

print(
    f"{'Run':<25} {'Sc':3} {'b':3}  {'Acc':7} {'F1':7} {'ForgA':7} {'ForgF':7} {'IntrA':7} {'IntrF':7}  {'Status'}"
)
print("-" * 98)

all_done = True
for tag, sc, mem, method in RUNS:
    log_path = os.path.join(LOGDIR, f"{tag}.log")
    data = parse_log(log_path)
    if data is None:
        all_done = False
        exps = 0
        if os.path.exists(log_path):
            exps = len(
                re.findall(
                    r"Completed experience", open(log_path, errors="replace").read()
                )
            )
        print(
            f"{tag:<25} {sc:3} {mem:3}  {'—':7} {'—':7} {'—':7} {'—':7} {'—':7} {'—':7}  INCOMPLETE ({exps}/4 exp)"
        )
        continue
    m = ci_metrics(data) if sc == "ci" else cii_metrics(data)
    print(
        f"{tag:<25} {sc:3} {mem:3}  {fmt(m['acc'])} {fmt(m['f1'])} {fmt(m['forg_acc'])} {fmt(m['forg_f1'])} {fmt(m['intr_acc'])} {fmt(m['intr_f1'])}  OK ({data['n_exp']}/4 exp)"
    )

print()
if all_done:
    print("ALL RUNS COMPLETE — ready to update manuscript.")
    sys.exit(0)
else:
    print("Some runs still in progress.")
    sys.exit(1)
