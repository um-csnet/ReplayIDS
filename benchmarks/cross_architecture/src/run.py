"""
run.py — CI/CII benchmark: 2 protocols x 2 models x 4 strategies (+ oracle).

For each protocol (CI, CII) and model (MLP, CNN):
  * train a Joint/Oracle model on all experiences at once -> overall acc/F1
    (upper bound, used for intransigence),
  * for each strategy, train across the 4-experience stream, evaluating on every
    seen experience's test set after each step to build a forgetting matrix.

Reported per run (matching paper Table 5/6):
  overall accuracy, macro-F1 (combined test, after final experience),
  average forgetting (Acc, F1), intransigence (Oracle - overall, Acc & F1).

Outputs: results/summary_<protocol>.csv, results/metrics.json,
results/cm_<protocol>_<model>_<strategy>.png
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch

import models
from strategies import set_seed, STRATEGIES, _weights, _train
import torch.nn as nn
from evaluate import evaluate, acc_f1, save_confusion

RES = (
    Path(__file__).resolve().parents[1] / "results"
)  # prepared_*.npz always read from here
OUT = RES  # outputs (summary/metrics/cm) — set from --outdir


def load_stream(protocol):
    d = np.load(RES / f"prepared_{protocol}.npz")
    n = int(d["n_experiences"])
    exps = [
        (d[f"Xtr_{e}"], d[f"ytr_{e}"], d[f"Xte_{e}"], d[f"yte_{e}"]) for e in range(n)
    ]
    return exps, (d["Xte_all"], d["yte_all"]), int(d["n_features"])


def train_oracle(model_name, exps, nfeat, device, cfg):
    set_seed(cfg["seed"])
    net = models.build(model_name, nfeat).to(device)
    X = np.vstack([e[0] for e in exps])
    y = np.concatenate([e[1] for e in exps])
    ce = nn.CrossEntropyLoss(weight=_weights(y, device))
    _train(net, X, y, device, cfg, lambda m, lg, xb, yb: ce(lg, yb))
    return net


def avg_forgetting(R):
    """R[k][j] = metric on test j after experience k (j<=k). Mean over j of max_k R[k][j]-R[last][j]."""
    n = len(R)
    if n < 2:
        return 0.0
    last = n - 1
    vals = []
    for j in range(last):  # exclude the final experience (never revisited)
        seen = [R[k][j] for k in range(j, n) if R[k][j] is not None]
        vals.append(max(seen) - R[last][j])
    return float(np.mean(vals))


def run_stream(model_name, strat_name, exps, combined, nfeat, device, cfg):
    set_seed(cfg["seed"])
    net = models.build(model_name, nfeat).to(device)
    strat = STRATEGIES[strat_name](cfg)
    predictor = getattr(strat, "predict_fn", None)  # iCaRL NME; None -> softmax argmax
    n = len(exps)
    Racc = [[None] * n for _ in range(n)]
    Rf1 = [[None] * n for _ in range(n)]
    seen_classes = set()
    print(f"\n=== {cfg['protocol'].upper()} / {model_name.upper()} / {strat_name} ===")
    for k, (Xtr, ytr, _, _) in enumerate(exps):
        seen_classes |= set(np.unique(ytr).tolist())
        print(
            f"  [E{k}] classes={sorted(set(np.unique(ytr).tolist()))} train={len(ytr)}"
        )
        strat.train_experience(net, Xtr, ytr, sorted(seen_classes), device)
        strat.after_experience(
            net, Xtr, ytr, device
        )  # update state (buffer/fisher/teacher/means)
        for j in range(k + 1):  # eval after update (no weight change)
            a, f = acc_f1(net, exps[j][2], exps[j][3], device, predictor=predictor)
            Racc[k][j], Rf1[k][j] = a, f

    m = evaluate(net, combined[0], combined[1], device, predictor=predictor)
    save_confusion(
        combined[1],
        m["_pred"],
        f"{cfg['protocol'].upper()} / {model_name.upper()} / {strat_name}",
        OUT / f"cm_{cfg['protocol']}_{model_name}_{strat_name}.png",
    )
    m.pop("_pred")
    return {
        "protocol": cfg["protocol"],
        "model": model_name,
        "strategy": strat_name,
        "overall_accuracy": m["accuracy"],
        "overall_macro_f1": m["macro_f1"],
        "avg_forgetting_acc": avg_forgetting(Racc),
        "avg_forgetting_f1": avg_forgetting(Rf1),
        "recall_per_class": m["recall_per_class"],
        "acc_matrix": Racc,
        "f1_matrix": Rf1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["ci", "cii"])
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn"])
    ap.add_argument(
        "--strategies", nargs="+", default=["naive", "replay", "ewc", "lwf"]
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ewc-lambda", type=float, default=1e4)
    ap.add_argument("--buffer-frac", type=float, default=0.10)
    ap.add_argument("--lwf-alpha", type=float, default=0.5)
    ap.add_argument("--lwf-T", type=float, default=2.0)
    ap.add_argument("--icarl-memory", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=str(RES))
    a = ap.parse_args()
    device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    global OUT
    OUT = Path(a.outdir)
    OUT.mkdir(parents=True, exist_ok=True)

    base = dict(
        epochs=a.epochs,
        batch=a.batch,
        lr=a.lr,
        wd=a.wd,
        seed=a.seed,
        ewc_lambda=a.ewc_lambda,
        buffer_frac=a.buffer_frac,
        lwf_alpha=a.lwf_alpha,
        lwf_T=a.lwf_T,
        icarl_memory=a.icarl_memory,
    )
    print("Device:", device, "| config:", base)

    runs = []
    for proto in a.protocols:
        exps, combined, nfeat = load_stream(proto)
        cfg = {**base, "protocol": proto}
        # oracle per model (upper bound for intransigence)
        oracle = {}
        for mn in a.models:
            print(f"\n--- ORACLE {proto.upper()} / {mn.upper()} ---")
            onet = train_oracle(mn, exps, nfeat, device, cfg)
            om = evaluate(onet, combined[0], combined[1], device)
            om.pop("_pred")
            oracle[mn] = (om["accuracy"], om["macro_f1"])
            print(
                f"    oracle overall acc={om['accuracy']:.4f} f1={om['macro_f1']:.4f}"
            )
        for mn in a.models:
            for st in a.strategies:
                r = run_stream(mn, st, exps, combined, nfeat, device, cfg)
                oa, of = oracle[mn]
                r["intransigence_acc"] = oa - r["overall_accuracy"]
                r["intransigence_f1"] = of - r["overall_macro_f1"]
                r["oracle_accuracy"], r["oracle_macro_f1"] = oa, of
                print(
                    f"    overall acc={r['overall_accuracy']:.4f} f1={r['overall_macro_f1']:.4f} "
                    f"forget(acc)={r['avg_forgetting_acc']:.4f} intrans(acc)={r['intransigence_acc']:.4f}"
                )
                runs.append(r)

        # per-protocol summary CSV
        hdr = "model,strategy,overall_acc,macro_f1,avg_forget_acc,avg_forget_f1,intrans_acc,intrans_f1,oracle_acc"
        rows = [hdr]
        for r in runs:
            if r["protocol"] != proto:
                continue
            rows.append(
                ",".join(
                    str(x)
                    for x in [
                        r["model"],
                        r["strategy"],
                        f"{r['overall_accuracy']:.4f}",
                        f"{r['overall_macro_f1']:.4f}",
                        f"{r['avg_forgetting_acc']:.4f}",
                        f"{r['avg_forgetting_f1']:.4f}",
                        f"{r['intransigence_acc']:.4f}",
                        f"{r['intransigence_f1']:.4f}",
                        f"{r['oracle_accuracy']:.4f}",
                    ]
                )
            )
        (OUT / f"summary_{proto}.csv").write_text("\n".join(rows) + "\n")
        print(f"\n[{proto.upper()}] summary:\n" + "\n".join(rows))

    (OUT / "metrics.json").write_text(
        json.dumps({"config": base, "runs": runs}, indent=2)
    )
    print("\nWrote results/summary_*.csv, results/metrics.json, results/cm_*.png")


if __name__ == "__main__":
    main()
