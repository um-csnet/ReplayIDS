"""
data_prep.py — CIC-IDS-2017 -> CI and CII experience streams (paper's Table 3).

Canonical 8-class label space (matches ReplayIDS config / paper Table 3):
  0 Benign, 1 DoS GoldenEye, 2 DoS Hulk, 3 DoS Slowhttptest,
  4 DoS slowloris, 5 FTP-Patator, 6 Heartbleed, 7 SSH-Patator

Experience partitioning (Table 3):
  CI  : E0{0,1} E1{2,3} E2{4,5} E3{6,7}         (Benign only in E0)
  CII : E0{0,1} E1{0,2,3} E2{0,4,5} E3{0,6,7}   (Benign anchored, new instances/exp)

Preprocessing: clean inf/NaN, then StandardScaler fit on the union of all
training data (offline standardisation, as in the paper's *_standardised.csv),
transform all. Output: results/prepared_ci.npz and results/prepared_cii.npz,
each with per-experience train/test arrays + a combined (all-class) test set.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "raw" / "CICIDS2017"
RES = Path(__file__).resolve().parents[1] / "results"
FILES = ["Tuesday-WorkingHours.pcap_ISCX.csv", "Wednesday-workingHours.pcap_ISCX.csv"]

LABELS = {
    "BENIGN": 0,
    "DoS GoldenEye": 1,
    "DoS Hulk": 2,
    "DoS Slowhttptest": 3,
    "DoS slowloris": 4,
    "FTP-Patator": 5,
    "Heartbleed": 6,
    "SSH-Patator": 7,
}
CLASS_NAMES = [k for k, _ in sorted(LABELS.items(), key=lambda kv: kv[1])]
NUM_CLASSES = 8

CI_EXP = [[0, 1], [2, 3], [4, 5], [6, 7]]
CII_EXP = [[0, 1], [0, 2, 3], [0, 4, 5], [0, 6, 7]]
BENIGN = 0
TEST_SIZE = 0.40
MAX_TRAIN_PER_CLASS = (
    50000  # cap majority classes in TRAIN (benign 500k, Hulk 138k) so the
)
# transformer is tractable and attack-class forgetting stays measurable;
# test split is left full for honest evaluation.
SEED = 42


def load_all(data_dir=DATA):
    frames = []
    for f in FILES:
        df = pd.read_csv(data_dir / f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        lc = df.columns[-1]
        df[lc] = df[lc].str.strip()
        frames.append(df[df[lc].isin(LABELS)])
    df = pd.concat(frames, ignore_index=True)
    lc = df.columns[-1]
    y = df[lc].map(LABELS).to_numpy(np.int64)
    X = df.drop(columns=[lc]).to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y[keep]


def per_class_split(X, y):
    """train/test indices per class (stratified trivially by splitting each class)."""
    tr, te = {}, {}
    rng = np.random.default_rng(SEED)
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(len(idx) * TEST_SIZE))
        te[c] = idx[:n_te]
        tr_c = idx[n_te:]
        if len(tr_c) > MAX_TRAIN_PER_CLASS:  # cap majority classes in TRAIN only
            tr_c = rng.choice(tr_c, size=MAX_TRAIN_PER_CLASS, replace=False)
        tr[c] = tr_c
    return tr, te


def build_stream(X, y, tr, te, exp_classes, anchor_benign):
    """Return per-experience (Xtr,ytr,Xte,yte). Benign instances are partitioned
    across experiences when anchor_benign (CII); otherwise Benign lives in E0 (CI)."""
    n = len(exp_classes)
    # split benign indices into n disjoint chunks for CII anchoring
    btr = np.array_split(tr[BENIGN], n) if anchor_benign else None
    bte = np.array_split(te[BENIGN], n) if anchor_benign else None
    stream = []
    for e, classes in enumerate(exp_classes):
        tri, tei = [], []
        for c in classes:
            if c == BENIGN:
                tri.append(btr[e] if anchor_benign else tr[BENIGN])
                tei.append(bte[e] if anchor_benign else te[BENIGN])
            else:
                tri.append(tr[c])
                tei.append(te[c])
        tri, tei = np.concatenate(tri), np.concatenate(tei)
        stream.append((X[tri], y[tri], X[tei], y[tei]))
    return stream


def save_stream(path, stream):
    d = {
        "n_experiences": np.int64(len(stream)),
        "n_features": np.int64(stream[0][0].shape[1]),
    }
    Xte_all, yte_all = [], []
    for e, (Xtr, ytr, Xte, yte) in enumerate(stream):
        d[f"Xtr_{e}"] = Xtr.astype(np.float32)
        d[f"ytr_{e}"] = ytr
        d[f"Xte_{e}"] = Xte.astype(np.float32)
        d[f"yte_{e}"] = yte
        Xte_all.append(Xte)
        yte_all.append(yte)
    d["Xte_all"] = np.vstack(Xte_all).astype(np.float32)
    d["yte_all"] = np.concatenate(yte_all)
    np.savez_compressed(path, **d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DATA)
    args = parser.parse_args()
    RES.mkdir(exist_ok=True)
    print("Loading Tuesday + Wednesday ...")
    X, y = load_all(args.raw_dir)
    print(f"  total {X.shape}, per-class = {np.bincount(y, minlength=8)}")
    tr, te = per_class_split(X, y)

    # scaler fit on union of all training indices
    all_tr = np.concatenate([tr[c] for c in range(NUM_CLASSES)])
    scaler = StandardScaler().fit(X[all_tr])
    X = scaler.transform(X).astype(np.float32)

    for name, exp, anchor in [("ci", CI_EXP, False), ("cii", CII_EXP, True)]:
        stream = build_stream(X, y, tr, te, exp, anchor)
        save_stream(RES / f"prepared_{name}.npz", stream)
        print(f"\n{name.upper()} stream:")
        for e, (Xtr, ytr, Xte, yte) in enumerate(stream):
            print(
                f"  E{e} classes={sorted(set(ytr.tolist()))} "
                f"train={len(ytr)} test={len(yte)}  train_counts={np.bincount(ytr, minlength=8)}"
            )
    # drop the stale 2-task file from the previous design
    (RES / "prepared.npz").unlink(missing_ok=True)
    print("\nSaved prepared_ci.npz, prepared_cii.npz")


if __name__ == "__main__":
    main()
