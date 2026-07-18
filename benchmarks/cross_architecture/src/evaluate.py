"""evaluate.py — combined-test metrics + confusion matrix for a trained model."""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from models import NUM_CLASSES
from data_prep import CLASS_NAMES


@torch.no_grad()
def predict(model, X, device, batch=4096, predictor=None):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).to(device)
        if predictor is not None:  # iCaRL NME path
            out.append(predictor(model, xb).cpu().numpy())
        else:
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def acc_f1(model, X, y, device, predictor=None):
    """accuracy + macro-F1 over the classes present in this test set (for forgetting tracking)."""
    pred = predict(model, X, device, predictor=predictor)
    present = sorted(set(y.tolist()))
    return (
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
    )


def evaluate(model, X, y, device, predictor=None):
    """Return metrics dict incl. per-class recall (keyed by class name)."""
    pred = predict(model, X, device, predictor=predictor)
    labels = list(range(NUM_CLASSES))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_per_class": {CLASS_NAMES[i]: float(rec[i]) for i in labels},
        "_pred": pred,
    }


def save_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    # row-normalise (recall view) so rare classes are visible
    with np.errstate(all="ignore"):
        cmn = cm / cm.sum(1, keepdims=True)
    cmn = np.nan_to_num(cmn)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    f"{cmn[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cmn[i, j] > 0.5 else "black",
                    fontsize=6,
                )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
