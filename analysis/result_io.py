"""Read ReplayIDS text logs into structured, versionable metrics."""

from __future__ import annotations

import re
from pathlib import Path


EXP = re.compile(
    r"Completed experience (\d+) - Overall accuracy: ([\d.]+), Macro F1: ([\d.]+)"
)
BWT_ACC = re.compile(r"Overall BWT \(Accuracy\): (-?[\d.]+)")
BWT_F1 = re.compile(r"Overall BWT \(F1-Score\): (-?[\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    checkpoints = [
        {"experience": int(exp), "accuracy": float(acc), "macro_f1": float(f1)}
        for exp, acc, f1 in EXP.findall(text)
    ]
    bwt_acc = BWT_ACC.search(text)
    bwt_f1 = BWT_F1.search(text)
    return {
        "complete": bool(checkpoints and bwt_acc and bwt_f1),
        "checkpoints": checkpoints,
        "final": checkpoints[-1] if checkpoints else None,
        "bwt_accuracy": float(bwt_acc.group(1)) if bwt_acc else None,
        "bwt_macro_f1": float(bwt_f1.group(1)) if bwt_f1 else None,
    }
