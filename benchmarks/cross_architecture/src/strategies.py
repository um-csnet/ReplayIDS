"""
strategies.py — the four CL strategies as experience-stream trainers.

Each strategy trains across a sequence of experiences (Table 3). Task-0 is plain
weighted CE for all; strategies differ in how later experiences are trained and
what state they carry forward:

  Naive  : nothing carried; sequential fine-tuning (lower bound).
  Replay : a class-balanced buffer of past samples, interleaved into each step.
  EWC    : per-experience diagonal Fisher penalties (separate EWC).
  LwF    : frozen previous-experience teacher, KD on all previously seen classes.

Class-weighted CE keeps rare attacks (Heartbleed, slowloris) learnable so that
forgetting is measurable.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

NUM_CLASSES = 8


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _weights(y, device):
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    w = np.clip(
        w, None, 20.0
    )  # cap so a 7-sample class (Heartbleed) can't dominate the loss
    full = np.zeros(NUM_CLASSES, np.float32)
    full[present] = w
    return torch.tensor(full, device=device)


def _train(model, X, y, device, cfg, loss_fn):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(
        ds, batch_size=cfg["batch"], shuffle=True, num_workers=2, pin_memory=True
    )
    for ep in range(cfg["epochs"]):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(model, logits, xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # tame tiny-experience blow-ups
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"      epoch {ep + 1}/{cfg['epochs']} loss={tot / len(X):.4f}")


def _fisher(model, X, y, device, n=4096, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    ce = nn.CrossEntropyLoss()
    fisher = {
        n_: torch.zeros_like(p) for n_, p in model.named_parameters() if p.requires_grad
    }
    model.eval()
    for i in idx:
        model.zero_grad()
        xb = torch.from_numpy(X[i : i + 1]).to(device)
        yb = torch.tensor([y[i]], device=device)
        ce(model(xb), yb).backward()
        for n_, p in model.named_parameters():
            if p.grad is not None:
                fisher[n_] += p.grad.detach() ** 2
    for n_ in fisher:
        fisher[n_] /= len(idx)
    star = {
        n_: p.detach().clone() for n_, p in model.named_parameters() if p.requires_grad
    }
    return fisher, star


# ---------------- strategies ----------------


class Naive:
    def __init__(self, cfg):
        self.cfg = cfg

    def train_experience(self, model, X, y, seen, device):
        ce = nn.CrossEntropyLoss(weight=_weights(y, device))
        _train(model, X, y, device, self.cfg, lambda m, lg, xb, yb: ce(lg, yb))

    def after_experience(self, model, X, y, device):
        pass


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bufX = np.empty((0, 0), np.float32)
        self.bufY = np.empty((0,), np.int64)

    def train_experience(self, model, X, y, seen, device):
        if len(self.bufY):
            X = np.vstack([X, self.bufX])
            y = np.concatenate([y, self.bufY])
        ce = nn.CrossEntropyLoss(weight=_weights(y, device))
        _train(model, X, y, device, self.cfg, lambda m, lg, xb, yb: ce(lg, yb))

    def after_experience(self, model, X, y, device):
        # add a class-balanced random subset of this experience to the buffer
        rng = np.random.default_rng(self.cfg["seed"])
        per = max(1, int(len(X) * self.cfg["buffer_frac"] / max(1, len(np.unique(y)))))
        pick = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            pick.append(rng.choice(ci, size=min(per, len(ci)), replace=False))
        pick = np.concatenate(pick)
        addX, addY = X[pick], y[pick]
        self.bufX = addX if self.bufX.size == 0 else np.vstack([self.bufX, addX])
        self.bufY = np.concatenate([self.bufY, addY])
        print(f"      replay buffer now {len(self.bufY)} samples")


class EWC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = []  # list of (fisher, star)

    def train_experience(self, model, X, y, seen, device):
        ce = nn.CrossEntropyLoss(weight=_weights(y, device))
        lam = self.cfg["ewc_lambda"]
        tasks = self.tasks

        def loss_fn(m, lg, xb, yb):
            pen = 0.0
            for fisher, star in tasks:
                for n_, p in m.named_parameters():
                    if n_ in fisher:
                        pen = pen + (fisher[n_] * (p - star[n_]) ** 2).sum()
            return ce(lg, yb) + 0.5 * lam * pen

        # ponytail: ewc_lambda is the tuning knob if forgetting stays high; sweep {1e3,1e4,1e5}
        _train(model, X, y, device, self.cfg, loss_fn)

    def after_experience(self, model, X, y, device):
        self.tasks.append(_fisher(model, X, y, device, seed=self.cfg["seed"]))
        print(
            f"      EWC: stored Fisher for {len(self.tasks)} task(s), lambda={self.cfg['ewc_lambda']:g}"
        )


class LwF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.teacher = None
        self.seen = []

    def train_experience(self, model, X, y, seen, device):
        ce = nn.CrossEntropyLoss(weight=_weights(y, device))
        a, T = self.cfg["lwf_alpha"], self.cfg["lwf_T"]
        teacher, old = self.teacher, self.seen
        if teacher is None or not old:
            _train(model, X, y, device, self.cfg, lambda m, lg, xb, yb: ce(lg, yb))
            return
        old_t = torch.tensor(sorted(old), device=device)

        def loss_fn(m, lg, xb, yb):
            with torch.no_grad():
                t_old = teacher(xb).index_select(1, old_t)
            s_old = lg.index_select(1, old_t)
            kd = F.kl_div(
                F.log_softmax(s_old / T, 1),
                F.softmax(t_old / T, 1),
                reduction="batchmean",
            ) * (T * T)
            return (1 - a) * ce(lg, yb) + a * kd

        print(f"      LwF: distilling on seen classes {sorted(old)} (alpha={a}, T={T})")
        _train(model, X, y, device, self.cfg, loss_fn)

    def after_experience(self, model, X, y, device):
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.seen = sorted(set(self.seen) | set(np.unique(y).tolist()))


class ICaRL:
    """iCaRL (Rebuffi et al. 2017): herding exemplars + BCE distillation + nearest-mean-of-
    exemplars (NME) classification in feature space. Reference: avalanche.training.ICaRL."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.exemplars = {}  # class_id -> raw feature rows (m, F)
        self.class_means = {}  # class_id -> L2-normalised feature mean (D,)
        self.teacher = None
        self.old_classes = []

    def _feats(self, model, X, device, batch=4096):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i : i + batch]).to(device)
                out.append(model.get_features(xb).cpu().numpy())
        return np.vstack(out)

    @staticmethod
    def _herding(feats_norm, m):
        mu = feats_norm.mean(0)
        w = mu.copy()
        chosen = []
        for _ in range(min(m, len(feats_norm))):
            i = int(np.argmax(feats_norm @ w))
            chosen.append(i)
            w = w + mu - feats_norm[i]
        return np.array(chosen)

    def train_experience(self, model, X, y, seen, device):
        if self.exemplars:
            bX = np.vstack(list(self.exemplars.values()))
            bY = np.concatenate(
                [np.full(len(v), c, np.int64) for c, v in self.exemplars.items()]
            )
            X = np.vstack([X, bX])
            y = np.concatenate([y, bY])
        teacher, old = self.teacher, self.old_classes

        def loss_fn(m, lg, xb, yb):
            tgt = F.one_hot(yb.long(), NUM_CLASSES).float()
            if teacher is not None and old:
                with torch.no_grad():
                    old_sig = torch.sigmoid(teacher(xb))
                oi = torch.tensor(old, device=xb.device)
                tgt[:, oi] = old_sig[:, oi]  # soft distillation targets for old classes
            return F.binary_cross_entropy_with_logits(lg, tgt)

        _train(model, X, y, device, self.cfg, loss_fn)

    def after_experience(self, model, X, y, device):
        K = self.cfg["icarl_memory"]
        new = [c for c in np.unique(y).tolist() if c not in self.exemplars]
        m = max(1, K // (len(self.exemplars) + len(new)))
        for c in new:  # herd new-class exemplars
            Xc = X[y == c]
            f = self._feats(model, Xc, device)
            fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            self.exemplars[c] = Xc[self._herding(fn, m)]
        for c in list(self.exemplars):  # prune old sets to new budget
            self.exemplars[c] = self.exemplars[c][:m]
        self.class_means = {}  # recompute means with updated extractor
        for c, Xc in self.exemplars.items():
            f = self._feats(model, Xc, device)
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            mu = f.mean(0)
            mu = mu / (np.linalg.norm(mu) + 1e-8)
            self.class_means[c] = torch.tensor(mu, dtype=torch.float32)
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.old_classes = sorted(self.exemplars.keys())
        total = sum(len(v) for v in self.exemplars.values())
        print(
            f"      iCaRL: {total} exemplars over {len(self.class_means)} classes (m={m}/class)"
        )

    def predict_fn(self, model, xb):  # NME: nearest L2-normalised class mean
        model.eval()
        with torch.no_grad():
            f = F.normalize(model.get_features(xb), dim=1)
        classes = sorted(self.class_means.keys())
        means = torch.stack([self.class_means[c].to(xb.device) for c in classes])
        loc = (f @ means.T).argmax(1).cpu().tolist()
        return torch.tensor([classes[i] for i in loc], device=xb.device)


STRATEGIES = {"naive": Naive, "replay": Replay, "ewc": EWC, "lwf": LwF, "icarl": ICaRL}
