"""
cl_strategies.py — LwF, EWC and iCaRL for the ReplayIDS TabTransformer.

These are additive, flag-guarded continual-learning strategies. They are adapted
from the sibling benchmark's strategies.py to ReplayIDS's:
  * (x_categ, x_cont) TabTransformer forward signature, and
  * experience-loop structure in main.py (adjust_model + train_model per exp).

Nothing here runs unless main.py selects strategy in {"lwf", "ewc", "icarl"};
the ER / Naive / LF / MP paths never import or touch this module.

Public surface used by main.py:
  LwFState   : make_aux_loss_fn(seen_classes) -> aux_loss_fn or None; update(model, seen)
  EWCState   : make_aux_loss_fn() -> aux_loss_fn or None; consolidate(model, loader, device)
  ICaRLState : make_aux_loss_fn(num_out) -> aux_loss_fn; update(...); predict(...) [NME]

aux_loss_fn signature matches train_model's contract:
    aux_loss_fn(model, logits, x_categ, x_cont, y) -> scalar tensor
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Feature extraction: penultimate activation = input to model.mlp.mlp[-1]
# --------------------------------------------------------------------------- #
class _FeatureHook:
    """Captures the input to the classifier head (the penultimate features)."""

    def __init__(self, model):
        self._feat = None
        self._handle = model.mlp.mlp[-1].register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        # forward_pre_hook receives args tuple; the head input is inp[0]
        self._feat = inp[0].detach()

    @property
    def features(self):
        return self._feat

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def extract_features(model, x_categ, x_cont, device):
    """Run a forward pass and return the penultimate features (B, D)."""
    hook = _FeatureHook(model)
    was_training = model.training
    model.eval()
    _ = model(x_categ.to(device), x_cont.to(device))
    feats = hook.features
    hook.remove()
    if was_training:
        model.train()
    return feats


# --------------------------------------------------------------------------- #
# LwF — frozen teacher + KL distillation on previously-seen-class logits
# --------------------------------------------------------------------------- #
class LwFState:
    def __init__(self, alpha=0.5, T=2.0):
        self.alpha = alpha
        self.T = T
        self.teacher = None
        self.seen_before = []  # class ids known BEFORE the current experience

    def make_aux_loss_fn(self, device):
        """Return an aux_loss_fn distilling old-class logits, or None on exp 0."""
        if self.teacher is None or not self.seen_before:
            return None
        teacher = self.teacher
        old_t = torch.tensor(sorted(self.seen_before), device=device, dtype=torch.long)
        a, T = self.alpha, self.T
        # cache for occasional reporting
        self.last_aux = None

        def aux_loss_fn(model, logits, x_categ, x_cont, y):
            with torch.no_grad():
                t_logits = teacher(x_categ, x_cont)
            # only distill columns that the teacher actually had
            n_old = t_logits.shape[1]
            cols = old_t[old_t < n_old]
            t_old = t_logits.index_select(1, cols)
            s_old = logits.index_select(1, cols)
            kd = F.kl_div(
                F.log_softmax(s_old / T, dim=1),
                F.softmax(t_old / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            self.last_aux = float(a * kd.detach().item())
            # main.py already computes plain CE; scale it there is not possible,
            # so we only ADD the distillation term (weighted by alpha).
            return a * kd

        return aux_loss_fn

    def update(self, model, seen_classes_set):
        """Freeze a teacher AFTER the experience and record seen classes."""
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.seen_before = sorted(set(seen_classes_set))


# --------------------------------------------------------------------------- #
# EWC — diagonal Fisher penalty (separate-EWC accumulation)
# --------------------------------------------------------------------------- #
class EWCState:
    def __init__(self, ewc_lambda=1000.0, fisher_samples=4096):
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.tasks = []  # list of (fisher_dict, star_dict)

    def make_aux_loss_fn(self, device):
        """Return an aux_loss_fn adding the EWC penalty, or None if no task stored."""
        if not self.tasks:
            return None
        lam = self.ewc_lambda
        tasks = self.tasks
        self.last_aux = None

        def aux_loss_fn(model, logits, x_categ, x_cont, y):
            pen = 0.0
            params = dict(model.named_parameters())
            for fisher, star in tasks:
                for n_, f in fisher.items():
                    p = params.get(n_)
                    if p is None or p.shape != f.shape:
                        continue  # head grew: skip mismatched (new) params
                    pen = pen + (f * (p - star[n_]) ** 2).sum()
            aux = 0.5 * lam * pen
            self.last_aux = float(aux.detach().item()) if torch.is_tensor(aux) else 0.0
            return aux

        return aux_loss_fn

    def consolidate(self, model, train_ds, device, criterion, seed=42):
        """Compute diagonal Fisher on a subset of this experience's train data."""
        rng = np.random.default_rng(seed)
        n = min(self.fisher_samples, len(train_ds))
        idx = rng.choice(len(train_ds), size=n, replace=False)

        fisher = {
            n_: torch.zeros_like(p)
            for n_, p in model.named_parameters()
            if p.requires_grad
        }
        was_training = model.training
        model.eval()
        for i in idx:
            (x_categ, x_cont), y = train_ds[int(i)]
            x_categ = x_categ.unsqueeze(0).to(device)
            x_cont = x_cont.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x_categ, x_cont), y)
            loss.backward()
            for n_, p in model.named_parameters():
                if p.grad is not None and n_ in fisher:
                    fisher[n_] += p.grad.detach() ** 2
        for n_ in fisher:
            fisher[n_] /= max(1, n)
        star = {
            n_: p.detach().clone()
            for n_, p in model.named_parameters()
            if p.requires_grad
        }
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        self.tasks.append((fisher, star))


# --------------------------------------------------------------------------- #
# iCaRL — herding exemplars + BCE distillation + nearest-mean classification
# --------------------------------------------------------------------------- #
class _ExemplarDataset(Dataset):
    """Wraps a flat list of ((x_categ, x_cont), y) samples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ICaRLState:
    def __init__(self, memory=2000):
        self.memory = memory
        self.exemplars = {}  # class_id -> list of ((x_categ, x_cont), y) samples
        self.class_means = {}  # class_id -> L2-normalised feature mean tensor (D,)
        self.teacher = None
        self.old_classes = []

    # ----- exemplar access for the dataloader -----
    def exemplar_samples(self):
        out = []
        for c in self.exemplars:
            out.extend(self.exemplars[c])
        return out

    def exemplar_dataset(self):
        return _ExemplarDataset(self.exemplar_samples())

    # ----- training loss: BCE with distillation on old classes -----
    def make_aux_loss_fn(self, num_out, device):
        """
        Returns an aux_loss_fn that REPLACES the CE term with BCE-with-distillation.
        Because train_model adds aux to the CE, we return (bce - ce_placeholder)?  No.
        Instead main.py passes a criterion that returns 0 for iCaRL, and the whole
        loss is produced here. See main.py wiring.
        """
        teacher = self.teacher
        old = list(self.old_classes)

        def aux_loss_fn(model, logits, x_categ, x_cont, y):
            n_out = logits.shape[1]
            tgt = F.one_hot(y.long(), num_classes=n_out).float()
            if teacher is not None and old:
                with torch.no_grad():
                    t_logits = teacher(x_categ, x_cont)
                n_old_out = t_logits.shape[1]
                old_sig = torch.sigmoid(t_logits)
                cols = [c for c in old if c < n_old_out and c < n_out]
                if cols:
                    ci = torch.tensor(cols, device=logits.device)
                    tgt[:, ci] = old_sig[:, ci]
            return F.binary_cross_entropy_with_logits(logits, tgt)

        return aux_loss_fn

    # ----- herding + means, after an experience -----
    @staticmethod
    def _herding(feats_norm, m):
        mu = feats_norm.mean(0)
        w = mu.copy()
        chosen = []
        for _ in range(min(m, len(feats_norm))):
            i = int(np.argmax(feats_norm @ w))
            chosen.append(i)
            w = w + mu - feats_norm[i]
        return np.array(chosen, dtype=int)

    def _feats_for_samples(self, model, samples, device, batch=1024):
        """Extract penultimate features for a list of dataset samples."""
        feats = []
        for i in range(0, len(samples), batch):
            chunk = samples[i : i + batch]
            xc = torch.stack([s[0][0] for s in chunk])  # x_categ
            xn = torch.stack([s[0][1] for s in chunk])  # x_cont
            f = extract_features(model, xc, xn, device)
            feats.append(f.cpu().numpy())
        return np.vstack(feats)

    def update(self, model, exp, device):
        """Herd exemplars for new classes, prune, and recompute class means."""
        train_ds = exp.train_ds
        labels = train_ds.labels.astype(int)
        new_classes = [c for c in exp.class_ids if c not in self.exemplars]

        m = max(1, self.memory // max(1, len(self.exemplars) + len(new_classes)))

        for c in new_classes:
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            samples = [train_ds[int(i)] for i in idxs]
            f = self._feats_for_samples(model, samples, device)
            fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            pick = self._herding(fn, m)
            self.exemplars[c] = [samples[i] for i in pick]

        # prune every class down to the new per-class budget
        for c in list(self.exemplars):
            self.exemplars[c] = self.exemplars[c][:m]

        # recompute L2-normalised class means with the updated extractor
        self.class_means = {}
        for c, samples in self.exemplars.items():
            f = self._feats_for_samples(model, samples, device)
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            mu = f.mean(0)
            mu = mu / (np.linalg.norm(mu) + 1e-8)
            self.class_means[c] = torch.tensor(mu, dtype=torch.float32)

        # freeze teacher for next experience's distillation
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.old_classes = sorted(self.exemplars.keys())

        total = sum(len(v) for v in self.exemplars.values())
        print(
            f"      iCaRL: {total} exemplars over {len(self.class_means)} classes "
            f"(m={m}/class)"
        )

    # ----- NME prediction for the eval path -----
    @torch.no_grad()
    def predict(self, model, x_categ, x_cont, device):
        """Nearest-mean-of-exemplars classification in feature space."""
        f = extract_features(model, x_categ, x_cont, device)
        f = F.normalize(f, dim=1)
        classes = sorted(self.class_means.keys())
        means = torch.stack([self.class_means[c].to(device) for c in classes])
        loc = (f @ means.T).argmax(1).cpu().tolist()
        return torch.tensor([classes[i] for i in loc], device=device)
