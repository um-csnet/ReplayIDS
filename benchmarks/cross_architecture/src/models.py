"""
models.py — the two architectures from the handover spec.

MLP: input -> two hidden layers (256, 128) ReLU -> 8-way head.
SGM-CNN: reshape the 1-D flow features into a 2-D grid (pad to 9x9), then
2-3 Conv2d + MaxPool + FC. Both use a fixed 8-way head (global label space);
Task-2 classes are simply untrained until Task 2 begins, which is the standard
class-incremental setup and equivalent to a dynamically expanded head.
"""

import math
import torch
import torch.nn as nn

NUM_CLASSES = 8


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x.float())

    def get_features(self, x):  # penultimate (128-d), for iCaRL NME
        return self.net[:-1](x.float())


class SGMCNN(nn.Module):
    """Spatial-reshape CNN: pad features to a square grid and treat as a 1-channel image."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.side = math.ceil(math.sqrt(input_dim))  # 78 -> 9  (9x9 = 81)
        self.pad = self.side * self.side - input_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 9->4
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4->2
        )
        feat = 64 * (self.side // 4) * (self.side // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.float()
        if self.pad:
            x = torch.nn.functional.pad(x, (0, self.pad))
        x = x.view(x.size(0), 1, self.side, self.side)
        return self.classifier(self.features(x))

    def get_features(self, x):  # penultimate (128-d), for iCaRL NME
        x = x.float()
        if self.pad:
            x = torch.nn.functional.pad(x, (0, self.pad))
        x = x.view(x.size(0), 1, self.side, self.side)
        return self.classifier[:-1](self.features(x))


class TabTransformer(nn.Module):
    """FT-Transformer (tabular transformer, all features continuous) — the paper's
    TabTransformer family. Uses FTTransformer because CICIDS features are all
    continuous and we lack the categorical-index file; tokenises every feature and
    applies self-attention. Config mirrors the paper: dim 32, depth 6, heads 10."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        from tab_transformer_pytorch import FTTransformer

        self.net = FTTransformer(
            categories=(),
            num_continuous=input_dim,
            dim=32,
            depth=6,
            heads=10,
            dim_out=num_classes,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
        self._empty = None

    def forward(self, x):
        x = x.float()
        cat = x.new_empty((x.size(0), 0), dtype=torch.long)
        return self.net(cat, x)

    def get_features(self, x):  # CLS token (32-d) via pre-hook on to_logits
        feat = {}
        h = self.net.to_logits.register_forward_pre_hook(
            lambda m, inp: feat.__setitem__("f", inp[0])
        )
        try:
            self.forward(x)
        finally:
            h.remove()
        return feat["f"]


def build(name, input_dim):
    return {"mlp": MLP, "cnn": SGMCNN, "ftt": TabTransformer}[name](input_dim)


if __name__ == "__main__":
    # smoke test: both models accept a 78-feature batch and emit 8 logits
    for n in ("mlp", "cnn"):
        m = build(n, 78)
        out = m(torch.randn(4, 78))
        assert out.shape == (4, 8), (n, out.shape)
        print(n, "ok", out.shape)
