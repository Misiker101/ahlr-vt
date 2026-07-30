"""
the variant registry used to generate the parameter-scaled 
ablation family (depth 2/4/6/8/12).
"""

import os
import math

import torch
import torch.nn as nn
import torchvision.models as models

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Variant registry -- single source of truth used by train.py, evaluate.py,
# and stats.py so every downstream table/test picks up new variants
# automatically once they're added here.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "AHLR-VT": dict(vit_depth=12),        # proposed, full ViT-Base/16 encoder (~87.04M params)
    "Hybrid-ViT-d8": dict(vit_depth=8),    # ~58.69M
    "Hybrid-ViT-d6": dict(vit_depth=6),    # ~44.52M
    "Hybrid-ViT-d4": dict(vit_depth=4),    # ~30.34M
    "Hybrid-ViT-d2": dict(vit_depth=2),    # ~16.16M -- closest param match to ResNet18-Transformer-CTC (17.83M)
}


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridCNNViT(nn.Module):
    """CNN + ViT-Base/16 encoder + CTC head.

    `vit_depth` truncates the 12-block ViT-Base/16 encoder to the first N
    blocks, giving a same-CNN, same-embedding-dim family of models for a
    true single-factor depth ablation / parameter-scaled efficiency
    comparison. vit_depth=12 reproduces the original TrueHybridViT_NoGRU.
    """

    def __init__(self, num_classes, hidden_dim=256, vit_depth=12):
        super().__init__()
        assert 1 <= vit_depth <= 12, "ViT-Base/16 has 12 blocks; choose vit_depth in [1, 12]"
        self.vit_depth = vit_depth

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
        )

        self.bridge = nn.Linear(hidden_dim * 4, 768)
        self.pos_encoder = PositionalEncoding1D(768)

        vit = models.vit_b_16(weights=None)
        self.vit_layers = vit.encoder.layers[:vit_depth]
        self.vit_ln = vit.encoder.ln

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        features = self.bridge(features)
        features = self.pos_encoder(features)
        trans_out = self.vit_layers(features)
        trans_out = self.vit_ln(trans_out)
        return self.classifier(trans_out)


# Backward-compat alias for the original notebook's class name
TrueHybridViT_NoGRU = HybridCNNViT


def build_model(variant_name, num_classes):
    cfg = MODEL_CONFIGS[variant_name]
    return HybridCNNViT(num_classes=num_classes, vit_depth=cfg["vit_depth"])


def ckpt_paths(variant_name):
    tag = variant_name.replace(" ", "_")
    return (
        os.path.join(CKPT_DIR, f"checkpoint_{tag}.pth"),
        os.path.join(CKPT_DIR, f"best_{tag}.pth"),
    )


def load_variant_for_eval(variant_name, num_classes, device):
    _, best_path = ckpt_paths(variant_name)
    if not os.path.exists(best_path):
        return None
    model = build_model(variant_name, num_classes).to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
