"""
USAGE
-----
Train the proposed model:
    python ahlr_vt_pipeline.py train --variant AHLR-VT --fresh

Train the Hybrid-CNN-ViT depth-ablation variants:
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d8
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d6
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d4
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d2

Train the Pure-ViT (no CNN) stem-ablation variants:
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d12
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d8
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d6
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d4
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d2

Run the full Part-2 suite across every variant trained so far:
    python ahlr_vt_pipeline.py validate
"""

import os
import io
import math
import time
import random
import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import torchvision.models as models

import cv2
import albumentations as A
import Levenshtein
from scipy import stats

RESULTS_DIR = "results"
CKPT_DIR = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Identical to the notebook.
VOCAB_PATH = "vocab.txt"
TRAIN_DIR = "Dataset/train_images/output_lines"
VAL_DIR = "Dataset/val_images/output_linesval"
TEST_DIR = "Dataset/test_images/output_linestest_cleaned"

# ---------------------------------------------------------------------------
# Model variant registry
# "family" selects TrueHybridViT_NoGRU (with CNN) vs. PureViT (no CNN).
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # Hybrid CNN + ViT family. AHLR-VT (vit_depth=12)
    "AHLR-VT":       dict(family="hybrid", vit_depth=12),  # proposed, ~87.04M params
    "Hybrid-ViT-d8": dict(family="hybrid", vit_depth=8),    # ~58.69M
    "Hybrid-ViT-d6": dict(family="hybrid", vit_depth=6),    # ~44.52M
    "Hybrid-ViT-d4": dict(family="hybrid", vit_depth=4),    # ~30.34M
    "Hybrid-ViT-d2": dict(family="hybrid", vit_depth=2),    # ~16.16M

    # Pure ViT family -- no CNN feature extractor. Native patch embedding
    # instead, with patch height = full image height so the token sequence
    # stays 1D/left-to-right (required for CTC). patch_width=4 matches the
    # hybrid CNN's 4x width downsampling, so sequence length T is the same
    # across families at any given depth -- isolating "CNN present or not"
    # as the only varying factor for the stem ablation.
    "Pure-ViT-d12":  dict(family="pure", vit_depth=12, patch_width=4),  # ~85.49M
    "Pure-ViT-d8":   dict(family="pure", vit_depth=8,  patch_width=4),   # ~57.14M
    "Pure-ViT-d6":   dict(family="pure", vit_depth=6,  patch_width=4),   # ~42.97M
    "Pure-ViT-d4":   dict(family="pure", vit_depth=4,  patch_width=4),   # ~28.79M
    "Pure-ViT-d2":   dict(family="pure", vit_depth=2,  patch_width=4),   # ~14.62M
}


class AmharicDataset(Dataset):
    def __init__(self, root_dir, vocab_path, img_height=64, augment=False):
        self.root_dir = root_dir
        self.img_height = img_height
        self.augment = augment

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f.readlines()]

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        assert "[BLANK]" in self.char_to_idx, "Vocab missing [BLANK] at index 0"
        assert "<SPACE>" in self.char_to_idx, "Vocab missing <SPACE> token"
        assert "<UNK>" in self.char_to_idx, "Vocab missing <UNK> token"

        self.samples = []
        self._load_samples()

        if self.augment:
            self.transform = A.Compose([
                A.SafeRotate(limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=(-0.1, 0.0),
                                    rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.ElasticTransform(alpha=1, sigma=20, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ])

    def _load_samples(self):
        for writer_dir in os.listdir(self.root_dir):
            writer_path = os.path.join(self.root_dir, writer_dir)
            if not os.path.isdir(writer_path):
                continue
            for file in os.listdir(writer_path):
                if file.endswith(".png"):
                    img_path = os.path.join(writer_path, file)
                    txt_path = img_path.replace(".png", ".txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        self.samples.append((img_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.augment:
            augmented = self.transform(image=img)
            img = augmented["image"]

        h, w = img.shape
        new_w = int(w * (self.img_height / h))  # no floor guard -- matches notebook exactly
        img = cv2.resize(img, (new_w, self.img_height))

        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        seq = []
        for c in text:
            if c == " ":
                seq.append(self.char_to_idx["<SPACE>"])
            else:
                seq.append(self.char_to_idx.get(c, self.char_to_idx["<UNK>"]))
        target = torch.tensor(seq, dtype=torch.long)

        return img, target, text, filename


def collate_fn(batch):
    images, targets, texts, filenames = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad_width = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, pad_width, 0, 0), value=1.0)
        padded_images.append(padded_img)
    padded_images = torch.stack(padded_images)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_images, padded_targets, target_lengths, texts, filenames


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
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


class TrueHybridViT_NoGRU(nn.Module):
    

    def __init__(self, num_classes, hidden_dim=256, vit_depth=12):
        super(TrueHybridViT_NoGRU, self).__init__()
        assert 1 <= vit_depth <= 12

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


class PureViT(nn.Module):
    """Stem-ablation counterpart to TrueHybridViT_NoGRU: NO CNN feature
    extractor. Tokenization is a single non-overlapping Conv2d patch
    embedding with kernel/stride = (img_height, patch_width) -- each patch
    spans the FULL line height and `patch_width` pixels horizontally, so
    the token sequence stays 1D/left-to-right, matching native ViT
    patchification (Dosovitskiy et al., 2021) adapted to line-shaped input.
    """

    def __init__(self, num_classes, img_height=64, patch_width=4, vit_depth=12, embed_dim=768):
        super().__init__()
        assert 1 <= vit_depth <= 12

        self.patch_embed = nn.Conv2d(
            in_channels=1, out_channels=embed_dim,
            kernel_size=(img_height, patch_width), stride=(img_height, patch_width),
        )
        self.pos_encoder = PositionalEncoding1D(embed_dim)

        vit = models.vit_b_16(weights=None)
        self.vit_layers = vit.encoder.layers[:vit_depth]
        self.vit_ln = vit.encoder.ln

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        patches = self.patch_embed(x)                 # [B, embed_dim, 1, W//patch_width]
        b, c, h, w = patches.size()
        assert h == 1
        features = patches.squeeze(2).permute(0, 2, 1)  # [B, T, embed_dim]
        features = self.pos_encoder(features)
        trans_out = self.vit_layers(features)
        trans_out = self.vit_ln(trans_out)
        return self.classifier(trans_out)


def build_model(variant_name, num_classes):
    cfg = MODEL_CONFIGS[variant_name]
    if cfg["family"] == "hybrid":
        return TrueHybridViT_NoGRU(num_classes=num_classes, vit_depth=cfg["vit_depth"])
    elif cfg["family"] == "pure":
        return PureViT(num_classes=num_classes, patch_width=cfg["patch_width"], vit_depth=cfg["vit_depth"])
    else:
        raise ValueError(f"Unknown family in MODEL_CONFIGS['{variant_name}']: {cfg['family']}")


def ckpt_paths(variant_name):
    tag = variant_name.replace(" ", "_")
    return (
        os.path.join(CKPT_DIR, f"checkpoint_{tag}.pth"),
        os.path.join(CKPT_DIR, f"best_{tag}.pth"),
    )


def calculate_metrics(preds, targets, idx_to_char):
    total_cer, total_wer = 0.0, 0.0
    num_samples = len(preds)

    for p, t in zip(preds, targets):
        target_str = "".join([idx_to_char[idx.item()] for idx in t if idx != 0])
        target_str = target_str.replace("<SPACE>", " ").replace("[BLANK]", "")

        pred_str = ""
        prev_char = None
        for idx in p:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                pred_str += idx_to_char[idx]
            prev_char = idx
        pred_str = pred_str.replace("<SPACE>", " ").replace("[BLANK]", "")

        cer = Levenshtein.distance(pred_str, target_str) / max(len(target_str), 1)
        pred_words = pred_str.split()
        target_words = target_str.split()
        wer = Levenshtein.distance(pred_words, target_words) / max(len(target_words), 1)

        total_cer += cer
        total_wer += wer

    return total_cer / num_samples, total_wer / num_samples


def decode_batch_predictions(outputs, idx_to_char):
    preds = torch.argmax(outputs, dim=-1)
    decoded_batch = []
    for b in range(preds.size(0)):
        pred_idx = preds[b]
        decoded_text = []
        for i in range(len(pred_idx)):
            if pred_idx[i] != 0 and (i == 0 or pred_idx[i] != pred_idx[i - 1]):
                char = idx_to_char[pred_idx[i].item()]
                if char == "<SPACE>":
                    decoded_text.append(" ")
                elif char != "<UNK>":
                    decoded_text.append(char)
        decoded_batch.append("".join(decoded_text))
    return decoded_batch


def compute_batch_word_distance(gt_text, pred_text):
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    unique_words = list(set(gt_words + pred_words))
    word_to_char_map = {word: chr(idx) for idx, word in enumerate(unique_words)}
    gt_encoded = "".join([word_to_char_map[w] for w in gt_words])
    pred_encoded = "".join([word_to_char_map[w] for w in pred_words])
    return Levenshtein.distance(gt_encoded, pred_encoded), len(gt_words)


def validate_model(model, val_loader, device, criterion, idx_to_char):
    model.eval()
    val_loss = 0.0
    total_cer, total_wer = 0.0, 0.0

    with torch.no_grad():
        for images, targets, target_lengths, texts, filenames in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            with autocast(device_type="cuda"):
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            preds = preds.permute(1, 0)
            batch_cer, batch_wer = calculate_metrics(preds, targets, idx_to_char)
            total_cer += batch_cer
            total_wer += batch_wer

    avg_loss = val_loss / len(val_loader)
    avg_cer = total_cer / len(val_loader)
    avg_wer = total_wer / len(val_loader)
    return avg_loss, avg_cer, avg_wer


def train_model(variant_name, train_dataset, val_dataset, max_epochs=100,
                 patience=None, fresh=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{variant_name}] Using device: {device}")
    if device.type != "cuda":
        print(f"[{variant_name}] WARNING: this training loop uses "
              f"autocast(device_type='cuda') exactly as the notebook did, "
              f"which assumes a CUDA GPU is present.")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    num_classes = len(train_dataset.vocab)
    model = build_model(variant_name, num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = GradScaler()

    writer = SummaryWriter(f"runs/{variant_name.replace(' ', '_')}")

    checkpoint_path, best_model_path = ckpt_paths(variant_name)

    if fresh:
        for p in (checkpoint_path, best_model_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"[{variant_name}] --fresh: removed existing {p}")

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if os.path.exists(checkpoint_path):
        print(f"[{variant_name}] *** RESUMING from {checkpoint_path} *** "
              f"(pass --fresh from scratch -- e.g. if this "
              f"checkpoint was written by an earlier, different version of this script)")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        scaler.load_state_dict(checkpoint["scaler"])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        print(f"[{variant_name}] Resuming from epoch {start_epoch}")

    accumulation_steps = 4

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (images, targets, target_lengths, texts, filenames) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            with autocast(device_type="cuda"):
                outputs = model(images).permute(1, 0, 2)
                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if batch_idx % 50 == 0:
                preds = torch.argmax(outputs.detach(), dim=-1)
                preds = preds.permute(1, 0)
                cer, wer = calculate_metrics(preds, targets, train_dataset.idx_to_char)
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Training/Loss", loss.item() * accumulation_steps, global_step)
                writer.add_scalar("Training/CER", cer, global_step)
                writer.add_scalar("Training/WER", wer, global_step)
                print(f"[{variant_name}] Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Train Loss: {loss.item() * accumulation_steps:.4f} "
                      f"| Train CER: {cer:.4f} | Train WER: {wer:.4f}")

        print(f"[{variant_name}] Running validation...")
        val_loss, val_cer, val_wer = validate_model(model, val_loader, device, criterion, train_dataset.idx_to_char)

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/CER", val_cer, epoch)
        writer.add_scalar("Validation/WER", val_wer, epoch)
        print(f"[{variant_name}] --> Epoch {epoch} Summary | Val Loss: {val_loss:.4f} "
              f"| Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}")

        checkpoint_state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss, "epochs_no_improve": epochs_no_improve,
            "variant_name": variant_name,
        }
        torch.save(checkpoint_state, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_state["best_val_loss"] = best_val_loss
            torch.save(checkpoint_state, best_model_path)
            print(f"[{variant_name}] *** New best model saved with Val Loss: {val_loss:.4f} ***")
        else:
            epochs_no_improve += 1
            # Only stops early if you explicitly opted in via --patience;
            # the notebook itself never stops early.
            if patience is not None and epochs_no_improve >= patience:
                print(f"[{variant_name}] Early stopping at epoch {epoch} "
                      f"(no val-loss improvement for {patience} epochs).")
                break

    return best_model_path


def evaluate_on_test_set_logged(model, data_loader, criterion, idx_to_char, device, model_name="AHLR-VT"):
    model.eval()
    rows = []
    total_test_loss = 0.0

    with torch.no_grad():
        for images, padded_targets, target_lengths, texts, filenames in data_loader:
            images = images.to(device)
            padded_targets = padded_targets.to(device)

            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1).permute(1, 0, 2)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0),
                                        dtype=torch.long, device=device)
            loss = criterion(log_probs, padded_targets, input_lengths, target_lengths)
            total_test_loss += loss.item() * batch_size

            predicted_texts = decode_batch_predictions(outputs, idx_to_char)

            for i in range(batch_size):
                gt, pred, fn = texts[i], predicted_texts[i], filenames[i]
                char_dist = Levenshtein.distance(gt, pred)
                word_dist, word_len = compute_batch_word_distance(gt, pred)

                rows.append({
                    "filename": fn, "gt": gt, "pred": pred,
                    "char_distance": char_dist, "char_length": max(len(gt), 1),
                    "word_distance": word_dist, "word_length": max(word_len, 1),
                    "line_cer": char_dist / max(len(gt), 1),
                    "line_wer": word_dist / max(word_len, 1),
                })

    df = pd.DataFrame(rows)
    corpus_cer = df["char_distance"].sum() / df["char_length"].sum() * 100
    corpus_wer = df["word_distance"].sum() / df["word_length"].sum() * 100
    avg_loss = total_test_loss / len(data_loader.dataset)

    print(f"[{model_name}] Test Loss: {avg_loss:.4f} | Corpus CER: {corpus_cer:.2f}% | Corpus WER: {corpus_wer:.2f}%")

    csv_path = os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_test_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{model_name}] Per-line predictions saved to {csv_path}")

    return df, corpus_cer, corpus_wer


def load_variant_for_eval(variant_name, num_classes, device):
    _, best_path = ckpt_paths(variant_name)
    if not os.path.exists(best_path):
        return None
    model = build_model(variant_name, num_classes).to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Part 2.1 -- Efficiency & complexity profile. covers both the
# hybrid's CNN+bridge and the pure-ViT's patch_embed (whichever the model
# actually has), so one function serves both families.
# ---------------------------------------------------------------------------
def get_mean_test_width(dataset, img_height=64, n_samples=500):
    idxs = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    widths = []
    for i in idxs:
        img, _, _, _ = dataset[i]
        widths.append(img.shape[2])
    return int(sum(widths) / len(widths))


def count_parameters_breakdown(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stem_params = sum(p.numel() for n, p in model.named_parameters()
                       if n.startswith(("cnn", "bridge", "patch_embed")))
    vit_params = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith(("vit_layers", "vit_ln")))
    head_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier"))

    breakdown = {
        "Total (M)": total_params / 1e6,
        "Trainable (M)": trainable_params / 1e6,
        "Stem params (M)": stem_params / 1e6,
        "ViT encoder (M)": vit_params / 1e6,
        "Classifier head (M)": head_params / 1e6,
    }
    return total_params, breakdown


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def get_flops(model, input_shape=(1, 1, 64, 400)):
    try:
        from ptflops import get_model_complexity_info
        model.eval()
        macs, params = get_model_complexity_info(
            model, input_shape[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        return macs * 2 / 1e9
    except ImportError:
        pass
    try:
        from thop import profile
        model.eval()
        dummy = torch.randn(*input_shape)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9
    except ImportError:
        print("Neither `ptflops` nor `thop` is installed. Install one to report FLOPs:")
        print("    pip install ptflops")
        return None


def measure_latency_throughput(model, device, img_height=64, img_width=400,
                                batch_sizes=(1, 8), n_warmup=10, n_runs=50):
    model.eval()
    model.to(device)
    results = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, 1, img_height, img_width).to(device)

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            timings = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

            peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                           if device.type == "cuda" else float("nan"))

        timings = torch.tensor(timings)
        per_line_ms = timings / bs
        results[bs] = {
            "Latency mean (ms/line)": per_line_ms.mean().item(),
            "Latency std (ms/line)": per_line_ms.std().item(),
            "Throughput (lines/sec)": 1000.0 / per_line_ms.mean().item(),
            "Peak GPU memory (MB)": peak_mem_mb,
        }
    return results


def build_efficiency_row(model, device, dataset, model_name):
    mean_width = get_mean_test_width(dataset)
    total_params, breakdown = count_parameters_breakdown(model)
    size_mb = get_model_size_mb(model)
    gflops = get_flops(model, input_shape=(1, 1, 64, mean_width))
    latency = measure_latency_throughput(model, device, img_width=mean_width)
    return {
        "Model": model_name,
        "Params (M)": round(breakdown["Total (M)"], 2),
        "Stem params (M)": round(breakdown["Stem params (M)"], 2),
        "ViT params (M)": round(breakdown["ViT encoder (M)"], 2),
        "Checkpoint size (MB)": round(size_mb, 1),
        "GFLOPs (per line)": round(gflops, 2) if gflops is not None else "N/A",
        "Latency bs=1 (ms/line)": round(latency[1]["Latency mean (ms/line)"], 2),
        "Latency bs=8 (ms/line)": round(latency[8]["Latency mean (ms/line)"], 2),
        "Throughput bs=8 (lines/s)": round(latency[8]["Throughput (lines/sec)"], 1),
        "Peak GPU mem bs=8 (MB)": (round(latency[8]["Peak GPU memory (MB)"], 1)
                                    if device.type == "cuda" else "N/A"),
    }


def build_efficiency_table_all_variants(test_dataset, num_classes, device, trained_variants):
    rows = []
    for variant_name in trained_variants:
        model = load_variant_for_eval(variant_name, num_classes, device)
        if model is None:
            continue
        rows.append(build_efficiency_row(model, device, test_dataset, variant_name))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "efficiency_all_variants.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Part 2.3 -- Bootstrap CI. Identical math to the notebook.
# ---------------------------------------------------------------------------
def bootstrap_corpus_ci(df, distance_col, length_col, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    distances = df[distance_col].to_numpy()
    lengths = df[length_col].to_numpy()

    point_estimate = distances.sum() / lengths.sum() * 100
    boot_estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = distances[idx].sum() / lengths[idx].sum() * 100

    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_estimates, [alpha, 100 - alpha])
    return point_estimate, lower, upper, boot_estimates


# ---------------------------------------------------------------------------
# Part 2.4 -- Paired significance testing. run_all_comparisons, EVERY pairwise
# combination of trained variants, so it
# scales automatically as we train more Hybrid-ViT-dN / Pure-ViT-dN checkpoints.
# ---------------------------------------------------------------------------
def paired_significance_test(csv_a, csv_b, name_a="AHLR-VT", name_b="Baseline", metric="line_cer"):
    df_a = pd.read_csv(csv_a)[["filename", metric]].rename(columns={metric: f"{metric}_a"})
    df_b = pd.read_csv(csv_b)[["filename", metric]].rename(columns={metric: f"{metric}_b"})

    merged = df_a.merge(df_b, on="filename", how="inner")
    n_matched = len(merged)
    if n_matched == 0:
        raise ValueError("No overlapping filenames between the two result sets -- check that both "
                          "models were evaluated on the identical test_loader / dataset ordering.")

    a = merged[f"{metric}_a"].to_numpy()
    b = merged[f"{metric}_b"].to_numpy()
    diff = a - b

    t_stat, t_pval = stats.ttest_rel(a, b)
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_matched, size=n_matched)
        boot_diffs[i] = diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")

    return {
        "Model A": name_a, "Model B": name_b, "Metric": metric, "N matched lines": n_matched,
        "Mean A": a.mean(), "Mean B": b.mean(), "Mean diff (A-B)": diff.mean(),
        "95% CI diff": f"[{ci_lo:.5f}, {ci_hi:.5f}]",
        "Paired t-stat": t_stat, "Paired t p-value": t_pval,
        "Wilcoxon stat": w_stat, "Wilcoxon p-value": w_pval,
        "Cohen's d (paired)": cohens_d,
        "Significant (p<0.05, t-test)": t_pval < 0.05,
        "Significant (p<0.05, Wilcoxon)": (w_pval < 0.05) if not np.isnan(w_pval) else "N/A",
    }


def run_all_comparisons(trained_variants):
    """Runs the paired comparison for EVERY pair of trained variants (both
    Hybrid-ViT-dN and Pure-ViT-dN together), on both line_cer and line_wer.
    This is what makes Part 2.4 generic across whatever we've trained.
    """
    all_results = []
    for name_a, name_b in itertools.combinations(trained_variants, 2):
        csv_a = os.path.join(RESULTS_DIR, f"{name_a.replace(' ', '_')}_test_predictions.csv")
        csv_b = os.path.join(RESULTS_DIR, f"{name_b.replace(' ', '_')}_test_predictions.csv")
        if not (os.path.exists(csv_a) and os.path.exists(csv_b)):
            continue
        for metric in ("line_cer", "line_wer"):
            all_results.append(paired_significance_test(csv_a, csv_b, name_a, name_b, metric))

    results_df = pd.DataFrame(all_results)
    if len(results_df):
        results_df.to_csv(os.path.join(RESULTS_DIR, "significance_tests_summary.csv"), index=False)
        print(results_df.to_string(index=False))
    else:
        print("Fewer than 2 trained variants have predictions logged -- need at least 2 to run "
              "pairwise significance tests.")
    return results_df


def build_substitution_confusion(df, top_n=25):
    confusion = Counter()
    insertion_count = 0
    deletion_count = 0

    for _, row in df.iterrows():
        gt, pred = row["gt"], row["pred"]
        ops = Levenshtein.editops(gt, pred)
        for tag, src_pos, dst_pos in ops:
            if tag == "replace":
                confusion[(gt[src_pos], pred[dst_pos])] += 1
            elif tag == "insert":
                insertion_count += 1
            elif tag == "delete":
                deletion_count += 1

    total_subs = sum(confusion.values())
    top_confusions = confusion.most_common(top_n)
    conf_df = pd.DataFrame(top_confusions, columns=["(gt, pred)", "count"])
    conf_df["ground_truth"] = conf_df["(gt, pred)"].apply(lambda x: x[0])
    conf_df["predicted"] = conf_df["(gt, pred)"].apply(lambda x: x[1])
    conf_df = conf_df[["ground_truth", "predicted", "count"]]

    composition = {"substitution": total_subs, "insertion": insertion_count, "deletion": deletion_count}
    return conf_df, confusion, composition


def cer_vs_length_analysis(df, n_bins=8):
    df = df.copy()
    df["length_bin"] = pd.qcut(df["char_length"], q=n_bins, duplicates="drop")
    grouped = df.groupby("length_bin", observed=True).apply(
        lambda g: pd.Series({
            "n_lines": len(g),
            "mean_gt_length": g["char_length"].mean(),
            "corpus_cer_pct": g["char_distance"].sum() / g["char_length"].sum() * 100,
        })
    ).reset_index()
    return grouped


def ctc_prefix_beam_search(log_probs, idx_to_char, beam_width=10, blank_idx=0):
    T, V = log_probs.shape
    log_probs = log_probs.cpu().numpy()
    NEG_INF = -1e10
    beams = {(): (0.0, NEG_INF)}

    def log_sum_exp(a, b):
        if a == NEG_INF: return b
        if b == NEG_INF: return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    for t in range(T):
        next_beams = {}
        top_k = min(beam_width * 2, V)
        top_indices = np.argpartition(log_probs[t], -top_k)[-top_k:]

        for prefix, (p_b, p_nb) in beams.items():
            p_total_prev = log_sum_exp(p_b, p_nb)
            for c in top_indices:
                p_c = log_probs[t, c]
                if c == blank_idx:
                    entry = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (log_sum_exp(entry[0], p_total_prev + p_c), entry[1])
                    continue

                end_char = prefix[-1] if len(prefix) > 0 else None
                if c == end_char:
                    new_prefix = prefix + (c,)
                    entry = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (entry[0], log_sum_exp(entry[1], p_b + p_c))
                    entry_same = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (entry_same[0], log_sum_exp(entry_same[1], p_nb + p_c))
                else:
                    new_prefix = prefix + (c,)
                    entry = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (entry[0], log_sum_exp(entry[1], p_total_prev + p_c))

        scored = [(pfx, log_sum_exp(pb, pnb)) for pfx, (pb, pnb) in next_beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {pfx: next_beams[pfx] for pfx, _ in scored[:beam_width]}

    best_prefix = max(beams.items(), key=lambda kv: log_sum_exp(*kv[1]))[0]
    decoded_text = []
    for c in best_prefix:
        char = idx_to_char[c]
        if char == "<SPACE>":
            decoded_text.append(" ")
        elif char != "<UNK>":
            decoded_text.append(char)
    return "".join(decoded_text)


def compare_greedy_vs_beam(model, data_loader, idx_to_char, device, n_lines=300, beam_width=10, seed=42):
    model.eval()
    rng = random.Random(seed)

    all_samples = []
    with torch.no_grad():
        for images, padded_targets, target_lengths, texts, filenames in data_loader:
            images = images.to(device)
            outputs = model(images)
            log_probs_batch = torch.nn.functional.log_softmax(outputs, dim=-1)
            for i in range(images.size(0)):
                all_samples.append((log_probs_batch[i].cpu(), texts[i]))

    subset = rng.sample(all_samples, min(n_lines, len(all_samples)))

    greedy_char_dist, greedy_char_len = 0, 0
    beam_char_dist, beam_char_len = 0, 0
    greedy_time, beam_time = 0.0, 0.0

    for log_probs, gt in subset:
        t0 = time.perf_counter()
        pred_idx = torch.argmax(log_probs, dim=-1)
        greedy_text = []
        for i in range(len(pred_idx)):
            if pred_idx[i] != 0 and (i == 0 or pred_idx[i] != pred_idx[i - 1]):
                ch = idx_to_char[pred_idx[i].item()]
                if ch == "<SPACE>": greedy_text.append(" ")
                elif ch != "<UNK>": greedy_text.append(ch)
        greedy_pred = "".join(greedy_text)
        greedy_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        beam_pred = ctc_prefix_beam_search(log_probs, idx_to_char, beam_width=beam_width)
        beam_time += time.perf_counter() - t0

        greedy_char_dist += Levenshtein.distance(greedy_pred, gt)
        greedy_char_len += max(len(gt), 1)
        beam_char_dist += Levenshtein.distance(beam_pred, gt)
        beam_char_len += max(len(gt), 1)

    greedy_cer = greedy_char_dist / greedy_char_len * 100
    beam_cer = beam_char_dist / beam_char_len * 100

    summary = pd.DataFrame([
        {"Decoding": "Greedy", "CER (%)": round(greedy_cer, 3),
         "Avg. decode time (ms/line)": round(greedy_time / len(subset) * 1000, 3)},
        {"Decoding": f"Beam search (width={beam_width})", "CER (%)": round(beam_cer, 3),
         "Avg. decode time (ms/line)": round(beam_time / len(subset) * 1000, 3)},
    ])
    return summary


def main():
    parser = argparse.ArgumentParser(description="AHLR-VT / Pure-ViT training and Part-2 validation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train one architecture variant")
    p_train.add_argument("--variant", required=True, choices=list(MODEL_CONFIGS.keys()))
    p_train.add_argument("--max-epochs", type=int, default=100)
    p_train.add_argument("--patience", type=int, default=None,
                          help="Disabled by default (matches the notebook: always runs to "
                               "--max-epochs). Pass a value to opt into early stopping.")
    p_train.add_argument("--fresh", action="store_true",
                          help="Delete any existing checkpoint for this variant before starting, "
                               "instead of resuming from it.")

    p_val = sub.add_parser("validate", help="Run the full Part-2 suite over every trained variant")
    p_val.add_argument("--skip-beam", action="store_true",
                        help="Skip the greedy-vs-beam comparison (slowest step, ~300 lines x beam "
                             "search per variant).")
    p_val.add_argument("--beam-n-lines", type=int, default=300)

    args = parser.parse_args()

    train_dataset = AmharicDataset(TRAIN_DIR, VOCAB_PATH, augment=True)
    val_dataset = AmharicDataset(VAL_DIR, VOCAB_PATH, augment=False)
    test_dataset = AmharicDataset(TEST_DIR, VOCAB_PATH, augment=False)
    num_classes = len(train_dataset.vocab)
    idx_to_char = train_dataset.idx_to_char
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        train_model(args.variant, train_dataset, val_dataset,
                    max_epochs=args.max_epochs, patience=args.patience, fresh=args.fresh)

    elif args.command == "validate":
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        trained_variants = [v for v in MODEL_CONFIGS if load_variant_for_eval(v, num_classes, device) is not None]
        if not trained_variants:
            print("No trained checkpoints found in checkpoints/ yet -- run `train --variant ...` first.")
            return
        print(f"Found trained checkpoints for: {trained_variants}")

        # 2.2: per-line logged evaluation for every trained variant
        corpus_rows = {}
        for variant_name in trained_variants:
            model = load_variant_for_eval(variant_name, num_classes, device)
            df, corpus_cer, corpus_wer = evaluate_on_test_set_logged(
                model, test_loader, criterion, idx_to_char, device, model_name=variant_name)
            corpus_rows[variant_name] = df

        # 2.1: efficiency table across every trained variant
        eff_df = build_efficiency_table_all_variants(test_dataset, num_classes, device, trained_variants)
        print(eff_df.to_string(index=False))

        # 2.3: bootstrap CI, per variant
        bootstrap_results = {}
        for variant_name, df in corpus_rows.items():
            cer_point, cer_lo, cer_hi, _ = bootstrap_corpus_ci(df, "char_distance", "char_length")
            wer_point, wer_lo, wer_hi, _ = bootstrap_corpus_ci(df, "word_distance", "word_length")
            bootstrap_results[variant_name] = dict(
                cer_point=cer_point, cer_lo=cer_lo, cer_hi=cer_hi,
                wer_point=wer_point, wer_lo=wer_lo, wer_hi=wer_hi)
            pd.DataFrame([bootstrap_results[variant_name]]).to_csv(
                os.path.join(RESULTS_DIR, f"bootstrap_ci_{variant_name.replace(' ', '_')}.csv"), index=False)
            print(f"[{variant_name}] CER {cer_point:.2f}% [{cer_lo:.2f}, {cer_hi:.2f}] | "
                  f"WER {wer_point:.2f}% [{wer_lo:.2f}, {wer_hi:.2f}]")

        # 2.4: paired significance -- every pair of trained variants
        run_all_comparisons(trained_variants)

        # confusion analysis, length robustness, greedy-vs-beam,
        # run for EVERY trained variant (not just one hardcoded reference model)
        for variant_name in trained_variants:
            df = corpus_rows[variant_name]
            tag = variant_name.replace(" ", "_")

            conf_df, _, composition = build_substitution_confusion(df, top_n=25)
            conf_df.to_csv(os.path.join(RESULTS_DIR, f"confusion_top25_{tag}.csv"), index=False)
            print(f"[{variant_name}] error composition: {composition}")

            length_df = cer_vs_length_analysis(df)
            length_df.to_csv(os.path.join(RESULTS_DIR, f"cer_vs_length_{tag}.csv"), index=False)

            if not args.skip_beam:
                model = load_variant_for_eval(variant_name, num_classes, device)
                beam_df = compare_greedy_vs_beam(model, test_loader, idx_to_char, device,
                                                  n_lines=args.beam_n_lines)
                beam_df.to_csv(os.path.join(RESULTS_DIR, f"greedy_vs_beam_{tag}.csv"), index=False)
                print(f"[{variant_name}] greedy vs. beam:\n{beam_df.to_string(index=False)}")

        # 2.9: consolidated manuscript summary table -- one row per variant
        summary_rows = []
        for variant_name in trained_variants:
            eff_row = eff_df[eff_df["Model"] == variant_name]
            if eff_row.empty or variant_name not in bootstrap_results:
                continue
            bs = bootstrap_results[variant_name]
            row = eff_row.iloc[0].to_dict()
            row["CER (%)"] = round(bs["cer_point"], 2)
            row["CER 95% CI"] = f"[{bs['cer_lo']:.2f}, {bs['cer_hi']:.2f}]"
            row["WER (%)"] = round(bs["wer_point"], 2)
            row["WER 95% CI"] = f"[{bs['wer_lo']:.2f}, {bs['wer_hi']:.2f}]"
            summary_rows.append(row)
        if summary_rows:
            manuscript_summary_df = pd.DataFrame(summary_rows)
            manuscript_summary_df.to_csv(os.path.join(RESULTS_DIR, "manuscript_summary_table.csv"), index=False)
            print("\n--- Manuscript summary table (one row per variant) ---")
            print(manuscript_summary_df.to_string(index=False))
            print("\n--- Markdown table for the paper ---\n")
            print(manuscript_summary_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
