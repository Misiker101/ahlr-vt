"""
train.py -- trains one architecture variant at a time.

Usage:
    python -m src.train --variant AHLR-VT
    python -m src.train --variant Hybrid-ViT-d4 --max-epochs 100 --patience 10
"""

from .dependency_check import check_dependencies
check_dependencies()  # exits with a clear message if `pip install -r requirements.txt` wasn't run

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from .model import MODEL_CONFIGS, build_model, ckpt_paths
from .dataset import get_datasets, collate_fn
from .evaluate import calculate_metrics, validate_model


def train_variant(variant_name, train_dataset, val_dataset, max_epochs=100,
                   patience=10, batch_size=8, accumulation_steps=4, lr=1e-4,
                   weight_decay=1e-5, seed=None, num_workers=0):
    """Real patience-based early stopping on validation-loss plateau (the
    original notebook's train_model() ran a fixed 100 epochs without an
    actual stopping rule, despite the manuscript describing early stopping)."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{variant_name}] Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)

    num_classes = len(train_dataset.vocab)
    model = build_model(variant_name, num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = GradScaler()
    writer = SummaryWriter(f"runs/{variant_name.replace(' ', '_')}")

    checkpoint_path, best_model_path = ckpt_paths(variant_name)
    start_epoch, best_val_loss, epochs_no_improve = 0, float("inf"), 0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        scaler.load_state_dict(ckpt["scaler"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"[{variant_name}] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad()
        for batch_idx, (images, targets, target_lengths, texts, filenames) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            with autocast(device_type=device.type):
                outputs = model(images).permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths) / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if batch_idx % 50 == 0:
                preds = torch.argmax(outputs.detach(), dim=-1).permute(1, 0)
                cer, wer = calculate_metrics(preds, targets, train_dataset.idx_to_char)
                gstep = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Training/Loss", loss.item() * accumulation_steps, gstep)
                writer.add_scalar("Training/CER", cer, gstep)
                writer.add_scalar("Training/WER", wer, gstep)
                print(f"[{variant_name}] Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss {loss.item()*accumulation_steps:.4f} | CER {cer:.4f} | WER {wer:.4f}")

        val_loss, val_cer, val_wer = validate_model(model, val_loader, device, criterion, train_dataset.idx_to_char)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/CER", val_cer, epoch)
        writer.add_scalar("Validation/WER", val_wer, epoch)
        print(f"[{variant_name}] --> Epoch {epoch} | Val Loss {val_loss:.4f} | Val CER {val_cer:.4f} | Val WER {val_wer:.4f}")

        state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss, "epochs_no_improve": epochs_no_improve,
            "variant_name": variant_name, "vit_depth": MODEL_CONFIGS[variant_name]["vit_depth"],
        }
        torch.save(state, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            state["best_val_loss"] = best_val_loss
            torch.save(state, best_model_path)
            print(f"[{variant_name}] *** New best model (Val Loss {val_loss:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{variant_name}] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train one AHLR-VT architecture variant")
    parser.add_argument("--variant", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vocab-path", type=str, default="vocab.json")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    args = parser.parse_args()

    train_dataset, val_dataset, _, _ = get_datasets(vocab_path=args.vocab_path, cache_dir=args.hf_cache_dir)

    train_variant(
        args.variant, train_dataset, val_dataset,
        max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
    )


if __name__ == "__main__":
    main()
