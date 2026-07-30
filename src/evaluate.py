"""
evaluate.py -- CTC decoding/metrics, per-line logged test-set evaluation
(the input every stats.py function needs), and the efficiency/complexity
profile (params, FLOPs, latency, throughput, memory).
"""

import os
import io
import time
import random

import numpy as np
import pandas as pd
import torch
import Levenshtein

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics / decoding
# ---------------------------------------------------------------------------
def calculate_metrics(preds, targets, idx_to_char):
    total_cer, total_wer = 0.0, 0.0
    num_samples = len(preds)
    for p, t in zip(preds, targets):
        target_str = "".join(idx_to_char[idx.item()] for idx in t if idx != 0)
        target_str = target_str.replace("<SPACE>", " ").replace("[BLANK]", "")

        pred_str, prev_char = "", None
        for idx in p:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                pred_str += idx_to_char[idx]
            prev_char = idx
        pred_str = pred_str.replace("<SPACE>", " ").replace("[BLANK]", "")

        cer = Levenshtein.distance(pred_str, target_str) / max(len(target_str), 1)
        pred_words, target_words = pred_str.split(), target_str.split()
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
    gt_words, pred_words = gt_text.split(), pred_text.split()
    unique_words = list(set(gt_words + pred_words))
    word_to_char_map = {w: chr(i) for i, w in enumerate(unique_words)}
    gt_encoded = "".join(word_to_char_map[w] for w in gt_words)
    pred_encoded = "".join(word_to_char_map[w] for w in pred_words)
    return Levenshtein.distance(gt_encoded, pred_encoded), len(gt_words)


def validate_model(model, val_loader, device, criterion, idx_to_char):
    model.eval()
    val_loss, total_cer, total_wer = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, targets, target_lengths, texts, filenames in val_loader:
            images, targets = images.to(device), targets.to(device)
            with torch.autocast(device_type=device.type):
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1).permute(1, 0)
            batch_cer, batch_wer = calculate_metrics(preds, targets, idx_to_char)
            total_cer += batch_cer
            total_wer += batch_wer
    n = len(val_loader)
    return val_loss / n, total_cer / n, total_wer / n


# ---------------------------------------------------------------------------
# Per-line logged test-set evaluation -- the DataFrame every stats.py
# function (bootstrap CI, paired significance, confusion, length robustness)
# consumes.
# ---------------------------------------------------------------------------
def evaluate_on_test_set_logged(model, data_loader, criterion, idx_to_char, device, model_name):
    model.eval()
    rows, total_test_loss = [], 0.0
    with torch.no_grad():
        for images, padded_targets, target_lengths, texts, filenames in data_loader:
            images, padded_targets = images.to(device), padded_targets.to(device)
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1).permute(1, 0, 2)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=device)
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
    print(f"[{model_name}] Test Loss {avg_loss:.4f} | Corpus CER {corpus_cer:.2f}% | Corpus WER {corpus_wer:.2f}%")

    csv_path = os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_test_predictions.csv")
    df.to_csv(csv_path, index=False)
    return df, corpus_cer, corpus_wer


# ---------------------------------------------------------------------------
# Efficiency & complexity profile (params, FLOPs, latency, throughput, memory)
# ---------------------------------------------------------------------------
def get_mean_test_width(dataset, img_height=64, n_samples=500):
    idxs = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    widths = [dataset[i][0].shape[2] for i in idxs]
    return int(sum(widths) / len(widths))


def count_parameters_breakdown(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("cnn", "bridge")))
    vit_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("vit_layers", "vit_ln")))
    head_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier"))
    return total_params, {
        "Total (M)": total_params / 1e6, "Trainable (M)": trainable_params / 1e6,
        "CNN backbone (M)": cnn_params / 1e6, "ViT encoder (M)": vit_params / 1e6,
        "Classifier head (M)": head_params / 1e6,
    }


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def get_flops(model, input_shape=(1, 1, 64, 400)):
    try:
        from ptflops import get_model_complexity_info
        model.eval()
        macs, _ = get_model_complexity_info(model, input_shape[1:], as_strings=False,
                                              print_per_layer_stat=False, verbose=False)
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
        print("Install `ptflops` or `thop` to report FLOPs.")
        return None


def measure_latency_throughput(model, device, img_height=64, img_width=400,
                                batch_sizes=(1, 8), n_warmup=10, n_runs=50):
    model.eval(); model.to(device)
    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, 1, img_height, img_width).to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
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
        "CNN params (M)": round(breakdown["CNN backbone (M)"], 2),
        "ViT params (M)": round(breakdown["ViT encoder (M)"], 2),
        "Checkpoint size (MB)": round(size_mb, 1),
        "GFLOPs (per line)": round(gflops, 2) if gflops is not None else "N/A",
        "Latency bs=1 (ms/line)": round(latency[1]["Latency mean (ms/line)"], 2),
        "Latency bs=8 (ms/line)": round(latency[8]["Latency mean (ms/line)"], 2),
        "Throughput bs=8 (lines/s)": round(latency[8]["Throughput (lines/sec)"], 1),
        "Peak GPU mem bs=8 (MB)": (round(latency[8]["Peak GPU memory (MB)"], 1)
                                    if device.type == "cuda" else "N/A"),
    }


def build_efficiency_table_all_variants(test_dataset, num_classes, device):
    from .model import MODEL_CONFIGS, load_variant_for_eval
    rows = []
    for variant_name in MODEL_CONFIGS:
        model = load_variant_for_eval(variant_name, num_classes, device)
        if model is None:
            print(f"[skip: efficiency] no checkpoint for {variant_name} yet.")
            continue
        rows.append(build_efficiency_row(model, device, test_dataset, variant_name))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "efficiency_all_variants.csv"), index=False)
    return df
