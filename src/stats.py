"""
stats.py -- Part 2 statistical validation: bootstrap confidence intervals,
paired significance testing (t-test + Wilcoxon + Cohen's d), character
confusion analysis, line-length robustness, and greedy-vs-beam decoding
comparison. Also the top-level CLI that runs the whole suite across every
trained variant.

Usage (after training at least AHLR-VT, and optionally the Hybrid-ViT-dN
variants):
    python -m src.stats
"""

from .dependency_check import check_dependencies
check_dependencies()  # exits with a clear message if `pip install -r requirements.txt` wasn't run

import os
import math
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats
import Levenshtein

from .model import MODEL_CONFIGS, load_variant_for_eval
from .dataset import get_datasets, collate_fn
from .evaluate import (
    evaluate_on_test_set_logged, build_efficiency_table_all_variants,
    decode_batch_predictions,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2.3 -- Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_corpus_ci(df, distance_col, length_col, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    distances, lengths = df[distance_col].to_numpy(), df[length_col].to_numpy()
    point_estimate = distances.sum() / lengths.sum() * 100
    boot_estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = distances[idx].sum() / lengths[idx].sum() * 100
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_estimates, [alpha, 100 - alpha])
    return point_estimate, lower, upper, boot_estimates


# ---------------------------------------------------------------------------
# 2.4 -- Paired significance testing
# ---------------------------------------------------------------------------
def paired_significance_test(csv_a, csv_b, name_a, name_b, metric="line_cer"):
    df_a = pd.read_csv(csv_a)[["filename", metric]].rename(columns={metric: f"{metric}_a"})
    df_b = pd.read_csv(csv_b)[["filename", metric]].rename(columns={metric: f"{metric}_b"})
    merged = df_a.merge(df_b, on="filename", how="inner")
    n_matched = len(merged)
    if n_matched == 0:
        raise ValueError("No overlapping filenames -- both models must be evaluated on the identical test set.")

    a, b = merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()
    diff = a - b

    t_stat, t_pval = scipy_stats.ttest_rel(a, b)
    try:
        w_stat, w_pval = scipy_stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    rng = np.random.default_rng(42)
    boot_diffs = np.empty(10000)
    for i in range(10000):
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


def run_all_comparisons(proposed_name="AHLR-VT", baseline_names=None):
    proposed_csv = os.path.join(RESULTS_DIR, f"{proposed_name.replace(' ', '_')}_test_predictions.csv")
    if baseline_names is None:
        baseline_names = [n for n in MODEL_CONFIGS if n != proposed_name]

    all_results = []
    for name in baseline_names:
        baseline_csv = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}_test_predictions.csv")
        if not (os.path.exists(proposed_csv) and os.path.exists(baseline_csv)):
            print(f"[skip: significance] missing predictions CSV for {proposed_name} or {name}.")
            continue
        for metric in ("line_cer", "line_wer"):
            all_results.append(paired_significance_test(proposed_csv, baseline_csv,
                                                          proposed_name, name, metric))
    results_df = pd.DataFrame(all_results)
    if len(results_df):
        results_df.to_csv(os.path.join(RESULTS_DIR, "significance_tests_summary.csv"), index=False)
        print(results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# 2.5 -- Character confusion analysis
# ---------------------------------------------------------------------------
def build_substitution_confusion(df, top_n=25):
    confusion = Counter()
    insertion_count = deletion_count = 0
    for _, row in df.iterrows():
        gt, pred = row["gt"], row["pred"]
        for tag, src_pos, dst_pos in Levenshtein.editops(gt, pred):
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


# ---------------------------------------------------------------------------
# 2.6 -- Length robustness
# ---------------------------------------------------------------------------
def cer_vs_length_analysis(df, n_bins=8):
    df = df.copy()
    df["length_bin"] = pd.qcut(df["char_length"], q=n_bins, duplicates="drop")
    grouped = df.groupby("length_bin", observed=True).apply(
        lambda g: pd.Series({
            "n_lines": len(g), "mean_gt_length": g["char_length"].mean(),
            "corpus_cer_pct": g["char_distance"].sum() / g["char_length"].sum() * 100,
        })
    ).reset_index()
    return grouped


# ---------------------------------------------------------------------------
# 2.7 -- Greedy vs. beam search
# ---------------------------------------------------------------------------
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
    greedy_char_dist = greedy_char_len = beam_char_dist = beam_char_len = 0
    greedy_time = beam_time = 0.0

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

        greedy_char_dist += Levenshtein.distance(greedy_pred, gt); greedy_char_len += max(len(gt), 1)
        beam_char_dist += Levenshtein.distance(beam_pred, gt); beam_char_len += max(len(gt), 1)

    summary = pd.DataFrame([
        {"Decoding": "Greedy", "CER (%)": round(greedy_char_dist / greedy_char_len * 100, 3),
         "Avg. decode time (ms/line)": round(greedy_time / len(subset) * 1000, 3)},
        {"Decoding": f"Beam search (width={beam_width})",
         "CER (%)": round(beam_char_dist / beam_char_len * 100, 3),
         "Avg. decode time (ms/line)": round(beam_time / len(subset) * 1000, 3)},
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "greedy_vs_beam_comparison.csv"), index=False)
    return summary


# ---------------------------------------------------------------------------
# Full Part-2 validation CLI -- loops over every trained variant
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset, vocab = get_datasets()
    num_classes = len(vocab)
    idx_to_char = test_dataset.idx_to_char
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    trained_variants = [v for v in MODEL_CONFIGS if load_variant_for_eval(v, num_classes, device) is not None]
    if not trained_variants:
        print("No trained checkpoints found in checkpoints/ yet -- run `python -m src.train --variant ...` first.")
        return
    print(f"Found trained checkpoints for: {trained_variants}")

    # 2.1 + 2.2: per-line logged evaluation for every trained variant
    # -> results/<variant>_test_predictions.csv, one file per variant
    corpus_rows = {}
    for variant_name in trained_variants:
        model = load_variant_for_eval(variant_name, num_classes, device)
        df, corpus_cer, corpus_wer = evaluate_on_test_set_logged(
            model, test_loader, criterion, idx_to_char, device, variant_name)
        corpus_rows[variant_name] = df

    # efficiency table across every trained variant (one combined CSV, one row per variant)
    eff_df = build_efficiency_table_all_variants(test_dataset, num_classes, device)
    print(eff_df.to_string(index=False))

    # 2.3: bootstrap CI, computed per variant
    # -> results/bootstrap_ci_<variant>.csv
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

    # 2.4-2.7: everything relative to AHLR-VT
    # -> results/significance_tests_summary.csv, confusion_top25.csv,
    #    cer_vs_length.csv, greedy_vs_beam_comparison.csv
    if "AHLR-VT" in trained_variants:
        run_all_comparisons(proposed_name="AHLR-VT",
                             baseline_names=[v for v in trained_variants if v != "AHLR-VT"])

        ahlrvt_df = corpus_rows["AHLR-VT"]

        conf_df, _, composition = build_substitution_confusion(ahlrvt_df, top_n=25)
        conf_df.to_csv(os.path.join(RESULTS_DIR, "confusion_top25.csv"), index=False)
        print(composition)

        length_df = cer_vs_length_analysis(ahlrvt_df)
        length_df.to_csv(os.path.join(RESULTS_DIR, "cer_vs_length.csv"), index=False)

        model = load_variant_for_eval("AHLR-VT", num_classes, device)
        beam_df = compare_greedy_vs_beam(model, test_loader, idx_to_char, device)
        print(beam_df.to_string(index=False))
    else:
        print("AHLR-VT not trained yet -- skipping significance tests / confusion / "
              "length robustness / beam search (these are all defined relative to the proposed model).")

    # 2.9: consolidated manuscript summary table -- one row per variant
    # -> results/manuscript_summary_table.csv
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


if __name__ == "__main__":
    main()
