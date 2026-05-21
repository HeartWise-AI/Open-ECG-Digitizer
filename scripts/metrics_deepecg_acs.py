"""Merge shard CSVs, compute AUROC, AUPRC, sens/spec/PPV/NPV at thresholds."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)

OUT = "/volume/Open-ECG-Digitizer/sandbox/deepecg_acs_eval"
REGISTERED_THRESHOLD = 0.047  # from predict_folder.py:60 (Youden, n=4037)
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]


def at_threshold(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def _s(n, d): return float(n) / float(d) if d > 0 else float("nan")
    return {
        "threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "sens": _s(tp, tp+fn), "spec": _s(tn, tn+fp),
        "ppv": _s(tp, tp+fp), "npv": _s(tn, tn+fn),
        "accuracy": _s(tp+tn, tp+tn+fp+fn),
        "f1": _s(2*tp, 2*tp+fp+fn),
    }


def main():
    shards = [f"{OUT}/shard_0{i}_of_03.csv" for i in range(3)]
    dfs = [pd.read_csv(s, low_memory=False) for s in shards]
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate by png (shouldn't be any but safe)
    df = df.drop_duplicates(subset=["png"], keep="first").reset_index(drop=True)
    print(f"Total merged rows: {len(df):,}")

    # Separate errors from successes
    has_prob = df["acs_prob"].notna()
    errors = df[~has_prob].copy()
    success = df[has_prob].copy()
    print(f"  With ACS prob:  {len(success):,}")
    print(f"  Errors/missing: {len(errors):,}")

    # Load label from per_image merged subset to be safe
    y = success["label"].values.astype(int)
    p = success["acs_prob"].values.astype(float)

    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())
    print(f"  Positives (Acute_Obstruction=1): {pos_n:,}")
    print(f"  Negatives: {neg_n:,}  (prevalence = {pos_n/len(y)*100:.2f}%)")

    # Discrimination
    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    print(f"\nAUROC = {auroc:.4f}")
    print(f"AUPRC = {auprc:.4f}")

    # Youden from full n
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best = int(np.argmax(j))
    youden_thr = float(thr[best])
    youden_j = float(j[best])
    print(f"\nYouden-optimal threshold (in-sample, n={len(y):,}): {youden_thr:.4f}  (J={youden_j:.4f})")

    m_reg = at_threshold(y, p, REGISTERED_THRESHOLD)
    m_you = at_threshold(y, p, youden_thr)

    print(f"\n--- At registered threshold {REGISTERED_THRESHOLD} (predict_folder.py:60) ---")
    for k, v in m_reg.items():
        print(f"  {k:<10} {v}")
    print(f"\n--- At in-sample Youden {youden_thr:.4f} ---")
    for k, v in m_you.items():
        print(f"  {k:<10} {v}")

    # Top-vessel accuracy among TRUE ACS+ samples
    master = pd.read_csv(
        "/volume/Open-ECG-Digitizer/sandbox/deepecg_acs_test_subset.csv",
        low_memory=False,
    )
    have_vessel_cols = [v for v in VESSEL_CLASSES if v in master.columns]
    if len(have_vessel_cols) == len(VESSEL_CLASSES):
        master_slim = master[["png"] + VESSEL_CLASSES].set_index("png")
        success = success.merge(master_slim, left_on="png", right_index=True,
                                how="left", suffixes=("", "_true"))
    else:
        print(f"\n(Per-vessel AUC skipped: test CSV has no vessel ground-truth cols; "
              f"only binary Acute_Obstruction.)")
        # Write report now and exit
        report = {
            "n_total_processed": int(len(df)),
            "n_with_prob": int(len(success)),
            "n_errors": int(len(errors)),
            "n_positive": pos_n,
            "n_negative": neg_n,
            "prevalence": pos_n/len(y),
            "auroc": auroc,
            "auprc": auprc,
            "registered": {**m_reg, "source": "predict_folder.py:60"},
            "in_sample_youden": {**m_you, "youden_j": youden_j,
                                 "warning": f"threshold chosen on same n={len(y)}"},
        }
        with open(f"{OUT}/metrics_full.json", "w") as fp:
            json.dump(report, fp, indent=2)
        success.to_csv(f"{OUT}/all_success.csv", index=False)
        print(f"\nWrote {OUT}/metrics_full.json and all_success.csv")
        return
    vessel_true = success[VESSEL_CLASSES].copy()
    # Some datasets may have NaN vessel labels for unknown culprit — only scored when label=1 and
    # one vessel column is 1.
    acs_pos = success[success["label"] == 1].copy()
    # AUC per vessel among ACS+ with known vessel label
    per_vessel_auc = {}
    for v in VESSEL_CLASSES:
        if v not in acs_pos.columns:
            continue
        sub = acs_pos[acs_pos[v].notna()]
        true = sub[v].values.astype(int)
        pred = sub[v.lower() if v == "Left_Main" else v.lower()].values.astype(float) if v.lower() in sub.columns else sub["left_main"].values.astype(float) if v == "Left_Main" else None
        # More robustly:
        pred_col = {"LAD":"lad","RCA":"rca","LCX":"lcx","Left_Main":"left_main"}[v]
        pred = sub[pred_col].values.astype(float)
        if len(np.unique(true)) < 2:
            per_vessel_auc[v] = {"auc": float("nan"), "n": len(sub), "pos": int(true.sum())}
        else:
            per_vessel_auc[v] = {
                "auc": float(roc_auc_score(true, pred)),
                "auprc": float(average_precision_score(true, pred)),
                "n": len(sub),
                "pos": int(true.sum()),
            }

    print(f"\n--- Per-vessel AUC among ACS+ cases ---")
    for v, m in per_vessel_auc.items():
        auc = m.get("auc")
        ap = m.get("auprc")
        print(f"  {v:<10}  n={m['n']:>5} pos={m['pos']:>4}  "
              f"AUC={auc if auc is None else f'{auc:.4f}'}  "
              f"AP={ap if ap is None else f'{ap:.4f}' if ap is not None else 'NA'}")

    # Top-1 accuracy: argmax among 4 predicted vessel probs = the labeled culprit (for cases
    # with exactly one vessel == 1)
    single_culprit = acs_pos[acs_pos[VESSEL_CLASSES].sum(axis=1) == 1].copy()
    if len(single_culprit) > 0:
        true_v = single_culprit[VESSEL_CLASSES].idxmax(axis=1)
        pred_v = single_culprit[["lad","rca","lcx","left_main"]].idxmax(axis=1).map(
            {"lad":"LAD","rca":"RCA","lcx":"LCX","left_main":"Left_Main"})
        top1 = (true_v.values == pred_v.values).mean()
        print(f"\nTop-1 vessel accuracy (single-culprit ACS+ cases, n={len(single_culprit):,}): {top1:.4f}")

    # Write report
    report = {
        "n_total_processed": int(len(df)),
        "n_with_prob": int(len(success)),
        "n_errors": int(len(errors)),
        "n_positive": pos_n,
        "n_negative": neg_n,
        "prevalence": pos_n/len(y),
        "auroc": auroc,
        "auprc": auprc,
        "registered": {**m_reg, "source": "predict_folder.py:60"},
        "in_sample_youden": {**m_you, "youden_j": youden_j,
                             "warning": f"threshold chosen on same n={len(y)}"},
        "per_vessel_auc_among_acs_pos": per_vessel_auc,
    }
    if len(single_culprit) > 0:
        report["vessel_top1_accuracy_among_single_culprit_acs_pos"] = float(top1)
        report["n_single_culprit_acs_pos"] = int(len(single_culprit))
    with open(f"{OUT}/metrics_full.json", "w") as fp:
        json.dump(report, fp, indent=2)
    success.to_csv(f"{OUT}/all_success.csv", index=False)
    print(f"\nWrote {OUT}/metrics_full.json and all_success.csv")


if __name__ == "__main__":
    main()
