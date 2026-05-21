"""Merge SSL v2 eval shards; compute AUROC/AUPRC per label vs Acute_Obstruction.

ACS-relevant labels (see Notion DeepECG-SSL v2 page):
    Acute MI, ST elevation * (anterior/septal/inferior/lateral/posterior),
    Q wave * (anterior/septal/inferior/lateral/posterior),
    ST depression * (anterior/septal/inferior/lateral),
    T wave inversion * (anterior/inferior/lateral/septal).

Also computes:
    - max-over-ACS-labels aggregated score
    - AUROC table and per-label Youden flag counts
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

OUT = "/volume/Open-ECG-Digitizer/sandbox/sslv2_eval"

ACS_LABELS = [
    "Acute MI",
    "ST elevation (anterior - V3-V4)",
    "ST elevation (septal - V1-V2)",
    "ST elevation (inferior - II, III, aVF)",
    "ST elevation (lateral - I, aVL, V5-V6)",
    "ST elevation (posterior - V7-V8-V9)",
    "Q wave (anterior - V3-V4)",
    "Q wave (septal- V1-V2)",
    "Q wave (inferior - II, III, aVF)",
    "Q wave (lateral- I, aVL, V5-V6)",
    "Q wave (posterior - V7-V9)",
    "ST depression (anterior - V3-V4)",
    "ST depression (inferior - II, III, aVF)",
    "ST depression (lateral - I, avL, V5-V6)",
    "ST depression (septal- V1-V2)",
    "T wave inversion (anterior - V3-V4)",
    "T wave inversion (inferior - II, III, aVF)",
    "T wave inversion (lateral -I, aVL, V5-V6)",
    "T wave inversion (septal- V1-V2)",
    "Acute pericarditis",
    "Early repolarization",
]

# Notion Youden thresholds (v2, MHI test n=287K)
V2_YOUDEN = {
    "Acute MI": 0.0064,
    "ST elevation (anterior - V3-V4)": 0.0009,
    "ST elevation (septal - V1-V2)": 0.0063,
    "ST elevation (inferior - II, III, aVF)": 0.0101,
    "ST elevation (lateral - I, aVL, V5-V6)": 0.0328,
    "ST elevation (posterior - V7-V8-V9)": 0.0001,
    "Q wave (inferior - II, III, aVF)": 0.0534,
    "Q wave (septal- V1-V2)": 0.0071,
    "Q wave (anterior - V3-V4)": 0.0046,
    "Q wave (lateral- I, aVL, V5-V6)": 0.0003,
    "ST depression (inferior - II, III, aVF)": 0.0118,
    "ST depression (anterior - V3-V4)": 0.0210,
    "ST depression (lateral - I, avL, V5-V6)": 0.0419,
    "ST depression (septal- V1-V2)": 0.0493,
    "T wave inversion (inferior - II, III, aVF)": 0.0075,
    "T wave inversion (anterior - V3-V4)": 0.0083,
    "T wave inversion (lateral -I, aVL, V5-V6)": 0.0003,
    "T wave inversion (septal- V1-V2)": 0.0029,
    "Acute pericarditis": 0.0120,
    "Early repolarization": 0.0030,
}


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(j[i]), float(tpr[i]), float(1-fpr[i])


def main():
    dfs = [pd.read_csv(f"{OUT}/shard_0{i}_of_03.csv", low_memory=False)
           for i in range(3)]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("png").reset_index(drop=True)
    print(f"Merged rows: {len(df):,}")

    has_pred = df[ACS_LABELS[0]].notna()
    errors = int((~has_pred).sum())
    success = df[has_pred].copy()
    print(f"Success: {len(success):,}  Errors: {errors}")
    y = success["Acute_Obstruction"].values.astype(int)
    print(f"Positives: {int(y.sum())}  Negatives: {int((y==0).sum())}")

    # Per-label AUROC vs Acute_Obstruction
    rows = []
    for lbl in ACS_LABELS:
        if lbl not in success.columns:
            continue
        p = success[lbl].values.astype(float)
        if np.isnan(p).any():
            mask = ~np.isnan(p)
            p, y_m = p[mask], y[mask]
        else:
            y_m = y
        try:
            auc = roc_auc_score(y_m, p)
            ap = average_precision_score(y_m, p)
        except Exception:
            auc, ap = float("nan"), float("nan")
        thr_you, j_you, sens_you, spec_you = youden(y_m, p)
        thr_notion = V2_YOUDEN.get(lbl)
        if thr_notion is not None:
            pred = (p >= thr_notion).astype(int)
            tp = int(((pred == 1) & (y_m == 1)).sum())
            fp = int(((pred == 1) & (y_m == 0)).sum())
            tn = int(((pred == 0) & (y_m == 0)).sum())
            fn = int(((pred == 0) & (y_m == 1)).sum())
            sens_n = tp / (tp + fn) if tp + fn else float("nan")
            spec_n = tn / (tn + fp) if tn + fp else float("nan")
            ppv_n = tp / (tp + fp) if tp + fp else float("nan")
        else:
            sens_n = spec_n = ppv_n = float("nan")
            tp = fp = tn = fn = None
        rows.append({
            "label": lbl,
            "auroc": auc,
            "auprc": ap,
            "in_sample_youden": thr_you,
            "youden_j": j_you,
            "notion_threshold": thr_notion,
            "at_notion_sens": sens_n,
            "at_notion_spec": spec_n,
            "at_notion_ppv": ppv_n,
            "TP_notion": tp, "FP_notion": fp, "TN_notion": tn, "FN_notion": fn,
        })
    tbl = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print("\n== Per-label AUROC / AUPRC (vs Acute_Obstruction) ==")
    with pd.option_context("display.max_colwidth", 60,
                           "display.width", 160,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(tbl[["label", "auroc", "auprc", "notion_threshold",
                   "at_notion_sens", "at_notion_spec", "at_notion_ppv"]].to_string(index=False))

    # Aggregated: max over ACS labels per row
    agg = success[ACS_LABELS].max(axis=1)
    auc_agg = roc_auc_score(y, agg)
    ap_agg = average_precision_score(y, agg)
    thr_y, j_y, sens_y, spec_y = youden(y, agg)
    print(f"\n== Aggregated max-over-ACS-labels score ==")
    print(f"  AUROC = {auc_agg:.4f}")
    print(f"  AUPRC = {ap_agg:.4f}")
    print(f"  Youden threshold = {thr_y:.4f} (J={j_y:.4f}, sens={sens_y:.3f}, spec={spec_y:.3f})")

    # Save
    tbl.to_csv(f"{OUT}/per_label_metrics.csv", index=False)
    with open(f"{OUT}/metrics_summary.json", "w") as fp:
        json.dump({
            "n_success": int(len(success)),
            "n_errors": errors,
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "aggregated_max_acs": {
                "auroc": float(auc_agg),
                "auprc": float(ap_agg),
                "youden_threshold": thr_y,
                "sens": sens_y, "spec": spec_y, "j": j_y,
            },
            "per_label": tbl.to_dict(orient="records"),
        }, fp, indent=2)
    print(f"\nWrote {OUT}/per_label_metrics.csv + metrics_summary.json")


if __name__ == "__main__":
    main()
