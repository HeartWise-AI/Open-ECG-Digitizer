"""Eval ACS dual-head ONNX on sandbox/test_ecgs labeled images.

Labels parsed from filenames:
    *_NO_STEMI_*  -> 0 negative
    *_STEMI_*     -> 1 positive  (includes STEMI_MISSED cases)
    *_UNKNOWN_*   -> skipped
Cropped variants (_cropped, _dino_crop) of the same case are reported
separately but excluded from the primary metrics to avoid leakage.

Outputs:
    - per-image prob CSV
    - metrics JSON with AUROC, AP, confusion/sens/spec/PPV/NPV/F1 at
      (a) in-sample Youden threshold and
      (b) pre-registered threshold 0.047 (Youden-optimal, n=4037;
          see scripts/predict_folder.py:60).
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from yacs.config import CfgNode as CN

sys.path.insert(0, "/volume/Open-ECG-Digitizer")
sys.path.insert(0, "/volume/Open-ECG-Digitizer/scripts")

from eval_layout_classifier import (LAYOUT_CLASSES, ROT_CLASSES,
                                    LayoutOrientModel, resize_pad)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
TARGET_LEN = 2500
PTBXL_POWER_RATIO = 3.003154
REGISTERED_THRESHOLD = 0.047  # Youden-optimal, n=4037 (predict_folder.py:60)


def label_from_name(name: str):
    base = os.path.splitext(os.path.basename(name))[0]
    if "UNKNOWN" in base:
        return None
    if "NO_STEMI" in base or "NO_ACS" in base:
        return 0
    if "STEMI" in base or "_ACS" in base:
        return 1
    return None


def is_cropped_variant(name: str) -> bool:
    b = os.path.basename(name)
    return "_cropped" in b or "_dino_crop" in b


def undo_rotation(img, rot):
    if rot > 0:
        img = np.rot90(img, k=(4 - rot))
    return np.ascontiguousarray(img)


def classify_layout_rot(img_path, ckpt, backbone, device, crop=512):
    m = LayoutOrientModel(num_layouts=len(LAYOUT_CLASSES),
                          num_rotations=len(ROT_CLASSES),
                          num_flips=1, backbone=backbone).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    img = cv2.imread(img_path)
    x = resize_pad(img, crop)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        lo, ro, *_ = m(t.to(device))
    return img, int(torch.softmax(lo, 1).argmax()), int(torch.softmax(ro, 1).argmax())


def _inference_cfg(device):
    return {
        "SIGNAL_EXTRACTOR": {"class_path": "src.model.signal_extractor.SignalExtractor", "KWARGS": {}},
        "PERSPECTIVE_DETECTOR": {"class_path": "src.model.perspective_detector.PerspectiveDetector", "KWARGS": {"num_thetas": 250}},
        "DEWARPER": {"class_path": "src.model.dewarper.Dewarper", "KWARGS": {"abs_peak_threshold": 0.1}},
        "SEGMENTATION_MODEL": {"class_path": "src.model.unet.UNet",
                               "weight_path": "./weights/unet_multilayout/best_weights.pt",
                               "KWARGS": {"num_in_channels": 3, "num_out_channels": 4,
                                          "dims": [32, 64, 128, 256, 320, 320, 320, 320], "depth": 2}},
        "CROPPER": {"class_path": "src.model.cropper.Cropper",
                    "KWARGS": {"granularity": 80, "percentiles": [0.02, 0.98], "alpha": 0.85}},
        "PIXEL_SIZE_FINDER": {"class_path": "src.model.pixel_size_finder.PixelSizeFinder",
                              "KWARGS": {"min_number_of_grid_lines": 30,
                                         "max_number_of_grid_lines": 70,
                                         "lower_grid_line_factor": 0.3}},
        "LAYOUT_IDENTIFIER": {"class_path": "src.model.lead_identifier.LeadIdentifier",
                              "config_path": "src/config/lead_layouts_all.yml",
                              "unet_config_path": "src/config/lead_name_unet.yml",
                              "unet_weight_path": "./weights/lead_name_unet_weights_07072025.pt",
                              "KWARGS": {"debug": False, "device": device, "possibly_flipped": False}},
    }


_WRAPPER = None
def _wrapper(device):
    global _WRAPPER
    if _WRAPPER is None:
        from src.model.inference_wrapper import InferenceWrapper
        _WRAPPER = InferenceWrapper(config=CN(_inference_cfg(device)), device=device,
                                    resample_size=3000, rotate_on_resample=True,
                                    enable_timing=False, apply_dewarping=False)
    return _WRAPPER


def digitize(img_path, device, hint=None):
    from torchvision.io import ImageReadMode, read_image
    img = read_image(img_path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1: img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4: img = img[:3]
    got = _wrapper(device)(img.unsqueeze(0), layout_should_include_substring=hint)
    canon = got.get("signal", {}).get("canonical_lines")
    layout = got.get("layout_name", "Unknown")
    cost = float(got.get("signal", {}).get("layout_matching_cost", float("inf")))
    return canon, layout, cost


def canonical_to_ecg(canon):
    data = canon.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    n_leads = data.shape[0]
    df = pd.DataFrame(data.T, columns=LEAD_ORDER[:n_leads])
    leads = []
    for name in LEAD_ORDER:
        v = np.nan_to_num(df[name].values.astype(np.float32), nan=0.0) \
            if name in df.columns else np.zeros(len(df), dtype=np.float32)
        leads.append(v)
    ecg = np.stack(leads, axis=0)
    if ecg.shape[1] != TARGET_LEN:
        ecg = np.stack([resample(l, TARGET_LEN) for l in ecg], axis=0).astype(np.float32)
    return (ecg * PTBXL_POWER_RATIO).astype(np.float32)


def metrics_at_threshold(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def _safe(num, den):
        return float(num) / float(den) if den > 0 else float("nan")
    sens = _safe(tp, tp + fn)
    spec = _safe(tn, tn + fp)
    ppv = _safe(tp, tp + fp)
    npv = _safe(tn, tn + fn)
    acc = _safe(tp + tn, tp + tn + fp + fn)
    f1 = _safe(2 * tp, 2 * tp + fp + fn)
    return {"threshold": float(thr), "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn), "sens": sens, "spec": spec,
            "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1}


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(j[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sandbox/test_ecgs")
    ap.add_argument("--out_dir", default="sandbox/test_ecgs/_acs_eval")
    ap.add_argument("--acs_onnx", default="weights/exported/acs_dual_head.onnx")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layout_ckpt",
                    default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    ap.add_argument("--layout_backbone", default="dinov3_b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Collect images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.dir, e)))
    paths = sorted(paths)

    rows = []
    import onnxruntime as ort
    sess = ort.InferenceSession(args.acs_onnx, providers=["CPUExecutionProvider"])

    for i, p in enumerate(paths, 1):
        y = label_from_name(p)
        if y is None:
            print(f"[{i}/{len(paths)}] SKIP (unlabeled): {os.path.basename(p)}")
            continue
        is_crop = is_cropped_variant(p)
        t0 = time.time()
        try:
            img, lidx, ridx = classify_layout_rot(
                p, args.layout_ckpt, args.layout_backbone, device)
            corrected = undo_rotation(img, ridx)
            with tempfile.TemporaryDirectory() as td:
                dig = os.path.join(td, "in.png")
                cv2.imwrite(dig, corrected)
                canon, layout_used, cost = digitize(
                    dig, str(device), hint=LAYOUT_CLASSES[lidx])
            if canon is None:
                raise RuntimeError("digitizer returned no canonical signal")
            ecg = canonical_to_ecg(canon)
            out = sess.run(None, {"ecg_12lead": ecg[None]})
            acs = float(out[0].squeeze())
            vessels = dict(zip(VESSEL_CLASSES,
                               out[1].squeeze().astype(float).tolist()))
            top = max(vessels.items(), key=lambda kv: kv[1])
        except Exception as e:
            print(f"[{i}/{len(paths)}] ERROR on {os.path.basename(p)}: {e}")
            rows.append({"image": os.path.basename(p), "label": y,
                         "is_variant": is_crop, "acs_prob": None,
                         "top_vessel": None, "top_vessel_prob": None,
                         "error": str(e)})
            continue
        print(f"[{i}/{len(paths)}] {os.path.basename(p):<45} "
              f"y={y} acs={acs:.4f} top={top[0]}={top[1]:.3f}  "
              f"[{time.time()-t0:.1f}s]")
        rows.append({"image": os.path.basename(p), "label": y,
                     "is_variant": is_crop,
                     "layout_pred": LAYOUT_CLASSES[lidx],
                     "rot_pred": ROT_CLASSES[ridx],
                     "digitizer_layout": layout_used, "digitizer_cost": cost,
                     "acs_prob": acs,
                     "lad": vessels["LAD"], "rca": vessels["RCA"],
                     "lcx": vessels["LCX"], "left_main": vessels["Left_Main"],
                     "top_vessel": top[0], "top_vessel_prob": top[1]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_image.csv"), index=False)

    def case_key(name):
        b = os.path.splitext(os.path.basename(name))[0]
        return b.replace("_cropped", "").replace("_dino_crop", "")

    df["case_key"] = df["image"].apply(case_key)
    primary_rows = []
    digitization_failures = []
    for _, grp in df.groupby("case_key"):
        # Prefer a non-variant row with a valid prob; else first successful variant.
        cand = grp[~grp["is_variant"] & grp["acs_prob"].notna()]
        if len(cand) == 0:
            cand = grp[grp["acs_prob"].notna()]
        if len(cand) == 0:
            digitization_failures.append(grp.iloc[0]["image"])
            continue
        primary_rows.append(cand.iloc[0])
    primary = pd.DataFrame(primary_rows)
    y = primary["label"].values.astype(int)
    p = primary["acs_prob"].values.astype(float)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap_score = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    yj_thr, yj_val = youden_threshold(y, p) if len(np.unique(y)) > 1 else (float("nan"), float("nan"))
    reg_metrics = metrics_at_threshold(y, p, REGISTERED_THRESHOLD)
    yj_metrics = metrics_at_threshold(y, p, yj_thr)

    report = {
        "dataset": {
            "n_total_images": len(df),
            "n_labeled_unique_cases": int(len(primary)),
            "n_positive": int(primary["label"].sum()),
            "n_negative": int((primary["label"] == 0).sum()),
            "excluded_variants": int(df["is_variant"].sum()),
            "digitization_failures": digitization_failures,
            "caveat": ("Sample is tiny. AUROC/PPV/NPV from this set are "
                       "anecdotal; in-sample Youden overfits this n."),
        },
        "model": {"acs_onnx": args.acs_onnx,
                  "head": "dual_head (ACS + 4-class vessel)"},
        "metrics": {
            "auroc": auc,
            "average_precision": ap_score,
            "at_registered_threshold": {**reg_metrics,
                                        "source": "predict_folder.py:60 "
                                                  "(Youden-optimal, n=4037)"},
            "at_in_sample_youden": {**yj_metrics,
                                    "youden_j": yj_val,
                                    "warning": "Chosen on this same tiny set; "
                                               "not generalizable."},
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fp:
        json.dump(report, fp, indent=2)

    print()
    print("=" * 80)
    print(f"Dataset: n={len(primary)}  (pos={int(primary['label'].sum())}, "
          f"neg={int((primary['label']==0).sum())}, "
          f"variants excluded={int(df['is_variant'].sum())})")
    print(f"AUROC = {auc:.4f}")
    print(f"AP    = {ap_score:.4f}")
    print()
    print(f"At registered threshold {REGISTERED_THRESHOLD} "
          "(Youden-optimal, n=4037):")
    for k, v in reg_metrics.items():
        print(f"  {k:<10} {v}")
    print()
    print(f"At in-sample Youden threshold {yj_thr:.4f} "
          f"(J={yj_val:.3f}) — WARNING: overfits this tiny N:")
    for k, v in yj_metrics.items():
        print(f"  {k:<10} {v}")
    print("=" * 80)
    print(f"Per-image: {os.path.join(args.out_dir, 'per_image.csv')}")
    print(f"Metrics  : {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
