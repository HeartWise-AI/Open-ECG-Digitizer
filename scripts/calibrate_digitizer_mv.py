"""Derive a single global scale factor so digitized signals match raw-MHI mV range.

For each sample:
    - Load raw MHI npy, scale by 0.00488 (documented MHI ADC→mV).
    - Digitize PNG → canonical_lines (digitizer output).
    - Compute per-lead stats (P50 |x|, P95 |x|).
    - Ratio = raw_mV / digitizer_out per lead.

Final scale factor = median of per-image, per-lead ratios (robust to outliers).
Save per-image summary CSV + recommended scale.
"""
import argparse
import os
import sys
import tempfile
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from yacs.config import CfgNode as CN

sys.path.insert(0, "/volume/Open-ECG-Digitizer")
sys.path.insert(0, "/volume/Open-ECG-Digitizer/scripts")

from eval_layout_classifier import (LAYOUT_CLASSES, ROT_CLASSES,
                                    LayoutOrientModel, resize_pad)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
MHI_SCALE = 0.00488
TARGET_LEN = 2500


def _inference_cfg(device):
    return {
        "SIGNAL_EXTRACTOR": {"class_path": "src.model.signal_extractor.SignalExtractor", "KWARGS": {}},
        "PERSPECTIVE_DETECTOR": {"class_path": "src.model.perspective_detector.PerspectiveDetector",
                                 "KWARGS": {"num_thetas": 250}},
        "DEWARPER": {"class_path": "src.model.dewarper.Dewarper",
                     "KWARGS": {"abs_peak_threshold": 0.1}},
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
                              "KWARGS": {"debug": False, "device": device,
                                         "possibly_flipped": False}},
    }


def undo_rotation(img, rot):
    if rot > 0:
        img = np.rot90(img, k=(4 - rot))
    return np.ascontiguousarray(img)


def classify(model, img, device, crop=512):
    x = resize_pad(img, crop)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        lo, ro, *_ = model(t.to(device))
    return int(torch.softmax(lo, 1).argmax()), int(torch.softmax(ro, 1).argmax())


def digitize(wrapper, img_path, hint=None):
    from torchvision.io import ImageReadMode, read_image
    img = read_image(img_path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1:
        img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4:
        img = img[:3]
    got = wrapper(img.unsqueeze(0), layout_should_include_substring=hint)
    canon = got.get("signal", {}).get("canonical_lines")
    layout = got.get("layout_name", "Unknown")
    cost = float(got.get("signal", {}).get("layout_matching_cost", float("inf")))
    return canon, layout, cost


def stats_12lead(ecg):
    """ecg shape (12, N). Returns dict per-lead P50|x|, P95|x|."""
    out = {}
    for i, name in enumerate(LEAD_ORDER):
        x = np.abs(ecg[i])
        out[f"{name}_p50"] = float(np.median(x))
        out[f"{name}_p95"] = float(np.percentile(x, 95))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_csv", default="sandbox/mv_calibration_sample.csv")
    ap.add_argument("--png_dir", default="/media/data1/ravram/DeepECG/ecg_png_parquet")
    ap.add_argument("--out_csv", default="sandbox/mv_calibration_results.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--layout_ckpt",
                    default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    ap.add_argument("--layout_backbone", default="dinov3_b")
    args = ap.parse_args()

    df = pd.read_csv(args.sample_csv, low_memory=False).head(args.n)
    device = torch.device(args.device)

    classifier = LayoutOrientModel(num_layouts=len(LAYOUT_CLASSES),
                                   num_rotations=len(ROT_CLASSES),
                                   num_flips=1, backbone=args.layout_backbone).to(device)
    classifier.load_state_dict(torch.load(args.layout_ckpt, map_location=device,
                                          weights_only=True))
    classifier.eval()

    from src.model.inference_wrapper import InferenceWrapper
    wrapper = InferenceWrapper(config=CN(_inference_cfg(str(device))), device=str(device),
                               resample_size=3000, rotate_on_resample=True,
                               enable_timing=False, apply_dewarping=False)

    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        png = os.path.join(args.png_dir, str(r["png"]))
        npy = str(r["npy_path"])
        if not os.path.exists(png) or not os.path.exists(npy):
            continue
        try:
            raw = np.load(npy).squeeze()  # may be (2500, 12) or (12, 2500)
            if raw.ndim == 3:
                raw = raw.squeeze(-1)
            if raw.shape[0] != 12:
                raw = raw.T
            raw_mv = raw * MHI_SCALE
            if raw_mv.shape[1] != TARGET_LEN:
                raw_mv = np.stack([resample(l, TARGET_LEN) for l in raw_mv]).astype(np.float32)
            raw_stats = stats_12lead(raw_mv)

            img = cv2.imread(png)
            lidx, ridx = classify(classifier, img, device)
            corrected = undo_rotation(img, ridx)
            with tempfile.TemporaryDirectory() as td:
                dig_in = os.path.join(td, "in.png")
                cv2.imwrite(dig_in, corrected)
                canon, _lay, _cost = digitize(wrapper, dig_in, hint=LAYOUT_CLASSES[lidx])
            if canon is None:
                raise RuntimeError("no canonical")
            data = canon.squeeze().cpu().numpy()
            if data.ndim == 1:
                data = data[None, :]
            # Pad to 12 leads if needed
            if data.shape[0] < 12:
                pad = np.zeros((12 - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.concatenate([data, pad], axis=0)
            if data.shape[1] != TARGET_LEN:
                data = np.stack([resample(l, TARGET_LEN) for l in data]).astype(np.float32)
            dig_stats = stats_12lead(data)

            rec = {"png": r["png"]}
            for k in raw_stats:
                rec[f"raw_{k}"] = raw_stats[k]
                rec[f"dig_{k}"] = dig_stats[k]
            # Per-image ratios (P95 abs) — one per lead
            for name in LEAD_ORDER:
                dig95 = dig_stats[f"{name}_p95"]
                rec[f"{name}_ratio_p95"] = raw_stats[f"{name}_p95"] / dig95 \
                    if dig95 > 1e-9 else float("nan")
            rows.append(rec)
            if (len(rows)) % 10 == 0:
                elapsed = time.time() - t0
                print(f"[{len(rows)}/{args.n}] {elapsed/len(rows):.1f}s/img", flush=True)
        except Exception as e:
            print(f"ERR {r['png']}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    ratio_cols = [c for c in out.columns if c.endswith("_ratio_p95")]
    all_ratios = out[ratio_cols].values.ravel()
    all_ratios = all_ratios[np.isfinite(all_ratios)]
    print(f"\nTotal per-lead ratios: {len(all_ratios)}")
    print(f"Median ratio (raw_mV / digitizer): {np.median(all_ratios):.4f}")
    print(f"Mean: {np.mean(all_ratios):.4f}  P25: {np.percentile(all_ratios,25):.4f}  "
          f"P75: {np.percentile(all_ratios,75):.4f}")
    per_lead = {}
    for name in LEAD_ORDER:
        vals = out[f"{name}_ratio_p95"].dropna().values
        per_lead[name] = np.median(vals) if len(vals) else np.nan
        print(f"  {name:<4}  median ratio = {per_lead[name]:.4f}")
    print(f"\nRecommended global scale = {np.median(all_ratios):.4f}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
