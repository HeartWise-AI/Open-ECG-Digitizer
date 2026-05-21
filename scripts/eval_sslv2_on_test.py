"""Resumable shard worker: run SSL v2 (77-label) on digitized test ECGs.

Pipeline:
    1. Classify layout + rotation (DINOv3-b)
    2. Undo rotation
    3. Digitize (multilayout U-Net → canonical_lines (12, 3000) with NaN fills)
    4. nan_to_num → * 0.001 (empirical digitizer→mV) → resample 2500
    5. SSL v2 ONNX → 77 logits → sigmoid → permute by WCR_COLUMN_CONVERSION
    6. Write 77 labeled columns + Acute_Obstruction + metadata

Resumable: skips rows already in shard CSV.
"""
import argparse
import csv
import os
import sys
import tempfile
import time

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
TARGET_LEN = 2500
DIG_TO_MV = 0.001

# Copied from /volume/DeepECG_Docker/utils/constants.py
ECG_PATTERNS = [
    "Sinusal", "Regular", "Monomorph", "QS complex in V1-V2-V3",
    "R complex in V5-V6", "T wave inversion (inferior - II, III, aVF)",
    "Left bundle branch block", "RaVL > 11 mm", "SV1 + RV5 or RV6 > 35 mm",
    "T wave inversion (lateral -I, aVL, V5-V6)",
    "T wave inversion (anterior - V3-V4)", "Left axis deviation",
    "Left ventricular hypertrophy", "Bradycardia",
    "Q wave (inferior - II, III, aVF)", "Afib", "Irregularly irregular",
    "Atrial tachycardia (>= 100 BPM)",
    "Nonspecific intraventricular conduction delay",
    "Premature ventricular complex", "Polymorph",
    "T wave inversion (septal- V1-V2)", "Right bundle branch block",
    "Ventricular paced", "ST elevation (anterior - V3-V4)",
    "ST elevation (septal - V1-V2)", "1st degree AV block",
    "Premature atrial complex", "Atrial flutter", "rSR' in V1-V2",
    "qRS in V5-V6-I, aVL", "Left anterior fascicular block",
    "Right axis deviation", "2nd degree AV block - mobitz 1",
    "ST depression (inferior - II, III, aVF)", "Acute pericarditis",
    "ST elevation (inferior - II, III, aVF)", "Low voltage",
    "Regularly irregular", "Junctional rhythm", "Left atrial enlargement",
    "ST elevation (lateral - I, aVL, V5-V6)", "Atrial paced",
    "Right ventricular hypertrophy", "Delta wave",
    "Wolff-Parkinson-White (Pre-excitation syndrome)", "Prolonged QT",
    "ST depression (anterior - V3-V4)", "QRS complex negative in III",
    "Q wave (lateral- I, aVL, V5-V6)", "Supraventricular tachycardia",
    "ST downslopping", "ST depression (lateral - I, avL, V5-V6)",
    "2nd degree AV block - mobitz 2", "U wave", "R/S ratio in V1-V2 >1",
    "RV1 + SV6 > 11 mm", "Left posterior fascicular block",
    "Right atrial enlargement", "ST depression (septal- V1-V2)",
    "Q wave (septal- V1-V2)", "Q wave (anterior - V3-V4)", "ST upslopping",
    "Right superior axis", "Ventricular tachycardia",
    "ST elevation (posterior - V7-V8-V9)",
    "Ectopic atrial rhythm (< 100 BPM)", "Lead misplacement",
    "Third Degree AV Block", "Acute MI", "Early repolarization",
    "Q wave (posterior - V7-V9)", "Bi-atrial enlargement", "LV pacing",
    "Brugada", "Ventricular Rhythm", "no_qrs",
]
assert len(ECG_PATTERNS) == 77

WCR_COLUMN_CONVERSION = [
    15, 23, 16, 1, 57, 63, 73, 41, 39, 36, 2, 29, 30, 65, 34, 12, 55, 56, 21, 8,
    42, 71, 37, 50, 13, 38, 46, 24, 49, 9, 66, 26, 40, 4, 22, 0, 11, 74, 64, 7,
    76, 58, 33, 70, 17, 6, 28, 69, 44, 61, 32, 72, 45, 25, 75, 18, 14, 5, 3, 31,
    27, 67, 62, 10, 43, 51, 52, 47, 19, 68, 53, 48, 60, 20, 59, 54, 35,
]
assert sorted(WCR_COLUMN_CONVERSION) == list(range(77))


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


def undo_rot(img, r):
    return np.ascontiguousarray(np.rot90(img, k=(4 - r)) if r > 0 else img)


def classify_with(model, img, device, crop=512):
    x = resize_pad(img, crop)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        lo, ro, *_ = model(t.to(device))
    return int(torch.softmax(lo, 1).argmax()), int(torch.softmax(ro, 1).argmax())


def digitize(wrapper, img_path, hint):
    from torchvision.io import ImageReadMode, read_image
    img = read_image(img_path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1: img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4: img = img[:3]
    got = wrapper(img.unsqueeze(0), layout_should_include_substring=hint)
    return (got.get("signal", {}).get("canonical_lines"),
            got.get("layout_name", "Unknown"),
            float(got.get("signal", {}).get("layout_matching_cost", float("inf"))))


def canon_to_mv(canon):
    data = canon.squeeze().cpu().numpy()
    if data.ndim == 1: data = data[None, :]
    if data.shape[0] < 12:
        data = np.concatenate([data,
                               np.full((12 - data.shape[0], data.shape[1]),
                                       np.nan, dtype=data.dtype)], axis=0)
    data = np.nan_to_num(data, nan=0.0) * DIG_TO_MV
    if data.shape[1] != TARGET_LEN:
        data = np.stack([resample(l, TARGET_LEN) for l in data])
    return data.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_csv", default="sandbox/sslv2_eval_1000.csv")
    ap.add_argument("--png_dir", default="/media/data1/ravram/DeepECG/ecg_png_parquet")
    ap.add_argument("--out_dir", default="sandbox/sslv2_eval")
    ap.add_argument("--sslv2_onnx",
                    default="/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.onnx")
    ap.add_argument("--layout_ckpt",
                    default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    ap.add_argument("--layout_backbone", default="dinov3_b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"shard_{args.shard:02d}_of_{args.nshards:02d}.csv")

    done = set()
    if os.path.isfile(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            done = set(prev["png"].dropna().astype(str).tolist())
        except Exception:
            done = set()

    df = pd.read_csv(args.subset_csv, low_memory=False)
    df = df[(df.index % args.nshards) == args.shard].reset_index(drop=True)

    device = torch.device(args.device)
    print(f"[shard {args.shard}/{args.nshards}] total={len(df)} done={len(done)} device={args.device}",
          flush=True)

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

    import onnxruntime as ort
    sess = ort.InferenceSession(args.sslv2_onnx, providers=["CPUExecutionProvider"])

    fields = ["png", "xml_path", "Acute_Obstruction", "layout_pred", "rot_pred",
              "digitizer_layout", "digitizer_cost", "seconds", "error"] + ECG_PATTERNS
    new_file = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    fp = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fp, fieldnames=fields)
    if new_file:
        writer.writeheader()
        fp.flush()

    t_start = time.time()
    n_here = 0
    total = len(df)
    processed = len(done)
    for i, row in df.iterrows():
        png = str(row["png"])
        if png in done:
            continue
        img_path = os.path.join(args.png_dir, png)
        rec = {k: "" for k in fields}
        rec["png"] = png
        rec["xml_path"] = row.get("xml_path", "")
        rec["Acute_Obstruction"] = int(row.get("Acute_Obstruction", 0))
        t0 = time.time()
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError("png missing")
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            lidx, ridx = classify_with(classifier, img, device)
            rec["layout_pred"] = LAYOUT_CLASSES[lidx]
            rec["rot_pred"] = ROT_CLASSES[ridx]
            corrected = undo_rot(img, ridx)
            with tempfile.TemporaryDirectory() as td:
                dig_in = os.path.join(td, "in.png")
                cv2.imwrite(dig_in, corrected)
                canon, layout_used, cost = digitize(wrapper, dig_in, LAYOUT_CLASSES[lidx])
            if canon is None:
                raise RuntimeError("no canonical")
            rec["digitizer_layout"] = layout_used
            rec["digitizer_cost"] = float(cost)
            ecg = canon_to_mv(canon)
            logits = sess.run(None, {"source": ecg[None]})[0].squeeze()
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs_ord = probs[WCR_COLUMN_CONVERSION]
            for name, p in zip(ECG_PATTERNS, probs_ord.tolist()):
                rec[name] = float(p)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(time.time() - t0, 2)
        writer.writerow(rec)
        fp.flush()
        processed += 1
        n_here += 1
        if n_here % 20 == 0:
            elapsed = time.time() - t_start
            rate = n_here / max(elapsed, 1e-3)
            remaining = (total - processed) / max(rate, 1e-6)
            print(f"[shard {args.shard}] {processed}/{total} "
                  f"({processed/total*100:.1f}%) {rate:.2f} img/s "
                  f"ETA {remaining/60:.1f} min  "
                  f"(AcuteMI={rec.get('Acute MI','ERR')})", flush=True)
    fp.close()
    print(f"[shard {args.shard}] DONE written={n_here}", flush=True)


if __name__ == "__main__":
    main()
