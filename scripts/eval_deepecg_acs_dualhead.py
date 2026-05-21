"""Resumable shard worker: run dual-head ACS ONNX on a slice of the DeepECG test set.

Per image:
    1. Classify layout + rotation (DINOv3-b classifier)
    2. Undo rotation
    3. Digitize (multilayout U-Net → canonical 12-lead)
    4. Resample to 2500 samples * PTBXL_POWER_RATIO
    5. Dual-head ONNX → ACS prob + 4 vessel probs

Writes one CSV per shard. Resumable: on startup, rows already present in the
shard CSV are skipped. Safe to kill/restart workers.

Usage (3-GPU example):
    python scripts/eval_deepecg_acs_dualhead.py \
        --subset_csv sandbox/deepecg_acs_test_subset.csv \
        --png_dir /media/data1/ravram/DeepECG/ecg_png_parquet \
        --out_dir sandbox/deepecg_acs_eval \
        --shard 0 --nshards 3 --device cuda:0 &
    python scripts/eval_deepecg_acs_dualhead.py ... --shard 1 --device cuda:2 &
    python scripts/eval_deepecg_acs_dualhead.py ... --shard 2 --device cuda:3 &
"""
import argparse
import csv
import os
import sys
import tempfile
import time
import traceback

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
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
TARGET_LEN = 2500
PTBXL_POWER_RATIO = 3.003154

FIELDS = ["png", "xml_path", "label", "acs_prob",
          "lad", "rca", "lcx", "left_main",
          "top_vessel", "top_vessel_prob",
          "layout_pred", "rot_pred", "digitizer_layout", "digitizer_cost",
          "seconds", "error"]


def undo_rotation(img, rot):
    if rot > 0:
        img = np.rot90(img, k=(4 - rot))
    return np.ascontiguousarray(img)


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


def load_layout_classifier(ckpt, backbone, device):
    m = LayoutOrientModel(num_layouts=len(LAYOUT_CLASSES),
                          num_rotations=len(ROT_CLASSES),
                          num_flips=1, backbone=backbone).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def classify_with(model, img, device, crop=512):
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
    layout_used = got.get("layout_name", "Unknown")
    cost = float(got.get("signal", {}).get("layout_matching_cost", float("inf")))
    return canon, layout_used, cost


def canonical_to_ecg(canon):
    data = canon.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    n = data.shape[0]
    df = pd.DataFrame(data.T, columns=LEAD_ORDER[:n])
    leads = []
    for name in LEAD_ORDER:
        v = np.nan_to_num(df[name].values.astype(np.float32), nan=0.0) \
            if name in df.columns else np.zeros(len(df), dtype=np.float32)
        leads.append(v)
    ecg = np.stack(leads, axis=0)
    if ecg.shape[1] != TARGET_LEN:
        ecg = np.stack([resample(l, TARGET_LEN) for l in ecg], axis=0).astype(np.float32)
    return (ecg * PTBXL_POWER_RATIO).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_csv", required=True)
    ap.add_argument("--png_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--acs_onnx", default="weights/exported/acs_dual_head.onnx")
    ap.add_argument("--layout_ckpt",
                    default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    ap.add_argument("--layout_backbone", default="dinov3_b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"shard_{args.shard:02d}_of_{args.nshards:02d}.csv")
    # Resume: collect pngs already processed
    done = set()
    if os.path.isfile(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            done = set(prev["png"].dropna().astype(str).tolist())
        except Exception:
            done = set()

    # Build shard (stable slicing by index modulo)
    df = pd.read_csv(args.subset_csv, low_memory=False)
    df = df[(df.index % args.nshards) == args.shard].reset_index(drop=True)
    total = len(df)

    # Load models once
    device = torch.device(args.device)
    print(f"[shard {args.shard}/{args.nshards}] total={total} done={len(done)} "
          f"device={args.device}", flush=True)

    classifier = load_layout_classifier(args.layout_ckpt, args.layout_backbone, device)

    from src.model.inference_wrapper import InferenceWrapper
    wrapper = InferenceWrapper(config=CN(_inference_cfg(str(device))), device=str(device),
                               resample_size=3000, rotate_on_resample=True,
                               enable_timing=False, apply_dewarping=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.acs_onnx, providers=["CPUExecutionProvider"])

    # Append mode; write header only if file is new
    new_file = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    fp = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fp, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        fp.flush()

    t_start = time.time()
    processed = len(done)
    n_done_here = 0
    for i, row in df.iterrows():
        png = str(row.get("png") or "")
        if not png:
            xml = str(row.get("xml_path") or "")
            png = os.path.basename(xml) + ".png" if xml else ""
        if not png or png in done:
            continue
        img_path = os.path.join(args.png_dir, png)
        rec = {k: "" for k in FIELDS}
        rec["png"] = png
        rec["xml_path"] = row.get("xml_path", "")
        rec["label"] = int(row.get("Acute_Obstruction", 0))
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
            corrected = undo_rotation(img, ridx)
            with tempfile.TemporaryDirectory() as td:
                dig_in = os.path.join(td, "in.png")
                cv2.imwrite(dig_in, corrected)
                canon, layout_used, cost = digitize(wrapper, dig_in, hint=LAYOUT_CLASSES[lidx])
            if canon is None:
                raise RuntimeError("digitizer returned no canonical signal")
            rec["digitizer_layout"] = layout_used
            rec["digitizer_cost"] = float(cost)
            ecg = canonical_to_ecg(canon)
            out = sess.run(None, {"ecg_12lead": ecg[None]})
            acs = float(out[0].squeeze())
            vessels = out[1].squeeze().astype(float).tolist()
            rec["acs_prob"] = acs
            rec["lad"], rec["rca"], rec["lcx"], rec["left_main"] = vessels
            top_i = int(np.argmax(vessels))
            rec["top_vessel"] = VESSEL_CLASSES[top_i]
            rec["top_vessel_prob"] = float(vessels[top_i])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            # Keep shard going on individual failures
        rec["seconds"] = round(time.time() - t0, 2)
        writer.writerow(rec)
        fp.flush()
        processed += 1
        n_done_here += 1
        if n_done_here % 20 == 0:
            elapsed = time.time() - t_start
            rate = n_done_here / max(elapsed, 1e-3)
            remaining = (total - processed) / max(rate, 1e-6)
            print(f"[shard {args.shard}] {processed}/{total} "
                  f"({processed/total*100:.1f}%) "
                  f"{rate:.2f} img/s  ETA {remaining/3600:.1f} h  "
                  f"last={rec.get('acs_prob','ERR')}", flush=True)
    fp.close()
    print(f"[shard {args.shard}] DONE  written={n_done_here}  total_rows={processed}", flush=True)


if __name__ == "__main__":
    main()
