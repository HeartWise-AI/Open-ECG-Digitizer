"""Single-image ECG report: Layout+Rotation -> Digitize -> ACS + Top Vessel + LVEF.

Prints a per-label table with model file, threshold, probability, and flag.
Thresholds are printed alongside each label so the provenance is obvious.

Usage:
    python scripts/run_full_ecg_report.py \\
        --image sandbox/test_ecgs/3x4R1_R0_NO_STEMI_1.png \\
        --out_dir sandbox/test_ecgs/_report_out/3x4R1_R0_NO_STEMI_1

Model files (defaults):
    Layout+Rot   : weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt
    U-Net seg    : weights/unet_multilayout/best_weights.pt
    Lead-id U-Net: weights/lead_name_unet_weights_07072025.pt
    ACS + Vessel : weights/exported/acs_dual_head.onnx
    LVEF <=40%   : heartwise/wcr_lvef_equal_under_40  (auto-downloaded to weights/)
    LVEF <50%    : heartwise/wcr_lvef_under_50        (auto-downloaded to weights/)
    WCR SSL base : /volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt

Thresholds (defaults, editable via flags):
    ACS           : 0.047   -- Youden-optimal, n=4037 test set (predict_folder.py)
                               sens 74.1%, spec 87.8%.
                               Source head: augment-v4 WCR; reused for dual-head ONNX.
    Top vessel    : argmax   -- no scalar threshold; pick argmax among 4 vessel probs,
                               conditional on ACS+.
    LVEF<=40%     : 0.5     -- PLACEHOLDER; no operating threshold saved in repo.
    LVEF<50%      : 0.5     -- PLACEHOLDER; no operating threshold saved in repo.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from yacs.config import CfgNode as CN

sys.path.insert(0, "/volume/Open-ECG-Digitizer")
sys.path.insert(0, "/volume/Open-ECG-Digitizer/scripts")
sys.path.insert(0, "/volume/DeepECG_Docker")
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

from eval_layout_classifier import (
    LAYOUT_CLASSES, ROT_CLASSES, LayoutOrientModel, resize_pad,
)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
TARGET_LEN = 2500
PTBXL_POWER_RATIO = 3.003154

ACS_ONNX_DEFAULT = "weights/exported/acs_dual_head.onnx"
WCR_SSL_BASE = "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"
LVEF40_REPO = "heartwise/wcr_lvef_equal_under_40"
LVEF50_REPO = "heartwise/wcr_lvef_under_50"
LVEF_LOCAL_ROOT = "weights"


def undo_rotation(img, rot_label):
    if rot_label > 0:
        img = np.rot90(img, k=(4 - rot_label))
    return np.ascontiguousarray(img)


def classify(img_path, ckpt, backbone, device, crop=512):
    model = LayoutOrientModel(
        num_layouts=len(LAYOUT_CLASSES), num_rotations=len(ROT_CLASSES),
        num_flips=1, backbone=backbone,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    img = cv2.imread(img_path)
    x = resize_pad(img, crop)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        lo, ro, *_ = model(t.to(device))
    lp = torch.softmax(lo, 1)[0].cpu().numpy()
    rp = torch.softmax(ro, 1)[0].cpu().numpy()
    return img, lp, rp


def _inference_cfg(device):
    return {
        "SIGNAL_EXTRACTOR": {"class_path": "src.model.signal_extractor.SignalExtractor", "KWARGS": {}},
        "PERSPECTIVE_DETECTOR": {"class_path": "src.model.perspective_detector.PerspectiveDetector",
                                 "KWARGS": {"num_thetas": 250}},
        "DEWARPER": {"class_path": "src.model.dewarper.Dewarper", "KWARGS": {"abs_peak_threshold": 0.1}},
        "SEGMENTATION_MODEL": {"class_path": "src.model.unet.UNet",
                               "weight_path": "./weights/unet_multilayout/best_weights.pt",
                               "KWARGS": {"num_in_channels": 3, "num_out_channels": 4,
                                          "dims": [32, 64, 128, 256, 320, 320, 320, 320], "depth": 2}},
        "CROPPER": {"class_path": "src.model.cropper.Cropper",
                    "KWARGS": {"granularity": 80, "percentiles": [0.02, 0.98], "alpha": 0.85}},
        "PIXEL_SIZE_FINDER": {"class_path": "src.model.pixel_size_finder.PixelSizeFinder",
                              "KWARGS": {"min_number_of_grid_lines": 30, "max_number_of_grid_lines": 70,
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
        _WRAPPER = InferenceWrapper(
            config=CN(_inference_cfg(device)), device=device,
            resample_size=3000, rotate_on_resample=True,
            enable_timing=False, apply_dewarping=False,
        )
    return _WRAPPER


def digitize(img_path, device, layout_hint=None):
    from torchvision.io import ImageReadMode, read_image
    img = read_image(img_path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1:
        img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4:
        img = img[:3]
    got = _wrapper(device)(img.unsqueeze(0),
                           layout_should_include_substring=layout_hint)
    canonical = got.get("signal", {}).get("canonical_lines")
    layout_used = got.get("layout_name", "Unknown")
    cost = float(got.get("signal", {}).get("layout_matching_cost", float("inf")))
    return canonical, layout_used, cost


def csv_to_ecg(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    leads = []
    for name in LEAD_ORDER:
        v = np.nan_to_num(df[name].values.astype(np.float32), nan=0.0) \
            if name in df.columns else np.zeros(len(df), dtype=np.float32)
        leads.append(v)
    ecg = np.stack(leads, axis=0)
    if ecg.shape[1] != TARGET_LEN:
        ecg = np.stack([resample(l, TARGET_LEN) for l in ecg], axis=0).astype(np.float32)
    return (ecg * PTBXL_POWER_RATIO).astype(np.float32)


def run_acs_dual_head(ecg, onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"ecg_12lead": ecg[None]})
    acs = float(out[0].squeeze())
    vessels = dict(zip(VESSEL_CLASSES, out[1].squeeze().astype(float).tolist()))
    return acs, vessels


def _ensure_lvef_weights(repo_id, local_root):
    """Download HF repo to weights/<name>/ if not already present."""
    local = os.path.join(local_root, repo_id.split("/")[-1])
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "base_ssl.pt")):
        return local
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=local)
    if not os.path.isfile(os.path.join(local, "base_ssl.pt")) and os.path.isfile(WCR_SSL_BASE):
        os.symlink(WCR_SSL_BASE, os.path.join(local, "base_ssl.pt"))
    return local


def _load_wcr_lvef(weights_dir, device):
    from fairseq_signals.utils import checkpoint_utils
    ssl = os.path.join(weights_dir, "base_ssl.pt")
    ft = next((os.path.join(weights_dir, f) for f in os.listdir(weights_dir)
               if f.endswith(".pt") and f != "base_ssl.pt"), None)
    if ft is None:
        raise FileNotFoundError(f"No fine-tuned .pt in {weights_dir}")
    model, _, _ = checkpoint_utils.load_model_and_task(
        ft, arg_overrides={"model_path": ssl}, suffix="",
    )
    return model.to(device).eval(), ft


def run_wcr_lvef(model, ecg, device):
    x = torch.from_numpy(ecg).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(source=x)
        logits = model.get_logits(out) if hasattr(model, "get_logits") else out["out"]
        prob = torch.sigmoid(logits).squeeze().item()
    return float(prob)


def format_row(label, model_file, prob, threshold, flag_yes="POS", flag_no="NEG"):
    if prob is None:
        prob_s, flag = "N/A", "—"
    else:
        prob_s = f"{prob:.4f}"
        flag = flag_yes if prob >= threshold else flag_no
    thr_s = "argmax" if threshold is None else f"{threshold:.3f}"
    return (f"{label:<16} {model_file:<58} {thr_s:>8}  "
            f"{prob_s:>8}  {flag:>6}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layout_ckpt",
                   default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    p.add_argument("--layout_backbone", default="dinov3_b")
    p.add_argument("--acs_onnx", default=ACS_ONNX_DEFAULT)
    p.add_argument("--acs_threshold", type=float, default=0.047,
                   help="Youden-optimal from test set (n=4037)")
    p.add_argument("--lvef40_threshold", type=float, default=0.5,
                   help="PLACEHOLDER; no repo-stored threshold")
    p.add_argument("--lvef50_threshold", type=float, default=0.5,
                   help="PLACEHOLDER; no repo-stored threshold")
    p.add_argument("--skip_lvef", action="store_true",
                   help="Skip LVEF heads (avoids HF download)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Step 1: layout + rotation
    t0 = time.time()
    img, lp, rp = classify(args.image, args.layout_ckpt, args.layout_backbone, device)
    lidx, ridx = int(lp.argmax()), int(rp.argmax())
    layout_name, rot_name = LAYOUT_CLASSES[lidx], ROT_CLASSES[ridx]
    print(f"[1/4] Layout={layout_name} ({lp[lidx]*100:.1f}%) "
          f"Rotation={rot_name} ({rp[ridx]*100:.1f}%)  [{time.time()-t0:.1f}s]")
    corrected = undo_rotation(img, ridx)

    # Step 2: digitize
    t1 = time.time()
    with tempfile.TemporaryDirectory() as td:
        dig_in = os.path.join(td, "in.png")
        cv2.imwrite(dig_in, corrected)
        canonical, layout_used, cost = digitize(dig_in, str(device), layout_hint=layout_name)
    if canonical is None:
        raise RuntimeError("Digitizer produced no canonical signal")
    print(f"[2/4] Digitizer matched '{layout_used}' (cost={cost:.2f})  [{time.time()-t1:.1f}s]")

    import pandas as pd
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    df = pd.DataFrame(data.T, columns=LEAD_ORDER[:data.shape[0]])
    csv_path = os.path.join(args.out_dir, "signals.csv")
    df.to_csv(csv_path, index=False)
    ecg = csv_to_ecg(csv_path)

    # Step 3: ACS + vessel
    t2 = time.time()
    acs_prob, vessel_probs = run_acs_dual_head(ecg, args.acs_onnx)
    top_vessel = max(vessel_probs.items(), key=lambda kv: kv[1])
    print(f"[3/4] ACS={acs_prob:.3f} top_vessel={top_vessel[0]}={top_vessel[1]:.3f}  "
          f"[{time.time()-t2:.1f}s]")

    # Step 4: LVEF
    lvef40 = lvef50 = None
    lvef40_file = lvef50_file = "SKIPPED"
    if not args.skip_lvef:
        t3 = time.time()
        try:
            d40 = _ensure_lvef_weights(LVEF40_REPO, LVEF_LOCAL_ROOT)
            m40, lvef40_file = _load_wcr_lvef(d40, device)
            lvef40 = run_wcr_lvef(m40, ecg, device)
            del m40
            d50 = _ensure_lvef_weights(LVEF50_REPO, LVEF_LOCAL_ROOT)
            m50, lvef50_file = _load_wcr_lvef(d50, device)
            lvef50 = run_wcr_lvef(m50, ecg, device)
            del m50
            print(f"[4/4] LVEF<=40%={lvef40:.3f} LVEF<50%={lvef50:.3f}  "
                  f"[{time.time()-t3:.1f}s]")
        except Exception as e:
            print(f"[4/4] LVEF heads failed: {e}")
    else:
        print("[4/4] LVEF heads skipped (--skip_lvef)")

    # Report
    print()
    print("=" * 110)
    print(f"{'Label':<16} {'Model file':<58} {'Thresh':>8}  {'Prob':>8}  {'Flag':>6}")
    print("-" * 110)
    print(format_row("ACS", os.path.basename(args.acs_onnx),
                     acs_prob, args.acs_threshold, "ACS+", "ACS-"))
    for v, pv in vessel_probs.items():
        tag = "TOP" if v == top_vessel[0] else ""
        print(f"{'vessel:'+v:<16} {os.path.basename(args.acs_onnx):<58} "
              f"{'argmax':>8}  {pv:>8.4f}  {tag:>6}")
    print(format_row("LVEF<=40%", os.path.basename(lvef40_file) if lvef40_file != 'SKIPPED' else 'SKIPPED',
                     lvef40, args.lvef40_threshold, "LOW_EF", "OK"))
    print(format_row("LVEF<50%", os.path.basename(lvef50_file) if lvef50_file != 'SKIPPED' else 'SKIPPED',
                     lvef50, args.lvef50_threshold, "LOW_EF", "OK"))
    print("=" * 110)
    print("Threshold provenance:")
    print(f"  ACS         = {args.acs_threshold} (Youden-optimal, n=4037; "
          "see scripts/predict_folder.py:60)")
    print(f"  Top vessel  = argmax over {VESSEL_CLASSES} (no scalar threshold)")
    print(f"  LVEF<=40%   = {args.lvef40_threshold} (PLACEHOLDER; no repo-stored threshold)")
    print(f"  LVEF<50%    = {args.lvef50_threshold} (PLACEHOLDER; no repo-stored threshold)")

    payload = {
        "input": args.image,
        "layout": {"name": layout_name, "conf": float(lp[lidx])},
        "rotation": {"name": rot_name, "conf": float(rp[ridx])},
        "digitizer": {"layout_used": layout_used, "cost": cost},
        "acs": {"prob": acs_prob, "threshold": args.acs_threshold,
                "flag": "ACS+" if acs_prob >= args.acs_threshold else "ACS-",
                "model_file": args.acs_onnx},
        "vessel": {"probs": vessel_probs, "top": top_vessel[0], "top_prob": top_vessel[1],
                   "rule": "argmax", "model_file": args.acs_onnx},
        "lvef_le_40": {"prob": lvef40, "threshold": args.lvef40_threshold,
                       "model_file": lvef40_file},
        "lvef_lt_50": {"prob": lvef50, "threshold": args.lvef50_threshold,
                       "model_file": lvef50_file},
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nreport.json + signals.csv in {args.out_dir}")


if __name__ == "__main__":
    main()
