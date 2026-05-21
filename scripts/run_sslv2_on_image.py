"""Run SSL v2 ONNX + old dual-head ACS ONNX on a single ECG image.

Uses empirical mV calibration factor 0.001 (digitizer canonical * 0.001 → mV),
derived from 50 MHI samples vs raw .npy*0.00488.

Caveats:
    - canonical_lines is 12 leads x 3000, but each non-rhythm lead only has
      ~2.5 s of real data (1 column chunk); the rest are NaN → 0. SSL v2 was
      trained on full 10 s recordings, so probs should be read with this
      distribution shift in mind.
    - Notion v2 Youden thresholds are applied for ACS-relevant labels, assuming
      the model's 77-label output order matches DeepECG-Docker ECG_PATTERNS.
      (If label order differs, the threshold flags will be wrong — flag visually.)
"""
import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import resample
from yacs.config import CfgNode as CN

sys.path.insert(0, "/volume/Open-ECG-Digitizer")
sys.path.insert(0, "/volume/Open-ECG-Digitizer/scripts")

from eval_layout_classifier import (LAYOUT_CLASSES, ROT_CLASSES,
                                    LayoutOrientModel, resize_pad)

# 77-label list copied from /volume/DeepECG_Docker/utils/constants.py:ECG_PATTERNS
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

# Permutation used by DeepECG-Docker to reorder raw WCR output → ECG_PATTERNS:
# permuted[i] = raw[WCR_COLUMN_CONVERSION[i]]
# (From /volume/DeepECG_Docker/utils/analysis_pipeline.py:393 and constants.py:477)
WCR_COLUMN_CONVERSION = [
    15, 23, 16, 1, 57, 63, 73, 41, 39, 36, 2, 29, 30, 65, 34, 12, 55, 56, 21, 8,
    42, 71, 37, 50, 13, 38, 46, 24, 49, 9, 66, 26, 40, 4, 22, 0, 11, 74, 64, 7,
    76, 58, 33, 70, 17, 6, 28, 69, 44, 61, 32, 72, 45, 25, 75, 18, 14, 5, 3, 31,
    27, 67, 62, 10, 43, 51, 52, 47, 19, 68, 53, 48, 60, 20, 59, 54, 35,
]
assert sorted(WCR_COLUMN_CONVERSION) == list(range(77))

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
TARGET_LEN = 2500
DIG_TO_MV = 0.001  # empirical from 50 MHI calibration

# Youden thresholds from Notion DeepECG-SSL v2 (2026-03-16), ACS-relevant subset
V2_YOUDEN_ACS = {
    "Acute MI": 0.0064,
    "ST elevation (anterior - V3-V4)": 0.0009,
    "ST elevation (septal - V1-V2)": 0.0063,
    "ST elevation (lateral - I, aVL, V5-V6)": 0.0328,
    "ST elevation (inferior - II, III, aVF)": 0.0101,
    "ST elevation (posterior - V7-V8-V9)": 0.0001,
    "Q wave (septal- V1-V2)": 0.0071,
    "Q wave (anterior - V3-V4)": 0.0046,
    "Q wave (inferior - II, III, aVF)": 0.0534,
    "Q wave (lateral- I, aVL, V5-V6)": 0.0003,
    "Q wave (posterior - V7-V9)": 0.0034,  # placeholder, n/a in Notion
    "ST depression (anterior - V3-V4)": 0.0210,
    "ST depression (inferior - II, III, aVF)": 0.0118,
    "ST depression (lateral - I, avL, V5-V6)": 0.0419,
    "ST depression (septal- V1-V2)": 0.0493,
    "T wave inversion (anterior - V3-V4)": 0.0083,
    "T wave inversion (inferior - II, III, aVF)": 0.0075,
    "T wave inversion (lateral -I, aVL, V5-V6)": 0.0003,
    "Acute pericarditis": 0.0120,
    "Early repolarization": 0.0030,
}


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


def classify(ckpt, backbone, img, device):
    m = LayoutOrientModel(num_layouts=len(LAYOUT_CLASSES),
                          num_rotations=len(ROT_CLASSES),
                          num_flips=1, backbone=backbone).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    x = resize_pad(img, 512)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        lo, ro, *_ = m(t.to(device))
    return int(torch.softmax(lo, 1).argmax()), int(torch.softmax(ro, 1).argmax())


def digitize(wrapper, path, hint):
    from torchvision.io import ImageReadMode, read_image
    img = read_image(path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1: img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4: img = img[:3]
    got = wrapper(img.unsqueeze(0), layout_should_include_substring=hint)
    return (got.get("signal", {}).get("canonical_lines"),
            got.get("layout_name", "Unknown"),
            float(got.get("signal", {}).get("layout_matching_cost", float("inf"))))


def canon_to_mv_2500(canon):
    """12-lead canonical → mV-scaled, NaN→0, resampled to 2500."""
    data = canon.squeeze().cpu().numpy()
    if data.ndim == 1: data = data[None, :]
    if data.shape[0] < 12:
        data = np.concatenate([data,
                               np.full((12 - data.shape[0], data.shape[1]),
                                       np.nan, dtype=data.dtype)], axis=0)
    data = np.nan_to_num(data, nan=0.0)
    data = data * DIG_TO_MV
    if data.shape[1] != TARGET_LEN:
        data = np.stack([resample(l, TARGET_LEN) for l in data]).astype(np.float32)
    return data.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sslv2_onnx",
                    default="/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.onnx")
    ap.add_argument("--dualhead_onnx",
                    default="weights/exported/acs_dual_head.onnx")
    ap.add_argument("--layout_ckpt",
                    default="weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt")
    ap.add_argument("--layout_backbone", default="dinov3_b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    img = cv2.imread(args.image)
    lidx, ridx = classify(args.layout_ckpt, args.layout_backbone, img, device)
    print(f"Layout={LAYOUT_CLASSES[lidx]}  Rotation={ROT_CLASSES[ridx]}")
    corrected = undo_rot(img, ridx)

    from src.model.inference_wrapper import InferenceWrapper
    wrapper = InferenceWrapper(config=CN(_inference_cfg(str(device))), device=str(device),
                               resample_size=3000, rotate_on_resample=True,
                               enable_timing=False, apply_dewarping=False)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "in.png")
        cv2.imwrite(p, corrected)
        canon, layout_used, cost = digitize(wrapper, p, LAYOUT_CLASSES[lidx])
    print(f"Digitizer: layout={layout_used}  cost={cost:.2f}")
    if canon is None:
        raise RuntimeError("No canonical signal")

    ecg_mv = canon_to_mv_2500(canon)  # (12, 2500), mV
    print(f"ECG shape {ecg_mv.shape}  range mV: "
          f"[{ecg_mv.min():.3f}, {ecg_mv.max():.3f}]  "
          f"P95|x|={np.percentile(np.abs(ecg_mv), 95):.3f}")

    # SSL v2
    sslv2 = ort.InferenceSession(args.sslv2_onnx, providers=["CPUExecutionProvider"])
    logits_v2 = sslv2.run(None, {"source": ecg_mv[None]})[0]
    probs_v2 = 1.0 / (1.0 + np.exp(-logits_v2.squeeze()))
    # Dual-head (uses same mV-scaled input; original path used PTBXL_POWER_RATIO,
    # so this may differ from earlier runs)
    dh = ort.InferenceSession(args.dualhead_onnx, providers=["CPUExecutionProvider"])
    dh_out = dh.run(None, {"ecg_12lead": ecg_mv[None]})
    acs_dh = float(dh_out[0].squeeze())
    vessel_dh = dict(zip(VESSEL_CLASSES, dh_out[1].squeeze().astype(float).tolist()))
    top_dh = max(vessel_dh.items(), key=lambda kv: kv[1])

    # Permute raw model output to ECG_PATTERNS order
    assert len(probs_v2) == 77, f"unexpected SSL v2 output len {len(probs_v2)}"
    probs_v2_permuted = probs_v2[WCR_COLUMN_CONVERSION]
    label_prob = dict(zip(ECG_PATTERNS, probs_v2_permuted.tolist()))

    # ACS-relevant report
    print("\n" + "=" * 88)
    print(f"{'Label':<48} {'Prob':>8}  {'Thresh':>7}  Flag")
    print("-" * 88)
    for lbl, thr in V2_YOUDEN_ACS.items():
        p = label_prob.get(lbl)
        if p is None:
            print(f"{lbl:<48} {'N/A':>8}  {thr:>7.4f}  (label not in ECG_PATTERNS)")
            continue
        flag = "POS" if p >= thr else ""
        print(f"{lbl:<48} {p:>8.4f}  {thr:>7.4f}  {flag}")
    print()
    print(f"Dual-head ACS (old ONNX, mV-scaled input): {acs_dh:.4f}  "
          f"Top vessel: {top_dh[0]}={top_dh[1]:.3f}")
    print("=" * 88)

    # Dump top-20 v2 predictions overall
    ranked = sorted(label_prob.items(), key=lambda kv: -kv[1])[:20]
    print("\nTop 20 SSL v2 probabilities (all 77):")
    for lbl, p in ranked:
        thr = V2_YOUDEN_ACS.get(lbl)
        tag = f" (>= youden {thr})" if (thr is not None and p >= thr) else ""
        print(f"  {lbl:<55} {p:.4f}{tag}")

    # JSON
    out = {
        "input": args.image,
        "layout": LAYOUT_CLASSES[lidx],
        "rotation": ROT_CLASSES[ridx],
        "digitizer": {"layout_used": layout_used, "cost": cost},
        "mv_scale_used": DIG_TO_MV,
        "sslv2_probs": {k: float(v) for k, v in label_prob.items()},
        "sslv2_acs_flags": {lbl: {"prob": float(label_prob.get(lbl, float('nan'))),
                                  "threshold": thr,
                                  "positive": bool(label_prob.get(lbl, 0) >= thr)
                                             if lbl in label_prob else None}
                            for lbl, thr in V2_YOUDEN_ACS.items()},
        "dualhead_acs": acs_dh,
        "dualhead_vessels": vessel_dh,
        "dualhead_top_vessel": top_dh[0],
    }
    with open(os.path.join(args.out_dir, "report_sslv2.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    np.save(os.path.join(args.out_dir, "ecg_mv.npy"), ecg_mv)
    print(f"\nSaved: {args.out_dir}/report_sslv2.json  and ecg_mv.npy")


if __name__ == "__main__":
    main()
