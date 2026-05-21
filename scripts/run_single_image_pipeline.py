"""End-to-end single-image pipeline:
    1. Layout + rotation classification (dinov3_b)
    2. Rotation correction (undo CCW rotation)
    3. Digitization via multi-layout U-Net → 12-lead CSV
    4. ACS + culprit-vessel localization (acs_dual_head.onnx)

Writes a per-image output folder with:
    original.png, corrected.png, annotated.png, prediction.json,
    signals.csv (12-lead digitized), plots/*.png (digitizer diagnostics).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import yaml
from scipy.signal import resample
from yacs.config import CfgNode as CN

sys.path.insert(0, '/volume/Open-ECG-Digitizer')
sys.path.insert(0, '/volume/Open-ECG-Digitizer/scripts')

from eval_layout_classifier import (
    LAYOUT_CLASSES, ROT_CLASSES, LayoutOrientModel, resize_pad,
)

VESSEL_CLASSES = ['LAD', 'RCA', 'LCX', 'Left_Main']
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PTBXL_POWER_RATIO = 3.003154
TARGET_LENGTH = 2500


def undo_rotation(img, rot_label):
    if rot_label > 0:
        img = np.rot90(img, k=(4 - rot_label))
    return np.ascontiguousarray(img)


def classify(img_path, ckpt, backbone, device, crop_size=512):
    model = LayoutOrientModel(
        num_layouts=len(LAYOUT_CLASSES),
        num_rotations=len(ROT_CLASSES),
        num_flips=1, backbone=backbone,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    img = cv2.imread(img_path)
    x = resize_pad(img, crop_size)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = t.to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(t)
    lo, ro = out[0], out[1]
    layout_probs = torch.softmax(lo, dim=1)[0].cpu().numpy()
    rot_probs = torch.softmax(ro, dim=1)[0].cpu().numpy()
    return img, layout_probs, rot_probs


_INFERENCE_WRAPPER = None


def _inference_config(device):
    return {
        'SIGNAL_EXTRACTOR': {
            'class_path': 'src.model.signal_extractor.SignalExtractor',
            'KWARGS': {}},
        'PERSPECTIVE_DETECTOR': {
            'class_path': 'src.model.perspective_detector.PerspectiveDetector',
            'KWARGS': {'num_thetas': 250}},
        'DEWARPER': {
            'class_path': 'src.model.dewarper.Dewarper',
            'KWARGS': {'abs_peak_threshold': 0.1}},
        'SEGMENTATION_MODEL': {
            'class_path': 'src.model.unet.UNet',
            'weight_path': './weights/unet_multilayout/best_weights.pt',
            'KWARGS': {'num_in_channels': 3, 'num_out_channels': 4,
                       'dims': [32, 64, 128, 256, 320, 320, 320, 320],
                       'depth': 2}},
        'CROPPER': {
            'class_path': 'src.model.cropper.Cropper',
            'KWARGS': {'granularity': 80, 'percentiles': [0.02, 0.98],
                       'alpha': 0.85}},
        'PIXEL_SIZE_FINDER': {
            'class_path': 'src.model.pixel_size_finder.PixelSizeFinder',
            'KWARGS': {'min_number_of_grid_lines': 30,
                       'max_number_of_grid_lines': 70,
                       'lower_grid_line_factor': 0.3}},
        'LAYOUT_IDENTIFIER': {
            'class_path': 'src.model.lead_identifier.LeadIdentifier',
            'config_path': 'src/config/lead_layouts_all.yml',
            'unet_config_path': 'src/config/lead_name_unet.yml',
            'unet_weight_path': './weights/lead_name_unet_weights_07072025.pt',
            'KWARGS': {'debug': False, 'device': device,
                       'possibly_flipped': False}},
    }


def _load_inference_wrapper(device):
    global _INFERENCE_WRAPPER
    if _INFERENCE_WRAPPER is None:
        from src.model.inference_wrapper import InferenceWrapper
        cfg_node = CN(_inference_config(device))
        _INFERENCE_WRAPPER = InferenceWrapper(
            config=cfg_node,
            device=device,
            resample_size=3000,
            rotate_on_resample=True,
            enable_timing=False,
            apply_dewarping=False,
        )
    return _INFERENCE_WRAPPER


def digitize_image(image_path, device, layout_hint=None):
    """Run the digitizer on a single image, optionally filtering candidate
    layouts by substring. Returns (canonical_signal, layout_name_used, cost).
    """
    from torchvision.io import ImageReadMode, read_image
    img = read_image(image_path, mode=ImageReadMode.RGB)
    if img.shape[0] == 1:
        img = img.expand(3, *img.shape[1:])
    elif img.shape[0] == 4:
        img = img[:3]
    img = img.unsqueeze(0)
    wrapper = _load_inference_wrapper(device)
    got = wrapper(img, layout_should_include_substring=layout_hint)
    canonical = got.get('signal', {}).get('canonical_lines')
    layout_used = got.get('layout_name', 'Unknown')
    cost = got.get('signal', {}).get('layout_matching_cost', float('inf'))
    return canonical, layout_used, float(cost), got


def run_acs_dual_head(csv_path, onnx_path):
    import pandas as pd
    import onnxruntime as ort
    df = pd.read_csv(csv_path)
    leads = []
    for name in LEAD_ORDER:
        if name in df.columns:
            v = np.nan_to_num(df[name].values.astype(np.float32), nan=0.0)
        else:
            v = np.zeros(len(df), dtype=np.float32)
        leads.append(v)
    ecg = np.stack(leads, axis=0)  # (12, N)
    if ecg.shape[1] != TARGET_LENGTH:
        ecg = np.stack([resample(l, TARGET_LENGTH) for l in ecg], axis=0).astype(np.float32)
    ecg = ecg * PTBXL_POWER_RATIO
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'ecg_12lead': ecg[None].astype(np.float32)})
    acs_prob = float(out[0].squeeze())
    vessel_probs = out[1].squeeze().astype(float).tolist()
    return acs_prob, dict(zip(VESSEL_CLASSES, vessel_probs)), ecg


def annotate(img, lines):
    out = img.copy()
    h, w = out.shape[:2]
    pad, line_h = 14, 36
    box_h = pad * 2 + line_h * len(lines)
    box = out[:box_h, :].copy()
    out[:box_h, :] = cv2.addWeighted(box, 0.3, np.zeros_like(box), 0.7, 0)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (pad, pad + line_h * (i + 1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def plot_signals(ecg, out_path, fs=250):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(ecg.shape[1]) / fs
    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    for i, (lead, ax) in enumerate(zip(LEAD_ORDER, axes)):
        ax.plot(t, ecg[i], linewidth=0.8, color='black')
        ax.set_ylabel(lead, rotation=0, ha='right', va='center', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t[-1])
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Digitized 12-lead ECG', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--layout_ckpt',
                   default='weights/layout_classifier_dinov3b_v5/best_layout_classifier.pt')
    p.add_argument('--layout_backbone', default='dinov3_b')
    p.add_argument('--acs_onnx',
                   default='weights/exported/acs_dual_head.onnx')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # -- Step 1: Layout + rotation
    t0 = time.time()
    img, layout_probs, rot_probs = classify(
        args.image, args.layout_ckpt, args.layout_backbone, device)
    layout_idx = int(layout_probs.argmax())
    rot_idx = int(rot_probs.argmax())
    layout_name = LAYOUT_CLASSES[layout_idx]
    rot_name = ROT_CLASSES[rot_idx]
    layout_conf = float(layout_probs[layout_idx])
    rot_conf = float(rot_probs[rot_idx])
    print(f"[1/3] Layout={layout_name} ({layout_conf*100:.1f}%), "
          f"Rotation={rot_name} ({rot_conf*100:.1f}%)  [{time.time()-t0:.1f}s]")

    # -- Step 2: rotation correction
    corrected = undo_rotation(img, rot_idx)
    cv2.imwrite(os.path.join(args.out_dir, 'original.png'),
                cv2.resize(img, (1200, int(img.shape[0] * 1200 / img.shape[1])))
                if img.shape[1] > 1200 else img)
    cv2.imwrite(os.path.join(args.out_dir, 'corrected.png'),
                cv2.resize(corrected, (1400, int(corrected.shape[0] * 1400 / corrected.shape[1])))
                if corrected.shape[1] > 1400 else corrected)

    # -- Step 3: Digitize corrected image, passing layout hint from classifier
    t1 = time.time()
    with tempfile.TemporaryDirectory() as td:
        dig_input = os.path.join(td, 'input.png')
        cv2.imwrite(dig_input, corrected)
        device_str = str(device)
        canonical, layout_used, cost, _got = digitize_image(
            dig_input, device_str, layout_hint=layout_name,
        )
    print(f"[2/3] Digitizer matched layout='{layout_used}' "
          f"(cost={cost:.2f}, hint='{layout_name}')  [{time.time()-t1:.1f}s]")

    if canonical is None:
        raise RuntimeError("Digitizer produced no canonical signal — check U-Net output")

    # Write signals.csv in the same format as src.digitize.save_timeseries_csv
    import pandas as pd
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    n_leads = data.shape[0]
    col_names = LEAD_ORDER[:n_leads]
    df_csv = pd.DataFrame(data.T, columns=col_names)
    df_csv.to_csv(os.path.join(args.out_dir, 'signals.csv'), index=False)

    # -- Step 4: ACS + vessel localization
    t2 = time.time()
    signal_csv = os.path.join(args.out_dir, 'signals.csv')
    acs_prob, vessel_probs, ecg_12lead = run_acs_dual_head(
        signal_csv, args.acs_onnx)
    top_vessel = max(vessel_probs.items(), key=lambda kv: kv[1])
    print(f"[3/3] ACS={acs_prob:.3f} | "
          f"top vessel: {top_vessel[0]}={top_vessel[1]:.3f}  "
          f"[{time.time()-t2:.1f}s]")

    # signals plot
    plot_signals(ecg_12lead, os.path.join(args.out_dir, 'ecg_12lead.png'))

    # annotated corrected image
    vessel_str = ", ".join(f"{k}:{v*100:.0f}%" for k, v in vessel_probs.items())
    lines = [
        f"Layout: {layout_name} ({layout_conf*100:.1f}%)",
        f"Rotation: {rot_name} -> corrected ({rot_conf*100:.1f}%)",
        f"ACS probability: {acs_prob*100:.1f}%",
        f"Culprit vessel (top): {top_vessel[0]} {top_vessel[1]*100:.1f}%",
        f"All vessels: {vessel_str}",
    ]
    disp = cv2.imread(os.path.join(args.out_dir, 'corrected.png'))
    cv2.imwrite(os.path.join(args.out_dir, 'annotated.png'),
                annotate(disp, lines))

    # JSON
    prediction = {
        'input': args.image,
        'layout': {
            'prediction': layout_name,
            'confidence': layout_conf,
            'probs': {LAYOUT_CLASSES[i]: float(layout_probs[i])
                      for i in range(len(LAYOUT_CLASSES))},
        },
        'rotation': {
            'prediction': rot_name,
            'confidence': rot_conf,
            'probs': {ROT_CLASSES[i]: float(rot_probs[i])
                      for i in range(len(ROT_CLASSES))},
        },
        'acs': {
            'probability': acs_prob,
        },
        'vessel_localization': {
            'probabilities': vessel_probs,
            'top_vessel': top_vessel[0],
            'top_vessel_prob': top_vessel[1],
        },
    }
    with open(os.path.join(args.out_dir, 'prediction.json'), 'w') as fp:
        json.dump(prediction, fp, indent=2)
    print(f"\nAll outputs in {args.out_dir}")


if __name__ == '__main__':
    main()
