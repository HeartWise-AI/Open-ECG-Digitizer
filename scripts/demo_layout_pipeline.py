"""End-to-end inference demo: for each image in a folder, predict layout +
rotation, undo the rotation, and write a per-image output folder containing
the original, the corrected image, and a prediction JSON.

Output layout:
  out_dir/
    <image_stem>/
      original.png
      corrected.png      # rotation undone using predicted rot label
      annotated.png      # corrected + text overlay with predicted layout/rot
      prediction.json
    summary.csv          # one row per image
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import torch

from eval_layout_classifier import (
    FLIP_CLASSES, LAYOUT_CLASSES, ROT_CLASSES,
    LayoutOrientModel, resize_pad,
)


def undo_rotation(img, rot_label):
    """Undo a CCW rotation of k*90° by rotating back (4-k)*90°."""
    if rot_label > 0:
        img = np.rot90(img, k=(4 - rot_label))
    return np.ascontiguousarray(img)


def annotate(img, text_lines):
    out = img.copy()
    h, w = out.shape[:2]
    pad = 12
    line_h = 32
    box_h = pad * 2 + line_h * len(text_lines)
    box = out[:box_h, :].copy()
    overlay = np.zeros_like(box)
    out[:box_h, :] = cv2.addWeighted(box, 0.35, overlay, 0.65, 0)
    for i, line in enumerate(text_lines):
        cv2.putText(out, line, (pad, pad + line_h * (i + 1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--backbone', required=True,
                   choices=['resnet18', 'dinov2_s', 'dinov2_b',
                            'dinov3_s', 'dinov3_b', 'dinov3_l'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--crop_size', type=int, default=512)
    p.add_argument('--no_flip', action='store_true')
    p.add_argument('--ext', default='png,jpg,jpeg')
    p.add_argument('--max_out_size', type=int, default=1200,
                   help='Max side length for saved images (downscaled for speed)')
    args = p.parse_args()

    device = torch.device(args.device)
    exts = [e.strip().lower() for e in args.ext.split(',')]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.folder, f'*.{ext}')))
        files.extend(glob.glob(os.path.join(args.folder, f'*.{ext.upper()}')))
    files = sorted(set(files))
    print(f"Found {len(files)} images in {args.folder}")
    if not files:
        return

    num_flips = 1 if args.no_flip else len(FLIP_CLASSES)
    model = LayoutOrientModel(
        num_layouts=len(LAYOUT_CLASSES),
        num_rotations=len(ROT_CLASSES),
        num_flips=num_flips,
        backbone=args.backbone,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
    print(f"Loaded {args.backbone} ({n_params:.1f}M params)\n")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    with torch.no_grad():
        for f in files:
            img = cv2.imread(f)
            if img is None:
                print(f"SKIP (unreadable): {os.path.basename(f)}")
                continue

            # Model input: resize-pad to crop_size
            x = resize_pad(img, args.crop_size)
            t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.amp.autocast('cuda'):
                out = model(t)
            lo, ro = out[0], out[1]
            layout_probs = torch.softmax(lo, dim=1)[0].cpu().numpy()
            rot_probs = torch.softmax(ro, dim=1)[0].cpu().numpy()
            layout_idx = int(layout_probs.argmax())
            rot_idx = int(rot_probs.argmax())
            layout_name = LAYOUT_CLASSES[layout_idx]
            rot_name = ROT_CLASSES[rot_idx]
            layout_conf = float(layout_probs[layout_idx])
            rot_conf = float(rot_probs[rot_idx])

            # Downscale original for output (keep aspect ratio)
            h, w = img.shape[:2]
            scale = args.max_out_size / max(h, w) if max(h, w) > args.max_out_size else 1.0
            disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
            corrected = undo_rotation(disp, rot_idx)
            top3 = sorted(
                [(LAYOUT_CLASSES[i], float(layout_probs[i]))
                 for i in range(len(LAYOUT_CLASSES))],
                key=lambda p: -p[1])[:3]
            rot_line = " ".join(
                f"{ROT_CLASSES[i]}:{rot_probs[i]*100:.0f}%"
                for i in range(len(ROT_CLASSES))
            )
            annotated = annotate(corrected, [
                f"pred layout: {layout_name}  ({layout_conf*100:.1f}%)",
                f"top3: {top3[0][0]} {top3[0][1]*100:.1f}%, "
                f"{top3[1][0]} {top3[1][1]*100:.1f}%, "
                f"{top3[2][0]} {top3[2][1]*100:.1f}%",
                f"pred rotation: {rot_name} (corrected)  [{rot_line}]",
            ])

            stem = os.path.splitext(os.path.basename(f))[0]
            sub = os.path.join(args.out_dir, stem)
            os.makedirs(sub, exist_ok=True)
            cv2.imwrite(os.path.join(sub, 'original.png'), disp)
            cv2.imwrite(os.path.join(sub, 'corrected.png'), corrected)
            cv2.imwrite(os.path.join(sub, 'annotated.png'), annotated)
            with open(os.path.join(sub, 'prediction.json'), 'w') as fp:
                json.dump({
                    'input': f,
                    'layout': layout_name,
                    'layout_conf': layout_conf,
                    'layout_top3': top3,
                    'layout_probs': {LAYOUT_CLASSES[i]: float(layout_probs[i])
                                     for i in range(len(LAYOUT_CLASSES))},
                    'rotation': rot_name,
                    'rotation_conf': rot_conf,
                    'rotation_probs': {ROT_CLASSES[i]: float(rot_probs[i])
                                       for i in range(len(ROT_CLASSES))},
                }, fp, indent=2)

            summary_rows.append({
                'file': os.path.basename(f),
                'layout': layout_name,
                'layout_conf': f"{layout_conf:.3f}",
                'top2_layout': top3[1][0],
                'top2_conf': f"{top3[1][1]:.3f}",
                'top3_layout': top3[2][0],
                'top3_conf': f"{top3[2][1]:.3f}",
                'rotation': rot_name,
                'rotation_conf': f"{rot_conf:.3f}",
                'rot0_p': f"{rot_probs[0]:.3f}",
                'rot90_p': f"{rot_probs[1]:.3f}",
                'rot180_p': f"{rot_probs[2]:.3f}",
                'rot270_p': f"{rot_probs[3]:.3f}",
            })
            print(f"{os.path.basename(f):<35} -> {layout_name:<22} "
                  f"({layout_conf*100:.1f}%) | {rot_name} ({rot_conf*100:.1f}%) | "
                  f"top3: {top3[0][0]}/{top3[1][0]}/{top3[2][0]}")

    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} per-image folders + summary.csv under {args.out_dir}")


if __name__ == '__main__':
    main()
