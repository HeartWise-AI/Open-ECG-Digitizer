"""Evaluate layout classifier on a folder of phone-style ECG images.

Expects filenames of the form: <prefix>_<layout>_<rest>.png where <layout>
matches one of LAYOUT_CLASSES. Runs layout + rotation prediction (no ground
truth rotation — just prints predicted rotation).
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch

from eval_layout_classifier import (
    FLIP_CLASSES, LAYOUT_CLASSES, ROT_CLASSES,
    LayoutOrientModel, resize_pad,
)


def extract_layout(fname: str):
    stem = os.path.splitext(os.path.basename(fname))[0]
    # Longest-match so that 'standard_3x4_with_r1' isn't truncated to 'standard_3x4'
    for layout in sorted(LAYOUT_CLASSES, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(layout)}(_|$)", stem):
            return layout
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--backbone', required=True,
                   choices=['resnet18', 'dinov2_s', 'dinov2_b',
                            'dinov3_s', 'dinov3_b', 'dinov3_l'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--crop_size', type=int, default=512)
    p.add_argument('--no_flip', action='store_true')
    p.add_argument('--ext', default='png,jpg,jpeg')
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
    print(f"Loaded {args.backbone} ({n_params:.1f}M params) from {args.ckpt}\n")

    correct = 0
    labeled = 0
    per_class = {}
    rows = []
    with torch.no_grad():
        for f in files:
            gt = extract_layout(f)
            img = cv2.imread(f)
            if img is None:
                print(f"SKIP (unreadable): {os.path.basename(f)}")
                continue
            img = resize_pad(img, args.crop_size)
            t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.amp.autocast('cuda'):
                out = model(t)
            lo, ro = out[0], out[1]
            fo = out[2] if len(out) == 3 else None
            layout_pred = LAYOUT_CLASSES[lo.argmax(1).item()]
            rot_pred = ROT_CLASSES[ro.argmax(1).item()]
            flip_pred = '-' if fo is None else FLIP_CLASSES[fo.argmax(1).item()]
            ok = ' ' if gt is None else ('✓' if gt == layout_pred else '✗')
            if gt is not None:
                labeled += 1
                per_class.setdefault(gt, {'n': 0, 'ok': 0})
                per_class[gt]['n'] += 1
                if gt == layout_pred:
                    correct += 1
                    per_class[gt]['ok'] += 1
            rows.append((os.path.basename(f), gt, layout_pred, rot_pred, flip_pred, ok))

    name_w = max(len(r[0]) for r in rows)
    print(f"{'file':<{name_w}}  {'gt layout':<22} {'pred layout':<22} {'rot':<7} {'flip':<8} ok")
    print('-' * (name_w + 70))
    for fname, gt, lp, rp, fp, ok in rows:
        gts = gt or '(unknown)'
        print(f"{fname:<{name_w}}  {gts:<22} {lp:<22} {rp:<7} {fp:<8} {ok}")

    if labeled:
        acc = 100 * correct / labeled
        print(f"\nLayout accuracy on {labeled} labeled images: {correct}/{labeled} = {acc:.1f}%")
        print("\nPer-class:")
        for c in sorted(per_class):
            s = per_class[c]
            print(f"  {c:<22} {s['ok']}/{s['n']}")


if __name__ == '__main__':
    main()
