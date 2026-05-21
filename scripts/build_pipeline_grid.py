"""Stitch all `annotated.png` files under an out_dir into a single grid image."""
import argparse
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True,
                   help='pipeline_demo_* directory containing per-image subfolders')
    p.add_argument('--grid_path', required=True)
    p.add_argument('--cols', type=int, default=3)
    p.add_argument('--tile_w', type=int, default=700)
    p.add_argument('--tile_h', type=int, default=500)
    p.add_argument('--title', default='')
    args = p.parse_args()

    subs = sorted([d for d in os.listdir(args.out_dir)
                   if os.path.isdir(os.path.join(args.out_dir, d))])
    panels = []
    for sub in subs:
        img_path = os.path.join(args.out_dir, sub, 'annotated.png')
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        scale = min(args.tile_w / w, args.tile_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((args.tile_h, args.tile_w, 3), 240, dtype=np.uint8)
        y_off = (args.tile_h - new_h) // 2
        x_off = (args.tile_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        # Add caption bar with filename
        cap_h = 36
        caption = np.zeros((cap_h, args.tile_w, 3), dtype=np.uint8)
        cv2.putText(caption, sub[:80], (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tile = np.vstack([caption, canvas])
        panels.append(tile)

    if not panels:
        print(f"no panels found in {args.out_dir}")
        return

    # Normalize tile heights to max height (pad with black)
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    rows = []
    for i in range(0, len(padded), args.cols):
        chunk = padded[i:i + args.cols]
        while len(chunk) < args.cols:
            chunk.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)

    if args.title:
        title_h = 60
        title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, args.title, (16, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)
        grid = np.vstack([title_bar, grid])

    os.makedirs(os.path.dirname(args.grid_path) or '.', exist_ok=True)
    cv2.imwrite(args.grid_path, grid)
    print(f"Wrote {grid.shape[1]}x{grid.shape[0]} grid with {len(panels)} panels -> {args.grid_path}")


if __name__ == '__main__':
    main()
