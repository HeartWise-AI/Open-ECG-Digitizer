"""Test-only evaluation for layout classifier checkpoints.

Reproduces the same stratified split (random_state=42) used by
train_layout_classifier.py and evaluates the saved best checkpoint
on the held-out test set. Self-contained to avoid the training
script's heavy import chain (wandb, matplotlib, etc).
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models


LAYOUT_CLASSES = [
    'cabrera_12x1', 'cabrera_6x1_limb', 'precordial_3x2', 'precordial_6x1',
    'standard_12x1', 'standard_3x1', 'standard_3x4', 'standard_3x4_with_r1',
    'standard_3x4_with_r2', 'standard_3x4_with_r3', 'standard_6x1_limb',
    'standard_6x2', 'standard_6x2_with_r1',
]
LAYOUT_TO_IDX = {name: i for i, name in enumerate(LAYOUT_CLASSES)}
ROT_CLASSES = ['rot0', 'rot90', 'rot180', 'rot270']
FLIP_CLASSES = ['no_flip', 'hflip']


def apply_orientation(img, rot_label, flip_label):
    if rot_label > 0:
        img = np.rot90(img, k=rot_label)
    if flip_label:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


def resize_pad(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class LayoutTestDataset(Dataset):
    """Test-time dataset: deterministic per-sample rotation seeded by index
    (matches the random rotation sampling of the training script's test eval)."""

    def __init__(self, image_paths, labels, crop_size=512, rng_seed=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.rng = np.random.default_rng(rng_seed)
        self.rot_labels = self.rng.integers(0, 4, size=len(image_paths)).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.ones((self.crop_size, self.crop_size, 3), dtype=np.uint8) * 255
        label = self.labels[idx]
        rot_label = self.rot_labels[idx]
        flip_label = 0
        img = apply_orientation(img, rot_label, flip_label)
        img = resize_pad(img, self.crop_size)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, label, rot_label, flip_label


class LayoutOrientModel(nn.Module):
    def __init__(self, num_layouts=13, num_rotations=4, num_flips=2,
                 backbone: str = 'resnet18'):
        super().__init__()
        self.backbone_name = backbone
        self.has_flip_head = num_flips > 1
        if backbone == 'resnet18':
            base = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 512
            self.layout_head = nn.Linear(feat_dim, num_layouts)
            self.rot_head = nn.Linear(feat_dim, num_rotations)
            if self.has_flip_head:
                self.flip_head = nn.Linear(feat_dim, num_flips)
        elif backbone.startswith('dino'):
            import timm
            model_name = {
                'dinov2_s': 'vit_small_patch14_dinov2',
                'dinov2_b': 'vit_base_patch14_dinov2',
                'dinov3_s': 'vit_small_patch16_dinov3',
                'dinov3_b': 'vit_base_patch16_dinov3',
                'dinov3_l': 'vit_large_patch16_dinov3',
            }[backbone]
            self.features = timm.create_model(model_name, pretrained=True,
                                              num_classes=0)
            feat_dim = self.features.num_features
            orient_dim = 256
            self.layout_head = nn.Sequential(
                nn.Linear(feat_dim + orient_dim, 256),
                nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, num_layouts),
            )
            orient_base = models.resnet18(weights=None)
            self.orient_cnn = nn.Sequential(
                orient_base.conv1, orient_base.bn1, orient_base.relu,
                orient_base.maxpool,
                orient_base.layer1, orient_base.layer2, orient_base.layer3,
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.rot_head = nn.Linear(orient_dim, num_rotations)
            if self.has_flip_head:
                self.flip_head = nn.Linear(orient_dim, num_flips)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def load_state_dict(self, state_dict, strict: bool = True):
        # Older checkpoints trained with num_flips=1 still contain a vestigial
        # flip_head (single-logit, receives no gradient). Drop those keys when
        # this model instance was built without a flip head.
        if not self.has_flip_head:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('flip_head.')}
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        if self.backbone_name.startswith('dino'):
            dino_feat = self.features(x)
            orient_feat = self.orient_cnn(x)
            combined = torch.cat([dino_feat, orient_feat], dim=1)
            layout = self.layout_head(combined)
            rot = self.rot_head(orient_feat)
            if self.has_flip_head:
                return layout, rot, self.flip_head(orient_feat)
            return layout, rot
        feat = self.features(x).flatten(1)
        layout = self.layout_head(feat)
        rot = self.rot_head(feat)
        if self.has_flip_head:
            return layout, rot, self.flip_head(feat)
        return layout, rot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--backbone', required=True,
                   choices=['resnet18', 'dinov2_s', 'dinov2_b',
                            'dinov3_s', 'dinov3_b', 'dinov3_l'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--crop_size', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_flip', action='store_true')
    p.add_argument('--max_images', type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)

    df = pd.read_csv(args.manifest)
    df = df[df['layout'].isin(LAYOUT_CLASSES)].copy()
    df['label'] = df['layout'].map(LAYOUT_TO_IDX)
    if args.max_images and args.max_images < len(df):
        df = df.sample(args.max_images, random_state=42)

    _, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df['label'])
    _, test_df = train_test_split(test_df, test_size=0.5, random_state=42,
                                   stratify=test_df['label'])
    print(f"Test set size: {len(test_df)}")

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
    print(f"Loaded {args.backbone} ({n_params:.1f}M params) from {args.ckpt}")

    test_ds = LayoutTestDataset(
        test_df['png_path'].tolist(), test_df['label'].tolist(),
        crop_size=args.crop_size,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_layout_p, all_layout_y = [], []
    all_rot_p, all_rot_y = [], []
    all_flip_p, all_flip_y = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, y_layout, y_rot, y_flip in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
            lo, ro = out[0], out[1]
            fo = out[2] if len(out) == 3 else None
            all_layout_p.extend(lo.argmax(1).cpu().tolist())
            all_layout_y.extend(y_layout.tolist())
            all_rot_p.extend(ro.argmax(1).cpu().tolist())
            all_rot_y.extend(y_rot.tolist())
            if fo is not None:
                all_flip_p.extend(fo.argmax(1).cpu().tolist())
                all_flip_y.extend(y_flip.tolist())
    print(f"Inference time: {time.time() - t0:.1f}s")

    layout_acc = accuracy_score(all_layout_y, all_layout_p) * 100
    print("\n--- Layout Classification (Test) ---")
    print(f"Test Accuracy: {layout_acc:.3f}%")
    print(classification_report(
        all_layout_y, all_layout_p,
        labels=list(range(len(LAYOUT_CLASSES))),
        target_names=LAYOUT_CLASSES, digits=3, zero_division=0))

    rot_acc = accuracy_score(all_rot_y, all_rot_p) * 100
    print("\n--- Rotation Classification (Test) ---")
    print(f"Test Accuracy: {rot_acc:.3f}%")
    print(classification_report(
        all_rot_y, all_rot_p,
        labels=list(range(len(ROT_CLASSES))),
        target_names=ROT_CLASSES, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_rot_y, all_rot_p,
                           labels=list(range(len(ROT_CLASSES)))))

    if args.no_flip:
        print("\n--- Flip Classification (Test) --- skipped (--no_flip)")
    else:
        flip_acc = accuracy_score(all_flip_y, all_flip_p) * 100
        print("\n--- Flip Classification (Test) ---")
        print(f"Test Accuracy: {flip_acc:.3f}%")
        print(classification_report(
            all_flip_y, all_flip_p,
            labels=list(range(len(FLIP_CLASSES))),
            target_names=FLIP_CLASSES, digits=3, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(all_flip_y, all_flip_p,
                               labels=list(range(len(FLIP_CLASSES)))))


if __name__ == '__main__':
    main()
