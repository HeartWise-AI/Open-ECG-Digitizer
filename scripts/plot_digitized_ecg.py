"""Plot a digitized 12-lead CSV as a clinical-style ECG figure.

Produces two figures:
    ecg_stacked.png   - 12 leads stacked vertically, 1 column
    ecg_3x4.png       - 3x4 grid with rhythm strip (clinical paper layout)
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Traditional clinical 3x4 display order (by column)
COL_LAYOUT = [
    ['I',  'aVR', 'V1', 'V4'],
    ['II', 'aVL', 'V2', 'V5'],
    ['III','aVF', 'V3', 'V6'],
]


def add_ecg_grid(ax, mm_per_s=25, mv_per_mm=0.1, seconds=2.5):
    """Draw standard ECG grid (small=1mm, large=5mm), red on white."""
    ax.set_facecolor('#fffafa')
    # minor = 0.04 s = 0.1 mV ; major = 0.2 s = 0.5 mV
    for x in np.arange(0, seconds + 1e-6, 0.04):
        ax.axvline(x, color='#ffcccc', linewidth=0.4, zorder=0)
    for x in np.arange(0, seconds + 1e-6, 0.2):
        ax.axvline(x, color='#ff8888', linewidth=0.7, zorder=0)
    y_range = 2.0  # mV total range per lead box
    for y in np.arange(-y_range, y_range + 1e-6, 0.1):
        ax.axhline(y, color='#ffcccc', linewidth=0.4, zorder=0)
    for y in np.arange(-y_range, y_range + 1e-6, 0.5):
        ax.axhline(y, color='#ff8888', linewidth=0.7, zorder=0)


def load_signals(csv_path):
    df = pd.read_csv(csv_path)
    leads = {}
    for name in LEAD_ORDER:
        if name in df.columns:
            v = df[name].values.astype(float)
        else:
            v = np.full(len(df), np.nan)
        leads[name] = v
    # Guess sample rate from length: digitizer resamples to 3000 by default,
    # representing 10 seconds at 300 Hz.
    n = len(df)
    # heuristic: if length close to multiples of 250*T
    fs = 300 if n >= 1500 else 250
    t = np.arange(n) / fs
    return leads, t, fs


def plot_stacked(leads, t, fs, out_path, title):
    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    for i, name in enumerate(LEAD_ORDER):
        ax = axes[i]
        v = leads[name]
        # Convert to mV (digitizer output is ~ -voltage*1000? normalize by max)
        ax.plot(t, v, linewidth=0.9, color='black')
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t[-1])
        if i < 11:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_clinical_3x4(leads, t, fs, out_path, title, rhythm_lead='II'):
    """3-row × 4-column layout with a rhythm strip at the bottom."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.15], hspace=0.35, wspace=0.15)

    n_total = len(t)
    seconds_per_panel = 2.5
    samples_per_panel = int(seconds_per_panel * fs)

    # Each of the 3 rows × 4 columns shows one lead for 2.5s
    for col in range(4):
        for row in range(3):
            name = COL_LAYOUT[row][col]
            ax = fig.add_subplot(gs[row, col])
            start = col * samples_per_panel
            end = min(start + samples_per_panel, n_total)
            v = leads[name][start:end]
            tt = t[start:end] - t[start]
            ax.plot(tt, v, color='black', linewidth=1.0)
            ax.set_xlim(0, seconds_per_panel)
            ax.grid(True, which='both', alpha=0.3, color='#ffbbbb')
            ax.set_facecolor('#fffafa')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.02, 0.92, name, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#333')

    # Rhythm strip (full length)
    ax_r = fig.add_subplot(gs[3, :])
    ax_r.plot(t, leads[rhythm_lead], color='black', linewidth=0.9)
    ax_r.set_xlim(0, t[-1])
    ax_r.grid(True, alpha=0.3, color='#ffbbbb')
    ax_r.set_facecolor('#fffafa')
    ax_r.set_xlabel('Time (s)')
    ax_r.text(0.005, 0.92, f'Rhythm: {rhythm_lead}',
              transform=ax_r.transAxes, fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=15, y=0.995)
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--title', default='Digitized 12-lead ECG')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    leads, t, fs = load_signals(args.csv)
    print(f"Loaded {len(t)} samples, estimated fs={fs} Hz, "
          f"duration={t[-1]:.1f}s")
    for name in LEAD_ORDER:
        n_ok = np.sum(~np.isnan(leads[name]))
        print(f"  {name:<4} {n_ok}/{len(t)} non-nan")

    stacked = os.path.join(args.out_dir, 'ecg_stacked.png')
    clinical = os.path.join(args.out_dir, 'ecg_3x4.png')
    plot_stacked(leads, t, fs, stacked, args.title + ' (stacked)')
    plot_clinical_3x4(leads, t, fs, clinical, args.title + ' (3x4 + rhythm)')
    print(f"\nWrote:\n  {stacked}\n  {clinical}")


if __name__ == '__main__':
    main()
