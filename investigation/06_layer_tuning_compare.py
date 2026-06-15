"""Visualize displacement-field structure for the OLD vs NEW layer configs.

Shows both dx fields and a still-frame of each layer's individual contribution,
so the eddy scale is visible at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = REPO_ROOT / "investigation" / "out"


def kolmogorov_screen_fft(h, w, pixel_scale_m, r0, L0, seed):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h, d=pixel_scale_m)[:, None]
    fx = np.fft.fftfreq(w, d=pixel_scale_m)[None, :]
    k2 = (2 * np.pi * fx) ** 2 + (2 * np.pi * fy) ** 2
    psd = 0.023 * (r0 ** (-5.0 / 3)) * (k2 + (1.0 / L0) ** 2) ** (-11.0 / 6)
    psd[0, 0] = 0.0
    noise = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    field_F = noise * np.sqrt(psd) * np.sqrt(h * w) * pixel_scale_m
    return np.fft.ifft2(field_F).real


OLD_LAYERS = [
    {"r0": 0.05, "L0": 200.0, "pixel_scale_m": 0.10, "weight": 1.00, "label": "old L1: L0=200m, ps=0.10"},
    {"r0": 0.10, "L0": 60.0,  "pixel_scale_m": 0.05, "weight": 0.60, "label": "old L2: L0=60m,  ps=0.05"},
    {"r0": 0.15, "L0": 20.0,  "pixel_scale_m": 0.02, "weight": 0.35, "label": "old L3: L0=20m,  ps=0.02"},
]
NEW_LAYERS = [
    {"r0": 0.10, "L0": 5.0,   "pixel_scale_m": 0.5, "weight": 0.60, "label": "new L1: L0=5.0m, ps=0.5"},
    {"r0": 0.10, "L0": 2.0,   "pixel_scale_m": 0.2, "weight": 0.80, "label": "new L2: L0=2.0m, ps=0.2"},
    {"r0": 0.10, "L0": 0.6,   "pixel_scale_m": 0.1, "weight": 1.00, "label": "new L3: L0=0.6m, ps=0.1"},
]

H, W = 720, 1280


def render_layers(layers: list[dict], seed_base: int = 100) -> tuple[list[np.ndarray], np.ndarray]:
    fields = []
    for i, lc in enumerate(layers):
        scr = kolmogorov_screen_fft(H, W, lc["pixel_scale_m"], lc["r0"], lc["L0"],
                                     seed_base + i)
        scr = scr / (scr.std() + 1e-9) * lc["weight"]
        fields.append(scr)
    summed = np.sum(fields, axis=0)
    summed *= 2.5 / (summed.std() + 1e-9)  # final amplitude renorm
    return fields, summed


old_fields, old_sum = render_layers(OLD_LAYERS, seed_base=100)
new_fields, new_sum = render_layers(NEW_LAYERS, seed_base=100)

vmin, vmax = -3, 3

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for col, (layer_field, label) in enumerate(zip(old_fields, OLD_LAYERS)):
    axes[0, col].imshow(layer_field, cmap="RdBu_r", vmin=-2, vmax=2)
    axes[0, col].set_title(label["label"], fontsize=10)
    axes[0, col].axis("off")
axes[0, 3].imshow(old_sum, cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 3].set_title("old SUM (final dx)", fontsize=10, weight="bold")
axes[0, 3].axis("off")

for col, (layer_field, label) in enumerate(zip(new_fields, NEW_LAYERS)):
    axes[1, col].imshow(layer_field, cmap="RdBu_r", vmin=-2, vmax=2)
    axes[1, col].set_title(label["label"], fontsize=10)
    axes[1, col].axis("off")
axes[1, 3].imshow(new_sum, cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 3].set_title("new SUM (final dx)", fontsize=10, weight="bold")
axes[1, 3].axis("off")

fig.suptitle("Layer-by-layer displacement structure: OLD (top, global bulging) vs NEW (bottom, local shimmer)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "layer_tuning_compare.png", dpi=110)
plt.close(fig)
print(f"Saved: {OUT_DIR / 'layer_tuning_compare.png'}")

# Also print eddy size estimate
def eddy_px(L0, pixel_scale_m):
    return L0 / pixel_scale_m

print("\nEddy outer-scale size in image pixels:")
for lc in OLD_LAYERS:
    print(f"  {lc['label']:<30s} -> {eddy_px(lc['L0'], lc['pixel_scale_m']):>6.0f} px")
for lc in NEW_LAYERS:
    print(f"  {lc['label']:<30s} -> {eddy_px(lc['L0'], lc['pixel_scale_m']):>6.0f} px")
