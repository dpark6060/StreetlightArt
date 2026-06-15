"""Investigation 02 — Generate Kolmogorov phase screens with aotools.

Measures:
  - Initial generation time (PhaseScreenKolmogorov constructor)
  - Per-row add_row() time
  - Spatial frequency content vs the current sine-based displacement field

Outputs:
  investigation/out/phase_screen.png
  investigation/out/phase_screen_displacement_x.png
  investigation/out/phase_screen_displacement_y.png
  investigation/out/phase_screen_psd.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_screen_generation(nx: int, pixel_scale: float, r0: float, L0: float, n_runs: int = 3):
    """Time the PhaseScreenKolmogorov constructor at the given resolution."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        scr = PhaseScreenKolmogorov(nx_size=nx, pixel_scale=pixel_scale, r0=r0, L0=L0)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times), scr


def time_add_row(scr: PhaseScreenKolmogorov, n_rows: int = 200):
    """Time add_row() over n_rows iterations."""
    times = []
    for _ in range(n_rows):
        t0 = time.perf_counter()
        scr.add_row()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


def visualise_screen(scr_array: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(scr_array, cmap="RdBu_r")
    ax.set_title(f"Kolmogorov phase screen ({scr_array.shape[1]}x{scr_array.shape[0]})")
    plt.colorbar(im, ax=ax, label="phase (rad)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def visualise_displacement(dx: np.ndarray, dy: np.ndarray, prefix: Path) -> None:
    for arr, name in [(dx, "dx"), (dy, "dy")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(arr, cmap="RdBu_r")
        ax.set_title(f"Displacement {name} (px), shape={arr.shape}")
        plt.colorbar(im, ax=ax, label="px")
        fig.tight_layout()
        fig.savefig(str(prefix) + f"_{name}.png", dpi=120)
        plt.close(fig)


def radial_psd(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially-averaged power spectral density of a 2D array."""
    F = np.fft.fftshift(np.fft.fft2(arr - arr.mean()))
    P = np.abs(F) ** 2
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(arr.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    rmax = min(cy, cx)
    bins = np.bincount(r.ravel(), weights=P.ravel(), minlength=rmax + 1)
    counts = np.bincount(r.ravel(), minlength=rmax + 1)
    counts = np.where(counts == 0, 1, counts)
    psd = bins / counts
    return np.arange(len(psd))[1:rmax], psd[1:rmax]


def main() -> None:
    print("=" * 60)
    print("Test 1: Generate phase screen at 1280x720 (full preview res)")
    print("=" * 60)
    # Use square 1280 first since aotools requires square input
    nx = 1024  # PhaseScreenKolmogorov requires nx_size to be 2*N for some N
    pixel_scale = 0.05  # meters/px in atmospheric phase space
    r0 = 0.15
    L0 = 100.0
    print(f"nx={nx}, pixel_scale={pixel_scale}, r0={r0}, L0={L0}")
    gen_times, scr = time_screen_generation(nx, pixel_scale, r0, L0, n_runs=3)
    print(f"Initial generation: mean {gen_times.mean()*1000:.1f} ms (min {gen_times.min()*1000:.1f})")

    # Crop to 1280x720 for display
    arr = scr.scrn  # (nx, nx)
    print(f"Phase screen shape: {arr.shape}")
    print(f"  min: {arr.min():.3f}, max: {arr.max():.3f}, std: {arr.std():.3f}")

    # We need at least 1280 for width — generate a wider screen
    # aotools doesn't directly support non-square, but we can use a nx=2048 screen.
    print()
    print("=" * 60)
    print("Test 2: nx=2048 screen (covers 1280x720 with margin)")
    print("=" * 60)
    nx2 = 2048
    gen2_times, scr2 = time_screen_generation(nx2, pixel_scale, r0, L0, n_runs=2)
    print(f"Initial generation: mean {gen2_times.mean()*1000:.1f} ms")

    # Crop to 720x1280
    arr2 = scr2.scrn[:720, :1280]
    visualise_screen(arr2, OUT_DIR / "phase_screen.png")
    print(f"Saved cropped phase screen: {OUT_DIR / 'phase_screen.png'}")

    print()
    print("=" * 60)
    print("Test 3: add_row() timing")
    print("=" * 60)
    row_times = time_add_row(scr2, n_rows=200)
    print(f"add_row() per call: mean {row_times.mean()*1000:.2f} ms, "
          f"median {np.median(row_times)*1000:.2f} ms, "
          f"max {row_times.max()*1000:.2f} ms")

    print()
    print("=" * 60)
    print("Test 4: Convert to displacement field (gradient of phase screen)")
    print("=" * 60)
    # Use TWO independent phase screens — one for dx and one for dy
    print("Generating second screen for dy...")
    _, scr_dy = time_screen_generation(nx2, pixel_scale, r0 * 1.4, L0, n_runs=1)
    arr_dy = scr_dy.scrn[:720, :1280]

    # Normalize each to ±target_amplitude_px at the horizon
    # A typical heat shimmer amplitude is 1-3 px at the apparent horizon.
    target_amp = 3.0
    dx = arr2 / np.std(arr2) * target_amp / 3.0  # ~3-sigma maps to target_amp
    dy = arr_dy / np.std(arr_dy) * target_amp / 3.0
    print(f"dx range: [{dx.min():.2f}, {dx.max():.2f}] px, std {dx.std():.2f}")
    print(f"dy range: [{dy.min():.2f}, {dy.max():.2f}] px, std {dy.std():.2f}")

    visualise_displacement(dx, dy, OUT_DIR / "phase_screen_displacement")
    print(f"Saved displacement visualizations: {OUT_DIR}/phase_screen_displacement_*.png")

    print()
    print("=" * 60)
    print("Test 5: Power spectrum vs sine field (sanity check)")
    print("=" * 60)
    # Compare PSD of Kolmogorov screen vs the current 2-octave sine field
    H, W = 720, 1280
    sy, sx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    f1, f2 = 0.003, 0.007
    s1, s2 = 0.4, 0.6
    t = 0.0
    norm_y = (H - 1 - sy) / (H - 1)
    distance_scale = norm_y ** 1.5
    sine_dx = 3.0 * distance_scale * (
        np.sin(f1 * sy + s1 * t) + 0.5 * np.sin(f2 * sy + 0.7 * sx + s2 * t)
    )

    k_kolm, psd_kolm = radial_psd(arr2)
    k_sine, psd_sine = radial_psd(sine_dx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k_kolm, psd_kolm, label="Kolmogorov phase screen", color="C0")
    ax.loglog(k_sine, psd_sine, label="2-octave sine (current)", color="C3")
    # Reference k^(-11/3) = 1D PSD slope from k^(-5/3) phase struct func via integration
    k_ref = np.array([2.0, 200.0])
    ax.loglog(k_ref, 1e6 * k_ref ** (-11.0 / 3.0), "k--", alpha=0.5,
              label="k^-11/3 reference")
    ax.set_xlabel("spatial freq k (cycles/img)")
    ax.set_ylabel("radial PSD")
    ax.set_title("Power spectral density: Kolmogorov vs sine")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_screen_psd.png", dpi=120)
    plt.close(fig)
    print(f"Saved PSD comparison: {OUT_DIR / 'phase_screen_psd.png'}")

    print()
    print("=" * 60)
    print("Test 6: Coarse vs full-resolution (upsample 320x180 -> 1280x720)")
    print("=" * 60)
    nx_coarse = 256
    gen3_times, scr3 = time_screen_generation(nx_coarse, pixel_scale * 4.0, r0, L0, n_runs=3)
    print(f"Coarse 256x256 generation: mean {gen3_times.mean()*1000:.1f} ms")
    coarse = scr3.scrn[:180, :320]
    # Upsample using bilinear via PIL
    from PIL import Image
    pil = Image.fromarray((coarse * 50.0 + 128).clip(0, 255).astype(np.uint8))
    upsampled = np.asarray(pil.resize((1280, 720), Image.BILINEAR)).astype(np.float32)
    upsampled = (upsampled - 128) / 50.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(coarse, cmap="RdBu_r")
    axes[0].set_title("Coarse (320x180)")
    axes[1].imshow(upsampled, cmap="RdBu_r")
    axes[1].set_title("Bilinear upsampled to 1280x720")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_screen_coarse_vs_full.png", dpi=120)
    plt.close(fig)
    print(f"Saved coarse vs full: {OUT_DIR / 'phase_screen_coarse_vs_full.png'}")


if __name__ == "__main__":
    main()
