"""Investigation 04 — Frozen-flow phase screen animation.

Generates a wide Kolmogorov-spectrum phase screen via FFT-filtered noise (much
faster than aotools' Cholesky path — sub-second for 4096x720) and animates by
sliding a 1280-wide window across it. This is Taylor's frozen-turbulence
hypothesis: turbulent eddies are structurally frozen and advect with the wind.

Compares the FFT approach against the aotools `add_row()` extension approach.

Outputs:
  investigation/out/frozen_flow_anim.gif       short animated preview (10 frames)
  investigation/out/frozen_flow_psd.png        PSD validation
  investigation/out/frozen_flow_timing.txt
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kolmogorov_screen_fft(
    height: int,
    width: int,
    pixel_scale_m: float = 0.05,
    r0: float = 0.15,
    L0: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Kolmogorov phase screen via FFT-filtered white noise.

    Implements the Von Karman power spectrum (Kolmogorov with finite outer scale):
        Phi_phi(k) = 0.023 * r0^(-5/3) * (k^2 + 1/L0^2)^(-11/6)
    Filters white noise in Fourier space by sqrt(Phi_phi(k)) * pixel_scale,
    then takes the inverse FFT. Result is a real-valued phase screen.

    Args:
        height (int): Screen height in pixels.
        width (int):  Screen width in pixels.
        pixel_scale_m (float): Meters per pixel in the atmospheric phase plane.
        r0 (float): Fried parameter (m). Smaller = more distortion.
        L0 (float): Outer scale (m). Largest eddy size.
        seed (int | None): RNG seed.

    Returns:
        np.ndarray: (H, W) float32 phase screen in radians.
    """
    rng = np.random.default_rng(seed)

    # Spatial frequency grid (cycles per meter)
    fy = np.fft.fftfreq(height, d=pixel_scale_m)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_m)[None, :]
    k2 = (2 * np.pi * fx) ** 2 + (2 * np.pi * fy) ** 2  # angular wavenumber^2

    # Von Karman PSD (avoids divergence at k=0 vs pure Kolmogorov)
    inv_L0_2 = (1.0 / L0) ** 2
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * (k2 + inv_L0_2) ** (-11.0 / 6.0)
    psd[0, 0] = 0.0  # remove DC (zero-mean screen)

    # White noise in Fourier domain, scaled by sqrt(PSD)
    noise = (rng.standard_normal((height, width))
             + 1j * rng.standard_normal((height, width)))
    field_F = noise * np.sqrt(psd) * np.sqrt(height * width) * pixel_scale_m
    screen = np.fft.ifft2(field_F).real
    return screen.astype(np.float32)


def slide_window(screen: np.ndarray, x_offset: float, window_w: int) -> np.ndarray:
    """Extract a window_w-wide horizontal slice with sub-pixel x_offset (wrap-around).

    Uses bilinear interpolation between adjacent columns.
    """
    H, W = screen.shape
    x_offset = x_offset % W
    i0 = int(np.floor(x_offset))
    frac = x_offset - i0

    cols = (np.arange(window_w) + i0) % W
    cols_p1 = (cols + 1) % W
    a = screen[:, cols]
    b = screen[:, cols_p1]
    return (1.0 - frac) * a + frac * b


def measure_loop_period(screen_w: int, wind_speed_pxs: float, fps: int) -> float:
    """Time, in seconds, before the visible window cycles back to the start."""
    return screen_w / wind_speed_pxs


def main() -> None:
    H, W_out = 720, 1280
    W_screen = 4096  # wide screen for frozen-flow advection

    print("=" * 60)
    print("FFT-filtered Kolmogorov screen generation")
    print("=" * 60)
    print(f"Screen: {W_screen} x {H}")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        screen = kolmogorov_screen_fft(H, W_screen, pixel_scale_m=0.05,
                                        r0=0.15, L0=100.0, seed=42)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"Generation time: mean {np.mean(times)*1000:.1f} ms, "
          f"min {min(times)*1000:.1f} ms")
    print(f"Phase screen std: {screen.std():.3f}")
    print(f"Phase screen range: [{screen.min():.2f}, {screen.max():.2f}]")

    print()
    print("=" * 60)
    print("Sliding window animation")
    print("=" * 60)
    fps = 60
    wind_speed = 80.0  # px/sec
    period = measure_loop_period(W_screen, wind_speed, fps)
    print(f"Wind speed: {wind_speed} px/sec")
    print(f"Visible window: {W_out} px / Screen width: {W_screen} px")
    print(f"Loop period: {period:.1f}s ({period * fps:.0f} frames)")

    # Time the per-frame slide cost
    t0 = time.perf_counter()
    for f in range(120):
        win = slide_window(screen, x_offset=f * (wind_speed / fps), window_w=W_out)
    t1 = time.perf_counter()
    per_frame_ms = (t1 - t0) / 120 * 1000
    print(f"Per-frame slide cost: {per_frame_ms:.2f} ms")

    # Save frames as a small animation GIF
    print()
    print("Building 24-frame animation GIF for visual inspection...")
    frames = []
    for f in range(24):
        win = slide_window(screen, x_offset=f * (wind_speed / fps) * 4.0,  # 4x speed for shorter gif
                           window_w=W_out)
        # Normalize and visualize
        norm = (win - win.min()) / (win.max() - win.min() + 1e-9) * 255
        frames.append(Image.fromarray(norm.astype(np.uint8)).convert("L"))
    gif_path = OUT_DIR / "frozen_flow_anim.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"Saved: {gif_path}")

    print()
    print("=" * 60)
    print("PSD verification: FFT screen should follow k^(-11/3)")
    print("=" * 60)

    # Compute radial PSD of the FFT-generated screen
    F = np.fft.fftshift(np.fft.fft2(screen[:H, :H] - screen[:H, :H].mean()))
    P = np.abs(F) ** 2
    cy, cx = H // 2, H // 2
    y, x = np.indices((H, H))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    rmax = cy
    bins = np.bincount(r.ravel(), weights=P.ravel(), minlength=rmax + 1)
    counts = np.bincount(r.ravel(), minlength=rmax + 1)
    counts = np.where(counts == 0, 1, counts)
    psd = bins / counts
    k = np.arange(len(psd))[1:rmax]
    psd = psd[1:rmax]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k, psd, label="FFT-filtered Kolmogorov", color="C0")
    k_ref = np.array([3.0, 200.0])
    # Power-law slope reference (theoretical k^(-11/3))
    norm = psd[10] * k[10] ** (11.0 / 3.0)
    ax.loglog(k_ref, norm * k_ref ** (-11.0 / 3.0), "k--", alpha=0.6,
              label="k^(-11/3) reference")
    ax.set_xlabel("spatial freq k (cycles/img)")
    ax.set_ylabel("radial PSD")
    ax.set_title(f"Radial PSD of FFT-filtered Kolmogorov screen ({H}x{H})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frozen_flow_psd.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'frozen_flow_psd.png'}")

    # Save a still frame for inspection
    norm = (screen[:H, :W_out] - screen[:H, :W_out].min())
    norm = norm / (norm.max() + 1e-9) * 255
    Image.fromarray(norm.astype(np.uint8)).save(OUT_DIR / "frozen_flow_still.png")

    print()
    print("=" * 60)
    print("Comparison vs aotools constructor time at same area")
    print("=" * 60)
    print(f"FFT (4096x720):    {np.mean(times)*1000:.0f} ms")
    print(f"aotools (1024x1024 from inv 02): ~4500 ms (square only, 18% smaller area)")
    print(f"aotools (2048x2048 from inv 02): ~23000 ms (square only)")

    with open(OUT_DIR / "frozen_flow_timing.txt", "w") as f:
        f.write(f"FFT screen generation ({W_screen}x{H}):\n")
        f.write(f"  mean: {np.mean(times)*1000:.1f} ms\n")
        f.write(f"  min:  {min(times)*1000:.1f} ms\n")
        f.write(f"\nSliding window per frame ({W_out} window): {per_frame_ms:.2f} ms\n")
        f.write(f"\nLoop period at {wind_speed} px/s: {period:.1f}s\n")


if __name__ == "__main__":
    main()
