"""Investigation 03 — Benchmark image warping methods at 1280x720.

Compares:
  1. numpy fancy indexing (current)        — nearest neighbour
  2. scipy.ndimage.map_coordinates          — bilinear
  3. torch.nn.functional.grid_sample (CPU)  — bilinear
  4. torch.nn.functional.grid_sample (MPS)  — bilinear, Apple Metal

For each, runs N warmup + M timed iterations, reports mean / median fps.
Also writes a sample output image to inspect quality differences.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 720, 1280
WARMUP = 5
ITER = 30


def make_test_frame() -> np.ndarray:
    """Synthetic test frame with diagonal stripes (so warp is visible)."""
    yy, xx = np.indices((H, W))
    band = ((xx + yy) // 12) % 2
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (band * 200 + 30).astype(np.uint8)
    rgb[..., 1] = ((1 - band) * 180 + 40).astype(np.uint8)
    rgb[..., 2] = 80
    # Add a few colored squares for shimmer reference
    for cy, cx, color in [(360, 320, (255, 80, 80)), (360, 640, (80, 255, 80)),
                          (360, 960, (80, 80, 255))]:
        rgb[cy - 30: cy + 30, cx - 30: cx + 30] = color
    return rgb


def make_displacement(amp: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random displacement field, ±amp px."""
    rng = np.random.default_rng(seed)
    dx_lo = rng.standard_normal((H // 8, W // 8)).astype(np.float32) * amp
    dy_lo = rng.standard_normal((H // 8, W // 8)).astype(np.float32) * amp
    dx = np.asarray(Image.fromarray(dx_lo).resize((W, H), Image.BICUBIC))
    dy = np.asarray(Image.fromarray(dy_lo).resize((W, H), Image.BICUBIC))
    return dx, dy


def warp_numpy(frame: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Current approach: integer rounding + fancy indexing."""
    h, w = frame.shape[:2]
    sy = np.arange(h, dtype=np.float32)[:, None]
    sx = np.arange(w, dtype=np.float32)[None, :]
    src_y = np.clip(sy + dy, 0, h - 1).astype(np.int32)
    src_x = np.clip(sx + dx, 0, w - 1).astype(np.int32)
    return frame[src_y, src_x]


def warp_scipy(frame: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """scipy.ndimage.map_coordinates with bilinear (order=1)."""
    h, w = frame.shape[:2]
    sy = np.arange(h, dtype=np.float32)[:, None]
    sx = np.arange(w, dtype=np.float32)[None, :]
    src_y = np.clip(sy + dy, 0, h - 1)
    src_x = np.clip(sx + dx, 0, w - 1)
    out = np.empty_like(frame)
    for c in range(3):
        out[..., c] = ndi.map_coordinates(
            frame[..., c], [src_y, src_x], order=1, mode="nearest"
        )
    return out


def warp_torch(frame: np.ndarray, dx: np.ndarray, dy: np.ndarray, device: str) -> np.ndarray:
    """torch grid_sample with bilinear interpolation."""
    h, w = frame.shape[:2]
    img = torch.from_numpy(frame).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    sy = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
    sx = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0)
    dxg = torch.from_numpy(dx).to(device)
    dyg = torch.from_numpy(dy).to(device)
    src_x = (sx + dxg).clamp(0, w - 1)
    src_y = (sy + dyg).clamp(0, h - 1)
    # grid_sample expects normalized [-1, 1] coordinates
    grid_x = (src_x / (w - 1)) * 2.0 - 1.0
    grid_y = (src_y / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    out_np = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return out_np


def warp_torch_preloaded(img_t: torch.Tensor, dx_t: torch.Tensor, dy_t: torch.Tensor,
                          base_grid_x: torch.Tensor, base_grid_y: torch.Tensor) -> torch.Tensor:
    """Variant where image and grid bases are already on device — measures pure compute."""
    h = img_t.shape[2]
    w = img_t.shape[3]
    src_x = (base_grid_x + dx_t).clamp(0, w - 1)
    src_y = (base_grid_y + dy_t).clamp(0, h - 1)
    grid_x = (src_x / (w - 1)) * 2.0 - 1.0
    grid_y = (src_y / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def benchmark(fn, *args, label: str, sync_fn=None) -> dict:
    for _ in range(WARMUP):
        fn(*args)
    if sync_fn:
        sync_fn()
    times = []
    for _ in range(ITER):
        t0 = time.perf_counter()
        result = fn(*args)
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    fps_mean = 1.0 / arr.mean()
    fps_median = 1.0 / np.median(arr)
    print(f"{label:<40s} mean {arr.mean()*1000:6.2f}ms ({fps_mean:5.0f} fps), "
          f"median {np.median(arr)*1000:6.2f}ms ({fps_median:5.0f} fps)")
    return {"label": label, "mean_ms": arr.mean() * 1000, "median_ms": np.median(arr) * 1000,
            "fps_mean": fps_mean, "fps_median": fps_median, "result": result}


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print()

    frame = make_test_frame()
    dx, dy = make_displacement(amp=3.0, seed=0)
    print(f"Frame {frame.shape}, displacement std dx={dx.std():.2f}, dy={dy.std():.2f}")
    print()

    results = []

    # 1. numpy
    print("--- 1. numpy fancy indexing (nearest, current approach) ---")
    r = benchmark(warp_numpy, frame, dx, dy, label="numpy fancy index (nearest)")
    results.append(r)
    Image.fromarray(r["result"]).save(OUT_DIR / "warp_numpy.png")

    # 2. scipy
    print("--- 2. scipy.ndimage.map_coordinates (bilinear) ---")
    r = benchmark(warp_scipy, frame, dx, dy, label="scipy.ndimage map_coords (bilinear)")
    results.append(r)
    Image.fromarray(r["result"]).save(OUT_DIR / "warp_scipy.png")

    # 3. torch CPU
    print("--- 3. torch.nn.functional.grid_sample CPU (bilinear) ---")
    r = benchmark(warp_torch, frame, dx, dy, "cpu", label="torch grid_sample CPU (bilinear)")
    results.append(r)
    Image.fromarray(r["result"]).save(OUT_DIR / "warp_torch_cpu.png")

    # 4. torch MPS
    if torch.backends.mps.is_available():
        print("--- 4. torch.nn.functional.grid_sample MPS (bilinear) ---")
        try:
            r = benchmark(
                warp_torch, frame, dx, dy, "mps",
                label="torch grid_sample MPS (bilinear, full pipeline)",
                sync_fn=torch.mps.synchronize,
            )
            results.append(r)
            Image.fromarray(r["result"]).save(OUT_DIR / "warp_torch_mps.png")
        except Exception as e:
            print(f"MPS failed: {e!r}")

        # 5. torch MPS preloaded — measures pure compute, not host transfer
        print("--- 5. torch grid_sample MPS preloaded (no per-frame H<->D) ---")
        device = "mps"
        img_t = torch.from_numpy(frame).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        dx_t = torch.from_numpy(dx).to(device)
        dy_t = torch.from_numpy(dy).to(device)
        sy = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        sx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)
        try:
            r = benchmark(
                warp_torch_preloaded, img_t, dx_t, dy_t, sx, sy,
                label="torch grid_sample MPS (preloaded compute only)",
                sync_fn=torch.mps.synchronize,
            )
            results.append(r)
        except Exception as e:
            print(f"MPS preloaded failed: {e!r}")
    else:
        print("MPS not available, skipping.")

    # Save results table
    print()
    print("=" * 72)
    print(f"Summary at {W}x{H}, displacement amplitude ~3px:")
    print("=" * 72)
    for r in results:
        print(f"  {r['label']:<55s} {r['median_ms']:6.2f}ms  ({r['fps_median']:5.0f} fps)")

    # Write out a side-by-side quality comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        result = r.get("result")
        if isinstance(result, torch.Tensor):
            continue  # preloaded run keeps tensor on MPS — skip rather than convert
        if result is None:
            continue
        # show only a 200x400 crop near red square for detail
        crop = result[330:430, 270:670]
        ax.imshow(crop)
        ax.set_title(f"{r['label']} ({r['fps_median']:.0f} fps median)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "warp_quality_comparison.png", dpi=120)
    print(f"\nSaved quality comparison: {OUT_DIR / 'warp_quality_comparison.png'}")

    with open(OUT_DIR / "warp_benchmark.txt", "w") as f:
        f.write(f"Resolution: {W}x{H}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"MPS available: {torch.backends.mps.is_available()}\n")
        f.write(f"Iterations: {ITER} (after {WARMUP} warmup)\n\n")
        for r in results:
            f.write(f"{r['label']:<55s} median {r['median_ms']:6.2f}ms  "
                    f"({r['fps_median']:5.0f} fps median, {r['fps_mean']:5.0f} fps mean)\n")


if __name__ == "__main__":
    main()
