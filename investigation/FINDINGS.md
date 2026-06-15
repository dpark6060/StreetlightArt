# Heat Haze Investigation — Findings

Investigation of the plan in `~/.claude/plans/the-heat-haze-animation-imperative-hennessy.md`.
All prototype code is in `investigation/` and outputs (images, .npy, stats) in `investigation/out/`.

---

## TL;DR / Recommendation

| Decision | Recommendation | Confidence |
|---|---|---|
| Phase screen generation | **Pure-numpy FFT-filtered Von Karman noise**, NOT aotools | High |
| Animation strategy | **Frozen-flow** (sliding window over a wide screen) | High |
| Layer count | **3 layers** at different scales/wind vectors | Medium — single-layer is also acceptable |
| Warping backend | **`torch.nn.functional.grid_sample` on MPS** | High |
| Depth normalization | Hardcode to `atmosphere.max_distance` from config (≈20km) | High |
| Phase screen resolution | Native 1280×720 is fast enough w/ FFT (~30ms); no upsampling needed | High |
| Bilinear vs nearest | Bilinear — visible improvement, no real cost difference w/ MPS | High |
| Sky distortion | Apply distortion at horizon-clamped distance (no special-case sky mask) | Medium |

The investigation plan's broad strokes hold up. The biggest revisions are
(a) **don't use aotools' Cholesky path** — too slow at native resolution, and
(b) **don't try to derive horizon distance from the depth map** — use a
config-defined atmospheric fade distance instead.

---

## Investigation 1 — Depth Map Extraction

`investigation/01_depth_map.py` modifies the raycast logic from `_raycast_to_flat_map`
to also retain `t_val` (parametric ray distance, meters), masks sky as NaN, and
visualizes the per-pixel depth.

**Key findings:**

- Median ground-hit depth at the configured camera: **8,110 m** (8 km)
- Ray-distance distribution is **highly long-tailed** because of the
  near-horizon asymptote. With camera tilted slightly down, the topmost row
  with any ground hits has dz ≈ −0.0003, yielding t ≈ 5,000,000 m.
  See `out/depth_stats.txt`:

  ```
  count:  588800 / 921600 (63.9% ground hits)
  min:    4026.1 m
  median: 8110.2 m
  p99:    417995.4 m
  max:    6078695.5 m   <-- this is one ray nearly tangent to z=0
  ```

- The investigation plan suggested clamping sky pixels to "the mean or maximum
  `t_val` along the horizon row." **This does not work in practice.** No row
  has a meaningfully bounded max — the ray-distribution transitions abruptly
  from "all sky" to "all ground" within a single row, and that row's pixels
  all have near-horizontal rays with t_val in the millions. Even the median
  of that row is ~5,000 km.
- **Alternative that does work:** clamp depths to the existing
  `atmosphere.max_distance` from config (currently 20 km). This is the
  distance beyond which atmospheric haze already fades the scene to nothing,
  so it is the natural ceiling for "atmospheric path that still matters."
- For the prototype: `distance_scale = (clip(depth, 0, max_haze) / max_haze) ** 1.5`.
  Median distance_scale across the image was **0.894**, max **1.0**. Foreground
  (roads near the camera at ~4km) gets scale ~0.07 — nearly no shimmer, which
  matches reality.

**Files:** `out/depth_map.npy`, `out/depth_map_visualization.png`, `out/depth_histogram.png`.

---

## Investigation 2 — aotools Kolmogorov Phase Screens

`investigation/02_aotools_phase_screen.py` installs aotools, generates a
`PhaseScreenKolmogorov`, and times it.

**Timing (Apple M4 Pro):**

| Resolution | Initial gen | `add_row()` per frame |
|---|---|---|
| 1024×1024 | **4.5 s**  | (not measured) |
| 2048×2048 | **23.0 s** | 3.4 ms |
| 256×256   | 0.28 s     | (not measured) |

The plan estimated 50–200 ms for initial generation. Actual cost is **20–100×
higher** because the constructor runs a Cholesky factorization that scales as
O(nx³). This is unworkable for any interactive editing of camera or config.

**`add_row()` is fast enough for animation** at 3.4 ms per frame, but it is
useful only with the `aotools` constructor. The pure-numpy alternative
(below) is faster end-to-end and supports non-square shapes natively.

**aotools is NOT recommended** — keep it out of the dependency list.

---

## Investigation 4 — Pure-numpy FFT-filtered Von Karman Screen

The replacement: filter complex white noise in Fourier space by
`sqrt(Φ_φ(k))` where `Φ_φ(k) = 0.023 · r0^(-5/3) · (k² + 1/L0²)^(-11/6)` is
the Von Karman PSD (Kolmogorov with finite outer scale, removes the k=0
divergence). This is what aotools does internally — minus the iterative
Cholesky path used for `add_row()`.

```python
def kolmogorov_screen_fft(height, width, pixel_scale_m, r0, L0, seed):
    fy = np.fft.fftfreq(height, d=pixel_scale_m)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_m)[None, :]
    k2 = (2 * np.pi * fx)**2 + (2 * np.pi * fy)**2
    psd = 0.023 * r0**(-5.0/3) * (k2 + (1.0/L0)**2)**(-11.0/6)
    psd[0, 0] = 0.0
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(...) + 1j * rng.standard_normal(...)
    field_F = noise * np.sqrt(psd) * np.sqrt(height * width) * pixel_scale_m
    return np.fft.ifft2(field_F).real
```

**Timing (4096×720 wide screen for frozen flow, M4 Pro):**

- mean: **119 ms**, min: 113 ms (*35× faster than aotools' equivalent area*)
- Per-frame slide cost (1280-wide window): 3.2 ms

**PSD validation:** the FFT output's radial PSD follows `k^(-11/3)` over the
inertial range (k ≈ 10–200 cycles per image side). See
`out/frozen_flow_psd.png`. This is the correct atmospheric power law.

**Visual sanity:** `out/frozen_flow_still.png` shows correlated patches at
multiple scales, no periodicity. `out/frozen_flow_anim.gif` shows smooth
advection across the wide screen.

---

## Investigation 3 — Warping Backend Benchmark

`investigation/03_warp_benchmark.py` benchmarks four warping methods at 1280×720
with a ±3 px displacement field. Median frame time over 30 iterations after 5
warmup.

| Method | Median ms | fps | Quality |
|---|---|---|---|
| numpy fancy index (current) | 17.0 | 59  | nearest, **visible step artifacts** |
| scipy.ndimage.map_coordinates | 58.5 | 17 | bilinear |
| torch grid_sample CPU         |  8.5 | 118 | bilinear |
| **torch grid_sample MPS (full pipeline)** | 6.7 | **149** | bilinear |
| **torch grid_sample MPS (preloaded image)** | **1.2** | **805** | bilinear |

Plan estimate for numpy was "~300 fps" — actual is ~60 fps with the displacement
field calculation included. Plan estimate for MPS was "unknown — needs test" —
actual is 150 fps full pipeline, 800 fps if the image stays on device.

**Recommendation:** torch grid_sample MPS, with the base image preloaded once
when `base_frame_dirty=True` (camera moved) and only the dx/dy field uploaded
each frame. Quality comparison in `out/warp_quality_comparison.png` — bilinear
is visibly cleaner on the diagonal stripes test pattern.

**On the torch dependency:** PyTorch 2.x with MPS pulls in ~250 MB. Justified
here because (a) MPS provides 5–10× headroom over the current numpy approach,
(b) the bilinear interpolation actually makes the heat haze look like a
real refraction effect rather than blocky pixel-shuffle, and (c) future work
(GPU-accelerated phase screen advection) gets a direct path.

---

## Layer Tuning — Critical Constraint

**Outer-scale-in-pixels = `L0 / pixel_scale_m` must be smaller than the image
extent.** When the outer scale is larger than the image, the entire visible
region sits in the PSD's flat low-k plateau — the whole frame becomes ONE big
random eddy that slowly bulges and contracts. This is *not* heat shimmer; it
is global "warping."

The first layer config tried in `05_end_to_end.py` had this failure:

| Layer | L0 | pixel_scale_m | eddy_px = L0/ps |
|---|---|---|---|
| old L1 | 200 m | 0.10 | **2000 px**  (1.5× image width!) |
| old L2 | 60 m  | 0.05 | 1200 px |
| old L3 | 20 m  | 0.02 | 1000 px |

The new config keeps all eddies within the visible image:

| Layer | L0 | pixel_scale_m | eddy_px |
|---|---|---|---|
| new L1 | 5.0 m | 0.5 | 10 px |
| new L2 | 2.0 m | 0.2 | 10 px |
| new L3 | 0.6 m | 0.1 | 6 px  |

See `out/layer_tuning_compare.png` for the visual evidence of the difference.

**Tuning guidance for the implementation:** target eddy_px ≈ 3–50 px across
the layer stack. Anything > image_width / 4 will read as global warping
rather than shimmer. Anything < 3 px will alias to noise. The art-direction
sweet spot is probably:

| Layer role | eddy_px target | wind speed (px/s) |
|---|---|---|
| Visible-scale eddies | 25–40 px | slow (~30) |
| Mid-scale | 8–15 px  | medium (~70) |
| Fine flicker | 3–6 px   | fast (~120) |

This is also the semantic that should be exposed in the TOML config —
`eddy_size_px` is more intuitive to tune than the physically-correct
(`L0`, `pixel_scale_m`) pair.

---

## Investigation 5+6 — End-to-End Integration

`investigation/05_end_to_end.py` wires depth × multi-layer phase screens × torch
warping. Three modes: `still`, `video`, `perf`.

**Three-layer config used in the prototype:**

| Layer | r0 (m) | L0 (m) | px_scale | weight | wind (px/s)         |
|------:|-------:|-------:|---------:|-------:|---------------------|
| 1 (slow large) | 0.05 | 200 | 0.10 | 1.00 | (80, 8)   |
| 2 (mid)        | 0.10 | 60  | 0.05 | 0.60 | (110, −5) |
| 3 (fast fine)  | 0.15 | 20  | 0.02 | 0.35 | (160, 12) |

Each layer is generated as an independent FFT screen (×2: dx and dy), then
advected at its own wind vector. Per-pixel `dx, dy` is the weighted sum,
multiplied by the depth-scaled amplitude.

**Per-frame end-to-end cost: 56.1 ms (17.8 fps).** Breakdown (estimated):
- 3-layer numpy slide_window: ~45 ms (numpy fancy-index allocations dominate)
- depth scale multiply + H→D copy + grid_sample + D→H copy: ~6 ms
- python overhead: ~5 ms

**The bottleneck is the numpy slide loop, not the GPU warp.** Two
optimizations the implementation should consider:

1. **Keep displacement on device.** Generate the wide phase screen on numpy
   at startup, upload to MPS once, then do all slicing/advection on the GPU
   (`torch.gather` or fancy indexing). This removes 3 H→D copies per frame.
2. **Pre-allocate output arrays in HazeField.** The current `_slide` path
   creates ~200 MB of intermediate arrays per call.

Either should reach 60 fps comfortably. For an art project where frame rate
above 30 fps is sufficient, the current 17.8 fps is already enough for live
preview.

**Phase screen build cost (one-time):** 712 ms (3 layers × 2 screens × 4096
columns × 720 rows). Acceptable startup cost.

**Visuals:**
- `out/end_to_end_t1.5.png` — Kolmogorov-haze frame at t=1.5s
- `out/end_to_end_sine_baseline.png` — current sine-based haze for comparison
- `out/end_to_end_anim.gif` — 90-frame 30fps animation

In stills the difference is subtle — the underlying scene has very fine high-
contrast detail (road network) that gets shifted in both implementations. The
perceptual difference is in **motion**: the sine version shows
diagonally-traveling wave fronts (the giveaway), the Kolmogorov version
shimmers organically with no preferred direction or frequency.

The PSD plot in `out/phase_screen_psd.png` confirms the difference at the
spectral level: the sine field has sharp peaks at the carrier frequencies
(k ≈ 2 and k ≈ 140), while the Kolmogorov screen has the continuous
`k^(-11/3)` slope of real turbulence.

---

## FFT Phase-Shift Dead End (Inv 6 from plan)

Confirmed dead end. The Fourier shift theorem says
`F{f(x − a)} = e^(-i2πfa) · F(f)`. This is a **global** phase shift —
multiplying every frequency component by the same complex exponential
translates the entire image by a single (a, b) vector. There is no Fourier-
domain operation that expresses spatially-varying per-pixel displacement;
that would require the displacement field to be linear in position
(an affine warp), which heat shimmer is not.

FFT is still essential for **generating** the phase screen (filtering noise
by the Von Karman spectrum), as used in inv 4. It is just not useful for
**applying** the warp.

---

## Recommended Implementation Path

Now that the prototype works, the cleanest path to merging this into the codebase:

### Phase A — Replace `apply_heat_haze` with the Kolmogorov pipeline

1. Add `aotools`-free Kolmogorov screen generator (`kolmogorov_screen_fft`)
   to `rendering/heat_haze.py` (new module).
2. Add `HazeField` class (multi-layer frozen flow). Keep numpy for now;
   optimize to MPS later if frame rate needs improvement.
3. Modify `_raycast_to_flat_map` to also return `t_val` as a 4th value
   (depth_map). Update existing call sites to unpack 4 values. Update tests.
4. Add `compute_haze_distance_scale(depth, max_haze_distance, exponent)` —
   reads `atmosphere.max_distance` and `[heat_haze].distance_exponent`.
5. Replace `apply_heat_haze` body. Keep the same signature (frame, t, config)
   plus a new `distance_scale` argument computed once per camera move.

### Phase B — Switch warping backend to torch MPS

1. Add `torch>=2.1` to dependencies (or make it optional with numpy fallback).
2. Add `TorchWarper` class that lazily moves the base frame to MPS once per
   camera move and warps each frame.
3. Wire into `cmd_preview` and `cmd_animate`. The `base_frame_dirty` flag
   already triggers a re-upload at the right moment.

### Phase C — New config schema

```toml
[heat_haze]
enabled = true
max_distance_m = 20000.0          # falls back to atmosphere.max_distance if absent
amplitude_px = 2.5                # std of dx/dy at distance_scale=1.0
distance_exponent = 1.5
seed = 42
backend = "torch_mps"             # "torch_mps" | "torch_cpu" | "numpy_nearest"
interpolation = "bilinear"

[[heat_haze.layers]]
r0 = 0.05
L0 = 200.0
pixel_scale_m = 0.10
weight = 1.00
wind_x_pxs = 80.0
wind_y_pxs = 8.0

[[heat_haze.layers]]
r0 = 0.10
# ... etc
```

The plan's proposed flat config (single r0, L0, etc.) hides the multi-layer
structure. Since the multi-layer stack is what makes the result look
naturalistic vs single-octave, the config should expose layers explicitly.

### Open questions for the user before implementing

1. **Torch dependency acceptable?** Adds ~250MB, but unlocks MPS warp. If
   no, fallback path is torch CPU (118 fps still beats current 60 fps with
   bilinear) — same code path, drop "MPS" from device.
2. **Should `apply_heat_haze` accept a depth map argument?** Cleanest API.
   Otherwise we'd thread the depth map through render_pipeline. The
   `AnimationState` dataclass in `main.py` already stores `base_frame` —
   the depth map would naturally live next to it.
3. **Should the sky region get distortion at all?** The depth map clamps
   sky pixels to `max_haze_distance`, so they get full distortion. That
   makes stars shimmer, which is physically correct for a long atmospheric
   path. If the art direction prefers stable stars, mask the sky region
   (`~ground_mask`) to receive zero distortion.

---

## Files Produced

```
investigation/
├── 01_depth_map.py
├── 02_aotools_phase_screen.py
├── 03_warp_benchmark.py
├── 04_frozen_flow.py
├── 05_end_to_end.py
├── debug_horizon.py
├── FINDINGS.md  (this file)
└── out/
    ├── depth_map.npy
    ├── depth_map_visualization.png
    ├── depth_histogram.png
    ├── depth_stats.txt
    ├── phase_screen.png
    ├── phase_screen_displacement_dx.png
    ├── phase_screen_displacement_dy.png
    ├── phase_screen_psd.png
    ├── phase_screen_coarse_vs_full.png
    ├── frozen_flow_anim.gif
    ├── frozen_flow_still.png
    ├── frozen_flow_psd.png
    ├── frozen_flow_timing.txt
    ├── warp_numpy.png / warp_scipy.png / warp_torch_cpu.png / warp_torch_mps.png
    ├── warp_quality_comparison.png
    ├── warp_benchmark.txt
    ├── base_frame.png
    ├── end_to_end_t0.0.png / end_to_end_t1.5.png / end_to_end_t3.0.png
    ├── end_to_end_sine_baseline.png
    └── end_to_end_anim.gif
```

## New dependencies installed during investigation

For investigation/prototyping (added to project venv via `uv pip install`):

- `aotools==1.0.7` — evaluated and rejected; **uninstall if not needed**
- `numba==0.65.1` (aotools dep) — **uninstall**
- `llvmlite==0.47.0` (numba dep) — **uninstall**
- `scipy==1.17.1` (aotools dep + benchmark) — keep or remove based on needs
- `torch==2.11.0` — **keep if Phase B is approved**

Run `uv pip uninstall aotools numba llvmlite` when the investigation is closed.
