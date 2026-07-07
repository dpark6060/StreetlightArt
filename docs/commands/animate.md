# `animate` — Heat haze over a rendered image

```bash
# Live interactive preview
uv run streets-at-night animate
uv run streets-at-night animate --image output/render.png --width 1280 --height 720

# Render an MP4 offline (any resolution, no real-time constraint)
uv run streets-at-night animate --record output/shimmer.mp4 --quality medium
uv run streets-at-night animate --record output/shimmer.mp4 --quality high --duration 30 --seamless
```

Loads a static image (typically the output of [`render`](render.md)) and applies physically-motivated heat shimmer on top of it. The original camera (from `scene_config.toml`) is reconstructed so that per-pixel atmospheric path length is known — distortion magnitude scales with depth.

Two modes:

- **Pygame mode** (default): real-time interactive window at 1280×720 (or `--width`/`--height`). Useful for live tuning.
- **Record mode** (`--record FILE`): renders frames offline, encodes to MP4. Use this when you want **higher resolution than your display can play in real time**, or a self-contained video file you can share.

The base image is read once; output frames can be saved on demand with `P` (pygame mode) or are written directly to MP4 (record mode).

> **In the UI:** the **Animate** tab of [`ui`](ui.md) has a base-image picker, a
> live Preview toggle, and Record controls (path, fps, duration, seamless) with a
> progress bar. It's disabled with a message if `[heat_haze] enabled = false`.

## Flags — pygame mode

| Flag | Default | Description |
|---|---|---|
| `--image` | `output/render.png` | Source image to animate |
| `--width` | `1280` | Display window width |
| `--height` | `720` | Display window height |

## Quick reference — record mode

```bash
# Quick loop, low res, seamless — good for iteration
uv run streets-at-night animate \
    --record output/shimmer.mp4 \
    --duration 10 --seamless

# Print-resolution video (~2.5 min wall time per 10s of video)
uv run streets-at-night animate \
    --record output/shimmer_high.mp4 \
    --quality high --duration 20 --seamless

# One natural loop of the slowest layer (the default duration), at 4K
uv run streets-at-night animate \
    --record output/shimmer.mp4 \
    --quality medium
# ~117 s output with the default duration; pass --duration if that's too long.
```

## Flags — record mode

Activated by setting `--record FILE.mp4`. All pygame flags are ignored.

| Flag | Default | Description |
|---|---|---|
| `--record` | _(required)_ | Output MP4 path. Setting this triggers record mode. |
| `--quality` | `low` | Resolution preset: `low` (1920×1080), `medium` (3840×2160), `high` (7680×4320) |
| `--width` | _(preset)_ | Override the preset width |
| `--height` | _(preset)_ | Override the preset height |
| `--duration` | _(auto)_ | Video duration in seconds. Default = 1× the slowest layer's cycle period (one full loop of the dominant motion). |
| `--fps` | `30` | Output frame rate |
| `--seamless` | `false` | Crossfade the tail of the video with the start so it loops without a visible jump |
| `--crossfade-seconds` | `0.5` | Crossfade duration when `--seamless` is set |

The encoder is H.264 with CRF≈18 in a yuv420p MP4 container — the bundled `ffmpeg` binary from `imageio-ffmpeg` is used, no external install required.

### Default duration

For each layer, the cyclic period along an axis is `screen_dim / |wind_along_axis|`. Across all layers the longest period sets the natural "loop length"; the default `--duration` is `1 ×` that. With the default config this works out to ~117 s (≈ 2 minutes) at 30 fps = 3510 frames, which is still long for casual iteration. **Pass an explicit `--duration` for shorter clips** — 5–10 s is plenty to see the shimmer character.

### Seamless looping

When `--seamless` is set, the last `crossfade_seconds × fps` frames are blended with the first `crossfade_seconds × fps` frames using a linear ramp, so output[N-1] equals frame[0] and the loop transition is visually continuous. The trade-off is that the crossfade region itself shows a slight motion-morph rather than the natural shimmer.

For a true mathematically-exact loop you'd need to pick a duration that's a common multiple of every layer's period, which is fiddly with arbitrary wind values; `--seamless` sidesteps that.

### Performance

Roughly: per-frame cost in record mode is the same as in pygame mode (~50 ms at 1920×1080 on Apple M4 Pro), plus encoding. So a 10-second 1080p clip takes ~5 seconds wall time. At `--quality high` (7680×4320) per-frame cost rises to ~250 ms and a 10-second clip takes ~2.5 minutes.

Phase-screen build cost scales with output resolution since `HazeField` auto-bumps screen dimensions to at least the output size: ~2 s at low, ~10 s at medium, ~30 s at high.

## Controls (pygame mode)

| Key | Action |
|---|---|
| `P` | Save current frame to `output/animate_<timestamp>.png` |
| `ESC` | Quit |

## What's actually happening

This is the short version. The full physics derivation, math, diagrams, and references are in [`docs/heat_haze.html`](../heat_haze.html).

The pipeline:

1. **Per-pixel depth map.** Each screen pixel raycasts against the ground plane to get its atmospheric path length in meters. Sky pixels get a configurable fraction of the max-haze distance.
2. **Multi-layer Kolmogorov phase screens.** N tall noise fields with the correct atmospheric power spectrum ($\Phi \propto k^{-11/3}$) are generated once at startup.
3. **Frozen-flow advection.** Each layer slides through its phase screen at its own wind vector (Taylor's hypothesis — eddies advect with the wind faster than they evolve internally).
4. **Depth-aware scaling.** The summed displacement field is scaled per-pixel by `(depth / max_distance)^exponent` so close objects barely shimmer and distant ones shimmer fully.
5. **Bilinear warp via PyTorch MPS.** The base frame stays resident on the GPU; only the displacement field is uploaded each frame. ~150 fps end-to-end at 1280×720 on Apple Silicon.

---

## Tuning the look (`[heat_haze]` config section)

All knobs live in `scene_config.toml` under `[heat_haze]` (top-level scalars) and `[[heat_haze.layers]]` (one entry per atmospheric layer). Defaults are tuned for a moderately distant urban scene.

### Top-level scalars

| Key | Type | Default | Effect |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch. `false` = no warp at all. |
| `amplitude_px` | float | `0.5` | Standard deviation of pixel displacement at the horizon, **at 1920 px wide (the `low` preset).** Automatically scaled to higher resolutions so the shimmer looks identical at any quality level. **Lower = subtler shimmer; higher = more violent.** Real heat shimmer at km-range scenes is sub-pixel to ~1 px; 0.5 is a good art-piece value. Above ~3 the image starts to look like a TV signal. |
| `max_distance_m` | float | `20000.0` | Distance ceiling in meters. Beyond this, distortion saturates at maximum (atmospheric haze already fades the scene to nothing here). Match your `[atmosphere].max_distance`. |
| `distance_exponent` | float | `1.5` | Falloff steepness from horizon to camera. `1.0` = linear (foreground still shimmers a fair bit). `2.0` = quadratic (foreground is much calmer than horizon). `0.5` = flatten the falloff so even foreground shimmers. |
| `sky_distance_factor` | float | `0.5` | Sky pixels' "effective distance" as a fraction of `max_distance_m`. `0.0` = sky doesn't shimmer (stars locked). `1.0` = sky shimmers as much as the farthest ground. `0.5` (default) = sky shimmer is `0.5^1.5 ≈ 35%` of max ground shimmer — visible but quieter than horizon ground. |
| `seed` | int | `42` | RNG seed. Same seed = same shimmer pattern. |
| `screen_width_px` | int | `2048` | Width of the cached phase screen. Loop period along x = `screen_width_px / wind_x_pxs`. Larger = longer loop, more memory. |
| `screen_height_px` | int | `4096` | Height of the cached phase screen. Loop period along y = `screen_height_px / wind_y_pxs`. Larger = longer loop, more memory. The default favours the y axis because rising heat (positive `wind_y_pxs`) is the natural direction. |

### Per-layer settings

Each `[[heat_haze.layers]]` block contributes one independent atmospheric layer. The output displacement field is the weighted sum of all layers. Three layers is a good default balance — one big and slow, one medium, one fine and fast.

| Key | Type | Effect |
|---|---|---|
| `r0` | float | Fried parameter (m). Smaller = stronger turbulence (more low-frequency power). Most tunings can leave this at `0.10`; vary `weight` instead. |
| `L0` | float | Outer scale (m). Sets the layer's largest eddy size. **Eddy size in image pixels ≈ `L0 / pixel_scale_m`.** Keep this number well below your image dimensions or you'll see global bulging instead of localized shimmer. |
| `pixel_scale_m` | float | Phase-plane pixel scale (m/px) at 1920 px wide. Together with `L0` determines eddy size in pixels; automatically scaled so eddies are the same fraction of the image at any resolution. |
| `weight` | float | Mix coefficient. Layer's contribution to the summed displacement is normalized to unit std then multiplied by `weight`. Layers with larger `weight` dominate. |
| `wind_x_pxs` | float | Horizontal advection (pixels/sec) at 1920 px wide. Positive = features move LEFT across the screen. Automatically scaled with resolution so motion speed is perceptually consistent. |
| `wind_y_pxs` | float | Vertical advection (pixels/sec) at 1920 px wide. **Positive = features move UP** (natural direction for hot air rising). Automatically scaled with resolution. |

### Eddy size — the most important tuning concept

The **outer-scale-in-pixels** of a layer is `L0 / pixel_scale_m`. This is the size of the largest correlated patch in that layer's displacement field.

| Eddy size in px | What you see |
|---|---|
| `> image_width / 4` | **Failure mode.** The whole frame becomes one big eddy that bulges and contracts. The image-extent constraint is real — at this scale you've zoomed out below the PSD's flat low-k plateau. |
| `30–50` | Visible turbulent patches. Reads as "atmospheric distortion." |
| `5–15` | Localized shimmer. The dominant heat-haze look. |
| `2–5` | Fine-grained flicker / sparkle. |
| `< 2` | Aliases to noise — the structure is below the screen Nyquist. |

A well-balanced multi-layer config covers a range of eddy sizes from one layer to the next.

### Default 3-layer config

```toml
[heat_haze]
enabled = true
amplitude_px = 0.5
max_distance_m = 20000.0
distance_exponent = 1.5
sky_distance_factor = 0.5
seed = 42
screen_width_px = 2048
screen_height_px = 4096

[[heat_haze.layers]]
r0 = 0.10
L0 = 2.5
pixel_scale_m = 0.5     # eddy = 2.5/0.5 = 5 px (largest, slowest)
weight = 0.6
wind_x_pxs = 4.0
wind_y_pxs = 35.0

[[heat_haze.layers]]
r0 = 0.10
L0 = 1.0
pixel_scale_m = 0.2     # eddy = 1.0/0.2 = 5 px
weight = 0.8
wind_x_pxs = -3.0
wind_y_pxs = 70.0

[[heat_haze.layers]]
r0 = 0.10
L0 = 0.3
pixel_scale_m = 0.1     # eddy = 0.3/0.1 = 3 px (finest, fastest)
weight = 1.0
wind_x_pxs = 8.0
wind_y_pxs = 120.0
```

### Tuning recipes

| Goal | What to change |
|---|---|
| More dramatic shimmer overall | Increase `amplitude_px` (try `1.0` or `1.5`) |
| Subtler, almost-not-there | Decrease `amplitude_px` (try `0.2`) |
| Slower rise / lazier motion | Lower all `wind_y_pxs` proportionally |
| Faster, more violent rise | Higher `wind_y_pxs` |
| Distort horizon only, foreground calm | Increase `distance_exponent` (try `2.5`) |
| Distort across more of the frame | Decrease `distance_exponent` (try `0.8`) |
| Stars / sky shimmer more | Increase `sky_distance_factor` toward `1.0` |
| Lock the sky | `sky_distance_factor = 0.0` |
| Bigger turbulent patches | Multiply each layer's `L0` by 2× (eddy size doubles) |
| Finer texture | Multiply each layer's `L0` by 0.5× |
| Visible loop too short | Increase `screen_height_px` (e.g. `8192`) — more memory, longer loop |
| Sideways drift instead of rising | Swap each layer's `wind_x_pxs` and `wind_y_pxs` |

### Loop period

Each layer's frozen-flow window cycles when it has traveled `screen_dim` pixels along its wind axis. With the defaults the per-layer y-axis loops are 117 s, 58 s, and 34 s. Because the three layers have incommensurate periods their summed pattern doesn't repeat for the LCM of those — usually many minutes. If you do see a perceptible cycle, increase `screen_height_px`.

### Performance

| Phase | Cost | When |
|---|---|---|
| Phase screen generation | ~2 s for 6 screens at default size | Once at startup |
| Depth map raycast | ~100 ms | Once at startup (animate) or per camera move (preview) |
| Per-frame warp | ~6 ms (≈150 fps) | Each frame, MPS backend |
| Per-frame frozen-flow slide | ~45 ms | Each frame, numpy |

End-to-end frame rate on Apple M4 Pro is ~18 fps for 3 layers — currently bottlenecked by the multi-layer numpy slide. The warp itself is much faster than that (under 2 ms on MPS); the slide is the optimization opportunity if you need higher fps.

## Further reading

- [`docs/heat_haze.html`](../heat_haze.html) — full physics/math write-up: Kolmogorov spectrum, Von Karman PSD, FFT noise filtering, frozen flow, depth math, MPS warp benchmarks.
- [`investigation/FINDINGS.md`](../../investigation/FINDINGS.md) — the prototype investigation that drove the design: aotools rejected, MPS benchmarks, depth normalization gotchas.
- [`docs/animation_design.md`](../animation_design.md) — original design doc (predates the Kolmogorov rewrite; kept for historical context).
