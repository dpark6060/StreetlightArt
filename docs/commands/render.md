# `render` — Static perspective render

```bash
uv run streets-at-night render
uv run streets-at-night render --quality medium --output output/myrender.png
uv run streets-at-night render --quality high --no-lights
uv run streets-at-night render --width 4096 --height 2160 --format tiff
```

Renders a single perspective image of the city scene to disk using the Pillow backend. This is the **export path** — it produces print-quality stills. For interactive camera placement use [`preview`](preview.md). For animation use [`animate`](animate.md).

> **In the UI:** the **Render** tab of [`ui`](ui.md) offers an output picker, a
> quality dropdown, and a per-pass progress bar, then displays the finished
> image. The CLI is still the way to do scripted, headless, or `--width`/
> `--height`/`--no-lights` renders — those flags are CLI-only.

## What it produces

The full pass list (in order):

1. **Sky / ground fill + horizon glow** — a vertical gradient with a warm horizon band
2. **Stars** (if `[stars].stars_enabled = true`)
3. **Tiled ground texture** — colored zones (commercial, residential, parks, water, road surfaces)
4. **Roads** — perspective-projected lines using the `[roads.export]` palette (very dark; only visible where lit)
5. **Light glow halos** — atmospheric blooms around each light source
6. **Light illumination splats** — soft pools of color on the ground around each light
7. **Atmospheric haze + vignette** — distance-based color blending and edge darkening

The result is a single PNG (or TIFF) at the configured quality preset's resolution.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--quality` | `low` | Resolution preset: `low`, `medium`, `high` |
| `--output` | `output/render.png` | Output file path |
| `--cache-dir` | `cache` | Directory for cached OSM data and ground tiles |
| `--no-lights` | `false` | Skip illumination and light passes (much faster, no glow / pools / cars) |
| `--width` | _(preset)_ | Override output width (independent of preset) |
| `--height` | _(preset)_ | Override output height (independent of preset) |
| `--format` | _(auto)_ | Force output format: `png` or `tiff`. Auto-detected from extension otherwise. |

## Quality presets

| Preset | Resolution | Use case |
|---|---|---|
| `low` | 1920×1080 | Quick iteration |
| `medium` | 3840×2160 | High-DPI screens, social-media stills |
| `high` | 7680×4320 | Print (the artistic target) |

Presets also control internal resolutions for the illumination buffer and the ground tile pixel-density. See [`[quality]`](../config.md#qualitylow--qualitymedium--qualityhigh).

## Cache behavior

- **OSM data** — downloaded once, persisted as `.gpkg` files. See [`verify`](verify.md).
- **Ground tiles** — generated from `[zones]` config and persisted under `cache/ground_tiles_<hash>/`. Tiles regenerate when zone colors change.
- **Flat map** — not used by `render` (only by `preview` and `animate`).

A clean render of the default scene at `--quality high` takes 2–4 minutes the first time; subsequent renders with the same zone config are mostly limited by the perspective passes (~30–60 s).

## Output format

| Extension | Format | Notes |
|---|---|---|
| `.png` | PNG (8-bit RGB) | Default. Lossless, smaller files. |
| `.tiff` / `.tif` | TIFF (16-bit RGB) | Better for print pipelines that want 16-bit color depth. |

You can also force the format with `--format png` or `--format tiff` regardless of the output filename extension.

## Related config

- [`[camera]`](../config.md#camera) — viewpoint and look-at
- [`[bounds]`](../config.md#bounds) — area of OSM data to download
- [`[quality.*]`](../config.md#qualitylow--qualitymedium--qualityhigh) — resolution presets
- [`[roads.export.*]`](../config.md#roadsexportroad_type) — road colors and widths
- [`[zones]`](../config.md#zones) — ground tile colors and house placement
- [`[lights.*]`](../config.md#lights) — streetlights, cars, commercial lights
- [`[atmosphere]`](../config.md#atmosphere) — haze, scatter, horizon glow, vignette
- [`[stars]`](../config.md#stars) — optional star field
