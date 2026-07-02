# Streets at Night

Generates aerial perspective images of city streets at night using OpenStreetMap data. The configured target is San Bernardino, CA; any bounding box in the southern California UTM 11N zone works without code changes.

The pipeline produces:

- **Static prints** — high-resolution PNG/TIFF up to 7680×4320 with road network, zoned ground texture, streetlights/car/commercial/house lights, atmospheric haze, horizon glow, stars, and corner vignette.
- **Live preview** — interactive 1280×720 pygame window for camera placement, with optional Kolmogorov heat shimmer.
- **Animation** — multi-layer Kolmogorov heat haze applied to a static render, with depth-aware distortion and Apple-MPS-accelerated bilinear warping.

## Setup

```bash
uv sync
```

Tested on macOS / Apple Silicon. The MPS backend on PyTorch is required for real-time animation; CPU-only works but is slower.

## Quick start

```bash
# 1. Confirm your OSM bounding box
uv run streets-at-night verify

# 2. Frame a shot interactively (WASD/arrows/etc; press C to save the camera, ESC to quit)
uv run streets-at-night preview

# 3. Render a final image
uv run streets-at-night render --quality high

# 4. Animate heat shimmer over the rendered image
uv run streets-at-night animate --image output/render.png

# Tune lighting/atmosphere live with sliders, then save a config
uv run streets-at-night devmode
```

## Commands

Each subcommand has its own reference page:

| Command | Purpose | Docs |
|---|---|---|
| `verify` | Sanity-check OSM data with a top-down plot | [`docs/commands/verify.md`](docs/commands/verify.md) |
| `render` | Static perspective render to PNG/TIFF | [`docs/commands/render.md`](docs/commands/render.md) |
| `preview` | Interactive pygame camera placement | [`docs/commands/preview.md`](docs/commands/preview.md) |
| `animate` | Heat shimmer over a static render | [`docs/commands/animate.md`](docs/commands/animate.md) |
| `devmode` | Live slider tuning of lights & atmosphere | [`docs/commands/devmode.md`](docs/commands/devmode.md) |

All commands accept a global `--config` flag to point at an alternate TOML file (default: `scene_config.toml`).

## Configuration

`scene_config.toml` controls every scene parameter. The full reference (every section, every field, defaults, and examples) is at:

➡️ **[`docs/config.md`](docs/config.md)**

Heat-haze tuning has its own dedicated section in [`docs/commands/animate.md`](docs/commands/animate.md#tuning-the-look-heat_haze-config-section) — eddy size, wind vectors, layer balance, recipes.

## How it works

For curious readers there are two long-form explainers:

- **[`docs/heat_haze.html`](docs/heat_haze.html)** — physics deep-dive on the heat shimmer pipeline: Kolmogorov turbulence, Von Karman PSD, FFT-filtered noise, frozen-flow advection, MPS warp benchmarks. Diagrams, equations, the works.
- **[`docs/animation_design.md`](docs/animation_design.md)** — original architecture sketch (predates the Kolmogorov rewrite; kept for historical context).
- **[`investigation/FINDINGS.md`](investigation/FINDINGS.md)** — the prototype investigation that drove the current heat haze design.

## Road types

Roads are classified into six types from OSM `highway` tags. Both the export and preview render paths have separate per-type color/width palettes (see [`config.md`](docs/config.md#roadsexportroad_type)).

| Type | Description |
|---|---|
| Highway | Motorways and trunk roads |
| Major Road | Primary and secondary roads |
| Collector | Tertiary roads |
| Residential | Residential streets |
| Service | Service roads, alleys, driveways |
| Minor | Unclassified and other minor roads |

## Status

| Stage | Description |
|---|---|
| Stage 1 ✅ | OSM data pipeline, road classification, verification plot |
| Stage 2 ✅ | Coordinate system, camera model, perspective projection |
| Stage 3 ✅ | Pygame live preview, Scene facade, spatial index |
| Stage 4 ✅ | Land use zones, tiled ground texture, zone colors |
| Stage 5 ✅ | Streetlights, car lights, commercial lights, house lights, export glow and illumination |
| Stage 6 ✅ | Export render with glow, illumination, atmosphere, vignette, stars |
| Stage 6a ✅ | Animation: Kolmogorov heat haze (depth-aware, MPS-accelerated) |
| Stage 7 | 3D depth illusion export |
