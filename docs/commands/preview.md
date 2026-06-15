# `preview` — Interactive camera placement

```bash
uv run streets-at-night preview
uv run streets-at-night preview --quality medium
uv run streets-at-night preview --no-lights
```

Opens a 1280×720 pygame window with live camera navigation. **This is how you frame a shot.** Move around the scene with WASD/QE/arrows, then press `C` to write the current camera position back to `scene_config.toml`, then run [`render`](render.md) to produce a final image at full quality.

The preview uses a **flat-map warp** rather than per-segment 3D projection: a top-down rasterized road map is generated once (and cached on disk), then each frame raycasts every screen pixel through the camera and samples that flat map. This holds 60 fps even on the full San Bernardino dataset because the per-segment cost has been eliminated.

If `[heat_haze].enabled = true`, the live preview applies the same Kolmogorov heat shimmer used by [`animate`](animate.md) on top of every frame.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--quality` | `low` | Resolution preset used **only** when saving screenshots with `P`. The preview window itself is always 1280×720. |
| `--cache-dir` | `cache` | Directory for OSM cache and the flat-map cache |
| `--no-lights` | `false` | Hide light dots in the preview (faster) |

## Controls

| Key / input | Action |
|---|---|
| W / S | Move forward / backward |
| A / D | Strafe left / right |
| Q / E | Raise / lower altitude |
| ↑ / ↓ | Tilt camera up / down |
| ← / → | Pan camera left / right |
| `+` / `-` | Zoom (adjust FOV) |
| Scroll wheel | Adjust altitude |
| Right-click drag | Rotate view |
| `R` | Reset camera to default position |
| `P` | Save screenshot (uses `--quality` preset, written to `output/screenshot_<timestamp>.png`) |
| `C` | Write current camera position to `scene_config.toml` |
| `ESC` | Quit |

The HUD overlay shows FPS, segment count, current altitude, and FOV.

## How it works

1. Load OSM data, classify roads, build the Scene with spatial index.
2. Generate (or load from cache) a top-down PNG of the entire road network — the **flat map**. This is the slow step on first run; it's cached as `cache/flat_map_<hash>.png` and is invalidated only when the road set or render colors change.
3. Pygame loop:
   - If the camera moved since last frame, re-raycast every screen pixel through the camera onto the flat map → that frame's "base frame".
   - If heat haze is enabled, apply [the Kolmogorov haze pipeline](animate.md) on top of the base frame.
   - Blit, present, repeat.

A camera move triggers a full base-frame regeneration plus a depth-map recompute (if heat haze is on). On a stationary camera it's just the per-frame haze warp — typically <10 ms.

## Saving camera positions

Press `C` at any time to dump the current camera position, look-at, FOV, and altitude back to `[camera]` in your TOML file. The other sections of the file are preserved. This is the supported workflow for finding a good shot interactively and then rendering it at print quality:

```bash
uv run streets-at-night preview               # frame the shot, press C
uv run streets-at-night render --quality high # render at the saved camera
```

## Related config

- [`[camera]`](../config.md#camera) — initial position and orientation
- [`[bounds]`](../config.md#bounds) — area of OSM data
- [`[roads.preview.*]`](../config.md#roadspreviewroad_type) — bright road colors used in the preview window
- [`[heat_haze]`](../config.md#heat_haze) — see [`animate.md`](animate.md) for full tuning guidance
