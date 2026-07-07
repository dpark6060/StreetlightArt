# `ui` — Tabbed workbench

```bash
uv run streets-at-night ui
uv run streets-at-night ui --tab Render
uv run streets-at-night ui --cache-dir /tmp/osm_cache
```

Opens a single pygame + pygame-gui window that wraps every command as a tab —
**Verify**, **Preview**, **Render**, **Animate**, **Devmode** — behind a shared
config bar. This is the fastest way to work: load a config once, then move
between framing a shot, rendering it, tuning lights, and animating without
re-launching anything or editing TOML by hand.

Every tab operates on the same in-memory config and the same OSM/scene caches,
so a bounding box you set in Verify, a camera you frame in Preview, and lights
you tune in Devmode all feed the same Render.

The individual CLI subcommands still exist and behave exactly as before (see
their own pages); the UI is a convenience layer on top, not a replacement.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--tab` | `Verify` | Tab to open on launch: `Verify`, `Preview`, `Render`, `Animate`, `Devmode` |
| `--cache-dir` | `cache` | Directory for OSM `.gpkg` and flat-map caches |

The global `--config` flag picks the starting TOML (default: `scene_config.toml`).
If that file does not exist, the UI opens in a blank / save-as state — nothing
is written to disk until you explicitly Save.

## Window layout

Three fixed regions, top to bottom:

1. **Tab bar** — one button per tab. The current tab's button is shown
   disabled as the active indicator. **While a background job is running the
   entire tab bar is disabled** so you cannot switch away from an in-progress
   render, verify, or record and lose its result.
2. **Config bar** — a `Config:` field showing the loaded TOML path, a **Browse**
   button (opens a file dialog), and a **Load** button. Loading a config
   replaces the working config for *all* tabs and invalidates the cached scene.
   A bad or missing path is reported and ignored; it never crashes the window.
3. **Tab content** — the active tab's view (usually a render/preview area on the
   left, controls on the right).

## The working config

The UI holds one in-memory "working config" — the single source of truth every
tab reads and writes. **Nothing touches disk until you Save.** Editing bounds in
Verify, coordinates in Preview, or light values in Devmode mutates this working
config; each tab's Save (or Render, for Preview) is what persists it.

- If a config file is loaded, Save writes back to that path (patching only the
  relevant section where it can — e.g. Verify rewrites just `[bounds]`, Preview
  just `[camera]`).
- If no config is loaded, Save opens a dialog; picking a new path writes the
  full working config there and adopts it as the current file.

## Background work

Slow operations — downloading OSM data, building the scene, rendering, encoding
an MP4 — run on a single background worker so the window stays responsive. Only
**one job runs at a time**; a tab's action buttons disable while it is busy, and
(as noted above) the tab bar locks so the result can't be orphaned. Progress
shows in the tab's status label and, where relevant, a progress bar.

## The tabs

### Verify

N/S/E/W bounding-box fields (populated from the loaded config), a **Rerender**
button that downloads/caches OSM data and shows `output/verify.png` in the view,
and **Save** which writes `[bounds]` back to the config. Use this first when
changing your area. Full detail: [`verify.md`](verify.md).

### Preview

The flat-map warp preview with **explicit camera controls** in addition to the
usual WASD/mouse (which still work while the cursor is over the view):

- **Move step (m)** field + N / S / E / W / Up / Down buttons — pan the camera by
  exactly that many meters.
- **Rotate step (deg)** field + CCW / CW / Pitch+ / Pitch- buttons, plus **Reset
  rotation** (re-aims at the configured target).
- Six live coordinate fields (camera lat/lon/altitude, target lat/lon/altitude)
  that update as you move, and that you can type into directly.
- A **Render** button that applies the typed coordinates, re-renders, and writes
  `[camera]` back to the config **on every click**.

The view renders every frame the camera moves, so held-key flight is live.
Background: the scene + flat map load once (shared with the other tabs).
Related CLI: [`preview.md`](preview.md).

### Render

An **Output** file picker, a **quality** dropdown (`low` / `medium` / `high`), a
**Render** button, and a per-pass **progress bar**. The finished image is shown
letterboxed in the view and written to the chosen path (`.tiff`/`.tif` → TIFF,
else PNG). The scene is reused from cache when the quality matches the last
build; otherwise it rebuilds. Full flag/preset detail: [`render.md`](render.md).

### Animate

A **base image** picker (default `output/render.png`), a **Preview** toggle that
plays live heat-haze shimmer in the view, and **Record** controls (output path,
fps, duration, seamless toggle, crossfade) that encode an MP4 in the background
with frame-level progress. Requires `[heat_haze] enabled = true` — if it's off,
the buttons are disabled with an explanatory message. Full tuning guidance:
[`animate.md`](animate.md).

### Devmode

Live lighting/atmosphere tuning against a synthetic test scene (or a real OSM
strip via the **Real Tiles** toggle), with a dynamic property panel and a
draggable minimap. This replaced the old standalone `devmode` window. Full
detail: [`devmode.md`](devmode.md).

## Relationship to the CLI

| You want to… | UI | CLI |
|---|---|---|
| Check a bounding box | Verify tab | `verify` |
| Frame a camera | Preview tab | `preview` |
| Render a still | Render tab | `render` |
| Animate shimmer | Animate tab | `animate` |
| Tune lights/atmosphere | Devmode tab | `devmode` (also opens the UI) |
| Batch / scripted / headless render | — | `render` (use the CLI flags) |

High-resolution or scripted renders are still best driven from the `render` CLI
with explicit flags; the UI is for interactive work.

## Related config

- [`docs/config.md`](../config.md) — every config section and field
- All per-tab behavior maps onto the same `scene_config.toml` the CLI uses
