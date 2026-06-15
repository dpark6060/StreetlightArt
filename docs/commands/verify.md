# `verify` — OSM data sanity-check plot

```bash
uv run streets-at-night verify
uv run streets-at-night verify --output output/mycheck.png
uv run streets-at-night verify --cache-dir /tmp/osm_cache
```

Downloads (or loads cached) OSM road, land use, and building data for the bounding box in `[bounds]` and produces a top-down matplotlib plot showing what was loaded. **Run this first when changing your bounding box** to confirm the data covers the area you expect before kicking off a long render.

The verify pass does not run any of the rendering pipeline — it stops after the matplotlib figure is written.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--output` | `output/verify.png` | Output image path |
| `--cache-dir` | `cache` | Directory for cached OSM `.gpkg` files |

## What's in the plot

- Roads color-coded by classification (highway / major / collector / residential / service / minor)
- Land use polygons (commercial, industrial, parks/forests, water)
- Building footprints
- The configured camera bounding box overlaid as a rectangle

## Cache behavior

The first run downloads three datasets from OSM via `osmnx`:

- `roads_<hash>.gpkg`
- `land_use_<hash>.gpkg`
- `buildings_<hash>.gpkg`

The `<hash>` is derived from the bounding box, so changing `[bounds]` triggers a fresh download. Cache files are typically 30–100 MB depending on density; a fresh download for the default San Bernardino bbox takes ~30–60 seconds depending on OSM API load.

## Related config

- [`[bounds]`](../config.md#bounds) — controls what gets downloaded
- [`[camera]`](../config.md#camera) — drives the camera-frustum overlay on the plot
