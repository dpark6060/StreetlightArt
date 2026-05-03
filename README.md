# Streets at Night

Generates perspective images of city streets at night as seen from altitude, using real OpenStreetMap data. Currently renders road networks with type-based coloring; land use zones, building footprints, lighting, and glow effects are coming in later stages.

---

## Setup

```bash
uv sync
```

---

## Commands

All commands accept a global `--config` flag to specify an alternate config file (default: `scene_config.toml`).

### verify

Downloads (or loads cached) OSM data and produces a top-down matplotlib map showing road types and land use. Run this first to confirm your bounding box and data before rendering.

```bash
uv run python -m streets_at_night verify
uv run python -m streets_at_night verify --output output/mycheck.png
uv run python -m streets_at_night verify --cache-dir /tmp/osm_cache
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `output/verify.png` | Output image path |
| `--cache-dir` | `cache` | Directory for cached OSM files |

---

### render

Renders a static perspective image to disk using Pillow. Output includes: zone-colored tiled ground texture, road lines with type-based coloring, light glow halos around each light source, and soft ground illumination splats. The ground tiles are cached in `cache/` keyed to your zone color config — changing zone colors will generate a new tile set automatically.

```bash
uv run python -m streets_at_night render
uv run python -m streets_at_night render --quality medium --output output/myrender.png
```

| Flag | Default | Description |
|---|---|---|
| `--quality` | `low` | Resolution preset: `low`, `medium`, `high` |
| `--output` | `output/render.png` | Output image path |
| `--cache-dir` | `cache` | Directory for cached OSM files |

Quality presets: `low` = 1920×1080, `medium` = 3840×2160, `high` = 7680×4320

---

### preview

Opens an interactive pygame window at 1280×720 for live camera navigation. The HUD shows FPS, segment count, altitude, and FOV. The preview renders roads with type-based coloring, zone ground fill (small zones only — large zones have a known polygon clipping limitation), and colored light dots for streetlights, car headlights, and taillights.

```bash
uv run python -m streets_at_night preview
uv run python -m streets_at_night preview --quality medium
```

| Flag | Default | Description |
|---|---|---|
| `--quality` | `low` | Preset used when saving screenshots with P |
| `--cache-dir` | `cache` | Directory for cached OSM files |

---

## Preview Controls

| Input | Action |
|---|---|
| W / S | Move forward / backward |
| A / D | Strafe left / right |
| Q / E | Raise / lower altitude |
| ↑ / ↓ | Tilt camera up / down |
| ← / → | Pan camera left / right |
| + / - | Zoom (adjust FOV) |
| Scroll wheel | Adjust altitude |
| Right-click drag | Rotate view |
| R | Reset camera to default position |
| P | Export screenshot (saved to `output/`) |
| C | Save current camera position to `scene_config.toml` |
| ESC | Quit |

---

## Configuration

`scene_config.toml` at the project root controls all scene parameters. The file is optional — if it doesn't exist, all defaults are used. You only need to specify values you want to override; everything else falls back to built-in defaults. Deep merge applies at every nesting level, so you can override a single sub-key without repeating the whole section.

### `[camera]`

Camera position and orientation in WGS84.

| Key | Type | Default | Description |
|---|---|---|---|
| `lat` | float | 34.2349 | Camera latitude |
| `lon` | float | -117.3116 | Camera longitude |
| `altitude_ft` | float | 5186.0 | Camera height in feet |
| `target_lat` | float | 34.1083 | Look-at point latitude |
| `target_lon` | float | -117.2968 | Look-at point longitude |
| `target_altitude_ft` | float | 1000.0 | Look-at point height in feet |
| `fov` | float | 60.0 | Field of view in degrees |
| `near` | float | 100.0 | Near clipping plane in meters |
| `far` | float | 60000.0 | Far clipping plane in meters |

### `[bounds]`

Bounding box for the OSM data download. The larger this is, the longer the initial download and the more segments in the scene.

| Key | Type | Default | Description |
|---|---|---|---|
| `north` | float | 34.35 | Northern edge in degrees |
| `south` | float | 33.95 | Southern edge in degrees |
| `east` | float | -117.05 | Eastern edge in degrees |
| `west` | float | -117.55 | Western edge in degrees |

### `[quality.low]` / `[quality.medium]` / `[quality.high]`

Output resolution for static renders. Also controls the internal illumination resolution and ground tile detail.

| Key | Type | Default (low / medium / high) | Description |
|---|---|---|---|
| `width` | int | 1920 / 3840 / 7680 | Output width in pixels |
| `height` | int | 1080 / 2160 / 4320 | Output height in pixels |
| `illumination_resolution_multiplier` | float | 0.5 / 0.75 / 1.0 | Scale factor for light map resolution |
| `ground_tile_resolution` | float | 4.0 / 2.0 / 1.0 | Meters per pixel for ground tiles (lower = finer) |

### `seed`

Integer seed for reproducible procedural generation (house colors, light placement). Default: `42`.

### `[roads.export.<road_type>]`

Road line colors and widths for the **export** render path. Defaults are very dark asphalt tones — roads are barely visible without illumination, but slightly brighter than the surrounding ground (`#080706`), so they pop where streetlights and car headlights illuminate them.

Each road type (`highway`, `major_road`, `collector`, `residential`, `service`, `minor`) has two fields:

| Field | Type | Description |
|---|---|---|
| `color` | str | Hex color of the road line |
| `width` | int | Base line width in pixels (scaled by perspective distance) |

Defaults:

| Road Type | Color | Width |
|---|---|---|
| `highway` | `"#141210"` | 4 |
| `major_road` | `"#121010"` | 3 |
| `collector` | `"#100E0E"` | 2 |
| `residential` | `"#0E0C0C"` | 1 |
| `service` | `"#0C0A0A"` | 1 |
| `minor` | `"#0A0908"` | 1 |

Example — make highways brighter in export:

```toml
[roads.export.highway]
color = "#1E1A16"
width = 5
```

### `[roads.preview.<road_type>]`

Road line colors and widths for the **interactive preview**. Same structure as export, but defaults are bright colors for easy navigation.

| Road Type | Color | Width |
|---|---|---|
| `highway` | `"#FFD700"` (gold) | 4 |
| `major_road` | `"#FF6B35"` (orange) | 3 |
| `collector` | `"#FFFFFF"` (white) | 2 |
| `residential` | `"#888888"` (gray) | 1 |
| `service` | `"#555555"` (dark gray) | 1 |
| `minor` | `"#64647A"` (slate) | 1 |

### `[zones.colors]`

Base fill colors for each zone type. These represent very dark night ambient light — in production use they are nearly black. Set bright colors here when debugging zone boundaries.

| Key | Default | Controls |
|---|---|---|
| `commercial` | `"#0A0A0C"` | Commercial and retail areas |
| `industrial` | `"#080808"` | Industrial areas |
| `park_forest` | `"#040804"` | Parks, gardens, forests |
| `water` | `"#040408"` | Water bodies |
| `road_surface` | `"#0C0C0C"` | Road surface fill in ground tiles |
| `default_ground` | `"#080706"` | Unzoned areas |

### `[zones.house]`

Parameters for synthetic house generation within residential zones.

| Key | Type | Default | Description |
|---|---|---|---|
| `yard_base_color` | str | `"#060A04"` | Base hex color for yard fill |
| `yard_variation` | int | 2 | Per-channel RGB noise added to yard color |
| `min_lights` | int | 0 | Minimum lights per house |
| `max_lights` | int | 5 | Maximum lights per house |
| `light_weights` | list[float] | `[5.0, 3.0, 1.5, 0.5, 0.3, 0.1]` | Probability weights for 0–5 lights |
| `footprint_coverage` | float | 0.65 | Fraction of lot area covered by house footprint |
| `lot_width` | float | 15.0 | Synthetic lot width in meters |
| `lot_depth` | float | 25.0 | Synthetic lot depth in meters |
| `setback` | float | 10.0 | Distance from road edge to house front in meters |

### `[zones.house.exterior_colors]`

Weighted color pool for house exterior color selection. Each entry is a `["#hexcolor", weight]` pair. Higher weight = more likely to be picked.

```toml
[zones.house.exterior_colors]
colors = [
    ["#0D0A08", 1.0],
    ["#0C0907", 1.0],
    ["#0B0808", 1.0],
]
```

### `[zones]` tile settings

| Key | Type | Default | Description |
|---|---|---|---|
| `tile_size_meters` | float | 256.0 | World-space size of each ground tile in meters |
| `tile_resolution_px` | int | 128 | Pixel resolution per tile |

### `[lights.streetlights]`

Streetlight fixtures per road type. Each road type (`highway`, `major_road`, `collector`, `residential`, `service`) has a pool of weighted fixture entries. When placing lights along a segment, one entry is drawn from the pool by weight — the selected fixture type is used for the entire segment. A `profile = null` entry (i.e., `weight` with no profile) represents no lights on that segment.

Each fixture profile has these fields:

| Field | Type | Description |
|---|---|---|
| `weight` | float | Relative probability this fixture type is selected for a segment |
| `color` | str | Hex color of the light point (the glow dot) |
| `height_min` / `height_max` | float | Pole height range in meters |
| `brightness_min` / `brightness_max` | float | Brightness scalar range (0–1) |
| `glow_radius` | float | Radius of the light glow halo in meters |
| `illumination_radius` | float | Radius of ground illumination splat in meters |
| `illumination_intensity` | float | Opacity of the ground illumination splat (0–1) |
| `spacing_min` / `spacing_max` | float | Distance between lights along the segment in meters |
| `side` | str | Placement side: `"both"`, `"alternating"`, or `"one"` |

To override a road type's fixture pool, replace the entire pool for that road type. The example below replaces highway lights with a single fixture type (cool LED white):

```toml
[[lights.streetlights.highway]]
weight = 1.0

[lights.streetlights.highway.profile]
color = "#E8F0FF"
height_min = 12.0
height_max = 15.0
brightness_min = 0.85
brightness_max = 1.0
glow_radius = 25.0
illumination_radius = 40.0
illumination_intensity = 0.9
spacing_min = 40.0
spacing_max = 60.0
side = "both"
```

### Light Rendering: glow_radius vs illumination_radius

Every light has two independent visual effects that run in separate render passes.

**`glow_radius`** — the atmospheric halo around the light source itself. Drawn in `light_pass.py` as concentric rings of decreasing opacity that approximate a gaussian falloff. Measured in world meters; projected to screen pixels via `glow_radius / camera_distance × focal_length`. Glow radius expands with camera distance using the atmospheric scatter formula (`glow_radius × (1 + distance / scatter_distance)`), so distant lights appear hazier and bloom wider. This is the bright orb visible around each light source.

**`illumination_radius`** — the soft pool of colored light falling on the ground around the source. Drawn in `illumination_pass.py` as concentric rings on a separate RGBA layer, then gaussian-blurred before compositing. Also measured in world meters and perspective-projected, but with a fixed 4× screen scale factor applied before blurring — so pools appear noticeably wider and softer than halos. Illumination radius does **not** expand with camera distance. Pool intensity is controlled independently by `illumination_intensity` (0–1).

**How they interact:** the two passes are completely independent. Glow is "light in the air"; illumination is "light on the ground." Zeroing out either has no effect on the other. Both composite additively on top of the ground layer, so overlapping pools and halos accumulate naturally.

**Quick tuning:**

| Goal | What to change |
|---|---|
| Larger bright orb | Increase `glow_radius` |
| Larger / brighter ground pool | Increase `illumination_radius` or `illumination_intensity` |
| Distant lights bloom more | Decrease `scatter_distance` in `[atmosphere]` |
| Keep orb, remove ground pool | Set `illumination_intensity = 0.0` |

---

### `[lights.cars]`

Controls car density, headlight/taillight separation, and color pools per road type.

| Key | Type | Default | Description |
|---|---|---|---|
| `separation_min` / `separation_max` | float | 1.6 / 2.4 | Distance between headlight pair and taillight pair in meters |
| `density_highway_min` / `density_highway_max` | float | 25.0 / 50.0 | Spacing between cars on highways in meters |
| `density_major_min` / `density_major_max` | float | 40.0 / 80.0 | Spacing on major roads |
| `density_collector_min` / `density_collector_max` | float | 80.0 / 150.0 | Spacing on collector roads |
| `density_residential_min` / `density_residential_max` | float | 150.0 / 400.0 | Spacing on residential roads |
| `density_service_min` / `density_service_max` | float | 300.0 / 800.0 | Spacing on service roads |

Higher spacing values = fewer cars. The actual spacing for each segment is drawn uniformly from the min/max range.

### `[lights.cars.headlight_colors]` / `[lights.cars.taillight_colors]`

Weighted color pools for car headlights and taillights. Each entry is `["#hexcolor", weight]`.

Default headlight pool: 50% bright white, 30% warm white, 15% yellow, 5% blue-white HID.

Default taillight pool: 60% red, 25% dark red, 15% red-orange.

```toml
[lights.cars.headlight_colors]
colors = [
    ["#FFFFFF", 0.5],
    ["#FFF4E0", 0.3],
    ["#FFEE80", 0.15],
    ["#CCE0FF", 0.05],
]

[lights.cars.taillight_colors]
colors = [
    ["#FF2020", 0.6],
    ["#CC1010", 0.25],
    ["#FF3010", 0.15],
]
```

> **Note:** Car color pools and separation are parsed from TOML but not yet wired to the runtime placement code — the placer currently uses hardcoded values. Car density IS config-driven. Full passthrough is tracked in `stage5_deferred.md` item #8.

### `[lights.commercial]`

Controls lights placed in commercial zones: signs, parking lots, and storefronts. Each sub-type has a color pool, brightness range, glow radius, and illumination radius.

**Sign lights:**

| Key | Type | Default | Description |
|---|---|---|---|
| `sign_brightness_min` / `sign_brightness_max` | float | 0.6 / 0.9 | Brightness range |
| `sign_glow_radius` | float | 12.0 | Glow halo radius in meters |
| `sign_illumination_radius` | float | 20.0 | Ground illumination radius in meters |

Default sign color pool: 35% warm white, 12% red, 12% blue, 10% green, 10% yellow, 8% purple, 8% orange, 5% pink.

```toml
[lights.commercial.sign_colors]
colors = [
    ["#FFF8E0", 0.35],
    ["#FF2020", 0.12],
    ["#2040FF", 0.12],
]
```

**Parking lot lights:**

| Key | Type | Default | Description |
|---|---|---|---|
| `parking_brightness_min` / `parking_brightness_max` | float | 0.75 / 0.9 | Brightness range |
| `parking_glow_radius` | float | 20.0 | Glow halo radius in meters |
| `parking_illumination_radius` | float | 30.0 | Ground illumination radius in meters |
| `parking_grid_spacing_min` / `parking_grid_spacing_max` | float | 20.0 / 30.0 | Grid spacing between lights in meters |

Default parking color pool: 70% LED white, 30% sodium.

**Storefront lights:**

| Key | Type | Default | Description |
|---|---|---|---|
| `storefront_brightness_min` / `storefront_brightness_max` | float | 0.5 / 0.7 | Brightness range |
| `storefront_glow_radius` | float | 8.0 | Glow halo radius in meters |
| `storefront_illumination_radius` | float | 12.0 | Ground illumination radius in meters |
| `storefront_spacing_min` / `storefront_spacing_max` | float | 5.0 / 8.0 | Spacing between storefront lights along building edge in meters |

Default storefront color pool: 60% warm white, 25% warm yellow, 15% fluorescent.

> **Note:** Commercial light configs are parsed from TOML but not yet wired to the runtime placement code — factories currently use hardcoded values. Full passthrough is tracked in `stage5_deferred.md` item #6.

### `[lights.house_lights]`

Controls lights placed in residential zone houses (window, porch, interior).

| Key | Type | Default | Description |
|---|---|---|---|
| `brightness_min` / `brightness_max` | float | 0.3 / 0.6 | Brightness range |
| `glow_radius` | float | 6.0 | Glow halo radius in meters |
| `illumination_radius` | float | 8.0 | Ground illumination radius in meters |
| `illumination_intensity` | float | 0.35 | Opacity of ground illumination splat (0–1) |

Default color pool: 45% warm yellow, 25% warm white, 15% amber, 15% TV blue.

```toml
[lights.house_lights.colors]
colors = [
    ["#FFE4A0", 0.45],
    ["#FFF0D0", 0.25],
    ["#FFCC80", 0.15],
    ["#D0E0FF", 0.15],
]
```

> **Note:** House light configs are parsed from TOML but not yet wired to the runtime placement code — the factory currently uses hardcoded values. Full passthrough is tracked in `stage5_deferred.md` item #7.

### `[heat_haze]`

Controls the atmospheric thermal ripple animation applied in the interactive preview. Heat haze is a post-process pixel shift applied each frame on top of a cached base frame — it does not affect the export render path.

The effect simulates the way hot air distorts light near a city at night: lights near the horizon shimmer and waver, while lights close to the camera are mostly still.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable or disable the effect entirely |
| `amplitude` | float | `3.0` | Maximum pixel displacement at the horizon. Controls how violently the image shimmers. At `3.0`, horizon features shift by ~1–3 pixels per frame. Values above `6.0` start to look like a bad TV signal rather than heat. |
| `freq1` | float | `0.003` | Primary spatial frequency of the wave pattern in cycles per pixel. Lower values produce broad, slow undulations; higher values produce tighter, faster ripples. |
| `freq2` | float | `0.007` | Secondary spatial frequency, layered on top of `freq1` to break up regularity. Should be roughly 2× `freq1` for a natural look. |
| `speed1` | float | `0.4` | Animation speed of the primary wave in radians per second. At `0.4`, a full shimmer cycle takes ~15 seconds. |
| `speed2` | float | `0.6` | Animation speed of the secondary wave. Setting it to a non-integer multiple of `speed1` prevents the two waves from phase-locking into a repeating pattern. |
| `distance_exponent` | float | `1.5` | Controls how steeply the distortion fades from the horizon toward the foreground. `1.0` = linear falloff; `2.0` = quadratic (much less distortion near the camera). Values below `1.0` spread the effect toward the foreground. |

**Visual intuition:**

| Goal | What to change |
|---|---|
| More dramatic shimmer at the horizon | Increase `amplitude` |
| Slower, lazier waves | Decrease `speed1` and `speed2` |
| Tighter, finer ripple texture | Increase `freq1` and `freq2` |
| Distortion only near the horizon | Increase `distance_exponent` (try `2.5`) |
| Distortion across more of the frame | Decrease `distance_exponent` (try `0.8`) |
| Disable without removing from config | Set `enabled = false` |

**Example — subtle shimmer:**
```toml
[heat_haze]
enabled = true
amplitude = 1.5
freq1 = 0.002
freq2 = 0.005
speed1 = 0.25
speed2 = 0.4
distance_exponent = 2.0
```

**Example — aggressive heat distortion:**
```toml
[heat_haze]
enabled = true
amplitude = 6.0
freq1 = 0.005
freq2 = 0.011
speed1 = 0.7
speed2 = 1.1
distance_exponent = 1.0
```

---

### Minimal override example

Only override the camera altitude and bounds — everything else stays at defaults:

```toml
[camera]
altitude_ft = 8000.0

[bounds]
north = 34.20
south = 34.05
east = -117.20
west = -117.40
```

### Tune road visibility for export

Make highway roads slightly brighter so they stand out more under illumination:

```toml
[roads.export.highway]
color = "#1E1A16"
width = 5
```

### Debug zone colors

Use obviously distinct colors to verify zone classification before tuning final aesthetics:

```toml
[zones.colors]
commercial = "#FF0000"    # red
industrial = "#FF8800"    # orange
park_forest = "#00FF00"   # green
water = "#0000FF"         # blue
road_surface = "#FF00FF"  # magenta
default_ground = "#222222"
```

---

## Road Types

Roads are classified into six types from OSM `highway` tags. In the **export** render, roads use dark asphalt colors and are only visible where illuminated by lights. In the **preview**, roads use bright colors for navigation. Both palettes are configurable via `[roads.export]` and `[roads.preview]` in the TOML.

| Type | Description |
|---|---|
| Highway | Motorways and trunk roads |
| Major Road | Primary and secondary roads |
| Collector | Tertiary roads |
| Residential | Residential streets |
| Service | Service roads, alleys, driveways |
| Minor | Unclassified and other minor roads |

---

## Status

| Stage | Description |
|---|---|
| Stage 1 ✅ | OSM data pipeline, road classification, verification plot |
| Stage 2 ✅ | Coordinate system, camera model, perspective projection |
| Stage 3 ✅ | Pygame live preview, Scene facade, spatial index |
| Stage 4 ✅ | Land use zones, tiled ground texture, zone colors |
| Stage 5 ✅ | Streetlights, car lights, commercial lights, house lights, export glow and illumination |
| Stage 6 ✅ | Export render with glow, illumination, atmosphere, vignette, stars |
| Stage 6a 🔄 | Animation: heat haze (config + apply_heat_haze done; moving cars pending) |
| Stage 7 | 3D depth illusion export |
