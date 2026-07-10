# `scene_config.toml` reference

Every parameter that controls scene generation, rendering, and animation lives in `scene_config.toml` at the project root. The file is **optional** — if it doesn't exist, all defaults are used. You only need to specify values you want to override; everything else falls back to built-in defaults defined in `src/streets_at_night/config/defaults.py`.

Deep merge applies at every nesting level, so you can override a single sub-key without repeating the whole section. The exception is **list-valued** keys (e.g. `[[heat_haze.layers]]`) — providing any layer entries replaces the entire default list.

## Sections

- [`[camera]`](#camera)
- [`[bounds]`](#bounds)
- [`[quality.low]` / `[quality.medium]` / `[quality.high]`](#qualitylow--qualitymedium--qualityhigh)
- [`seed`](#seed)
- [`[roads.export.<road_type>]`](#roadsexportroad_type)
- [`[roads.preview.<road_type>]`](#roadspreviewroad_type)
- [`[zones.colors]`](#zonescolors)
- [`[zones.house]`](#zoneshouse)
- [`[zones.house.exterior_colors]`](#zoneshouseexterior_colors)
- [`[zones]` tile settings](#zones-tile-settings)
- [`[lights.streetlights]`](#lightsstreetlights)
- [Light Rendering: glow_radius vs illumination_radius](#light-rendering-glow_radius-vs-illumination_radius)
- [`[lights.cars]`](#lightscars)
- [`[lights.commercial]`](#lightscommercial)
- [`[atmosphere]`](#atmosphere)
- [`[stars]`](#stars)
- [`[heat_haze]`](#heat_haze)
- [`[devmode]`](#devmode)
- [Examples](#examples)

---

## `[camera]`

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

Used by: [`render`](commands/render.md), [`preview`](commands/preview.md), [`animate`](commands/animate.md).

## `[bounds]`

Bounding box for the OSM data download. The larger this is, the longer the initial download and the more segments in the scene.

| Key | Type | Default | Description |
|---|---|---|---|
| `north` | float | 34.35 | Northern edge in degrees |
| `south` | float | 33.95 | Southern edge in degrees |
| `east` | float | -117.05 | Eastern edge in degrees |
| `west` | float | -117.55 | Western edge in degrees |

Used by: [`verify`](commands/verify.md), [`render`](commands/render.md), [`preview`](commands/preview.md).

## `[quality.low]` / `[quality.medium]` / `[quality.high]`

Output resolution for static renders. Also controls the internal illumination resolution and ground tile detail.

| Key | Type | Default (low / medium / high) | Description |
|---|---|---|---|
| `width` | int | 1920 / 3840 / 7680 | Output width in pixels |
| `height` | int | 1080 / 2160 / 4320 | Output height in pixels |
| `illumination_resolution_multiplier` | float | 0.5 / 0.75 / 1.0 | Scale factor for the light buffer's resolution |
| `ground_tile_resolution` | float | 4.0 / 2.0 / 1.0 | Meters per pixel for ground tiles (lower = finer) |

Used by: [`render`](commands/render.md), [`preview`](commands/preview.md) (for screenshots).

## `seed`

Top-level integer seed for reproducible procedural generation (house colors, light placement, star field). Default: `42`.

## `[roads.export.<road_type>]`

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

## `[roads.preview.<road_type>]`

Road line colors and widths for the **interactive preview**. Same structure as export, but defaults are bright colors for easy navigation.

| Road Type | Color | Width |
|---|---|---|
| `highway` | `"#FFD700"` (gold) | 4 |
| `major_road` | `"#FF6B35"` (orange) | 3 |
| `collector` | `"#FFFFFF"` (white) | 2 |
| `residential` | `"#888888"` (gray) | 1 |
| `service` | `"#555555"` (dark gray) | 1 |
| `minor` | `"#64647A"` (slate) | 1 |

## `[zones.colors]`

Base fill colors for each zone type. These represent very dark night ambient light — in production use they are nearly black. Set bright colors here when debugging zone boundaries.

| Key | Default | Controls |
|---|---|---|
| `commercial` | `"#0A0A0C"` | Commercial and retail areas |
| `industrial` | `"#080808"` | Industrial areas |
| `park_forest` | `"#040804"` | Parks, gardens, forests |
| `water` | `"#040408"` | Water bodies |
| `road_surface` | `"#0C0C0C"` | Road surface fill in ground tiles |
| `default_ground` | `"#080706"` | Unzoned areas |

## `[zones.house]`

Parameters for synthetic house generation within residential zones.

| Key | Type | Default | Description |
|---|---|---|---|
| `yard_base_color` | str | `"#060A04"` | Base hex color for yard fill |
| `yard_variation` | int | 2 | Per-channel RGB noise added to yard color |
| `footprint_coverage` | float | 0.65 | Fraction of lot area covered by house footprint |
| `lot_width` | float | 15.0 | Synthetic lot width in meters |
| `lot_depth` | float | 25.0 | Synthetic lot depth in meters |
| `setback` | float | 10.0 | Distance from road edge to house front in meters |

## `[zones.house.exterior_colors]`

Weighted color pool for house exterior color selection. Each entry is a `["#hexcolor", weight]` pair. Higher weight = more likely to be picked.

```toml
[zones.house.exterior_colors]
colors = [
    ["#0D0A08", 1.0],
    ["#0C0907", 1.0],
    ["#0B0808", 1.0],
]
```

## `[zones]` tile settings

| Key | Type | Default | Description |
|---|---|---|---|
| `tile_size_meters` | float | 256.0 | World-space size of each ground tile in meters |
| `tile_resolution_px` | int | 128 | Pixel resolution per tile |

## `[lights.streetlights]`

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

## Light Rendering: glow_radius vs illumination_radius

Every light has two independent visual effects that run in separate render passes.

**`glow_radius`** — the atmospheric halo around the light source itself. Drawn in `light_pass.py` as concentric rings of decreasing opacity that approximate a Gaussian falloff. Measured in world meters; projected to screen pixels via `glow_radius / camera_distance × focal_length`. Glow radius expands with camera distance using the atmospheric scatter formula (`glow_radius × (1 + distance / scatter_distance)`), so distant lights appear hazier and bloom wider. This is the bright orb visible around each light source.

**`illumination_radius`** — the soft pool of colored light falling on the ground around the source. Drawn in `illumination_pass.py` as concentric rings on a separate RGBA layer, then Gaussian-blurred before compositing. Also measured in world meters and perspective-projected, but with a fixed 4× screen scale factor applied before blurring — so pools appear noticeably wider and softer than halos. Illumination radius does **not** expand with camera distance. Pool intensity is controlled independently by `illumination_intensity` (0–1).

**How they interact:** the two passes are completely independent. Glow is "light in the air"; illumination is "light on the ground." Zeroing out either has no effect on the other. Both composite additively on top of the ground layer, so overlapping pools and halos accumulate naturally.

**Quick tuning:**

| Goal | What to change |
|---|---|
| Larger bright orb | Increase `glow_radius` |
| Larger / brighter ground pool | Increase `illumination_radius` or `illumination_intensity` |
| Distant lights bloom more | Decrease `scatter_distance` in `[atmosphere]` |
| Keep orb, remove ground pool | Set `illumination_intensity = 0.0` |

## `[lights.cars]`

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

> **Note:** Car color pools and separation are parsed from TOML but not yet wired to the runtime placement code — the placer currently uses hardcoded values. Car density IS config-driven.

## `[lights.commercial]`

Controls lights placed in commercial zones: signs, parking lots, and storefronts. Each sub-type has a color pool, brightness range, glow radius, and illumination radius.

**Sign lights:**

| Key | Type | Default | Description |
|---|---|---|---|
| `sign_brightness_min` / `sign_brightness_max` | float | 0.6 / 0.9 | Brightness range |
| `sign_glow_radius` | float | 12.0 | Glow halo radius in meters |
| `sign_illumination_radius` | float | 20.0 | Ground illumination radius in meters |

Default sign color pool: 35% warm white, 12% red, 12% blue, 10% green, 10% yellow, 8% purple, 8% orange, 5% pink.

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

> **Note:** Commercial light configs are parsed from TOML but not yet wired to the runtime placement code — factories currently use hardcoded values.

## `[atmosphere]`

Controls distance-based color blending, horizon glow, atmospheric haze, and the corner vignette in the export render.

| Key | Type | Default | Description |
|---|---|---|---|
| `falloff_distance` | float | 8000.0 | Distance (m) at which lights begin to dim noticeably |
| `haze_color` | str | `"#3A2010"` | Tint color blended with distant lights and ground |
| `scatter_distance` | float | 10000.0 | Distance (m) at which atmospheric scatter doubles the glow radius |
| `max_distance` | float | 20000.0 | Distance (m) at which lights fade out completely |
| `glow_expansion_enabled` | bool | `false` | Enable distance-based glow radius expansion |
| `glow_max_alpha` | int | 80 | Max alpha (0–255) for the additional glow halo |
| `glow_falloff_steepness` | float | 1.0 | Steepness of the alpha rolloff for the additional glow |
| `min_brightness_threshold` | float | 0.05 | Below this brightness a light is skipped entirely |
| `illumination_brightness` | int | 60 | Default brightness scalar (0–255) for illumination splats |
| `illumination_min_radius` | float | 15.0 | Minimum ground illumination radius (px) regardless of distance |
| `horizon_glow_color` | str | `"#28180A"` | Color of the warm band along the horizon |
| `horizon_glow_width_px` | int | 120 | Vertical extent of the horizon glow band |
| `horizon_glow_peak_alpha` | int | 90 | Maximum alpha of the horizon glow |
| `haze_enabled` | bool | `true` | Master switch for the distant-haze pass |
| `haze_alpha` | int | 30 | Maximum alpha of the haze tint |
| `vignette_strength` | float | 0.35 | Corner darkening intensity (0 = none) |
| `vignette_radius` | float | 0.7 | Inner radius (fraction of screen) where vignette starts |

## `[stars]`

Optional star field rendered into the sky region of the export. Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `stars_enabled` | bool | `false` | Master switch |
| `count` | int | 200 | Number of stars |
| `brightness_min` / `brightness_max` | float | 0.03 / 0.15 | Brightness range (0–1) |
| `color_variation` | int | 15 | Per-channel RGB jitter on the white base |
| `radius` | float | 1.0 | Star sprite radius in pixels |

## `[heat_haze]`

Controls the multi-layer Kolmogorov heat shimmer used by [`preview`](commands/preview.md) and [`animate`](commands/animate.md).

This section has its own dedicated tuning guide — see **[`commands/animate.md`](commands/animate.md#tuning-the-look-heat_haze-config-section)** for full discussion of eddy size, wind direction, layer balance, and recipe presets. The raw fields:

**Top-level scalars:**

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master on/off switch. `false` = no distortion at all. |
| `amplitude_px` | float | `0.5` | **Overall shimmer intensity.** How many pixels the image wavers at the horizon (defined at 1920 px wide; scales automatically at higher resolutions). Turn this up for more dramatic heat, down for subtler shimmer. Values above ~2 start to look like a bad TV signal. |
| `max_distance_m` | float | `20000.0` | **How far away "full shimmer" is.** Distortion ramps up with distance and saturates at this point. Should match `[atmosphere].max_distance` so the most-distorted ground is also the most haze-faded. |
| `distance_exponent` | float | `1.5` | **How quickly shimmer fades toward the camera.** `1.0` = linear falloff (foreground shimmers noticeably). `2.0` = quadratic (only the far distance shimmers; foreground is calm). `0.5` = very gentle falloff (shimmer spreads across the whole frame). |
| `sky_distance_factor` | float | `0.5` | **How much the sky/stars shimmer relative to the horizon ground.** `0.0` = sky is locked (no shimmer). `1.0` = sky shimmers as hard as the farthest ground. `0.5` means sky gets `0.5^exponent ≈ 35%` of the maximum distortion — visible but quieter than the horizon. |
| `seed` | int | `42` | Random seed. Change this to get a different shimmer pattern; same value always produces the same result. |
| `screen_width_px` | int | `2048` | Internal phase-screen width. Larger = longer before the horizontal pattern repeats. Rarely needs changing unless you see a visible horizontal cycle. |
| `screen_height_px` | int | `4096` | Internal phase-screen height. Larger = longer before the vertical (rising-heat) pattern repeats. Increase if you see a repeating loop in long animations. |

**Per-layer settings** (one or more `[[heat_haze.layers]]` entries):

| Key | Type | Description |
|---|---|---|
| `r0` | float | Fried parameter (m) — turbulence strength |
| `L0` | float | Outer scale (m) — largest eddy size |
| `pixel_scale_m` | float | Phase-plane meters per pixel |
| `weight` | float | Mix coefficient |
| `wind_x_pxs` | float | Horizontal advection (px/s); positive = features move LEFT |
| `wind_y_pxs` | float | Vertical advection (px/s); positive = features move UP |

**Eddy size in pixels = `L0 / pixel_scale_m`**. Keep this well below the image dimensions or the whole frame bulges as one eddy. See [`commands/animate.md`](commands/animate.md) for the full guide.

## `[devmode]`

Configuration for the [Devmode tab](commands/devmode.md) (the live lighting/atmosphere tuning view in the [`ui`](commands/ui.md)). These fields set the devmode scene's render resolution and its camera; the production `render`/`preview` commands ignore them.

| Key | Type | Default | Description |
|---|---|---|---|
| `render_width` | int | 896 | Devmode render width |
| `render_height` | int | 720 | Devmode render height |
| `[devmode.camera]` | table | _(see source)_ | ENU-space camera with explicit pos/target/fov fields |

---

## Examples

### Minimal override

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

### Lock the sky during animation

Stars stay perfectly still while the ground shimmers:

```toml
[heat_haze]
sky_distance_factor = 0.0
```

### Aggressive heat shimmer

Doubled amplitude, faster rise:

```toml
[heat_haze]
amplitude_px = 1.5

[[heat_haze.layers]]
r0 = 0.10
L0 = 2.5
pixel_scale_m = 0.5
weight = 0.6
wind_x_pxs = 8.0
wind_y_pxs = 70.0
# ... matching y-doubled values for the other two layers
```
