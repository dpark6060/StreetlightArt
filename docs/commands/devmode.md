# `devmode` — Interactive parameter tuning

```bash
uv run streets-at-night devmode
uv run streets-at-night devmode --real-tiles
```

Opens a 1280×720 window split into a **render view** (left) and a **control panel** (right). **This is how you dial in the look** — lighting colors, glow, ground pools, atmosphere — without editing `scene_config.toml` by hand and re-running `render` each time. Drag sliders, click **Re-render**, and when it looks right click **Save Config** to write a TOML you can hand to [`render`](render.md).

By default it renders a compact **synthetic scene**: rows of representative roads (one per road type) plus cars, commercial, and house lights, so every light category is on screen at once for fast comparison. Pass `--real-tiles` to tune against an actual strip of OSM road data instead.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--real-tiles` | `false` | Tune against a real OSM road strip instead of the synthetic test scene |
| `--cache-dir` | `cache` | Directory for OSM cache files (only used with `--real-tiles`) |

The global `--config` flag picks the starting TOML (default: `scene_config.toml`).

## Layout

The control panel has three zones, top to bottom:

1. **Section list** — pick which parameter group to edit.
2. **Sliders** — every tunable in the selected section, each with a live value box you can also type into directly.
3. **Action buttons** — `Re-render`, `Save Config`, `Load Config`.

### Sections

| Section | Controls |
|---|---|
| Atmosphere | Extinction/airlight coefficient (`falloff_distance`), airlight strength (`haze_alpha`) and color, the glow core + power-law-wing knobs (`glow_wing_exponent`, `glow_tail_strength`, `glow_cutoff_mult`), illumination brightness, vignette |
| Streetlights: Highway … Service | Per-road-type fixture profile: **color temperature** (`color_temp_k`) + **saturation**, **beam angle** (drives the ground pool from pole height), glow radius, illumination intensity, brightness/height/spacing ranges |
| Cars | Headlight / taillight brightness, glow, and ground-pool cone |
| House Lights | Window/porch brightness, glow, ground-pool cone |
| Sign / Parking / Storefront Lights | Commercial light brightness, glow, ground-pool cone |
| Devmode Camera | Position, target, and FOV of the devmode camera (ENU meters) |

> Streetlight ground pools are derived physically from **pole height + beam angle**, so they have a `beam_angle_deg` slider instead of an explicit radius. Cars, signs, and house lights still use the authored cone model (`illumination_radius` / `falloff` / `inner_fraction`), so those sliders remain.

### Camera navigation

The same keys as [`preview`](preview.md) move the camera in the render view; a camera move auto-re-renders:

| Key / input | Action |
|---|---|
| W / S / A / D | Move forward / back / strafe |
| Q / E | Raise / lower altitude |
| ↑ / ↓ / ← / → | Tilt and pan |
| Right-click drag | Rotate view |
| `R` | Reset camera |
| `ESC` | Quit |

Screenshot (`P`) and save-camera (`C`) are **not** wired in devmode — use **Save Config** to persist your work, and the **Devmode Camera** section to set the camera numerically.

## Workflow

1. Pick a section, drag sliders (or type exact values in the value boxes).
2. Click **Re-render**. This applies every widget value to the in-memory config, **rebuilds the scene**, and renders. Rebuilding is required because most light parameters (color, beam angle, height, spacing) are baked in when lights are placed — moving a slider alone does nothing until you re-render.
3. Repeat until it looks right.
4. Click **Save Config**, enter a filename (default `scene_config_devmode.toml`), and it writes a full TOML.
5. Render it at print quality:

```bash
uv run streets-at-night render --config scene_config_devmode.toml --quality high
```

**Load Config** reads an existing TOML back into the panel, so you can iterate on a saved file across sessions.

## Notes

- Saved TOMLs are complete round-trips: color temperature, saturation, beam angle, the glow wing knobs, the `color_temp_shift_k` master lever, and all atmosphere fields are preserved.
- Adjusting a streetlight `color_temp_k` / `saturation` slider writes that value to **every** fixture in that road type, collapsing its sodium/LED mix to one temperature.
- `color_temp_shift_k` is a global additive Kelvin offset applied to *all* streetlights at once — the "retrofit the whole city from sodium to LED in one number" lever. It is intentionally **not** a per-section slider; set it directly in `[lights.streetlights]` of your TOML when you want a global shift on top of the per-type colors.

## Related config

- [`[lights.streetlights]`](../config.md) — fixture pools, color temperature, beam angle, master shift
- [`[atmosphere]`](../config.md) — extinction, airlight, glow wings, vignette
- [`[devmode]`](../config.md) — devmode camera defaults
