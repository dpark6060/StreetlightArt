# Window atmosphere + illumination soft knee

**Date:** 2026-07-10
**Status:** Approved (user delegated execution end-to-end)

## Problem

1. Building windows skip both atmospheric treatments every other light gets:
   no distance extinction (`_get_window_brightness` is a pure random draw) and
   no glow (`window_glow_px` defaults to 0.0). Result: razor-sharp, uniformly
   bright pinpoints at any distance.
2. Ground illumination pools accumulate additively and unclamped; a dozen
   overlapping parking-lot pools sum to a washed-out gray puddle. User wants
   stacked lights to still read brighter than one — tunably — not clamped to
   a single-light look.

## Changes

### 1. Window extinction + glow default

- New `BuildingConfig.window_extinction_strength: float` (default 1.0).
  In the window accumulation path (`building_pass.py`), window brightness is
  multiplied by `exp(-distance_m * strength / falloff_distance)` where
  `falloff_distance` is `scene.config.atmosphere.falloff_distance` — the same
  extinction law `light_pass.py` applies to every light core. The distance is
  the per-building camera distance already available in
  `_accumulate_building_windows`. `strength = 0.0` reproduces today's
  behavior exactly; 1.0 is full parity with other lights.
- `window_glow_px` default changes 0.0 → 1.2 (`config/defaults.py`); the
  glow blur path (`_apply_window_glow`) already exists and is already a
  devmode slider.
- Devmode: `window_extinction_strength` appears automatically in the
  "Buildings: Windows" section via the `window_` prefix. `field_meta` needs a
  range rule only if no existing substring rule matches ("strength" has no
  rule today — add `("extinction_strength", FieldMeta(0.0, 2.0))`).

### 2. Illumination map soft knee

- New dataclass `IlluminationConfig` in `config/schema.py`:
  `knee_start: float` (default 1.5), `knee_max: float` (default 4.0), in
  linear illumination-map intensity units. New field
  `LightConfig.illumination: IlluminationConfig`; defaults block
  `[lights.illumination]` in `config/defaults.py`. Old TOMLs without the
  section load via the defaults merge.
- Knee math, applied element-wise to the illumination map once, after all
  pools are painted (end of the map-build step in `illumination_pass.py`),
  operating on the map's luminance scale per channel:

  ```
  disabled (identity) when knee_max <= knee_start
  y = x                                          for x <= knee_start
  y = knee_start + span * over / (over + span)   for x >  knee_start
      where over = x - knee_start, span = knee_max - knee_start
  ```

  Properties: identity below `knee_start`; smooth (C1) at the knee; brighter
  stacks are always brighter (monotonic); asymptote at `knee_max`. Vectorized
  numpy, no per-pixel Python.
- Devmode: new `LightKind("Illumination: Ground Pools",
  config.lights.illumination, None)` in `light_catalog.py` (placed after the
  commercial kinds). `field_meta`: add `("knee", FieldMeta(0.0, 10.0))`.

## Testing

- Window extinction: brightness scales by the expected exponential for a
  known distance/falloff; strength 0 reproduces the pre-change brightness;
  strength appears in the Buildings: Windows catalog-driven panel (existing
  dynamic-panel tests cover field discovery generically — a targeted test
  asserts the field is on BuildingConfig and field_meta returns 0–2).
- Knee: parametrized input→output table for the curve (below start identity,
  monotonic above, asymptote below knee_max, disabled when max <= start);
  integration test that a map with stacked pools compresses while a
  single-pool region is untouched.
- Config round-trip: old TOML without `[lights.illumination]` loads with
  defaults; full suite green.

## Out of scope

- Any change to parking light placement/density.
- Exposure / falloff_distance value changes (user tunes those in TOML/devmode).
- Knee for the screen-space light-core/glow buffer (pools only).
