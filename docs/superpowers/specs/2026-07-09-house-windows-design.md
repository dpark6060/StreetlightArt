# House Windows: light houses via building windows, remove the house-light point system

**Date:** 2026-07-09
**Status:** Approved

## Problem

Devmode exposes two overlapping ways to edit residential lighting: "House Lights"
(the legacy ground-level point-light system) and "Buildings: Windows" (the emissive
window grid on extruded building geometry). Houses already render as building
geometry — `placement/building_builder.py` wraps every house footprint into a
`Building` — but they never get windows: `_get_window_row_count` computes
`floor((height − sill_margin − roof_margin) / floor_height_m)`, and with defaults
(house height 4.5m, sill 0.9m, roof 0.6m, floor 3.2m) that is `floor(3.0 / 3.2) = 0`
rows. Any building under ~4.7m renders as a dark box, so houses are lit only by the
legacy point lights.

Decision: houses are lit by real windows on their geometry; the house-light point
system is removed entirely (including its ground illumination pools — yards go dark
by choice; `window_glow_px` and a possible future window ground-spill pass are the
levers if suburbs later feel too black).

## Changes

### 1. Single-row window fallback (`rendering/building_pass.py`)

In `_get_window_row_count`, when the standard formula yields 0 rows but a window
physically fits in the vertical band (`usable >= config.window_height_m`), return 1.

- `usable = height − window_roof_margin_m − window_sill_margin_m` (unchanged).
- No change to `_get_window_row_centers`: its centering math already places a single
  row at the center of the usable band (verified: on a 4.5m house the row center
  lands at 2.4m; window top 2.95m is below the 3.9m roofline).
- Applies to every building kind, garages included. Walls whose usable band is
  smaller than one window (roughly under 2.6m tall at defaults) still get 0 rows.

### 2. Residential lit fraction

- Add `window_lit_fraction_residential: float` to `BuildingConfig`
  (`config/schema.py`) with default `0.4` in `config/defaults.py`.
- In the window lit-test in `building_pass.py`, select the fraction by building
  kind: kinds in `{"house", "detached", "residential"}` use
  `window_lit_fraction_residential`; all others keep `window_lit_fraction`
  (default 0.12). Synthetic-lot houses are tagged `"house"`, so they're covered.
- Devmode: the existing "Buildings: Windows" catalog entry uses the `window_`
  field prefix, so the new field appears automatically; add a `field_meta` slider
  range of 0.0–1.0.

### 3. Remove the house-light point system

- Delete `models/house_light.py` and `placement/house_light_placer.py`, plus their
  test files.
- `models/house.py`: remove `num_lights`, `light_positions`, `get_lights`,
  `_pick_num_lights`, `_compute_light_positions`, and the module-level perimeter
  helpers (`_place_along_perimeter`, `_inset_toward_centroid`). `House` keeps
  `footprint`, `base_color`, and `yard_color` — ground tiles still consume them.
- `config/schema.py`: remove `HouseLightConfig` and `LightConfig.house_lights`;
  remove `min_lights`, `max_lights`, `light_weights` from `HouseConfig`.
  `config/defaults.py`: remove the matching default entries. The serde layer
  (`_generic_from_dict`) only reads keys that match schema fields, so old TOMLs
  containing `[lights.house_lights]` still load without error.
- `placement/scene_builder.py`: remove `_run_house_light_placer`, the house child
  RNG, the `house_lights` element of `populate_lights`' return, and the summary
  line.
- `devmode/light_catalog.py`: remove the "House Lights" `LightKind`.
- `devmode/synthetic_scene.py`: remove the "House Lights" row and its light
  creation; the existing "Houses" buildings row demonstrates the window fallback.
- Remove stale `[lights.house_lights]` (and `[lights.house_lights.colors]`)
  sections from `scene_config.toml` and `scene_config_NewDevmode.toml`, and the
  now-removed house `min_lights`/`max_lights`/`light_weights` keys if present.
- Update CLAUDE.md structure/notes that reference `house_light.py`,
  `house_light_placer.py`, and house lights.

## Testing

- New: `_get_window_row_count` fallback — returns 1 when 0 rows but a window fits;
  returns 0 when the usable band is smaller than `window_height_m`; unchanged for
  multi-row heights.
- New: lit-fraction selection — residential kinds use the residential fraction,
  other kinds use the global one.
- Update/remove tests touching `House` lights, `house_light_placer`,
  `scene_builder` house wiring, `light_catalog`, and `synthetic_scene`.
- Run the full suite (`uv run pytest tests/`) and report pass/fail counts.

## Out of scope

- Window ground-spill illumination (windows lighting the yard) — possible
  follow-up, not part of this change.
- Any change to commercial lights, streetlights, or car lights.
