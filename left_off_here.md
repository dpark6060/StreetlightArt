# Where We Left Off — 2026-03-24

## Session Summary

Fixed the road color issue: export roads were drawn with bright debug colors
(gold highways, orange major roads, white collectors) making every road visible
regardless of lighting. Replaced with a config-driven system — export now uses
dark asphalt tones (`#141210` down to `#0A0908`) so roads are only visible where
illuminated by lights. Preview keeps bright colors for navigation. Both palettes
are fully TOML-configurable via `[roads.export]` and `[roads.preview]`.

Also updated the README with comprehensive documentation of all TOML parameters,
including the commercial, house, and car light config sections that were
previously undocumented.

Started at 400 tests, ended at 414.

## Current State

- **Stages 1–5**: Complete.
- **Stage 6** (atmosphere): Not started.
- **Stage 7** (Blender export): Not started.

## What Works

- `uv run streets-at-night preview` — interactive preview with bright road colors,
  zone colors (small zones only — clipping bug), and light dots. Press C to save
  camera. WASD/mouse to fly.
- `uv run streets-at-night render --quality low` — export with dark asphalt roads,
  ground tiles, light glow halos, and ground illumination splats. Roads now only
  appear where lights illuminate them.
- All 414 tests pass.
- Road colors fully TOML-configurable (separate export and preview palettes).
- Streetlight fixture pools fully config-driven.
- Car density config-driven.

## What Changed This Session

- `config/schema.py` — Added `RoadStyleEntry`, `RoadStyleConfig`, `RoadsConfig`
  dataclasses. Added `roads` field to `SceneConfig`.
- `config/defaults.py` — Added `roads` section with dark export and bright preview
  defaults.
- `config/loader.py` — Added `_build_roads_config`, `_build_road_style_config`,
  `_build_road_style_entry` builders.
- `rendering/road_pass.py` — New `render_roads_export` (reads `scene.config.roads.export`).
  `render_roads_from_scene` now reads `scene.config.roads.preview`. Both share
  `_render_roads` internal. Removed dead `_draw_scene_road_segment` and
  `_draw_road_segment`. Added `_build_style_table` to convert config to render
  style table.
- `rendering/render_pipeline.py` — Export passes now use `render_roads_export`
  instead of `render_roads_from_scene`.
- `tests/test_road_pass.py` — New file, 10 tests for style table building, export
  rendering, fallback behavior.
- `tests/test_config_loader.py` — 4 new tests for road config loading and partial
  TOML overrides.
- 4 existing test files updated to include `roads` param in `SceneConfig` fixtures.
- `README.md` — Full config reference for all TOML parameters.

## Known Issues (consolidated from stage4_deferred.md and stage5_deferred.md)

### Must fix before Stage 6

1. **Polygon vertex clipping** — `_project_polygon` in `ground_pass.py` drops
   off-screen vertices instead of clipping. Large zones invisible in preview at
   oblique camera angles. Needs Sutherland-Hodgman clipping. Export path
   (tile-based) is unaffected.

### Should fix soon (config passthrough)

These all have the same pattern: TOML config is parsed into typed dataclasses,
but the runtime code uses hardcoded values instead of reading from config.

2. **Commercial light factories** — `create_sign_light`, `create_parking_lot_light`,
   `create_storefront_light` receive `{}` instead of `CommercialLightConfig`.
3. **House light factory** — `create_house_light` receives `{}` instead of
   `HouseLightConfig`.
4. **Car color/brightness** — `CarLightPair.__init__` uses hardcoded color pools.
   Car density IS config-driven. Separation range is hardcoded.

Impact: you can't tune these light types via `scene_config.toml` yet. Streetlight
fixture pools and road colors ARE fully config-driven.

### Can defer

5. **Atmospheric perspective** — distant lights render at full brightness. Needs
   distance-based dimming, diffusion, and warm color shift. Natural fit for Stage 6.
6. **Export tile sampling performance** — per-pixel Python loop. Vectorize with numpy
   if high-res export is too slow.
7. **Architecture plan stale** — `.claude/work/architecture-plan.md` has outdated
   signatures for zone_builder, ground_tiles, scene_builder.
8. **Parking lot light height** — placed at z=0, spec says 8-12m poles.
9. **`populate_lights` prints to console** — should be caller's responsibility.

## Design Decisions Made (not written elsewhere)

All decisions from the 2026-03-23 session still apply (see git history or
stage5_deferred.md). New this session:

- **Separate export vs preview road styles**: Export uses dark asphalt
  (`#141210`–`#0A0908`), preview uses bright navigation colors
  (`#FFD700`–`#64647A`). Both configurable via TOML. The render pipeline calls
  `render_roads_export` for export and `render_roads_from_scene` for preview.

- **Fallback style tables**: `road_pass.py` has hardcoded `_FALLBACK_EXPORT_STYLE`
  and `_FALLBACK_PREVIEW_STYLE` in case `scene.config.roads` is missing (e.g.,
  tests with mock scenes that don't set the attribute). Normal code path always
  reads from config.

- **Debug zone colors removed from config**: The `[zones.colors]` overrides in
  `scene_config.toml` are now commented out. The default near-black values in
  `defaults.py` are used for production renders.

## Next Session: Recommended Plan

1. **Add `--no-lights` CLI flag** — Skip light/illumination passes for debugging
   road and ground appearance in isolation.

2. **Fix config passthrough** (#2–4 above) — Wire `CommercialLightConfig`,
   `HouseLightConfig`, and `CarConfig` color/brightness fields through to the
   runtime factories. Same pattern as the streetlight config fix. Unlocks TOML
   tuning for all light types.

3. **Run a render and evaluate** — `uv run streets-at-night render --quality low`.
   Look at whether the dark asphalt roads + illumination interaction looks right.
   Adjust road export colors in TOML if needed.

4. **Atmospheric dimming (Stage 6 entry)** — Distance-based brightness falloff +
   warm color shift + glow expansion for distant lights. Highest-impact visual
   change remaining. Makes the aerial illusion convincing.

5. **Polygon clipping** (#1) — Sutherland-Hodgman for preview zone visibility.
   Needed if evaluating atmospheric effects in the interactive preview.

Defer GPU/parallelization until a high-quality export proves too slow.
