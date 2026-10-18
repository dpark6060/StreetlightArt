# CLAUDE.md

## Product

Renders an aerial night view of San Bernardino, CA using OpenStreetMap road data, a 3D perspective camera, a linear-HDR light/atmosphere pipeline, and a flat-map warp preview. Personal art project targeting large-format print.

**What it does:**

1. **`verify`** — Downloads road, land use, and building data from OSM via `osmnx` for a bounding box (default: San Bernardino area, `scene_config.toml [bounds]`). Caches raw data as `.gpkg` files in `cache/`. Classifies roads by OSM `highway` tag into 6 `RoadType` categories. Saves a matplotlib verification plot to `output/verify.png`.
2. **`preview`** — Builds a fully populated `Scene` (roads, zones, streetlights, cars, commercial lights, and lit-window buildings; ENU meters via the origin's UTM zone; spatial index at 500m cell size). Generates a flat top-down road map (`PIL.Image`, cached as `cache/flat_map_*.png`). Runs a pygame event loop at 1280x720 where each frame raycasts every screen pixel through the camera to the z=0 ground plane and samples the flat map, then optionally applies heat-haze distortion. WASD/QE/arrows/mouse-drag for camera control. P for screenshot, R for reset, C to save the current camera position back to the TOML config, ESC to quit. `--no-lights` skips light rendering for a faster preview.
3. **`render`** — Builds the same fully populated Scene, but renders via `PillowBackend` at a configurable quality preset (low=1920x1080, medium=3840x2160, high=7680x4320, overridable with `--width`/`--height`). Runs the full linear-HDR pass pipeline (ground, roads, illumination, lights, atmosphere, compositor) with progress reporting per pass. Saves PNG or TIFF to `output/render.png` (`--format`, `--no-lights` to skip illumination/light passes).
4. **`animate`** — Animates Kolmogorov heat-haze distortion over a previously rendered still image (`output/render.png` by default). Two modes: an interactive pygame window (default, press P to save a frame, ESC to quit), or `--record FILE` which renders frames offline through `imageio`/ffmpeg to an MP4 at a chosen quality/resolution, optionally with `--seamless` looping via a crossfade tail. Requires `[heat_haze] enabled = true` in the config; the depth map used for haze distortion is rebuilt from the config camera at the target resolution.
5. **`devmode`** — Alias that opens the tabbed `ui` directly on its Devmode tab (`run_ui(..., initial_tab="Devmode")`). Keeps `--cache-dir`; the old `--real-tiles` flag is gone (the tab has a synthetic/real toggle). The standalone `cmd_devmode` event loop and hand-written control panel were removed in the Fable UI rework — the Devmode tab now owns scene rendering and panel construction.
6. **`ui`** — Launches the tabbed pygame + pygame-gui window (`ui/app.py`, `run_ui`) that wraps every command as a tab (Verify, Preview, Render, Animate, Devmode) behind a shared config bar. `--tab` picks the tab to open on launch (default `Verify`); `--cache-dir` sets the OSM cache directory. Tabs run work off the main thread via `ui/jobs.py` (`JobRunner`) and share an `AppState` (`ui/app_state.py`) holding the working `SceneConfig`, its path, and the cache dir.

7. **`tune`** — Opens a separate real-time GPU window (`gpu/tune_app.py`, wgpu on Metal + rendercanvas/glfw + imgui) with a human-scale scene: an E-W road, a small shop with lit windows/sign/storefronts, a 4.7 m car, a 1.75 m person, and 4 poles at the selected fixture's sampled spacing. Panel tabs: Road / Commercial (streetlight pools, parking — picking one mounts it on the poles — head/taillights), Building (windows, signs, storefronts), Scene (illumination brightness, knee, grade, glow shape). Drag orbits, shift+drag pans, scroll zooms, R resets. Save writes via `config_writer`. The light math is a WGSL port of the CPU passes; `tests/test_gpu_parity.py` renders on the GPU and compares against the numpy functions (skipped without a wgpu adapter).

**Key constraints:**

- Camera defaults: position at 34.23°N 117.31°W at 5186ft, target at 34.11°N 117.30°W at 1000ft, 60° FOV, near=100m, far=60000m
- Coordinate system is ENU (East-North-Up) in meters, origin at camera/target midpoint
- UTM zone is derived from the coordinate origin longitude/latitude (`get_utm_epsg` in `camera/coordinates.py`), so any city projects correctly; the zone EPSG is exposed as `CoordinateTransformer.utm_epsg`
- Road classification drops footways, cycleways, paths, pedestrian, tracks, steps, bridleway
- Flat map capped at 8000px in either dimension; resolution auto-scales coarser if exceeded
- Zones, streetlights, cars, commercial lights, atmosphere, and heat haze are all implemented (not stubs); houses render as extruded building geometry with lit windows (`models/building.py`, `placement/building_builder.py`, `rendering/building_pass.py`) rather than point lights, with a single-row window fallback for one-story heights and a residential lit fraction of 0.4 for house/detached/residential kinds; `zone_builder.py` builds residential/commercial/dark zones from OSM land use + building footprints, `RoadSegment.fixture_profile` drives per-segment streetlight sampling
- Export lighting is a linear-HDR pipeline: light contributions accumulate unclamped in a float32 linear buffer (`PillowBackend._light`), and a single compositor pass at the end applies exposure, ACES tone mapping, and sRGB encoding — intermediate passes never clamp
- Atmosphere includes airlight/extinction: the same `falloff_distance` extinction coefficient that dims lights with range also drives a depth-based airlight veil in `atmosphere_pass.py`
- Streetlight ground pools can be "physical" (derived from pole height + `beam_angle_deg`, when both are set) or a legacy authored two-zone cone (inner flat zone + falloff curve), selected per-fixture in `illumination_pass.py`
- Heat haze (`rendering/heat_haze.py`) is MPS-accelerated via `torch.nn.functional.grid_sample`; it falls back to whatever `torch` device is available if MPS isn't present (see `select_torch_device`)
- Devmode config edits only persist when explicitly saved via its Save button; the CLI's other commands always read `scene_config.toml` (or `--config`) directly
- The `ui` and `devmode` subcommands both dispatch to `run_ui`; `main()` passes `config_path=None` when the `--config` file does not exist so the UI opens in a blank/save-as state
- The UI holds one in-memory "working config" (`AppState.config`) shared by every tab; edits mutate it and only reach disk on a tab's Save/Render. Slow work (OSM load, verify, render, MP4 record) runs on a single background `JobRunner` — one job at a time, and the tab bar is disabled while a job runs so a result can't be delivered to the wrong tab

## Structure

```
scene_config.toml          # Camera, bounds, quality presets, lights, zones, atmosphere, heat_haze, devmode
pyproject.toml             # uv/hatchling build config
cache/                     # OSM .gpkg caches, flat map PNG/JSON cache, ground tile cache
output/                    # Rendered images, screenshots, recorded animations
src/streets_at_night/
  __main__.py              # python -m entry, calls main()
  main.py                  # CLI: verify/preview/render/animate/ui/devmode subcommands, scene loading
  scene_loader.py          # load_scene / bounds_to_bbox — OSM→Scene assembly shared by CLI + UI
  config/
    schema.py              # Dataclasses: SceneConfig and all nested config sections
    defaults.py            # DEFAULTS dict — base config values
    loader.py              # load_config() — TOML → SceneConfig with deep merge
    layers.py              # Place/look layering: `look = "..."` merges DEFAULTS ← look ← place; split_for_save routes saves back
    serde.py               # dataclass_from_dict / dataclass_to_dict reflection walkers
    camera_io.py           # save_camera_position — write camera pose back to a TOML config
    paths.py               # resolve_cache_dir and other path helpers
  models/
    road_types.py          # RoadType enum (HIGHWAY..MINOR)
    road_segment.py        # RoadSegment dataclass — geometry, type, width, fixture_profile
    light_source.py        # LightSource dataclass — universal light model consumed by renderer
    streetlight.py         # FixtureProfile, FixturePool, StreetLight factory
    car.py                 # CarLightPair — paired headlight/taillight set for one car
    house.py                # House data model — footprint and colors
    building.py              # Building dataclass — extruded footprint geometry for mass/window rendering
    commercial.py            # Factory functions for commercial zone light sources
    zones.py                # ResidentialZone, CommercialZone, DarkZone data containers
    color_pool.py            # Weighted random color selection from a named palette
    range_param.py           # Numeric parameter with configurable min/max + distribution
  data/
    osm_loader.py           # load_roads/land_use/buildings — osmnx download + .gpkg cache
    road_classifier.py      # classify_roads — OSM highway tag → RoadType
    verify_plot.py          # save_verification_plot — matplotlib dark-theme figure
    zone_builder.py         # build_zones — OSM land use/buildings → residential/commercial/dark zones
  camera/
    camera.py               # Camera class — position/target/FOV, view+projection matrices
    coordinates.py          # CoordinateTransformer — WGS84↔ENU via pyproj, UTM zone from origin
    projection.py           # project_points (vectorized), frustum_cull_segments, near-plane clip
    controls.py             # handle_input — pygame WASD/mouse/scroll → camera mutations, HUD
  scene/
    scene.py                # Scene facade — road_segments + zones + lights + camera + spatial index
    spatial_index.py        # SpatialIndex — uniform grid, 500m cells, O(1) insert/query
    ground_tiles.py         # Tile-based ground color cache for the export ground pass
  placement/
    scene_builder.py        # build_scene / populate_lights — orchestrates all placers into a Scene
    streetlight_placer.py   # Place streetlights along road segments using fixture profiles
    car_placer.py            # Place CarLightPair instances along road segments
    commercial_placer.py     # Signs, storefronts, and parking lot lights in commercial zones
    building_builder.py      # Build Building objects from OSM footprints + residential house lots
    geometry_utils.py        # Shared 2D geometry helpers (perpendicular_2d, etc.) for placers
  rendering/
    backend.py               # RenderBackend ABC — draw_line/draw_polygon/get_image/add_linear/etc.
    pillow_backend.py        # PillowBackend — sRGB substrate + float32 linear light buffer for export
    pygame_backend.py        # PygameBackend — pygame.Surface for interactive preview
    render_pipeline.py       # render_scene_frame(_with_progress), get_preview_passes(_warp), get_export_passes
    color_space.py            # sRGB<->linear conversions, ACES tone mapping
    raycast.py                # build_ground_gbuffer — shared per-pixel ground raycast (G-buffer)
    render_utils.py           # Shared helpers: brightness falloff, camera distance, horizon calc
    flat_map.py               # FlatMapMeta, generate_flat_map, get_flat_map (with disk cache)
    warp_preview.py           # Per-pixel raycast preview — _build_ray_directions, _raycast_to_flat_map
    road_pass.py              # render_roads_from_scene / render_roads_export — road projection + drawing
    ground_pass.py            # render_ground / render_ground_export — sky/ground fill, tiled zone colors
    illumination_pass.py      # render_illumination — world-space ground light pools, raycast-sampled
    light_pass.py             # render_lights_preview / render_lights_export — cores + glow halos
    building_pass.py          # Building mass render — extruded walls/roofs + lit windows (additive linear)
    atmosphere_pass.py        # render_atmosphere — horizon glow + depth-based airlight veil
    compositor.py             # render_composite — stars, vignette, exposure, tone-map, sRGB encode
    heat_haze.py               # HeatHaze — multi-layer Kolmogorov phase-screen distortion (torch/MPS)
    animate_service.py         # build_animate_haze / load_animate_base_frame / record_haze[_seamless] — heat-haze animation helpers shared by CLI + Animate tab
  ui/
    app.py                     # UIApp + run_ui — window, tab bar, shared config bar, event loop
    app_state.py               # AppState — working SceneConfig, config_path, cache_dir
    jobs.py                    # JobRunner — runs tab work off-thread, polled per frame
    imaging.py                 # compute_viewport_fit + PIL↔pygame surface helpers (was in cmd_devmode)
    tabs/
      base.py                  # Tab ABC — build/destroy/update/draw/handle_event/refresh_from_state
      verify_tab.py            # Verify tab — bounds entry + OSM verification plot
      preview_tab.py           # Preview tab — interactive warp preview + camera controls
      render_tab.py            # Render tab — quality/output controls + export render
      animate_tab.py           # Animate tab — heat-haze preview/record over a still
      devmode_tab.py           # Devmode tab — synthetic/real scene, dynamic light-kind panel, draggable minimap
    widgets/
      file_picker.py           # FilePickerRow — text entry + browse dialog
      labeled_entry.py         # LabeledEntry — label + text entry row
  gpu/                         # wgpu renderer — currently the `tune` scene; grows into the city renderer
    orbit_camera.py            # OrbitCamera — orbit/pan/zoom around a ground pivot → camera.Camera
    tuning_scene.py            # build_tuning_scene — road, shop, car, person, poles; lights via real factories
    gpu_scene.py               # Pack geometry/pool lights/sprites/uniforms into float32 arrays for upload
    renderer.py                # TuningRenderer — HDR scene pass, sprite pass (source-occlusion), composite
    tuning_view.py             # TuningView — scene + renderer glue (window and headless renders)
    field_editor.py            # imgui slider rows for a LightKind (reuses devmode field_meta/light_catalog)
    tune_app.py                # run_tune — glfw window, imgui panel, input, Save
    shaders/*.wgsl             # common, illumination (falloff ports), geometry, sprites, composite
  devmode/
    synthetic_scene.py        # Builds the 12-row synthetic ENU test scene (incl. a Houses row of buildings and two parking-lot cluster rows)
    real_scene.py              # Loads a clipped strip of real OSM road data for devmode
    devmode_camera.py          # Builds an ENU-space Camera from DevmodeCameraConfig
    dynamic_panel.py           # build_dynamic_section — data-driven slider panel from field metadata
    field_meta.py              # Per-field slider ranges for the dynamic panel; widens to contain the current value (never clamps)
    light_catalog.py           # Enumerates the light config fields the devmode panel exposes
    render_helpers.py          # should_render_full / build_render_passes / render_devmode_scene (was in cmd_devmode)
    widget_sync.py             # Paired slider/entry/toggle update handlers used by the Devmode tab
    minimap.py                 # Top-down minimap overlay for the devmode scene
    panel_sections.py          # build_atmosphere_section + build_devmode_camera_section (light-section builders removed in change 09)
    slider_registry.py         # Maps pygame-gui slider widgets to SceneConfig field paths
    config_bridge.py           # Two-way sync between SceneConfig and slider widget values
    config_writer.py           # Serializes SceneConfig back to a TOML-compatible dict and writes it
    label_pass.py              # Render pass that draws row identification labels onto the scene
  utils/
    colors.py                  # Shared color utilities (hex_to_rgb, kelvin_to_rgb, etc.)
tests/
  conftest.py                  # Shared fixtures
  test_*.py                    # One test file per source module (see tests/ directory listing)
```

**Data flow (render):**

```
main.py:main()
  -> load_config(toml_path)                      # TOML → SceneConfig
  -> cmd_render(config, quality, ...)
       -> _load_scene(config, quality, cache_dir)
            |-- load_roads/land_use/buildings(bbox, cache_dir)  # osmnx download or .gpkg cache
            |-- classify_roads(roads_raw)          # highway tag → RoadType column
            |-- create_transformer(origin_lat/lon) # pyproj UTM transformer (zone from origin)
            |-- create_camera(pos, target, ...)    # Camera with view/proj matrices
            `-- build_scene(roads, transformer, camera, config, land_use, buildings)
                 |-- convert_road_geometry()       # WGS84 LineStrings → ENU (N,3) arrays
                 |-- build_zones()                 # land use + buildings → residential/commercial/dark zones
                 |-- _build_buildings_if_configured() # OSM footprints + residential lots → Building list
                 |-- scene.build_spatial_index()   # uniform grid, 500m cells
                 `-- populate_lights()             # streetlight/car/commercial placers → scene lights
       -> get_export_passes(cache_dir, no_lights, quality_preset)
       -> render_scene_frame_with_progress(scene, PillowBackend, passes)
            |-- render_ground_export      # sky/ground fill, tiled per-zone colors
            |-- render_roads_export       # per-segment 3D projection + frustum culling
            |-- render_buildings          # extruded walls/roofs (opaque) + lit windows (additive linear)
            |-- render_illumination       # world-space light-pool map, raycast-sampled, additive linear
            |-- render_lights_export      # bright cores + gaussian glow halos, additive linear
            |-- render_atmosphere         # horizon glow + airlight veil, additive linear
            `-- render_composite          # resolve_linear() -> exposure -> ACES tonemap -> sRGB -> paste
       -> image.save(output_path)
```

**Data flow (animate --record):**

```
cmd_animate_record(args, config, heat_haze_cfg)
  |-- _load_animate_base_frame(image_path, w, h)   # load + resize a pre-rendered PNG/TIFF
  |-- _build_animate_haze(config, ...)             # build Camera, compute_depth_map, construct HeatHaze
  `-- _record_haze[_seamless](haze, fps, n_frames, out_path)
       -> haze.apply(t) per frame                  # phase-screen advection -> torch grid_sample warp
       -> imageio writer.append_data(frame)         # streamed to ffmpeg/libx264 MP4
```

**Conventions:**

- All config dataclasses in `config/schema.py`; defaults in `config/defaults.py`
- `RoadType` enum centralizes road categories; `_HIGHWAY_TO_ROAD_TYPE` dict in `road_classifier.py` maps OSM tags
- Render passes are `Callable[[Scene, Camera, RenderBackend], Any]` — pluggable via lists in `render_pipeline.py`
- `RenderBackend` ABC has two implementations; preview uses `blit_pil_image`, export uses `draw_line`/`add_linear`/`resolve_linear`
- Export passes share a single per-frame ground raycast G-buffer (`raycast.build_ground_gbuffer`) rather than recomputing it per pass
- All light-emitting passes (illumination, lights, atmosphere) write into the backend's linear buffer via `add_linear` — never draw light directly into the sRGB substrate; only the compositor tone-maps and encodes
- Coordinate conversion always flows WGS84 → origin's UTM zone → ENU offset; `CoordinateTransformer` encapsulates origin and zone
- Devmode config edits go through `config_bridge`/`slider_registry`, which map slider widgets to dataclass field paths on `SceneConfig` by tuple path (string = attribute, int = list index)

## Tech

- **Language**: Python >=3.11
- **Package manager**: `uv`
  - Install: `uv sync`
  - Run: `uv run streets-at-night <command>`
  - Test: `uv run pytest tests/`
- **Build system**: hatchling; package name `streets-at-night`

**Runtime dependencies:**

| Package     | Constraint | Purpose                                    |
|-------------|------------|--------------------------------------------|
| osmnx       | >=1.9      | Download OSM road/building/land use data   |
| geopandas   | >=0.14     | GeoDataFrame for spatial data              |
| shapely     | >=2.0      | Geometry operations on OSM features        |
| numpy       | >=1.26     | Array math, vectorized raycast             |
| matplotlib  | >=3.8      | Verification plot rendering                |
| pyproj      | >=3.6      | WGS84 ↔ UTM coordinate transformation     |
| tomli-w     | >=1.0      | TOML writing (config round-trip, devmode save) |
| pygame      | >=2.5      | Interactive preview/devmode window + input |
| pygame-gui  | >=0.6      | Devmode control panel widgets              |
| Pillow      | >=10.0     | Flat map generation, warp output, export   |
| torch       | >=2.1      | MPS/CPU-accelerated heat-haze frame warping (`grid_sample`) |
| imageio[ffmpeg] | >=2.34 | MP4 encoding for `animate --record`        |
| wgpu        | >=0.32     | GPU rendering (Metal) for `tune`           |
| rendercanvas, glfw | >=2.7, >=2.11 | Window + event loop for `tune`  |
| imgui-bundle | >=1.92.0,<1.92.900 | `tune` control panel. Capped: 1.92.900 dropped `ImDrawData.cmd_lists_count`, which wgpu 0.32's imgui backend uses |

**Dev dependencies:**

| Package | Constraint | Purpose    |
|---------|------------|------------|
| pytest  | >=7.0      | Test runner|

**Test patterns**: pytest with `unittest.mock`. Fixtures at file top. Naming: `test_method_scenario_behavior`. One file per source module. `spec=True` on mocks. `MagicMock()` for Scene (no spec). Tests prefixed with `# Radical`.
