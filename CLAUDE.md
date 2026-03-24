# CLAUDE.md

## Product

Renders an aerial night view of San Bernardino, CA using OpenStreetMap road data, a 3D perspective camera, and a flat-map warp preview. Personal art project targeting large-format print.

**What it does:**

1. **`verify`** — Downloads road, land use, and building data from OSM via `osmnx` for a bounding box (default: San Bernardino area, `scene_config.toml [bounds]`). Caches raw data as `.gpkg` files in `cache/`. Classifies roads by OSM `highway` tag into 6 `RoadType` categories. Saves a matplotlib verification plot to `output/verify.png`.
2. **`preview`** — Builds a `Scene` (roads converted to ENU meters via UTM Zone 11N, spatial index built at 500m cell size). Generates a flat top-down road map (`PIL.Image`, cached as `cache/flat_map_*.png`). Runs a pygame event loop at 1280x720 where each frame raycasts every screen pixel through the camera to the z=0 ground plane and samples the flat map. WASD/QE/arrows/mouse-drag for camera control. P for screenshot, R for reset, ESC to quit.
3. **`render`** — Builds the same Scene, but renders via `PillowBackend` at a configurable quality preset (low=1920x1080, medium=3840x2160, high=7680x4320). Uses per-segment 3D projection with frustum culling and near-plane clipping. Saves PNG to `output/render.png`.

**Key constraints:**

- Camera defaults: position at 34.23°N 117.31°W at 5186ft, target at 34.11°N 117.30°W at 1000ft, 60° FOV, near=100m, far=60000m
- Coordinate system is ENU (East-North-Up) in meters, origin at camera/target midpoint
- UTM zone is hardcoded to 11N (EPSG:32611) — only correct for Southern California
- Road classification drops footways, cycleways, paths, pedestrian, tracks, steps, bridleway
- Flat map capped at 8000px in either dimension; resolution auto-scales coarser if exceeded
- Stages 4-7 (zones, lights, atmosphere, Blender export) are planned but not yet implemented; `zone_builder.py` and `RoadSegment.fixture_profile` are stubs

## Structure

```
scene_config.toml          # Camera, bounds, quality presets
pyproject.toml             # uv/hatchling build config
cache/                     # OSM .gpkg caches + flat map PNG/JSON cache
output/                    # Rendered images, screenshots
src/streets_at_night/
  __main__.py              # python -m entry, calls main()
  main.py                  # CLI: verify/preview/render subcommands, scene loading
  config/
    schema.py              # Dataclasses: SceneConfig, CameraConfig, QualityPreset, BoundsConfig
    defaults.py            # DEFAULTS dict — base config values
    loader.py              # load_config() — TOML → SceneConfig with deep merge
  models/
    road_types.py          # RoadType enum (HIGHWAY..MINOR)
    road_segment.py        # RoadSegment dataclass, default widths
  data/
    osm_loader.py          # load_roads/land_use/buildings — osmnx download + .gpkg cache
    road_classifier.py     # classify_roads — OSM highway tag → RoadType
    verify_plot.py         # save_verification_plot — matplotlib dark-theme figure
    zone_builder.py        # Stub — returns empty lists (Stage 4)
  camera/
    camera.py              # Camera class — position/target/FOV, view+projection matrices
    coordinates.py         # CoordinateTransformer — WGS84↔ENU via pyproj UTM 11N
    projection.py          # project_points (vectorized), frustum_cull_segments, near-plane clip
    controls.py            # handle_input — pygame WASD/mouse/scroll → camera mutations, HUD
  scene/
    scene.py               # Scene facade — road_segments + camera + config + spatial index
    spatial_index.py       # SpatialIndex — uniform grid, 500m cells, O(1) insert/query
  placement/
    scene_builder.py       # build_scene — roads GDF → RoadSegments → Scene with spatial index
  rendering/
    backend.py             # RenderBackend ABC — draw_line, draw_polygon, get_image, etc.
    pillow_backend.py      # PillowBackend — PIL/ImageDraw for static export
    pygame_backend.py      # PygameBackend — pygame.Surface for interactive preview
    render_pipeline.py     # render_scene_frame, get_preview_passes_warp, get_export_passes
    flat_map.py            # FlatMapMeta, generate_flat_map, get_flat_map (with disk cache)
    warp_preview.py        # Per-pixel raycast preview — _build_ray_directions, _raycast_to_flat_map
    road_pass.py           # render_roads_from_scene — per-segment 3D projection + drawing
    ground_pass.py         # render_ground — sky/ground fill + horizon glow band
tests/
  conftest.py
  test_camera.py
  test_config_loader.py
  test_coordinates.py
  test_projection.py
  test_road_classifier.py
  test_scene.py
  test_warp_preview.py     # Tests for warp_preview + flat_map modules
```

**Data flow:**

```
main.py:main()
  -> load_config(toml_path)                      # TOML → SceneConfig
  -> cmd_preview(args, config)                    # or cmd_verify / cmd_render
       -> _load_scene(config, quality, cache_dir)
            |-- load_roads(bbox, cache_dir)       # osmnx download or .gpkg cache
            |-- classify_roads(roads_raw)          # highway tag → RoadType column
            |-- create_transformer(origin_lat/lon) # pyproj UTM 11N transformer
            |-- create_camera(pos, target, ...)    # Camera with view/proj matrices
            `-- build_scene(roads, transformer, camera, config)
                 |-- convert_road_geometry()       # WGS84 LineStrings → ENU (N,3) arrays
                 |-- _build_road_segments()        # geometry + type → RoadSegment list
                 `-- scene.build_spatial_index()   # uniform grid, 500m cells
       -> get_flat_map(scene, cache_dir)           # generate or load cached flat map
       -> get_preview_passes_warp(flat_map, meta)  # closure capturing flat_map
       -> pygame event loop
            |-- handle_input(events, camera, ...)  # WASD/mouse → camera.move()/rotate()
            `-- render_scene_frame(scene, backend, passes=[_warp_pass])
                 -> render_warp_preview(scene, camera, backend, flat_map, flat_meta)
                      |-- _build_ray_directions()  # meshgrid → per-pixel world rays
                      |-- _raycast_to_flat_map()   # ray-ground intersect → flat map px
                      `-- _sample_flat_map()        # numpy fancy index → PIL output
```

**Conventions:**

- All config dataclasses in `config/schema.py`; defaults in `config/defaults.py`
- `RoadType` enum centralizes road categories; `_HIGHWAY_TO_ROAD_TYPE` dict in `road_classifier.py` maps OSM tags
- Style tables (`_ROAD_STYLE`, `_PREVIEW_ROAD_STYLE`, `_FLAT_MAP_ROAD_STYLE`) map `RoadType` → color/width
- Render passes are `Callable[[Scene, Camera, RenderBackend], Any]` — pluggable via lists
- `RenderBackend` ABC with two implementations; preview uses `blit_pil_image`, export uses `draw_line`
- Coordinate conversion always flows WGS84 → UTM 11N → ENU offset; `CoordinateTransformer` encapsulates origin

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
| tomli-w     | >=1.0      | TOML writing (config round-trip)           |
| pygame      | >=2.5      | Interactive preview window + input         |
| Pillow      | >=10.0     | Flat map generation, warp output, export   |

**Dev dependencies:**

| Package | Constraint | Purpose    |
|---------|------------|------------|
| pytest  | >=7.0      | Test runner|

**Test patterns**: pytest with `unittest.mock`. Fixtures at file top. Naming: `test_method_scenario_behavior`. One file per source module. `spec=True` on mocks. `MagicMock()` for Scene (no spec). Tests prefixed with `# Radical`.
