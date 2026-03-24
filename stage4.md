# Stage 4: Object Model Foundation & Ground Plane

## Goal
Two things in this stage:
1. Build the shared object model infrastructure (ColorPool, RangeParam, LightSource base,
   zone objects, House) — see `object_model.md` for the full hierarchy
2. Render the ground plane using zone objects (House, ResidentialZone, CommercialZone, DarkZone)
   with appropriate base colors, using tiled ground textures

The ground is dimly lit at this stage — dark colors that represent moonlight-level ambient.
Lights illuminate these surfaces in stage 5.

## Architecture Reference
See `.claude/work/architecture-plan.md` for the full module map and design decisions.

## Part 1: Object Model Infrastructure

### Shared Utilities
Build first, used by everything else:

```
src/streets_at_night/
    models/
        __init__.py
        color_pool.py        # ColorPool class
        range_param.py       # RangeParam class
        light_source.py      # LightSource base class
        road_types.py        # RoadType enum (already exists from stage 1)
```

**ColorPool**: weighted random color selection. Takes a list of (hex_color, weight) tuples.
`pick()` returns an RGB tuple. All color selection in the project goes through this.
Must accept a seeded RNG for reproducibility.

**RangeParam**: numeric parameter with min/max and distribution. `pick()` returns a float.
Supports uniform, normal, exponential distributions. Also accepts seeded RNG.

**LightSource**: base dataclass that the renderer consumes. Fields:
position, color, brightness, size, glow_radius, illumination_radius, illumination_intensity.
The renderer only sees LightSource objects — it doesn't know about cars, houses, etc.

### Zone Objects
```
src/streets_at_night/
    models/
        zones.py             # ResidentialZone, CommercialZone, DarkZone
        house.py             # House class (footprint, base_color, yard_color, get_lights())
        road_segment.py      # RoadSegment class (geometry + type only, NO lights)
```

**RoadSegment**: geometry and metadata only. **Does not own lights.** Streetlights and cars
are placed by the placement engine (stage 5) and stored in the Scene's flat light collection.

**House**: see `object_model.md`. On init:
- Receives a footprint polygon (from OSM or zone subdivision)
- Picks `base_color` from house exterior ColorPool
- Picks `yard_color` (dark green with slight variation)
- Picks `num_lights` from weighted distribution (0-5, heavily weighted toward 0-1)
- Computes light positions relative to footprint (but doesn't create LightSource objects
  yet — that happens when `get_lights()` is called by the placement engine in stage 5)

**ResidentialZone**: owns a boundary polygon and a list of Houses. On init:
- Takes OSM building footprints within the boundary
- Where footprints are missing, subdivides empty space into synthetic lots
  (simple grid — doesn't need to be accurate, just non-overlapping)
- Creates a House for each lot

**CommercialZone**: owns a boundary polygon, building footprints, and parking area polygons.
**Does not own lights** — SignLight, ParkingLotLight, StorefrontLight are created by the
commercial_placer (stage 5) and stored in the Scene.

**DarkZone**: parks, water, forest. Boundary polygon + base_color. No lights, no subdivision.

### Configuration
The TOML config system was set up in stage 1. In this stage, extend it with light and
zone parameters:

```
src/streets_at_night/
    config/
        schema.py            # Add HouseConfig, CommercialConfig, zone color definitions
        defaults.py          # Add default color pools, parameter ranges, probabilities
```

All color pools and parameter ranges are defined in `defaults.py` and can be overridden
in `scene_config.toml`.

## Part 2: Ground Plane Rendering

### Tiled Ground Textures
The ground texture is too large to fit in memory as a single image (40km at 1px=2m =
20000×20000 pixels). Use tiles instead:

**`scene/ground_tiles.py`**:
- 256m × 256m tiles at 1px = 2m resolution (128×128 pixel tiles)
- ~160×160 tiles for the full area
- Each tile colored based on zone type and building footprints:
  - Residential: per-house `base_color` for footprints, `yard_color` between them
  - Commercial: uniform dark grey
  - Park/forest: dark green
  - Water: dark blue
  - Default: dark brown-grey
- Tiles cached to `cache/ground_tiles/` after first generation
- Only tiles visible in the current camera frustum are loaded/rendered

### Layer Order (back to front)
1. Sky (flat color, above horizon)
2. Default ground (very dark grey-brown, #0A0808)
3. DarkZone fills (parks = very dark green, water = very dark blue)
4. CommercialZone fills (very dark cool grey)
5. ResidentialZone fills — per-House rendering:
   - House footprints in their `base_color`
   - Yard space in `yard_color`
6. Road surfaces (dark grey, slightly lighter than surroundings)

### Zone Base Colors (very dark — night ambient before any light)
```
House exteriors:   ColorPool of very dark warm tones (#0D0A08, #0C0907, #0B0808, etc.)
Yards:             #060A04 base with ±(2,2,2) random variation per house
Commercial:        #0A0A0C (very dark cool grey)
Industrial:        #080808 (very dark neutral grey)
Park/Forest:       #040804 (very dark green)
Water:             #040408 (very dark blue)
Road surface:      #0C0C0C (dark grey)
Default ground:    #080706 (very dark brown-grey)
```

These are intentionally almost-black. Light illumination in stage 5 reveals the actual colors.

### Ground Rendering via RenderBackend

Both preview and export use the same `ground_pass.py` rendering logic, but with
backend-appropriate detail levels:

**Preview (pygame backend)**:
- Project zone boundary vertices → `backend.draw_polygon()` with flat zone colors
- Houses rendered as zone-level flat color (no per-house variation — too slow)
- Toggle with "G" key
- Must not kill FPS

**Export (Pillow backend)**:
- Query visible ground tiles from `ground_tiles.py`
- Sample tile textures for each screen pixel below horizon
- Supports full per-house color variation and yard detail

### Scene Updates
Add zones and ground tiles to the Scene:
```python
class Scene:
    road_segments: list[RoadSegment]
    residential_zones: list[ResidentialZone]
    commercial_zones: list[CommercialZone]
    dark_zones: list[DarkZone]
    camera: Camera
    config: SceneConfig
    spatial_index: SpatialIndex  # now also indexes zone boundaries
    # lights added in stage 5
```

### Residential Zone Subdivision (where OSM footprints are missing)
OSM building coverage may be spotty. For areas within a residential zone boundary but
lacking footprint data:
- Subdivide into rectangular lots (~15m x 25m, with variation)
- Offset from road edges by ~10m (front setback)
- Place a rectangular "house" footprint within each lot (~60-70% lot coverage)
- This is approximate — from 5000ft it just needs to read as "houses with yards"

### New/Modified Files Summary
```
src/streets_at_night/
    models/
        __init__.py
        color_pool.py
        range_param.py
        light_source.py
        road_types.py          # already exists
        zones.py
        house.py
        road_segment.py
    scene/
        ground_tiles.py        # Tiled ground texture generation + caching
    rendering/
        ground_pass.py         # Ground rendering logic (backend-agnostic)
    data/
        zone_builder.py        # Build zones from OSM land use + building data
    config/
        schema.py              # Add zone/house config dataclasses
        defaults.py            # Add zone/house default parameters
```

## Dependencies (new)
- None beyond what's already installed

## Acceptance Criteria
- [ ] ColorPool and RangeParam work correctly with seeded RNG
- [ ] House objects generate with random but reproducible colors and light counts
- [ ] ResidentialZone correctly wraps OSM footprints and generates synthetic lots where missing
- [ ] Zone objects (commercial, dark) correctly built from OSM land use data
- [ ] RoadSegment holds geometry and type only — no light references
- [ ] Ground tiles generated and cached for visible area
- [ ] Ground plane visible in pygame preview (dark colored zones, road surfaces)
- [ ] High-res export shows per-house color variation via ground tiles
- [ ] Parks and water bodies are visually distinct dark regions
- [ ] Ground colors are very dark (night ambient)
- [ ] Same seed → same scene (reproducible randomness)
- [ ] SceneConfig can override all default color pools and parameters via TOML
- [ ] Scene facade holds road segments, zones, camera, config, and spatial index

## Notes
- The House class computes light positions on init but doesn't create LightSource objects
  until `get_lights()` is called by the placement engine in stage 5. The House needs to
  know its light count to exist as a complete object, but actual light creation is deferred.
- Don't over-invest in the synthetic lot subdivision. It's a fallback for missing OSM data
  and doesn't need to look architecturally realistic. A grid of rectangles is fine.
- Ground tiles at 128×128 pixels each are small enough to fit many in memory simultaneously.
  At 1px=2m, each tile covers 256×256m. Only tiles in the camera frustum need to be loaded.
