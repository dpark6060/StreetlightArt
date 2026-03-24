# Stage 4 Deferred Items

Items identified during code review and architecture review that were not addressed
in the initial Stage 4 implementation. These should be resolved before or during
Stage 5 as noted.

## 1. Zone Spatial Indexing (resolve during Stage 5)

**What:** `Scene.build_spatial_index()` only indexes road segments. Zone boundaries
are not indexed. `ground_pass.py` iterates all zones linearly with a bounding-box
pre-check.

**Why it matters:** Stage 5 light placement will query zones per road segment to
determine light characteristics. Without spatial indexing, this becomes
O(segments × zones). For the current San Bernardino dataset that's manageable, but
it will degrade with larger bounds.

**Fix:** Extend `build_spatial_index()` to also insert zone boundaries by their
bounding box. Add `get_zones_in_rect()` query method (same pattern as
`get_roads_in_rect()`). Update `ground_pass.py` to use it instead of iterating all
zones.

**When:** Start of Stage 5, before building the placement engine.

---

## 2. Polygon Vertex Clipping in Preview (resolve before Stage 6)

**What:** `_project_polygon` in `ground_pass.py` drops vertices that fail NDC bounds
check instead of clipping them. When a large zone polygon straddles the camera's near
plane, some vertices are behind the camera and are silently discarded. The remaining
vertices form an incorrect screen polygon.

**Why it matters:** At the default camera angle (oblique, high altitude), most large
zone polygons have ALL vertices outside screen bounds, meaning they are skipped
entirely in the preview. The preview still looks okay because the flat ground color
fills the background, but zone coloring is only visible for smaller zones that happen
to project fully on-screen.

**Fix:** Port the near-plane clipping logic from `road_pass.py` (which already
handles this for line segments) to work with polygon vertex rings. Alternatively,
use Sutherland-Hodgman polygon clipping against the view frustum.

**When:** Before Stage 6 (atmosphere), since atmospheric effects depend on correct
zone visibility. Not critical for Stage 5 (lights don't depend on preview zone
rendering).

---

## 3. Export Tile Sampling Performance (resolve if needed)

**What:** `_sample_tiles` in `ground_pass.py` uses a per-pixel Python loop to sample
ground tile colors during export rendering. At 7680×4320 (high quality), this is
~33 million pixels iterated in pure Python.

**Why it matters:** Export will be slow at high resolution. Functionally correct but
could take minutes per frame.

**Fix:** Vectorize with numpy — batch-compute tile coordinates for all ground pixels
at once, load tiles into a numpy array, and use fancy indexing to sample. Similar
pattern to `warp_preview.py`'s `_raycast_to_flat_map`.

**When:** Only if export speed is actually a problem. Low quality (1920×1080) should
be fine. Try it first before optimizing.

---

## 4. Architecture Plan Updates (resolve now or next session)

**What:** Three entries in `.claude/work/architecture-plan.md` no longer match the
implementation:

1. **`data/zone_builder.py`** — Plan says `build_zones(land_use_gdf, buildings_gdf)`.
   Actual signature: `build_zones(land_use_gdf, buildings_gdf, transformer,
   zone_config, rng)`. WGS84→ENU conversion happens inside zone_builder, not as a
   separate step. Plan data flow steps 4 and 5 should be merged.

2. **`scene/ground_tiles.py`** — Plan says `get_tile(tile_x, tile_y, zones, config)`.
   Actual: `get_tile(tile_x, tile_y, grid, scene, cache_dir)`. Plan should also
   document `GroundTileGrid` dataclass and `create_tile_grid()` function.

3. **`placement/scene_builder.py`** — Plan describes it as "orchestrating all light
   placers." Actual responsibility: scene construction (roads + zones → Scene). Light
   placement is deferred to Stage 5 and will be a separate function or module.

**When:** Anytime. Low priority — the code is the source of truth. But worth doing
before Stage 5 planning so the plan doesn't mislead.
