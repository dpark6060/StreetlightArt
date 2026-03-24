# Stage 5 Deferred Items

Items identified during Stage 5 implementation and review that were not addressed.

## Carried from Stage 4

### 2. Polygon Vertex Clipping in Preview (resolve before Stage 6)

**What:** `_project_polygon` in `ground_pass.py` drops vertices outside NDC bounds
instead of clipping. Large zone polygons are invisible in the preview at oblique
camera angles.

**Fix:** Sutherland-Hodgman polygon clipping against the view frustum.

**When:** Before Stage 6 (atmosphere).

---

### 3. Export Tile Sampling Performance (resolve if needed)

**What:** `_sample_tiles` in `ground_pass.py` uses a per-pixel Python loop.

**Fix:** Vectorize with numpy if export is too slow at high resolution.

**When:** Only if export speed is actually a problem.

---

### 4. Architecture Plan Updates

**What:** `.claude/work/architecture-plan.md` has stale entries for zone_builder,
ground_tiles, and scene_builder signatures/responsibilities.

**When:** Anytime.

---

## New in Stage 5

### 5. Atmospheric Perspective for Distant Lights (resolve in Stage 6)

**What:** stage5.md specifies that distant lights should dim, diffuse, and color-shift
toward warm haze. This was not implemented — lights render at full brightness
regardless of distance.

**Why it matters:** Without atmospheric perspective, the scene looks flat — distant
lights are as bright as nearby ones, which breaks the aerial night view illusion.

**Fix:** In `light_pass.py`, apply a distance-based brightness curve and color shift:
```
brightness *= exp(-distance / falloff_distance)
color = lerp(light_color, haze_color, distance / max_distance)
glow_radius *= 1 + distance / scatter_distance
```
Make falloff_distance, haze_color, and scatter_distance configurable.

**When:** Stage 6 (atmosphere) or as a quick addition before.

---

### 6. Commercial Placers Use Empty Config Dict

**What:** `create_sign_light`, `create_parking_lot_light`, and `create_storefront_light`
in `commercial.py` accept a `config: dict` parameter but it's always passed as `{}`.
The actual `CommercialLightConfig` values from TOML are never read at runtime — all
parameters are hardcoded in the factory functions.

**Fix:** Similar to the streetlight config fix — pass `config.lights.commercial` through
the placer and have the factory functions read color pools and parameter ranges from it.

**When:** Next session or when commercial light tuning is needed.

---

### 7. House Light Placer Uses Empty Config Dict

**What:** Same issue as #6. `create_house_light` receives `{}` instead of
`config.lights.house_lights`. The `HouseLightConfig` is parsed but not used at runtime.

**Fix:** Pass `config.lights.house_lights` through the placer to
`House.get_lights(rng)` and then to `create_house_light`.

**When:** Next session or when house light tuning is needed.

---

### 8. Car Config Parameters Partially Hardcoded

**What:** `CarLightPair.__init__` uses hardcoded color pools and brightness ranges
instead of reading from `CarConfig`. The car density ranges ARE read from config
correctly, but per-car color and brightness are hardcoded.

**Fix:** Pass `CarConfig` through to `CarLightPair` and use its `headlight_colors`,
`taillight_colors`, `separation_min`, `separation_max` fields.

**When:** Next session or when car light tuning is needed.

---

### 9. `populate_lights` Mixes State Mutation with Console I/O

**What:** `populate_lights` in `scene_builder.py` both mutates the scene and prints
a summary. These are two responsibilities.

**Fix:** Move the print calls to `build_scene` or `main.py`.

**When:** Low priority. Doesn't affect functionality.

---

### 10. Parking Lot Light Height

**What:** Parking lot lights are placed at z=0. The spec mentions 8-12m poles.

**Fix:** Set z-coordinate from `RangeParam(8, 12)` or from config.

**When:** When the renderer uses z-height for glow calculations.
