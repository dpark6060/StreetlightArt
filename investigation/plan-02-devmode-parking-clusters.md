# Plan 02 — Two parking-lot clusters (medium + high density) in the devmode synthetic scene

## 1. Answer

### How the synthetic scene is laid out today

`src/streets_at_night/devmode/synthetic_scene.py` builds a 10-row ENU test scene.
Rows are horizontal strips along X (0..10000m, `_ROW_X_START`/`_ROW_X_END`, lines 58-59),
stacked along Y at a uniform 200m pitch defined by `ROW_LABELS` (lines 35-46):

| idx | label | y |
|---|---|---|
| 0-4 | Highway..Service | 900..100 |
| 5 | Cars | -100 |
| 6 | Sign Lights | -300 |
| 7 | Parking Lights | -500 |
| 8 | Storefront | -700 |
| 9 | Houses | -900 |

- `build_synthetic_scene()` (lines 68-93): camera → road segments → buildings → Scene →
  spatial index → `_populate_road_lights()` → `_populate_direct_light_rows()`.
- `_build_road_segments()` (lines 144-185) creates **9** segments: rows 0-4 (road types with
  streetlights), row 5 (car-only, `fixture_profile=None`), rows 6-8 (wide 20m RESIDENTIAL
  surface strips under the direct-light rows, `fixture_profile=None`). Row 9 (Houses) has no
  segment.
- Rows 6-8 place one light every 80m (`_DIRECT_LIGHT_SPACING`, line 65) via
  `_populate_direct_light_rows()` (lines 412-442). The existing "Parking Lights" row (row 7)
  is a **single 1-D line** of `create_parking_lot_light()` calls at z=8 (lines 482-502) — it
  shows the fixture but not the grid/density behavior of a real lot.
- Houses row: `_build_synthetic_buildings()` (lines 239-273) cycles `_BUILDING_TEMPLATES`
  every 200m.
- Determinism: one master `random.Random(config.seed)`, child RNGs derived per subsystem
  (lines 89-91, 427-429).

### How parking lights are generated in the real pipeline

`src/streets_at_night/placement/commercial_placer.py`:

- `_place_parking_grid(polygon, config, rng, pool)` (lines 318-349): samples one spacing from
  `rng.uniform(config.commercial.parking_grid_spacing_min, parking_grid_spacing_max)`, computes
  grid points inside the polygon with `_compute_grid_points()` (lines 352-374, vectorized
  `shapely.contains_xy`), and creates one `create_parking_lot_light()` per point at a random
  pole height in 8-12m (`_PARKING_LIGHT_HEIGHT_MIN/MAX`, lines 21-22).
- The factory `create_parking_lot_light()` (`src/streets_at_night/models/commercial.py`
  lines 55-94) reads all `parking_*` fields from `CommercialLightConfig`: color pool,
  brightness min/max, glow radius, and the ConeIllumination (radius/falloff/inner fraction).

Config (`config/schema.py` lines 179-188, `config/defaults.py` lines 428-442):
`parking_grid_spacing_min = 20.0`, `parking_grid_spacing_max = 30.0` (so default lots average
~25m pole pitch), brightness 0.75-0.9, glow 20, illum radius 30 / intensity 0.8 / falloff 1.5 /
inner fraction 0.3.

### Labels, minimap, sliders

- Labels: `devmode/render_helpers.py` line 160 appends `make_label_pass(ROW_LABELS, camera)`;
  `devmode/label_pass.py` anchors every label at world `[0, y, 0]` (line 71). Any entry
  appended to `ROW_LABELS` is labeled automatically.
- Minimap: draws **road segments only** (`devmode/minimap.py` lines 157-175) and the world
  extent comes from road segment geometry (`ui/tabs/devmode_tab.py` `_compute_world_extent`,
  lines 742-765). So new rows need a road segment to appear on the minimap and to extend its
  extent.
- Sliders: the "Commercial: Parking" panel section (`devmode/light_catalog.py` line 105,
  `field_prefix="parking_"`) edits `config.lights.commercial` in place, and applying sliders
  re-runs `build_synthetic_scene()` (`ui/tabs/devmode_tab.py` `_rebuild_scene_in_place`,
  lines 497-509). Anything the synthetic scene derives from `config.lights.commercial` —
  including `parking_grid_spacing_min/max` — therefore responds to the sliders with no panel
  changes.

### Recommended placement + density

Add **two new rows** appended to `ROW_LABELS`, keeping the 200m pitch (this reuses the existing
per-row label and minimap machinery for free; putting both clusters in one row would require
extending the label pass to support per-X anchors):

| idx | label | y | lot X extent | density |
|---|---|---|---|---|
| 10 | `Parking Lot Med`  | -1100.0 | 1600..1900 | config spacing as-is (scale 1.0, ~25m pitch → ~12x4 ≈ 50 poles) |
| 11 | `Parking Lot High` | -1300.0 | 2400..2700 | config spacing x 0.5 (~12.5m pitch → ~24x9 ≈ 200 poles) |

Each lot is a 300m x 120m axis-aligned rectangle (120m depth keeps 80m clearance to the
neighboring row). X extents 1600-1900 / 2400-2700 sit inside the default devmode frustum
(camera at (-500, 0, 800) looking at (5000, 0, 0), `config/defaults.py` lines 578-592) and are
staggered along X so the two grids read side by side without one occluding the other.
"Medium" = the real-pipeline default grid (20-30m spacing). "High" = the same sampled spacing
scaled by 0.5 (10-15m), expressed as a scale so the density *relationship* is preserved when
the user drags the spacing sliders.

Lights are placed by the real placer (`_place_parking_grid`, made public with a new
`spacing_scale` parameter) on real shapely polygons — same grid logic, same 8-12m pole
heights, same factory — so every `parking_*` slider drives both clusters.

Each cluster row also gets a road segment spanning its lot X extent with `width = 120`
(RESIDENTIAL, `fixture_profile=None`): it renders a dark asphalt-like pad under the grid and
puts the rows on the minimap / inside its extent.

---

## 2. Code change plan

Written for a code-writing agent with repo access and no other context. Repo conventions:
`uv` for everything (`uv run pytest tests/`), Google docstrings, type hints, `typing as t`,
snake_case with approved prefixes, private `_helpers`, one test file per module,
`test_method_scenario_behavior` naming, test files start with `# Radical`.

### Change A — `src/streets_at_night/placement/commercial_placer.py`

Make the parking grid placer public and add a density scale.

1. Rename `_place_parking_grid` (line 318) to `place_parking_grid` and add a
   `spacing_scale: float = 1.0` keyword parameter (after `pool`):

```python
def place_parking_grid(
    polygon: Polygon,
    config: LightConfig,
    rng: random.Random,
    pool: t.Optional[ColorPool] = None,
    spacing_scale: float = 1.0,
) -> t.List[LightSource]:
```

2. Inside, scale the sampled spacing:

```python
    spacing = spacing_scale * rng.uniform(
        config.commercial.parking_grid_spacing_min,
        config.commercial.parking_grid_spacing_max,
    )
    spacing = max(spacing, 1.0)  # guard against degenerate grids
```

   Keep the rest of the body unchanged (`_compute_grid_points`, 8-12m pole height,
   `create_parking_lot_light`). Update the docstring: mention `spacing_scale` multiplies the
   sampled grid spacing (values < 1.0 pack poles more densely) and note the 1.0m floor.

3. Update the one internal caller `_place_parking_lights` (line 161):
   `lights.extend(place_parking_grid(area, config, rng, pools.parking))`.

### Change B — `src/streets_at_night/devmode/synthetic_scene.py`

1. Module docstring (lines 1-9) and `build_synthetic_scene` docstring (lines 69-82): 10 rows →
   12 rows; mention rows 10-11 are two rectangular parking-lot grids (medium vs high density).

2. Append to `ROW_LABELS` (after `("Houses", -900.0)`, line 45):

```python
    ("Parking Lot Med",  -1100.0),
    ("Parking Lot High", -1300.0),
]
```

3. New module constants (near `_DIRECT_LIGHT_SPACING`, line 65):

```python
# Parking cluster rows (indices 10-11): lot X extents, depth, and the
# spacing scale applied to the config's sampled parking grid spacing.
_PARKING_LOT_DEPTH = 120.0
_PARKING_MED_X: t.Tuple[float, float] = (1600.0, 1900.0)
_PARKING_HIGH_X: t.Tuple[float, float] = (2400.0, 2700.0)
_PARKING_MED_SPACING_SCALE = 1.0
_PARKING_HIGH_SPACING_SCALE = 0.5
```

4. Import at top: `from ..placement.commercial_placer import place_parking_grid` and
   `from shapely.geometry import box` (shapely is already a runtime dep; commercial_placer
   has no devmode imports, so no cycle).

5. In `_build_road_segments()` (lines 144-185): after the rows 6-8 loop, append one pad
   segment per cluster row (indices 9 and 10 in the returned list). Update the function
   docstring ("Nine road segments" → "Eleven"; describe the two lot pads).

```python
    for x_range, (_, y_offset) in zip(
        (_PARKING_MED_X, _PARKING_HIGH_X), ROW_LABELS[10:12]
    ):
        segments.append(RoadSegment(
            geometry=_make_lot_geometry(x_range, y_offset),
            road_type=RoadType.RESIDENTIAL,
            width=_PARKING_LOT_DEPTH,
            fixture_profile=None,
        ))
```

6. New helper (mirror `_make_row_geometry`'s multi-point requirement — sub-segment endpoints
   must project on-screen, so include intermediate vertices):

```python
def _make_lot_geometry(x_range: t.Tuple[float, float], y: float) -> np.ndarray:
    """Create a short multi-point geometry spanning one parking lot's X extent.

    Args:
        x_range (tuple[float, float]): (x_start, x_end) of the lot in ENU meters.
        y (float): Y coordinate (ENU meters) for this row.

    Returns:
        np.ndarray: (4, 3) geometry array from x_start to x_end at z=0.
    """
    x_values = np.linspace(x_range[0], x_range[1], 4)
    geom = np.zeros((len(x_values), 3))
    geom[:, 0] = x_values
    geom[:, 1] = y
    return geom
```

7. In `build_synthetic_scene()` (line 91), after `_populate_direct_light_rows(...)`, add
   `_populate_parking_clusters(scene, config, rng)`. Appending the call last keeps existing
   rows' RNG draws (and therefore existing visuals) unchanged for a given seed.

8. New functions at module bottom:

```python
def _populate_parking_clusters(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> None:
    """Place the medium- and high-density parking lot clusters (rows 10-11).

    Both clusters reuse the real commercial placer (place_parking_grid) on
    rectangular lot polygons, so every parking_* config field — including the
    grid spacing sliders — drives them. The high-density cluster halves the
    sampled grid spacing via spacing_scale.

    Args:
        scene (Scene): The scene to populate.
        config (SceneConfig): Scene configuration providing parking params.
        rng (random.Random): Master RNG; a child RNG is derived from it.
    """
    cluster_rng = random.Random(rng.randint(0, 2**32))
    lights: t.List[LightSource] = []
    lights.extend(_create_parking_cluster(
        _PARKING_MED_X, ROW_LABELS[10][1], _PARKING_MED_SPACING_SCALE, config, cluster_rng
    ))
    lights.extend(_create_parking_cluster(
        _PARKING_HIGH_X, ROW_LABELS[11][1], _PARKING_HIGH_SPACING_SCALE, config, cluster_rng
    ))
    scene.add_lights(lights)


def _create_parking_cluster(
    x_range: t.Tuple[float, float],
    y: float,
    spacing_scale: float,
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Create one rectangular parking-lot light grid via the real placer.

    Args:
        x_range (tuple[float, float]): (x_start, x_end) of the lot in ENU meters.
        y (float): Row center Y coordinate (ENU meters).
        spacing_scale (float): Multiplier on the sampled grid spacing.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for this cluster.

    Returns:
        list[LightSource]: Parking lot lights inside the lot polygon.
    """
    half_depth = _PARKING_LOT_DEPTH / 2.0
    lot = box(x_range[0], y - half_depth, x_range[1], y + half_depth)
    return place_parking_grid(lot, config.lights, rng, spacing_scale=spacing_scale)
```

### Change C — no changes needed (verify only)

- `devmode/label_pass.py`, `devmode/render_helpers.py`: labels come from `ROW_LABELS`
  automatically; anchors at `[0, y, 0]` match every other row.
- `devmode/minimap.py`, `ui/tabs/devmode_tab.py`: extent and road overlay derive from
  `scene.road_segments`, which now include the two lot pads.
- `devmode/light_catalog.py` / `field_meta.py`: the "Commercial: Parking" section already
  exposes all `parking_*` fields including `parking_grid_spacing_min/max`; the tab rebuilds
  the synthetic scene on apply, so both clusters respond. Verify by dragging
  `parking grid spacing` sliders in the Devmode tab.

### Change D — `CLAUDE.md` (repo root)

Two stale references: the Structure entry for `devmode/synthetic_scene.py` says "10-row
synthetic ENU test scene" and the synthetic_scene module description in the constraints
section. Update to 12 rows and mention the two parking-lot cluster rows.

### Tests

#### `tests/test_commercial_placer.py`

- Update the import at line 22 (`_place_parking_grid` → `place_parking_grid`) and the
  existing `_place_parking_grid` call sites (lines ~348-366).
- Add (logic-method tests, no mocking of internals):

```python
def test_place_parking_grid_spacing_scale_half_produces_more_lights(
    parking_polygon, light_config
):
    """A 0.5 spacing_scale yields a denser grid than the default scale."""
    baseline = place_parking_grid(parking_polygon, light_config, random.Random(7))

    dense = place_parking_grid(
        parking_polygon, light_config, random.Random(7), spacing_scale=0.5
    )

    assert len(dense) > len(baseline)


def test_place_parking_grid_spacing_scale_default_matches_unscaled_behavior(
    parking_polygon, light_config
):
    """spacing_scale=1.0 reproduces the same grid as omitting the argument."""
    a = place_parking_grid(parking_polygon, light_config, random.Random(7))

    b = place_parking_grid(
        parking_polygon, light_config, random.Random(7), spacing_scale=1.0
    )

    assert len(a) == len(b)
```

  (Identical seeds make the sampled base spacing equal, so the comparison is deterministic.
  `random` is already imported at the top of that test file; if not, add it there — never
  inside a test.)

#### `tests/test_devmode_synthetic_scene.py`

- Update `test_row_labels_uniform_200m_pitch` (lines 485-490): 12 offsets, `900.0 .. -1300.0`.
- Update `test_synthetic_scene_has_correct_road_segments` (line 186) and
  `test_synthetic_scene_buildings_row_adds_no_road_segment` (line 178): expect **11** segments;
  reword docstrings ("rows 0-8 plus two parking lot pads", "Houses row adds no segment of its
  own").
- Add, using the existing `scene_config` fixture (lights outside y < -1000 can only be
  parking-cluster lights — streetlights/cars occupy y >= -100 rows, direct rows are
  y = -300..-700, and the Houses row emits no LightSource):

```python
def test_build_synthetic_scene_places_lights_in_both_parking_cluster_rows(
    scene_config: SceneConfig,
) -> None:
    """Both parking cluster rows receive at least one light."""
    scene = build_synthetic_scene(scene_config)

    med = [l for l in scene.lights if -1160.0 <= l.position[1] <= -1040.0]
    high = [l for l in scene.lights if -1360.0 <= l.position[1] <= -1240.0]
    assert len(med) > 0
    assert len(high) > 0


def test_build_synthetic_scene_high_density_cluster_has_more_lights_than_medium(
    scene_config: SceneConfig,
) -> None:
    """The high-density cluster packs more poles into the same lot size."""
    scene = build_synthetic_scene(scene_config)

    med = [l for l in scene.lights if -1160.0 <= l.position[1] <= -1040.0]
    high = [l for l in scene.lights if -1360.0 <= l.position[1] <= -1240.0]
    assert len(high) > len(med)


def test_parking_cluster_segments_are_unlit_residential_pads(
    scene_config: SceneConfig,
) -> None:
    """Segments 9-10 are wide RESIDENTIAL pads with no fixture profile."""
    scene = build_synthetic_scene(scene_config)

    for i in (9, 10):
        assert scene.road_segments[i].road_type == RoadType.RESIDENTIAL
        assert scene.road_segments[i].fixture_profile is None
        assert scene.road_segments[i].width == 120.0
```

- Run the full suite afterward: `uv run pytest tests/` and report pass/fail counts. Also grep
  for any other test pinned to 9 segments or 10 rows:
  `rg -n "road_segments\) == 9|ROW_LABELS" tests/`.

### Open design decisions (recommendations made)

1. **Two rows vs. one row with two lots.** Recommended two rows (indices 10-11) — per-row
   labels and minimap support come free; a single row would need `label_pass` extended to
   per-X anchors. If print real estate on the devmode map ever matters, revisit.
2. **High density = spacing_scale 0.5** (10-15m effective) rather than hardcoded meters, so the
   medium/high relationship survives slider changes. Alternative: separate absolute spacing
   constants — rejected because it decouples the clusters from the panel sliders.
3. **120m-wide road segment as the lot pad.** Slightly unusual (road pass draws a very thick
   polyline, with end caps overshooting the lot ends a little), but it reuses existing passes
   and gets minimap/extent handling free. Alternative — no pad — loses the minimap dot and the
   asphalt surface under the pools. If the fat polyline looks wrong in practice, drop the pads
   to `width=20.0` like rows 6-8 and accept a thin strip.
4. **Label anchors sit at x=0**, ~1.6km left of the lots (consistent with every other row, all
   anchored at x=0). Acceptable; if it reads poorly, a follow-up could extend `ROW_LABELS` to
   optional (label, x, y) triples — out of scope here.
