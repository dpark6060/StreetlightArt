# House Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Houses render lit windows on their building geometry; the legacy ground-level house-light point-light system is removed entirely.

**Architecture:** Three moves: (1) a single-row fallback in the window-grid row math so one-story buildings (~4.5m) get one centered window row instead of zero; (2) a `window_lit_fraction_residential` config field so house-kind buildings light ~40% of windows while towers keep the sparse 12%; (3) full removal of the house-light point-light system (`models/house_light.py`, `placement/house_light_placer.py`, `HouseLightConfig`, and all wiring).

**Tech Stack:** Python 3.11, uv, pytest, numpy, dataclass-based config with reflection serde.

**Spec:** `docs/superpowers/specs/2026-07-09-house-windows-design.md`

## Global Constraints

- Package manager is `uv`: run tests with `uv run pytest tests/...`.
- Test conventions (see user rules UnitTests.md): imports at top of file, double quotes, naming `test_methodundertest_scenario_behavior`, Arrange/Act/Assert with whitespace, fixtures at top of file, parametrize only for pure input→output tables.
- Old TOML configs containing `[lights.house_lights]` or `min_lights`/`max_lights`/`light_weights` keys MUST still load without error. No loader code change is needed — `_generic_from_dict` in `config/serde.py` only reads keys matching schema fields — but Task 3 adds a regression test proving it.
- `House` keeps `footprint`, `base_color`, `yard_color` — ground tiles consume them.
- Never revert unrelated working-tree changes; read files before editing and treat current content as authoritative.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Key math (referenced by Task 1 tests)

Window rows = `floor((height − window_roof_margin_m − window_sill_margin_m) / floor_height_m)`.
Defaults: sill 0.9, roof 0.6, floor 3.2, window height 1.1, window pitch 2.4, side margin 0.8.
A 4.5m house: usable = 3.0, `floor(3.0/3.2)` = 0 rows today. The single-row fallback fires when
rows == 0 and usable ≥ `window_height_m`. `_get_window_row_centers` already centers a single row
in the band: z = sill + (usable − floor_height)/2 + floor_height/2 = 0.9 + (3.0−3.2)/2 + 1.6 = **2.4**.
On a 10m wall: usable width 8.4, `floor(8.4/2.4)` = **3 columns**.

---

### Task 1: Single-row window fallback

**Files:**
- Modify: `src/streets_at_night/rendering/building_pass.py` (`_get_window_row_count`, ~line 582)
- Test: `tests/test_building_pass.py`

**Interfaces:**
- Consumes: existing `_get_window_row_count(height, config)`, `build_window_grid(...)`, and the `building_config` / `south_wall_edge` fixtures already in `tests/test_building_pass.py`.
- Produces: `_get_window_row_count` returns 1 (not 0) whenever `0 < usable < floor_height_m` and `usable >= config.window_height_m`. No signature change.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_building_pass.py`, next to the existing `_get_window_row_count` tests (~line 346). `pytest` is already imported.

```python
def test_get_window_row_count_returns_fallback_row_for_short_house(building_config):
    # Act — 4.5m house: usable 3.0m is less than one 3.2m floor, but a 1.1m window fits
    rows = _get_window_row_count(4.5, building_config)

    # Assert
    assert rows == 1


def test_get_window_row_count_returns_zero_when_no_window_fits(building_config):
    # Act — 2.0m wall: usable 0.5m is smaller than window_height_m 1.1m
    rows = _get_window_row_count(2.0, building_config)

    # Assert
    assert rows == 0


def test_get_window_row_count_multi_floor_heights_unaffected_by_fallback(building_config):
    # Act — 10m building: floor(8.5 / 3.2) = 2 rows, fallback path not involved
    rows = _get_window_row_count(10.0, building_config)

    # Assert
    assert rows == 2


def test_build_window_grid_short_house_gets_single_centered_row(south_wall_edge, building_config):
    # Arrange
    p0, p1, normal = south_wall_edge

    # Act — 4.5m house on a 10m wall
    cells = build_window_grid(
        p0, p1, 4.5, normal, edge_index=0, window_seed=1, config=building_config
    )

    # Assert — 1 row x 3 cols; row center z = 0.9 + (3.0 - 3.2)/2 + 1.6 = 2.4,
    # so windows span 1.85..2.95, below the 3.9m roofline
    assert len(cells) == 3
    for cell in cells:
        assert cell.corners[0][2] == pytest.approx(1.85)
        assert cell.corners[2][2] == pytest.approx(2.95)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_building_pass.py -k "fallback or no_window_fits or multi_floor or single_centered" -v`
Expected: `test_get_window_row_count_returns_fallback_row_for_short_house` and `test_build_window_grid_short_house_gets_single_centered_row` FAIL (rows == 0 / cells == []); the other two PASS (they pin current behavior).

- [ ] **Step 3: Implement the fallback**

Replace `_get_window_row_count` in `src/streets_at_night/rendering/building_pass.py`:

```python
def _get_window_row_count(height: float, config: BuildingConfig) -> int:
    """Compute how many window rows fit in a wall's vertical band.

    Buildings too short for a full floor still get one centered row when a
    window physically fits in the band, so one-story houses render lit
    windows instead of dark boxes.

    Args:
        height (float): Building height in meters.
        config (BuildingConfig): Window margin/spacing configuration.

    Returns:
        int: Number of rows that fit (0 when not even one window fits).
    """
    usable = height - config.window_roof_margin_m - config.window_sill_margin_m
    if usable <= 0.0:
        return 0
    rows = int(np.floor(usable / config.floor_height_m))
    if rows == 0 and usable >= config.window_height_m:
        return 1
    return rows
```

- [ ] **Step 4: Run the building-pass suite**

Run: `uv run pytest tests/test_building_pass.py -v`
Expected: all PASS. The pre-existing "too short" tests use height 1.0 (usable negative) and are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/rendering/building_pass.py tests/test_building_pass.py
git commit -m "Building windows: single-row fallback so one-story houses get a window row

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Residential window lit fraction

**Files:**
- Modify: `src/streets_at_night/config/schema.py` (`BuildingConfig`, ~line 382)
- Modify: `src/streets_at_night/config/defaults.py` (`"buildings"` dict, ~line 565)
- Modify: `src/streets_at_night/rendering/building_pass.py` (`_accumulate_building_windows` ~line 691, `_accumulate_window_cell` ~line 738)
- Modify: `tests/test_building_pass.py` (`building_config` fixture ~line 41), `tests/test_building_builder.py` (any `BuildingConfig(` constructions, ~line 91 area)
- Test: `tests/test_building_pass.py`

**Interfaces:**
- Consumes: `Building.building_kind: str` (raw OSM tag, `"house"` for synthetic-lot houses), `_is_window_lit(rng, lit_fraction)` (unchanged).
- Produces: `BuildingConfig.window_lit_fraction_residential: float` (default 0.4); module constant `_RESIDENTIAL_KINDS = frozenset({"house", "detached", "residential"})`; helper `_get_lit_fraction(building_kind: str, config: BuildingConfig) -> float`; `_accumulate_window_cell` gains a `lit_fraction: float` parameter (positioned after `config`). Devmode's "Buildings: Windows" panel picks the field up automatically via the `window_` prefix, and `field_meta.py`'s existing `("lit_fraction", FieldMeta(0.0, 1.0))` rule provides the slider range — no devmode changes needed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_building_pass.py` (the fixture sets `window_lit_fraction=0.12`; Step 3 adds `window_lit_fraction_residential=0.4` to it):

```python
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("house", 0.4),
        ("detached", 0.4),
        ("residential", 0.4),
        ("apartments", 0.12),
        ("retail", 0.12),
        ("yes", 0.12),
    ],
)
def test_get_lit_fraction_selects_fraction_by_building_kind(kind, expected, building_config):
    # Act
    fraction = _get_lit_fraction(kind, building_config)

    # Assert
    assert fraction == expected
```

Add `_get_lit_fraction` to the module's import list at the top of the test file (same `from streets_at_night.rendering.building_pass import ...` statement the other private helpers use).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_building_pass.py -k "lit_fraction_selects" -v`
Expected: FAIL with `ImportError: cannot import name '_get_lit_fraction'`.

- [ ] **Step 3: Implement**

`src/streets_at_night/config/schema.py` — in `BuildingConfig`, directly after `window_lit_fraction: float`:

```python
    window_lit_fraction: float
    window_lit_fraction_residential: float
```

`src/streets_at_night/config/defaults.py` — in the `"buildings"` dict, directly after `"window_lit_fraction": 0.12,`:

```python
        "window_lit_fraction": 0.12,
        "window_lit_fraction_residential": 0.4,
```

`tests/test_building_pass.py` `building_config` fixture — after `window_lit_fraction=0.12,` add `window_lit_fraction_residential=0.4,`. Do the same in every `BuildingConfig(` construction in `tests/test_building_builder.py` (use `rg -n "BuildingConfig\(" tests/` to find them all; those fixtures can use `window_lit_fraction_residential=0.4`).

`src/streets_at_night/rendering/building_pass.py` — add near the other module constants (top of file, after `_WINDOW_MIN_QUAD_PX`-style constants):

```python
# Building kinds that use the residential window lit fraction. Synthetic-lot
# house footprints are tagged "house" by building_builder, so they're covered.
_RESIDENTIAL_KINDS: t.FrozenSet[str] = frozenset({"house", "detached", "residential"})
```

Add the helper next to `_is_window_lit`:

```python
def _get_lit_fraction(building_kind: str, config: BuildingConfig) -> float:
    """Return the window lit fraction for a building's kind.

    Args:
        building_kind (str): Raw OSM ``building`` tag value.
        config (BuildingConfig): Window configuration.

    Returns:
        float: ``window_lit_fraction_residential`` for house-like kinds,
        ``window_lit_fraction`` for everything else.
    """
    if building_kind in _RESIDENTIAL_KINDS:
        return config.window_lit_fraction_residential
    return config.window_lit_fraction
```

In `_accumulate_building_windows`, compute the fraction once per building (before the `for wall in facing_walls:` loop):

```python
    force_subpixel = distance_m >= _WINDOW_SUBPIXEL_DISTANCE_M
    lit_fraction = _get_lit_fraction(building.building_kind, config)
```

and pass it through the `_accumulate_window_cell` call:

```python
            if _accumulate_window_cell(
                cell, view_matrix, proj_matrix, width, height, config, lit_fraction,
                force_subpixel, window_buffer,
            ):
```

In `_accumulate_window_cell`, add the parameter after `config` and use it (also add an Args line to the docstring):

```python
def _accumulate_window_cell(
    cell: WindowCell,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    config: BuildingConfig,
    lit_fraction: float,
    force_subpixel: bool,
    window_buffer: np.ndarray,
) -> bool:
```

```python
    rng = _get_window_rng(cell.rng_key)
    if not _is_window_lit(rng, lit_fraction):
        return False
```

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest tests/test_building_pass.py tests/test_building_builder.py tests/test_config_loader.py -v`
Expected: all PASS (any existing test that calls `_accumulate_window_cell` directly must be updated to pass `lit_fraction` — check with `rg -n "_accumulate_window_cell" tests/`).

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/config/schema.py src/streets_at_night/config/defaults.py \
        src/streets_at_night/rendering/building_pass.py \
        tests/test_building_pass.py tests/test_building_builder.py
git commit -m "Building windows: per-kind lit fraction, residential default 0.4

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Remove house-light generation (models, placer, scene builder, synthetic scene)

**Files:**
- Delete: `src/streets_at_night/models/house_light.py`, `src/streets_at_night/placement/house_light_placer.py`, `tests/test_house_light.py`, `tests/test_house_light_placer.py`
- Modify: `src/streets_at_night/models/house.py`, `src/streets_at_night/placement/scene_builder.py`, `src/streets_at_night/devmode/synthetic_scene.py`, `src/streets_at_night/config/schema.py` (`HouseConfig`), `src/streets_at_night/config/defaults.py` (`"zones"."house"`)
- Modify tests: `tests/test_house.py`, `tests/test_scene_builder.py`, `tests/test_devmode_synthetic_scene.py`, `tests/test_config_zones.py`, `tests/test_config_loader.py`, plus the mechanical `HouseConfig(` kwarg sweep in `tests/test_zones.py`, `tests/test_zone_builder.py`, `tests/test_ground_pass.py`, `tests/test_ground_tiles.py`, `tests/test_scene.py`, `tests/test_devmode_real_scene.py`, `tests/test_building_builder.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `House` exposes only `footprint`, `base_color`, `yard_color` (constructor signature unchanged: `House(footprint, config, rng, color_pool=None)`). `HouseConfig` loses `min_lights`, `max_lights`, `light_weights`. `populate_lights(scene, config, rng)` returns a **3-tuple** `(streetlights, car_lights, commercial_lights)`. `_create_child_rngs` returns a 3-tuple. Synthetic scene has **10 rows (0–9)**: roads 0–4, cars 5, sign 6, parking 7, storefront 8, houses 9 — and **9 road segments** (rows 0–8).

- [ ] **Step 1: Update tests first (they define the target state)**

1. Delete `tests/test_house_light.py` and `tests/test_house_light_placer.py` (`git rm`).
2. `tests/test_house.py`: remove the `house_light_config` fixture and its `HouseLightConfig` import; delete the tests exercising `num_lights`, `light_positions`, and `get_lights` (lines ~106, ~135–166 area: `test_get_lights_returns_light_sources_for_each_position`, `test_get_lights_returns_empty_list_when_no_lights`, and any `num_lights` range assertions). Remove `min_lights=`, `max_lights=`, `light_weights=` kwargs from every `HouseConfig(` construction. Keep all footprint/base_color/yard_color tests.
3. `tests/test_scene_builder.py`: remove every `@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")` decorator and its corresponding mock argument; delete `test_populate_lights_calls_house_light_placer`; update tuple unpacking from 4 elements to 3 wherever `populate_lights` results are asserted. Remove the `min_lights`/`max_lights`/`light_weights` kwargs from its `HouseConfig(` fixture.
4. `tests/test_devmode_synthetic_scene.py`: the "exactly 10 road segments (rows 0-9)" test becomes 9 segments (rows 0–8); "Segments 6-9" direct-light/fixture-profile/Y-position tests become segments 6–8; delete any house-light row assertions; remove the three `HouseConfig` kwargs from its fixture.
5. `tests/test_config_zones.py`: delete `test_house_config_defaults_min_max_lights` and `test_house_config_defaults_light_weights` (~lines 110–124).
6. Kwarg sweep: in `tests/test_zones.py`, `tests/test_zone_builder.py`, `tests/test_ground_pass.py`, `tests/test_ground_tiles.py`, `tests/test_scene.py`, `tests/test_devmode_real_scene.py`, `tests/test_building_builder.py`, remove the `min_lights=...`, `max_lights=...`, `light_weights=[...]` lines from every `HouseConfig(` construction. Verify none remain: `rg -n "min_lights|max_lights|light_weights" tests/` → no hits.
7. `tests/test_config_loader.py`: add a stale-key regression test:

```python
def test_load_config_ignores_stale_house_light_keys(tmp_path):
    # Arrange — a pre-removal TOML with house-light keys that no longer exist
    stale = tmp_path / "stale.toml"
    stale.write_text(
        "\n".join(
            [
                "[zones.house]",
                "min_lights = 0",
                "max_lights = 5",
                "",
                "[lights.house_lights]",
                "brightness_min = 0.3",
            ]
        )
    )

    # Act
    result = load_config(stale)

    # Assert — loads without error, stale keys simply dropped
    assert not hasattr(result.zones.house, "min_lights")
    assert not hasattr(result.lights, "house_lights")
```

(Check the file's existing fixtures — if `load_config` is imported under a different name or a helper writes TOMLs, follow that pattern.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_house.py tests/test_scene_builder.py tests/test_devmode_synthetic_scene.py tests/test_config_zones.py tests/test_config_loader.py -v`
Expected: FAIL — `HouseConfig` still requires the removed kwargs (`TypeError: missing ... required positional arguments`), `populate_lights` still returns 4 elements, synthetic scene still builds 10 segments.

- [ ] **Step 3: Remove the system from src**

1. `git rm src/streets_at_night/models/house_light.py src/streets_at_night/placement/house_light_placer.py`
2. `src/streets_at_night/config/schema.py` — `HouseConfig`: delete the `min_lights: int`, `max_lights: int`, `light_weights: t.List[float]` fields. (Leave `HouseLightConfig` / `LightConfig.house_lights` for Task 4.)
3. `src/streets_at_night/config/defaults.py` — `"zones"."house"`: delete the `"min_lights"`, `"max_lights"`, `"light_weights"` entries (~lines 59–61).
4. `src/streets_at_night/models/house.py`:
   - `__init__`: delete `self.num_lights = ...` and `self.light_positions = ...`; drop `HouseLightConfig` from the imports; update the class docstring (remove the "light positions"/"get_lights" sentences).
   - Delete `get_lights`, `_pick_num_lights`, `_compute_light_positions`, and the module-level `_place_along_perimeter` and `_inset_toward_centroid`. Keep `_clamp` (yard color uses it).
   - Update the module docstring to `"""House data model — footprint and colors."""`.
5. `src/streets_at_night/placement/scene_builder.py`:
   - Remove `from .house_light_placer import collect_house_lights` and delete `_run_house_light_placer`.
   - `populate_lights` becomes:

```python
def populate_lights(
    scene: Scene, config: SceneConfig, rng: random.Random
) -> t.Tuple[t.List, t.List, t.List]:
    """Run all light placers and add results to the scene.

    Creates deterministic child RNGs per placer from the master seed, then
    runs each placer in turn. Results are added to the scene via add_lights
    and add_car_pairs.

    Args:
        scene (Scene): The scene to populate. Must have road_segments and
            zone lists set before calling.
        config (SceneConfig): Loaded scene configuration with lights section.
        rng (random.Random): Master RNG seeded from config.seed. Child RNGs
            are derived from this to keep each placer independent.

    Returns:
        Tuple[List, List, List]: (streetlights, car_lights,
            commercial_lights) — all generated LightSources by type,
            already added to the scene.
    """
    streetlight_rng, car_rng, commercial_rng = _create_child_rngs(rng)

    streetlights = _run_streetlight_placer(scene, config, streetlight_rng)
    car_lights, car_pairs = _run_car_placer(scene, config, car_rng)
    commercial_lights = _run_commercial_placer(scene, config, commercial_rng)

    scene.add_lights(streetlights)
    scene.add_lights(car_lights)
    scene.add_lights(commercial_lights)
    scene.add_car_pairs(car_pairs)

    return streetlights, car_lights, commercial_lights
```

   - `_create_child_rngs` returns three RNGs (drop the fourth `random.Random(rng.randint(0, 2**32))` and update the return-type hint/docstring).
   - `build_scene`: `streetlights, car_lights, commercial_lights = populate_lights(...)` and `_print_light_summary(streetlights, car_lights, commercial_lights)`.
   - `_print_light_summary`: drop the `house_lights` parameter, its docstring line, its term in `total`, and the `House lights` print line.
   - Module docstring: "...add streetlights, cars, and commercial lights." (drop "and house lights").
6. `src/streets_at_night/devmode/synthetic_scene.py`:
   - Remove `from ..models.house_light import create_house_light`.
   - `ROW_LABELS`: delete `("House Lights",   -300.0),`. Resulting order: Highway, Major Road, Collector, Residential, Service, Cars, Sign Lights, Parking Lights, Storefront, Houses (10 entries, indices 0–9). Keep the remaining y-offsets unchanged.
   - `_build_road_segments`: the surface-segment loop slice `ROW_LABELS[6:10]` → `ROW_LABELS[6:9]`; docstring: nine segments, rows 6–8 are the three direct-light surface rows.
   - `_populate_direct_light_rows`: remove `house_rng`, `house_y`, and the `_create_house_light_row` call; `sign_y = ROW_LABELS[6][1]`, `parking_y = ROW_LABELS[7][1]`, `storefront_y = ROW_LABELS[8][1]`; docstring: "Row 6: sign lights. Row 7: parking lights. Row 8: storefront lights."
   - Delete `_create_house_light_row`.
   - `_build_synthetic_buildings`: `y = ROW_LABELS[10][1]` → `y = ROW_LABELS[9][1]`; the comment above `_BUILDING_TEMPLATES` about short houses growing no window rows is now stale — update it to note short houses get the single-row fallback.
   - Module + `build_synthetic_scene` docstrings: 10 rows total; rows 6–8 direct lights (sign, parking, storefront); row 9 buildings.
7. Confirm nothing else imports the deleted modules: `rg -n "house_light" src/` → only docstring mentions in `models/illumination.py` / `models/light_source.py` may remain (Task 5 cleans those).

- [ ] **Step 4: Run the affected suites, then the full suite**

Run: `uv run pytest tests/test_house.py tests/test_scene_builder.py tests/test_devmode_synthetic_scene.py tests/test_config_zones.py tests/test_config_loader.py -v`
Expected: all PASS.
Run: `uv run pytest tests/`
Expected: all PASS — this catches any file missed in the kwarg sweep or an import of the deleted modules (e.g. `tests/test_zone_builder.py`, UI tab tests building scenes). Fix any straggler the same way before committing. Note `tests/test_light_catalog.py`, `tests/test_dynamic_panel.py`, `tests/test_ui_devmode_tab.py`, and `tests/test_config_lights.py` still reference `config.lights.house_lights` — that field still exists (Task 4 removes it), so they should still pass here.

- [ ] **Step 5: Commit**

```bash
git add -A src/streets_at_night tests
git commit -m "Remove house-light point-light system; houses are lit by windows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Remove the HouseLightConfig config surface and devmode catalog entry

**Files:**
- Modify: `src/streets_at_night/config/schema.py` (delete `HouseLightConfig` ~line 205, `LightConfig.house_lights` ~line 225)
- Modify: `src/streets_at_night/config/defaults.py` (delete the `"house_lights"` block, ~lines 464–481)
- Modify: `src/streets_at_night/devmode/light_catalog.py` (delete the "House Lights" kind, line 103)
- Modify tests: `tests/test_light_catalog.py`, `tests/test_config_lights.py`, `tests/test_dynamic_panel.py`, `tests/test_ui_devmode_tab.py`

**Interfaces:**
- Consumes: Task 3 (nothing in src references `create_house_light`/`collect_house_lights` anymore).
- Produces: `LightConfig` has exactly `streetlights`, `cars`, `commercial`. `build_light_catalog` no longer emits a "House Lights" `LightKind`; the only house-lighting surface in devmode is "Buildings: Windows".

- [ ] **Step 1: Update tests first**

1. `tests/test_light_catalog.py` (~lines 65–66): delete the `by_label["House Lights"]` assertions; if a test asserts the full catalog label list or count, remove "House Lights" from it.
2. `tests/test_config_lights.py`: delete `test_light_config_has_house_lights` (~line 85) and `test_house_light_colors_non_empty` (~line 186), and the `HouseLightConfig` import.
3. `tests/test_dynamic_panel.py` (~lines 103–116, `test_build_dynamic_section_color_pool_entries_prefilled`): the test builds `LightKind("House Lights", config.lights.house_lights, None)` to exercise a color-pool section. Retarget it the same way the test directly above it already does: `kind = LightKind("Commercial: Parking", config.lights.commercial, "parking_")` and `pool = config.lights.commercial.parking_colors`. Keep the assertions otherwise identical.
4. `tests/test_ui_devmode_tab.py` (~lines 109–110): `tab._build_section("House Lights")` → `tab._build_section("Commercial: Signs")` (any surviving catalog label works; keep the test's intent — leave a section and return).

- [ ] **Step 2: Run tests to verify current state**

Run: `uv run pytest tests/test_light_catalog.py tests/test_config_lights.py tests/test_dynamic_panel.py tests/test_ui_devmode_tab.py -v`
Expected: retargeted tests PASS already (they no longer touch house lights); nothing should FAIL except tests that would only fail after the schema change — if everything passes, proceed.

- [ ] **Step 3: Remove the config surface**

1. `src/streets_at_night/config/schema.py`: delete the whole `HouseLightConfig` dataclass and the `house_lights: HouseLightConfig` field on `LightConfig`.
2. `src/streets_at_night/config/defaults.py`: delete the `"house_lights": {...}` block inside `"lights"` (including its comment lines).
3. `src/streets_at_night/devmode/light_catalog.py`: delete `kinds.append(LightKind("House Lights", config.lights.house_lights, None))` and its surrounding blank lines.
4. Verify no live references remain: `rg -n "HouseLightConfig|house_lights" src/ tests/` → zero hits.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/`
Expected: all PASS. Report the pass count.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/config/schema.py src/streets_at_night/config/defaults.py \
        src/streets_at_night/devmode/light_catalog.py \
        tests/test_light_catalog.py tests/test_config_lights.py \
        tests/test_dynamic_panel.py tests/test_ui_devmode_tab.py
git commit -m "Remove HouseLightConfig and the devmode House Lights section

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Clean config TOMLs, docs, and stray docstrings

**Files:**
- Modify: `scene_config.toml` (keys ~lines 46–55, sections ~lines 478–492), `scene_config_NewDevmode.toml` (same shape; file has uncommitted working-tree changes — read it first and edit the current content)
- Modify: `docs/config.md`, `docs/commands/devmode.md`, `docs/commands/render.md`, `CLAUDE.md`
- Modify: `src/streets_at_night/models/illumination.py` (line 20), `src/streets_at_night/models/light_source.py` (line 16)

**Interfaces:**
- Consumes: Tasks 3–4 (schema no longer has the keys being deleted; loader tolerates them anyway, so this is cosmetic hygiene, not correctness).
- Produces: no repo file mentions the house-light system except historical specs/plans.

- [ ] **Step 1: Clean the TOML configs**

In both `scene_config.toml` and `scene_config_NewDevmode.toml`:
1. In `[zones.house]`: delete the `min_lights`, `max_lights`, and multi-line `light_weights = [...]` entries.
2. Delete the entire `[lights.house_lights]` section and its `[lights.house_lights.colors]` subsection (through the end of that table's keys).
3. Add `window_lit_fraction_residential = 0.4` to the `[buildings]` section if the file has one, next to `window_lit_fraction` (check with `rg -n "window_lit_fraction" scene_config.toml scene_config_NewDevmode.toml`; skip if the file has no `[buildings]` section — defaults cover it).

Verify: `uv run python -c "from streets_at_night.config.loader import load_config; load_config('scene_config.toml'); load_config('scene_config_NewDevmode.toml'); print('ok')"`
Expected: `ok`.

- [ ] **Step 2: Update docs**

1. `docs/config.md`: remove the `[lights.house_lights]` TOC entry (line 23), the `min_lights`/`max_lights`/`light_weights` table rows (~lines 136–138), and the whole `## [lights.house_lights]` section (~line 301 onward through its color-pool table). Add a row for `window_lit_fraction_residential` (type float, default 0.4, "Lit fraction for house/detached/residential buildings") to the `[buildings]` window table if the doc has one.
2. `docs/commands/devmode.md` and `docs/commands/render.md`: `rg -n -i "house light" docs/commands/` and rewrite each hit — devmode's editable-sections list drops "House Lights"; render's light summary drops the house-lights line. Mention houses are lit via building windows where a replacement sentence is needed.
3. `CLAUDE.md`:
   - Structure listing: delete the `house_light.py` and `house_light_placer.py` lines; update `house.py`'s description to "House data model — footprint and colors"; also add the missing `building.py` / `building_builder.py` / `building_pass.py` lines while in there (they exist but are absent from the listing).
   - Prose: update "streetlights, cars, commercial and house lights" phrasing in the Product section and the `populate_lights()` line in the render data flow to drop house lights; note houses render as building geometry with a single-row window fallback and a residential lit fraction.
4. `src/streets_at_night/models/illumination.py` line 20 and `src/streets_at_night/models/light_source.py` line 16: drop "house lights" from the example lists in the docstrings.

- [ ] **Step 3: Full suite + smoke check**

Run: `uv run pytest tests/`
Expected: all PASS. Report the count.
Verify no references remain outside specs/plans: `rg -n -i "house.light" src/ docs/ tests/ CLAUDE.md scene_config.toml scene_config_NewDevmode.toml | rg -v "superpowers"` → zero hits.

- [ ] **Step 4: Commit**

```bash
git add scene_config.toml scene_config_NewDevmode.toml docs/config.md docs/commands \
        CLAUDE.md src/streets_at_night/models/illumination.py \
        src/streets_at_night/models/light_source.py
git commit -m "Clean house-light remnants from configs, docs, and docstrings

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
