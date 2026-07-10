# Window Atmosphere + Illumination Soft Knee Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Windows dim with distance like every other light and get a default glow; overlapping ground illumination pools compress through a tunable soft knee — both tunable from devmode.

**Architecture:** Window extinction reuses the exact law from `render_utils.apply_brightness_falloff` (`exp(-d/falloff_distance)`), computed once per building and threaded into the window-cell accumulation like `lit_fraction` already is; a new `window_extinction_strength` field scales the exponent (0 = today's behavior). The knee is one vectorized function applied to the illumination map after painting: identity below `knee_start`, asymptotic to `knee_max` above it, disabled when `knee_max <= knee_start`. New `IlluminationConfig` under `[lights.illumination]`; devmode picks both up (window field via the existing `window_` prefix section; knee via a new catalog entry).

**Tech Stack:** Python 3.11, uv, pytest, numpy, pygame_gui devmode panel (reflection-driven; no widget code needed).

**Spec:** `docs/superpowers/specs/2026-07-10-window-haze-illum-knee-design.md`

## Global Constraints

- `uv run pytest tests/...` for all test runs.
- Test conventions (UnitTests.md): imports at top, double quotes, Arrange/Act/Assert, naming `test_methodundertest_scenario_behavior`, parametrize only for pure input→output tables.
- `window_extinction_strength = 0.0` must reproduce pre-change window brightness bit-for-bit (factor exactly 1.0, not ≈1.0).
- Knee must be monotonic (brighter stacks stay brighter) and identity below `knee_start`; `knee_max <= knee_start` disables it entirely.
- Old TOMLs without the new keys/section must load via the defaults merge (no loader changes).
- Working tree has unrelated uncommitted user changes (deleted TOML files) — never stage them; `git add` explicit paths only.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Window extinction + glow default

**Files:**
- Modify: `src/streets_at_night/config/schema.py` (`BuildingConfig`, after `window_glow_px`)
- Modify: `src/streets_at_night/config/defaults.py` (`"buildings"` dict: `window_glow_px` 0.0→1.2, add `window_extinction_strength`)
- Modify: `src/streets_at_night/rendering/building_pass.py` (`render_buildings` loop ~line 154, `_accumulate_building_windows` ~line 700, `_accumulate_window_cell` ~line 749)
- Modify: `src/streets_at_night/devmode/field_meta.py` (`_FIELD_META_RULES`)
- Test: `tests/test_building_pass.py`, `tests/test_building_builder.py` (fixture kwarg), `tests/test_field_meta.py`

**Interfaces:**
- Consumes: `scene.config.atmosphere.falloff_distance` (already read as `atmos` nowhere in building_pass — fetch via `getattr(scene.config, "atmosphere", None)`); per-candidate `distance` already computed in the `render_buildings` loop; `math` is already imported or add it.
- Produces: `BuildingConfig.window_extinction_strength: float` (default 1.0); `_get_window_extinction(distance_m, strength, falloff_distance_m) -> float`; `_accumulate_building_windows(..., extinction: float)` and `_accumulate_window_cell(..., lit_fraction, extinction, force_subpixel, window_buffer)` — `extinction` positioned directly after `lit_fraction`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_building_pass.py` (import `_get_window_extinction` in the existing import block):

```python
def test_get_window_extinction_matches_exponential_law():
    # Act — same law as render_utils.apply_brightness_falloff
    factor = _get_window_extinction(8000.0, 1.0, 8000.0)

    # Assert
    assert factor == pytest.approx(math.exp(-1.0))


def test_get_window_extinction_strength_zero_is_exactly_one():
    # Act
    factor = _get_window_extinction(12000.0, 0.0, 8000.0)

    # Assert — bit-for-bit parity with pre-extinction behavior
    assert factor == 1.0


def test_get_window_extinction_nonpositive_falloff_is_one():
    # Act — degenerate config must not divide by zero
    factor = _get_window_extinction(5000.0, 1.0, 0.0)

    # Assert
    assert factor == 1.0
```

`math` must be imported at the top of the test file if it isn't already.

Add to `tests/test_field_meta.py`, following its existing rule-lookup test pattern:

```python
def test_get_field_meta_extinction_strength_range():
    # Act
    meta = get_field_meta("window_extinction_strength", 1.0)

    # Assert
    assert (meta.min_val, meta.max_val) == (0.0, 2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_building_pass.py -k extinction tests/test_field_meta.py -k extinction -v`
Expected: FAIL — `ImportError: cannot import name '_get_window_extinction'`; the field_meta test fails on whatever fallback range the generic path returns.

- [ ] **Step 3: Implement**

`config/schema.py` — in `BuildingConfig`, directly after `window_glow_px: float`:

```python
    window_glow_px: float
    window_extinction_strength: float
```

`config/defaults.py` — in `"buildings"`: change `"window_glow_px": 0.0,` to `"window_glow_px": 1.2,` and add `"window_extinction_strength": 1.0,` directly after it.

`devmode/field_meta.py` — add to `_FIELD_META_RULES`, BEFORE any broader rule that could substring-match first (check the list order; no current rule matches "extinction_strength" or "strength", so append near the top for clarity):

```python
    ("extinction_strength", FieldMeta(0.0, 2.0)),
```

`rendering/building_pass.py`:

```python
def _get_window_extinction(distance_m: float, strength: float, falloff_distance_m: float) -> float:
    """Compute the atmospheric extinction factor for a building's windows.

    Same exponential law as render_utils.apply_brightness_falloff, with a
    strength knob scaling the exponent: 0.0 disables extinction exactly,
    1.0 matches the attenuation applied to every other light core.

    Args:
        distance_m (float): Camera distance to the building, in meters.
        strength (float): Extinction strength multiplier.
        falloff_distance_m (float): Atmosphere falloff distance in meters.

    Returns:
        float: Multiplier in (0, 1]; exactly 1.0 when strength is 0 or the
        falloff distance is non-positive.
    """
    if strength == 0.0 or falloff_distance_m <= 0.0:
        return 1.0
    return math.exp(-distance_m * strength / falloff_distance_m)
```

(`import math` at the top if not present.)

In `render_buildings`, read the falloff once before the loop (atmosphere is a required config section, but guard like the pass guards `buildings`):

```python
    atmosphere = getattr(scene.config, "atmosphere", None)
    falloff_m = atmosphere.falloff_distance if atmosphere is not None else 0.0
```

and inside the loop, alongside the existing window call:

```python
        extinction = _get_window_extinction(
            distance, building_config.window_extinction_strength, falloff_m
        )
        windows_drawn += _accumulate_building_windows(
            building, facing_walls, distance, view_matrix, proj_matrix, width, height,
            building_config, window_buffer, extinction,
        )
```

(Match the actual current parameter order of `_accumulate_building_windows` — it gained no trailing params since; append `extinction: float` as the last parameter with a docstring line, then pass it to `_accumulate_window_cell` directly after `lit_fraction`.)

In `_accumulate_window_cell`, add `extinction: float` after `lit_fraction` and apply it:

```python
    linear_color = _get_window_linear_color(rng, config)
    brightness = _get_window_brightness(rng, config) * extinction
```

- [ ] **Step 4: Update fixtures and run**

Add `window_extinction_strength=1.0,` to every `BuildingConfig(` construction in tests (`rg -n "BuildingConfig\(" tests/` — expected: `tests/test_building_pass.py`, `tests/test_building_builder.py`). Any test that calls `_accumulate_window_cell` or `_accumulate_building_windows` directly gains the new argument (`rg -n "_accumulate_" tests/`).

Run: `uv run pytest tests/test_building_pass.py tests/test_building_builder.py tests/test_field_meta.py tests/test_config_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/config/schema.py src/streets_at_night/config/defaults.py \
        src/streets_at_night/rendering/building_pass.py src/streets_at_night/devmode/field_meta.py \
        tests/test_building_pass.py tests/test_building_builder.py tests/test_field_meta.py
git commit -m "Windows: distance extinction (tunable strength) + default glow 1.2px

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Illumination soft knee + devmode section

**Files:**
- Modify: `src/streets_at_night/config/schema.py` (new `IlluminationConfig`; `LightConfig.illumination`)
- Modify: `src/streets_at_night/config/defaults.py` (`[lights.illumination]` block)
- Modify: `src/streets_at_night/rendering/illumination_pass.py` (`render_illumination` ~line 110, new `_apply_soft_knee`)
- Modify: `src/streets_at_night/devmode/light_catalog.py` (new kind after the commercial kinds)
- Modify: `src/streets_at_night/devmode/field_meta.py` (`("knee", FieldMeta(0.0, 10.0))` rule)
- Modify: `docs/commands/devmode.md` (one line in the sections list) and `docs/config.md` if it documents `[lights]` subsections (check; add a `[lights.illumination]` table mirroring its house-lights-era style)
- Test: `tests/test_illumination_pass.py`, `tests/test_light_catalog.py`, `tests/test_config_lights.py`, `tests/test_field_meta.py`

**Interfaces:**
- Consumes: `illum_map` (H, W, 3) float32 produced by `_build_illumination_map` (`illumination_pass.py:110`); `scene.config.lights`.
- Produces: `IlluminationConfig(knee_start: float, knee_max: float)`; `LightConfig.illumination: IlluminationConfig` with a `default_factory` so hand-built `LightConfig` instances keep working; `_apply_soft_knee(illum_map: np.ndarray, knee_start: float, knee_max: float) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/test_illumination_pass.py` (follow its existing imports/fixtures):

```python
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (1.5, 1.5),          # exactly at knee_start: identity
        (4.0, 1.5 + 2.5 * 2.5 / (2.5 + 2.5)),   # over=2.5, span=2.5 -> 2.75
        (14.0, 1.5 + 2.5 * 12.5 / (12.5 + 2.5)),  # -> ~3.5833
    ],
)
def test_apply_soft_knee_curve_table(value, expected):
    # Arrange
    illum_map = np.full((2, 2, 3), value, dtype=np.float32)

    # Act
    result = _apply_soft_knee(illum_map, knee_start=1.5, knee_max=4.0)

    # Assert
    assert result[0, 0, 0] == pytest.approx(expected, rel=1e-5)


def test_apply_soft_knee_is_monotonic_above_start():
    # Arrange — stacked pools must stay brighter than fewer pools
    values = np.array([[[2.0, 4.0, 8.0]]], dtype=np.float32)

    # Act
    result = _apply_soft_knee(values, knee_start=1.5, knee_max=4.0)

    # Assert
    assert result[0, 0, 0] < result[0, 0, 1] < result[0, 0, 2]
    assert result[0, 0, 2] < 4.0


def test_apply_soft_knee_disabled_when_max_not_above_start():
    # Arrange
    illum_map = np.full((1, 1, 3), 9.0, dtype=np.float32)

    # Act
    result = _apply_soft_knee(illum_map, knee_start=2.0, knee_max=2.0)

    # Assert — identity, same values back
    assert np.array_equal(result, illum_map)
```

`tests/test_light_catalog.py` (mirror the existing per-kind assertions):

```python
def test_build_light_catalog_includes_illumination_kind(config):
    # Act
    kinds = build_light_catalog(config)
    by_label = {kind.label: kind for kind in kinds}

    # Assert
    assert by_label["Illumination: Ground Pools"].target is config.lights.illumination
    assert by_label["Illumination: Ground Pools"].field_prefix is None
```

(Adapt the fixture name to the file's existing `config` fixture.)

`tests/test_config_lights.py`:

```python
def test_light_config_has_illumination_knee_defaults(loaded_config: SceneConfig):
    # Assert
    assert loaded_config.lights.illumination.knee_start == 1.5
    assert loaded_config.lights.illumination.knee_max == 4.0
```

(`IlluminationConfig` import if the file asserts isinstance elsewhere — match its style.)

`tests/test_field_meta.py`:

```python
def test_get_field_meta_knee_range():
    # Act
    meta = get_field_meta("knee_start", 1.5)

    # Assert
    assert (meta.min_val, meta.max_val) == (0.0, 10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_illumination_pass.py -k knee tests/test_light_catalog.py -k illumination tests/test_config_lights.py -k illumination tests/test_field_meta.py -k knee -v`
Expected: FAIL — `_apply_soft_knee` import error, missing catalog label, missing config attribute.

- [ ] **Step 3: Implement**

`config/schema.py` — new dataclass directly above `LightConfig`:

```python
@dataclass
class IlluminationConfig:
    """Ground illumination-map post-processing parameters.

    The soft knee compresses stacked (overlapping) light pools: values up to
    ``knee_start`` pass through unchanged; above it they compress smoothly
    and asymptotically approach ``knee_max``, so overlapping pools still read
    brighter than a single pool but can never blow out. Setting
    ``knee_max <= knee_start`` disables the knee.
    """

    knee_start: float = 1.5
    knee_max: float = 4.0
```

and on `LightConfig` (import `field` is already imported in the module):

```python
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
```

`config/defaults.py` — inside `"lights"`, after the `"commercial"` block:

```python
        "illumination": {
            "knee_start": 1.5,
            "knee_max": 4.0,
        },
```

`rendering/illumination_pass.py` — new function near the map builder:

```python
def _apply_soft_knee(illum_map: np.ndarray, knee_start: float, knee_max: float) -> np.ndarray:
    """Compress stacked illumination above knee_start toward knee_max.

    Identity below ``knee_start``; above it, values compress smoothly
    (C1-continuous at the knee) and asymptotically approach ``knee_max`` —
    monotonic, so brighter stacks always stay brighter. Disabled (returns
    the input unchanged) when ``knee_max <= knee_start``.

    Args:
        illum_map (np.ndarray): (H, W, 3) float32 linear illumination map.
        knee_start (float): Intensity where compression begins.
        knee_max (float): Asymptotic intensity ceiling.

    Returns:
        np.ndarray: The compressed map (may be the input array when disabled).
    """
    if knee_max <= knee_start:
        return illum_map
    span = knee_max - knee_start
    over = np.maximum(illum_map - knee_start, 0.0)
    compressed = knee_start + span * over / (over + span)
    return np.where(illum_map > knee_start, compressed, illum_map).astype(np.float32)
```

In `render_illumination`, right after `_build_illumination_map(...)` returns (~line 110):

```python
    knee = scene.config.lights.illumination
    illum_map = _apply_soft_knee(illum_map, knee.knee_start, knee.knee_max)
```

(If `scene.config.lights` may be absent in some devmode/test scenes, guard with `getattr` the way the pass guards other config access — check how the function reads `scene.config.atmosphere` at ~line 112 and follow that pattern.)

`devmode/light_catalog.py` — after the commercial kinds, before the buildings-windows block:

```python
    kinds.append(LightKind("Illumination: Ground Pools", config.lights.illumination, None))
```

`devmode/field_meta.py` — add `("knee", FieldMeta(0.0, 10.0)),` to `_FIELD_META_RULES`.

Docs: one line in `docs/commands/devmode.md`'s section list ("Illumination: Ground Pools — soft-knee compression for overlapping ground light pools (knee start / max)"); check `docs/config.md` for a `[lights]` area and add a small `[lights.illumination]` table (`knee_start` 1.5, `knee_max` 4.0, one-line descriptions).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_illumination_pass.py tests/test_light_catalog.py tests/test_config_lights.py tests/test_field_meta.py tests/test_config_loader.py -v`
Expected: all PASS.
Then the full suite: `uv run pytest tests/`
Expected: all PASS; report the count.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/config/schema.py src/streets_at_night/config/defaults.py \
        src/streets_at_night/rendering/illumination_pass.py src/streets_at_night/devmode/light_catalog.py \
        src/streets_at_night/devmode/field_meta.py docs/commands/devmode.md docs/config.md \
        tests/test_illumination_pass.py tests/test_light_catalog.py tests/test_config_lights.py tests/test_field_meta.py
git commit -m "Illumination: tunable soft knee for stacked ground pools + devmode section

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
