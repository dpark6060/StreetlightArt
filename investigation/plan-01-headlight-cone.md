# Plan 01 — One directional headlight cone per car

## Part 1 — Answer

### How headlight ground illumination works today

Every car is a `CarLightPair` (`src/streets_at_night/models/car.py`) holding exactly two
`LightSource` instances offset perpendicular to the travel bearing (`_create_lights`,
car.py:112-138). The travel bearing IS available: it's stored on the pair
(`self.travel_bearing`, car.py:59) and derived in the placer from the road tangent
(`_bearing_from_tangent`, `placement/car_placer.py:173-185`; lane/direction assignment at
car_placer.py:188-253). Convention: radians, 0 = East, pi/2 = North, ENU meters.

At render time each caller runs `pair.resolve_facing(camera.get_forward())`
(main.py:445-446 and 651-652, `devmode/render_helpers.py:183-184`,
`ui/tabs/render_tab.py:111-112`). Cars approaching the camera get headlight properties,
receding cars get taillight properties. `_apply_headlight_properties` (car.py:200-211)
sets on BOTH lights:

```python
light.illumination = ConeIllumination(
    self._config.headlight_illumination_radius,      # 15.0 m
    self._config.headlight_illumination_falloff,     # 1.5
    self._config.headlight_illumination_inner_fraction,  # 0.3
)
```

`ConeIllumination` (`models/illumination.py:14-32`) is despite its name a **radially
symmetric** two-zone pool: flat disc out to `inner_fraction * radius`, then a
smoothstep^falloff fade to zero at `radius`. It has no direction. The illumination pass
(`rendering/illumination_pass.py`) paints it in `_paint_illumination_blob`
(illumination_pass.py:404-488): purely a function of radial distance
`dist_m = sqrt((grid_x-cx)^2 + (grid_y-cy)^2) * resolution` (line 470), dispatched to
`_cone_falloff` (lines 519-544) or, for `BeamIllumination` streetlights, to
`_physical_falloff` (lines 491-516). So today each car paints **two overlapping circles**
on the ground, one per headlight.

Supporting plumbing:
- `_effective_pool_radius` (illumination_pass.py:235-264) returns one scalar radius per
  light, shared by AABB culling (`_is_light_visible`, lines 267-301) and the paint window.
- Lights with `illumination is None` are skipped entirely (illumination_pass.py:347).
- Per-light brightness is attenuated by camera-distance extinction before painting
  (lines 478-482); painted color is `srgb255_to_linear(light.color)`.
- Config: `CarConfig` (`config/schema.py:133-164`), defaults in
  `config/defaults.py:358-404`, TOML at `scene_config.toml:341` and
  `scene_config_NewDevmode.toml:348` (`headlight_illumination_radius`).
- No light type currently carries a horizontal direction. `BeamIllumination` is the only
  "directional" model and it only points straight down (angle around a vertical axis).

### Recommended design

**Model.** Add a third illumination model, `DirectionalConeIllumination`, carrying a 2D
unit heading `(ux, uy)` in ENU plus cone shape parameters. Footprint math for a ground
point at world offset `(dx, dy)` from the apex:

```
along   = dx*ux + dy*uy                     # meters ahead of the car (signed)
lateral = -dx*uy + dy*ux                    # meters left/right of the axis
axial falloff   = existing two-zone _cone_falloff(along, length, inner_fraction, falloff),
                  zero where along <= 0 (nothing behind the car)
angular falloff = smoothstep from 1 -> 0 as the off-axis angle
                  atan2(|lateral|, along) sweeps the outer 30% of the half-angle
total = axial * angular
```

This reuses the repo's existing axial falloff curve and adds a soft-edged angular wedge —
a filled circular sector with a feathered rim and a distance fade, which is exactly what a
headlight pool reads as from altitude.

**One cone per car, carried by a third LightSource.** Attaching the cone to one of the
two existing point lights is wrong twice over: `resolve_facing` swaps the point lights'
color/brightness to red taillight values for receding cars (the cone would turn red and
dim), and the pool would be camera-dependent when it is physically a world-space fact.
Instead `CarLightPair` gains a third, illumination-only `LightSource` at the car center:
headlight color/brightness (sampled once at init, never swapped), `size=0`,
`glow_radius=0`, and the `DirectionalConeIllumination`. To keep it from drawing a bright
dot in the light passes (`_core_radius` floors at 1 px — light_pass.py:298-316),
`LightSource` gains `emits_core: bool = True`; both light passes skip `False`. The
illumination pass needs no gating — it already only reads `illumination`.

**Direction source:** `math.cos/sin(self.travel_bearing)` — already stored on the pair.

**Live devmode editing:** the existing headlight sliders work without a scene rebuild
because `_apply_headlight_properties` re-reads `self._config` on every `resolve_facing`.
The cone must do the same: refresh the cone light's model inside `resolve_facing`, not
only in `__init__`.

**New config fields** (on `CarConfig`, `[lights.cars]` in TOML):

| Field | Default | Unit | Meaning |
|---|---|---|---|
| `headlight_beam_length` | 25.0 | m | Cone reach ahead of the car (replaces `headlight_illumination_radius`) |
| `headlight_beam_angle_deg` | 40.0 | deg | Full horizontal cone angle |

`headlight_illumination_falloff`, `headlight_illumination_inner_fraction`, and
`headlight_illumination_intensity` are kept and reused (axial fade exponent, flat
fraction of length, pool strength). `headlight_illumination_radius` is removed —
`serde._generic_from_dict` (config/serde.py:81-90) iterates dataclass fields only, so a
stale key in an old TOML is silently ignored, no migration needed.

**Devmode exposure is automatic.** The "Cars: Headlights" section
(`devmode/light_catalog.py:98`) reflects every `headlight_`-prefixed float on `CarConfig`
via `dynamic_panel.build_dynamic_section` (dynamic_panel.py:61-70). The new
`headlight_beam_angle_deg` already hits the existing `("beam_angle", 10..170)` rule in
`devmode/field_meta.py:46`; add one rule for `beam_length`.

---

## Part 2 — Code change plan

Written for an implementer with repo access and no other context. Repo root:
`/Users/davidparker/Documents/Personal/Projects/Art/StreetsAtNight`. Run tests with
`uv run pytest tests/`. Style: match the existing vectorized numpy style of
`illumination_pass.py`; Google docstrings; `typing as t`; helper prefixes limited to
`get_/set_/is_/validate_/process_/build_/add_/run_/delete_/orchestrate_` (private
helpers `_`-prefixed).

IMPORTANT: `scene_config_NewDevmode.toml` has uncommitted user edits. Edit keys in place;
never regenerate or reformat that file.

### 1. `src/streets_at_night/models/illumination.py`

Add after `BeamIllumination` (line 48) and update the union (line 52):

```python
@dataclass(frozen=True)
class DirectionalConeIllumination:
    """Oriented ground cone cast forward from the light along a heading.

    A wedge-shaped pool on the ground: full intensity near the apex fading
    to zero at ``length`` along the heading (same two-zone axial curve as
    ConeIllumination), feathered to zero at the cone's angular edge, and
    zero everywhere behind the apex. Used for car headlights — one cone per
    car, pointing in the travel direction.

    Attributes:
        direction: Unit (east, north) heading of the cone axis in ENU.
        length: Cone reach in meters along the heading.
        beam_angle_deg: Full horizontal cone angle in degrees.
        falloff: Axial fade exponent (same semantics as ConeIllumination).
        inner_fraction: Fraction of ``length`` at full brightness, in [0, 1].
    """

    direction: t.Tuple[float, float]
    length: float
    beam_angle_deg: float
    falloff: float
    inner_fraction: float


IlluminationModel = t.Union[
    ConeIllumination, BeamIllumination, DirectionalConeIllumination, None
]
```

Update the module docstring's model list.

### 2. `src/streets_at_night/models/light_source.py`

Add one field at the end of `LightSource` (after `illumination`, line 46), plus an
Attributes-docstring entry:

```python
    emits_core: bool = True
```

Docstring: "When False, the light contributes ground illumination only — the light
passes draw no core dot or glow halo for it. Used by illumination-carrier lights such
as the per-car headlight cone."

### 3. `src/streets_at_night/models/car.py`

- Import `DirectionalConeIllumination` alongside `ConeIllumination` (line 10).
- `__init__` (lines 33-74): after `self._lights = self._create_lights(...)`, build the
  carrier:

```python
        self._cone_light = self._make_cone_light(position)
        self._apply_cone_properties()
```

- New private helper (near `_make_placeholder_light`, line 165):

```python
    def _make_cone_light(self, position: np.ndarray) -> LightSource:
        """Create the illumination-only headlight-cone carrier at the car center."""
        return LightSource(
            position=position.copy(),
            color=self._headlight_color,
            brightness=self._headlight_brightness,
            size=0.0,
            glow_radius=0.0,
            illumination_intensity=0.0,
            illumination=None,
            emits_core=False,
        )
```

- New private helper (with the resolve helpers, after `_get_travel_direction`,
  line 190):

```python
    def _apply_cone_properties(self) -> None:
        """Refresh the cone carrier from config so devmode edits apply live."""
        cfg = self._config
        direction = (math.cos(self.travel_bearing), math.sin(self.travel_bearing))
        self._cone_light.illumination_intensity = cfg.headlight_illumination_intensity
        self._cone_light.illumination = DirectionalConeIllumination(
            direction=direction,
            length=cfg.headlight_beam_length,
            beam_angle_deg=cfg.headlight_beam_angle_deg,
            falloff=cfg.headlight_illumination_falloff,
            inner_fraction=cfg.headlight_illumination_inner_fraction,
        )
```

- `resolve_facing` (lines 80-98): add `self._apply_cone_properties()` after the
  headlight/taillight branch (unconditionally — the cone is camera-independent).
- `get_lights` (lines 100-106): return `[*self._lights, self._cone_light]`; update the
  docstring to "Always exactly three lights: two point lights plus the illumination-only
  headlight-cone carrier."
- `_apply_headlight_properties` (lines 200-211): replace the `ConeIllumination`
  assignment with `light.illumination = None` — the point lights no longer carry ground
  pools in headlight mode (the cone carrier does). Taillight mode
  (`_apply_taillight_properties`, lines 213-224) is unchanged: receding cars keep their
  small red pools AND still cast the forward headlight cone.
- Update the class docstring ("a paired set ... always two" is no longer the whole
  story) and the module docstring.

### 4. `src/streets_at_night/rendering/illumination_pass.py`

- Import `DirectionalConeIllumination` next to `BeamIllumination` (line 25).

- `_effective_pool_radius` (lines 235-264): add before the final `return model.radius`:

```python
    if isinstance(model, DirectionalConeIllumination):
        return model.length
```

  (Conservative: the cone fits inside a circle of radius `length` centered on the apex,
  so AABB culling and the paint window stay correct. Update the docstring.)

- `_paint_illumination_blob` (lines 404-488): restructure the delta computation so the
  directional branch gets signed world-space deltas. Replace lines 464-476 with:

```python
    grid_y, grid_x = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )

    # Signed world-space deltas from the light. NOTE the y flip: map row 0 is
    # max_y (north edge), so increasing grid_y means decreasing world y.
    dx_m = (grid_x - cx) * resolution
    dy_m = (cy - grid_y) * resolution
    dist_m = np.hypot(dx_m, dy_m)

    model = light.illumination
    if is_physical:
        falloff = _physical_falloff(dist_m, height, radius_m)
    elif isinstance(model, DirectionalConeIllumination):
        falloff = _directional_cone_falloff(dx_m, dy_m, model)
    else:
        falloff = _cone_falloff(dist_m, radius_m, model.inner_fraction, model.falloff)
```

  `np.hypot(dx_m, dy_m)` is numerically identical to the existing
  `sqrt(dpx^2 + dpy^2) * resolution` for the radial models (the y sign squares away), so
  existing behavior is preserved. Update the function docstring's model list, and flag
  the y-flip in a comment — getting `dy_m`'s sign wrong silently mirrors every cone
  north/south, which no radial test would ever catch.

- New helper next to `_cone_falloff` (after line 544):

```python
# Fraction of the cone half-angle over which the wedge edge feathers to zero.
_CONE_EDGE_SOFTNESS: float = 0.3


def _directional_cone_falloff(
    dx_m: np.ndarray,
    dy_m: np.ndarray,
    model: DirectionalConeIllumination,
) -> np.ndarray:
    """Compute the oriented headlight-cone falloff.

    Projects each ground offset onto the cone axis: ``along`` (meters ahead
    of the apex) drives the same two-zone axial fade as ConeIllumination;
    the off-axis angle drives a smoothstep feather over the outer
    _CONE_EDGE_SOFTNESS of the half-angle. Points behind the apex are zero.

    Args:
        dx_m (np.ndarray): Signed east offsets from the apex, in meters.
        dy_m (np.ndarray): Signed north offsets from the apex, in meters.
        model (DirectionalConeIllumination): Cone geometry and fade params.

    Returns:
        np.ndarray: Falloff multiplier in [0, 1], same shape as dx_m.
    """
    ux, uy = model.direction
    along = dx_m * ux + dy_m * uy
    lateral = -dx_m * uy + dy_m * ux

    axial = _cone_falloff(
        np.maximum(along, 0.0), model.length, model.inner_fraction, model.falloff
    )

    half_angle = math.radians(model.beam_angle_deg / 2.0)
    off_axis = np.arctan2(np.abs(lateral), np.maximum(along, 1e-6))
    edge_start = half_angle * (1.0 - _CONE_EDGE_SOFTNESS)
    edge = np.clip(
        (half_angle - off_axis) / max(half_angle - edge_start, 1e-6), 0.0, 1.0
    )
    edge = edge * edge * (3.0 - 2.0 * edge)

    return np.where(along > 0.0, axial * edge, 0.0).astype(np.float32)
```

### 5. `src/streets_at_night/rendering/light_pass.py`

Skip core-less lights in both passes:

- `render_lights_preview` draw loop (lines 62-67): first statement inside the loop:
  `if not light.emits_core: continue` (before drawing / incrementing `drawn`).
- `render_lights_export` (lines 119-137): same guard at the top of the
  `for i in visible_indices:` loop body, before any accumulation.

(Do not fold the check into `_get_visible_indices` — it takes arrays, not lights, and a
per-light guard in the loop is simpler.)

### 6. `src/streets_at_night/config/schema.py`

In `CarConfig` (lines 133-164): delete `headlight_illumination_radius` (line 156) and
append at the end of the class, with defaults so the many existing
positional-free `CarConfig(...)` constructions in tests keep working:

```python
    headlight_beam_length: float = 25.0
    headlight_beam_angle_deg: float = 40.0
```

(`CarConfig` currently has zero defaulted fields, so adding defaulted fields at the end
is legal. Values must match defaults.py.) Extend the class docstring: cone reach in
meters / full angle in degrees; one cone per car along the travel bearing.

### 7. `src/streets_at_night/config/defaults.py`

In the `"cars"` block (lines 358-404): remove `"headlight_illumination_radius": 15.0`
(line 395) and add:

```python
            "headlight_beam_length": 25.0,
            "headlight_beam_angle_deg": 40.0,
```

### 8. `scene_config.toml` and `scene_config_NewDevmode.toml`

- `scene_config.toml:341`: replace `headlight_illumination_radius = 15.0` with
  `headlight_beam_length = 25.0` and `headlight_beam_angle_deg = 40.0`.
- `scene_config_NewDevmode.toml:348`: replace
  `headlight_illumination_radius = 22.22...` with the two new keys (use
  `headlight_beam_length = 25.0`, `headlight_beam_angle_deg = 40.0` — the old radius was
  a circle, not a beam length; do not carry the value over). Touch nothing else in this
  file — it has uncommitted user edits.

### 9. `src/streets_at_night/devmode/field_meta.py`

Add one rule to `_FIELD_META_RULES` (lines 38-57), adjacent to the existing
`("beam_angle", ...)` entry (order vs. it doesn't matter — the substrings don't
overlap):

```python
    ("beam_length", FieldMeta(0.0, 80.0)),
```

`headlight_beam_angle_deg` already matches `("beam_angle", FieldMeta(10.0, 170.0))`.
No changes to `light_catalog.py` or `dynamic_panel.py` — the `headlight_` prefix section
picks the new floats up by reflection, and `config_writer`/`serde` handle plain floats
generically.

### 10. Tests

Conventions: one file per module, `test_method_scenario_behavior` names, imports at top,
Arrange/Act/Assert, double quotes, no mocking of plain data.

**`tests/test_illumination_pass.py`** — add (reuse the file's existing light-construction
helper style; build `LightSource(..., illumination=DirectionalConeIllumination(...))`):
- `test_directional_cone_falloff_zero_behind_apex` — direction east `(1.0, 0.0)`,
  points with `dx_m < 0` all return 0.
- `test_directional_cone_falloff_full_on_axis_inside_inner_fraction` — on-axis point at
  `0.5 * inner_fraction * length` returns ~1.0.
- `test_directional_cone_falloff_zero_beyond_length` — on-axis point past `length` is 0.
- `test_directional_cone_falloff_zero_outside_half_angle` — point at an off-axis angle
  greater than `beam_angle_deg / 2` returns 0.
- `test_paint_illumination_blob_directional_cone_respects_y_flip` — the load-bearing
  test: light at map center with `direction=(0.0, 1.0)` (north); assert painted energy
  lies in rows ABOVE the center row (`grid_y < cy`) and rows below the center are ~0.
  This catches the y-flip sign bug.
- `test_effective_pool_radius_directional_cone_uses_length` — mirrors the two existing
  `_effective_pool_radius` tests (lines 331-352).

**`tests/test_car.py`** — update and add:
- Fixture `car_config` (lines 33-73) and the three inline `CarConfig(...)` constructions
  (lines ~288, ~342, ~396, ~465): drop `headlight_illumination_radius=`; the new fields
  can ride on their dataclass defaults or be passed explicitly.
- `test_get_lights_returns_exactly_two` → rename to
  `test_get_lights_returns_three_lights` (two point + cone carrier); same for the
  car.py:88 section comment.
- `test_get_lights_cone_carrier_emits_no_core` — third light has `emits_core is False`.
- `test_cone_carrier_direction_matches_travel_bearing` — bearing 0 → cone
  `direction == (1.0, 0.0)` (use `pytest.approx`).
- `test_cone_carrier_uses_headlight_color_and_beam_config` — cone light color equals the
  sampled headlight color; `illumination.length/beam_angle_deg/falloff/inner_fraction`
  come from config.
- `test_resolve_facing_away_keeps_headlight_cone` — after resolving to taillights, the
  cone carrier still holds a `DirectionalConeIllumination` with headlight color.
- `test_resolve_facing_toward_camera_clears_point_light_pools` — after headlight
  resolve, the two point lights have `illumination is None`.
- Update `test_headlight_illumination_radius_comes_from_config` (line 516) → replace
  with `test_headlight_beam_length_comes_from_config` /
  `test_headlight_beam_angle_comes_from_config` asserting the cone carrier's model, and
  re-point `test_headlight_illumination_intensity_comes_from_config` (line 535) at the
  cone carrier.

**`tests/test_car_placer.py`** — `test_place_cars_each_pair_contributes_two_lights`
(line 96) → three lights per pair (rename accordingly). Check other count-based asserts
(`len(lights) == 2 * len(pairs)` style) throughout the file.

**`tests/test_light_pass.py`** — add
`test_render_lights_export_skips_emits_core_false_light` and the preview twin: a bright
`emits_core=False` light in view is not counted in `lights_drawn` and leaves the buffer
dark; a control light with `emits_core=True` is drawn.

**`tests/test_light_source.py`** — `test_light_source_emits_core_defaults_true`.

**`tests/test_field_meta.py`** — add a row to the parametrized rules table (line 30):
`("headlight_beam_length", 25.0, 0.0, 80.0, False)`.

**`tests/test_config_lights.py`** — add
`test_car_beam_fields_valid` (length > 0, 0 < angle < 180 on `loaded_config`).

Also sweep `rg -n "headlight_illumination_radius" src tests` after the change — it must
come back empty.

Run the full suite (`uv run pytest tests/`) and report pass/fail counts.

### Open design decisions (recommendations made)

1. **Receding cars still cast the forward cone** (headlight color, headlight
   brightness), even while showing red taillight points. Recommended: yes — the pool is
   a world-space fact independent of the camera, and it reads well (pools ahead of
   receding cars). If it looks wrong in practice, gate `_apply_cone_properties` behind
   the facing branch instead; one-line change.
2. **Third carrier light + `emits_core` flag** vs. hanging the cone on one of the two
   existing point lights. Recommended: carrier light. The point lights' color/brightness
   are swapped by `resolve_facing` (red for receding cars), so a cone attached to one
   would change color with camera angle. The `emits_core` default of `True` makes the
   flag invisible to every other light type.
3. **Cone apex at car center**, not the front bumper — the ~2 m error is at/below the
   illumination map resolution (2 m/px). Not worth a config field.
4. **`_effective_pool_radius` returns `length`** (circle circumscribing the cone) rather
   than a tight oriented AABB. Wastes some paint-window pixels for narrow cones; correct
   and simple. Optimize only if profiling shows it matters.
5. **Taillight pools unchanged** (small radial `ConeIllumination` per point light).
   Making them a rear cone is out of scope; the model now exists if wanted later.
6. **`headlight_illumination_radius` removed, not deprecated.** serde ignores stale TOML
   keys, both checked-in TOMLs are updated in this change, and keeping a dead field
   would leave a phantom slider in the devmode panel.
