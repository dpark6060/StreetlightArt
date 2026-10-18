# Plan 03 — Streetlight ground-pool size: research, units, recommended values, unit-label feature

Investigation date: 2026-07-10. Config under scrutiny: `scene_config_NewDevmode.toml` (the devmode
working config, currently modified on branch `UI-rework`).

---

## Part 1 — Answers

### 1a. Real-world photometrics: how big should a streetlight pool actually be?

**Mounting heights (US practice):**

| Road class | Mounting height | Source |
|---|---|---|
| Residential / local | 20–26 ft (6–8 m); municipal specs commonly 25 ft, some cap at 20 ft | [Lightway](https://www.lightwaytraffic.com/what-is-the-standard-height-for-a-street-light/), [Chandler AZ TDM6](https://www.chandleraz.gov/sites/default/files/documents/imported/UDM_TDM6.pdf) |
| Collector | ~30 ft (9 m) at intersections | [Chandler AZ TDM6](https://www.chandleraz.gov/sites/default/files/documents/imported/UDM_TDM6.pdf) |
| Arterial / major road | 30–40 ft (9–12 m); non-residential max 40 ft | [Chandler AZ TDM6](https://www.chandleraz.gov/sites/default/files/documents/imported/UDM_TDM6.pdf), [Inlux](https://www.inluxsolar.com/how-tall-are-street-lights/) |
| Freeway / highway | 30–70 ft (9–21 m, incl. high-mast) | [FHWA Lighting Handbook](https://highways.dot.gov/safety/other/visibility/fhwa-lighting-handbook-august-2012/7-lighting-application) |

The project's `height_min`/`height_max` values (residential 5–7 m, collector 7–9 m, major 8–10 m,
highway 12–15 m) already match reality well. **The pole heights are not the problem.**

**Lit-pool radius vs mounting height (MH):** the IES roadway classifications define lateral throw as
the distance to the *half-maximum-candlepower* trace, in multiples of MH:

- Type II: half-max trace falls at 1.0–1.75 MH street-side (narrow roads, residential)
- Type III: 1.75–2.75 MH (wider roads, arterials, parking)
- Longitudinal: "Short" = 1.0–2.25 MH, "Medium" = 2.25–3.75 MH each way along the road

Sources: [AGi32 IES road report reference](https://docs.agi32.com/PhotometricToolbox/Content/Road_Report_Tool/road_report_summary.htm),
[LED Light Expert Type II guide](https://www.ledlightexpert.com/type-ii-light-pattern-street-light-guide),
[Penglight IESNA distribution types](https://www.penglight.com/what-are-light-distribution-types-defined-by-iesna-and-nema/).

So the *useful/visible* pool extends roughly **1–2× mounting height from the pole**; beyond ~2.75 MH
the luminaire is below half-max output by definition even for the widest street-side throws. Pole
spacing rules of thumb are 2.5–3× pole height for continuous coverage
([Lightmart](https://www.lightmart.com/blog/the-guide-to-light-pole-spacing-coverage/)) and municipal
design starts at 7:1 spacing:MH for non-continuous residential lighting
([CDOT lighting guidelines](https://www.codot.gov/safety/traffic-safety/assets/documents/cdot-lighting-design-guideline.pdf)) —
meaning residential pools are isolated islands, not overlapping strips.

**Expected pool radii (radius ≈ 1.25–2 × MH):**

| Road class | MH | Realistic pool radius | Pool diameter |
|---|---|---|---|
| Residential | 6–8 m | 6–10 m | 12–20 m |
| Collector | 7–9 m | 9–14 m | 18–28 m |
| Arterial | 9–12 m | 14–24 m | 28–48 m |
| Freeway | 12–15 m | 18–30 m | 36–60 m |

**The visual reference — house size:** typical new-construction US suburban lots are ~8,500–11,000
sq ft with 50–60 ft (15–18 m) frontage (common lot 60'×120'), and after 5 ft side setbacks the house
footprint width is ~40 ft (**~12 m**). Sources:
[SellofLand lot-size guide](https://www.sellofland.com/post/how-big-is-a-lot-of-land),
[Infinity Construction footprint calculator](https://infinityconstructiongc.com/what-size-house-will-fit-on-my-lot).

**So a realistic residential pool (12–20 m diameter) should look about one house wide.** That's the
target the render should hit.

### Why the current render's pools look too big

Streetlight pools are physical: `radius = pole_height × tan(beam_angle_deg / 2)`
(`illumination_pass.py:255-264`). The current beam angles in `scene_config_NewDevmode.toml` put
`tan(beam/2)` at 2.5–3.5 (vs the realistic 1.25–2), and the collector value is catastrophically broken:

| Class (devmode TOML) | height | beam_angle_deg | tan(beam/2) | **Current pool radius** | Realistic |
|---|---|---|---|---|---|
| Highway (lines 96, 113) | 12–15 m | 147.70° | 3.45 | **41–52 m** (dia 83–104 m) | 18–30 m |
| Major road (130, 147, 164) | 8–10 m | 147.0° | 3.38 | **27–34 m** (dia 54–68 m) | 14–24 m |
| Collector (181, 198, 215) | 7–9 m | **179.0°** | 114.6 | **802–1031 m** (dia 1.6–2.1 km!) | 9–14 m |
| Residential (235, 252, 269, 286) | 5–7 m | 136.0° | 2.48 | **12–17 m** (dia 25–35 m) | 6–10 m |
| Service (306, 323) | 4–6 m | 138.31° | 2.63 | **10–16 m** (dia 21–32 m) | 5–8 m |

Three compounding problems:

1. **Collector beam 179°** is a devmode slider accident (the `field_meta.py:46` "beam_angle" rule
   caps at 170, but `_contain_value` widening let 179 through). tan(89.5°) = 114.6, so every collector
   light paints a ~900 m radius blob. Besides the visual wash, `illumination_pass.py:99-102` sets the
   map padding to `max(200, max_light_radius)` — one 179° light inflates the whole illumination map
   by ~1 km on every side.
2. Highway/major beams ~147–148° give tan ≈ 3.4, pools 2× too wide. Residential 136° (tan 2.48) makes
   the pool diameter ~30 m — two-plus house frontages instead of one.
3. **`illumination_intensity` was cranked to 1.48–2.13** (design range is 0–1, `field_meta.py:49`;
   defaults are 0.4–0.9). The physical falloff `1/(1+(r/h)²)^1.5` (`illumination_pass.py:511`) has a
   long dim skirt; multiplying it by ~2 lifts that skirt above visibility, so the *perceived* pool
   edge moves outward even at a fixed radius.

### 1b. Units — definitive answer

**Every radius/height/spacing number in the config and devmode text boxes is world-space ENU meters.**
Nothing is in pixels or degrees except `beam_angle_deg` (degrees) and `color_temp_k` (Kelvin).

Evidence:

- `models/streetlight.py:30-37` — `FixtureProfile` docstring: "height: Pole height **in meters**",
  "glow_radius: Atmospheric glow halo radius **in meters**", "spacing: Distance between consecutive
  lights **in meters**". Positions are "3-element (x, y, z) ENU position **in meters**"
  (`streetlight.py:126-127`).
- `models/illumination.py:23,30` — `ConeIllumination.radius`: "Ground pool radius **in meters**".
  This is the model behind every `*_illumination_radius` box (cars: `models/car.py:207,220`;
  commercial signs/parking/storefronts: `models/commercial.py:47,89,130`).
- `rendering/illumination_pass.py:255-264` — `_effective_pool_radius` returns
  `height * math.tan(radians(beam_angle_deg / 2.0))` where height is the light's ENU z in meters;
  docstring: "Effective ground pool radius **in meters**".
- `rendering/illumination_pass.py:448-450` — meters are converted to map pixels only at paint time:
  `radius_px = radius_m / resolution`, with `resolution` in meters/pixel
  (`_BASE_ILLUMINATION_RESOLUTION = 2.0` m/px, `illumination_pass.py:36-37,107`). Ground distances
  are likewise meters: `dist_m = sqrt(...) * resolution` (`illumination_pass.py:470`).
- `rendering/light_pass.py:319-330` — `glow_radius` is world-space meters, perspective-projected to
  screen pixels via `glow_radius / distance * focal_length`; docstring: "World-space glow radius
  **in meters**".

One subtlety: **streetlights have no radius box at all.** Their pool size is derived from two boxes
the panel does show — `Height Min`/`Height Max` (meters) and `Beam Angle Deg` (degrees). Only cars
and commercial lights expose a direct `... Illumination Radius` box (meters).

Dead field found along the way: `atmosphere.illumination_min_radius` (`config/schema.py:269`,
`config/defaults.py:480`) is defined but referenced nowhere in the rendering code. Candidate for
removal in a cleanup pass.

### 1c. Recommended devmode values

Devmode sections are named `Streetlight: <Road Type> (pool N)` (`devmode/light_catalog.py:76`); the
pool entries within a road class currently carry identical values, so **type the same numbers into
every pool of that class** (Highway pools 0–1, Major Road 0–2, Collector 0–2, Residential 0–3,
Service 0–1). Panel field labels are the title-cased field names (`dynamic_panel.py:88-101`).

Target: pool radius ≈ 1.2–1.9 × pole height, i.e. `beam_angle_deg = 2·atan(k)` for k in that range.

| Section | Field (as shown in panel) | Current | **Recommended** | Resulting pool radius |
|---|---|---|---|---|
| Streetlight: Highway (pools 0–1) | Beam Angle Deg | 147.70 | **120** | 21–26 m (was 41–52) |
| | Illumination Intensity | 1.481 | **0.9** | |
| Streetlight: Major Road (pools 0–2) | Beam Angle Deg | 147.0 | **115** | 12.6–15.7 m (was 27–34) |
| | Illumination Intensity | 0.75 | 0.75 (keep) | |
| Streetlight: Collector (pools 0–2) | Beam Angle Deg | **179.0** | **110** | 10.0–12.9 m (was 802–1031) |
| | Illumination Intensity | 2.130 | **0.65** | |
| Streetlight: Residential (pools 0–3) | Beam Angle Deg | 136.0 | **100** | 6.0–8.3 m (was 12.4–17.3) |
| | Illumination Intensity | 1.852 | **0.5** | |
| Streetlight: Service (pools 0–1) | Beam Angle Deg | 138.31 | **100** | 4.8–7.2 m (was 10.5–15.8) |
| | Illumination Intensity | 1.944 | **0.4** | |

Leave `Height Min`/`Height Max` alone — they're already realistic and they feed both the radius and
the physical falloff shape. With the recommended residential values the pool diameter is 12–17 m —
almost exactly one house frontage (12–15 m), which is what real aerial night photos show.

Secondary (cone-model) radii in the same config that are oversized relative to their defaults:

| Section | Field | Current | Default | Suggested |
|---|---|---|---|---|
| Cars: Headlights | Illumination Radius | 22.2 | 15.0 | **10–15** (low-beam hot pool on asphalt) |
| Commercial: Parking | Illumination Radius | 71.7 | 30.0* | **20–25** (25–40 ft lot poles ≈ 1.5–2 MH ≈ 15–25 m) |
| Commercial: Signs | Illumination Radius | 13.3 | 8.0* | 8–12 |
| Commercial: Storefronts | Illumination Radius | 14.8 | 12.0* | 10–14 |

*Defaults from `config/defaults.py` `[lights.commercial]` block; verify exact numbers when editing —
the parking value at 71.7 m is the standout offender (a 143 m diameter pool per lot fixture).

Perception note: after typing these in, the pools will also read *dimmer* at the edge because the
intensity multipliers return to ≤ 1.0. If overall ground light feels too weak afterward, raise
`Atmosphere → illumination_brightness` (global, currently 60/255) rather than re-inflating
per-fixture intensities — that preserves relative pool sizes.

---

## Part 2 — Code change plan: unit suffixes in devmode panel labels

Written to be executed standalone by a code-writing agent with repo access. No behavior changes
outside label text. Do not change any config values as part of this plan (see Optional section).

### Problem

Devmode panel rows show bare field names ("Glow Radius", "Height Min", "Falloff Distance") with no
indication that the values are meters. Add a unit suffix, e.g. "Glow Radius (m)".

### Design

Units live in `devmode/field_meta.py` as an ordered substring-rule list — the exact pattern already
used there for slider ranges (`_FIELD_META_RULES`, `field_meta.py:38-57`), so ordering semantics and
testing style are established. A new `get_field_unit(field_name) -> str` is consumed by
`dynamic_panel._display_name` (all reflected light sections) and by
`panel_sections._add_slider_row` (the hand-written Atmosphere / Devmode Camera sections).

Important constraint: `SliderDescriptor.label` is an identity key — it's the dirty-set key
(`slider_registry.py:64,83-89`) and in the dynamic panel it's `f"{kind.label}:{field_name}"`
(`dynamic_panel.py:205-211`). **Do not bake units into `descriptor.label`**; only change the text
passed to `UILabel`.

### Change 1 — `src/streets_at_night/devmode/field_meta.py`

Append after `_FIELD_META_RULES` (module currently ends at line 124):

```python
# Ordered (substring, unit) rules for display labels. First substring match
# wins; fields with no match are unitless. Kept separate from
# _FIELD_META_RULES because ranges and presentation change independently.
_FIELD_UNIT_RULES: t.List[t.Tuple[str, str]] = [
    ("color_temp", "K"),        # color_temp_k and color_temp_shift_k
    ("beam_angle", "deg"),
    ("exposure", "stops"),
    ("distance", "m"),          # falloff_distance, scatter_distance
    ("radius", "m"),            # glow_radius, *_illumination_radius
    ("height", "m"),
    ("spacing", "m"),
    ("separation", "m"),
    ("density", "m/car"),       # car density fields are meters between cars
    ("pos_", "m"),              # devmode camera pos_x/y/z
    ("target_", "m"),           # devmode camera target_x/y/z
    ("fov", "deg"),
]


def get_field_unit(field_name: str) -> str:
    """Return the display unit for a field, or "" when unitless.

    Args:
        field_name (str): The dataclass field name being displayed.

    Returns:
        str: Unit string (e.g. "m", "deg", "K"), empty when no rule matches.
    """
    for substring, unit in _FIELD_UNIT_RULES:
        if substring in field_name:
            return unit
    return ""
```

Notes for the implementer:

- `("radius", "m")` intentionally also matches `vignette_radius` and `StarsConfig.radius`; neither is
  currently shown in a devmode section (atmosphere section lists fields explicitly,
  `panel_sections.py:68-79`), so no wrong label appears. If `vignette_radius` is ever added to the
  panel, insert a `("vignette_radius", "")` rule above it.
- `("density", "m/car")` is correct: car densities are "meters between cars"
  (`placement/car_placer.py:123`, `config/defaults.py:383`).
- Update the module docstring (lines 1-16) with one sentence mentioning the unit rules.

### Change 2 — `src/streets_at_night/devmode/dynamic_panel.py`

Replace `_display_name` (`dynamic_panel.py:88-101`). Match the unit against the **full** field name
(prefix included) so `headlight_glow_radius` with prefix `headlight_` still resolves. Strip a
trailing title-cased token that duplicates the unit so "Color Temp K" becomes "Color Temp (K)" and
"Beam Angle Deg" becomes "Beam Angle (deg)", not "Color Temp K (K)".

```python
def _display_name(field_name: str, prefix: t.Optional[str]) -> str:
    """Build a human label: strip prefix, title-case, append the unit suffix.

    The unit is looked up from the full field name via field_meta.get_field_unit.
    A trailing name token that spells the unit ("Color Temp K", "Beam Angle
    Deg") is dropped before the suffix is appended, so units are never doubled.

    Args:
        field_name (str): Dataclass field name.
        prefix (str | None): Prefix to strip, or None.

    Returns:
        str: Display label (e.g. "Glow Radius (m)", "Color Temp (K)").
    """
    name = field_name
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    label = name.replace("_", " ").title()
    unit = get_field_unit(field_name)
    if not unit:
        return label
    if label.lower().endswith(f" {unit.lower()}"):
        label = label[: -(len(unit) + 1)]
    return f"{label} ({unit})"
```

Import: add `get_field_unit` to the existing `from .field_meta import get_field_meta` line
(`dynamic_panel.py:23`).

`_display_name` is called once per field in `build_dynamic_section` (`dynamic_panel.py:65`) and the
result flows to the `UILabel` in `_add_slider_row`/`_add_bool_row`/`_add_text_row` and the color-pool
header — every row type gets the suffix for free. Bool and str fields have no matching unit rules,
so they are unchanged.

### Change 3 — `src/streets_at_night/devmode/panel_sections.py`

The Atmosphere and Devmode Camera sections use `descriptor.label` directly as the `UILabel` text
(`panel_sections.py:170-175`). In `_add_slider_row`, change the label construction only:

```python
from .field_meta import get_field_unit          # add to imports (line ~9)

# inside _add_slider_row, replacing text=descriptor.label:
unit = get_field_unit(descriptor.config_path[-1])
label_text = f"{descriptor.label} ({unit})" if unit else descriptor.label
...
    text=label_text,
```

Do not modify the `SliderDescriptor` instances or `register_slider` call — the descriptor label must
stay the raw identity string. This gives "falloff_distance (m)", "scatter_distance (m)",
"pos_x (m)" … "fov (deg)"; unitless fields (`glow_max_alpha`, `haze_alpha`, etc.) are untouched.
`_add_bool_button_row` and `_add_text_entry_row` need no change (booleans/colors have no units).

### Tests

Per repo conventions (one test file per module, `# Radical` prefix, imports at top, parametrize only
for pure input→output tables).

**`tests/test_field_meta.py`** — add a parametrized table test (this is a pure input→output table,
so parametrization is allowed):

```python
@pytest.mark.parametrize(
    "field_name, expected_unit",
    [
        ("glow_radius", "m"),
        ("headlight_illumination_radius", "m"),
        ("height_min", "m"),
        ("spacing_max", "m"),
        ("falloff_distance", "m"),
        ("separation_min", "m"),
        ("density_highway_min", "m/car"),
        ("beam_angle_deg", "deg"),
        ("fov", "deg"),
        ("color_temp_k", "K"),
        ("color_temp_shift_k", "K"),
        ("exposure", "stops"),
        ("pos_x", "m"),
        ("brightness_min", ""),
        ("illumination_intensity", ""),
        ("side", ""),
    ],
)
def test_get_field_unit_matches_rules(field_name, expected_unit):
    assert get_field_unit(field_name) == expected_unit
```

Add `get_field_unit` to the file's imports.

**`tests/test_dynamic_panel.py`** — the display label is only observable on the `UILabel` widgets in
`registry.extra_widgets`, so assert through the public `build_dynamic_section` (existing headless
fixtures `headless_manager` / `scroll` / `config` at `test_dynamic_panel.py:21-33` already cover
setup). Add:

```python
def test_build_dynamic_section_labels_include_units(headless_manager, scroll, config):
    # Arrange
    profile = config.lights.streetlights.highway[0].profile
    kind = LightKind("Streetlight: Highway (pool 0)", profile, None)
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)
    texts = {w.text for w in registry.extra_widgets if isinstance(w, pygame_gui.elements.UILabel)}

    # Assert — metered fields get (m), unit-named fields collapse, unitless stay bare.
    assert "Height Min (m)" in texts
    assert "Glow Radius (m)" in texts
    assert "Beam Angle (deg)" in texts
    assert "Color Temp (K)" in texts
    assert "Illumination Intensity" in texts
```

Also update the two existing label-sensitive spots if they fail: none of the current tests assert
display text (they assert `descriptor.label`, which this plan does not change — see
`test_dynamic_panel.py:80-81`), so no existing test should need edits.

**`panel_sections`** has no test file today; creating `tests/test_panel_sections.py` just for the
suffix would be new scope — acceptable to skip, but if added, mirror the dynamic-panel test: build
the atmosphere section with the headless manager and assert `"falloff_distance (m)"` appears among
`registry.extra_widgets` label texts.

Run the full suite afterward (`uv run pytest tests/`) and report pass/fail counts.

### Optional section — default value changes (separate commit, only if requested)

The label feature above is safe on its own. If the recommended pool sizes should become code
defaults rather than hand-typed devmode edits:

1. `config/defaults.py` `[lights.streetlights]` `beam_angle_deg` per profile:
   highway 150.0 → **120.0** (defaults.py:113,130), major_road 148.0 → **115.0** (:150,167,184),
   collector 140.0 → **110.0** (:204,221,238), residential 136.0 → **100.0** (:260,277,294,311),
   service 135.0 → **100.0** (:332,349). Leave `illumination_intensity` defaults as-is (0.4–0.9 —
   they were never the problem; the devmode TOML drifted).
2. `devmode/field_meta.py:46` — tighten the "beam_angle" rule from `FieldMeta(10.0, 170.0)` to
   `FieldMeta(60.0, 150.0)`. At 170° the derived radius is already 11.4× pole height (~90 m for a
   collector); the current cap invites exactly the 179° accident found in the working TOML. The
   widening logic (`_contain_value`) still lets an existing out-of-range TOML value display without
   clamping, so this is not a data-loss change.
3. `scene_config.toml` / `scene_config_NewDevmode.toml`: same beam values as (1); in the NewDevmode
   file also restore `illumination_intensity` to 0.9/0.75/0.65/0.5/0.4 per class (current values
   1.48–2.13 are slider-drift). These TOMLs are user-authored working files — confirm with the user
   before editing, per the "respect user changes" rule.
4. Cleanup candidate (separate change): delete dead `atmosphere.illumination_min_radius`
   (`config/schema.py:269`, `config/defaults.py:480`) — unreferenced by any render pass.
