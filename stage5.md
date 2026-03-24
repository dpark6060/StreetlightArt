# Stage 5: Light Objects, Placement & Rendering

## Goal
Build the light object hierarchy (StreetLight subtypes, CarLightPair, CommercialLight subtypes,
HouseLight), place them in the scene using road segments and zone objects from stage 4, and
render them with glow effects and ground illumination.

This is the core artistic stage. See `object_model.md` for the full class hierarchy and
`.claude/work/architecture-plan.md` for module responsibilities.

## Part 1: Light Object Classes

### File Structure
```
src/streets_at_night/
    models/
        streetlight.py       # StreetLight base + HighwayLight, MajorRoadLight,
                             #   CollectorLight, ResidentialLight subclasses
                             #   Factory: StreetLight.create_for_road_type()
        car.py               # CarLightPair class (stores travel_bearing, not camera-relative)
        commercial.py        # SignLight, ParkingLotLight, StorefrontLight
        house_light.py       # HouseLight class (created by House.get_lights())
    config/
        defaults.py          # Add all light-specific default configs
```

### FixtureProfile & FixturePool (per-road-segment lighting)

Streetlights are consistent **per road segment**. A given street has all the same fixture
type at the same spacing. Variation happens between streets, not within them.

**FixtureProfile** defines the lighting character of an entire road segment:
- fixture_color (single RGB — all lights on this road match)
- height, brightness, glow_radius, illumination_radius, illumination_intensity
- spacing (meters between lights)
- side ("both", "one", "alternating")

**FixturePool** is a weighted list of possible FixtureProfiles per road type. One profile
is rolled per RoadSegment at construction time and stored in `road_segment.fixture_profile`.
If the roll lands on "no lights", fixture_profile is None and the placer skips that segment.

Per-road-type pools (configurable in TOML):

**Highway FixturePool** (always has lights):
- 90% high-pressure sodium: #FFA040, height 12-15m, spacing 40-60m, both sides, bright
- 10% LED retrofit: #E0E8FF, same geometry

**Major Road FixturePool** (always has lights):
- 55% sodium: #FFA040, height 8-10m, spacing 35-50m, both sides
- 35% LED white: #E0E8FF
- 10% warm LED: #FFE080

**Collector FixturePool**:
- 40% sodium: #FFA040, height 7-9m, spacing 45-65m, alternating sides
- 40% LED white: #E0E8FF
- 20% warm yellow: #FFE080

**Residential FixturePool** (most variety, including no lights):
- 25% no lights at all
- 30% warm yellow: #FFE080, height 5-7m, spacing 80-120m, one side
- 25% LED white: #E0E8FF
- 15% sodium: #FFA040
- 5% decorative warm: #FFCC80, shorter 4-5m poles, spacing 60-80m, both sides

**Service FixturePool**:
- 70% no lights
- 20% LED white: short poles, wide spacing
- 10% sodium

### StreetLight Hierarchy

**StreetLight(LightSource)** — base for road-mounted lights:
- All fields (color, height, brightness, glow, illumination) come from the parent
  RoadSegment's FixtureProfile. No per-light color randomization.
- Factory: `StreetLight.create_from_profile(fixture_profile, position, road_type)`

Subclasses exist for road-type-specific placement behavior:

**HighwayLight(StreetLight)**: tall, bright, always present
**MajorRoadLight(StreetLight)**: medium, always present
**CollectorLight(StreetLight)**: between major and residential
**ResidentialLight(StreetLight)**: shorter, dimmer, 25% of segments have none

### CarLightPair
Always instantiated as a pair. Never a single light.

**Important**: Cars store absolute `travel_bearing`, NOT camera-relative facing. Headlight
vs taillight is resolved at **render time** via `resolve_facing(camera_forward)`. This
keeps placement valid when the camera moves in the preview.

On init:
- Takes: position on road centerline, travel_bearing (absolute radians), config, rng
- Picks separation from RangeParam(1.6, 2.4) — roughly car width
- Computes two light positions offset perpendicular to travel_bearing by ±(separation/2)
- Creates two LightSource instances with placeholder color (resolved at render time)

`resolve_facing(camera_forward)`:
- Computes dot product of travel_bearing vs camera look direction
- If car traveling toward camera → sets headlight colors from headlight pool:
  - 50% bright white (#FFFFFF), 30% warm white (#FFF4E0), 15% yellow (#FFEE80),
    5% blue-white HID (#CCE0FF)
  - brightness: 0.7-0.9, glow_radius: 8m
  - Forward illumination cone: 15m ahead, 8m wide
- If car traveling away → sets taillight colors from taillight pool:
  - 60% red (#FF2020), 25% dark red (#CC1010), 15% red-orange (#FF3010)
  - brightness: 0.3-0.5, glow_radius: 3m
  - Minimal illumination

`get_lights()` — returns the two LightSource instances

Car density per road type (configurable in TOML):
- Highway: RangeParam(25, 50)m between cars, both directions
- Major road: RangeParam(40, 80)m
- Collector: RangeParam(80, 150)m
- Residential: RangeParam(150, 400)m
- Service: RangeParam(300, 800)m

Car direction logic:
- Car placer assigns travel_bearing based on road bearing + lane side
- Highway lanes: same direction per side (left lane offset = one bearing, right = opposite)
- Bidirectional roads: 50/50 chance per car

### Commercial Light Types

**SignLight(LightSource)**:
- sign_size: (RangeParam(1, 6), RangeParam(0.5, 3)) meters
- color_pool: warm white 35%, red 12%, blue 12%, green 10%, yellow 10%, purple 8%,
  orange 8%, pink 5%
- brightness: 0.6-0.9 (signs are bright)
- glow_radius: proportional to sign_size
- illumination_radius: 15-25m (signs cast a lot of ambient light)
- Placed along road-facing edges of commercial zone buildings

**ParkingLotLight(LightSource)**:
- height: RangeParam(8, 12)
- grid_spacing: (RangeParam(20, 30), RangeParam(20, 30))
- color_pool: 70% LED white (#E0E8FF), 30% sodium (#FFA040)
- brightness: 0.75-0.9
- glow_radius: 20m, illumination_radius: 30m
- Placed in a grid within parking areas of commercial zones

**StorefrontLight(LightSource)**:
- color_pool: 60% warm white (#FFE8CC), 25% warm yellow (#FFE080), 15% fluorescent (#F0F0FF)
- brightness: 0.5-0.7
- glow_radius: 8m, illumination_radius: 12m
- Placed along building faces in commercial zones, spacing ~5-8m

### HouseLight(LightSource)
Created by `House.get_lights()`, which is called by the house_light_placer:
- light_type: "window" | "porch" | "interior"
- Window: warm yellow/white, small, low illumination (light through glass)
- Porch: brighter, amber/white, slightly larger illumination
- Interior (TV glow): cool blue-white, dim, very small illumination
- Color pool: 45% warm yellow (#FFE4A0), 25% warm white (#FFF0D0), 15% amber (#FFCC80),
  15% TV blue (#D0E0FF)

## Part 2: Light Placement Pipeline

### Placement Engine
```
src/streets_at_night/
    placement/
        __init__.py
        streetlight_placer.py   # Place streetlights along RoadSegments
        car_placer.py           # Place CarLightPairs along RoadSegments
        commercial_placer.py    # Place commercial lights in CommercialZones
        house_light_placer.py   # Call get_lights() on Houses in ResidentialZones
        scene_builder.py        # Orchestrates all placers, populates Scene
```

**scene_builder.py** is the orchestration point:
1. Takes: a Scene (already has road_segments and zones from stage 4) + config + rng
2. Creates a child RNG per placer (deterministic from master seed)
3. Calls each placer to generate light objects
4. Each domain object (CarLightPair, HighwayLight, House) creates its own LightSource(s)
5. Collects all LightSources and adds them to the Scene via `scene.add_lights()`
6. Also stores CarLightPair references in `scene.car_pairs` for render-time facing resolution

The renderer only sees `scene.lights` — a flat `list[LightSource]`. It doesn't know about
placement logic.

### Placement Logic Details

**Streetlight placement**:
- For each road segment, check `road_segment.fixture_profile`
- If None → skip (this road has no streetlights)
- Otherwise, walk along the segment's geometry at `fixture_profile.spacing` intervals
- Offset perpendicular to road direction by half the road width (place on road edge)
- Use `StreetLight.create_from_profile(fixture_profile, position, road_type)`
- All lights on the segment inherit the same color, height, brightness from the profile
- Handle "both sides" vs "one side" vs "alternating" per `fixture_profile.side`

**Car placement**:
- Walk along each road segment at random intervals (density from config)
- Offset slightly from centerline (left or right lane)
- Assign `travel_bearing` based on road bearing + lane side (NOT camera-relative)
- Create CarLightPair

**Commercial placement**:
- For each CommercialZone:
  - Find road-facing building edges → place SignLights and StorefrontLights
  - Use `parking_areas` polygons → place ParkingLotLight grids
- Denser placement near major road intersections

**House light activation**:
- Iterate through ResidentialZones → Houses
- Call `house.get_lights()` on each House
- Houses with `num_lights == 0` produce no lights
- Collect and add to Scene

### Scene Updates
After placement, the Scene is fully populated:
```python
class Scene:
    road_segments: list[RoadSegment]
    residential_zones: list[ResidentialZone]
    commercial_zones: list[CommercialZone]
    dark_zones: list[DarkZone]
    lights: list[LightSource]           # ALL lights, flat list
    car_pairs: list[CarLightPair]       # for render-time facing resolution
    camera: Camera
    config: SceneConfig
    spatial_index: SpatialIndex         # indexes roads, zones, AND lights
```

The spatial index is updated when lights are added — enables fast
`get_lights_near(x, y, radius)` queries for ground illumination.

## Part 3: Light Rendering

### Render Pipeline Updates
Add new passes to `render_pipeline.py`:

```python
def get_preview_passes():
    return [ground_pass, road_pass, light_pass_preview]

def get_export_passes():
    return [ground_pass, road_pass, illumination_pass, light_pass_export]
```

### Car Facing Resolution
Before any render pass that draws lights, the pipeline must resolve car facing:
```python
for car_pair in scene.car_pairs:
    car_pair.resolve_facing(scene.camera.forward)
```
This runs every frame in the preview (since camera moves) and once for export.

### Glow Effect (export only)
Each LightSource renders as:
1. **Core**: bright pixel(s) at the light's color and full brightness
2. **Glow halo**: gaussian falloff from center, additive blended
   - Radius in screen space = `glow_radius / distance_to_camera * focal_length`
   - Distant lights have smaller halos but are still visible
3. **Ground illumination**: light's color tints the ground within `illumination_radius`

### Ground Illumination (export only, `rendering/illumination_pass.py`)
```
illuminated_color = base_ground_color + light_color * illumination_intensity * falloff(d)
```
- `d` = distance from light to ground point (in world meters)
- `falloff(d)` = gaussian or inverse-square, clamped at `illumination_radius`
- Multiple lights accumulate additively
- For each ground pixel, query `scene.get_lights_near(x, y, max_illumination_radius)` via
  spatial index — only accumulate nearby lights, not all lights
- Render at configurable reduced resolution (`illumination_resolution_multiplier` from
  QualityPreset) and gaussian-upscale — it's inherently soft
- This is the effect of car headlights making the road glow, streetlights creating pools

### Atmospheric Perspective
Lights further from camera:
- Dimmer: `brightness *= atmospheric_curve(distance)`
- More diffused: glow radius increases slightly (scattering)
- Color shifts toward warm haze: blend toward (#332210) at extreme distance
- Beyond ~30km: individual lights blur into general glow (handled in stage 6)

### New Rendering Files
```
src/streets_at_night/
    rendering/
        light_pass.py          # Light core + glow (preview: dots; export: core + glow layer)
        illumination_pass.py   # Ground illumination accumulation (export only)
```

### Preview Behavior
**Preview (pygame)**: lights as colored dots via `backend.draw_circle()`. No glow, no
ground illumination — too slow for real-time.
- Toggle lights on/off with "L" key
- Toggle individual types with shift+1 through shift+5

### Export Behavior
**Export (Pillow)**: full multi-pass pipeline:
1. Ground layer (from stage 4 ground_pass)
2. Road surfaces (from road_pass)
3. Illumination pass: accumulate light contributions onto ground at reduced resolution,
   gaussian blur, upscale
4. Light core + glow pass: render on a separate black layer with additive blending
5. Composite all layers

## Dependencies (new)
- `scipy` — only if Pillow's GaussianBlur is insufficient for the glow rendering.
  Try Pillow first.

## Acceptance Criteria
- [ ] FixtureProfile assigned once per road segment — all lights on a segment match
- [ ] Some residential/service segments have no lights (fixture_profile is None)
- [ ] Different streets visibly have different light colors (not a random per-light mix)
- [ ] StreetLight.create_from_profile() factory creates lights from fixture profile
- [ ] CarLightPair stores travel_bearing (absolute), not camera-relative facing
- [ ] CarLightPair.resolve_facing() correctly determines headlights vs taillights
- [ ] CarLightPair always produces exactly two lights with correct separation
- [ ] Headlight and taillight pairs are visually distinguishable
- [ ] Car density varies correctly by road type
- [ ] Commercial zones have signs, parking lot grids, and storefront lights
- [ ] Residential zones have sparse, warm house lights
- [ ] Dark zones (parks, water) have no lights
- [ ] All lights stored in Scene.lights (not in road segments or zones)
- [ ] Spatial index queries return correct nearby lights for illumination
- [ ] Glow halos visible in export render
- [ ] Ground illumination shows pools of light on road surfaces in export
- [ ] Car headlights visibly illuminate the road ahead of them
- [ ] Atmospheric perspective dims/diffuses distant lights
- [ ] All parameters configurable via scene_config.toml
- [ ] Same seed produces identical light placement
- [ ] Preview shows light positions in real-time (dots, no glow)
- [ ] Moving camera in preview correctly updates headlight/taillight assignment
- [ ] High-res export renders full glow + ground illumination
- [ ] Scene reads as "city at night from the air" — the key artistic test

## Notes
- The object model is the whole point of this stage architecture. The placement
  engine creates objects, objects create their own lights, the renderer just
  draws LightSources. The Scene is the handoff point. Keep these boundaries clean.
- Car light pairs will merge into single dots at distance — that's correct and
  realistic. Don't fight it.
- Expect heavy iteration on the config values. The default numbers in this doc
  are starting points. The real values come from looking at the output and
  adjusting until it looks right. The TOML config makes this a file edit.
- Ground illumination is the most expensive render pass. The spatial index +
  reduced resolution rendering are both critical for keeping it tractable. Profile
  early. If still too slow, reduce illumination resolution further (1/4 instead
  of 1/2) and blur-upscale.
- For the preview, cache projected light screen positions and only recalculate
  when the camera moves. Lights don't move — only the camera does.
