# Object Model

This defines the configurable object hierarchy for the project. All objects pull from
parameter ranges and color pools on instantiation, making the scene highly tunable without
code changes.

See `.claude/work/architecture-plan.md` for the full module map and data flow.

## Shared Utilities

### ColorPool
Weighted random color selection, reused across all light-bearing objects.
Accepts a seeded RNG for reproducibility.
```
ColorPool
    colors: list[tuple[str, float]]   # (hex_color, weight)
    rng: Random                        # seeded RNG instance
    pick() -> tuple[int, int, int]     # Returns random RGB, weighted by probability
```

Example usage:
```python
sodium_pool = ColorPool([
    ("#FFA040", 0.70),  # sodium orange
    ("#E0E8FF", 0.20),  # LED white
    ("#FFE080", 0.10),  # soft yellow
], rng=rng)
color = sodium_pool.pick()
```

### RangeParam
Numeric parameter that picks from a range on each instantiation.
Accepts a seeded RNG for reproducibility.
```
RangeParam
    min_val: float
    max_val: float
    distribution: str = "uniform"   # "uniform", "normal", "exponential"
    rng: Random                      # seeded RNG instance
    pick() -> float
```

---

## Light Hierarchy

### LightSource (base)
The renderer interacts with this interface. Everything that emits light is a LightSource.
```
LightSource
    position: tuple[float, float]    # (x, y) in local meters
    color: tuple[int, int, int]      # RGB
    brightness: float                # 0.0 - 1.0
    size: float                      # apparent base size in meters
    glow_radius: float               # how far the glow extends
    illumination_radius: float       # how far it lights the ground
    illumination_intensity: float    # how strongly it lights the ground
```

### FixtureProfile
Streetlights are consistent **per road segment** — a given street has all the same fixture
type at the same spacing. Variation happens *between* streets, not within them.

A FixtureProfile is assigned once per RoadSegment and defines the lighting character of
that entire road.

```
FixtureProfile
    fixture_color: tuple[int, int, int]  # single color — all lights on this road match
    height: float                         # pole height
    brightness: float
    glow_radius: float
    illumination_radius: float
    illumination_intensity: float
    spacing: float                        # meters between lights
    side: str                             # "both", "one", "alternating"
```

Each road type has a **FixturePool** — a weighted list of possible FixtureProfiles that
get assigned per-segment. This is where the variety comes from.

#### Highway FixturePool
Highways are uniform — all segments get the same type (city/state standards):
- 90% high-pressure sodium: color=#FFA040, height=12-15m, spacing=40-60m, both sides
- 10% LED retrofit: color=#E0E8FF, same height/spacing

#### Major Road FixturePool
Some variety between streets:
- 55% sodium: color=#FFA040, height=8-10m, spacing=35-50m, both sides
- 35% LED white: color=#E0E8FF, same height/spacing
- 10% warm LED: color=#FFE080, same height/spacing

#### Collector Road FixturePool
- 40% sodium: color=#FFA040, height=7-9m, spacing=45-65m, alternating sides
- 40% LED white: color=#E0E8FF, same
- 20% warm yellow: color=#FFE080, same

#### Residential FixturePool
Most variety here — some streets have no lights at all:
- 25% no lights at all (common in suburban residential)
- 30% warm yellow: color=#FFE080, height=5-7m, spacing=80-120m, one side
- 25% LED white: color=#E0E8FF, same height/spacing
- 15% sodium: color=#FFA040, same height/spacing
- 5% decorative warm: color=#FFCC80, shorter poles (4-5m), spacing=60-80m, both sides

#### Service Road FixturePool
- 70% no lights
- 20% LED white: short poles, wide spacing
- 10% sodium: same

**Key point**: the fixture color is picked ONCE per road segment and applied to ALL lights
on that segment. Individual lights do NOT roll from a color pool — they inherit the
segment's fixture profile. This means one street is all sodium orange, the next is all
LED white, and a third has no lights. That's how real cities look.

### StreetLight(LightSource) — instantiated from FixtureProfile
```
StreetLight(LightSource)
    road_type: RoadType
    fixture_profile: FixtureProfile      # inherited from the parent road segment
```

All StreetLight fields (color, height, brightness, glow, illumination) come directly from
the FixtureProfile. No per-light randomization of color.

Subclasses still exist for road-type-specific behavior (placement logic, default pool):

#### HighwayLight(StreetLight)
- Tall poles, bright, consistent per-segment
- Always has lights (0% "no lights" probability)

#### MajorRoadLight(StreetLight)
- Medium poles
- Always has lights

#### CollectorLight(StreetLight)
- Between major and residential

#### ResidentialLight(StreetLight)
- Shorter poles, dimmer
- 25% chance the segment has no lights at all

---

### CarLightPair (always two lights)
A car is always a pair of lights. Never a single light.

**Important**: Cars store their absolute travel direction, NOT camera-relative facing.
Headlight vs taillight is resolved at **render time** by comparing travel_bearing to the
camera look direction. This keeps car placement valid when the camera moves in the preview.

```
CarLightPair
    position: tuple[float, float]    # center of the car on the road
    travel_bearing: float            # absolute direction of travel (radians)
    separation: float                # distance between the two lights (~1.8-2.2m)
    speed_class: str                 # "stopped", "slow", "normal", "fast"

    lights: tuple[LightSource, LightSource]  # the two individual lights, generated on init
```

On init:
- Pair separation randomized from RangeParam(1.6, 2.4)
- Two lights offset perpendicular to travel_bearing by ±(separation/2)
- Lights created with placeholder color (white) — actual color set by resolve_facing()

At render time:
- `resolve_facing(camera_forward)` — computes dot product of travel_bearing vs camera
  look direction. If car is moving toward camera → headlights; away → taillights.
  Sets appropriate colors from the headlight or taillight ColorPool.
- Headlight pool: 50% bright white, 30% warm white, 15% slight yellow, 5% blue-white (HID)
- Taillight pool: 60% red, 25% dark red, 15% red-orange
- Headlights: brightness 0.7-0.9, glow_radius 8m, forward illumination cone 15m ahead
- Taillights: brightness 0.3-0.5, glow_radius 3m, minimal illumination

---

### House
A house is a lot with a footprint and 0-N light sources.
```
House
    footprint: Polygon               # from OSM or generated via zone subdivision
    base_color: tuple[int, int, int] # the house's exterior color (dim, for ground rendering)
    num_lights: int                  # 0-5, randomly chosen with weighted distribution
    yard_color: tuple[int, int, int] # dark green-ish, slight variation per house
```

On init:
- `num_lights` drawn from weighted distribution:
  - 0 lights (dark house): 35%
  - 1 light: 25%
  - 2 lights: 20%
  - 3 lights: 12%
  - 4 lights: 6%
  - 5 lights: 2%
- `base_color` drawn from house exterior pool:
  - Dark beige, dark tan, dark brown, dark grey, dark terracotta (all very muted at night)
- Light positions computed relative to footprint edges/corners
- `yard_color` is a slight random variation of dark green

Key method:
- `get_lights()` — returns list of HouseLight instances (called by house_light_placer)

#### HouseLight(LightSource)
```
HouseLight(LightSource)
    light_type: str                  # "window", "porch", "interior"
    color_pool: ColorPool
```
- Window: warm yellow, warm white
- Porch: warm white, amber (brighter than window, smaller)
- Interior (TV glow): cool blue-white, slightly flickery appearance
- Color pool: 45% warm yellow, 25% warm white, 15% amber, 15% TV blue

---

### CommercialLight (base for business-area lights)

#### SignLight(CommercialLight, LightSource)
```
SignLight
    sign_size: tuple[float, float]   # width, height in meters
    color_pool: ColorPool            # neon variety
    flicker: bool                    # some signs flicker
```
- Bright, varied colors
- Color pool: warm white 35%, red 12%, blue 12%, green 10%, yellow 10%, purple 8%, orange 8%, pink 5%
- Size: RangeParam(1.0, 6.0) width, proportional height
- Positioned along road-facing building edges in commercial zones
- Large illumination radius (signs cast a lot of ambient light)

#### ParkingLotLight(CommercialLight, LightSource)
```
ParkingLotLight
    pole_height: float
    grid_spacing: tuple[float, float]  # (x_spacing, y_spacing)
```
- Evenly spaced in a grid (not random)
- Color: almost always LED white or sodium
- Bright, large illumination radius (parking lots are well-lit)
- Grid spacing: 20-30m typically

#### StorefrontLight(CommercialLight, LightSource)
```
StorefrontLight
    color_pool: ColorPool
```
- Warm white or yellow, positioned along building face
- Moderate brightness, smaller than sign lights
- Color pool: 60% warm white, 25% warm yellow, 15% fluorescent white

---

## Zone Objects (data containers, do NOT own lights)

Zones hold geometry and child objects (houses, building footprints) but do NOT store
references to placed lights. Lights are placed by the placement engine and stored in
the Scene's flat light collection.

### RoadSegment
Geometry, metadata, and assigned fixture profile. No light *instances* — those are
created by the placement engine and stored in the Scene.
```
RoadSegment
    geometry: LineString              # road centerline in local meters
    road_type: RoadType
    width: float                     # road width in meters (derived from type)
    bearing: float                   # average direction
    fixture_profile: FixtureProfile | None  # assigned on construction, None = no streetlights
```

The fixture profile is assigned when the RoadSegment is created (during data processing).
A `FixturePool` for the segment's road_type is rolled once — the result applies to the
entire segment. If `fixture_profile is None`, the placement engine skips streetlight
placement for this segment.

### ResidentialZone
```
ResidentialZone
    boundary: Polygon
    houses: list[House]              # generated from building footprints or subdivision
    lot_coverage: float              # what fraction of the zone is built (vs yards/streets)
```

### CommercialZone
```
CommercialZone
    boundary: Polygon
    building_footprints: list[Polygon]  # building shapes within the zone
    parking_areas: list[Polygon]        # open areas for parking lot light grids
```

### DarkZone (parks, water, forest)
```
DarkZone
    boundary: Polygon
    zone_type: str                   # "park", "water", "forest"
    base_color: tuple[int, int, int] # very dark green, dark blue, etc.
    # No lights (or extremely sparse path lights for parks)
```

---

## Scene Facade

The `Scene` class is the single object that owns all world state. Everything downstream
(placement, rendering) receives a Scene — not a bag of lists.

```
Scene
    road_segments: list[RoadSegment]
    residential_zones: list[ResidentialZone]
    commercial_zones: list[CommercialZone]
    dark_zones: list[DarkZone]
    lights: list[LightSource]                # flat list, populated by placement engine
    car_pairs: list[CarLightPair]            # for render-time facing resolution
    camera: Camera
    config: SceneConfig
    spatial_index: SpatialIndex

    get_lights_in_frustum(camera) -> list[LightSource]
    get_roads_in_frustum(camera) -> list[RoadSegment]
    get_zones_in_frustum(camera) -> list of zone objects
    get_lights_near(x, y, radius) -> list[LightSource]   # for illumination
    add_lights(lights) -> None                             # also indexes spatially
```

---

## Spatial Index

Uniform grid (500m × 500m cells) for fast spatial queries. Used for:
- Frustum culling (which roads/lights are visible?)
- Ground illumination (which lights are near this point?)
- Preview LOD (what to skip at distance?)

```
SpatialIndex
    __init__(bounds, cell_size=500)
    insert(item, position_or_bbox)
    query_rect(min_x, min_y, max_x, max_y) -> list
    query_radius(x, y, radius) -> list
```

Generic — works for roads, lights, and zones.

---

## Configuration Structure

All parameter ranges, color pools, and probabilities live in a TOML config file
(`scene_config.toml`) at the project root. Defaults are baked into `config/defaults.py`.
The user's TOML only needs to override what they want to change.

```python
@dataclass
class SceneConfig:
    # Camera
    camera: CameraConfig

    # Quality
    quality: QualityPreset

    # Streetlight configs per road type
    streetlight_configs: dict[RoadType, StreetLightConfig]

    # Car configs per road type
    car_configs: dict[RoadType, CarConfig]

    # House config
    house_config: HouseConfig

    # Commercial config
    commercial_config: CommercialConfig

    # Atmosphere
    atmosphere: AtmosphereConfig

    # Color pools (can be overridden per object type)
    color_pools: dict[str, ColorPool]

    # Global params
    seed: int
    max_render_distance: float
    ambient_ground_brightness: float
    bounding_box: tuple[float, float, float, float]  # (north, south, east, west)
```

Loaded via `config/loader.py`: reads TOML, overlays onto defaults, returns immutable
SceneConfig.

---

## Design Principles
1. **The renderer is dumb**: it receives a `Scene` with a flat list of `LightSource` objects
   and renders them. It doesn't know about cars vs houses. All the intelligence is in the
   object constructors and placement logic.
2. **Zones don't own lights**: zones hold geometry. The placement engine reads zone geometry,
   creates light objects, and stores them in the Scene. This decouples placement from the
   data model.
3. **Car facing is resolved at render time**: CarLightPair stores absolute travel direction.
   The render pipeline calls `resolve_facing(camera_forward)` before drawing. This keeps
   the preview correct when the camera moves.
4. **Randomness is seeded**: every random choice (color picks, light counts, car positions)
   uses a seeded RNG so renders are reproducible. Each placer gets its own RNG seeded
   deterministically from the master seed.
5. **Color pools are the primary tuning knob**: the fastest way to change the scene's feel
   is adjusting color weights in scene_config.toml.
6. **Objects own their lights**: a House generates its HouseLights. A CarLightPair generates
   its two LightSources. The placement engine creates objects; objects create their own lights.
   But the *containers* (zones, road segments) do not hold references to those lights.
