# Stage 1: Project Setup & OSM Data Pipeline

## Goal
Set up the project, pull OpenStreetMap data for the San Bernardino area, classify all road types,
extract land use zones, and produce a top-down verification plot proving the data is correct.

## Architecture Reference
See `.claude/work/architecture-plan.md` for the full module map and design decisions.

## Camera Context
The user's camera setup defines what area we need to pull:
- Camera: 34.2349°N, 117.3116°W at 5186ft (~1581m)
- Target: 34.1083°N, 117.2968°W at 1000ft (~305m)
- The camera is ~14km north of the target, looking south at ~5° below horizontal
- At this shallow angle, visible terrain extends 30-50km+ from the camera
- Pull a generous bounding box: ~40km x 40km centered between camera and target

## Technical Approach

### Project Structure
```
StreetsAtNight/
├── pyproject.toml
├── scene_config.toml              # User-editable scene parameters (TOML)
├── src/
│   └── streets_at_night/
│       ├── __init__.py
│       ├── main.py                # CLI entry point (verify / preview / render subcommands)
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py          # SceneConfig, CameraConfig, QualityPreset dataclasses
│       │   ├── defaults.py        # Default values as dict/dataclass
│       │   └── loader.py          # TOML → SceneConfig, with overlay logic
│       ├── data/
│       │   ├── __init__.py
│       │   ├── osm_loader.py      # Download and cache OSM data
│       │   ├── road_classifier.py # Classify roads into render categories
│       │   └── zone_builder.py    # OSM land use → zone objects (built in stage 4)
│       ├── models/
│       │   ├── __init__.py
│       │   └── road_types.py      # RoadType enum
│       ├── camera/                 # (built in stage 2)
│       ├── scene/                  # (built in stage 3-4)
│       ├── placement/             # (built in stage 5)
│       └── rendering/             # (built in stage 3+)
├── cache/                          # Cached OSM downloads (gitignored)
├── output/                         # Rendered images (gitignored)
└── tests/
```

### Dependencies
- `osmnx` — OSM data download, road networks, building footprints, land use
- `networkx` — comes with osmnx, graph operations
- `shapely` — geometry operations (comes with osmnx)
- `geopandas` — spatial dataframes (comes with osmnx)
- `matplotlib` — verification plots only (not used for final rendering)
- `numpy` — math
- Use `uv` as the package manager

### OSM Data to Pull
Using `osmnx`, we need three categories of data:

1. **Road network** (`osmnx.graph_from_bbox` or `graph_from_polygon`)
   - Returns a NetworkX MultiDiGraph with edges containing `highway` tags
   - OSM `highway` values we care about:
     - `motorway`, `motorway_link` → Highway
     - `trunk`, `trunk_link` → Highway
     - `primary`, `primary_link` → Major Road
     - `secondary`, `secondary_link` → Major Road
     - `tertiary`, `tertiary_link` → Collector Road
     - `residential` → Residential Street
     - `service` → Service Road (parking lots, driveways)
     - `unclassified` → Minor Road
     - `living_street` → Residential Street
   - Ignore: `footway`, `cycleway`, `path`, `pedestrian`, `track`, `steps`

2. **Land use zones** (`osmnx.features_from_bbox` with tags)
   - `landuse=residential` → Residential zone
   - `landuse=commercial`, `landuse=retail` → Commercial zone
   - `landuse=industrial` → Industrial zone
   - `leisure=park`, `landuse=grass`, `landuse=recreation_ground` → Park
   - `natural=water`, `waterway=riverbank` → Water
   - `landuse=forest`, `natural=wood` → Forest/nature

3. **Building footprints** (`osmnx.features_from_bbox` with `building=True`)
   - Used later to place house lights and define building zones
   - Tag with zone type based on which land use polygon they fall within

### Data Caching
- Save downloaded data to `cache/` as GeoPackage files (`.gpkg`)
- On subsequent runs, load from cache instead of re-downloading
- Cache key: bounding box coordinates + date

### Road Classification
Create a `RoadType` enum in `models/road_types.py` and a classifier in
`data/road_classifier.py` that maps OSM `highway` tags to our categories:
```python
class RoadType(Enum):
    HIGHWAY = "highway"
    MAJOR_ROAD = "major_road"
    COLLECTOR = "collector"
    RESIDENTIAL = "residential"
    SERVICE = "service"
```

Each road type will later drive:
- Line thickness in render
- Light type and density (via StreetLight subclass factory)
- Car density and speed
- Color in verification plot

### TOML Configuration
Set up the initial `scene_config.toml` with camera position, bounding box, and quality
presets. The config module (`config/schema.py`, `config/defaults.py`, `config/loader.py`)
should be built in this stage so that all subsequent stages can load config consistently.

```toml
[camera]
lat = 34.234880959491356
lon = -117.31157772457728
altitude_ft = 5186
target_lat = 34.10834722093596
target_lon = -117.29681238769862
target_altitude_ft = 1000
fov = 60

[bounds]
north = 34.35
south = 33.95
east = -117.05
west = -117.55

[quality.low]
width = 1920
height = 1080

[quality.medium]
width = 3840
height = 2160

[quality.high]
width = 7680
height = 4320

seed = 42
```

### CLI Entry Point
Set up `main.py` with argparse subcommands. In this stage only the `verify` subcommand
needs to work:
```bash
uv run python -m streets_at_night verify --output output/stage1_verification.png
```

### Verification Plot
The `verify` subcommand produces a matplotlib figure showing:
- All roads colored by RoadType (distinct colors per type)
- Land use zones as semi-transparent fills
- Building footprints as small grey shapes
- Camera position and target marked with dots
- Legend showing road types and land use categories
- Save to specified output path

## Acceptance Criteria
- [ ] `uv sync` installs all dependencies cleanly
- [ ] `scene_config.toml` exists with camera, bounds, and quality presets
- [ ] Config loads correctly via `config/loader.py`
- [ ] Running `verify` subcommand downloads OSM data (or loads from cache)
- [ ] Verification plot shows roads around San Bernardino colored by type
- [ ] At least 4 distinct road types are visible in the plot
- [ ] Land use zones (residential, commercial, parks, water) are visible
- [ ] Camera and target positions are marked on the plot
- [ ] Data caching works (second run doesn't re-download)

## Notes
- osmnx may take a while to download a 40x40km area — that's fine
- The bounding box should be generous; we'll cull to the camera frustum later
- Don't over-engineer the classifier — we can refine categories as we see what
  the data actually contains in this area
