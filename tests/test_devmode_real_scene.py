# Radical
"""Unit tests for the devmode real_scene module."""

from dataclasses import replace

import pytest

from streets_at_night.config.schema import (
    BoundsConfig,
    CameraConfig,
    ColorPoolConfig,
    HouseConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
)
from streets_at_night.devmode.real_scene import _build_strip_cache_filename


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zone_config() -> ZoneConfig:
    """Return a minimal ZoneConfig for testing."""
    return ZoneConfig(
        house=HouseConfig(
            exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
            yard_base_color="#060A04",
            yard_variation=2,
            min_lights=0,
            max_lights=5,
            light_weights=[5.0, 3.0, 1.5, 0.5, 0.3, 0.1],
            footprint_coverage=0.65,
            lot_width=15.0,
            lot_depth=25.0,
            setback=10.0,
        ),
        colors=ZoneColorsConfig(
            commercial="#0A0A0C",
            industrial="#080808",
            park_forest="#040804",
            water="#040408",
            road_surface="#0C0C0C",
            default_ground="#080706",
        ),
        tile_size_meters=256.0,
        tile_resolution_px=64,
    )


@pytest.fixture
def scene_config(zone_config, light_config, atmosphere_config) -> SceneConfig:
    """Return a minimal SceneConfig for testing."""
    style = RoadStyleConfig(
        highway=RoadStyleEntry(color="#141210", width=4),
        major_road=RoadStyleEntry(color="#121010", width=3),
        collector=RoadStyleEntry(color="#100E0E", width=2),
        residential=RoadStyleEntry(color="#0E0C0C", width=1),
        service=RoadStyleEntry(color="#0C0A0A", width=1),
        minor=RoadStyleEntry(color="#0A0908", width=1),
    )
    return SceneConfig(
        camera=CameraConfig(
            lat=34.2349,
            lon=-117.3116,
            altitude_ft=5186.0,
            target_lat=34.1,
            target_lon=-117.3,
            target_altitude_ft=0.0,
            fov=60.0,
        ),
        bounds=BoundsConfig(north=34.3, south=34.0, east=-117.1, west=-117.5),
        quality_presets={"low": QualityPreset(width=640, height=360)},
        seed=42,
        zones=zone_config,
        lights=light_config,
        roads=RoadsConfig(export=style, preview=style),
        atmosphere=atmosphere_config,
    )


# ---------------------------------------------------------------------------
# _build_strip_cache_filename
# ---------------------------------------------------------------------------


def test_build_strip_cache_filename_differs_for_different_target_lon(scene_config):
    # Arrange — same bounds and strip width, different camera target longitude
    other_camera = replace(scene_config.camera, target_lon=scene_config.camera.target_lon + 0.05)
    other_config = replace(scene_config, camera=other_camera)

    # Act
    filename1 = _build_strip_cache_filename(scene_config, strip_half_width_deg=0.025)
    filename2 = _build_strip_cache_filename(other_config, strip_half_width_deg=0.025)

    # Assert
    assert filename1 != filename2


def test_build_strip_cache_filename_differs_for_different_bounds(scene_config):
    # Arrange — same target_lon and strip width, different clip bounds
    other_bounds = replace(scene_config.bounds, north=scene_config.bounds.north + 0.1)
    other_config = replace(scene_config, bounds=other_bounds)

    # Act
    filename1 = _build_strip_cache_filename(scene_config, strip_half_width_deg=0.025)
    filename2 = _build_strip_cache_filename(other_config, strip_half_width_deg=0.025)

    # Assert
    assert filename1 != filename2


def test_build_strip_cache_filename_same_config_returns_same_filename(scene_config):
    # Act
    filename1 = _build_strip_cache_filename(scene_config, strip_half_width_deg=0.025)
    filename2 = _build_strip_cache_filename(scene_config, strip_half_width_deg=0.025)

    # Assert
    assert filename1 == filename2


def test_build_strip_cache_filename_ends_with_gpkg_extension(scene_config):
    # Act
    filename = _build_strip_cache_filename(scene_config, strip_half_width_deg=0.025)

    # Assert
    assert filename.endswith(".gpkg")
