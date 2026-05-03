"""Shared test fixtures for Streets at Night test suite."""

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import (
    _build_atmosphere_config,
    _build_light_config,
    _build_stars_config,
)
from streets_at_night.config.schema import AtmosphereConfig, LightConfig, StarsConfig


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return _build_light_config(DEFAULTS["lights"])


@pytest.fixture
def atmosphere_config() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return _build_atmosphere_config(DEFAULTS["atmosphere"])


@pytest.fixture
def stars_config() -> StarsConfig:
    """Return a default StarsConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return _build_stars_config(DEFAULTS["stars"])
