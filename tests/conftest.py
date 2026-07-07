"""Shared test fixtures for Streets at Night test suite."""

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import AtmosphereConfig, LightConfig, StarsConfig
from streets_at_night.config.serde import dataclass_from_dict


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return dataclass_from_dict(LightConfig, DEFAULTS["lights"])


@pytest.fixture
def atmosphere_config() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return dataclass_from_dict(AtmosphereConfig, DEFAULTS["atmosphere"])


@pytest.fixture
def stars_config() -> StarsConfig:
    """Return a default StarsConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return dataclass_from_dict(StarsConfig, DEFAULTS["stars"])
