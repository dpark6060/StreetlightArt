"""Shared test fixtures for Streets at Night test suite."""

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_light_config
from streets_at_night.config.schema import LightConfig


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig built from DEFAULTS for use in SceneConfig fixtures."""
    return _build_light_config(DEFAULTS["lights"])
