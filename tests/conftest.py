"""Shared test fixtures for Streets at Night test suite."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402
import pytest  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import AtmosphereConfig, LightConfig, StarsConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402


@pytest.fixture
def headless_pygame():
    """Initialise pygame under the dummy video driver, torn down after the test."""
    pygame.init()
    pygame.display.set_mode((1440, 900))
    yield
    pygame.quit()


@pytest.fixture
def headless_manager(headless_pygame):
    """Provide a headless pygame display + pygame_gui UIManager, torn down after the test."""
    return pygame_gui.UIManager((1440, 900))


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
