# Radical
"""Headless construction/smoke tests for the UI shell (app + tabs + widgets)."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app import UIApp, run_ui  # noqa: E402
from streets_at_night.ui.tabs.verify_tab import VerifyTab  # noqa: E402
from streets_at_night.ui.widgets.file_picker import FilePickerRow  # noqa: E402


@pytest.fixture
def headless_display():
    """Provide a headless pygame display for widget construction, then tear down."""
    pygame.init()
    pygame.display.set_mode((200, 200))
    yield
    pygame.quit()


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def test_uiapp_constructs_under_dummy_driver(config: SceneConfig):
    # Act
    app = UIApp(config, config_path=None, cache_dir=Path("cache"), initial_tab="Verify")

    # Assert — five tab buttons wired, Verify active, its button disabled.
    assert len(app._tab_buttons) == 5
    assert isinstance(app._active_tab, VerifyTab)
    active_button = next(b for b, t in app._tab_buttons.items() if t is app._active_tab)
    assert active_button.is_enabled is False

    pygame.quit()


def test_uiapp_unknown_initial_tab_falls_back_to_first(config: SceneConfig):
    # Act
    app = UIApp(config, config_path=None, cache_dir=Path("cache"), initial_tab="Nope")

    # Assert
    assert app._active_tab is app._tabs[0]

    pygame.quit()


def test_run_ui_is_callable():
    # Assert — public entry point exists and is callable.
    assert callable(run_ui)


def test_file_picker_row_builds_and_kills(headless_display):
    # Arrange
    manager = pygame_gui.UIManager((200, 200))

    # Act
    row = FilePickerRow(
        manager, None, pygame.Rect(0, 0, 180, 30), initial_text="a.toml"
    )

    # Assert
    assert row.path == Path("a.toml")
    row.set_path(Path("b.toml"))
    assert row.path == Path("b.toml")
    row.kill()


def test_file_picker_row_empty_text_yields_none_path(headless_display):
    # Arrange
    manager = pygame_gui.UIManager((200, 200))

    # Act
    row = FilePickerRow(manager, None, pygame.Rect(0, 0, 180, 30), initial_text="")

    # Assert
    assert row.path is None
    row.kill()
