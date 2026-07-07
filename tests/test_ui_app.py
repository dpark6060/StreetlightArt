# Radical
"""Headless construction/smoke tests for the UI shell (app + tabs + widgets)."""

import os
import threading
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app import UIApp, run_ui  # noqa: E402
from streets_at_night.ui.jobs import JobEvent  # noqa: E402
from streets_at_night.ui.tabs.preview_tab import PreviewTab  # noqa: E402
from streets_at_night.ui.tabs.verify_tab import VerifyTab  # noqa: E402


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


def test_refresh_tab_bar_busy_disables_all_buttons(config: SceneConfig):
    # Arrange — a job that blocks on the worker thread so jobs.busy stays True.
    app = UIApp(config, config_path=None, cache_dir=Path("cache"), initial_tab="Verify")
    release = threading.Event()
    app.jobs.submit(lambda progress: release.wait(5.0))

    # Act
    app._refresh_tab_bar()

    # Assert — no tab switch can start mid-job: every button is disabled.
    try:
        assert all(button.is_enabled is False for button in app._tab_buttons)
    finally:
        release.set()
        pygame.quit()


def test_refresh_tab_bar_idle_disables_only_active_button(config: SceneConfig):
    # Arrange
    app = UIApp(config, config_path=None, cache_dir=Path("cache"), initial_tab="Verify")

    # Act
    app._refresh_tab_bar()

    # Assert — active tab's button is the disabled indicator; the rest are enabled.
    for button, tab in app._tab_buttons.items():
        if tab is app._active_tab:
            assert button.is_enabled is False
        else:
            assert button.is_enabled is True

    pygame.quit()


def test_preview_tab_ignores_foreign_done_event(config: SceneConfig):
    # Arrange — an unbuilt Preview tab that never submitted a job of its own.
    app = UIApp(config, config_path=None, cache_dir=Path("cache"), initial_tab="Verify")
    preview = next(tab for tab in app._tabs if isinstance(tab, PreviewTab))
    assert preview._job_pending is False
    foreign = JobEvent("done", result=object())  # not a (scene, flat_map, meta) tuple

    # Act / Assert — a misrouted done result is dropped, not unpacked into a crash.
    preview.handle_job_event(foreign)
    assert preview._loaded is False

    pygame.quit()


def test_run_ui_is_callable():
    # Assert — public entry point exists and is callable.
    assert callable(run_ui)
