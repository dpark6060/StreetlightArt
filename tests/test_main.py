# Radical
"""Tests for streets_at_night.main CLI subcommand dispatch."""

from pathlib import Path
from unittest import mock

import pytest

from streets_at_night import main as main_module


@pytest.fixture
def patched_dispatch():
    """Patch config loading, cache resolution, and run_ui for dispatch tests.

    Yields the run_ui mock so tests can assert how the UI was launched without
    ever opening a pygame window.
    """
    with mock.patch.object(main_module, "load_config") as mock_load, mock.patch.object(
        main_module, "resolve_cache_dir", return_value=Path("cache")
    ), mock.patch("streets_at_night.ui.app.run_ui") as mock_run_ui:
        mock_load.return_value = mock.MagicMock()
        yield mock_run_ui


def test_main_ui_default_tab_dispatches_to_run_ui_on_verify(patched_dispatch):
    # Arrange
    argv = ["streets-at-night", "ui"]

    # Act
    with mock.patch("sys.argv", argv):
        main_module.main()

    # Assert
    patched_dispatch.assert_called_once()
    assert patched_dispatch.call_args.kwargs["initial_tab"] == "Verify"


def test_main_ui_explicit_tab_dispatches_to_run_ui_on_that_tab(patched_dispatch):
    # Arrange
    argv = ["streets-at-night", "ui", "--tab", "Render"]

    # Act
    with mock.patch("sys.argv", argv):
        main_module.main()

    # Assert
    assert patched_dispatch.call_args.kwargs["initial_tab"] == "Render"


def test_main_devmode_dispatches_to_run_ui_on_devmode_tab(patched_dispatch):
    # Arrange
    argv = ["streets-at-night", "devmode"]

    # Act
    with mock.patch("sys.argv", argv):
        main_module.main()

    # Assert
    patched_dispatch.assert_called_once()
    assert patched_dispatch.call_args.kwargs["initial_tab"] == "Devmode"


def test_main_ui_passes_none_config_path_when_file_missing(patched_dispatch):
    # Arrange — a config path that does not exist maps to None (blank UI state).
    argv = ["streets-at-night", "--config", "does_not_exist.toml", "ui"]

    # Act
    with mock.patch("sys.argv", argv):
        main_module.main()

    # Assert — second positional arg is the config path (None when missing).
    assert patched_dispatch.call_args.args[1] is None
