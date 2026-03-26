# Radical
"""Tests for devmode panel_sections helper functions."""

import pytest

from streets_at_night.config.schema import FixturePoolEntry, FixtureProfileConfig
from streets_at_night.devmode.panel_sections import (
    _find_first_profile_index,
    _get_all_profile_indices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile() -> FixtureProfileConfig:
    """Return a minimal FixtureProfileConfig for test use."""
    return FixtureProfileConfig(
        color="#FFAA00",
        height_min=5.0,
        height_max=8.0,
        brightness_min=0.5,
        brightness_max=0.8,
        glow_radius=12.0,
        illumination_radius=30.0,
        illumination_intensity=0.4,
        spacing_min=30.0,
        spacing_max=60.0,
        side="both",
    )


def _no_lights_entry() -> FixturePoolEntry:
    """Return a pool entry with no profile (no-lights)."""
    return FixturePoolEntry(weight=0.3, profile=None)


def _lit_entry() -> FixturePoolEntry:
    """Return a pool entry with a real profile."""
    return FixturePoolEntry(weight=0.7, profile=_make_profile())


# ---------------------------------------------------------------------------
# _find_first_profile_index
# ---------------------------------------------------------------------------


def test_find_first_profile_index_returns_zero_when_first_has_profile() -> None:
    """Returns 0 when the first entry has a profile."""

    entries = [_lit_entry(), _lit_entry()]

    result = _find_first_profile_index(entries)

    assert result == 0


def test_find_first_profile_index_skips_no_lights_entries() -> None:
    """Returns 1 when the first entry is no-lights and second has a profile."""

    entries = [_no_lights_entry(), _lit_entry()]

    result = _find_first_profile_index(entries)

    assert result == 1


def test_find_first_profile_index_returns_none_when_all_empty() -> None:
    """Returns None when every entry is a no-lights entry."""

    entries = [_no_lights_entry(), _no_lights_entry()]

    result = _find_first_profile_index(entries)

    assert result is None


def test_find_first_profile_index_returns_none_for_empty_list() -> None:
    """Returns None for an empty entry list."""

    result = _find_first_profile_index([])

    assert result is None


def test_find_first_profile_index_returns_correct_index_with_leading_nones() -> None:
    """Returns index 2 when first two entries are no-lights."""

    entries = [_no_lights_entry(), _no_lights_entry(), _lit_entry()]

    result = _find_first_profile_index(entries)

    assert result == 2


# ---------------------------------------------------------------------------
# _get_all_profile_indices
# ---------------------------------------------------------------------------


def test_get_all_profile_indices_returns_all_lit_indices() -> None:
    """Returns indices of every entry that has a profile."""

    entries = [_lit_entry(), _no_lights_entry(), _lit_entry(), _lit_entry()]

    result = _get_all_profile_indices(entries)

    assert result == [0, 2, 3]


def test_get_all_profile_indices_returns_empty_when_all_no_lights() -> None:
    """Returns an empty list when no entries have profiles."""

    entries = [_no_lights_entry(), _no_lights_entry()]

    result = _get_all_profile_indices(entries)

    assert result == []


def test_get_all_profile_indices_returns_empty_for_empty_list() -> None:
    """Returns an empty list for an empty entry list."""

    result = _get_all_profile_indices([])

    assert result == []


def test_get_all_profile_indices_with_single_lit_entry() -> None:
    """Returns [0] when there is a single entry with a profile."""

    entries = [_lit_entry()]

    result = _get_all_profile_indices(entries)

    assert result == [0]


def test_get_all_profile_indices_skips_none_profile_entries() -> None:
    """First and last entries are no-lights; only middle index is returned."""

    entries = [_no_lights_entry(), _lit_entry(), _no_lights_entry()]

    result = _get_all_profile_indices(entries)

    assert result == [1]
