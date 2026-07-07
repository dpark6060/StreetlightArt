# Radical
"""Tests for devmode config_bridge, focusing on extra_config_paths propagation."""

from unittest import mock

import pytest

from streets_at_night.config.schema import ColorPoolConfig
from streets_at_night.devmode.config_bridge import (
    _apply_color_pools,
    _apply_sliders,
    _format_value,
    _hex_to_tuple,
    _read_color_pools,
    _tuple_to_hex,
    apply_widgets_to_config,
    get_nested,
    read_config_to_widgets,
    set_nested,
)
from streets_at_night.devmode.slider_registry import SliderDescriptor, SliderRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_obj() -> object:
    """Return a simple nested object for path traversal tests."""

    class Inner:
        value = 10.0

    class Outer:
        inner = Inner()
        items = [5.0, 15.0, 25.0]

    return Outer()


# ---------------------------------------------------------------------------
# get_nested / set_nested
# ---------------------------------------------------------------------------


def test_get_nested_traverses_attribute_path(simple_obj: object) -> None:
    """get_nested returns the value at a multi-level attribute path."""

    result = get_nested(simple_obj, ("inner", "value"))

    assert result == 10.0


def test_get_nested_traverses_list_index(simple_obj: object) -> None:
    """get_nested returns the correct element at an integer list index."""

    result = get_nested(simple_obj, ("items", 1))

    assert result == 15.0


def test_set_nested_writes_to_attribute_path(simple_obj: object) -> None:
    """set_nested updates the value at a multi-level attribute path."""

    set_nested(simple_obj, ("inner", "value"), 99.0)

    assert simple_obj.inner.value == 99.0


def test_set_nested_writes_to_list_index(simple_obj: object) -> None:
    """set_nested updates the element at an integer list index."""

    set_nested(simple_obj, ("items", 2), 999.0)

    assert simple_obj.items[2] == 999.0


# ---------------------------------------------------------------------------
# _apply_sliders — extra_config_paths propagation
# ---------------------------------------------------------------------------


def test_apply_sliders_writes_to_primary_path() -> None:
    """_apply_sliders writes slider value to descriptor.config_path."""

    class Root:
        pass

    root = Root()
    root.field = 0.0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 42.5
    descriptor = SliderDescriptor(
        label="test",
        config_path=("field",),
        min_val=0.0,
        max_val=100.0,
    )
    registry.register_slider(mock_slider, descriptor)
    registry.mark_dirty(descriptor.label)

    _apply_sliders(registry, root)

    assert root.field == 42.5


def test_apply_sliders_writes_to_all_extra_paths() -> None:
    """_apply_sliders also writes to every path in extra_config_paths."""

    class Root:
        pass

    root = Root()
    root.primary = 0.0
    root.extra_a = 0.0
    root.extra_b = 0.0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 7.5
    descriptor = SliderDescriptor(
        label="test",
        config_path=("primary",),
        min_val=0.0,
        max_val=100.0,
        extra_config_paths=[("extra_a",), ("extra_b",)],
    )
    registry.register_slider(mock_slider, descriptor)
    registry.mark_dirty(descriptor.label)

    _apply_sliders(registry, root)

    assert root.primary == 7.5
    assert root.extra_a == 7.5
    assert root.extra_b == 7.5


def test_apply_sliders_does_not_write_extra_when_list_is_empty() -> None:
    """_apply_sliders does not fail and only writes primary when extra_config_paths is empty."""

    class Root:
        pass

    root = Root()
    root.only = 0.0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 3.0
    descriptor = SliderDescriptor(
        label="test",
        config_path=("only",),
        min_val=0.0,
        max_val=10.0,
        extra_config_paths=[],
    )
    registry.register_slider(mock_slider, descriptor)
    registry.mark_dirty(descriptor.label)

    _apply_sliders(registry, root)

    assert root.only == 3.0


def test_apply_sliders_casts_int_descriptor_to_int() -> None:
    """_apply_sliders rounds and casts the value to int for is_int descriptors."""

    class Root:
        pass

    root = Root()
    root.count = 0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 4.6
    descriptor = SliderDescriptor(
        label="count",
        config_path=("count",),
        min_val=0.0,
        max_val=10.0,
        is_int=True,
    )
    registry.register_slider(mock_slider, descriptor)
    registry.mark_dirty(descriptor.label)

    _apply_sliders(registry, root)

    assert root.count == 5
    assert isinstance(root.count, int)


def test_apply_sliders_updates_value_entry_text() -> None:
    """_apply_sliders calls set_text on the value entry with the formatted value."""

    class Root:
        pass

    root = Root()
    root.val = 0.0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 12.3456
    mock_entry = mock.MagicMock()
    descriptor = SliderDescriptor(
        label="val",
        config_path=("val",),
        min_val=0.0,
        max_val=100.0,
    )
    registry.register_slider(mock_slider, descriptor, value_entry=mock_entry)
    registry.mark_dirty(descriptor.label)

    _apply_sliders(registry, root)

    mock_entry.set_text.assert_called_once_with("12.346")


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


def test_apply_sliders_skips_slider_not_marked_dirty() -> None:
    """_apply_sliders leaves config untouched for a slider the user never moved.

    Regression test for the streetlight pool-mix flattening bug: a slider's
    displayed value is read from only the first profiled pool entry, but its
    extra_config_paths span every entry. Without the dirty gate, visiting a
    section and clicking Re-render (without touching anything) would
    broadcast that first entry's value over every other entry, destroying an
    authored fixture mix (e.g. sodium + LED).
    """

    class Root:
        pass

    root = Root()
    root.primary = 111.0
    root.extra_a = 222.0

    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 111.0
    descriptor = SliderDescriptor(
        label="untouched",
        config_path=("primary",),
        min_val=0.0,
        max_val=500.0,
        extra_config_paths=[("extra_a",)],
    )
    registry.register_slider(mock_slider, descriptor)
    # Note: registry.dirty is never marked for this slider.

    _apply_sliders(registry, root)

    assert root.primary == 111.0
    assert root.extra_a == 222.0


def test_format_value_formats_float_to_three_decimals() -> None:
    """_format_value returns three-decimal representation for non-int values."""

    result = _format_value(3.14159, is_int=False)

    assert result == "3.142"


def test_format_value_formats_int_as_integer_string() -> None:
    """_format_value returns integer string for is_int=True values."""

    result = _format_value(7.9, is_int=True)

    assert result == "8"


# ---------------------------------------------------------------------------
# _tuple_to_hex / _hex_to_tuple
# ---------------------------------------------------------------------------


def test_tuple_to_hex_converts_rgb_tuple() -> None:
    """_tuple_to_hex returns '#RRGGBB' uppercase hex string."""

    result = _tuple_to_hex((58, 32, 16))

    assert result == "#3A2010"


def test_hex_to_tuple_parses_hex_string() -> None:
    """_hex_to_tuple converts '#RRGGBB' to (R, G, B) integer tuple."""

    result = _hex_to_tuple("#3A2010")

    assert result == (58, 32, 16)


def test_hex_to_tuple_returns_black_on_invalid_input() -> None:
    """_hex_to_tuple falls back to (0, 0, 0) for invalid input."""

    result = _hex_to_tuple("not-a-color")

    assert result == (0, 0, 0)


# ---------------------------------------------------------------------------
# _read_color_pools / _apply_color_pools
# ---------------------------------------------------------------------------


class _Root:
    """A root holding one ColorPoolConfig at attribute `pool`."""

    def __init__(self, colors: list) -> None:
        self.pool = ColorPoolConfig(colors=colors)


def _entry(text: str = "") -> mock.MagicMock:
    """Return a mock widget entry whose get_text returns `text`."""
    entry = mock.MagicMock()
    entry.get_text.return_value = text
    return entry


def _register_pool(registry: SliderRegistry, rows: list) -> None:
    """Register a colour-pool binding at the ('pool', 'colors') path."""
    registry.register_color_pool(("pool", "colors"), rows)


def test_read_color_pools_fills_entries_from_config() -> None:
    """_read_color_pools pushes each colour's hex/weight into its row entries."""

    root = _Root([("#FF0000", 2.0), ("#00FF00", 0.5)])
    registry = SliderRegistry()
    rows = [(_entry(), _entry()), (_entry(), _entry())]
    _register_pool(registry, rows)

    _read_color_pools(root, registry)

    rows[0][0].set_text.assert_called_once_with("#FF0000")
    rows[0][1].set_text.assert_called_once_with("2.000")
    rows[1][0].set_text.assert_called_once_with("#00FF00")
    rows[1][1].set_text.assert_called_once_with("0.500")


def test_apply_color_pools_rebuilds_list_with_fresh_tuples() -> None:
    """_apply_color_pools writes the edited hex/weight as new (hex, weight) tuples."""

    root = _Root([("#FF0000", 2.0), ("#00FF00", 0.5)])
    registry = SliderRegistry()
    rows = [
        (_entry("#123456"), _entry("3.5")),
        (_entry("#00FF00"), _entry("0.5")),
    ]
    _register_pool(registry, rows)

    _apply_color_pools(registry, root)

    assert root.pool.colors == [("#123456", 3.5), ("#00FF00", 0.5)]
    assert all(isinstance(c, tuple) for c in root.pool.colors)


def test_apply_color_pools_blank_hex_falls_back_to_prior() -> None:
    """_apply_color_pools keeps the current hex when the entry is blank."""

    root = _Root([("#ABCDEF", 1.0)])
    registry = SliderRegistry()
    _register_pool(registry, [(_entry("   "), _entry("2.0"))])

    _apply_color_pools(registry, root)

    assert root.pool.colors == [("#ABCDEF", 2.0)]


def test_apply_color_pools_unparseable_weight_falls_back_to_prior() -> None:
    """_apply_color_pools keeps the current weight when the entry is not a float."""

    root = _Root([("#ABCDEF", 1.5)])
    registry = SliderRegistry()
    _register_pool(registry, [(_entry("#111111"), _entry("not-a-number"))])

    _apply_color_pools(registry, root)

    assert root.pool.colors == [("#111111", 1.5)]


def test_color_pool_round_trip_edits_one_leaves_others() -> None:
    """Read → edit one row → apply reflects the edit and leaves others unchanged."""

    root = _Root([("#FF0000", 2.0), ("#00FF00", 0.5)])
    registry = SliderRegistry()
    rows = [
        (_entry(), _entry()),
        (_entry(), _entry()),
    ]
    _register_pool(registry, rows)
    _read_color_pools(root, registry)

    # Simulate the user editing only row 0's hex; other entries echo what was read.
    rows[0][0].get_text.return_value = "#0000FF"
    rows[0][1].get_text.return_value = "2.000"
    rows[1][0].get_text.return_value = "#00FF00"
    rows[1][1].get_text.return_value = "0.500"

    _apply_color_pools(registry, root)

    assert root.pool.colors == [("#0000FF", 2.0), ("#00FF00", 0.5)]
