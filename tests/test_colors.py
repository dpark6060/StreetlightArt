# Radical
"""Tests for streets_at_night.utils.colors module."""

from streets_at_night.utils.colors import kelvin_to_rgb


# ---------------------------------------------------------------------------
# kelvin_to_rgb — blackbody temperature
# ---------------------------------------------------------------------------


def test_kelvin_to_rgb_warm_temp_is_red_dominant():
    # Act — low temperature (sodium-like)
    r, g, b = kelvin_to_rgb(2000.0)

    # Assert — warm light is red-dominant with little blue
    assert r > g > b


def test_kelvin_to_rgb_cool_temp_has_more_blue_than_warm():
    # Act
    _, _, warm_b = kelvin_to_rgb(2000.0)
    _, _, cool_b = kelvin_to_rgb(6500.0)

    # Assert — cooler light carries more blue
    assert cool_b > warm_b


def test_kelvin_to_rgb_zero_saturation_is_gray():
    # Act — saturation 0 collapses to luminance gray
    r, g, b = kelvin_to_rgb(2000.0, saturation=0.0)

    # Assert — all channels equal (within rounding)
    assert max(r, g, b) - min(r, g, b) <= 1


def test_kelvin_to_rgb_higher_saturation_widens_channel_spread():
    # Act
    base = kelvin_to_rgb(2000.0, saturation=1.0)
    boosted = kelvin_to_rgb(2000.0, saturation=1.5)

    # Assert — boosting saturation pushes channels further apart
    assert (max(boosted) - min(boosted)) > (max(base) - min(base))


def test_kelvin_to_rgb_clamps_below_minimum():
    # Act — anything under 1000K is treated as 1000K
    below = kelvin_to_rgb(500.0)
    floor = kelvin_to_rgb(1000.0)

    # Assert
    assert below == floor


def test_kelvin_to_rgb_clamps_above_maximum():
    # Act — anything over 40000K is treated as 40000K
    above = kelvin_to_rgb(50000.0)
    ceiling = kelvin_to_rgb(40000.0)

    # Assert
    assert above == ceiling


def test_kelvin_to_rgb_channels_in_byte_range():
    # Act
    rgb = kelvin_to_rgb(3500.0, saturation=1.4)

    # Assert — every channel is a valid 0-255 byte
    assert all(0 <= channel <= 255 for channel in rgb)
