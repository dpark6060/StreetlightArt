# Radical
"""Tests for streets_at_night.devmode.cmd_devmode render pass assembly."""

from unittest import mock

from streets_at_night.devmode.cmd_devmode import _build_render_passes
from streets_at_night.rendering.compositor import render_composite


def test_build_render_passes_includes_composite():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=False)

    # Assert — without composite the accumulated light buffer never resolves and
    # only the dim substrate (ground/roads) shows; lights are invisible.
    assert render_composite in passes


def test_build_render_passes_composite_is_last_when_no_labels():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=False)

    # Assert — composite must run after the light passes that fill the buffer
    assert passes[-1] is render_composite


def test_build_render_passes_labels_come_after_composite():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=True)

    # Assert — labels overlay the final composited frame, so they run last
    assert passes.index(render_composite) < len(passes) - 1
    assert passes[-1] is not render_composite
