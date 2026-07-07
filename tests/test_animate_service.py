# Radical
"""Tests for the heat-haze animate service: recording and duration logic."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from streets_at_night.rendering import animate_service
from streets_at_night.rendering.animate_service import default_video_duration, record_haze

_MODULE = "streets_at_night.rendering.animate_service"


def _stub_haze() -> mock.Mock:
    """Return a haze stub whose apply returns a tiny RGB frame."""
    haze = mock.Mock()
    haze.apply.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    return haze


def test_record_haze_appends_n_frames_and_closes_writer(tmp_path):
    # Arrange
    writer = mock.Mock()
    haze = _stub_haze()
    n_frames = 5

    # Act
    with mock.patch.object(animate_service, "open_video_writer", return_value=writer):
        record_haze(haze, fps=30, n_frames=n_frames, out_path=tmp_path / "out.mp4")

    # Assert
    assert writer.append_data.call_count == n_frames
    writer.close.assert_called_once()


def test_record_haze_closes_writer_when_frame_raises(tmp_path):
    # Arrange
    writer = mock.Mock()
    haze = _stub_haze()
    haze.apply.side_effect = [np.zeros((2, 2, 3), dtype=np.uint8), RuntimeError("gpu boom")]

    # Act / Assert
    with mock.patch.object(animate_service, "open_video_writer", return_value=writer):
        with pytest.raises(RuntimeError, match="gpu boom"):
            record_haze(haze, fps=30, n_frames=5, out_path=tmp_path / "out.mp4")

    # Assert — the finally block must still close the writer.
    writer.close.assert_called_once()


def test_record_haze_reports_ascending_fractions(tmp_path):
    # Arrange
    writer = mock.Mock()
    haze = _stub_haze()
    fractions: list = []
    n_frames = 4

    # Act
    with mock.patch.object(animate_service, "open_video_writer", return_value=writer):
        record_haze(
            haze,
            fps=30,
            n_frames=n_frames,
            out_path=tmp_path / "out.mp4",
            progress=lambda msg, frac: fractions.append(frac),
        )

    # Assert
    assert fractions == [pytest.approx(i / n_frames) for i in range(1, n_frames + 1)]
    assert fractions == sorted(fractions)


@pytest.mark.parametrize(
    "period, expected",
    [
        (0.0, 10.0),
        (2.0, 4.0),
        (7.0, 7.0),
    ],
)
def test_default_video_duration_from_loop_period(period, expected):
    # Arrange
    haze = mock.Mock()
    haze.loop_period_s.return_value = period

    # Act / Assert
    assert default_video_duration(haze) == expected
