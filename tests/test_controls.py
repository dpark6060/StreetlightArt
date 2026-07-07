# Radical
"""Unit tests for streets_at_night.camera.controls."""

import numpy as np
import pygame
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.camera.controls import FOV_MAX, FOV_MIN, FOV_SPEED, _apply_fov


class _FakeHeldKeys(dict):
    """Minimal stand-in for pygame.key.get_pressed(), defaulting unset keys to False."""

    def __getitem__(self, key: int) -> bool:
        return self.get(key, False)


@pytest.fixture
def camera():
    """Return a camera with a fixed starting FOV of 60 degrees."""
    return create_camera(
        position=np.array([0.0, -5000.0, 1500.0]),
        target=np.array([0.0, 0.0, 300.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


def test_apply_fov_plus_held_increases_fov_by_speed_times_dt(camera):
    # Arrange
    held_keys = _FakeHeldKeys({pygame.K_EQUALS: True})

    # Act
    _apply_fov(held_keys, camera, dt=0.5)

    # Assert
    assert camera.fov_deg == pytest.approx(60.0 + FOV_SPEED * 0.5)


def test_apply_fov_minus_held_decreases_fov_by_speed_times_dt(camera):
    # Arrange
    held_keys = _FakeHeldKeys({pygame.K_MINUS: True})

    # Act
    _apply_fov(held_keys, camera, dt=0.5)

    # Assert
    assert camera.fov_deg == pytest.approx(60.0 - FOV_SPEED * 0.5)


def test_apply_fov_scales_by_speed_mult(camera):
    # Arrange
    held_keys = _FakeHeldKeys({pygame.K_EQUALS: True})

    # Act
    _apply_fov(held_keys, camera, dt=0.5, speed_mult=2.0)

    # Assert
    assert camera.fov_deg == pytest.approx(60.0 + FOV_SPEED * 0.5 * 2.0)


def test_apply_fov_clamps_at_max(camera):
    # Arrange
    camera.fov_deg = FOV_MAX - 1.0
    held_keys = _FakeHeldKeys({pygame.K_EQUALS: True})

    # Act
    _apply_fov(held_keys, camera, dt=10.0)

    # Assert
    assert camera.fov_deg == FOV_MAX


def test_apply_fov_clamps_at_min(camera):
    # Arrange
    camera.fov_deg = FOV_MIN + 1.0
    held_keys = _FakeHeldKeys({pygame.K_MINUS: True})

    # Act
    _apply_fov(held_keys, camera, dt=10.0)

    # Assert
    assert camera.fov_deg == FOV_MIN


def test_apply_fov_no_keys_held_leaves_fov_unchanged(camera):
    # Arrange
    held_keys = _FakeHeldKeys()

    # Act
    _apply_fov(held_keys, camera, dt=0.5)

    # Assert
    assert camera.fov_deg == 60.0
