"""Keyboard and mouse input handlers that mutate camera state.

Processes pygame events and held-key state to move, rotate, and zoom
the camera. Also provides a HUD overlay renderer.
"""

import typing as t
from dataclasses import dataclass, field

import numpy as np
import pygame

from .camera import Camera


# Movement speeds
MOVE_SPEED: float = 200.0       # meters per second (forward/back/strafe)
ALTITUDE_SPEED: float = 100.0   # meters per second (raise/lower)
YAW_SPEED: float = 45.0         # degrees per second (arrow keys)
PITCH_SPEED: float = 30.0       # degrees per second (arrow keys)
FOV_SPEED: float = 10.0         # degrees per second (+/-)
MOUSE_SENSITIVITY: float = 0.3  # degrees per pixel
SPRINT_MULTIPLIER: float = 3.0  # speed boost when holding Left Shift

FOV_MIN: float = 15.0
FOV_MAX: float = 120.0
PITCH_CLAMP: float = 85.0


@dataclass
class InputResult:
    """Result of processing one frame of input events.

    Attributes:
        quit: True if the user requested quit.
        screenshot: True if the user pressed P.
        reset: True if the user pressed R.
        save_camera: True if the user pressed C.
    """

    quit: bool = False
    screenshot: bool = False
    reset: bool = False
    save_camera: bool = False


@dataclass
class MouseDragState:
    """Tracks right-mouse drag state for camera rotation.

    Attributes:
        dragging: True while the right mouse button is held.
        last_pos: Last recorded mouse position in pixels, or None.
    """

    dragging: bool = False
    last_pos: t.Optional[t.Tuple[int, int]] = None

    def clear(self) -> None:
        """Reset drag state to the initial idle condition."""
        self.dragging = False
        self.last_pos = None


def handle_input(
    events: list,
    camera: Camera,
    dt: float,
    held_keys: t.Any,
    drag_state: MouseDragState,
    original_position: np.ndarray,
    original_target: np.ndarray,
    original_fov: float,
) -> InputResult:
    """Process pygame events and held keys, applying camera mutations.

    Handles quit, discrete key presses (R, P, ESC, +/-), scroll-wheel
    altitude, held movement keys (W/S/A/D/Q/E/arrows), and right-mouse-drag
    rotation. Drag state is carried in the provided MouseDragState instance
    so callers can manage lifetime and reset it independently.

    Args:
        events (list): List of pygame.Event from pygame.event.get().
        camera (Camera): Camera to mutate in-place.
        dt (float): Delta time in seconds since last frame.
        held_keys: Result of pygame.key.get_pressed().
        drag_state (MouseDragState): Mutable drag state to read/update.
        original_position (np.ndarray): Camera position at startup for reset.
        original_target (np.ndarray): Camera target at startup for reset.
        original_fov (float): Camera FOV at startup for reset.

    Returns:
        InputResult: Flags for quit, screenshot, and reset.
    """
    result = InputResult()

    speed_mult = SPRINT_MULTIPLIER if held_keys[pygame.K_LSHIFT] else 1.0
    _process_discrete_events(events, camera, result, drag_state, speed_mult)
    _process_held_keys(held_keys, camera, dt)

    return result


def render_hud(
    surface: pygame.Surface,
    camera: Camera,
    fps: float,
    segments_drawn: int,
    font: pygame.font.Font,
) -> None:
    """Draw the HUD overlay onto a pygame surface.

    Renders FPS, segment count, camera altitude, and FOV into a semi-
    transparent box in the top-left corner.

    Args:
        surface (pygame.Surface): The surface to draw the HUD onto.
        camera (Camera): Camera to read state from.
        fps (float): Current frames per second.
        segments_drawn (int): Number of road segments drawn this frame.
        font (pygame.font.Font): Font for rendering HUD text.
    """
    lines = _build_hud_lines(camera, fps, segments_drawn)
    _draw_hud_background(surface, font, lines)
    _draw_hud_text(surface, font, lines)


# ---------------------------------------------------------------------------
# Private — event processing
# ---------------------------------------------------------------------------


def _process_discrete_events(
    events: list,
    camera: Camera,
    result: InputResult,
    drag_state: MouseDragState,
    speed_mult: float = 1.0,
) -> None:
    """Handle per-event (non-held) key, mouse, and quit events.

    Args:
        events (list): Pygame event list.
        camera (Camera): Camera to mutate.
        result (InputResult): Result object to update.
        drag_state (MouseDragState): Mouse drag state to read/update.
        speed_mult (float): Speed multiplier (e.g. sprint).
    """
    for event in events:
        if event.type == pygame.QUIT:
            result.quit = True
        elif event.type == pygame.KEYDOWN:
            _handle_keydown(event.key, camera, result)
        elif event.type == pygame.MOUSEWHEEL:
            _handle_scroll(event.y, camera, speed_mult)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            drag_state.dragging = True
            drag_state.last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            drag_state.clear()
        elif event.type == pygame.MOUSEMOTION and drag_state.dragging:
            _apply_mouse_rotation(event.pos, camera, drag_state)


def _handle_keydown(
    key: int,
    camera: Camera,
    result: InputResult,
) -> None:
    """Process a single KEYDOWN event.

    Args:
        key (int): pygame key constant.
        camera (Camera): Camera to mutate.
        result (InputResult): Result object to update.
    """
    if key == pygame.K_ESCAPE:
        result.quit = True
    elif key == pygame.K_r:
        result.reset = True
    elif key == pygame.K_p:
        result.screenshot = True
    elif key == pygame.K_c:
        result.save_camera = True
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        camera.fov_deg = min(FOV_MAX, camera.fov_deg + FOV_SPEED)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        camera.fov_deg = max(FOV_MIN, camera.fov_deg - FOV_SPEED)


def _handle_scroll(scroll_y: int, camera: Camera, speed_mult: float = 1.0) -> None:
    """Adjust camera altitude via scroll-wheel input.

    Positive scroll_y raises the camera; negative lowers it.

    Args:
        scroll_y (int): Scroll delta from pygame.MOUSEWHEEL event.
        camera (Camera): Camera to mutate.
        speed_mult (float): Speed multiplier (e.g. sprint).
    """
    altitude_delta = scroll_y * ALTITUDE_SPEED * 0.5 * speed_mult
    camera.move(np.array([0.0, 0.0, altitude_delta]))


def _process_held_keys(
    held_keys: t.Any,
    camera: Camera,
    dt: float,
) -> None:
    """Apply movement and rotation from held keys.

    Args:
        held_keys: pygame.key.get_pressed() result.
        camera (Camera): Camera to mutate.
        dt (float): Delta time in seconds.
    """
    speed_mult = SPRINT_MULTIPLIER if held_keys[pygame.K_LSHIFT] else 1.0
    _apply_translation(held_keys, camera, dt, speed_mult)
    _apply_rotation(held_keys, camera, dt, speed_mult)


def _apply_translation(
    held_keys: t.Any,
    camera: Camera,
    dt: float,
    speed_mult: float = 1.0,
) -> None:
    """Apply W/S/A/D/Q/E translation to the camera.

    W/S move forward/backward on the ground plane. A/D strafe left/right.
    Q/E move up/down along the world Z axis.

    Args:
        held_keys: pygame.key.get_pressed() result.
        camera (Camera): Camera to mutate.
        dt (float): Delta time in seconds.
        speed_mult (float): Speed multiplier (e.g. sprint).
    """
    forward_ground = _get_ground_forward(camera)
    right = camera.get_right()
    world_up = np.array([0.0, 0.0, 1.0])
    move = MOVE_SPEED * dt * speed_mult
    alt = ALTITUDE_SPEED * dt * speed_mult

    delta = np.zeros(3)
    if held_keys[pygame.K_w]:
        delta += forward_ground * move
    if held_keys[pygame.K_s]:
        delta -= forward_ground * move
    if held_keys[pygame.K_d]:
        delta += right * move
    if held_keys[pygame.K_a]:
        delta -= right * move
    if held_keys[pygame.K_e]:
        delta += world_up * alt
    if held_keys[pygame.K_q]:
        delta -= world_up * alt

    if np.any(delta != 0):
        camera.move(delta)


def _apply_rotation(
    held_keys: t.Any,
    camera: Camera,
    dt: float,
    speed_mult: float = 1.0,
) -> None:
    """Apply arrow key yaw/pitch rotation to the camera.

    Args:
        held_keys: pygame.key.get_pressed() result.
        camera (Camera): Camera to mutate.
        dt (float): Delta time in seconds.
        speed_mult (float): Speed multiplier (e.g. sprint).
    """
    yaw = 0.0
    pitch = 0.0

    if held_keys[pygame.K_LEFT]:
        yaw += YAW_SPEED * dt * speed_mult
    if held_keys[pygame.K_RIGHT]:
        yaw -= YAW_SPEED * dt * speed_mult
    if held_keys[pygame.K_UP]:
        pitch += PITCH_SPEED * dt * speed_mult
    if held_keys[pygame.K_DOWN]:
        pitch -= PITCH_SPEED * dt * speed_mult

    if yaw != 0.0 or pitch != 0.0:
        _rotate_clamped(camera, yaw, pitch)


def _rotate_clamped(camera: Camera, yaw_deg: float, pitch_deg: float) -> None:
    """Apply yaw/pitch rotation while clamping pitch to ±85°.

    Args:
        camera (Camera): Camera to mutate.
        yaw_deg (float): Yaw delta in degrees.
        pitch_deg (float): Pitch delta in degrees.
    """
    forward = camera.get_forward()
    current_pitch = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
    new_pitch = np.clip(current_pitch + pitch_deg, -PITCH_CLAMP, PITCH_CLAMP)
    clamped_pitch = new_pitch - current_pitch
    camera.rotate(yaw_deg, clamped_pitch)


# ---------------------------------------------------------------------------
# Private — mouse drag
# ---------------------------------------------------------------------------


def _apply_mouse_rotation(
    current_pos: t.Tuple[int, int],
    camera: Camera,
    drag_state: MouseDragState,
) -> None:
    """Apply camera rotation from mouse movement delta.

    Args:
        current_pos (tuple[int, int]): Current mouse position in pixels.
        camera (Camera): Camera to mutate.
        drag_state (MouseDragState): Drag state providing last_pos; updated in place.
    """
    if drag_state.last_pos is None:
        drag_state.last_pos = current_pos
        return

    dx = current_pos[0] - drag_state.last_pos[0]
    dy = current_pos[1] - drag_state.last_pos[1]
    drag_state.last_pos = current_pos

    yaw = -dx * MOUSE_SENSITIVITY
    pitch = -dy * MOUSE_SENSITIVITY
    _rotate_clamped(camera, yaw, pitch)


# ---------------------------------------------------------------------------
# Private — HUD helpers
# ---------------------------------------------------------------------------


def _build_hud_lines(
    camera: Camera,
    fps: float,
    segments_drawn: int,
) -> t.List[str]:
    """Build the list of HUD text lines.

    Args:
        camera (Camera): Camera to read state from.
        fps (float): Current frames per second.
        segments_drawn (int): Number of road segments drawn.

    Returns:
        list[str]: Lines to display in the HUD.
    """
    alt_m = camera.position[2]
    fov = camera.fov_deg
    return [
        f"FPS: {fps:.1f}",
        f"Segments: {segments_drawn}",
        f"Alt: {alt_m:.0f}m",
        f"FOV: {fov:.1f}°",
        "P=screenshot  R=reset  C=save cam",
    ]


def _draw_hud_background(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: t.List[str],
) -> None:
    """Draw a semi-transparent rectangle behind the HUD text.

    Args:
        surface (pygame.Surface): Target surface.
        font (pygame.font.Font): Font used to measure text height.
        lines (list[str]): HUD lines.
    """
    line_height = font.get_height() + 2
    box_w = 240
    box_h = len(lines) * line_height + 8

    overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surface.blit(overlay, (4, 4))


def _draw_hud_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: t.List[str],
) -> None:
    """Render HUD text lines onto the surface.

    Args:
        surface (pygame.Surface): Target surface.
        font (pygame.font.Font): Font for rendering text.
        lines (list[str]): HUD lines to render.
    """
    line_height = font.get_height() + 2
    x, y = 8, 8
    for line in lines:
        text_surf = font.render(line, True, (220, 220, 220))
        surface.blit(text_surf, (x, y))
        y += line_height


# ---------------------------------------------------------------------------
# Private — math helpers
# ---------------------------------------------------------------------------


def _get_ground_forward(camera: Camera) -> np.ndarray:
    """Return the forward vector projected onto the XY ground plane, normalized.

    Ignores the Z component so movement stays on the ground plane.

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (3,) unit vector on the ground plane.
    """
    fwd = camera.get_forward()
    ground = np.array([fwd[0], fwd[1], 0.0])
    norm = float(np.linalg.norm(ground))
    if norm < 1e-10:
        return np.array([0.0, 1.0, 0.0])
    return ground / norm
