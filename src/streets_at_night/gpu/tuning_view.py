"""Glue between the tuning scene, the per-frame packers, and the renderer.

Used by the interactive window and by headless renders (tests, PNG probes).
"""

import typing as t

import numpy as np
import wgpu

from ..camera.camera import Camera
from ..config.schema import SceneConfig
from ..rendering.color_space import srgb255_to_linear
from .gpu_scene import (
    build_opaque_vertices,
    build_window_vertices,
    pack_frame_uniforms,
    pack_pool_lights,
    pack_sprites,
    window_extinction,
)
from .renderer import TuningRenderer, get_device
from .tuning_scene import PoleFixture, TuningScene, build_tuning_scene

# Same night sky the CPU ground pass clears to.
SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)


class TuningView:
    """A built tuning scene plus the renderer that draws it.

    Attributes:
        config (SceneConfig): Working config; edits take effect on rebuild().
        scene (TuningScene): The current scene build.
    """

    def __init__(
        self,
        config: SceneConfig,
        pole_fixture: PoleFixture,
        output_format: str,
        device: t.Optional[wgpu.GPUDevice] = None,
    ) -> None:
        """Create the renderer and build the initial scene.

        Args:
            config (SceneConfig): Working config.
            pole_fixture (PoleFixture): Initial fixture mounted on the poles.
            output_format (str): Output texture format.
            device (wgpu.GPUDevice, optional): GPU device; defaults to the
                shared high-performance device.
        """
        self.config = config
        self.pole_fixture = pole_fixture
        self.road_lights = True
        self.building_lights = True
        self._encode_srgb = not output_format.endswith("-srgb")
        self._renderer = TuningRenderer(device or get_device(), output_format)
        self.scene: TuningScene = None  # set by rebuild()
        self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the scene and re-upload geometry from the current config."""
        self.scene = build_tuning_scene(
            self.config, self.pole_fixture, self.road_lights, self.building_lights
        )
        self._renderer.set_geometry(
            build_opaque_vertices(self.scene, self.config),
            build_window_vertices(self.scene),
        )

    def draw(self, camera: Camera, target: wgpu.GPUTextureView, size: t.Tuple[int, int]) -> None:
        """Render one frame into a texture view.

        Args:
            camera (Camera): Current camera.
            target (wgpu.GPUTextureView): Output view in the renderer's format.
            size (tuple[int, int]): (width, height) in pixels.
        """
        self._renderer.render(*self._frame_inputs(camera, size), target, size)

    def render_image(self, camera: Camera, size: t.Tuple[int, int]) -> np.ndarray:
        """Render one frame offscreen (rgba8unorm renderers only).

        Args:
            camera (Camera): Camera to render from.
            size (tuple[int, int]): (width, height) in pixels.

        Returns:
            np.ndarray: (H, W, 3) uint8 sRGB image.
        """
        return self._renderer.render_offscreen(*self._frame_inputs(camera, size), size)

    def read_hdr(self) -> np.ndarray:
        """Return the last frame's linear HDR scene buffer.

        Returns:
            np.ndarray: (H, W, 3) float32.
        """
        return self._renderer.read_hdr()

    def _frame_inputs(
        self, camera: Camera, size: t.Tuple[int, int]
    ) -> t.Tuple[bytes, np.ndarray, np.ndarray, t.Tuple[float, float, float]]:
        """Pack uniforms, pools, sprites, and sky color for one frame.

        Args:
            camera (Camera): Current camera.
            size (tuple[int, int]): (width, height) in pixels.

        Returns:
            tuple: (uniform bytes, pool records, sprite instances, sky linear RGB).
        """
        width, height = size
        lights = self.scene.lights
        pools = pack_pool_lights(lights, camera, self.config)
        pool_count = sum(1 for light in lights if light.illumination is not None)
        uniforms = pack_frame_uniforms(
            camera, width, height, self.config, pool_count,
            window_extinction(self.scene, self.config, camera), self._encode_srgb,
        )
        sprites = pack_sprites(lights, camera, width, self.config)
        return uniforms, pools, sprites, srgb255_to_linear(SKY_COLOR)
