"""wgpu renderer for the tuning scene: HDR scene pass, then a composite pass.

Pass 1 renders into an rgba16float HDR target with depth: opaque surfaces
(ground pools evaluated per pixel), emissive windows, then additive light
sprites. Pass 2 grades the HDR image into the output texture (a window
swapchain texture or an offscreen texture for tests and PNG export).
"""

import functools
import typing as t
from pathlib import Path

import numpy as np
import wgpu

from .gpu_scene import FRAME_UNIFORM_BYTES, SPRITE_FLOATS, VERTEX_FLOATS

HDR_FORMAT: str = wgpu.TextureFormat.rgba16float
DEPTH_FORMAT: str = wgpu.TextureFormat.depth32float

_SHADER_DIR: Path = Path(__file__).parent / "shaders"

# wgpu requires texture-to-buffer row pitches to be multiples of 256 bytes.
_ROW_ALIGN_BYTES: int = 256

_ADDITIVE_BLEND: dict = {
    "color": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
}


@functools.lru_cache(maxsize=1)
def get_device() -> wgpu.GPUDevice:
    """Return the process-wide GPU device (high-performance adapter).

    Returns:
        wgpu.GPUDevice: The device.
    """
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()


def _load_shader(*names: str) -> str:
    """Concatenate WGSL sources (WGSL has no #include).

    Args:
        *names (str): Shader file names in shaders/, in order.

    Returns:
        str: The combined WGSL source.
    """
    return "\n".join((_SHADER_DIR / name).read_text() for name in names)


class TuningRenderer:
    """Owns the pipelines, buffers, and render targets for the tuning scene."""

    def __init__(self, device: wgpu.GPUDevice, output_format: str) -> None:
        """Compile shaders and build pipelines.

        Args:
            device (wgpu.GPUDevice): GPU device.
            output_format (str): Texture format of the final output.
        """
        self._device = device
        self.output_format = output_format
        self._uniforms = device.create_buffer(
            size=FRAME_UNIFORM_BYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._scene_layout = device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ])
        self._sprite_layout = device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX,
                "texture": {"sample_type": wgpu.TextureSampleType.depth},
            },
        ])
        self._composite_layout = device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float},
            },
        ])
        self._build_pipelines()
        self._opaque_vertices: t.Optional[wgpu.GPUBuffer] = None
        self._opaque_count = 0
        self._window_vertices: t.Optional[wgpu.GPUBuffer] = None
        self._window_count = 0
        self._size: t.Tuple[int, int] = (0, 0)
        self._hdr: t.Optional[wgpu.GPUTexture] = None
        self._depth: t.Optional[wgpu.GPUTexture] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_geometry(self, opaque: np.ndarray, windows: np.ndarray) -> None:
        """Upload scene geometry (call only when the scene changes).

        Args:
            opaque (np.ndarray): (N, VERTEX_FLOATS) opaque triangle vertices.
            windows (np.ndarray): (M, VERTEX_FLOATS) emissive window vertices.
        """
        self._opaque_vertices, self._opaque_count = self._vertex_buffer(opaque)
        self._window_vertices, self._window_count = self._vertex_buffer(windows)

    def render(
        self,
        frame_uniforms: bytes,
        pool_lights: np.ndarray,
        sprites: np.ndarray,
        sky_linear: t.Tuple[float, float, float],
        target: wgpu.GPUTextureView,
        size: t.Tuple[int, int],
    ) -> None:
        """Render one frame into the target view.

        Args:
            frame_uniforms (bytes): Packed Frame uniform block.
            pool_lights (np.ndarray): (K, POOL_LIGHT_FLOATS) ground-pool records.
            sprites (np.ndarray): (S, SPRITE_FLOATS) light sprite instances.
            sky_linear (tuple[float, float, float]): Linear background color.
            target (wgpu.GPUTextureView): Output texture view (output_format).
            size (tuple[int, int]): (width, height) in pixels.
        """
        device = self._device
        self._ensure_targets(size)
        device.queue.write_buffer(self._uniforms, 0, frame_uniforms)
        pools = device.create_buffer_with_data(
            data=np.ascontiguousarray(pool_lights, dtype=np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        scene_group = device.create_bind_group(layout=self._scene_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._uniforms}},
            {"binding": 1, "resource": {"buffer": pools}},
        ])
        composite_group = device.create_bind_group(layout=self._composite_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._uniforms}},
            {"binding": 1, "resource": self._hdr.create_view()},
        ])

        encoder = device.create_command_encoder()
        scene_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": self._hdr.create_view(),
                "clear_value": (*sky_linear, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }],
            depth_stencil_attachment={
                "view": self._depth.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        scene_pass.set_bind_group(0, scene_group)
        if self._opaque_count:
            scene_pass.set_pipeline(self._opaque_pipeline)
            scene_pass.set_vertex_buffer(0, self._opaque_vertices)
            scene_pass.draw(self._opaque_count)
        if self._window_count:
            scene_pass.set_pipeline(self._window_pipeline)
            scene_pass.set_vertex_buffer(0, self._window_vertices)
            scene_pass.draw(self._window_count)
        scene_pass.end()

        if len(sprites):
            self._draw_sprites(encoder, sprites)

        composite_pass = encoder.begin_render_pass(color_attachments=[{
            "view": target,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }])
        composite_pass.set_pipeline(self._composite_pipeline)
        composite_pass.set_bind_group(0, composite_group)
        composite_pass.draw(3)
        composite_pass.end()

        device.queue.submit([encoder.finish()])

    def render_offscreen(
        self,
        frame_uniforms: bytes,
        pool_lights: np.ndarray,
        sprites: np.ndarray,
        sky_linear: t.Tuple[float, float, float],
        size: t.Tuple[int, int],
    ) -> np.ndarray:
        """Render one graded frame to an offscreen texture and read it back.

        Requires output_format rgba8unorm with encode_srgb set in the
        uniforms, so the composite shader does the sRGB encode itself.

        Args:
            frame_uniforms (bytes): Packed Frame uniforms.
            pool_lights (np.ndarray): Ground-pool records.
            sprites (np.ndarray): Sprite instances.
            sky_linear (tuple[float, float, float]): Linear background color.
            size (tuple[int, int]): (width, height) in pixels.

        Returns:
            np.ndarray: (H, W, 3) uint8 sRGB image.
        """
        width, height = size
        texture = self._device.create_texture(
            size=(width, height, 1),
            format=self.output_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.render(frame_uniforms, pool_lights, sprites, sky_linear, texture.create_view(), size)
        rows = self._read_texture(texture, width, height, bytes_per_pixel=4)
        return rows.reshape(height, -1, 4)[:, :width, :3].copy()

    def read_hdr(self) -> np.ndarray:
        """Read back the last frame's HDR scene target (before grading).

        Returns:
            np.ndarray: (H, W, 3) float32 linear radiance.
        """
        width, height = self._size
        rows = self._read_texture(self._hdr, width, height, bytes_per_pixel=8)
        return rows.view(np.float16).reshape(height, -1, 4)[:, :width, :3].astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipelines(self) -> None:
        """Compile the WGSL modules and create the four render pipelines."""
        device = self._device
        geometry = device.create_shader_module(
            code=_load_shader("common.wgsl", "illumination.wgsl", "geometry.wgsl")
        )
        sprites = device.create_shader_module(code=_load_shader("common.wgsl", "sprites.wgsl"))
        composite = device.create_shader_module(code=_load_shader("common.wgsl", "composite.wgsl"))
        scene_layout = device.create_pipeline_layout(bind_group_layouts=[self._scene_layout])

        vertex_layout = {
            "array_stride": VERTEX_FLOATS * 4,
            "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                {"format": wgpu.VertexFormat.float32, "offset": 24, "shader_location": 2},
            ],
        }
        sprite_layout = {
            "array_stride": SPRITE_FLOATS * 4,
            "step_mode": wgpu.VertexStepMode.instance,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x4, "offset": 0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32x4, "offset": 16, "shader_location": 1},
                {"format": wgpu.VertexFormat.float32x4, "offset": 32, "shader_location": 2},
            ],
        }

        self._opaque_pipeline = self._scene_pipeline(
            scene_layout, geometry, "vs_geometry", "fs_opaque", [vertex_layout],
            blend=None, depth_write=True, depth_compare=wgpu.CompareFunction.less,
        )
        self._window_pipeline = self._scene_pipeline(
            scene_layout, geometry, "vs_geometry", "fs_window", [vertex_layout],
            blend=_ADDITIVE_BLEND, depth_write=False,
            depth_compare=wgpu.CompareFunction.less_equal,
        )
        self._sprite_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._sprite_layout]),
            vertex={"module": sprites, "entry_point": "vs_sprite", "buffers": [sprite_layout]},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
            fragment={
                "module": sprites,
                "entry_point": "fs_sprite",
                "targets": [{"format": HDR_FORMAT, "blend": _ADDITIVE_BLEND}],
            },
        )
        self._composite_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._composite_layout]),
            vertex={"module": composite, "entry_point": "vs_fullscreen"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={
                "module": composite,
                "entry_point": "fs_composite",
                "targets": [{"format": self.output_format}],
            },
        )

    def _draw_sprites(self, encoder: wgpu.GPUCommandEncoder, sprites: np.ndarray) -> None:
        """Add light cores and glows onto the HDR target in their own pass.

        A separate pass so the vertex shader can read the finished scene
        depth for its source-occlusion test.

        Args:
            encoder (wgpu.GPUCommandEncoder): Frame command encoder.
            sprites (np.ndarray): (S, SPRITE_FLOATS) sprite instances.
        """
        device = self._device
        instances = device.create_buffer_with_data(
            data=np.ascontiguousarray(sprites, dtype=np.float32),
            usage=wgpu.BufferUsage.VERTEX,
        )
        group = device.create_bind_group(layout=self._sprite_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._uniforms}},
            {"binding": 1, "resource": self._depth.create_view()},
        ])
        sprite_pass = encoder.begin_render_pass(color_attachments=[{
            "view": self._hdr.create_view(),
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        sprite_pass.set_pipeline(self._sprite_pipeline)
        sprite_pass.set_bind_group(0, group)
        sprite_pass.set_vertex_buffer(0, instances)
        sprite_pass.draw(6, len(sprites))
        sprite_pass.end()

    def _scene_pipeline(
        self,
        layout: wgpu.GPUPipelineLayout,
        module: wgpu.GPUShaderModule,
        vs: str,
        fs: str,
        buffers: list,
        blend: t.Optional[dict],
        depth_write: bool,
        depth_compare: str,
    ) -> wgpu.GPURenderPipeline:
        """Create one HDR scene-pass pipeline.

        Args:
            layout (wgpu.GPUPipelineLayout): Pipeline layout.
            module (wgpu.GPUShaderModule): Shader module.
            vs (str): Vertex entry point.
            fs (str): Fragment entry point.
            buffers (list): Vertex buffer layouts.
            blend (dict | None): Blend state, None for opaque.
            depth_write (bool): Whether to write depth.
            depth_compare (str): Depth compare function.

        Returns:
            wgpu.GPURenderPipeline: The pipeline.
        """
        target = {"format": HDR_FORMAT}
        if blend is not None:
            target["blend"] = blend
        return self._device.create_render_pipeline(
            layout=layout,
            vertex={"module": module, "entry_point": vs, "buffers": buffers},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": DEPTH_FORMAT,
                "depth_write_enabled": depth_write,
                "depth_compare": depth_compare,
            },
            fragment={"module": module, "entry_point": fs, "targets": [target]},
        )

    def _vertex_buffer(self, vertices: np.ndarray) -> t.Tuple[t.Optional[wgpu.GPUBuffer], int]:
        """Upload vertices, or return (None, 0) for an empty array.

        Args:
            vertices (np.ndarray): (N, VERTEX_FLOATS) float32 vertices.

        Returns:
            tuple[GPUBuffer | None, int]: The buffer and vertex count.
        """
        if len(vertices) == 0:
            return None, 0
        buffer = self._device.create_buffer_with_data(
            data=np.ascontiguousarray(vertices, dtype=np.float32),
            usage=wgpu.BufferUsage.VERTEX,
        )
        return buffer, len(vertices)

    def _ensure_targets(self, size: t.Tuple[int, int]) -> None:
        """(Re)create the HDR and depth targets when the viewport size changes.

        Args:
            size (tuple[int, int]): (width, height) in pixels.
        """
        if size == self._size and self._hdr is not None:
            return
        width, height = size
        self._hdr = self._device.create_texture(
            size=(width, height, 1),
            format=HDR_FORMAT,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
            ),
        )
        self._depth = self._device.create_texture(
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._size = size

    def _read_texture(
        self, texture: wgpu.GPUTexture, width: int, height: int, bytes_per_pixel: int
    ) -> np.ndarray:
        """Copy a texture to host memory with row-aligned pitch.

        Args:
            texture (wgpu.GPUTexture): Texture with COPY_SRC usage.
            width (int): Width in pixels.
            height (int): Height in pixels.
            bytes_per_pixel (int): Bytes per texel.

        Returns:
            np.ndarray: (height, padded_row_bytes) uint8 rows.
        """
        row_bytes = width * bytes_per_pixel
        padded = (row_bytes + _ROW_ALIGN_BYTES - 1) // _ROW_ALIGN_BYTES * _ROW_ALIGN_BYTES
        data = self._device.queue.read_texture(
            {"texture": texture, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": padded, "rows_per_image": height},
            (width, height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(height, padded)

