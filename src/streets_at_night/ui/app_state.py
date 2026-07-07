"""Shared mutable state for the tabbed UI.

Pure data module: no pygame imports so it stays trivially testable. Tabs read
and mutate this single source of truth (config path, working config, cached
scene) via the app.
"""

import typing as t
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import SceneConfig
from ..scene.scene import Scene


@dataclass
class AppState:
    """Shared mutable state for the tabbed UI.

    Attributes:
        config_path: Path of the loaded config file, or None until one is
            loaded/saved.
        config: In-memory working copy of the scene config (the "memory
            config"); nothing touches disk until a Save action.
        cache_dir: Resolved cache directory.
        scene: Cached scene, built lazily by tabs.
        scene_quality: Quality preset the cached scene was built at.
        scene_dirty: True when the config changed since the scene was built.
    """

    config_path: t.Optional[Path]
    config: SceneConfig
    cache_dir: Path
    scene: t.Optional[Scene] = None
    scene_quality: t.Optional[str] = None
    scene_dirty: bool = True

    def invalidate_scene(self) -> None:
        """Drop the cached scene and mark it dirty so tabs rebuild it."""
        self.scene = None
        self.scene_dirty = True

    def set_config(self, config: SceneConfig, path: t.Optional[Path]) -> None:
        """Replace the working config and path, then invalidate the scene cache.

        Args:
            config (SceneConfig): New working config to adopt.
            path (Path | None): Path the config came from, or None.
        """
        self.config = config
        self.config_path = path
        self.invalidate_scene()
