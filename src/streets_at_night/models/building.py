"""Building data model — extruded footprint geometry for mass and window rendering."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Building:
    """A single building footprint with a resolved height and window seed.

    Data container only. Wall/roof extrusion and window-grid derivation
    happen at render time (see rendering/building_pass.py, Phase B/C) — the
    ``window_seed`` is stored here so window layout is reproducible without
    materializing per-window objects for every building in the scene.

    Attributes:
        footprint: (N, 2) ENU exterior ring in meters, CCW winding, with no
            closing duplicate vertex.
        height: Building height in meters, resolved from OSM tags or a
            per-type default (see placement.building_builder).
        building_kind: Raw OSM ``building`` tag value (e.g. "house",
            "retail"), used for stats/debug and height-default lookup.
        window_seed: Per-building seed for reproducible window layout.
    """

    footprint: np.ndarray
    height: float
    building_kind: str
    window_seed: int
