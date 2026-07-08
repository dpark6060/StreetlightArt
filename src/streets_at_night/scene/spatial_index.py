"""Uniform grid spatial index for fast bounding-box queries.

Cell size of 500m over a 40km area produces an 80x80 = 6400 cell grid,
giving O(1) insert and lookup for road segment culling.
"""

import typing as t

import numpy as np


class SpatialIndex:
    """Uniform grid spatial index. O(1) insert and lookup.

    Items are inserted into all cells that overlap their bounding box.
    Queries return all items in overlapping cells — callers may receive
    duplicates if an item spans multiple cells.

    Cell size of 500m over a 40km area = 80x80 = 6400 cells.
    """

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        cell_size: float = 500.0,
    ) -> None:
        """Initialise the spatial index.

        Args:
            min_x (float): World minimum x bound in meters.
            min_y (float): World minimum y bound in meters.
            max_x (float): World maximum x bound in meters.
            max_y (float): World maximum y bound in meters.
            cell_size (float): Grid cell size in meters. Default 500m.
        """
        self._min_x = min_x
        self._min_y = min_y
        self._cell_size = cell_size
        self._grid: t.Dict[t.Tuple[int, int], t.List[t.Any]] = {}
        self._item_count = 0
        self._max_cx = max(0, int((max_x - min_x) / cell_size))
        self._max_cy = max(0, int((max_y - min_y) / cell_size))

    def insert(
        self,
        item: t.Any,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> None:
        """Insert an item into all cells that overlap its bounding box.

        Args:
            item: The item to store.
            min_x (float): Bounding box minimum x.
            min_y (float): Bounding box minimum y.
            max_x (float): Bounding box maximum x.
            max_y (float): Bounding box maximum y.
        """
        cx_min, cy_min = self._world_to_cell(min_x, min_y)
        cx_max, cy_max = self._world_to_cell(max_x, max_y)

        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                key = (cx, cy)
                if key not in self._grid:
                    self._grid[key] = []
                self._grid[key].append(item)
                self._item_count += 1

    def insert_points(
        self,
        items: t.Sequence[t.Any],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        """Insert many point items in one call, computing cells vectorized.

        Equivalent to calling ``insert(item, x, y, x, y)`` for each item, but
        the cell index for every point is computed with a single vectorized
        floor-divide + clamp instead of two per-point ``_world_to_cell`` calls.
        The resulting cell assignment (and item order within a cell) is
        identical to the per-item path.

        Args:
            items (Sequence): Point items to store, aligned with xs/ys.
            xs (np.ndarray): (N,) world x coordinates.
            ys (np.ndarray): (N,) world y coordinates.
        """
        cxs = np.clip(
            ((xs - self._min_x) / self._cell_size).astype(int), 0, self._max_cx
        )
        cys = np.clip(
            ((ys - self._min_y) / self._cell_size).astype(int), 0, self._max_cy
        )

        grid = self._grid
        for item, cx, cy in zip(items, cxs.tolist(), cys.tolist()):
            key = (cx, cy)
            cell = grid.get(key)
            if cell is None:
                grid[key] = [item]
            else:
                cell.append(item)
        self._item_count += len(items)

    def query_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> t.List[t.Any]:
        """Return all items whose bounding box overlaps the query rectangle.

        May contain duplicates if items span multiple cells.

        Args:
            min_x (float): Query rectangle minimum x.
            min_y (float): Query rectangle minimum y.
            max_x (float): Query rectangle maximum x.
            max_y (float): Query rectangle maximum y.

        Returns:
            list: All items in overlapping cells. May contain duplicates.
        """
        cx_min, cy_min = self._world_to_cell(min_x, min_y)
        cx_max, cy_max = self._world_to_cell(max_x, max_y)
        return self._collect_cells(cx_min, cy_min, cx_max, cy_max)

    def query_radius(self, x: float, y: float, radius: float) -> t.List[t.Any]:
        """Return all items within radius of (x, y).

        Uses a conservative bounding box — result may contain duplicates
        and items beyond the exact radius.

        Args:
            x (float): Query center x in meters.
            y (float): Query center y in meters.
            radius (float): Search radius in meters.

        Returns:
            list: Items in cells overlapping the bounding circle. May contain duplicates.
        """
        return self.query_rect(x - radius, y - radius, x + radius, y + radius)

    def get_item_count(self) -> int:
        """Return total number of indexed items, counting duplicates across cells.

        Returns:
            int: Total item count summed across all cells.
        """
        return self._item_count

    def _world_to_cell(self, x: float, y: float) -> t.Tuple[int, int]:
        """Convert world coordinates to cell indices, clamped to valid range.

        Coordinates outside the world bounds are clamped to [0, max_cell] so
        query AABBs larger than the data do not produce negative indices or
        wasteful iteration over empty range keys.

        Args:
            x (float): World x coordinate.
            y (float): World y coordinate.

        Returns:
            tuple[int, int]: (cell_x, cell_y) integer indices in valid range.
        """
        cx = max(0, min(self._max_cx, int((x - self._min_x) / self._cell_size)))
        cy = max(0, min(self._max_cy, int((y - self._min_y) / self._cell_size)))
        return cx, cy

    def _collect_cells(
        self,
        cx_min: int,
        cy_min: int,
        cx_max: int,
        cy_max: int,
    ) -> t.List[t.Any]:
        """Collect all items from cells in the given cell range.

        Args:
            cx_min (int): Minimum cell x index.
            cy_min (int): Minimum cell y index.
            cx_max (int): Maximum cell x index.
            cy_max (int): Maximum cell y index.

        Returns:
            list: All items from the covered cells.
        """
        result = []
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                items = self._grid.get((cx, cy))
                if items:
                    result.extend(items)
        return result
