# Plan 04 — Reference-scale "car" marker in the devmode scene

A single red line, the length of an average car, drawn at the beginning of each devmode row
so the user can visually judge the scale of lights, pools, and houses. Devmode only.

---

## 1. Answer

### Car length: 4.7 m

Typical modern passenger cars cluster tightly: Corolla 4.63 m, Civic 4.67 m, Camry 4.88 m,
Accord 4.90 m; compact SUVs (RAV4, CR-V) are 4.6–4.7 m. The US fleet average passenger
vehicle is ~4.6–4.8 m. 4.7 m is the midpoint of that band and inside the requested
4.5–4.8 m range. Define it as a public module constant `CAR_LENGTH_M = 4.7` so the value is
importable by tests and greppable.

### Placement: one marker per row, transverse, near the row start

Row geometry facts (from `devmode/synthetic_scene.py`):

- 10 rows defined in `ROW_LABELS` as `(label, y_offset)` pairs; y from +900 (Highway) down
  to -900 (Houses) in 200 m steps.
- Every row runs along +X from `_ROW_X_START = 0.0` to `_ROW_X_END = 10000.0`. "The
  beginning of each road" is x = 0 on the row's centerline `y = y_offset`, z = 0.
- The existing label pass anchors its text at world `[0, y_offset, 0]`.
- The default devmode camera (config/defaults.py `"devmode"` section) sits at
  `(-500, 0, 800)` looking at `(5000, 0, 0)` — i.e. it looks almost straight **down +X,
  along the rows**.

That last fact decides the marker's orientation. A line laid along X (the "car parked on
the road" orientation) is nearly parallel to the default view direction and foreshortens to
roughly a point. So the marker is drawn **transverse — along Y, perpendicular to the road,
centered on the row centerline**:

```
p1 = (MARKER_X, y_offset - CAR_LENGTH_M / 2, 0.0)
p2 = (MARKER_X, y_offset + CAR_LENGTH_M / 2, 0.0)
```

with `_MARKER_X_M = 30.0` (a small inset from x=0 so the marker does not sit under the
projected label text, which draws at the `[0, y, 0]` anchor). Transverse orientation has a
bonus: it lies across the road, giving a direct visual comparison against road width
(highway default width is 12 m vs the 4.7 m car). z = 0 exactly — the things being judged
(pools, house footprints, road widths) are all ground-plane features, and there is no depth
buffer to fight.

All 10 rows get a marker, including the Houses row (row 9, no road segment) — judging house
scale is an explicit goal of the feature, and `ROW_LABELS` already carries the y offsets for
all 10 rows.

### Rendering approach

Copy the `devmode/label_pass.py` pattern: a new devmode-only module
`devmode/scale_marker_pass.py` exposing a factory that returns a `RenderPass`
(`Callable[[Scene, Camera, RenderBackend], Any]`). Registration happens only in
`devmode/render_helpers.build_render_passes()` under the existing `use_labels` flag — the
same gate that keeps row labels out of real-tiles devmode. The export pipeline
(`rendering/render_pipeline.get_export_passes`) and preview pipeline are untouched, so real
renders can never show it.

Draw order: **after `render_composite`, before the label pass**. Rationale:

- The marker is an annotation, not scene content. Drawing it post-composite means it is a
  constant pure red unaffected by exposure/tone-mapping — exactly like the yellow row
  labels. If it were drawn pre-composite it would shift color/brightness as the user tunes
  exposure, defeating its purpose as a fixed reference.
- It is conceptually a ground-plane feature, but "on top of roads" is automatic: everything
  post-composite draws over the resolved frame.
- Label pass last so label text is never occluded by the marker line.

Mechanics: unlike the label pass (which needs PIL `ImageDraw` for text), lines can use the
backend directly. `PillowBackend.draw_line` draws into `self._image`
(pillow_backend.py:37/60), and `render_composite` finishes with
`backend.paste_image(...)` into that same image (compositor.py:219), so a post-composite
`backend.draw_line` lands on top of the final frame on both backends. Projection reuses
`camera/projection.py` exactly as `rendering/road_pass.py` does: `frustum_cull_segments`
(near-plane clip + far cull) then `project_points_full` (use the `in_front_mask`, not the
on-screen mask — backends clip line primitives themselves).

Style: color `(255, 60, 60)` (pure-ish red; nothing in the night palette is red, so it
reads unambiguously as a marker), fixed width 2 px. Width is deliberately **not**
perspective-scaled (unlike road_pass's `_compute_line_width`): the marker's screen-space
*length* is the scale information; a constant 2 px stroke keeps it findable when the row
start is far away without pretending to be scene geometry.

### Minimap: no

`devmode/minimap.py` maps the full scene extent (10 km x 1.8 km) into a small screen rect
with a uniform fit scale. 4.7 m at that scale is well under one pixel — the marker would
be invisible or, if force-drawn at 1 px, would lie about its size. Skip it.

---

## 2. Code change plan

Target audience: Claude Opus code-writing agent with repo access and no other context.
Repo: `/Users/davidparker/Documents/Personal/Projects/Art/StreetsAtNight`. Run tests with
`uv run pytest tests/`.

### 2.1 New file: `src/streets_at_night/devmode/scale_marker_pass.py`

Mirror the structure and docstring style of
`src/streets_at_night/devmode/label_pass.py` (module docstring, factory returning a
closure, small private helpers, Google-style docstrings with types).

Note one deliberate deviation from `label_pass.py`: `make_label_pass(row_labels, camera)`
takes a `camera` argument that its inner closure never uses (the closure uses the `cam`
call parameter). Do **not** replicate that dead parameter — the new factory takes only
`row_labels`.

```python
"""Render pass that draws a car-length scale reference line at each row start.

A single red line of length CAR_LENGTH_M is drawn transverse to each synthetic
row (along Y, centered on the row centerline) near the row's beginning, so the
scale of lights, pools, roads, and houses can be judged against a familiar
object. Devmode-only: registered exclusively by devmode render helpers.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import frustum_cull_segments, project_points_full
from ..rendering.backend import RenderBackend
from ..rendering.render_pipeline import RenderPass

if t.TYPE_CHECKING:
    from ..scene.scene import Scene

# Length of an average passenger car in meters (Corolla 4.63, Civic 4.67,
# Camry 4.88 — 4.7 sits mid-band).
CAR_LENGTH_M: float = 4.7

# X inset from the row start so the marker clears the row label anchor at x=0.
_MARKER_X_M: float = 30.0

# Pure red reads as annotation — nothing in the night palette is red.
_MARKER_COLOR: t.Tuple[int, int, int] = (255, 60, 60)

# Fixed stroke width; the marker's projected LENGTH carries the scale info.
_MARKER_WIDTH_PX: int = 2


def make_scale_marker_pass(
    row_labels: t.List[t.Tuple[str, float]],
) -> RenderPass:
    """Create a render pass drawing a car-length reference line per row.

    Each row gets one transverse line of length CAR_LENGTH_M centered on the
    row centerline at x=_MARKER_X_M, z=0. Lines are near-plane clipped,
    projected, and drawn directly via backend.draw_line, so this pass must run
    after render_composite (it draws onto the resolved sRGB frame).

    Args:
        row_labels (list[tuple[str, float]]): (label_text, y_offset) pairs;
            only the y offsets are used.

    Returns:
        RenderPass: Callable conforming to (Scene, Camera, RenderBackend) -> Any.
    """
    def _scale_marker_pass(scene: "Scene", cam: Camera, backend: RenderBackend) -> None:
        """Draw the reference line for every row visible to the camera.

        Args:
            scene (Scene): The scene being rendered (unused; RenderPass protocol).
            cam (Camera): Camera providing view/projection matrices.
            backend (RenderBackend): Target render surface.
        """
        pairs = _build_marker_pairs(row_labels)
        view_matrix = cam.get_view_matrix()
        proj_matrix = cam.get_projection_matrix()
        visible = frustum_cull_segments(pairs, view_matrix, cam.near, cam.far)
        if not visible:
            return
        _draw_marker_edges(visible, view_matrix, proj_matrix, backend)

    return _scale_marker_pass


def _build_marker_pairs(
    row_labels: t.List[t.Tuple[str, float]],
) -> t.List[t.Tuple[np.ndarray, np.ndarray]]:
    """Build world-space endpoint pairs for each row's reference line.

    Args:
        row_labels (list[tuple[str, float]]): Row label definitions.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: One (p1, p2) pair per row; each
            line runs along Y, length CAR_LENGTH_M, centered on the row
            centerline at x=_MARKER_X_M, z=0.
    """
    half = CAR_LENGTH_M / 2.0
    return [
        (
            np.array([_MARKER_X_M, y - half, 0.0]),
            np.array([_MARKER_X_M, y + half, 0.0]),
        )
        for _, y in row_labels
    ]


def _draw_marker_edges(
    pairs: t.List[t.Tuple[np.ndarray, np.ndarray]],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    backend: RenderBackend,
) -> None:
    """Project near-clipped marker edges and draw the in-front ones.

    Uses the in-front mask (not the on-screen mask) — both backends clip line
    primitives to the surface, matching road_pass behavior.

    Args:
        pairs (list[tuple[np.ndarray, np.ndarray]]): Near-clipped world edges.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        backend (RenderBackend): Target render surface.
    """
    screen_size = backend.get_surface_size()
    points = np.array([point for pair in pairs for point in pair])
    screen_coords, in_front_mask, _ = project_points_full(
        points, view_matrix, proj_matrix, screen_size
    )
    for i in range(len(pairs)):
        idx1, idx2 = 2 * i, 2 * i + 1
        if not (in_front_mask[idx1] and in_front_mask[idx2]):
            continue
        sx1, sy1 = screen_coords[idx1]
        sx2, sy2 = screen_coords[idx2]
        backend.draw_line(
            (int(sx1), int(sy1)), (int(sx2), int(sy2)),
            _MARKER_COLOR, _MARKER_WIDTH_PX,
        )
```

Notes for the implementer:

- `frustum_cull_segments` and `project_points_full` are in
  `src/streets_at_night/camera/projection.py` (lines 156 and 50). Their contracts are
  exactly as used by `rendering/road_pass.py:_draw_segment` (road_pass.py:197-248).
- `Camera.near` / `Camera.far` are attributes set in `Camera.__init__`
  (`camera/camera.py:36-53`).
- 10 edges per frame — no need for road_pass's `_is_off_same_side` cheap reject or batching
  heroics beyond the single `project_points_full` call shown.
- Naming: repo Functions.md limits prefixes to get_/set_/is_/build_/etc., but the direct
  sibling `label_pass.py` established `make_label_pass`. Use `make_scale_marker_pass` for
  symmetry with the module it mirrors (flagged as an accepted deviation; see 2.5).

### 2.2 Edit: `src/streets_at_night/devmode/render_helpers.py`

`build_render_passes` (lines 120-161). Two changes:

1. Add the import next to the existing label pass import (line 28):

```python
from .label_pass import make_label_pass
from .scale_marker_pass import make_scale_marker_pass
```

2. Replace the trailing registration block (lines 159-161):

```python
    if use_labels:
        passes.append(make_label_pass(ROW_LABELS, scene.camera))
    return passes
```

with:

```python
    if use_labels:
        passes.append(make_scale_marker_pass(ROW_LABELS))
        passes.append(make_label_pass(ROW_LABELS, scene.camera))
    return passes
```

3. Update the function docstring (lines 125-139): the sentence "labels come last so they
   overlay the final composited frame" becomes something like "overlay passes come last so
   they draw onto the final composited frame: the scale-marker pass first, then labels so
   text is never occluded by the marker line." Also update the `use_labels` arg description:
   "When True, append the synthetic-row scale-marker and label passes."

No other registration points exist. `build_render_passes` is called only by
`render_devmode_scene` (render_helpers.py:189) and the Devmode tab flows through that
(`ui/tabs/devmode_tab.py:788`, which passes `use_labels=not self._use_real_tiles`). The
export pass list in `rendering/render_pipeline.get_export_passes` is untouched — real
renders cannot pick this up.

### 2.3 New test file: `tests/test_scale_marker_pass.py`

One test file per source module. Naming `test_method_scenario_behavior`, Arrange/Act/Assert
with whitespace, all imports at top, double quotes, `# Radical` first line (repo test
convention — every test file starts with it). Mock only the backend (architectural
boundary); use a real `Camera` — the projection math is the behavior under test.

```python
# Radical
"""Tests for devmode.scale_marker_pass: marker geometry and draw behavior."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import Camera
from streets_at_night.devmode.scale_marker_pass import (
    CAR_LENGTH_M,
    _build_marker_pairs,
    make_scale_marker_pass,
)
from streets_at_night.devmode.synthetic_scene import ROW_LABELS
from streets_at_night.rendering.backend import RenderBackend


@pytest.fixture
def devmode_camera() -> Camera:
    """Camera matching the default devmode pose: looks down +X along the rows."""
    return Camera(
        position=np.array([-500.0, 0.0, 800.0]),
        target=np.array([5000.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=896.0 / 720.0,
        near=100.0,
        far=60000.0,
    )


@pytest.fixture
def mock_backend() -> mock.MagicMock:
    """Backend mock with a real surface size for projection."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (896, 720)
    return backend


def test_build_marker_pairs_one_line_per_row_with_car_length():
    # Act
    pairs = _build_marker_pairs(ROW_LABELS)

    # Assert — one marker per row, exactly CAR_LENGTH_M long, on the ground plane.
    assert len(pairs) == len(ROW_LABELS)
    for p1, p2 in pairs:
        assert np.linalg.norm(p2 - p1) == pytest.approx(CAR_LENGTH_M)
        assert p1[2] == 0.0 and p2[2] == 0.0


def test_build_marker_pairs_centered_on_row_centerline():
    # Act
    pairs = _build_marker_pairs(ROW_LABELS)

    # Assert — each line's Y midpoint is the row's y offset.
    for (p1, p2), (_, y_offset) in zip(pairs, ROW_LABELS):
        assert (p1[1] + p2[1]) / 2.0 == pytest.approx(y_offset)


def test_scale_marker_pass_draws_lines_when_rows_in_front(devmode_camera, mock_backend):
    # Arrange
    render_pass = make_scale_marker_pass(ROW_LABELS)

    # Act
    render_pass(mock.MagicMock(), devmode_camera, mock_backend)

    # Assert — markers in front of the camera are drawn in the marker style.
    assert mock_backend.draw_line.call_count > 0
    for call in mock_backend.draw_line.call_args_list:
        assert call.args[2] == (255, 60, 60)


def test_scale_marker_pass_skips_rows_behind_camera(mock_backend):
    # Arrange — camera at the far end looking away from all markers (x=30).
    camera = Camera(
        position=np.array([20000.0, 0.0, 800.0]),
        target=np.array([30000.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=896.0 / 720.0,
        near=100.0,
        far=60000.0,
    )
    render_pass = make_scale_marker_pass(ROW_LABELS)

    # Act
    render_pass(mock.MagicMock(), camera, mock_backend)

    # Assert
    mock_backend.draw_line.assert_not_called()
```

Implementer must verify the `Camera` constructor call against
`src/streets_at_night/camera/camera.py:36-53` (signature confirmed:
`position, target, fov_deg, aspect_ratio, near, far`) and check how existing tests
construct cameras (e.g. `tests/test_road_pass.py`) — reuse a conftest fixture if one
already exists rather than duplicating.

The color assertion pins the module constant's value; if that feels too implementation-y,
assert `call.args[2] == scale_marker_pass._MARKER_COLOR` instead. Recommendation: import
the private constant and compare against it — the contract is "drawn in the marker style",
not "is this exact RGB".

### 2.4 Update existing test: `tests/test_render_helpers.py`

`test_build_render_passes_appends_label_pass_when_requested` (lines 88-98) asserts
`use_labels=True` adds exactly one trailing pass and `passes[-2] is render_composite`.
The change adds two. Update to:

```python
def test_build_render_passes_appends_overlay_passes_when_requested():
    # Arrange
    scene = mock.MagicMock()

    # Act
    without = build_render_passes(scene, gbuffer=None, use_labels=False)
    with_overlays = build_render_passes(scene, gbuffer=None, use_labels=True)

    # Assert — overlays add exactly two trailing passes after composite.
    assert len(with_overlays) == len(without) + 2
    assert with_overlays[-3] is render_composite
```

No other test in that file touches the trailing passes
(`test_build_render_passes_includes_composite_last_without_labels` uses
`use_labels=False` and stays valid).

### 2.5 Open design decisions (with recommendations)

1. **Marker orientation — transverse (along Y). Recommended as specced.** The default
   devmode camera looks almost exactly down +X, so an X-oriented ("parked car") line would
   foreshorten to near-zero screen length. If the user objects that a car should lie along
   the road, the fallback is drawing an L (one X leg + one Y leg); not recommended — more
   visual noise for no added information.
2. **Gate flag — reuse `use_labels`. Recommended.** The marker has the same lifecycle as
   row labels (synthetic scene only, off in real-tiles mode). Renaming the parameter to
   `use_overlays` would be more accurate but churns `render_devmode_scene`,
   `devmode_tab.py:788`, and four tests for zero behavior change. Keep `use_labels`,
   document that it now gates both overlay passes.
3. **Factory name — `make_scale_marker_pass`. Recommended despite Functions.md.** The repo's
   prefix whitelist has no `make_`, but the sibling `make_label_pass` set the local
   convention and the two modules should read as a pair. If the user prefers strict rule
   compliance, `build_scale_marker_pass` is the alternative (do not rename
   `make_label_pass` in this change).
4. **`_MARKER_X_M = 30.0`.** Any small positive inset works; 30 m clears the label anchor
   at x=0 and the first streetlight cluster without pushing the marker far from "the
   beginning". Trivially tunable module constant.
5. **Rows covered — all 10, including Houses. Recommended.** Judging house scale is a
   stated goal; `ROW_LABELS` already provides all anchors.
6. **Pre-existing wart, do not fix here:** `make_label_pass(row_labels, camera)` in
   `devmode/label_pass.py:17-20` takes a `camera` parameter its closure never uses (it uses
   the per-call `cam`). The new module intentionally omits it; leave `label_pass.py` alone
   in this change.

### 2.6 Verification

- `uv run pytest tests/` — full suite; report pass/fail counts.
- Manual: `uv run streets-at-night devmode` — confirm a short red tick crosses each row
  near its left end, labels still render on top, and toggling to real tiles removes it.
- Confirm `render` output is unchanged: `rg -n "scale_marker" src/streets_at_night/rendering/`
  must return nothing.
