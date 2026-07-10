# Devmode Color Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reflow the devmode synthetic scene rows, add live color-preview swatches next to every hex-valued text entry in the devmode panel, and open pygame_gui's colour picker dialog from a swatch click, applying picks through the existing config-apply path.

**Architecture:** Swatches are a post-build pass: after each section's widgets are built and populated (`read_config_to_widgets`), a new `devmode/color_swatch.py` module walks the section's `SliderRegistry`, shrinks each hex-valued entry, and drops a `UIImage` swatch in the freed slot — `dynamic_panel.py` and `panel_sections.py` layout code is untouched. The Devmode tab routes `MOUSEBUTTONDOWN` hits on swatches to a single `UIColourPickerDialog`, and the picked colour is written into the entry and applied via the tab's existing `_apply_active_section()` — no second write path.

**Tech Stack:** Python 3.11, uv, pytest, pygame-ce 2.5, pygame_gui 0.6.14 (`pygame_gui.windows.UIColourPickerDialog`, event `pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED` with `event.colour: pygame.Color` and `event.ui_element`).

**Spec:** `docs/superpowers/specs/2026-07-09-devmode-color-tools-design.md`

## Global Constraints

- Package manager is `uv`: run tests with `uv run pytest tests/...`.
- Test conventions (user rules UnitTests.md): imports at top, double quotes, Arrange/Act/Assert with blank lines, naming `test_methodundertest_scenario_behavior`, fixtures at top of file or conftest, parametrize only for pure input→output tables.
- Headless pygame-gui tests use the existing `headless_pygame` / `headless_manager` fixtures in `tests/conftest.py` — follow the patterns in `tests/test_dynamic_panel.py`.
- Hex detection pattern: `^#[0-9a-fA-F]{6}$` (case-insensitive input accepted; written values use the same `#%02X%02X%02X` uppercase format as `config_bridge._tuple_to_hex`).
- One colour dialog at a time; Cancel/close applies nothing; dialog is destroyed on section switch and tab destroy.
- The repo working tree has unrelated uncommitted user changes (deleted TOML files) — never stage them; `git add` explicit paths only.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Synthetic scene row reflow + Houses row templates

**Files:**
- Modify: `src/streets_at_night/devmode/synthetic_scene.py` (`ROW_LABELS` ~line 36, `_BUILDING_TEMPLATES` ~line 224)
- Test: `tests/test_devmode_synthetic_scene.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ROW_LABELS` (same 10 labels/order, new y-offsets), `_BUILDING_TEMPLATES` (6 entries). No signature changes; later tasks don't depend on this one.

- [ ] **Step 1: Update ROW_LABELS to a uniform 200m pitch**

```python
# Row layout: (label, y_offset_in_enu_meters)
ROW_LABELS: t.List[t.Tuple[str, float]] = [
    ("Highway",         900.0),
    ("Major Road",      700.0),
    ("Collector",       500.0),
    ("Residential",     300.0),
    ("Service",         100.0),
    ("Cars",           -100.0),
    ("Sign Lights",    -300.0),
    ("Parking Lights", -500.0),
    ("Storefront",     -700.0),
    ("Houses",         -900.0),
]
```

- [ ] **Step 2: Update _BUILDING_TEMPLATES and its comment**

```python
# Alternating (footprint_width_m, footprint_depth_m, height_m, kind) templates
# for the Houses row, chosen to exercise the window system's behaviors side by
# side: a 3.0m house below the single-row window fallback threshold (renders
# dark), 4.5m houses that get exactly one fallback row, a retail block on the
# global (sparse) lit fraction for contrast with the residential fraction, and
# apartment towers with multiple window rows. Cycled along the row by index.
_BUILDING_TEMPLATES: t.List[t.Tuple[float, float, float, str]] = [
    (12.0, 10.0, 4.5, "house"),
    (24.0, 16.0, 14.0, "apartments"),
    (10.0, 8.0, 3.0, "house"),
    (16.0, 12.0, 6.5, "retail"),
    (14.0, 11.0, 4.5, "house"),
    (28.0, 18.0, 22.0, "apartments"),
]
```

- [ ] **Step 3: Update tests**

In `tests/test_devmode_synthetic_scene.py`: most row tests read `ROW_LABELS` directly and adapt automatically (e.g. the y-offset test at ~line 472). Find and update any test with hardcoded y values or hardcoded template counts/heights (`rg -n "1100|-300.0|-500.0|-700.0|-900.0|_BUILDING_TEMPLATES|4.5|22.0" tests/test_devmode_synthetic_scene.py`). If a test asserts the template cycle (kinds or heights per building index), update the expected sequence to the 6-template list above. Add one test pinning the new invariants:

```python
def test_row_labels_uniform_200m_pitch() -> None:
    # Act
    offsets = [y for _, y in ROW_LABELS]

    # Assert — 10 rows from 900 down to -900 at a uniform 200m pitch
    assert offsets == [900.0, 700.0, 500.0, 300.0, 100.0, -100.0, -300.0, -500.0, -700.0, -900.0]
```

- [ ] **Step 4: Run the suite for this area**

Run: `uv run pytest tests/test_devmode_synthetic_scene.py tests/test_ui_devmode_tab.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/devmode/synthetic_scene.py tests/test_devmode_synthetic_scene.py
git commit -m "Devmode: reflow synthetic rows to uniform pitch, widen Houses row templates

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: color_swatch module + registry swatch tracking

**Files:**
- Create: `src/streets_at_night/devmode/color_swatch.py`
- Modify: `src/streets_at_night/devmode/slider_registry.py` (add `swatches` dict, include in `all_widgets()`)
- Test: `tests/test_devmode_color_swatch.py` (new; one test file per source module)

**Interfaces:**
- Consumes: `SliderRegistry` (`text_entries: Dict[entry, config_path]`, `color_pools: List[ColorPoolBinding]` where `binding.rows` is `List[Tuple[hex_entry, weight_entry]]`, `extra_widgets`, `all_widgets()`).
- Produces (Task 3 relies on these exact names):
  - `SliderRegistry.swatches: t.Dict[t.Any, t.Any]` — maps hex entry → swatch `UIImage`; included in `all_widgets()`.
  - `is_hex_color(text: str) -> bool`
  - `add_swatches(manager: pygame_gui.UIManager, registry: SliderRegistry) -> None`
  - `refresh_swatch(registry: SliderRegistry, entry: t.Any) -> None`
  - `find_swatch_entry_at(registry: SliderRegistry, pos: t.Tuple[int, int], panel_rect: pygame.Rect) -> t.Optional[t.Any]`
  - `color_to_hex(colour: pygame.Color) -> str`
  - Module constants `SWATCH_SIZE = 22`, `SWATCH_GAP = 4`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_devmode_color_swatch.py`. Follow `tests/test_dynamic_panel.py` for the headless fixtures (`headless_manager` from conftest; build a `UIScrollingContainer` the same way its `scroll` fixture does — copy that fixture into this file if it is file-local rather than in conftest; if it lives in conftest, use it directly). Tests:

```python
import pygame
import pygame_gui
import pytest

from streets_at_night.devmode.color_swatch import (
    SWATCH_GAP,
    SWATCH_SIZE,
    add_swatches,
    color_to_hex,
    find_swatch_entry_at,
    is_hex_color,
    refresh_swatch,
)
from streets_at_night.devmode.slider_registry import SliderRegistry


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#FFAA00", True),
        ("#ffaa00", True),
        ("#0D0A08", True),
        ("FFAA00", False),
        ("#FFAA0", False),
        ("#FFAA000", False),
        ("#GGAA00", False),
        ("", False),
        ("not a color", False),
    ],
)
def test_is_hex_color_pattern_table(text, expected):
    assert is_hex_color(text) == expected


def test_color_to_hex_formats_uppercase_rgb_dropping_alpha():
    assert color_to_hex(pygame.Color(255, 170, 0, 128)) == "#FFAA00"


def _make_entry(headless_manager, container, text):
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(100, 10, 120, 30),
        manager=headless_manager,
        container=container,
    )
    entry.set_text(text)
    return entry


def test_add_swatches_creates_swatch_for_hex_text_entry(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))

    # Act
    add_swatches(headless_manager, registry)

    # Assert — swatch exists, entry shrank and shifted right to make room
    assert entry in registry.swatches
    swatch = registry.swatches[entry]
    assert swatch.get_relative_rect().width == SWATCH_SIZE
    assert entry.get_relative_rect().x == 100 + SWATCH_SIZE + SWATCH_GAP
    assert entry.get_relative_rect().width == 120 - SWATCH_SIZE - SWATCH_GAP


def test_add_swatches_skips_non_hex_text_entry(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "some string")
    registry.register_text(entry, ("region_name",))

    # Act
    add_swatches(headless_manager, registry)

    # Assert
    assert registry.swatches == {}


def test_add_swatches_covers_color_pool_hex_entries(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    hex_entry = _make_entry(headless_manager, scroll.scrollable_container, "#0D0A08")
    weight_entry = _make_entry(headless_manager, scroll.scrollable_container, "1.000")
    registry.register_color_pool(("colors", "colors"), [(hex_entry, weight_entry)])

    # Act
    add_swatches(headless_manager, registry)

    # Assert — hex entry got a swatch, weight entry did not
    assert hex_entry in registry.swatches
    assert weight_entry not in registry.swatches


def test_refresh_swatch_updates_image_on_valid_hex(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FF0000")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    entry.set_text("#00FF00")
    refresh_swatch(registry, entry)

    # Assert — swatch surface repainted to the new colour
    swatch = registry.swatches[entry]
    assert swatch.image.get_at((SWATCH_SIZE // 2, SWATCH_SIZE // 2))[:3] == (0, 255, 0)


def test_refresh_swatch_keeps_last_color_on_invalid_hex(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FF0000")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    entry.set_text("#nope")
    refresh_swatch(registry, entry)

    # Assert
    swatch = registry.swatches[entry]
    assert swatch.image.get_at((SWATCH_SIZE // 2, SWATCH_SIZE // 2))[:3] == (255, 0, 0)


def test_find_swatch_entry_at_hit_and_miss(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)
    swatch = registry.swatches[entry]
    panel_rect = scroll.get_abs_rect()
    inside = swatch.get_abs_rect().center

    # Act / Assert
    assert find_swatch_entry_at(registry, inside, panel_rect) is entry
    assert find_swatch_entry_at(registry, (panel_rect.right + 50, panel_rect.bottom + 50), panel_rect) is None


def test_find_swatch_entry_at_ignores_swatch_outside_panel_rect(headless_manager, scroll):
    # Arrange — a panel rect that excludes the swatch simulates it being scrolled out of view
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)
    swatch = registry.swatches[entry]
    point = swatch.get_abs_rect().center
    off_panel = pygame.Rect(-500, -500, 10, 10)

    # Act / Assert
    assert find_swatch_entry_at(registry, point, off_panel) is None


def test_all_widgets_includes_swatches(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    widgets = registry.all_widgets()

    # Assert
    assert registry.swatches[entry] in widgets
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_devmode_color_swatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'streets_at_night.devmode.color_swatch'`.

- [ ] **Step 3: Implement**

`src/streets_at_night/devmode/slider_registry.py` — add to `__init__`:

```python
        self.swatches: t.Dict[t.Any, t.Any] = {}
```

with an Attributes docstring line (`swatches: Maps hex-valued UITextEntryLine → its UIImage colour swatch.`), and in `all_widgets()` add `widgets.extend(self.swatches.values())`.

`src/streets_at_night/devmode/color_swatch.py`:

```python
"""Colour-preview swatches for hex-valued panel entries.

A post-build pass over a section's SliderRegistry: every text entry whose
current text is a #RRGGBB colour gets a small UIImage swatch to its left
(the entry is shrunk to make room). Swatches live in
``SliderRegistry.swatches`` so section teardown kills them with everything
else. Click routing and the colour-picker dialog live in the Devmode tab;
this module only answers "which entry's swatch is at this screen position".
"""

import re
import typing as t

import pygame
import pygame_gui

from .slider_registry import SliderRegistry

SWATCH_SIZE = 22
SWATCH_GAP = 4

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_hex_color(text: str) -> bool:
    """Check whether text is a #RRGGBB hex colour string."""
    return bool(_HEX_COLOR_RE.match(text))


def color_to_hex(colour: pygame.Color) -> str:
    """Format a pygame Color as an uppercase #RRGGBB string (alpha dropped).

    Args:
        colour (pygame.Color): The colour to format.

    Returns:
        str: "#RRGGBB", matching config_bridge's hex format.
    """
    return f"#{colour.r:02X}{colour.g:02X}{colour.b:02X}"


def add_swatches(manager: pygame_gui.UIManager, registry: SliderRegistry) -> None:
    """Create a colour swatch beside every hex-valued entry in the registry.

    Must run AFTER the section's entries have their text populated (i.e.
    after read_config_to_widgets), because hex detection reads entry text.
    Non-hex entries are left untouched.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        registry (SliderRegistry): The section's registry; swatches are
            recorded in registry.swatches.
    """
    for entry in _iter_hex_candidates(registry):
        if entry in registry.swatches or not is_hex_color(entry.get_text()):
            continue
        registry.swatches[entry] = _create_swatch(manager, entry)


def refresh_swatch(registry: SliderRegistry, entry: t.Any) -> None:
    """Repaint an entry's swatch from its current text, if valid hex.

    Invalid hex leaves the swatch on its last valid colour. Entries without
    a swatch are ignored.

    Args:
        registry (SliderRegistry): The section's registry.
        entry: The UITextEntryLine whose swatch should refresh.
    """
    swatch = registry.swatches.get(entry)
    if swatch is None or not is_hex_color(entry.get_text()):
        return
    swatch.set_image(_fill_surface(entry.get_text()))


def find_swatch_entry_at(
    registry: SliderRegistry,
    pos: t.Tuple[int, int],
    panel_rect: pygame.Rect,
) -> t.Optional[t.Any]:
    """Return the entry whose swatch is under a screen position, if any.

    Only the part of a swatch inside ``panel_rect`` counts, so swatches
    scrolled outside the visible panel area don't take clicks.

    Args:
        registry (SliderRegistry): The section's registry.
        pos (tuple[int, int]): Screen position of the click.
        panel_rect (pygame.Rect): Absolute rect of the visible panel area.

    Returns:
        The hex entry owning the clicked swatch, or None.
    """
    for entry, swatch in registry.swatches.items():
        visible_part = swatch.get_abs_rect().clip(panel_rect)
        if visible_part.collidepoint(pos):
            return entry
    return None


def _iter_hex_candidates(registry: SliderRegistry) -> t.Iterator[t.Any]:
    """Yield every entry that may hold a hex colour: text entries + pool hex entries."""
    yield from registry.text_entries.keys()
    for binding in registry.color_pools:
        for hex_entry, _weight_entry in binding.rows:
            yield hex_entry


def _create_swatch(manager: pygame_gui.UIManager, entry: t.Any) -> pygame_gui.elements.UIImage:
    """Shrink an entry and place a colour-filled UIImage in the freed slot.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        entry: The UITextEntryLine to attach a swatch to.

    Returns:
        UIImage: The created swatch, vertically centered on the entry row.
    """
    rect = entry.get_relative_rect()
    swatch_y = rect.y + (rect.height - SWATCH_SIZE) // 2
    swatch_rect = pygame.Rect(rect.x, swatch_y, SWATCH_SIZE, SWATCH_SIZE)

    entry.set_relative_position((rect.x + SWATCH_SIZE + SWATCH_GAP, rect.y))
    entry.set_dimensions((rect.width - SWATCH_SIZE - SWATCH_GAP, rect.height))

    return pygame_gui.elements.UIImage(
        relative_rect=swatch_rect,
        image_surface=_fill_surface(entry.get_text()),
        manager=manager,
        container=entry.ui_container,
    )


def _fill_surface(hex_text: str) -> pygame.Surface:
    """Build a SWATCH_SIZE square surface filled with the given hex colour."""
    surface = pygame.Surface((SWATCH_SIZE, SWATCH_SIZE))
    surface.fill(pygame.Color(hex_text))
    return surface
```

Note: if `UIImage.set_image` does not exist in pygame_gui 0.6.14 under that name, check `dir(pygame_gui.elements.UIImage)` — 0.6.x exposes `set_image(new_image)`. If the swatch image assertion reads stale pixels, read the surface via `swatch.image` as the tests do.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_devmode_color_swatch.py tests/test_dynamic_panel.py -v`
Expected: all PASS (test_dynamic_panel confirms no regression from the registry change).

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/devmode/color_swatch.py src/streets_at_night/devmode/slider_registry.py tests/test_devmode_color_swatch.py
git commit -m "Devmode: colour swatch module + registry swatch tracking

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Devmode tab wiring — swatch creation, click → dialog, pick → apply

**Files:**
- Modify: `src/streets_at_night/ui/tabs/devmode_tab.py` (`_build_section` ~line 261-278, `handle_event` ~line 323, `_handle_entry_finished` ~line 371, `destroy` ~line 218-240)
- Modify: `docs/commands/devmode.md` (mention swatches + picker in the panel description)
- Test: `tests/test_ui_devmode_tab.py`

**Interfaces:**
- Consumes (from Task 2, exact names): `add_swatches(manager, registry)`, `refresh_swatch(registry, entry)`, `find_swatch_entry_at(registry, pos, panel_rect)`, `color_to_hex(colour)`, `is_hex_color(text)`.
- Consumes (existing): `self._scroll` (UIScrollingContainer; its `get_abs_rect()` is the visible panel area), `self._active_registry()`, `self._apply_active_section()`, `pygame_gui.windows.UIColourPickerDialog(rect, manager, initial_colour=pygame.Color(...), window_title=...)`, events `pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED` (attrs `.colour`, `.ui_element`) and `pygame_gui.UI_WINDOW_CLOSE`.
- Produces: tab attributes `self._color_dialog: t.Optional[UIColourPickerDialog]` and `self._color_dialog_entry: t.Optional[t.Any]` (both None when no dialog is open).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ui_devmode_tab.py`, following its existing build-a-tab fixture pattern (it already builds the tab headless and calls `tab._build_section(...)`). Exact fixture names/setup: mirror the file's existing tests.

```python
def test_build_section_creates_swatches_for_hex_entries(built_tab):
    # Arrange — the Atmosphere section registers haze_color as a text entry
    built_tab._build_section("Atmosphere")

    # Act
    registry = built_tab._active_registry()

    # Assert — at least the haze_color entry received a swatch
    assert len(registry.swatches) >= 1


def test_swatch_click_opens_colour_picker_dialog(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center

    # Act
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos})
    built_tab.handle_event(event)

    # Assert
    assert built_tab._color_dialog is not None
    assert built_tab._color_dialog_entry is entry


def test_second_swatch_click_does_not_stack_dialogs(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    _entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos})
    built_tab.handle_event(event)
    first_dialog = built_tab._color_dialog

    # Act
    built_tab.handle_event(event)

    # Assert — same dialog instance, not a second one
    assert built_tab._color_dialog is first_dialog


def test_colour_picked_sets_entry_applies_and_refreshes(built_tab):
    # Arrange — open the dialog for the haze_color entry
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    built_tab.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos}))

    # Act
    picked = pygame.event.Event(
        pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED,
        {"colour": pygame.Color(18, 52, 86), "ui_element": built_tab._color_dialog},
    )
    built_tab.handle_event(picked)

    # Assert — entry text set, config applied through the normal path, dialog refs cleared
    assert entry.get_text() == "#123456"
    assert built_tab.state.config.atmosphere.haze_color == (18, 52, 86)
    assert built_tab._color_dialog is None
    assert built_tab._color_dialog_entry is None


def test_section_switch_closes_open_colour_dialog(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    _entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    built_tab.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos}))

    # Act
    built_tab._build_section("Devmode Camera")

    # Assert
    assert built_tab._color_dialog is None
    assert built_tab._color_dialog_entry is None
```

Adjust the `built_tab` fixture name and the section label strings ("Atmosphere", "Devmode Camera") to whatever the file actually uses — read the file first; the section labels come from the tab's section list and existing tests reference them.

Also note: `test_colour_picked_sets_entry_applies_and_refreshes` asserts `haze_color == (18, 52, 86)` because config stores it as an RGB tuple (`serde: hex_color`); if the apply path stores hex strings for this field instead, assert accordingly after reading `config_bridge._hex_to_tuple` usage.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_devmode_tab.py -k "swatch or colour" -v`
Expected: FAIL — `registry.swatches` empty (add_swatches never called) and `AttributeError` on `_color_dialog`.

- [ ] **Step 3: Implement in devmode_tab.py**

1. Imports:

```python
from ...devmode.color_swatch import add_swatches, color_to_hex, find_swatch_entry_at, is_hex_color, refresh_swatch
from pygame_gui.windows import UIColourPickerDialog
```

2. In `__init__` (near `self._minimap`):

```python
        self._color_dialog: t.Optional[UIColourPickerDialog] = None
        self._color_dialog_entry: t.Optional[t.Any] = None
```

3. In `_build_section`, first close any open dialog, and after each of the three `read_config_to_widgets(...)` calls add the swatch pass:

```python
        self._close_color_dialog()
        ...
            height = build_atmosphere_section(self._manager, container, 0, self._static_registry, self.state.config)
            read_config_to_widgets(self.state.config, self._static_registry)
            add_swatches(self._manager, self._static_registry)
```

(same `add_swatches(self._manager, self._static_registry)` after the camera section's read, and `add_swatches(self._manager, self._light_registry)` after the dynamic section's read).

4. New helpers on the tab:

```python
    def _close_color_dialog(self) -> None:
        """Kill any open colour-picker dialog and clear its target entry."""
        if self._color_dialog is not None:
            self._color_dialog.kill()
        self._color_dialog = None
        self._color_dialog_entry = None

    def _open_color_dialog(self, entry: t.Any) -> None:
        """Open the colour picker seeded from a hex entry's current colour."""
        if self._color_dialog is not None:
            return
        initial = pygame.Color(entry.get_text()) if is_hex_color(entry.get_text()) else pygame.Color(255, 255, 255)
        self._color_dialog = UIColourPickerDialog(
            pygame.Rect(self._panel_rect.x - 420, self._panel_rect.y + 40, 400, 400),
            self._manager,
            initial_colour=initial,
            window_title="Pick colour",
        )
        self._color_dialog_entry = entry

    def _handle_colour_picked(self, colour: pygame.Color) -> None:
        """Write a picked colour into the target entry and apply it."""
        entry = self._color_dialog_entry
        if entry is not None:
            entry.set_text(color_to_hex(colour))
            refresh_swatch(self._active_registry(), entry)
            self._apply_active_section()
        self._color_dialog = None
        self._color_dialog_entry = None

    def _handle_swatch_click(self, pos: t.Tuple[int, int]) -> None:
        """Open the picker when a visible swatch is clicked."""
        if self._scroll is None:
            return
        entry = find_swatch_entry_at(self._active_registry(), pos, self._scroll.get_abs_rect())
        if entry is not None:
            self._open_color_dialog(entry)
```

If the picker dialog rect above lands off-window for the devmode layout (panel is on the right; `self._panel_rect.x - 420` places the dialog just left of it), clamp x to `max(10, ...)`.

5. In `handle_event`, add branches (order alongside the existing elif chain):

```python
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_swatch_click(event.pos)
        elif event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED and event.ui_element is self._color_dialog:
            self._handle_colour_picked(event.colour)
        elif event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element is self._color_dialog:
            self._color_dialog = None
            self._color_dialog_entry = None
```

Caution: `handle_event` currently starts with `self._frame_events.append(event)` and the camera passthrough consumes mouse events for the render viewport — the swatch hit test only fires for positions inside the panel (find_swatch_entry_at clips to the panel rect), so no conflict with camera drag or minimap; do not consume/stop the event, just act on it.

6. In `_handle_entry_finished`, refresh the swatch after the apply for any entry that has one:

```python
        self._apply_active_section()
        refresh_swatch(registry, entry)
```

7. In `destroy()`, call `self._close_color_dialog()` alongside the registry resets. (Swatches themselves are killed by `_reset_registry` via `all_widgets()` — verify `_reset_registry` kills `all_widgets()`; it does today.)

8. `docs/commands/devmode.md`: in the panel/editing description, add one short paragraph: hex color fields show a color swatch beside the entry; clicking the swatch opens a color picker dialog, and OK applies the picked color exactly like typing it.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ui_devmode_tab.py tests/test_devmode_color_swatch.py tests/test_dynamic_panel.py -v`
Expected: all PASS.
Then the full suite: `uv run pytest tests/`
Expected: all PASS; report the count.

- [ ] **Step 5: Commit**

```bash
git add src/streets_at_night/ui/tabs/devmode_tab.py tests/test_ui_devmode_tab.py docs/commands/devmode.md
git commit -m "Devmode: colour swatches open a picker dialog; picks apply via the normal path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
