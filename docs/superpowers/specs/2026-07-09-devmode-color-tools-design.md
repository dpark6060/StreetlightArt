# Devmode color tools: row reflow, hex swatches, color picker dialog

**Date:** 2026-07-09
**Status:** Approved (design approved verbally; user delegated execution end-to-end)

## Problem

1. The devmode synthetic scene has a 400m y-gap between the Cars and Sign Lights
   rows where the removed House Lights row used to sit, and its Houses row
   predates the residential-lit-fraction work, so it doesn't showcase what the
   panel now tunes.
2. Hex color values in the devmode panel (color-pool rows, string color fields,
   atmosphere haze color) are edited blind — no preview of the color, no picker.

## Changes

### 1. Synthetic scene row reflow + Houses row content

- `ROW_LABELS` y-offsets reflow to a uniform 200m pitch:
  Highway 900, Major Road 700, Collector 500, Residential 300, Service 100,
  Cars −100, Sign Lights −300, Parking Lights −500, Storefront −700,
  Houses −900. Row order and count (10 rows, indices 0–9) unchanged.
- `_BUILDING_TEMPLATES` in `devmode/synthetic_scene.py` gains coverage of the
  new window behaviors: a 3.0m house (below the single-row window fallback
  threshold — renders dark), the existing 4.5m houses (single window row), a
  6.5m `retail` block (global 0.12 lit fraction, contrasting with the 0.4
  residential fraction beside it), and two apartment towers (multi-row
  windows). Exact template list:

  ```python
  _BUILDING_TEMPLATES = [
      (12.0, 10.0, 4.5, "house"),
      (24.0, 16.0, 14.0, "apartments"),
      (10.0, 8.0, 3.0, "house"),      # below window fallback threshold — dark
      (16.0, 12.0, 6.5, "retail"),    # global lit fraction, contrast vs houses
      (14.0, 11.0, 4.5, "house"),
      (28.0, 18.0, 22.0, "apartments"),
  ]
  ```

- Existing synthetic-scene tests asserting y-offsets update to the new values.

### 2. Color preview swatches

- New module `devmode/color_swatch.py` owns swatch behavior; `dynamic_panel`
  and (via the shared registry) the static sections consume it.
- A swatch is a small square `pygame_gui.elements.UIImage` filled with the
  entry's current color, placed immediately left of the hex text entry; the
  entry shrinks by the swatch width + gap to compensate.
- Which entries get swatches: every color-pool hex entry, and every registered
  text entry (`SliderRegistry.text_entries`) whose current value renders as
  `#RRGGBB` (this covers `haze_color` in the Atmosphere section and any future
  string color field). Detection is by value pattern
  `^#[0-9a-fA-F]{6}$`, case-insensitive.
- `SliderRegistry` gains a swatch mapping (hex entry → swatch element) so
  lifecycle (build/destroy/rebuild on section switch) rides the existing
  `all_widgets()` path.
- Swatch refresh: when a hex entry finishes editing (existing
  `UI_TEXT_ENTRY_FINISHED` flow) the swatch refills from the entry text; an
  invalid hex leaves the swatch on its last valid color (matching the existing
  "invalid falls back" behavior of pool applies).

### 3. Color picker dialog

- Clicking a swatch opens `pygame_gui.windows.UIColourPickerDialog`
  (pygame_gui 0.6.14; constructor takes `rect`, `manager`,
  `initial_colour: pygame.Color`, `window_title`), seeded from the entry's
  current color. One dialog at a time; opening tracks which hex entry it
  targets.
- Click detection: the Devmode tab already handles raw pygame events (minimap
  drag); a `MOUSEBUTTONDOWN` handler tests the active registry's swatches via
  their absolute rects, ignoring swatches scrolled outside the panel's visible
  area.
- On `pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED`: format `event.colour` as
  `#RRGGBB` (alpha dropped), `set_text` it into the target entry, update the
  swatch, and run the tab's existing `_apply_active_section()` — the same
  apply path as a typed edit (color-pool tuple rebuild / `_hex_to_tuple` for
  haze color, scene invalidation). No second write path.
- Cancel/close changes nothing. The dialog is destroyed on section switch or
  tab destroy if still open.

## Testing

- Unit tests (headless pygame fixtures from `tests/conftest.py`):
  hex-pattern detection; swatch created next to pool hex entries and
  hex-valued text entries, not next to non-color entries; swatch color updates
  on entry-finished; picked-event flow sets entry text, refreshes swatch, and
  writes through to the config object; invalid hex leaves swatch unchanged;
  dialog lifecycle (single instance, cleaned up on destroy).
- Synthetic-scene tests updated for new y-offsets and template list.

## Out of scope

- Add/remove colors in a pool (still TOML-only, per the existing v1 note).
- Custom color-wheel widget (built-in dialog chosen deliberately).
- Alpha channels; all scene colors are RGB.
