# Code Review: heat_haze branch (rendering focus)

**Branch:** `heat_haze` · **Date:** 2026-06-14 · Point-in-time review.

Tests: `uv run pytest tests/` → **594 passed in 9.33s**. No failures.

Scope: uncommitted changes on `heat_haze`, with focus on `src/streets_at_night/rendering/`.

---

## Critical

None. No correctness bugs or structural violations that demand a rewrite. The heat-haze
math is well-commented and the new module decomposition (HazeField / TorchWarper /
HeatHaze) is clean.

---

## Major

### 1. `cmd_preview` is a god-function — main.py:306-427
~120 lines doing: lazy imports, scene load, flat-map gen, pygame init, the entire event
loop, camera-moved detection, base-frame rebuild, haze setup, car facing, blit, and HUD.
This fails the AND/THEN test badly. The loop body alone (lines 366-425) mixes input
dispatch, dirty-frame management, haze application, and presentation.

Decompose: pull a `_handle_preview_result(result, scene, anim, ...)`, a
`_rebuild_base_frame_if_dirty(anim, haze, ...)`, and a `_present_preview_frame(...)`.
Keep `cmd_preview` as the orchestrator. The render loop in `cmd_animate` (main.py:473-489)
is smaller but has the same shape — its event handling is fine inline, but it duplicates
the blit/flip dance that `cmd_preview` does differently (one uses
`backend.blit_pil_image`, the other hand-rolls `pygame.image.fromstring`). Two render
loops, two ways to get a numpy frame onto a pygame surface — pick one.

### 2. Per-row Python loops in atmosphere_pass.py — atmosphere_pass.py:79-88, 123-132
Both `_render_horizon_glow` and `_render_distance_haze` loop `for row in range(height)`
in Python, writing one RGBA row per iteration. At export resolution (high = 4320 rows)
that's 4320 Python iterations building an array numpy could fill in a couple vectorized
ops. The whole point of building an `(H, W, 4)` ndarray is to avoid this.

`_render_horizon_glow` vectorizes trivially:
```python
rows = np.arange(height)
offset = np.abs(rows - horizon_y)
fade = np.where(offset < half_width, (1.0 - offset / half_width) ** 2, 0.0)
alpha = (peak_alpha * fade).astype(np.uint8)
rgba[:, :, :3] = glow_color
rgba[:, :, 3] = alpha[:, None]
```
`_render_distance_haze` is the same idea — `_haze_alpha_for_row` is pure arithmetic on
`row`, so compute it for the whole `np.arange(height)` at once. This is the most obvious
perf footgun in the changed code. It runs on every export and (for the distance-haze
path) potentially every preview frame where the camera moved.

### 3. Lazy `from ..main import _resolve_cache_dir` to dodge a circular import — cmd_devmode.py:52
Importing a private helper from `main` inside a function body is a code smell that says
the dependency graph is wrong. `main` imports the whole world; `devmode` reaching back
into `main` for one path helper creates a cycle that only works because the import is
deferred. `_resolve_cache_dir` is a pure `(Path, SceneConfig) -> Path` function with no
business being in the CLI entry module — move it to a small shared module
(`config/paths.py` or similar) and import it normally from both `main` and `devmode`.

---

## Minor

### 4. `_compute_origin` duplicated verbatim — main.py:1071-1082 and real_scene.py:132-143
Identical body (camera/target lat-lon midpoint). Same root cause as #3: there's no shared
home for coordinate-origin logic, so it got copy-pasted. Consolidate into the same shared
module you create for `_resolve_cache_dir`.

### 5. `t.Any` used for config params that have real types — main.py
`heat_haze_cfg: t.Any` appears in `_build_animate_haze` (518), `cmd_animate_record` (564),
`_default_video_duration` (625), `_compute_preview_depth` (749), and `preset: t.Any` in
`_apply_resolution_overrides` (856). These are `HeatHazeConfig` and `QualityPreset` —
both already imported or trivially importable. The docstrings even say "Active
HeatHazeConfig". Annotate them; `t.Any` here defeats the type checker for no reason.

### 6. `import dataclasses` inside function bodies — main.py:799 and 874
`cmd_render` and `_apply_resolution_overrides` both `import dataclasses` locally. Stdlib,
no cycle risk, used unconditionally — hoist to the top of the file. (Lots of other local
imports in main.py are genuinely there to keep pygame/imageio off the import path for the
non-interactive commands; those are defensible. `dataclasses` isn't.)

### 7. Module docstring mismatch — heat_haze.py:1
The file is `heat_haze.py` but its module docstring opens "Atmospheric heat haze
pipeline" — that's correct. However `atmosphere_pass.py`'s docstring (line 8) references
"matching the pattern used by the light glow pass in light_pass.py" which is a fine
cross-reference. No action; flagging that I checked and the heat_haze header is accurate.

### 8. `compute_depth_map` re-masks `hits` redundantly — heat_haze.py:126-129
```python
hits = np.abs(dz) > 1e-10
t_val = np.where(hits, -cam_pos[2] / np.where(hits, dz, 1.0), -1.0)
hits = hits & (t_val > 0.0)
```
The nested `np.where(hits, dz, 1.0)` guards division by zero, then `t_val` is set to
`-1.0` where `not hits`, then `hits` is recomputed. It works and it's the same pattern as
`warp_preview._raycast_to_flat_map:132-135` (good — consistent), but the double-`where`
is doing the divide-by-zero guard twice. Readable enough; leave it, just noting the
duplication between the two modules. If you ever touch one, factor out a shared
`ray_ground_t(cam_z, dz)` helper since the two are now copies. (Note: `illumination_pass`
`_raycast_to_ground` is a third copy of this same intersection.)

---

## Health / smells (good)

- `apply_heat_haze` and the old freq1/freq2/speed1/speed2 config were cleanly removed —
  no orphaned references anywhere (grepped `apply_heat_haze`, all the old config keys).
  The dead test block in `test_warp_preview.py` was deleted with it. Clean cut-over.
- `test_heat_haze.py` is genuinely good coverage: deterministic-per-seed, amplitude
  linearity, sky-factor edge cases, the raises-without-base-frame and
  raises-without-depth-map contracts, and the orchestrator test (`HeatHaze.apply`) mocks
  the warper and asserts one level deep. This matches the testing rules well.
- Docstrings on the new rendering code are strong — they explain the WHY (Taylor frozen
  flow, Von Karman PSD, why sky pixels get a fraction of max distance, why
  pixel_scale_m scales down with resolution). This is the rare case where the physics is
  actually documented.

## Coverage gaps

- **atmosphere_pass.py has no test file.** `render_atmosphere`, `_render_horizon_glow`,
  `_render_distance_haze`, `_haze_alpha_for_row`, `_parse_hex_color` are all untested.
  `_haze_alpha_for_row` is pure arithmetic with branch logic (ground vs sky, zero-range
  guards) — cheap to test and exactly the kind of logic that breaks silently. The
  `>= half_width` / `** 2` falloff change in this diff went in untested.
- **main.py orchestration is untested** (cmd_preview, cmd_animate, cmd_animate_record,
  the seamless crossfade math in `_record_haze_seamless:694-700`). The crossfade alpha
  blend is real logic, not just I/O, and it's the kind of thing that's wrong-by-one
  (`alpha = j / max(1, crossfade_frames - 1)`, mirror indexing). Worth a test that feeds
  a fake `haze` and asserts the last output frame equals frame 0.
- `compute_horizon_y`'s new forward-direction logic (render_utils.py:42-50) — the test
  only added a `get_forward.return_value` mock to keep the existing test passing; there's
  no test for the south-facing-camera case the comment says this fix is *for*.

---

### Summary
- Critical: 0
- Major: 3
- Minor: 5

### Verdict
Approve with minor fixes

The branch is in good shape — tests pass, the haze pipeline is well-structured and
well-tested, the old approach was removed cleanly. Before merge I'd want #2 (vectorize the
atmosphere row loops — real perf cost at export res) and #3 (the main↔devmode import
cycle). #1 (decompose cmd_preview) is the right call but can be a follow-up since it's not
new in this diff — though the diff did grow it.

---

### Pre-existing issues (not in scope)
- `cmd_preview`'s size predates this branch in part; the diff added the haze setup blocks
  but the god-function shape was already there. Counted as Major #1 anyway since the diff
  made it worse.
- The `_render_distance_haze` / `_render_horizon_glow` row loops may predate the branch;
  the diff only changed the falloff formula (`** 2`) on one line. The perf issue is
  pre-existing but flagged because it's the active file.
