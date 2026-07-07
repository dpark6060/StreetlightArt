"""Reflection-based (de)serialization between config dataclasses and dicts.

``schema.py`` is the single source of truth for the config surface. Rather than
hand-writing a ``_build_*`` reader and a ``_*_to_dict`` writer per section (the
four-way sync that kept drifting — see the Fable audit change_09/change_18
docs), the load and save paths walk the dataclass definitions by reflection.

Two generic walkers do almost everything:

- ``dataclass_from_dict(cls, data)`` — build a dataclass from a merged config
  dict. Nested dataclasses recurse; ``list[DC]`` and ``dict[str, DC]`` recurse
  per element/value; a missing key falls back to the dataclass field default;
  ``int``/``float``-typed scalars are coerced.
- ``dataclass_to_dict(obj)`` — the inverse, producing a TOML-ready dict.
  ``None`` values are dropped (TOML has no null).

A tiny registry handles the cases that aren't pure field-by-field reflection:
color pools (``[[hex, weight], ...]`` <-> list of tuples), the fixture-profile
migration guard, and — declared as field metadata on ``schema.py`` — the
``quality``/``quality_presets`` key rename and the ``haze_color`` hex<->tuple
transform.
"""

import dataclasses
import typing as t

# cls -> callable(data: dict) -> instance   (bypasses generic field walk)
_FROM_HOOKS: t.Dict[type, t.Callable[[dict], t.Any]] = {}
# cls -> callable(obj) -> serialized value
_TO_HOOKS: t.Dict[type, t.Callable[[t.Any], t.Any]] = {}


def register_hooks(
    cls: type,
    from_dict: t.Optional[t.Callable[[dict], t.Any]] = None,
    to_dict: t.Optional[t.Callable[[t.Any], t.Any]] = None,
) -> None:
    """Register special-case (de)serializers for a dataclass type.

    Args:
        cls (type): The dataclass type to special-case.
        from_dict (callable, optional): Builds an instance from a dict,
            bypassing the generic field walk.
        to_dict (callable, optional): Serializes an instance to a TOML-ready
            value, bypassing the generic walk.
    """
    if from_dict is not None:
        _FROM_HOOKS[cls] = from_dict
    if to_dict is not None:
        _TO_HOOKS[cls] = to_dict


def _toml_key(f: dataclasses.Field) -> str:
    """Return the TOML key for a dataclass field (honoring a rename marker)."""
    return f.metadata.get("toml_key", f.name)


def _is_dataclass_type(tp: t.Any) -> bool:
    """Check whether tp is a dataclass type (not an instance)."""
    return dataclasses.is_dataclass(tp) and isinstance(tp, type)


def dataclass_from_dict(cls: type, data: dict) -> t.Any:
    """Construct a dataclass instance from a merged config dict.

    Args:
        cls (type): Target dataclass type.
        data (dict): Merged configuration dict (defaults overlaid with user
            TOML). Keys map to field names, except where a field declares a
            ``toml_key`` metadata rename.

    Returns:
        Any: A populated instance of ``cls``.
    """
    hook = _FROM_HOOKS.get(cls)
    if hook is not None:
        return hook(data)
    return _generic_from_dict(cls, data)


def _generic_from_dict(cls: type, data: dict) -> t.Any:
    """Build a dataclass by walking its fields (no hook dispatch on ``cls``)."""
    hints = t.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        key = _toml_key(f)
        if key not in data:
            continue  # rely on the dataclass field default
        kwargs[f.name] = _convert_from(hints[f.name], data[key], f)
    return cls(**kwargs)


def _convert_from(tp: t.Any, value: t.Any, f: t.Optional[dataclasses.Field]) -> t.Any:
    """Convert one raw config value to the type annotated on its field."""
    if value is None:
        return None

    if f is not None and f.metadata.get("serde") == "hex_color":
        return _hex_to_rgb_tuple(value)

    origin = t.get_origin(tp)

    if origin is t.Union:
        non_none = [a for a in t.get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _convert_from(non_none[0], value, None)
        return value

    if origin in (list, tuple):
        (elem_tp,) = t.get_args(tp)[:1] or (t.Any,)
        return [_convert_from(elem_tp, item, None) for item in value]

    if origin is dict:
        val_tp = t.get_args(tp)[1]
        return {k: _convert_from(val_tp, v, None) for k, v in value.items()}

    if _is_dataclass_type(tp):
        return dataclass_from_dict(tp, value)

    if tp is int and isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    return value


def dataclass_to_dict(obj: t.Any) -> dict:
    """Serialize a dataclass instance to a TOML-ready dict.

    ``None`` fields are dropped (TOML has no null). Nested dataclasses,
    lists, and dicts recurse. Field renames and hex-color transforms declared
    in the schema metadata are applied.

    Args:
        obj (Any): The dataclass instance to serialize.

    Returns:
        dict: A TOML-serializable dict.
    """
    hook = _TO_HOOKS.get(type(obj))
    if hook is not None:
        return hook(obj)

    result: dict = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        result[_toml_key(f)] = _convert_to(value, f)
    return result


def _convert_to(value: t.Any, f: t.Optional[dataclasses.Field]) -> t.Any:
    """Convert one dataclass value to a TOML-ready value."""
    if f is not None and f.metadata.get("serde") == "hex_color":
        return _rgb_tuple_to_hex(value)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)

    if isinstance(value, dict):
        return {k: _convert_to(v, None) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_convert_to(item, None) for item in value]

    return value


def _hex_to_rgb_tuple(value: str) -> t.Tuple[int, int, int]:
    """Parse a '#RRGGBB' hex string into an (R, G, B) int tuple."""
    hex_str = value.lstrip("#")
    return (
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16),
    )


def _rgb_tuple_to_hex(value: t.Sequence[int]) -> str:
    """Format an (R, G, B) int tuple as an uppercase '#RRGGBB' hex string."""
    r, g, b = value
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Special cases — registered explicitly, not scattered through the walkers.
# ---------------------------------------------------------------------------


def _register_special_cases() -> None:
    """Register the (de)serializers that aren't pure field reflection."""
    from .schema import ColorPoolConfig, FixtureProfileConfig

    # Color pool: TOML [[hex, weight], ...] <-> dataclass list of (hex, weight).
    register_hooks(
        ColorPoolConfig,
        from_dict=lambda data: ColorPoolConfig(
            colors=[tuple(entry) for entry in data["colors"]]
        ),
        to_dict=lambda pool: {"colors": [[color, weight] for color, weight in pool.colors]},
    )

    # Fixture profile: guard the Kelvin/beam-angle migration. A pre-migration
    # profile lacks these keys; fail with a clear message instead of a bare
    # KeyError (matches the historical loader behavior).
    def _fixture_profile_from_dict(data: dict):
        for key in ("color_temp_k", "beam_angle_deg"):
            if key not in data:
                raise ValueError(
                    "Streetlight profile in [lights.streetlights.*] is missing "
                    "'color_temp_k'/'beam_angle_deg' — this config predates the "
                    "Kelvin/beam-angle migration. Re-save it from devmode or "
                    "update the pool entries by hand."
                )
        return _generic_from_dict(FixtureProfileConfig, data)

    register_hooks(FixtureProfileConfig, from_dict=_fixture_profile_from_dict)


_register_special_cases()
