# Radical
"""Unit tests for streets_at_night.config.serde reflection walkers."""

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import (
    AtmosphereConfig,
    ColorPoolConfig,
    FixtureProfileConfig,
)
from streets_at_night.config.serde import dataclass_from_dict, dataclass_to_dict


# ---------------------------------------------------------------------------
# Toy dataclasses exercising each generic-walker feature
# ---------------------------------------------------------------------------


@dataclass
class _Leaf:
    a: int
    b: float = 1.5


@dataclass
class _Branch:
    leaf: _Leaf
    leaves: t.List[_Leaf]
    tag: str = "x"
    opt: t.Optional[_Leaf] = None
    mapping: t.Dict[str, _Leaf] = field(default_factory=dict)
    renamed: int = field(default=0, metadata={"toml_key": "aka"})


# ---------------------------------------------------------------------------
# dataclass_from_dict — generic behavior
# ---------------------------------------------------------------------------


def test_from_dict_recurses_into_nested_dataclass():
    # Act
    result = dataclass_from_dict(_Branch, {"leaf": {"a": 2}, "leaves": []})

    # Assert
    assert result.leaf == _Leaf(a=2, b=1.5)


def test_from_dict_recurses_into_list_of_dataclasses():
    # Act
    result = dataclass_from_dict(_Branch, {"leaf": {"a": 1}, "leaves": [{"a": 3}, {"a": 4, "b": 9.0}]})

    # Assert
    assert result.leaves == [_Leaf(a=3), _Leaf(a=4, b=9.0)]


def test_from_dict_recurses_into_dict_of_dataclasses():
    # Act
    result = dataclass_from_dict(
        _Branch, {"leaf": {"a": 1}, "leaves": [], "mapping": {"k": {"a": 7}}}
    )

    # Assert
    assert result.mapping == {"k": _Leaf(a=7)}


def test_from_dict_missing_key_uses_field_default():
    # Act — 'tag' and 'opt' omitted
    result = dataclass_from_dict(_Branch, {"leaf": {"a": 1}, "leaves": []})

    # Assert
    assert result.tag == "x"
    assert result.opt is None


def test_from_dict_optional_present_recurses():
    # Act
    result = dataclass_from_dict(_Branch, {"leaf": {"a": 1}, "leaves": [], "opt": {"a": 5}})

    # Assert
    assert result.opt == _Leaf(a=5)


def test_from_dict_honors_toml_key_rename():
    # Act — TOML key is 'aka', dataclass field is 'renamed'
    result = dataclass_from_dict(_Branch, {"leaf": {"a": 1}, "leaves": [], "aka": 42})

    # Assert
    assert result.renamed == 42


def test_from_dict_coerces_int_field_from_float():
    # Act — a whole-number float in TOML for an int-typed field
    result = dataclass_from_dict(_Leaf, {"a": 3.0})

    # Assert
    assert result.a == 3
    assert isinstance(result.a, int)


# ---------------------------------------------------------------------------
# dataclass_to_dict — generic behavior
# ---------------------------------------------------------------------------


def test_to_dict_drops_none_values():
    # Arrange
    branch = _Branch(leaf=_Leaf(a=1), leaves=[], opt=None)

    # Act
    result = dataclass_to_dict(branch)

    # Assert — TOML has no null, so opt is omitted entirely
    assert "opt" not in result


def test_to_dict_applies_toml_key_rename():
    # Arrange
    branch = _Branch(leaf=_Leaf(a=1), leaves=[], renamed=9)

    # Act
    result = dataclass_to_dict(branch)

    # Assert
    assert result["aka"] == 9
    assert "renamed" not in result


def test_to_dict_recurses_into_nested_and_lists():
    # Arrange
    branch = _Branch(leaf=_Leaf(a=1, b=2.0), leaves=[_Leaf(a=3)])

    # Act
    result = dataclass_to_dict(branch)

    # Assert
    assert result["leaf"] == {"a": 1, "b": 2.0}
    assert result["leaves"] == [{"a": 3, "b": 1.5}]


# ---------------------------------------------------------------------------
# Registered special cases
# ---------------------------------------------------------------------------


def test_color_pool_from_dict_produces_tuples():
    # Act
    result = dataclass_from_dict(ColorPoolConfig, {"colors": [["#FFAA00", 1.0], ["#001122", 2.0]]})

    # Assert
    assert result.colors == [("#FFAA00", 1.0), ("#001122", 2.0)]


def test_color_pool_to_dict_produces_lists():
    # Arrange
    pool = ColorPoolConfig(colors=[("#FFAA00", 1.0)])

    # Act
    result = dataclass_to_dict(pool)

    # Assert — TOML pairs are lists, not tuples
    assert result == {"colors": [["#FFAA00", 1.0]]}


def test_hex_color_field_parses_to_rgb_tuple():
    # Act — haze_color carries the hex_color serde marker
    result = dataclass_from_dict(
        AtmosphereConfig,
        {"falloff_distance": 1.0, "haze_color": "#204060", "scatter_distance": 2.0},
    )

    # Assert
    assert result.haze_color == (0x20, 0x40, 0x60)


def test_hex_color_field_serializes_back_to_hex():
    # Arrange
    atmosphere = dataclass_from_dict(
        AtmosphereConfig,
        {"falloff_distance": 1.0, "haze_color": "#204060", "scatter_distance": 2.0},
    )

    # Act
    result = dataclass_to_dict(atmosphere)

    # Assert
    assert result["haze_color"] == "#204060"


def test_fixture_profile_missing_migration_fields_raises():
    # Arrange — a pre-migration profile with no color_temp_k / beam_angle_deg
    stale = {"height_min": 5.0, "height_max": 6.0}

    # Act & Assert
    with pytest.raises(ValueError, match="Kelvin/beam-angle migration"):
        dataclass_from_dict(FixtureProfileConfig, stale)


def test_scene_config_quality_rename_round_trips():
    # Arrange — a pure-defaults config exercises the field<->key rename
    cfg = load_config(Path("nonexistent.toml"))

    # Act
    as_dict = dataclass_to_dict(cfg)

    # Assert — serialized under "quality", not "quality_presets"
    assert "quality" in as_dict
    assert "quality_presets" not in as_dict
    assert set(cfg.quality_presets.keys()) == set(as_dict["quality"].keys())
