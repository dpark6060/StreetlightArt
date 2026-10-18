# Radical
"""Unit tests for streets_at_night.rendering.flat_map."""

from PIL import Image

from streets_at_night.rendering.flat_map import save_flat_map_to_output


def test_save_flat_map_to_output_writes_region_named_png(tmp_path):
    # Arrange
    img = Image.new("RGB", (4, 4), (10, 20, 30))

    # Act
    result = save_flat_map_to_output(img, "rochester", tmp_path)

    # Assert
    assert result == tmp_path / "flat_map_rochester.png"
    assert result.exists()
    saved = Image.open(result).convert("RGB")
    assert saved.size == (4, 4)


def test_save_flat_map_to_output_empty_region_uses_plain_name(tmp_path):
    # Arrange
    img = Image.new("RGB", (4, 4))

    # Act
    result = save_flat_map_to_output(img, "", tmp_path)

    # Assert
    assert result == tmp_path / "flat_map.png"
    assert result.exists()


def test_save_flat_map_to_output_creates_missing_output_dir(tmp_path):
    # Arrange
    img = Image.new("RGB", (4, 4))
    output_dir = tmp_path / "nested" / "output"

    # Act
    result = save_flat_map_to_output(img, "rochester", output_dir)

    # Assert
    assert result.exists()
