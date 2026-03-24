"""Verification plot for OSM data — saves a matplotlib figure showing roads, land use, and buildings."""

import typing as t
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geopandas as gpd

from ..config.schema import CameraConfig, SceneConfig
from ..models.road_types import RoadType

matplotlib.use("Agg")


_ROAD_COLORS: t.Dict[RoadType, str] = {
    RoadType.HIGHWAY: "#FF6B6B",
    RoadType.MAJOR_ROAD: "#FFA500",
    RoadType.COLLECTOR: "#FFD700",
    RoadType.RESIDENTIAL: "#90EE90",
    RoadType.SERVICE: "#87CEEB",
    RoadType.MINOR: "#DDA0DD",
}

_ROAD_LINEWIDTHS: t.Dict[RoadType, float] = {
    RoadType.HIGHWAY: 2.0,
    RoadType.MAJOR_ROAD: 1.5,
    RoadType.COLLECTOR: 1.0,
    RoadType.RESIDENTIAL: 0.6,
    RoadType.SERVICE: 0.4,
    RoadType.MINOR: 0.4,
}

_LAND_USE_COLORS: t.Dict[str, str] = {
    "residential": "#228B22",
    "commercial": "#DC143C",
    "retail": "#DC143C",
    "industrial": "#696969",
    "park": "#006400",
    "recreation_ground": "#006400",
    "grass": "#006400",
    "garden": "#006400",
    "nature_reserve": "#006400",
    "water": "#00008B",
    "riverbank": "#00008B",
    "wood": "#013220",
    "forest": "#013220",
    "grassland": "#2E7D32",
}

_LAND_USE_TAG_COLUMNS: t.List[str] = ["landuse", "leisure", "natural", "waterway"]

_ROAD_PLOT_ORDER: t.List[RoadType] = [
    RoadType.MINOR,
    RoadType.SERVICE,
    RoadType.RESIDENTIAL,
    RoadType.COLLECTOR,
    RoadType.MAJOR_ROAD,
    RoadType.HIGHWAY,
]


def save_verification_plot(
    config: SceneConfig,
    roads_gdf: gpd.GeoDataFrame,
    land_use_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Build and save the OSM verification plot to disk.

    Args:
        config (SceneConfig): Scene configuration (used for camera markers).
        roads_gdf (GeoDataFrame): Classified road edges with 'road_type' column.
        land_use_gdf (GeoDataFrame): Land use polygons with OSM tags.
        buildings_gdf (GeoDataFrame): Building footprint polygons.
        output_path (Path): Destination PNG path.
    """
    fig = _build_verification_figure(config, roads_gdf, land_use_gdf, buildings_gdf)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def _build_verification_figure(
    config: SceneConfig,
    roads: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> plt.Figure:
    """Build the full verification matplotlib figure.

    Args:
        config (SceneConfig): Scene config (for camera/target markers).
        roads (GeoDataFrame): Classified road edges.
        land_use (GeoDataFrame): Land use polygons.
        buildings (GeoDataFrame): Building footprints.

    Returns:
        Figure: Matplotlib figure ready to save.
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    _plot_land_use(ax, land_use)
    _plot_buildings(ax, buildings)
    _plot_roads(ax, roads)
    _plot_camera_markers(ax, config.camera)
    _add_legend(ax, roads)
    _style_axes(ax)

    return fig


def _plot_land_use(ax: plt.Axes, land_use: gpd.GeoDataFrame) -> None:
    """Plot land use polygons as semi-transparent colored fills.

    Iterates over each known OSM tag column and delegates per-column
    rendering to _plot_tag_column.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        land_use (GeoDataFrame): Land use polygons with OSM tags.
    """
    if land_use.empty:
        return

    for tag_col in _LAND_USE_TAG_COLUMNS:
        if tag_col not in land_use.columns:
            continue
        _plot_tag_column(ax, land_use, tag_col, _LAND_USE_COLORS)


def _plot_tag_column(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    tag_col: str,
    color_map: t.Dict[str, str],
) -> None:
    """Plot all rows in gdf that match known values in color_map for a single tag column.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        gdf (GeoDataFrame): GeoDataFrame containing the tag column.
        tag_col (str): Column name to match against (e.g. 'landuse', 'leisure').
        color_map (dict[str, str]): Mapping of tag value to hex color string.
    """
    subset = gdf[gdf[tag_col].notna()]
    for tag_value, color in color_map.items():
        matching = subset[subset[tag_col] == tag_value]
        if matching.empty:
            continue
        matching.plot(ax=ax, color=color, alpha=0.3, linewidth=0)


def _plot_buildings(ax: plt.Axes, buildings: gpd.GeoDataFrame) -> None:
    """Plot building footprints as small grey shapes.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        buildings (GeoDataFrame): Building footprint polygons.
    """
    if buildings.empty:
        return
    buildings.plot(ax=ax, color="#808080", alpha=0.5, linewidth=0.2, edgecolor="#808080")


def _plot_roads(ax: plt.Axes, roads: gpd.GeoDataFrame) -> None:
    """Plot road edges colored by road type, in back-to-front z-order.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        roads (GeoDataFrame): Classified road edges with 'road_type' column.
    """
    for road_type in _ROAD_PLOT_ORDER:
        subset = roads[roads["road_type"] == road_type]
        if subset.empty:
            continue
        color = _ROAD_COLORS[road_type]
        linewidth = _ROAD_LINEWIDTHS[road_type]
        subset.plot(ax=ax, color=color, linewidth=linewidth)


def _plot_camera_markers(ax: plt.Axes, camera: CameraConfig) -> None:
    """Plot camera position and target as annotated markers.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        camera (CameraConfig): Camera position and target.
    """
    _plot_marker(
        ax,
        lon=camera.lon,
        lat=camera.lat,
        marker_style="*",
        color="yellow",
        markersize=14,
        label="Camera",
        annotation=f"Camera\n{camera.altitude_ft:.0f}ft",
    )
    _plot_marker(
        ax,
        lon=camera.target_lon,
        lat=camera.target_lat,
        marker_style="o",
        color="cyan",
        markersize=10,
        label="Target",
        annotation=f"Target\n{camera.target_altitude_ft:.0f}ft",
    )


def _plot_marker(
    ax: plt.Axes,
    lon: float,
    lat: float,
    marker_style: str,
    color: str,
    markersize: int,
    label: str,
    annotation: str,
) -> None:
    """Plot a single annotated marker at the given coordinates.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        lon (float): Longitude of the marker.
        lat (float): Latitude of the marker.
        marker_style (str): Matplotlib marker string (e.g. '*', 'o').
        color (str): Marker and annotation color.
        markersize (int): Size of the marker in points.
        label (str): Legend label for the marker.
        annotation (str): Text string for the nearby annotation.
    """
    ax.plot(lon, lat, marker=marker_style, color=color, markersize=markersize, zorder=10, label=label)
    ax.annotate(
        annotation,
        xy=(lon, lat),
        xytext=(8, 8),
        textcoords="offset points",
        color=color,
        fontsize=8,
    )


def _build_legend_handles(roads: gpd.GeoDataFrame) -> t.List[t.Any]:
    """Build the list of legend handles for road types, land use, and camera markers.

    Only road types present in the data are included.

    Args:
        roads (GeoDataFrame): Classified road edges with 'road_type' column.

    Returns:
        list: Combined legend handles for roads, land use categories, and camera markers.
    """
    road_patches = _build_road_patches(roads)
    land_use_patches = _build_land_use_patches()
    camera_patches = _build_camera_patches()
    return road_patches + land_use_patches + camera_patches


def _build_road_patches(roads: gpd.GeoDataFrame) -> t.List[mpatches.Patch]:
    """Build legend patches for road types present in the data.

    Args:
        roads (GeoDataFrame): Classified road edges.

    Returns:
        list[Patch]: One patch per present road type.
    """
    patches = []
    for road_type in RoadType:
        if roads[roads["road_type"] == road_type].empty:
            continue
        patch = mpatches.Patch(
            color=_ROAD_COLORS[road_type],
            label=road_type.value.replace("_", " ").title(),
        )
        patches.append(patch)
    return patches


def _build_land_use_patches() -> t.List[mpatches.Patch]:
    """Build legend patches for land use categories.

    Returns:
        list[Patch]: Fixed set of land use category patches.
    """
    return [
        mpatches.Patch(color="#228B22", alpha=0.5, label="Residential"),
        mpatches.Patch(color="#DC143C", alpha=0.5, label="Commercial/Retail"),
        mpatches.Patch(color="#696969", alpha=0.5, label="Industrial"),
        mpatches.Patch(color="#006400", alpha=0.5, label="Park/Green"),
        mpatches.Patch(color="#00008B", alpha=0.5, label="Water"),
        mpatches.Patch(color="#013220", alpha=0.5, label="Forest/Wood"),
    ]


def _build_camera_patches() -> t.List[plt.Line2D]:
    """Build legend handles for the camera and target markers.

    Returns:
        list[Line2D]: Camera and target legend handles.
    """
    return [
        plt.Line2D([0], [0], marker="*", color="yellow", linestyle="None", markersize=10, label="Camera"),
        plt.Line2D([0], [0], marker="o", color="cyan", linestyle="None", markersize=8, label="Target"),
    ]


def _add_legend(ax: plt.Axes, roads: gpd.GeoDataFrame) -> None:
    """Add a legend showing road types, land use categories, and camera markers.

    Args:
        ax (Axes): Matplotlib axes to add the legend to.
        roads (GeoDataFrame): Classified roads (to show only present types).
    """
    handles = _build_legend_handles(roads)
    ax.legend(
        handles=handles,
        loc="lower right",
        facecolor="#2a2a4e",
        edgecolor="#555577",
        labelcolor="white",
        fontsize=8,
        framealpha=0.8,
    )


def _style_axes(ax: plt.Axes) -> None:
    """Apply dark-theme styling and labels to the axes.

    Args:
        ax (Axes): Matplotlib axes to style.
    """
    ax.set_title(
        "Streets at Night — OSM Data Verification\nSan Bernardino, CA",
        color="white",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Longitude", color="white", fontsize=10)
    ax.set_ylabel("Latitude", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")
