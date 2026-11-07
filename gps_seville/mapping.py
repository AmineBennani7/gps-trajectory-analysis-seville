"""Folium maps.

`build_map` is the original single-trajectory map. `build_analysis_map` is the
dashboard map with one toggleable layer per result type. Both use OpenStreetMap
tiles with the attribution Folium adds by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from html import escape

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import HeatMap, PolyLineOffset

from gps_seville.analysis import TRAJ_ID


def _fmt_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes} min {seconds:02d} s"


def build_map(result, removed=None):
    """Return a folium.Map with the route, start/end, stops and removed points."""
    df = result.trajectory.df
    coords = list(zip(df.geometry.y, df.geometry.x, strict=True))
    minx, miny, maxx, maxy = df.geometry.total_bounds
    m = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=14, tiles="OpenStreetMap"
    )

    route = folium.FeatureGroup(name="Route")
    folium.PolyLine(coords, color="#1f77b4", weight=4, opacity=0.8).add_to(route)
    folium.Marker(
        coords[0],
        tooltip=f"Start {df.index[0]:%H:%M:%S}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route)
    folium.Marker(
        coords[-1],
        tooltip=f"End {df.index[-1]:%H:%M:%S}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(route)
    route.add_to(m)

    stops = folium.FeatureGroup(name="Stops")
    for i, row in enumerate(result.stops.itertuples(), start=1):
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            tooltip=f"Stop {i}: {row.start_time:%H:%M} - {row.end_time:%H:%M} "
            f"({_fmt_duration(row.duration_s)})",
            icon=folium.Icon(color="orange", icon="pause"),
        ).add_to(stops)
    stops.add_to(m)

    if removed is not None and len(removed):
        group = folium.FeatureGroup(name="Removed points (inside map area)", show=False)
        for row in removed.itertuples():
            try:
                lat, lon = float(row.latitude), float(row.longitude)
            except (TypeError, ValueError):
                continue
            if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                folium.CircleMarker(
                    [lat, lon],
                    radius=5,
                    color="black",
                    fill=True,
                    fill_color="black",
                    tooltip=row.reason,
                ).add_to(group)
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


@dataclass(frozen=True)
class MapLayers:
    trajectories: bool = True
    start_end: bool = True
    stops: bool = True
    rejected: bool = False
    raw_track: bool = False
    zones: bool = True
    buffers: bool = False
    frequent_locations: bool = False
    anomalies: bool = False
    heatmap: bool = False


def zoom_for_extent(
    lon_span: float, lat_span: float, width_px: int = 900, height_px: int = 600
) -> int:
    """Largest web-map zoom level at which the extent fits into width x height pixels."""
    for zoom in range(18, 1, -1):
        degrees_per_px = 360 / (256 * 2**zoom)
        if lon_span / degrees_per_px <= width_px and lat_span * 1.27 / degrees_per_px <= height_px:
            return zoom  # 1.27 ≈ 1/cos(38°): latitude degrees are longer on the map here
    return 2


OSM_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
CARTO_ATTRIBUTION = OSM_ATTRIBUTION + ' &copy; <a href="https://carto.com/attributions">CARTO</a>'
OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
CARTO_TILES = "https://{s}.basemaps.cartocdn.com/%s/{z}/{x}/{y}{r}.png?api_key=%s"
LIGHT_GREY = "Light grey (OpenStreetMap)"
TRACK_SPACING_PX = 3.0  # screen distance between overlapping trajectories
DEFAULT_BASEMAP = LIGHT_GREY
# Standard OSM tiles, desaturated and lightened in the browser so coloured
# trajectories stand out. Same tile server and usage policy as plain OSM.
GREY_CSS = "<style>.basemap-grey { filter: grayscale(1) brightness(1.06) contrast(0.82); }</style>"


def available_basemaps() -> list[str]:
    """Basemaps that can be shown. CARTO tiles need an API key (CARTO_API_KEY)."""
    names = [LIGHT_GREY, "OpenStreetMap"]
    if os.getenv("CARTO_API_KEY"):
        names += ["CARTO Positron", "CARTO Voyager"]
    return names


def _tile_layer(name: str, show: bool) -> folium.TileLayer:
    key = os.getenv("CARTO_API_KEY", "")
    carto = {"CARTO Positron": "light_all", "CARTO Voyager": "rastertiles/voyager"}
    if name in carto:
        return folium.TileLayer(
            CARTO_TILES % (carto[name], key),
            attr=CARTO_ATTRIBUTION,
            name=name,
            subdomains="abcd",
            max_zoom=20,
            overlay=False,
            control=True,
            show=show,
        )
    extra = {"className": "basemap-grey"} if name == LIGHT_GREY else {}
    return folium.TileLayer(
        OSM_TILES,
        attr=OSM_ATTRIBUTION,
        name=name,
        max_zoom=19,
        overlay=False,
        control=True,
        show=show,
        **extra,
    )


def add_basemaps(m: folium.Map, selected: str) -> None:
    """Add every available basemap; `selected` is shown first, the others are in the
    layer control (radio buttons)."""
    names = available_basemaps()
    if selected not in names:
        selected = DEFAULT_BASEMAP
    m.get_root().header.add_child(folium.Element(GREY_CSS))
    for name in [selected, *(n for n in names if n != selected)]:
        _tile_layer(name, show=name == selected).add_to(m)


def _swatch(colour: str, label: str) -> str:
    return f'<span style="color:{colour}">&#9632;</span> {escape(label)}'


def build_analysis_map(
    view,
    layers: MapLayers,
    zones: gpd.GeoDataFrame | None = None,
    rejected: pd.DataFrame | None = None,
    raw_track: gpd.GeoDataFrame | None = None,
    buffers: gpd.GeoDataFrame | None = None,
    locations: gpd.GeoDataFrame | None = None,
    anomalies: gpd.GeoDataFrame | None = None,
    estimated_position: tuple[float, float, str] | None = None,
    basemap: str = DEFAULT_BASEMAP,
) -> folium.Map:
    """Map of a filtered View. Each enabled layer becomes a FeatureGroup in the layer control.

    `estimated_position` is (lat, lon, label) for a position interpolated by
    MovingPandas between two fixes; it is drawn as a hollow ring to set it apart
    from observed GPS fixes.
    """
    pts = view.points
    if pts.empty:
        m = folium.Map(location=[37.385, -5.99], zoom_start=13, tiles=None, control_scale=True)
        add_basemaps(m, basemap)
        folium.LayerControl(collapsed=False).add_to(m)
        return m
    minx, miny, maxx, maxy = pts.total_bounds
    # Zoom from the data extent. fit_bounds is not used: inside a Streamlit tab the map
    # is created while hidden (zero size), and Leaflet then fits the whole world.
    m = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2],
        zoom_start=zoom_for_extent(maxx - minx, maxy - miny),
        tiles=None,
        control_scale=True,
    )
    add_basemaps(m, basemap)

    if layers.zones and zones is not None and not zones.empty:
        g = folium.FeatureGroup(name="Geofence zones")
        folium.GeoJson(
            zones[["name", "zone_type", "geometry"]].to_json(),
            style_function=lambda _: {
                "color": "#555555",
                "weight": 1.5,
                "dashArray": "4 4",
                "fillColor": "#888888",
                "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(["name", "zone_type"], aliases=["Zone", "Type"]),
        ).add_to(g)
        g.add_to(m)

    if layers.buffers and buffers is not None and not buffers.empty:
        g = folium.FeatureGroup(name=f"Route buffers ({int(buffers['buffer_m'].iloc[0])} m)")
        for row in buffers.itertuples():
            colour = view.colours.get(getattr(row, TRAJ_ID), "#0072B2")
            folium.GeoJson(
                gpd.GeoSeries([row.geometry]).to_json(),
                style_function=lambda _, c=colour: {
                    "color": c,
                    "weight": 0,
                    "fillColor": c,
                    "fillOpacity": 0.12,
                },
            ).add_to(g)
        g.add_to(m)

    if layers.heatmap:
        g = folium.FeatureGroup(name="Heatmap of fixes")
        HeatMap(
            list(zip(pts.geometry.y, pts.geometry.x, strict=True)),
            radius=12,
            blur=15,
            min_opacity=0.3,
        ).add_to(g)
        g.add_to(m)

    if layers.trajectories:
        n = len(view.lines)
        for i, row in enumerate(view.lines.itertuples()):
            tid = getattr(row, TRAJ_ID)
            colour = view.colours.get(tid, "#0072B2")
            g = folium.FeatureGroup(name=_swatch(colour, tid))
            coords = [(y, x) for x, y in row.geometry.coords]
            # Routes on the same streets are drawn side by side: a screen offset of a few
            # pixels (leaflet-polylineoffset). The coordinates themselves are not moved.
            # The offset is to the right of the direction of travel, so each trajectory
            # gets a different non-negative value; with symmetric +/- offsets a route
            # driven in the opposite direction would land on top of another one.
            offset = i * min(TRACK_SPACING_PX, 9 / max(n - 1, 1))
            PolyLineOffset(coords, offset=offset, color="#ffffff", weight=6, opacity=0.9).add_to(g)
            PolyLineOffset(
                coords,
                offset=offset,
                color=colour,
                weight=3.5,
                opacity=0.95,
                tooltip=f"{tid}<br>{row.start_t:%H:%M} - {row.end_t:%H:%M}",
            ).add_to(g)
            if layers.start_end:
                (x0, y0), (x1, y1) = row.geometry.coords[0], row.geometry.coords[-1]
                folium.Marker(
                    [y0, x0],
                    tooltip=f"Start {tid} {row.start_t:%H:%M:%S}",
                    icon=folium.Icon(color="green", icon="play"),
                ).add_to(g)
                folium.Marker(
                    [y1, x1],
                    tooltip=f"End {tid} {row.end_t:%H:%M:%S}",
                    icon=folium.Icon(color="red", icon="stop"),
                ).add_to(g)
            g.add_to(m)

    # drawn after the cleaned lines so jumps and zig-zags stay visible on top
    if layers.raw_track and raw_track is not None and not raw_track.empty:
        sel = raw_track[raw_track[TRAJ_ID].isin(view.lines[TRAJ_ID])]
        g = folium.FeatureGroup(name=_swatch("#222222", "Raw fixes before cleaning (dashed)"))
        for _, t in sel.groupby(TRAJ_ID):
            folium.PolyLine(
                list(zip(t.geometry.y, t.geometry.x, strict=True)),
                color="#222222",
                weight=1.5,
                opacity=0.9,
                dash_array="3 4",
            ).add_to(g)
        g.add_to(m)

    if layers.stops and not view.stops.empty:
        g = folium.FeatureGroup(name="Detected stops")
        for row in view.stops.itertuples():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=5 + min(row.duration_s / 120, 10),
                color="#333333",
                weight=1,
                fill=True,
                fill_color="#E69F00",
                fill_opacity=0.9,
                tooltip=(
                    f"Stop {row.vehicle_id}<br>{row.start_time:%Y-%m-%d %H:%M} - "
                    f"{row.end_time:%H:%M} ({_fmt_duration(row.duration_s)})"
                ),
            ).add_to(g)
        g.add_to(m)

    if layers.frequent_locations and locations is not None and not locations.empty:
        g = folium.FeatureGroup(name="Frequent locations (DBSCAN)")
        for row in locations.itertuples():
            folium.Circle(
                [row.geometry.y, row.geometry.x],
                radius=60,
                color="#CC79A7",
                weight=2,
                fill=True,
                fill_opacity=0.25,
                tooltip=(
                    f"Location {row.location_id}: {row.visits} visits, "
                    f"{row.vehicles} vehicle(s), {row.total_dwell_min} min"
                ),
            ).add_to(g)
        g.add_to(m)

    if layers.anomalies and anomalies is not None and not anomalies.empty:
        g = folium.FeatureGroup(name="Unusual movement")
        for row in anomalies.itertuples():
            folium.RegularPolygonMarker(
                [row.geometry.y, row.geometry.x],
                number_of_sides=3,
                radius=7,
                color="#D55E00",
                fill=True,
                fill_color="#D55E00",
                fill_opacity=0.8,
                tooltip=(
                    f"{row.vehicle_id} {row.timestamp:%H:%M:%S}<br>{row.metric} = "
                    f"{row.value} (z = {row.z_score})"
                ),
            ).add_to(g)
        g.add_to(m)

    if layers.rejected and rejected is not None and not rejected.empty:
        rejected = rejected[rejected["vehicle_id"].isin(view.lines["vehicle_id"])]
        g = folium.FeatureGroup(name="Rejected GPS records")
        lat = pd.to_numeric(rejected["latitude"], errors="coerce")
        lon = pd.to_numeric(rejected["longitude"], errors="coerce")
        near = lat.between(miny - 0.05, maxy + 0.05) & lon.between(minx - 0.05, maxx + 0.05)
        for i in rejected.index[near]:
            r = rejected.loc[i]
            folium.CircleMarker(
                [lat[i], lon[i]],
                radius=4,
                color="#000000",
                weight=1,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=1,
                tooltip=f"Rejected: {escape(str(r['reason']))}<br>{escape(str(r['timestamp']))}",
            ).add_to(g)
        g.add_to(m)

    if estimated_position is not None:
        lat_e, lon_e, label = estimated_position
        g = folium.FeatureGroup(name="Estimated position (interpolated)")
        folium.CircleMarker(
            [lat_e, lon_e], radius=9, color="#000000", weight=3, fill=False, tooltip=escape(label)
        ).add_to(g)
        g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
