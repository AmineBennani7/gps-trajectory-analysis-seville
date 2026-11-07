"""Optional road-network analysis with OSMnx (OpenStreetMap data).

Needs `pip install osmnx` and internet access to the Overpass API the first
time an area is downloaded; OSMnx caches responses in `cache/`.

`snap_to_roads` assigns each GPS fix to the nearest road edge. This is simple
nearest-edge snapping, not map matching: it does not use the order of the fixes
or network connectivity, so near junctions and parallel streets a fix can be
assigned to the wrong road. The snapping distance is reported so such cases can
be filtered.

Road data © OpenStreetMap contributors, available under the ODbL.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from gps_seville.analysis import TRAJ_ID
from gps_seville.exceptions import GpsAnalysisError
from gps_seville.logging_config import get_logger

log = get_logger(__name__)


def osmnx_available() -> bool:
    try:
        import osmnx  # noqa: F401
    except ImportError:
        return False
    return True


def download_roads(
    area: BaseGeometry,
    network_type: str = "drive",
    cache_folder: str = "cache/osmnx",
    timeout_s: int = 60,
) -> gpd.GeoDataFrame:
    """Download the road network inside `area` (WGS84 polygon) and return its edges.

    `timeout_s` limits each request to the Overpass API so the app does not hang
    when the server is slow or unreachable.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise GpsAnalysisError("Road analysis needs OSMnx: pip install osmnx") from exc
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_folder
    ox.settings.requests_timeout = timeout_s
    ox.settings.overpass_rate_limit = False  # do not wait on the server's slot status
    try:
        graph = ox.graph_from_polygon(area, network_type=network_type, retain_all=False)
    except Exception as exc:  # network errors, empty responses
        raise GpsAnalysisError(f"Could not download the road network: {exc}") from exc
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    log.info("Downloaded %d road edges", len(edges))
    return roads_from_edges(edges)


def roads_from_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep the edge columns used here; OSM tags can be lists, which become text."""

    def text(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    out = gpd.GeoDataFrame(edges[["geometry"]].copy(), geometry="geometry", crs=edges.crs)
    for col in ("name", "highway"):
        out[col] = edges[col].map(text) if col in edges else ""
    out["name"] = out["name"].replace("", "(unnamed)")
    return out


def snap_to_roads(
    points: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, crs: CRS, max_distance_m: float = 30.0
) -> gpd.GeoDataFrame:
    """Nearest road edge for each fix (GeoPandas sjoin_nearest in a projected CRS).

    Fixes farther than `max_distance_m` from every road get no road.
    """
    p = points[["timestamp", "vehicle_id", TRAJ_ID, "geometry"]].to_crs(crs)
    r = roads.to_crs(crs)
    j = gpd.sjoin_nearest(
        p,
        r[["name", "highway", "geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="snap_m",
    )
    j = j[~j.index.duplicated()]  # equidistant edges: keep one
    j = j.drop(columns="index_right").rename(columns={"name": "road", "highway": "road_type"})
    j["snap_m"] = j["snap_m"].round(1)
    return j.to_crs("EPSG:4326")


def road_usage(snapped: pd.DataFrame) -> pd.DataFrame:
    """Number of fixes and vehicles per road (fixes not near any road are left out)."""
    on_road = snapped.dropna(subset=["road"])
    if on_road.empty:
        return pd.DataFrame(columns=["road", "road_type", "fixes", "vehicles", "median_snap_m"])
    return (
        on_road.groupby(["road", "road_type"])
        .agg(
            fixes=("timestamp", "size"),
            vehicles=("vehicle_id", "nunique"),
            median_snap_m=("snap_m", "median"),
        )
        .reset_index()
        .sort_values("fixes", ascending=False)
    )
