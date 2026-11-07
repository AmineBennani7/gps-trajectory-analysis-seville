"""Coordinate reference systems.

GPS positions are stored in WGS84 (EPSG:4326, degrees). Lengths, areas and
buffers in metres need a projected CRS. For Seville the default is
ETRS89 / UTM zone 30N (EPSG:25830). With "auto", GeoPandas picks the UTM zone
that contains the data (via pyproj).

Within a city, WGS84 and ETRS89 differ by well under a metre today, which is
far below GPS error, so no datum transformation is attempted.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import CRS

from gps_seville.exceptions import ConfigError

WGS84 = "EPSG:4326"


def resolve_projected_crs(gdf: gpd.GeoDataFrame, setting: str) -> CRS:
    """Return a projected CRS in metres for `gdf` ("auto" or any pyproj input)."""
    if setting.lower() == "auto":
        crs = gdf.estimate_utm_crs()
    else:
        try:
            crs = CRS.from_user_input(setting)
        except Exception as exc:  # pyproj raises CRSError, keep the message for the user
            raise ConfigError(f"Unknown CRS '{setting}': {exc}") from exc
    if not crs.is_projected:
        raise ConfigError(f"{crs.to_string()} is not a projected CRS")
    unit = crs.axis_info[0].unit_name
    if unit not in ("metre", "meter"):
        raise ConfigError(f"{crs.to_string()} uses {unit}, a metre-based CRS is needed")
    return crs


def path_length_m(points: gpd.GeoSeries) -> float:
    """Length of the polyline through `points` (in order), in the units of their CRS."""
    if len(points) < 2:
        return 0.0
    x, y = points.x.to_numpy(), points.y.to_numpy()
    return float(np.hypot(np.diff(x), np.diff(y)).sum())
