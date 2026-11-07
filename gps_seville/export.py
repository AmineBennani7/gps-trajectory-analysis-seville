"""Export results as GeoJSON (RFC 7946, WGS84) and CSV."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def _serialisable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
        elif out[col].dtype == object:
            out[col] = out[col].map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
    return out


def to_geojson(gdf: gpd.GeoDataFrame) -> str:
    """GeoJSON text in WGS84 with timestamps as ISO 8601 strings."""
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return _serialisable(gdf).to_json(drop_id=True)


def to_csv(df: pd.DataFrame) -> str:
    """CSV text. Geometries are written as WKT."""
    out = pd.DataFrame(df).copy()
    if "geometry" in out.columns:
        out["geometry"] = gpd.GeoSeries(out["geometry"]).to_wkt()
    return out.to_csv(index=False)
