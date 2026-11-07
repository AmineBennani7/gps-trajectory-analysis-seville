"""Spatial analysis against geographic features: zones (polygons) and points of interest.

GeoPandas does the spatial joins (`sjoin`, `sjoin_nearest`), overlays and
reprojection; Shapely does the geometry. My own code is the geofence logic
(turning point-in-zone results into enter/exit events and dwell times) and the
summary tables.

Geofence events are based on GPS fixes, not on interpolated boundary crossings:
the entry time is the first fix inside a zone and the exit time the last fix
inside. With 5-second sampling this is accurate to about 5 seconds.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from gps_seville.analysis import TRAJ_ID
from gps_seville.exceptions import InputFormatError
from gps_seville.logging_config import get_logger

log = get_logger(__name__)


def _read(src: str | Path | bytes, name: str) -> gpd.GeoDataFrame:
    try:
        if isinstance(src, bytes):
            import io

            return gpd.read_file(io.BytesIO(src))
        return gpd.read_file(src)
    except Exception as exc:  # pyogrio raises several types; show one message
        raise InputFormatError(f"{name}: cannot read as GeoJSON ({exc})") from exc


def load_zones(src: str | Path | bytes, name: str = "zones") -> gpd.GeoDataFrame:
    """Read polygon zones (GeoJSON). Needs a `name` property. Returned in WGS84."""
    zones = _read(src, name)
    if "name" not in zones.columns:
        raise InputFormatError(f"{name}: zones need a 'name' property")
    zones = zones[zones.geometry.notna()]
    kinds = set(zones.geom_type)
    if not kinds or not kinds <= {"Polygon", "MultiPolygon"}:
        raise InputFormatError(f"{name}: zones must be Polygon or MultiPolygon, got {kinds}")
    if zones.crs is None:
        zones = zones.set_crs("EPSG:4326")
    invalid = ~zones.is_valid
    if invalid.any():
        log.warning("%s: repaired %d invalid polygon(s)", name, int(invalid.sum()))
        zones.loc[invalid, "geometry"] = zones.loc[invalid, "geometry"].make_valid()
    if "zone_type" not in zones.columns:
        zones["zone_type"] = "zone"
    return zones.to_crs("EPSG:4326")[["name", "zone_type", "geometry"]].reset_index(drop=True)


def load_pois(src: str | Path | bytes, name: str = "points of interest") -> gpd.GeoDataFrame:
    pois = _read(src, name)
    if "name" not in pois.columns or not set(pois.geom_type) <= {"Point"}:
        raise InputFormatError(f"{name}: expected Point features with a 'name' property")
    if pois.crs is None:
        pois = pois.set_crs("EPSG:4326")
    if "category" not in pois.columns:
        pois["category"] = ""
    return pois.to_crs("EPSG:4326")[["name", "category", "geometry"]].reset_index(drop=True)


# --- geofencing -------------------------------------------------------------------------


def points_in_zones(points: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join: one row per (point, zone containing it). Points outside all zones
    are dropped. A point inside overlapping zones appears once per zone."""
    joined = gpd.sjoin(
        points[["timestamp", "vehicle_id", TRAJ_ID, "geometry"]],
        zones[["name", "geometry"]],
        how="inner",
        predicate="within",
    )
    return pd.DataFrame(joined.rename(columns={"name": "zone"}).drop(columns="index_right"))


def geofence_events(
    points: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, max_gap_s: float
) -> pd.DataFrame:
    """Visits to each zone: entry time, exit time and dwell time per trajectory.

    A visit is a run of consecutive fixes inside the zone. It ends when the next
    fix is outside, or when the time to the next fix inside is longer than
    `max_gap_s` (the vehicle may have left during a signal gap).

    `entered` is False when the trajectory already starts inside the zone, and
    `exited` is False when it ends inside, so those times are the start/end of
    observation, not real crossings.
    """
    cols = [
        "vehicle_id",
        TRAJ_ID,
        "zone",
        "enter_time",
        "exit_time",
        "dwell_min",
        "fixes",
        "entered",
        "exited",
    ]
    inside = points_in_zones(points, zones)
    if inside.empty:
        return pd.DataFrame(columns=cols)

    # position of every fix in its trajectory, to tell "consecutive" from "next fix inside"
    order = points[[TRAJ_ID, "timestamp"]].sort_values([TRAJ_ID, "timestamp"])
    order["seq"] = order.groupby(TRAJ_ID).cumcount()
    last_seq = order.groupby(TRAJ_ID)["seq"].max()
    inside = inside.merge(order, on=[TRAJ_ID, "timestamp"], how="left")

    rows = []
    for (traj_id, zone), g in inside.sort_values("seq").groupby([TRAJ_ID, "zone"]):
        new_run = (g["seq"].diff() != 1) | (g["timestamp"].diff().dt.total_seconds() > max_gap_s)
        for _, run in g.groupby(new_run.cumsum()):
            start, end = run["timestamp"].iloc[0], run["timestamp"].iloc[-1]
            rows.append(
                {
                    "vehicle_id": run["vehicle_id"].iloc[0],
                    TRAJ_ID: traj_id,
                    "zone": zone,
                    "enter_time": start,
                    "exit_time": end,
                    "dwell_min": round((end - start).total_seconds() / 60, 1),
                    "fixes": len(run),
                    "entered": bool(run["seq"].iloc[0] > 0),
                    "exited": bool(run["seq"].iloc[-1] < last_seq[traj_id]),
                }
            )
    return pd.DataFrame(rows, columns=cols).sort_values("enter_time").reset_index(drop=True)


def zone_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Visits and total time per zone and vehicle."""
    if events.empty:
        return pd.DataFrame(columns=["zone", "vehicle_id", "visits", "total_min", "longest_min"])
    return (
        events.groupby(["zone", "vehicle_id"])
        .agg(
            visits=("dwell_min", "size"),
            total_min=("dwell_min", "sum"),
            longest_min=("dwell_min", "max"),
        )
        .round(1)
        .reset_index()
        .sort_values(["zone", "total_min"], ascending=[True, False])
    )


def stops_by_zone(stops: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join of detected stops with zones (left join: stops outside get no zone)."""
    if stops.empty:
        return pd.DataFrame(columns=["vehicle_id", "start_time", "duration_min", "zone"])
    j = gpd.sjoin(stops, zones[["name", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(
        {
            "vehicle_id": j["vehicle_id"],
            "start_time": j["start_time"],
            "duration_min": (j["duration_s"] / 60).round(1),
            "zone": j["name"].fillna("(outside zones)"),
        }
    )
    return out.reset_index(drop=True)


# --- intersections, buffers and proximity -----------------------------------------------


def route_zone_intersections(
    lines: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, crs: CRS
) -> pd.DataFrame:
    """Length of each trajectory inside each zone (GeoPandas overlay), in metres and %."""
    lp = lines[[TRAJ_ID, "vehicle_id", "geometry"]].to_crs(crs)
    zp = zones[["name", "geometry"]].to_crs(crs)
    total = dict(zip(lp[TRAJ_ID], lp.length, strict=True))
    parts = gpd.overlay(lp, zp, how="intersection", keep_geom_type=True)
    if parts.empty:
        return pd.DataFrame(columns=[TRAJ_ID, "vehicle_id", "zone", "length_m", "share_pct"])
    parts["length_m"] = parts.length.round(1)
    out = parts.groupby([TRAJ_ID, "vehicle_id", "name"], as_index=False)["length_m"].sum()
    out["share_pct"] = (100 * out["length_m"] / out[TRAJ_ID].map(total)).round(1)
    return out.rename(columns={"name": "zone"}).sort_values(
        [TRAJ_ID, "length_m"], ascending=[True, False]
    )


def route_buffers(lines: gpd.GeoDataFrame, crs: CRS, distance_m: float) -> gpd.GeoDataFrame:
    """Corridor of `distance_m` around each trajectory, returned in WGS84."""
    proj = lines[[TRAJ_ID, "vehicle_id", "geometry"]].to_crs(crs)
    proj["geometry"] = proj.buffer(distance_m)
    proj["buffer_m"] = distance_m
    proj["area_ha"] = (proj.area / 10_000).round(1)
    return proj.to_crs("EPSG:4326")


def pois_near_routes(
    lines: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, crs: CRS, distance_m: float
) -> pd.DataFrame:
    """Points of interest within `distance_m` of each route, with the exact distance."""
    buffers = route_buffers(lines, crs, distance_m)
    near = gpd.sjoin(pois, buffers[[TRAJ_ID, "vehicle_id", "geometry"]], predicate="within")
    if near.empty:
        return pd.DataFrame(columns=[TRAJ_ID, "vehicle_id", "poi", "category", "distance_m"])
    lp = lines.set_index(TRAJ_ID).to_crs(crs).geometry
    near_p = near.to_crs(crs)
    dist = [lp[t].distance(g) for t, g in zip(near_p[TRAJ_ID], near_p.geometry, strict=True)]
    out = pd.DataFrame(
        {
            TRAJ_ID: near[TRAJ_ID].to_numpy(),
            "vehicle_id": near["vehicle_id"].to_numpy(),
            "poi": near["name"].to_numpy(),
            "category": near["category"].to_numpy(),
            "distance_m": [round(d, 1) for d in dist],
        }
    )
    return out.sort_values([TRAJ_ID, "distance_m"]).reset_index(drop=True)


def nearest_poi_to_stops(stops: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, crs: CRS) -> pd.DataFrame:
    """For each stop, the nearest point of interest and its distance (sjoin_nearest)."""
    if stops.empty or pois.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "start_time", "duration_min", "nearest_poi", "distance_m"]
        )
    j = gpd.sjoin_nearest(
        stops.to_crs(crs),
        pois.to_crs(crs)[["name", "geometry"]],
        how="left",
        distance_col="distance_m",
    )
    j = j[~j.index.duplicated()]  # ties: keep one
    return pd.DataFrame(
        {
            "vehicle_id": j["vehicle_id"],
            "start_time": j["start_time"],
            "duration_min": (j["duration_s"] / 60).round(1),
            "nearest_poi": j["name"],
            "distance_m": j["distance_m"].round(1),
        }
    ).reset_index(drop=True)
