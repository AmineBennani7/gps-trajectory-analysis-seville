"""Analytics on top of the MovingPandas results (my own code).

Libraries used: scikit-learn DBSCAN for clustering stops, Shapely for the
Hausdorff and discrete Fréchet distances and buffers. The anomaly rule
(robust z-score) and the filters are written here.

All distances are computed in the projected CRS passed in (metres).
"""

from __future__ import annotations

import itertools
from datetime import date, time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import DBSCAN

from gps_seville.analysis import TRAJ_ID

# --- frequently visited locations -------------------------------------------------------


def frequent_locations(
    stops: gpd.GeoDataFrame, crs: CRS, eps_m: float, min_visits: int
) -> gpd.GeoDataFrame:
    """Group stops that lie within `eps_m` of each other (DBSCAN) into locations.

    Returns one row per location (cluster) with the number of visits, distinct
    vehicles, total and mean dwell time; the geometry is the mean stop position
    in WGS84. Stops that do not belong to any cluster are left out.
    """
    cols = [
        "location_id",
        "visits",
        "vehicles",
        "vehicle_ids",
        "total_dwell_min",
        "mean_dwell_min",
        "first_visit",
        "last_visit",
        "geometry",
    ]
    if len(stops) < min_visits:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

    proj = stops.to_crs(crs)
    xy = np.column_stack([proj.geometry.x, proj.geometry.y])
    labels = DBSCAN(eps=eps_m, min_samples=min_visits).fit_predict(xy)
    proj = proj.assign(cluster=labels)
    proj = proj[proj["cluster"] >= 0]

    rows = []
    for cid, g in proj.groupby("cluster"):
        rows.append(
            {
                "location_id": int(cid) + 1,
                "visits": len(g),
                "vehicles": g["vehicle_id"].nunique(),
                "vehicle_ids": ", ".join(sorted(g["vehicle_id"].unique())),
                "total_dwell_min": round(g["duration_s"].sum() / 60, 1),
                "mean_dwell_min": round(g["duration_s"].mean() / 60, 1),
                "first_visit": g["start_time"].min(),
                "last_visit": g["end_time"].max(),
                "geometry": shapely.Point(g.geometry.x.mean(), g.geometry.y.mean()),
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_crs("EPSG:4326")
    return out.sort_values("visits", ascending=False).reset_index(drop=True)


# --- unusual movements ------------------------------------------------------------------


def robust_z(values: pd.Series) -> pd.Series:
    """(x - median) / (1.4826 * MAD). Robust to the outliers it is meant to find.

    Returns zeros if the MAD is 0 (more than half of the values are equal).
    """
    med = values.median()
    mad = (values - med).abs().median()
    if np.isnan(mad) or mad <= 1e-9 * max(1.0, abs(med)):  # no spread (or rounding noise)
        return pd.Series(0.0, index=values.index)
    return (values - med) / (1.4826 * mad)


def detect_anomalies(
    points: gpd.GeoDataFrame, z_threshold: float, moving_kmh: float = 5.0
) -> gpd.GeoDataFrame:
    """Flag points with unusually high speed or harsh acceleration/braking for that vehicle.

    Both metrics are only compared among moving points (speed above
    `moving_kmh` at the point or the one before it). Parked fixes have speed and
    acceleration close to 0; including them makes the spread tiny and every
    normal start or stop looks unusual. Each flagged point gets the metric, its
    value and its robust z-score.
    """
    found = []
    for _vehicle, g in points.sort_values("timestamp").groupby("vehicle_id"):
        moving = g[(g["speed"] > moving_kmh) | (g["speed"].shift() > moving_kmh)]
        # (metric, values, two_sided): low speeds are not unusual, harsh braking is.
        checks = [
            ("speed_kmh", moving["speed"], False),
            ("acceleration_ms2", moving["acceleration"], True),
        ]
        for metric, values, two_sided in checks:
            values = values.dropna()
            if len(values) < 10:
                continue
            z = robust_z(values)
            hit = (z.abs() if two_sided else z) > z_threshold
            if hit.any():
                sel = g.loc[hit[hit].index].copy()
                sel["metric"] = metric
                sel["value"] = values[hit].round(2)
                sel["z_score"] = z[hit].round(1)
                found.append(sel)
    cols = ["timestamp", "vehicle_id", TRAJ_ID, "metric", "value", "z_score", "geometry"]
    if not found:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=points.crs)
    out = pd.concat(found)
    return gpd.GeoDataFrame(out[cols], geometry="geometry", crs=points.crs).sort_values("timestamp")


# --- route similarity -------------------------------------------------------------------


def _overlap_share(a: BaseGeometry, b: BaseGeometry, buffer_m: float) -> float:
    """Share of line `a`'s length that lies within `buffer_m` of line `b`.

    `a` is cut into pieces no longer than half the buffer; a piece counts as
    inside if its midpoint is within the distance. This matches an exact buffer
    intersection closely and is much faster on noisy GPS lines.
    """
    xy = shapely.get_coordinates(shapely.segmentize(a, max(buffer_m / 2, 1.0)))
    if len(xy) < 2:
        return 0.0
    seg_len = np.hypot(*np.diff(xy, axis=0).T)
    total = seg_len.sum()
    if total == 0:
        return 0.0
    mids = shapely.points((xy[1:] + xy[:-1]) / 2)
    shapely.prepare(b)
    inside = shapely.dwithin(mids, b, buffer_m)
    return float(seg_len[inside].sum() / total)


def route_similarity(lines: gpd.GeoDataFrame, crs: CRS, buffer_m: float) -> pd.DataFrame:
    """Pairwise comparison of trajectory lines.

    hausdorff_m: largest distance from any point of one route to the other route.
    frechet_m: discrete Fréchet distance over the GPS fixes; it respects the direction and
      order of travel (two routes driven in opposite directions score high).
    overlap_a_pct / overlap_b_pct: share of each route within `buffer_m` of the other.
    """
    proj = lines.to_crs(crs)
    geoms = dict(zip(proj[TRAJ_ID], proj.geometry, strict=True))
    vehicles = dict(zip(proj[TRAJ_ID], proj["vehicle_id"], strict=True))
    rows = []
    for a, b in itertools.combinations(sorted(geoms), 2):
        ga, gb = geoms[a], geoms[b]
        rows.append(
            {
                "trajectory_a": a,
                "trajectory_b": b,
                "same_vehicle": vehicles[a] == vehicles[b],
                "hausdorff_m": round(shapely.hausdorff_distance(ga, gb), 1),
                "frechet_m": round(shapely.frechet_distance(ga, gb), 1),
                "overlap_a_pct": round(100 * _overlap_share(ga, gb, buffer_m), 1),
                "overlap_b_pct": round(100 * _overlap_share(gb, ga, buffer_m), 1),
            }
        )
    return pd.DataFrame(rows)


# --- filters ----------------------------------------------------------------------------


def filter_points(
    points: gpd.GeoDataFrame,
    vehicles: list[str] | None = None,
    dates: tuple[date, date] | None = None,
    hours: tuple[time, time] | None = None,
    area: BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """Select points by vehicle, date range (inclusive), time of day and area (WGS84).

    The time-of-day window may wrap past midnight (e.g. 22:00-06:00).
    """
    mask = pd.Series(True, index=points.index)
    if vehicles:
        mask &= points["vehicle_id"].isin(vehicles)
    if dates:
        d = points["timestamp"].dt.date
        mask &= (d >= dates[0]) & (d <= dates[1])
    if hours:
        t = points["timestamp"].dt.time
        start, end = hours
        mask &= (t >= start) & (t <= end) if start <= end else (t >= start) | (t <= end)
    if area is not None:
        mask &= points.within(area)
    return points[mask]
