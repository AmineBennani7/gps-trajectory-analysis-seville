"""Filtered views of a PipelineResult, used by the dashboard.

The pipeline always runs on all records. Filters (vehicle, dates, time of day,
zone) are applied afterwards to what is shown and to the spatial analyses, so
changing a filter does not re-run cleaning or stop detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from gps_seville.analysis import TRAJ_ID
from gps_seville.analytics import filter_points
from gps_seville.cleaning import REASON_AREA, REASON_COORDINATE, REASON_RANGE, REASON_TIMESTAMP
from gps_seville.pipeline import PipelineResult

# Okabe-Ito colours (distinguishable with common colour-vision deficiencies)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


@dataclass(frozen=True)
class Filters:
    vehicles: tuple[str, ...] = ()
    dates: tuple[date, date] | None = None
    hours: tuple[time, time] | None = None
    zone: str | None = None


@dataclass
class View:
    points: gpd.GeoDataFrame
    lines: gpd.GeoDataFrame
    stops: gpd.GeoDataFrame
    trajectories: pd.DataFrame
    segments: pd.DataFrame
    colours: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.points.empty


def trajectory_colours(trajectory_ids: list[str]) -> dict[str, str]:
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(sorted(trajectory_ids))}


def lines_from_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One LineString per trajectory from (filtered) points, in time order."""
    rows = []
    for traj_id, g in points.sort_values("timestamp").groupby(TRAJ_ID):
        if len(g) < 2:
            continue
        rows.append(
            {
                TRAJ_ID: traj_id,
                "vehicle_id": g["vehicle_id"].iloc[0],
                "start_t": g["timestamp"].iloc[0],
                "end_t": g["timestamp"].iloc[-1],
                "geometry": LineString(zip(g.geometry.x, g.geometry.y, strict=True)),
            }
        )
    cols = [TRAJ_ID, "vehicle_id", "start_t", "end_t", "geometry"]
    return gpd.GeoDataFrame(rows, columns=cols, geometry="geometry", crs="EPSG:4326")


def apply_filters(
    result: PipelineResult, filters: Filters, zones: gpd.GeoDataFrame | None = None
) -> View:
    area: BaseGeometry | None = None
    if filters.zone and zones is not None:
        match = zones[zones["name"] == filters.zone]
        if not match.empty:
            area = match.geometry.iloc[0]

    pts = filter_points(
        result.points, list(filters.vehicles) or None, filters.dates, filters.hours, area
    )
    traj_ids = set(pts[TRAJ_ID])

    stops = result.stops
    if not stops.empty:
        stops = stops[stops["traj_id"].isin(traj_ids)]
        st = gpd.GeoDataFrame(stops.assign(timestamp=stops["start_time"]), crs=stops.crs)
        st = filter_points(st, None, filters.dates, filters.hours, area)
        stops = stops.loc[st.index]

    seg = result.segment_table
    if not seg.empty:
        seg = seg[seg[TRAJ_ID].isin(traj_ids)]
        if filters.dates:
            d = seg["start"].dt.date
            seg = seg[(d >= filters.dates[0]) & (d <= filters.dates[1])]

    trajectories = result.trajectory_table[result.trajectory_table[TRAJ_ID].isin(traj_ids)]
    return View(
        points=pts,
        lines=lines_from_points(pts),
        stops=stops,
        trajectories=trajectories,
        segments=seg,
        colours=trajectory_colours(sorted(result.trajectory_table[TRAJ_ID])),
    )


def raw_track(result: PipelineResult) -> gpd.GeoDataFrame:
    """Raw fixes with a time and coordinates inside the study area, before any cleaning.

    Used to draw the "before cleaning" track next to the cleaned one: it still
    contains GPS jumps, duplicates and low-accuracy fixes.
    """
    from gps_seville.analysis import assign_trajectory_ids
    from gps_seville.cleaning import parse_values, rejection_reasons

    parsed = parse_values(result.raw)
    reason = rejection_reasons(parsed, result.config.cleaning)
    drop = reason.isin([REASON_TIMESTAMP, REASON_COORDINATE, REASON_RANGE, REASON_AREA])
    keep = parsed[~drop].copy()
    if result.config.cleaning.reject_zero_coordinates:
        keep = keep[~((keep["latitude"] == 0) & (keep["longitude"] == 0))]
    keep = assign_trajectory_ids(keep).sort_values(["vehicle_id", "timestamp"])
    return gpd.GeoDataFrame(
        keep, geometry=gpd.points_from_xy(keep["longitude"], keep["latitude"]), crs="EPSG:4326"
    )
