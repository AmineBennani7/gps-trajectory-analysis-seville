"""Trajectory analysis with MovingPandas.

This module prepares input and collects results. The trajectory algorithms are
MovingPandas: OutlierCleaner (GPS jumps), add_speed / add_acceleration,
get_length (geodesic length on WGS84), TrajectoryStopDetector (stops),
ObservationGapSplitter and StopSplitter (journey segmentation).

My own code here: one trajectory per vehicle and day, the per-trajectory and
per-segment summary tables, and moving time / average moving speed derived
from the detected stops.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import geopandas as gpd
import pandas as pd

# MovingPandas warns about the optional Stone Soup package (smoothing), which is not used here.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Missing optional dependencies")
    import movingpandas as mpd
from movingpandas.unit_utils import UNITS

from gps_seville.config import StopConfig
from gps_seville.exceptions import InsufficientDataError
from gps_seville.logging_config import get_logger

log = get_logger(__name__)

KMH = UNITS(distance="km", time="h")
M_S2 = UNITS(distance="m", time="s", time2="s")
KM = UNITS(distance="km")
TRAJ_ID = "trajectory_id"
JUMP_REASON = "speed above {:g} km/h (MovingPandas OutlierCleaner)"


@dataclass
class TrajectoryResult:
    trajectory: mpd.Trajectory
    stops: gpd.GeoDataFrame
    outliers: pd.DataFrame
    summary: dict[str, Any]


# --- single trajectory (original API, used by the first version of the app) -------------


def build_trajectory(clean: pd.DataFrame, traj_id: str | None = None) -> mpd.Trajectory:
    """Create a MovingPandas Trajectory from cleaned records (WGS84)."""
    if len(clean) < 2:
        raise InsufficientDataError("At least two valid records are needed to build a trajectory.")
    gdf = gpd.GeoDataFrame(
        clean,
        geometry=gpd.points_from_xy(clean["longitude"], clean["latitude"]),
        crs="EPSG:4326",
    ).set_index("timestamp")
    traj_id = traj_id or str(clean["vehicle_id"].iloc[0])
    return mpd.Trajectory(gdf, traj_id)


def remove_speed_outliers(
    traj: mpd.Trajectory, max_speed_kmh: float
) -> tuple[mpd.Trajectory, pd.DataFrame]:
    """Drop GPS jumps with MovingPandas' OutlierCleaner.

    Returns (cleaned trajectory, removed rows).
    """
    cleaned = mpd.OutlierCleaner(traj).clean(v_max=max_speed_kmh, units=KMH)
    dropped = traj.df.index.difference(cleaned.df.index)
    cols = [
        c
        for c in ("row_id", "vehicle_id", TRAJ_ID, "latitude", "longitude", "accuracy_m", "source")
        if c in traj.df.columns
    ]
    outliers = traj.df.loc[dropped, cols].reset_index()
    outliers["reason"] = JUMP_REASON.format(max_speed_kmh)
    outliers["stage"] = "GPS jump removal"
    return cleaned, outliers


def detect_stops(
    traj: mpd.Trajectory | mpd.TrajectoryCollection, max_diameter_m: float, min_duration_min: float
) -> gpd.GeoDataFrame:
    """Detect stops with MovingPandas' TrajectoryStopDetector."""
    detector = mpd.TrajectoryStopDetector(traj)
    return detector.get_stop_points(
        max_diameter=max_diameter_m, min_duration=timedelta(minutes=min_duration_min)
    )


def _summary(traj: mpd.Trajectory, stops: gpd.GeoDataFrame) -> dict[str, Any]:
    duration = pd.Timedelta(traj.get_duration())
    stop_s = float(stops["duration_s"].sum()) if len(stops) else 0.0
    stop_time = pd.Timedelta(seconds=stop_s)
    moving_time = duration - stop_time
    distance_km = traj.get_length(units=KM)
    moving_h = moving_time.total_seconds() / 3600
    return {
        "points": len(traj.df),
        "start": traj.get_start_time(),
        "end": traj.get_end_time(),
        "duration": duration,
        "distance_km": distance_km,
        "stops": len(stops),
        "stop_time": stop_time,
        "moving_time": moving_time,
        "avg_moving_speed_kmh": distance_km / moving_h if moving_h > 0 else 0.0,
        "max_speed_kmh": float(traj.df["speed"].max()),
    }


def analyse(
    clean: pd.DataFrame,
    max_speed_kmh: float = 200,
    stop_diameter_m: float = 60,
    stop_min_duration_min: float = 5,
) -> TrajectoryResult:
    """Run the single-trajectory analysis on cleaned records."""
    traj = build_trajectory(clean)
    traj, outliers = remove_speed_outliers(traj, max_speed_kmh)
    traj.add_speed(overwrite=True, units=KMH)
    stops = detect_stops(traj, stop_diameter_m, stop_min_duration_min)
    return TrajectoryResult(traj, stops, outliers, _summary(traj, stops))


# --- several vehicles and days ----------------------------------------------------------


def assign_trajectory_ids(clean: pd.DataFrame) -> pd.DataFrame:
    """One trajectory per vehicle and calendar day: '<vehicle_id>_<YYYY-MM-DD>'."""
    out = clean.copy()
    out[TRAJ_ID] = out["vehicle_id"].astype(str) + "_" + out["timestamp"].dt.strftime("%Y-%m-%d")
    return out


def build_collection(clean: pd.DataFrame, min_points: int = 2) -> mpd.TrajectoryCollection:
    """Build a TrajectoryCollection (WGS84) from cleaned records."""
    if TRAJ_ID not in clean.columns:
        clean = assign_trajectory_ids(clean)
    counts = clean[TRAJ_ID].value_counts()
    keep = counts[counts >= max(min_points, 2)].index
    if len(keep) < len(counts):
        log.info(
            "Skipped %d trajectories with fewer than %d points", len(counts) - len(keep), min_points
        )
    clean = clean[clean[TRAJ_ID].isin(keep)]
    if clean.empty:
        raise InsufficientDataError(
            f"No trajectory has at least {max(min_points, 2)} valid records."
        )
    gdf = gpd.GeoDataFrame(
        clean,
        geometry=gpd.points_from_xy(clean["longitude"], clean["latitude"]),
        crs="EPSG:4326",
    ).set_index("timestamp")
    return mpd.TrajectoryCollection(gdf, traj_id_col=TRAJ_ID, obj_id_col="vehicle_id")


def remove_collection_outliers(
    tc: mpd.TrajectoryCollection, max_speed_kmh: float
) -> tuple[mpd.TrajectoryCollection, pd.DataFrame]:
    """Apply OutlierCleaner to every trajectory. Returns (collection, removed rows)."""
    cleaned, removed = [], []
    for traj in tc:
        t, out = remove_speed_outliers(traj, max_speed_kmh)
        cleaned.append(t)
        removed.append(out)
    outliers = pd.concat(removed, ignore_index=True) if removed else pd.DataFrame()
    return mpd.TrajectoryCollection(cleaned), outliers


def vehicle_of(traj: mpd.Trajectory) -> str:
    """Vehicle id of a trajectory. OutlierCleaner and the splitters do not keep obj_id,
    so it is read from the data column."""
    return str(traj.df["vehicle_id"].iloc[0])


def add_motion(tc: mpd.TrajectoryCollection) -> mpd.TrajectoryCollection:
    """Add speed (km/h) and acceleration (m/s²) columns with MovingPandas."""
    tc.add_speed(overwrite=True, units=KMH)
    tc.add_acceleration(overwrite=True, units=M_S2)
    return tc


def collection_stops(tc: mpd.TrajectoryCollection, cfg: StopConfig) -> gpd.GeoDataFrame:
    """Stops for all trajectories, with vehicle_id added."""
    stops = detect_stops(tc, cfg.max_diameter_m, cfg.min_duration_min)
    if stops.empty:
        return gpd.GeoDataFrame(
            columns=[
                "stop_id",
                "traj_id",
                "vehicle_id",
                "start_time",
                "end_time",
                "duration_s",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
    vehicles = {t.id: vehicle_of(t) for t in tc}
    stops = stops.reset_index()
    if stops.crs is None:  # MovingPandas leaves it unset for collections built from a list
        stops = stops.set_crs("EPSG:4326")
    stops["vehicle_id"] = stops["traj_id"].map(vehicles)
    return stops


def segment_journeys(
    tc: mpd.TrajectoryCollection, stop_cfg: StopConfig, max_gap_s: float
) -> mpd.TrajectoryCollection:
    """Split trajectories at observation gaps, then at stops (both MovingPandas splitters).

    Each resulting segment is one journey between two stops (or a gap).
    """
    gaps = mpd.ObservationGapSplitter(tc).split(gap=timedelta(seconds=max_gap_s))
    return mpd.StopSplitter(gaps).split(
        max_diameter=stop_cfg.max_diameter_m,
        min_duration=timedelta(minutes=stop_cfg.min_duration_min),
    )


def segment_table(segments: mpd.TrajectoryCollection) -> pd.DataFrame:
    """One row per journey segment. The splitters overwrite the trajectory_id column
    with the segment id, so the parent id (vehicle and day) is rebuilt here."""
    rows = []
    for seg in segments:
        start, end = seg.get_start_location(), seg.get_end_location()
        hours = seg.get_duration().total_seconds() / 3600
        km = seg.get_length(units=KM)
        vehicle = vehicle_of(seg)
        rows.append(
            {
                "segment_id": seg.id,
                "vehicle_id": vehicle,
                TRAJ_ID: f"{vehicle}_{pd.Timestamp(seg.get_start_time()):%Y-%m-%d}",
                "start": seg.get_start_time(),
                "end": seg.get_end_time(),
                "duration_min": round(hours * 60, 1),
                "distance_km": round(km, 3),
                "avg_speed_kmh": round(km / hours, 1) if hours > 0 else 0.0,
                "start_lat": start.y,
                "start_lon": start.x,
                "end_lat": end.y,
                "end_lon": end.x,
            }
        )
    return pd.DataFrame(rows)


def trajectory_table(tc: mpd.TrajectoryCollection, stops: gpd.GeoDataFrame) -> pd.DataFrame:
    """Summary per trajectory (vehicle and day)."""
    rows = []
    for traj in tc:
        own = stops[stops["traj_id"] == traj.id] if len(stops) else stops
        s = _summary(traj, own)
        acc = traj.df["acceleration"] if "acceleration" in traj.df else pd.Series(dtype=float)
        rows.append(
            {
                TRAJ_ID: traj.id,
                "vehicle_id": vehicle_of(traj),
                "date": s["start"].date(),
                "start": s["start"],
                "end": s["end"],
                "points": s["points"],
                "distance_km": round(s["distance_km"], 3),
                "duration_min": round(s["duration"].total_seconds() / 60, 1),
                "moving_min": round(s["moving_time"].total_seconds() / 60, 1),
                "stops": s["stops"],
                "stop_min": round(s["stop_time"].total_seconds() / 60, 1),
                "avg_moving_speed_kmh": round(s["avg_moving_speed_kmh"], 1),
                "max_speed_kmh": round(s["max_speed_kmh"], 1),
                "max_accel_ms2": round(float(acc.max()), 2) if len(acc) else None,
                "max_decel_ms2": round(float(acc.min()), 2) if len(acc) else None,
            }
        )
    return pd.DataFrame(rows)


def points_gdf(tc: mpd.TrajectoryCollection) -> gpd.GeoDataFrame:
    """All processed points (with speed/acceleration) as one GeoDataFrame in WGS84."""
    frames = []
    for traj in tc:
        df = traj.df.copy()
        df[TRAJ_ID] = traj.id
        frames.append(df)
    gdf = pd.concat(frames)
    gdf.index.name = "timestamp"
    return gpd.GeoDataFrame(gdf.reset_index(), geometry="geometry", crs="EPSG:4326")


def lines_gdf(tc: mpd.TrajectoryCollection) -> gpd.GeoDataFrame:
    """One LineString per trajectory (WGS84), from MovingPandas to_traj_gdf."""
    lines = tc.to_traj_gdf()
    lines = lines.rename(columns={"traj_id": TRAJ_ID})
    vehicles = {t.id: vehicle_of(t) for t in tc}
    lines["vehicle_id"] = lines[TRAJ_ID].map(vehicles)
    return lines


def position_at(traj: mpd.Trajectory, t: pd.Timestamp) -> tuple[float, float, bool]:
    """(lat, lon, observed) at time t.

    If t is the time of a GPS fix, that fix is returned with observed=True.
    Otherwise MovingPandas interpolates linearly between the fixes before and
    after t (observed=False): an estimate, not a measurement.
    """
    if not traj.get_start_time() <= t <= traj.get_end_time():
        raise InsufficientDataError(f"{t} is outside trajectory {traj.id}")
    observed = t in traj.df.index
    point = traj.get_position_at(t, method="nearest" if observed else "interpolated")
    return point.y, point.x, observed
