"""Trajectory analysis with MovingPandas.

This module only prepares the input and collects results. Distance, speed,
outlier removal and stop detection are done by MovingPandas.
"""

import warnings
from dataclasses import dataclass
from datetime import timedelta

import geopandas as gpd

# MovingPandas warns about the optional Stone Soup package (smoothing), which is not used here.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Missing optional dependencies")
    import movingpandas as mpd
import pandas as pd
from movingpandas.unit_utils import UNITS

KMH = UNITS(distance="km", time="h")


@dataclass
class TrajectoryResult:
    trajectory: mpd.Trajectory
    stops: gpd.GeoDataFrame
    outliers: pd.DataFrame
    summary: dict


def build_trajectory(clean, traj_id=None):
    """Create a MovingPandas Trajectory from cleaned records (WGS84)."""
    if len(clean) < 2:
        raise ValueError("At least two valid records are needed to build a trajectory.")
    gdf = gpd.GeoDataFrame(
        clean,
        geometry=gpd.points_from_xy(clean["longitude"], clean["latitude"]),
        crs="EPSG:4326",
    ).set_index("timestamp")
    traj_id = traj_id or str(clean["vehicle_id"].iloc[0])
    return mpd.Trajectory(gdf, traj_id)


def remove_speed_outliers(traj, max_speed_kmh):
    """Drop GPS jumps with MovingPandas' OutlierCleaner.

    Returns (cleaned trajectory, removed rows).
    """
    cleaned = mpd.OutlierCleaner(traj).clean(v_max=max_speed_kmh, units=KMH)
    dropped = traj.df.index.difference(cleaned.df.index)
    outliers = traj.df.loc[dropped, ["vehicle_id", "latitude", "longitude"]].reset_index()
    outliers["reason"] = f"speed above {max_speed_kmh:g} km/h (MovingPandas OutlierCleaner)"
    return cleaned, outliers


def detect_stops(traj, max_diameter_m, min_duration_min):
    """Detect stops with MovingPandas' TrajectoryStopDetector."""
    detector = mpd.TrajectoryStopDetector(traj)
    stops = detector.get_stop_points(
        max_diameter=max_diameter_m, min_duration=timedelta(minutes=min_duration_min)
    )
    return stops


def analyse(clean, max_speed_kmh=200, stop_diameter_m=60, stop_min_duration_min=5):
    """Run the full analysis on cleaned records."""
    traj = build_trajectory(clean)
    traj, outliers = remove_speed_outliers(traj, max_speed_kmh)
    traj.add_speed(overwrite=True, units=KMH)
    stops = detect_stops(traj, stop_diameter_m, stop_min_duration_min)

    duration = traj.get_duration()
    stop_time = pd.to_timedelta(stops["duration_s"].sum(), unit="s") if len(stops) else timedelta(0)
    moving_time = duration - stop_time
    distance_km = traj.get_length(units=UNITS(distance="km"))

    summary = {
        "points": len(traj.df),
        "start": traj.get_start_time(),
        "end": traj.get_end_time(),
        "duration": duration,
        "distance_km": distance_km,
        "stops": len(stops),
        "stop_time": stop_time,
        "moving_time": moving_time,
        "avg_moving_speed_kmh": distance_km / (moving_time.total_seconds() / 3600)
        if moving_time.total_seconds() > 0 else 0.0,
        "max_speed_kmh": float(traj.df["speed"].max()),
    }
    return TrajectoryResult(traj, stops, outliers, summary)
