"""GPS data quality indicators (my own code).

About accuracy: consumer GPS positions typically have a horizontal error of a
few metres in open sky and tens of metres near tall buildings or trees. The
synthetic data adds random errors of that size (standard deviation per axis
of about 1 m when parked, 3 m when moving, 20-35 m in one degraded block of
the fleet file). Nothing in this project measures or claims sub-metre accuracy.

About distance: summing straight lines between noisy positions overestimates
the distance travelled, because random errors add sideways zig-zags. The effect
is largest when the vehicle is parked: it moves 0 m but every new fix lands a
metre or two away from the last one. `distance_comparison` therefore reports
the distance accumulated during detected stops separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from gps_seville.analysis import TRAJ_ID
from gps_seville.cleaning import REASON_COORDINATE, REASON_RANGE, REASON_TIMESTAMP
from gps_seville.crs import path_length_m


@dataclass
class QualityReport:
    total_records: int
    valid_records: int
    rejections: pd.DataFrame  # stage, reason, records, share_pct
    per_vehicle: pd.DataFrame
    sampling: pd.DataFrame  # per trajectory
    distances: pd.DataFrame  # per trajectory

    @property
    def rejected_records(self) -> int:
        return self.total_records - self.valid_records

    @property
    def valid_share(self) -> float:
        return self.valid_records / self.total_records if self.total_records else 0.0


def rejection_summary(rejected: pd.DataFrame, total: int) -> pd.DataFrame:
    if rejected.empty:
        return pd.DataFrame(columns=["stage", "reason", "records", "share_pct"])
    out = rejected.groupby(["stage", "reason"], sort=False).size().rename("records").reset_index()
    out["share_pct"] = (100 * out["records"] / max(total, 1)).round(2)
    return out


def per_vehicle_quality(
    raw: pd.DataFrame, rejected: pd.DataFrame, cleaned: pd.DataFrame
) -> pd.DataFrame:
    total = raw.groupby("vehicle_id").size().rename("records")
    rej = rejected.groupby("vehicle_id").size().rename("rejected")
    out = pd.concat([total, rej], axis=1).fillna(0).astype(int)
    out["valid"] = out["records"] - out["rejected"]
    out["valid_pct"] = (100 * out["valid"] / out["records"]).round(1)
    if "accuracy_m" in cleaned.columns and cleaned["accuracy_m"].notna().any():
        out["median_accuracy_m"] = cleaned.groupby("vehicle_id")["accuracy_m"].median().round(1)
    return out.reset_index()


def sampling_quality(points: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
    """Sampling interval statistics per trajectory from the cleaned points."""
    rows = []
    for traj_id, g in points.sort_values("timestamp").groupby(TRAJ_ID):
        dt = g["timestamp"].diff().dt.total_seconds().dropna()
        rows.append(
            {
                TRAJ_ID: traj_id,
                "vehicle_id": g["vehicle_id"].iloc[0],
                "points": len(g),
                "median_interval_s": float(dt.median()) if len(dt) else None,
                "max_gap_s": float(dt.max()) if len(dt) else None,
                "gaps_over_threshold": int((dt > max_gap_s).sum()),
            }
        )
    return pd.DataFrame(rows)


def _orderable(parsed: pd.DataFrame) -> pd.DataFrame:
    """Records that have a time and coordinates in the valid WGS84 range."""
    bad = {REASON_TIMESTAMP, REASON_COORDINATE, REASON_RANGE}
    ok = parsed["_reason"].isna() | ~parsed["_reason"].isin(bad)
    return parsed[ok]


def _length_by_traj(df: pd.DataFrame, crs: CRS) -> pd.Series:
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(crs)
    gdf = gdf.sort_values("timestamp")
    return gdf.groupby(TRAJ_ID)["geometry"].apply(lambda g: path_length_m(g) / 1000)


def _stationary_km(points: gpd.GeoDataFrame, stops: pd.DataFrame, crs: CRS) -> pd.Series:
    """Distance accumulated between consecutive fixes that both fall inside a stop."""
    proj = points.to_crs(crs).sort_values("timestamp")
    out = {}
    for traj_id, g in proj.groupby(TRAJ_ID):
        own = stops[stops["traj_id"] == traj_id] if len(stops) else stops
        if own.empty:
            out[traj_id] = 0.0
            continue
        t = g["timestamp"]
        inside = pd.Series(False, index=g.index)
        for s in own.itertuples():
            inside |= (t >= s.start_time) & (t <= s.end_time)
        step = g.geometry.distance(g.geometry.shift())
        both = inside & inside.shift(fill_value=False)
        out[traj_id] = float(step[both].sum()) / 1000
    return pd.Series(out, name="stationary_km")


def distance_comparison(
    parsed: pd.DataFrame,
    after_checks: pd.DataFrame,
    cleaned_points: gpd.GeoDataFrame,
    stops: pd.DataFrame,
    crs: CRS,
) -> pd.DataFrame:
    """Distance per trajectory at each cleaning stage, in km, in a projected CRS.

    raw_km: every record with a time and in-range coordinates, in time order
    (includes zero coordinates, far-away points, duplicates and jumps).
    after_record_checks_km: after cleaning.clean_records.
    cleaned_km: after GPS jump removal (the distance shown in the dashboard).
    stationary_km: part of cleaned_km collected while stopped (GPS jitter).
    moving_km: cleaned_km - stationary_km, an estimate of distance actually driven.
    """
    from gps_seville.analysis import assign_trajectory_ids

    raw = assign_trajectory_ids(_orderable(parsed))
    out = pd.concat(
        [
            _length_by_traj(raw, crs).rename("raw_km"),
            _length_by_traj(assign_trajectory_ids(after_checks), crs).rename(
                "after_record_checks_km"
            ),
            _length_by_traj(cleaned_points, crs).rename("cleaned_km"),
            _stationary_km(cleaned_points, stops, crs),
        ],
        axis=1,
    )
    out = out[out["cleaned_km"].notna()].fillna(0.0)
    out["moving_km"] = out["cleaned_km"] - out["stationary_km"]
    out["jitter_share_pct"] = (100 * out["stationary_km"] / out["cleaned_km"]).round(1)
    out.index.name = TRAJ_ID
    return out.round(3).reset_index()
