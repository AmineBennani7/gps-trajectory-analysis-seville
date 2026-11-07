"""Shared fixtures and helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from gps_seville.pipeline import PipelineResult, run_pipeline
from gps_seville.readers import read_files

DATA = Path(__file__).resolve().parents[1] / "data"
SINGLE = DATA / "vehicle_gps_seville.csv"
FLEET = DATA / "fleet_gps_seville.csv"
ZONES = DATA / "zones_seville.geojson"
POIS = DATA / "poi_seville.geojson"
METERS_PER_DEG_LAT = 111_320.0


def make_track(
    vehicle: str = "T1",
    start: datetime = datetime(2026, 3, 10, 8, 0, 0),
    lat: float = 37.38,
    lon: float = -5.99,
    n: int = 21,
    step_m: float = 50,
    interval_s: int = 5,
    stop_after: int | None = None,
    stop_minutes: int = 0,
    east: bool = False,
) -> pd.DataFrame:
    """Straight track in Seville: `n` points `step_m` apart every `interval_s` seconds.

    Northbound by default, eastbound with east=True. With stop_after=i the vehicle
    waits `stop_minutes` at point i.
    """
    dlat = 0 if east else step_m / METERS_PER_DEG_LAT
    dlon = step_m / (METERS_PER_DEG_LAT * 0.7944) if east else 0  # cos(37.4°) ≈ 0.794
    rows, t = [], start
    for i in range(n):
        rows.append((vehicle, t, lat + i * dlat, lon + i * dlon))
        t += timedelta(seconds=interval_s)
        if stop_after == i:
            for _ in range(stop_minutes * 60 // interval_s):
                rows.append((vehicle, t, lat + i * dlat, lon + i * dlon))
                t += timedelta(seconds=interval_s)
    return pd.DataFrame(rows, columns=["vehicle_id", "timestamp", "latitude", "longitude"])


@pytest.fixture(scope="session")
def fleet_result() -> PipelineResult:
    return run_pipeline(read_files([(FLEET.name, FLEET)]))


@pytest.fixture(scope="session")
def single_result() -> PipelineResult:
    return run_pipeline(read_files([(SINGLE.name, SINGLE)]))
