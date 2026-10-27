from datetime import datetime, timedelta
from pathlib import Path

import folium
import pandas as pd
import pytest

from gps_seville.analysis import analyse
from gps_seville.cleaning import clean_records, load_records
from gps_seville.mapping import build_map

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "vehicle_gps_seville.csv"
METERS_PER_DEG_LAT = 111_320.0


def straight_track(stop_minutes=0, jump=False):
    """Northbound track: 1 km at ~36 km/h, optional stop in the middle."""
    t0 = datetime(2026, 3, 10, 8, 0, 0)
    lat, lon = 37.38, -5.99
    step = 50 / METERS_PER_DEG_LAT  # 50 m every 5 s
    rows, t = [], t0
    for i in range(21):
        rows.append((t, lat + i * step, lon))
        t += timedelta(seconds=5)
        if i == 10:
            for _ in range(stop_minutes * 12):
                rows.append((t, lat + i * step, lon))
                t += timedelta(seconds=5)
    df = pd.DataFrame(rows, columns=["timestamp", "latitude", "longitude"])
    if jump:
        df.loc[5, ["latitude", "longitude"]] = [37.418, -5.893]
    df["vehicle_id"] = "T1"
    return df


def test_distance_matches_known_track():
    result = analyse(straight_track())
    assert result.summary["distance_km"] == pytest.approx(1.0, rel=0.01)
    assert result.summary["max_speed_kmh"] == pytest.approx(36, rel=0.02)


def test_stop_is_detected():
    result = analyse(straight_track(stop_minutes=8), stop_diameter_m=30, stop_min_duration_min=5)
    assert result.summary["stops"] == 1
    assert result.stops["duration_s"].iloc[0] == pytest.approx(8 * 60, abs=10)


def test_short_pause_is_not_a_stop():
    result = analyse(straight_track(stop_minutes=2), stop_diameter_m=30, stop_min_duration_min=5)
    assert result.summary["stops"] == 0


def test_gps_jump_is_removed():
    result = analyse(straight_track(jump=True), max_speed_kmh=200)
    assert len(result.outliers) == 1
    assert result.summary["distance_km"] == pytest.approx(1.0, rel=0.01)


def test_sample_file_end_to_end():
    raw = load_records(SAMPLE)
    clean, removed = clean_records(raw)
    result = analyse(clean)
    assert len(removed) == 7
    assert len(result.outliers) == 2
    assert result.summary["stops"] == 3
    assert 8.5 < result.summary["distance_km"] < 11
    assert isinstance(build_map(result, removed), folium.Map)
