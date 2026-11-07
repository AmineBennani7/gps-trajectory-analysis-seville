from datetime import date, datetime, time

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from gps_seville import analytics as an
from gps_seville.pipeline import run_pipeline
from tests.conftest import make_track

UTM = "EPSG:25830"


def _lines(*coords_by_id):
    rows = [
        {"trajectory_id": tid, "vehicle_id": tid[0], "geometry": LineString(c)}
        for tid, c in coords_by_id
    ]
    return gpd.GeoDataFrame(rows, crs=UTM)


def test_similarity_identical_and_parallel_routes():
    base = [(0, 0), (1000, 0)]
    lines = _lines(("A1", base), ("A2", base), ("B1", [(0, 30), (1000, 30)]))
    s = an.route_similarity(lines, UTM, buffer_m=50).set_index(["trajectory_a", "trajectory_b"])
    same = s.loc[("A1", "A2")]
    assert same["hausdorff_m"] == 0 and same["overlap_a_pct"] == 100 and same["same_vehicle"]
    parallel = s.loc[("A1", "B1")]
    assert parallel["hausdorff_m"] == pytest.approx(30)
    assert parallel["overlap_a_pct"] == 100  # within the 50 m buffer
    assert not parallel["same_vehicle"]


def test_frechet_detects_opposite_direction():
    """Same geometry driven the other way: Hausdorff 0, Fréchet large."""
    lines = _lines(("A1", [(0, 0), (500, 0), (1000, 0)]), ("B1", [(1000, 0), (500, 0), (0, 0)]))
    row = an.route_similarity(lines, UTM, 50).iloc[0]
    assert row["hausdorff_m"] == 0
    assert row["frechet_m"] == pytest.approx(1000)


def test_partial_overlap_share():
    lines = _lines(("A1", [(0, 0), (1000, 0)]), ("B1", [(0, 0), (400, 0), (400, 1000)]))
    row = an.route_similarity(lines, UTM, 10).iloc[0]
    assert row["overlap_a_pct"] == pytest.approx(40, abs=1)


def test_same_route_on_two_days_is_most_similar(fleet_result):
    s = an.route_similarity(fleet_result.lines, fleet_result.projected_crs, 50)
    best = s.sort_values("frechet_m").iloc[0]
    assert {best["trajectory_a"], best["trajectory_b"]} == {
        "SEV-VEH-01_2026-03-10",
        "SEV-VEH-01_2026-03-11",
    }
    assert best["frechet_m"] < 50 and best["overlap_a_pct"] > 95


def test_frequent_locations_cluster_stops(fleet_result):
    loc = an.frequent_locations(fleet_result.stops, fleet_result.projected_crs, 150, 2)
    # Triana and Nervión are visited by two vehicles each
    top = loc[loc["vehicles"] == 2]
    assert len(top) == 2
    triana = gpd.GeoSeries([Point(-6.0034, 37.3851)], crs=4326).to_crs(UTM).iloc[0]
    assert top.to_crs(UTM).distance(triana).min() < 50
    assert (loc["visits"] >= 2).all()


def test_frequent_locations_empty_input(fleet_result):
    empty = fleet_result.stops.iloc[0:0]
    assert an.frequent_locations(empty, fleet_result.projected_crs, 150, 2).empty


def test_robust_z():
    z = an.robust_z(pd.Series([10, 11, 9, 10, 12, 8, 10, 50.0]))
    assert z.iloc[-1] > 10 and z.iloc[:-1].abs().max() < 2
    assert (an.robust_z(pd.Series([5.0] * 10)) == 0).all()


def test_speeding_van_is_flagged(fleet_result):
    a = an.detect_anomalies(fleet_result.points, 3.5)
    speed = a[a["metric"] == "speed_kmh"]
    assert set(speed["vehicle_id"]) == {"SEV-VAN-02"}
    assert speed["value"].min() > 60
    assert (speed["timestamp"].dt.strftime("%H:%M") <= "09:13").all()


def test_constant_speed_has_no_anomalies():
    r = run_pipeline(make_track(n=60))
    assert an.detect_anomalies(r.points, 3.5).empty


def test_filters(fleet_result):
    p = fleet_result.points
    assert set(an.filter_points(p, vehicles=["SEV-VAN-02"])["vehicle_id"]) == {"SEV-VAN-02"}
    day2 = an.filter_points(p, dates=(date(2026, 3, 11), date(2026, 3, 11)))
    assert set(day2["timestamp"].dt.date) == {date(2026, 3, 11)}
    morning = an.filter_points(p, hours=(time(9, 0), time(10, 0)))
    assert morning["timestamp"].dt.hour.isin([9, 10]).all()
    triana = box(-6.006, 37.383, -6.001, 37.387)
    inside = an.filter_points(p, area=triana)
    assert len(inside) > 0 and inside.within(triana).all()


def test_time_filter_wraps_midnight():
    pts = gpd.GeoDataFrame(
        {"vehicle_id": "A", "timestamp": [datetime(2026, 3, 10, h) for h in (1, 12, 23)]},
        geometry=[Point(0, 0)] * 3,
        crs=4326,
    )
    night = an.filter_points(pts, hours=(time(22), time(6)))
    assert night["timestamp"].dt.hour.tolist() == [1, 23]
