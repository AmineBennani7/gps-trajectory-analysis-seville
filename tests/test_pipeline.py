from dataclasses import replace

import pandas as pd
import pytest

from gps_seville.config import AppConfig, CleaningConfig
from gps_seville.exceptions import InsufficientDataError
from gps_seville.pipeline import run_pipeline
from tests.conftest import make_track


def test_single_vehicle_matches_original_analysis(single_result):
    """The pipeline gives the same figures as the first version on the original file."""
    r = single_result
    assert r.quality.total_records == 661
    assert r.quality.valid_records == 652
    row = r.trajectory_table.iloc[0]
    assert row["vehicle_id"] == "SEV-VEH-01"
    assert row["stops"] == 3
    assert row["distance_km"] == pytest.approx(9.67, abs=0.02)


def test_raw_cleaned_and_processed_are_separate(fleet_result):
    r = fleet_result
    assert len(r.raw) == r.quality.total_records == 2046
    assert len(r.cleaned) + len(r.rejected) == len(r.raw)
    assert "speed" not in r.cleaned.columns  # cleaned data has no derived values
    assert {"speed", "acceleration"} <= set(r.points.columns)
    assert set(r.rejected["row_id"]).isdisjoint(r.cleaned["row_id"])


def test_fleet_rejections_by_reason(fleet_result):
    counts = fleet_result.quality.rejections.set_index("reason")["records"].to_dict()
    assert counts["reported accuracy too low"] == 8
    assert counts["duplicate timestamp"] == 2
    assert counts["missing or unreadable timestamp"] == 1
    assert counts["speed above 200 km/h (MovingPandas OutlierCleaner)"] == 2


def test_one_trajectory_per_vehicle_and_day(fleet_result):
    t = fleet_result.trajectory_table
    assert sorted(t["trajectory_id"]) == [
        "SEV-TAXI-03_2026-03-11",
        "SEV-VAN-02_2026-03-10",
        "SEV-VEH-01_2026-03-10",
        "SEV-VEH-01_2026-03-11",
    ]
    assert fleet_result.vehicles == ["SEV-TAXI-03", "SEV-VAN-02", "SEV-VEH-01"]


def test_signal_gap_is_reported_and_splits_journey(fleet_result):
    s = fleet_result.quality.sampling.set_index("trajectory_id")
    assert s.loc["SEV-VAN-02_2026-03-10", "gaps_over_threshold"] == 1
    seg = fleet_result.segment_table
    # van: 2 stops and 1 gap -> 4 journeys
    assert (seg["vehicle_id"] == "SEV-VAN-02").sum() == 4


def test_distance_stages(fleet_result):
    d = fleet_result.quality.distances.set_index("trajectory_id")
    row = d.loc["SEV-VEH-01_2026-03-11"]  # day 2 has one GPS jump
    assert row["raw_km"] > row["cleaned_km"] + 5
    assert row["stationary_km"] > 0
    assert row["moving_km"] == pytest.approx(row["cleaned_km"] - row["stationary_km"], abs=0.002)
    # the projected length agrees with MovingPandas' geodesic length
    t = fleet_result.trajectory_table.set_index("trajectory_id")
    assert row["cleaned_km"] == pytest.approx(
        t.loc["SEV-VEH-01_2026-03-11", "distance_km"], rel=0.005
    )


def test_acceleration_on_known_track():
    """Constant speed except a stop: acceleration is ~0 while cruising."""
    r = run_pipeline(make_track(n=40, stop_after=20, stop_minutes=6))
    p = r.points
    cruising = p[(p["speed"] > 30) & (p["speed"].shift() > 30) & (p["speed"].shift(-1) > 30)]
    assert cruising["acceleration"].abs().max() < 0.05
    assert r.trajectory_table.loc[0, "stops"] == 1


def test_accuracy_rule_only_when_column_present():
    track = make_track()
    track["accuracy_m"] = 5.0
    track.loc[3, "accuracy_m"] = 80.0
    r = run_pipeline(track)
    assert r.quality.rejections["reason"].tolist() == ["reported accuracy too low"]
    off = replace(AppConfig(), cleaning=CleaningConfig(max_accuracy_m=None))
    assert run_pipeline(track, off).quality.rejected_records == 0


def test_bbox_can_be_disabled():
    track = make_track(lat=48.85, lon=2.35)  # Paris
    with pytest.raises(InsufficientDataError):
        run_pipeline(track)
    cfg = replace(AppConfig(), cleaning=CleaningConfig(bbox=None))
    assert run_pipeline(track, cfg).quality.valid_records == len(track)


def test_trajectory_with_one_valid_point_is_skipped():
    a = make_track("A")
    b = make_track("B").iloc[:1]
    r = run_pipeline(pd.concat([a, b], ignore_index=True))
    assert r.vehicles == ["A"]


def test_auto_crs_setting():
    cfg = replace(AppConfig(), spatial=replace(AppConfig().spatial, projected_crs="auto"))
    assert run_pipeline(make_track(), cfg).projected_crs.to_epsg() == 32630
