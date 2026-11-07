from datetime import date, time

import folium
import pandas as pd
import pytest

from gps_seville import analytics, spatial
from gps_seville.analysis import position_at
from gps_seville.exceptions import InsufficientDataError
from gps_seville.mapping import (
    LIGHT_GREY,
    MapLayers,
    available_basemaps,
    build_analysis_map,
    zoom_for_extent,
)
from gps_seville.views import Filters, apply_filters, raw_track
from tests.conftest import ZONES

ALL_LAYERS = MapLayers(
    rejected=True,
    raw_track=True,
    buffers=True,
    frequent_locations=True,
    anomalies=True,
    heatmap=True,
)


@pytest.fixture(scope="module")
def zones():
    return spatial.load_zones(ZONES)


def test_no_filters_keeps_everything(fleet_result, zones):
    v = apply_filters(fleet_result, Filters(), zones)
    assert len(v.points) == len(fleet_result.points)
    assert len(v.lines) == 4 and len(v.stops) == len(fleet_result.stops)
    assert len(v.segments) == len(fleet_result.segment_table)
    assert len(set(v.colours.values())) == 4


def test_vehicle_and_date_filters(fleet_result, zones):
    v = apply_filters(
        fleet_result,
        Filters(vehicles=("SEV-VEH-01",), dates=(date(2026, 3, 11), date(2026, 3, 11))),
        zones,
    )
    assert v.lines["trajectory_id"].tolist() == ["SEV-VEH-01_2026-03-11"]
    assert set(v.stops["traj_id"]) == {"SEV-VEH-01_2026-03-11"}
    assert set(v.segments["trajectory_id"]) == {"SEV-VEH-01_2026-03-11"}


def test_zone_and_hour_filters(fleet_result, zones):
    v = apply_filters(fleet_result, Filters(zone="Triana"), zones)
    triana = zones.loc[zones["name"] == "Triana", "geometry"].iloc[0]
    assert v.points.within(triana).all()
    assert (v.stops.within(triana)).all()
    morning = apply_filters(fleet_result, Filters(hours=(time(9), time(9, 59))), zones)
    assert set(morning.points["vehicle_id"]) == {"SEV-VAN-02"}


def test_raw_track_keeps_jumps_but_not_invalid_coordinates(fleet_result):
    raw = raw_track(fleet_result)
    assert len(raw) > len(fleet_result.cleaned)  # jumps, duplicates, low accuracy still there
    assert raw["latitude"].between(37.30, 37.45).all()
    assert not ((raw["latitude"] == 0) & (raw["longitude"] == 0)).any()


@pytest.mark.parametrize(
    "lon_span,lat_span,expected",
    [(0.032, 0.027, 14), (0.005, 0.004, 17), (0.3, 0.2, 11), (360, 170, 2)],
)
def test_zoom_for_extent(lon_span, lat_span, expected):
    assert zoom_for_extent(lon_span, lat_span) == expected


def test_map_with_all_layers(fleet_result, zones):
    v = apply_filters(fleet_result, Filters(), zones)
    crs = fleet_result.projected_crs
    m = build_analysis_map(
        v,
        ALL_LAYERS,
        zones,
        fleet_result.rejected,
        raw_track(fleet_result),
        spatial.route_buffers(v.lines, crs, 100),
        analytics.frequent_locations(v.stops, crs, 150, 2),
        analytics.detect_anomalies(v.points, 3.5),
        (37.385, -5.99, "test position"),
    )
    html = m.get_root().render()
    for name in [
        "Geofence zones",
        "Route buffers (100 m)",
        "Heatmap of fixes",
        "Raw fixes before cleaning (dashed)",
        "Detected stops",
        "Frequent locations (DBSCAN)",
        "Unusual movement",
        "Rejected GPS records",
        "Estimated position (interpolated)",
        "OpenStreetMap",
    ]:
        assert name in html, name


def test_map_with_no_points(fleet_result, zones):
    empty = apply_filters(fleet_result, Filters(vehicles=("nobody",)), zones)
    assert empty.empty
    assert isinstance(build_analysis_map(empty, MapLayers(), zones), folium.Map)


def test_position_at_observed_and_interpolated(fleet_result):
    traj = fleet_result.collection.get_trajectory("SEV-VEH-01_2026-03-10")
    t0 = traj.df.index[100]
    lat, _lon, observed = position_at(traj, t0)
    assert observed and lat == pytest.approx(traj.df.geometry.iloc[100].y)
    mid = t0 + pd.Timedelta(seconds=2)
    lat_m, _lon_m, observed_m = position_at(traj, mid)
    assert not observed_m
    p0, p1 = traj.df.geometry.iloc[100], traj.df.geometry.iloc[101]
    assert min(p0.y, p1.y) <= lat_m <= max(p0.y, p1.y)
    with pytest.raises(InsufficientDataError):
        position_at(traj, t0 - pd.Timedelta(days=1))


def test_basemaps_without_carto_key(monkeypatch, fleet_result, zones):
    monkeypatch.delenv("CARTO_API_KEY", raising=False)
    assert available_basemaps() == [LIGHT_GREY, "OpenStreetMap"]
    v = apply_filters(fleet_result, Filters(), zones)
    html = build_analysis_map(v, MapLayers(), zones).get_root().render()
    assert "basemap-grey" in html and "grayscale(1)" in html
    assert "leaflet.polylineoffset" in html and '"offset": 9.0' in html  # 4 routes side by side
    assert "tile.openstreetmap.org" in html and "openstreetmap.org/copyright" in html
    assert "cartocdn" not in html
    # a CARTO basemap requested without a key falls back to the default
    html = build_analysis_map(v, MapLayers(), zones, basemap="CARTO Voyager").get_root().render()
    assert "cartocdn" not in html


def test_basemaps_with_carto_key(monkeypatch, fleet_result, zones):
    monkeypatch.setenv("CARTO_API_KEY", "test-key")
    assert "CARTO Voyager" in available_basemaps()
    v = apply_filters(fleet_result, Filters(), zones)
    html = build_analysis_map(v, MapLayers(), zones, basemap="CARTO Voyager").get_root().render()
    assert "rastertiles/voyager" in html and "api_key=test-key" in html
    assert "carto.com/attributions" in html
