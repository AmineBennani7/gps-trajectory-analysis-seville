import json
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from gps_seville import roads
from gps_seville import spatial as sp
from gps_seville.exceptions import InputFormatError
from gps_seville.export import to_csv, to_geojson
from gps_seville.pipeline import run_pipeline
from tests.conftest import POIS, ZONES, make_track

UTM = "EPSG:25830"


@pytest.fixture(scope="module")
def zones():
    return sp.load_zones(ZONES)


@pytest.fixture(scope="module")
def pois():
    return sp.load_pois(POIS)


def _zone(name, minx, miny, maxx, maxy):
    return gpd.GeoDataFrame(
        {"name": [name], "zone_type": ["test"]}, geometry=[box(minx, miny, maxx, maxy)], crs=4326
    )


def test_load_zones_and_pois(zones, pois):
    assert len(zones) == 6 and zones.crs.to_epsg() == 4326
    assert {"Triana", "Nervión", "Casco Antiguo"} <= set(zones["name"])
    assert len(pois) == 7


def test_zone_file_errors():
    points = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "p"},
                "geometry": {"type": "Point", "coordinates": [-5.99, 37.38]},
            }
        ],
    }
    with pytest.raises(InputFormatError, match="Polygon"):
        sp.load_zones(json.dumps(points).encode())
    with pytest.raises(InputFormatError):
        sp.load_zones(b"not geojson")


def test_invalid_polygon_is_repaired():
    bowtie = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "bow"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-5.99, 37.38],
                            [-5.98, 37.39],
                            [-5.98, 37.38],
                            [-5.99, 37.39],
                            [-5.99, 37.38],
                        ]
                    ],
                },
            }
        ],
    }
    z = sp.load_zones(json.dumps(bowtie).encode())
    assert z.is_valid.all()


def test_enter_exit_and_dwell():
    """Northbound track: 21 fixes 50 m apart, 5-minute stop at fix 10, zone around it."""
    r = run_pipeline(make_track(n=21, stop_after=10, stop_minutes=5))
    lat10 = 37.38 + 10 * 50 / 111_320
    zone = _zone("Z", -5.991, lat10 - 0.0007, -5.989, lat10 + 0.0007)  # about ±78 m
    ev = sp.geofence_events(r.points, zone, max_gap_s=180)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["entered"] and e["exited"]
    assert 5 <= e["dwell_min"] <= 5.5
    assert e["enter_time"] < datetime(2026, 3, 10, 8, 1, 0) < e["exit_time"]


def test_start_inside_is_not_an_entry():
    r = run_pipeline(make_track(n=21))
    zone = _zone("Start", -5.991, 37.3795, -5.989, 37.3805)
    e = sp.geofence_events(r.points, zone, 180).iloc[0]
    assert not e["entered"] and e["exited"]


def test_gap_inside_zone_splits_visit():
    a = make_track(n=10, lat=37.38)
    b = make_track(n=10, lat=37.38, start=datetime(2026, 3, 10, 8, 10, 0))
    r = run_pipeline(pd.concat([a, b], ignore_index=True))
    zone = _zone("All", -5.995, 37.375, -5.985, 37.39)
    assert len(sp.geofence_events(r.points, zone, max_gap_s=180)) == 2
    assert len(sp.geofence_events(r.points, zone, max_gap_s=3600)) == 1


def test_no_points_in_zones():
    r = run_pipeline(make_track())
    far = _zone("Far", -5.90, 37.42, -5.89, 37.43)
    assert sp.geofence_events(r.points, far, 180).empty
    assert sp.zone_summary(sp.geofence_events(r.points, far, 180)).empty


def test_fleet_geofence(fleet_result, zones):
    ev = sp.geofence_events(fleet_result.points, zones, 180)
    triana = ev[(ev["zone"] == "Triana") & (ev["vehicle_id"] == "SEV-VEH-01")]
    assert len(triana) == 2  # once per day
    assert triana["dwell_min"].between(15, 18).all()
    summary = sp.zone_summary(ev)
    assert set(summary["zone"]) <= set(zones["name"])


def test_stops_by_zone(fleet_result, zones):
    s = sp.stops_by_zone(fleet_result.stops, zones)
    assert len(s) == len(fleet_result.stops)
    assert "Nervión" in set(s["zone"])


def test_route_zone_intersection_lengths(zones):
    line = gpd.GeoDataFrame(
        {"trajectory_id": ["T"], "vehicle_id": ["V"]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs=UTM,
    )
    zone = gpd.GeoDataFrame(
        {"name": ["half"], "zone_type": ["t"]}, geometry=[box(500, -10, 1500, 10)], crs=UTM
    )
    out = sp.route_zone_intersections(line.to_crs(4326), zone.to_crs(4326), UTM)
    assert out.iloc[0]["length_m"] == pytest.approx(500, abs=1)
    assert out.iloc[0]["share_pct"] == pytest.approx(50, abs=0.2)


def test_buffers_and_poi_proximity(pois):
    line = gpd.GeoDataFrame(
        {"trajectory_id": ["T"], "vehicle_id": ["V"]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs=UTM,
    )
    poi = gpd.GeoDataFrame(
        {"name": ["near", "far"], "category": ["x", "x"]},
        geometry=[Point(500, 80), Point(500, 300)],
        crs=UTM,
    )
    buf = sp.route_buffers(line.to_crs(4326), UTM, 100)
    assert buf.to_crs(UTM).area.iloc[0] == pytest.approx(2 * 100 * 1000 + 3.1416 * 100**2, rel=0.01)
    near = sp.pois_near_routes(line.to_crs(4326), poi.to_crs(4326), UTM, 100)
    assert near["poi"].tolist() == ["near"]
    assert near["distance_m"].iloc[0] == pytest.approx(80, abs=0.5)


def test_nearest_poi_to_stops(fleet_result, pois):
    n = sp.nearest_poi_to_stops(fleet_result.stops, pois, fleet_result.projected_crs)
    assert len(n) == len(fleet_result.stops)
    triana = n[n["nearest_poi"] == "Mercado de Triana"]
    assert len(triana) >= 2 and (triana["distance_m"] < 150).all()


def test_geojson_export_round_trip(fleet_result, zones):
    for gdf in (fleet_result.lines, fleet_result.stops, zones, fleet_result.points.head(50)):
        text = to_geojson(gdf)
        back = gpd.read_file(text)
        assert len(back) == len(gdf)
        assert json.loads(text)["type"] == "FeatureCollection"
    first = json.loads(to_geojson(fleet_result.stops))["features"][0]["properties"]
    assert first["start_time"].startswith("2026-03-")


def test_csv_export_writes_wkt(fleet_result):
    text = to_csv(fleet_result.stops)
    assert "POINT (" in text.splitlines()[1]


# --- roads (offline, with a hand-made road layer) ----------------------------------------


def test_snap_to_roads_offline():
    r = run_pipeline(make_track(n=21))  # northbound along lon -5.99
    roads_gdf = roads.roads_from_edges(
        gpd.GeoDataFrame(
            {"name": ["Calle Norte", ["A", "B"]], "highway": ["residential", "primary"]},
            geometry=[
                LineString([(-5.99002, 37.37), (-5.99002, 37.40)]),
                LineString([(-5.95, 37.37), (-5.95, 37.40)]),
            ],
            crs=4326,
        )
    )
    s = roads.snap_to_roads(r.points, roads_gdf, UTM, max_distance_m=30)
    assert (s["road"] == "Calle Norte").all()
    assert s["snap_m"].max() < 3
    usage = roads.road_usage(s)
    assert usage.iloc[0]["fixes"] == len(r.points)
    assert roads_gdf.loc[1, "name"] == "A, B"  # list tags become text


@pytest.mark.network
@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1")
def test_download_roads_live():
    area = box(-5.9940, 37.3810, -5.9910, 37.3830)  # around Puerta de Jerez
    edges = roads.download_roads(area)
    assert len(edges) > 0 and {"name", "highway"} <= set(edges.columns)
