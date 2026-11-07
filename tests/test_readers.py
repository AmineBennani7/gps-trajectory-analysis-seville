import json

import pytest

from gps_seville.exceptions import InputFormatError
from gps_seville.readers import RAW_COLUMNS, read_file, read_files
from tests.conftest import DATA, FLEET, SINGLE

GPX = DATA / "samples" / "taxi03_2026-03-11.gpx"
GEOJSON = DATA / "samples" / "van02_2026-03-10.geojson"


def test_csv_keeps_values_as_text():
    df = read_file(SINGLE)
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 661
    assert (df["vehicle_id"] == "SEV-VEH-01").all()
    assert "not-a-time" in set(df["timestamp"])


def test_csv_column_aliases():
    df = read_file(b"lat,lon,time,vehicle\n37.38,-5.99,2026-03-10 08:00:00,A\n", name="x.csv")
    assert df.loc[0, ["latitude", "longitude", "vehicle_id"]].tolist() == ["37.38", "-5.99", "A"]


def test_gpx_reads_track_points_and_missing_time():
    df = read_file(GPX)
    assert df["vehicle_id"].unique().tolist() == ["SEV-TAXI-03"]
    assert df["timestamp"].isna().sum() == 1  # the sample's first point has no <time>
    assert df["timestamp"].dropna().iloc[0].tzinfo is None  # UTC, stored without zone


def test_geojson_points():
    df = read_file(GEOJSON)
    assert df["vehicle_id"].unique().tolist() == ["SEV-VAN-02"]
    assert df["accuracy_m"].notna().all()


def test_geojson_linestring_with_times():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-5.99, 37.38], [-5.99, 37.381]],
                },
                "properties": {
                    "vehicle_id": "L1",
                    "coordTimes": ["2026-03-10T08:00:00Z", "2026-03-10T08:00:05Z"],
                },
            }
        ],
    }
    df = read_file(json.dumps(fc).encode(), name="line.geojson")
    assert len(df) == 2 and df["vehicle_id"].eq("L1").all()


def test_geojson_linestring_without_times_is_rejected():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-5.99, 37.38], [-5.99, 37.381]],
                },
            }
        ],
    }
    with pytest.raises(InputFormatError, match="coordTimes"):
        read_file(json.dumps(fc).encode(), name="line.geojson")


@pytest.mark.parametrize(
    "name,content",
    [
        ("bad.gpx", b"<gpx><not closed"),
        ("bad.geojson", b"{not json"),
        ("empty.csv", b""),
        ("data.xlsx", b"PK"),
    ],
)
def test_unreadable_files_raise_input_format_error(name, content):
    with pytest.raises(InputFormatError):
        read_file(content, name=name)


def test_read_files_combines_sources_with_row_ids():
    raw = read_files([(SINGLE.name, SINGLE), (GPX.name, GPX), (FLEET.name, FLEET)])
    assert raw["row_id"].is_unique
    assert set(raw["source"]) == {SINGLE.name, GPX.name, FLEET.name}
