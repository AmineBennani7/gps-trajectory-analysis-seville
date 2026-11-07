"""PostGIS tests.

The unit tests run everywhere. The integration tests need a running PostGIS
and DATABASE_URL, for example:
    docker compose up -d db
    DATABASE_URL=postgresql+psycopg://gps:<password>@localhost:5432/gps pytest -m postgis
They write to the tables gps_points, trajectories and zones.
"""

from datetime import datetime

import pytest

from gps_seville import db
from gps_seville import spatial as sp
from gps_seville.exceptions import DatabaseError
from tests.conftest import ZONES


def test_no_database_configured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.database_url() is None
    with pytest.raises(DatabaseError, match="DATABASE_URL"):
        db.get_engine()


def test_schema_has_spatial_indexes():
    sql = " ".join(db.SCHEMA_SQL)
    for table in ("gps_points", "trajectories", "zones"):
        assert f"ON {table} USING GIST (geom)" in sql
    assert "UNIQUE (vehicle_id, recorded_at)" in sql


def test_unreachable_database_raises_database_error():
    engine = db.get_engine("postgresql+psycopg://nobody:secret@127.0.0.1:1/none?connect_timeout=2")
    with pytest.raises(DatabaseError) as info:
        db.check_connection(engine)
    assert "secret" not in str(info.value)


postgis = pytest.mark.skipif(not db.database_url(), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import text

    e = db.get_engine()
    db.init_schema(e)
    with e.begin() as conn:
        conn.execute(text("TRUNCATE gps_points, trajectories, zones"))
    return e


@pytest.mark.postgis
@postgis
def test_store_and_query(engine, fleet_result):
    assert "POSTGIS" in db.check_connection(engine).upper()
    n = db.store_points(engine, fleet_result.points)
    assert n == len(fleet_result.points)
    assert db.store_points(engine, fleet_result.points) == 0  # duplicates are skipped
    db.store_trajectories(engine, fleet_result.lines)
    db.store_zones(engine, sp.load_zones(ZONES))
    assert db.table_counts(engine) == {
        "gps_points": len(fleet_result.points),
        "trajectories": len(fleet_result.lines),
        "zones": 6,
    }

    van = db.query_points(engine, vehicle_id="SEV-VAN-02")
    assert len(van) == (fleet_result.points["vehicle_id"] == "SEV-VAN-02").sum()
    day2 = db.query_points(engine, start=datetime(2026, 3, 11), end=datetime(2026, 3, 12))
    assert set(day2["recorded_at"].dt.date.astype(str)) == {"2026-03-11"}
    box = db.query_points(engine, bbox=(-6.006, 37.383, -6.001, 37.387))  # Triana
    assert len(box) > 0


@pytest.mark.postgis
@postgis
def test_spatial_queries(engine, fleet_result):
    # Mercado de Triana: every vehicle except none passes within 100 m; check distances.
    near = db.query_points_near(engine, -6.0029, 37.3857, 100)
    assert len(near) > 0 and near["distance_m"].max() <= 100
    assert near["distance_m"].is_monotonic_increasing

    in_triana = db.query_trajectories_in_zone(engine, "Triana")
    assert set(in_triana["trajectory_id"]) == set(fleet_result.lines["trajectory_id"])
    # PostGIS geodesic length inside Triana agrees with the GeoPandas overlay in UTM
    ref = sp.route_zone_intersections(
        fleet_result.lines, sp.load_zones(ZONES), fleet_result.projected_crs
    )
    ref = ref[ref["zone"] == "Triana"].set_index("trajectory_id")["length_m"]
    for row in in_triana.itertuples():
        assert row.length_in_zone_m == pytest.approx(ref[row.trajectory_id], rel=0.01)


@pytest.mark.postgis
@postgis
def test_trajectory_length_matches_movingpandas(engine, fleet_result):
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = dict(conn.execute(text("SELECT trajectory_id, length_m FROM trajectories")).all())
    for t in fleet_result.trajectory_table.itertuples():
        assert rows[t.trajectory_id] / 1000 == pytest.approx(t.distance_km, rel=0.001)


@pytest.mark.postgis
@postgis
def test_bbox_query_uses_gist_index(engine):
    plan = db.explain_bbox_query(engine, (-6.006, 37.383, -6.001, 37.387))
    assert "gps_points_geom_idx" in plan
