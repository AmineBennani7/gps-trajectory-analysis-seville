"""Optional PostgreSQL/PostGIS storage.

The app works without a database. To use one, set DATABASE_URL, for example
    postgresql+psycopg://user:password@localhost:5432/gps
(see .env.example and docker-compose.yml). Credentials are never stored in code.

Tables (created by `init_schema`):
- gps_points: cleaned fixes, geometry(Point, 4326), GiST index on geom and a
  B-tree index on (vehicle_id, recorded_at);
- trajectories: one LineString per vehicle and day with start/end times and
  geodesic length, GiST index on geom;
- zones: geofence polygons, GiST index.

Queries use PostGIS functions: ST_Intersects with ST_MakeEnvelope for
bounding boxes, ST_DWithin on geography for distances in metres, and
ST_Length on geography for lengths.

Needs: pip install sqlalchemy geoalchemy2 "psycopg[binary]"
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon

from gps_seville.analysis import TRAJ_ID
from gps_seville.exceptions import DatabaseError
from gps_seville.logging_config import get_logger

log = get_logger(__name__)

SCHEMA_SQL = [
    "CREATE EXTENSION IF NOT EXISTS postgis",
    """
    CREATE TABLE IF NOT EXISTS gps_points (
        id            BIGSERIAL PRIMARY KEY,
        vehicle_id    TEXT NOT NULL,
        trajectory_id TEXT NOT NULL,
        recorded_at   TIMESTAMP NOT NULL,
        speed_kmh     DOUBLE PRECISION,
        accel_ms2     DOUBLE PRECISION,
        accuracy_m    DOUBLE PRECISION,
        source        TEXT,
        geom          geometry(Point, 4326) NOT NULL,
        UNIQUE (vehicle_id, recorded_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS gps_points_geom_idx ON gps_points USING GIST (geom)",
    "CREATE INDEX IF NOT EXISTS gps_points_vehicle_time_idx "
    "ON gps_points (vehicle_id, recorded_at)",
    """
    CREATE TABLE IF NOT EXISTS trajectories (
        trajectory_id TEXT PRIMARY KEY,
        vehicle_id    TEXT NOT NULL,
        start_time    TIMESTAMP NOT NULL,
        end_time      TIMESTAMP NOT NULL,
        length_m      DOUBLE PRECISION,
        geom          geometry(LineString, 4326) NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS trajectories_geom_idx ON trajectories USING GIST (geom)",
    """
    CREATE TABLE IF NOT EXISTS zones (
        name      TEXT PRIMARY KEY,
        zone_type TEXT,
        geom      geometry(MultiPolygon, 4326) NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS zones_geom_idx ON zones USING GIST (geom)",
]


def database_url() -> str | None:
    """The configured URL from the DATABASE_URL environment variable, or None."""
    url = os.getenv("DATABASE_URL", "").strip()
    return url or None


def get_engine(url: str | None = None) -> Any:
    """SQLAlchemy engine for `url` or DATABASE_URL. Raises DatabaseError if missing."""
    url = url or database_url()
    if not url:
        raise DatabaseError("No database configured. Set DATABASE_URL (see .env.example).")
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise DatabaseError(
            'PostGIS support needs: pip install sqlalchemy geoalchemy2 "psycopg[binary]"'
        ) from exc
    return create_engine(url, pool_pre_ping=True, future=True)


@contextmanager
def _connect(engine: Any) -> Iterator[Any]:
    from sqlalchemy.exc import SQLAlchemyError

    try:
        with engine.begin() as conn:
            yield conn
    except SQLAlchemyError as exc:
        # The message can contain the host but SQLAlchemy hides the password.
        raise DatabaseError(f"Database error: {exc.__class__.__name__}: {exc.orig or exc}") from exc


def check_connection(engine: Any) -> str:
    """Return the PostGIS version, or raise DatabaseError."""
    from sqlalchemy import text

    with _connect(engine) as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        return str(conn.execute(text("SELECT postgis_full_version()")).scalar())


def init_schema(engine: Any) -> None:
    from sqlalchemy import text

    with _connect(engine) as conn:
        for stmt in SCHEMA_SQL:
            conn.execute(text(stmt))
    log.info("Database schema ready")


POINT_COLUMNS = {
    "vehicle_id": "vehicle_id",
    TRAJ_ID: "trajectory_id",
    "timestamp": "recorded_at",
    "speed": "speed_kmh",
    "acceleration": "accel_ms2",
    "accuracy_m": "accuracy_m",
    "source": "source",
}


def _points_frame(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.reindex(columns=[*POINT_COLUMNS, "geometry"]).rename(columns=POINT_COLUMNS)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=points.crs)
    for col in ("speed_kmh", "accel_ms2", "accuracy_m"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.rename_geometry("geom").to_crs("EPSG:4326")


def _stage(conn: Any, frame: gpd.GeoDataFrame, name: str) -> None:
    """Write `frame` to a staging table (replaced if it exists).

    A normal table is used because GeoPandas looks up the SRID of the target
    table with Find_SRID, which cannot see temporary tables. The caller drops
    it in the same transaction.
    """
    frame.to_postgis(name, conn, if_exists="replace", index=False)


def store_points(engine: Any, points: gpd.GeoDataFrame) -> int:
    """Insert processed points. Rows already stored (same vehicle and time) are skipped."""
    from sqlalchemy import text

    frame = _points_frame(points)
    with _connect(engine) as conn:
        _stage(conn, frame, "staging_points")
        result = conn.execute(
            text("""
            INSERT INTO gps_points (vehicle_id, trajectory_id, recorded_at, speed_kmh,
                                    accel_ms2, accuracy_m, source, geom)
            SELECT vehicle_id, trajectory_id, recorded_at, speed_kmh, accel_ms2, accuracy_m,
                   source, geom
            FROM staging_points
            ON CONFLICT (vehicle_id, recorded_at) DO NOTHING
        """)
        )
        inserted = result.rowcount
        conn.execute(text("DROP TABLE staging_points"))
    log.info("Stored %d new points (%d given)", inserted, len(frame))
    return inserted


def store_trajectories(engine: Any, lines: gpd.GeoDataFrame) -> int:
    """Insert or replace trajectory lines; length is computed by PostGIS (geodesic)."""
    from sqlalchemy import text

    frame = (
        lines[[TRAJ_ID, "vehicle_id", "start_t", "end_t", "geometry"]]
        .rename(columns={TRAJ_ID: "trajectory_id", "start_t": "start_time", "end_t": "end_time"})
        .rename_geometry("geom")
        .to_crs("EPSG:4326")
    )
    with _connect(engine) as conn:
        _stage(conn, frame, "staging_traj")
        conn.execute(
            text("""
            INSERT INTO trajectories (trajectory_id, vehicle_id, start_time, end_time,
                                      length_m, geom)
            SELECT trajectory_id, vehicle_id, start_time, end_time,
                   ST_Length(geom::geography), geom
            FROM staging_traj
            ON CONFLICT (trajectory_id) DO UPDATE
              SET vehicle_id = EXCLUDED.vehicle_id, start_time = EXCLUDED.start_time,
                  end_time = EXCLUDED.end_time, length_m = EXCLUDED.length_m,
                  geom = EXCLUDED.geom
        """)
        )
        conn.execute(text("DROP TABLE staging_traj"))
    return len(frame)


def store_zones(engine: Any, zones: gpd.GeoDataFrame) -> int:
    from sqlalchemy import text

    frame = zones[["name", "zone_type", "geometry"]].copy()
    frame["geometry"] = frame.geometry.apply(
        lambda g: g if g.geom_type == "MultiPolygon" else MultiPolygon([g])
    )
    frame = frame.rename_geometry("geom").to_crs("EPSG:4326")
    with _connect(engine) as conn:
        _stage(conn, frame, "staging_zones")
        conn.execute(
            text("""
            INSERT INTO zones (name, zone_type, geom)
            SELECT name, zone_type, geom FROM staging_zones
            ON CONFLICT (name) DO UPDATE SET zone_type = EXCLUDED.zone_type, geom = EXCLUDED.geom
        """)
        )
        conn.execute(text("DROP TABLE staging_zones"))
    return len(frame)


def _read(engine: Any, sql: str, params: dict[str, Any]) -> gpd.GeoDataFrame:
    from sqlalchemy import text

    with _connect(engine) as conn:
        return gpd.read_postgis(text(sql), conn, geom_col="geom", params=params)


def query_points(
    engine: Any,
    vehicle_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Points filtered by vehicle, time range and bounding box (minx, miny, maxx, maxy)."""
    where, params = [], {}
    if vehicle_id:
        where.append("vehicle_id = :vehicle_id")
        params["vehicle_id"] = vehicle_id
    if start:
        where.append("recorded_at >= :start")
        params["start"] = start
    if end:
        where.append("recorded_at <= :end")
        params["end"] = end
    if bbox:
        where.append("ST_Intersects(geom, ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))")
        params.update(dict(zip(["minx", "miny", "maxx", "maxy"], bbox, strict=True)))
    sql = "SELECT * FROM gps_points"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return _read(engine, sql + " ORDER BY vehicle_id, recorded_at", params)


def query_points_near(engine: Any, lon: float, lat: float, radius_m: float) -> gpd.GeoDataFrame:
    """Points within `radius_m` metres of (lon, lat), using geography for true distances."""
    sql = """
        SELECT *, ST_Distance(geom::geography,
                              ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS distance_m
        FROM gps_points
        WHERE ST_DWithin(geom::geography,
                         ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)
        ORDER BY distance_m
    """
    return _read(engine, sql, {"lon": lon, "lat": lat, "radius": radius_m})


def query_trajectories_in_zone(engine: Any, zone_name: str) -> pd.DataFrame:
    """Trajectories crossing a stored zone, with the length inside it (metres)."""
    sql = """
        SELECT t.trajectory_id, t.vehicle_id, t.start_time,
               ST_Length(ST_Intersection(t.geom, z.geom)::geography) AS length_in_zone_m,
               t.geom
        FROM trajectories t JOIN zones z ON ST_Intersects(t.geom, z.geom)
        WHERE z.name = :zone
        ORDER BY length_in_zone_m DESC
    """
    return _read(engine, sql, {"zone": zone_name})


def table_counts(engine: Any) -> dict[str, int]:
    from sqlalchemy import text

    with _connect(engine) as conn:
        return {
            t: int(conn.execute(text(f"SELECT count(*) FROM {t}")).scalar())
            for t in ("gps_points", "trajectories", "zones")
        }


def explain_bbox_query(engine: Any, bbox: tuple[float, float, float, float]) -> str:
    """EXPLAIN output for a bounding-box query, to check that the GiST index is used."""
    from sqlalchemy import text

    with _connect(engine) as conn:
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        rows = conn.execute(
            text(
                "EXPLAIN SELECT id FROM gps_points WHERE "
                "ST_Intersects(geom, ST_MakeEnvelope(:a, :b, :c, :d, 4326))"
            ),
            dict(zip("abcd", bbox, strict=True)),
        ).fetchall()
    return "\n".join(r[0] for r in rows)
