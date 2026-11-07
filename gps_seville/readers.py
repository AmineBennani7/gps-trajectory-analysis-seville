"""Read GPS records from CSV, GPX and GeoJSON into one raw table.

Every reader returns a pandas DataFrame with the columns in RAW_COLUMNS.
Values are kept as they appear in the file (text or numbers); parsing and
validation happen in cleaning.py, so a bad value becomes a rejected record
with a reason instead of an exception.

Timestamps with a time zone (for example GPX `...Z`) are converted to UTC and
stored without the zone, so they compare with naive CSV timestamps. Mixing files
recorded in different local time zones is not handled.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

import gpxpy
import gpxpy.gpx
import pandas as pd

from gps_seville.exceptions import InputFormatError
from gps_seville.logging_config import get_logger

log = get_logger(__name__)

RAW_COLUMNS = ["vehicle_id", "timestamp", "latitude", "longitude", "accuracy_m", "source"]

# Accepted alternative column names (lower case) for CSV and GeoJSON properties.
ALIASES = {
    "vehicle_id": ["vehicle_id", "vehicle", "device_id", "id", "track_id"],
    "timestamp": ["timestamp", "time", "datetime", "date_time", "recorded_at"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "long", "x"],
    "accuracy_m": ["accuracy_m", "accuracy", "hacc", "horizontal_accuracy"],
}

Source = str | Path | bytes | IO[bytes] | IO[str]


def _standardise(df: pd.DataFrame, source: str, default_vehicle: str) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    renamed = {}
    for target, names in ALIASES.items():
        for name in names:
            if name in lower:
                renamed[lower[name]] = target
                break
    df = df.rename(columns=renamed)
    missing = [c for c in ("timestamp", "latitude", "longitude") if c not in df.columns]
    if missing:
        raise InputFormatError(f"{source}: missing required column(s): {', '.join(missing)}")
    if "vehicle_id" not in df.columns:
        df["vehicle_id"] = default_vehicle
    if "accuracy_m" not in df.columns:
        df["accuracy_m"] = pd.NA
    df["vehicle_id"] = df["vehicle_id"].fillna(default_vehicle).astype(str)
    df["source"] = source
    return df[RAW_COLUMNS].reset_index(drop=True)


def _as_text(src: Source) -> str:
    if isinstance(src, (str, Path)):
        return Path(src).read_text(encoding="utf-8-sig")
    if isinstance(src, bytes):
        return src.decode("utf-8-sig")
    data = src.read()
    return data.decode("utf-8-sig") if isinstance(data, bytes) else data


def read_csv(src: Source, name: str = "csv", default_vehicle: str = "vehicle") -> pd.DataFrame:
    """Read a CSV. Values stay as text so invalid ones can be reported later."""
    try:
        df = pd.read_csv(io.StringIO(_as_text(src)), dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputFormatError(f"{name}: not a readable CSV file ({exc})") from exc
    return _standardise(df, name, default_vehicle)


def read_gpx(src: Source, name: str = "gpx", default_vehicle: str | None = None) -> pd.DataFrame:
    """Read track points from a GPX file with gpxpy. Each track name becomes a vehicle_id."""
    try:
        gpx = gpxpy.parse(_as_text(src))
    except (gpxpy.gpx.GPXException, ValueError) as exc:
        raise InputFormatError(f"{name}: not a valid GPX file ({exc})") from exc

    fallback = default_vehicle or Path(name).stem
    rows: list[dict[str, Any]] = []
    for track in gpx.tracks:
        vehicle = track.name or fallback
        for segment in track.segments:
            for p in segment.points:
                rows.append(
                    {
                        "vehicle_id": vehicle,
                        "timestamp": _naive_utc(p.time),
                        "latitude": p.latitude,
                        "longitude": p.longitude,
                    }
                )
    if not rows:
        raise InputFormatError(f"{name}: the GPX file has no track points")
    # GPX has HDOP (unitless) but no accuracy in metres, so accuracy_m stays empty.
    return _standardise(pd.DataFrame(rows), name, fallback)


def read_geojson(
    src: Source, name: str = "geojson", default_vehicle: str | None = None
) -> pd.DataFrame:
    """Read a GeoJSON FeatureCollection.

    Supported features:
    - Point with a timestamp property (one record per feature);
    - LineString with a list of times in `coordTimes` or `times` (one record per vertex).
    """
    try:
        data = json.loads(_as_text(src))
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{name}: not valid JSON ({exc})") from exc
    if data.get("type") != "FeatureCollection":
        raise InputFormatError(f"{name}: expected a GeoJSON FeatureCollection")

    fallback = default_vehicle or Path(name).stem
    rows: list[dict[str, Any]] = []
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        props = {k.lower(): v for k, v in (feature.get("properties") or {}).items()}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "Point" and coords:
            rows.append({**props, "longitude": coords[0], "latitude": coords[1]})
        elif gtype == "LineString" and coords:
            times = props.pop("coordtimes", None) or props.pop("times", None)
            if not times or len(times) != len(coords):
                raise InputFormatError(
                    f"{name}: LineString features need a coordTimes/times list "
                    "with one time per vertex"
                )
            for (lon, lat, *_), t in zip(coords, times, strict=True):
                rows.append({**props, "timestamp": t, "longitude": lon, "latitude": lat})
    if not rows:
        raise InputFormatError(f"{name}: no Point or timed LineString features found")
    return _standardise(pd.DataFrame(rows), name, fallback)


def _naive_utc(t: Any) -> Any:
    if t is None:
        return None
    ts = pd.Timestamp(t)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


READERS = {".csv": read_csv, ".gpx": read_gpx, ".geojson": read_geojson, ".json": read_geojson}


def read_file(src: Source, name: str | None = None) -> pd.DataFrame:
    """Read one file, choosing the reader from the file extension."""
    name = name or (str(src) if isinstance(src, (str, Path)) else "upload")
    suffix = Path(name).suffix.lower()
    reader = READERS.get(suffix)
    if reader is None:
        raise InputFormatError(
            f"{name}: unsupported file type '{suffix}'. Use CSV, GPX or GeoJSON."
        )
    df = reader(src, name=Path(name).name)
    log.info("Read %d records from %s", len(df), name)
    return df


def read_files(items: Iterable[tuple[str, Source]]) -> pd.DataFrame:
    """Read several (name, source) pairs into one raw table with a row_id column."""
    frames = [read_file(src, name=name) for name, src in items]
    if not frames:
        raise InputFormatError("No input files given")
    raw = pd.concat(frames, ignore_index=True)
    raw.insert(0, "row_id", range(len(raw)))
    return raw
