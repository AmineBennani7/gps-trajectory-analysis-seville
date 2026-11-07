"""Record-level validation of raw GPS records.

This is the first cleaning stage and is my own code (pandas filters). It
rejects records that cannot be valid positions on their own. GPS jumps, which
are only visible by comparing a point with its neighbours, are removed later by
MovingPandas' OutlierCleaner (analysis.remove_speed_outliers).

Each rejected record keeps its original values and gets one reason, the first
rule it failed, in the order of RULES below.
"""

from __future__ import annotations

import pandas as pd

from gps_seville.config import SEVILLE_BBOX, CleaningConfig
from gps_seville.readers import Source, read_csv

REQUIRED_COLUMNS = ["timestamp", "latitude", "longitude"]
__all__ = ["SEVILLE_BBOX", "clean_records", "load_records", "parse_values", "rejection_reasons"]

REASON_TIMESTAMP = "missing or unreadable timestamp"
REASON_COORDINATE = "missing or unreadable coordinate"
REASON_RANGE = "outside valid latitude/longitude range"
REASON_ZERO = "zero coordinates (0, 0)"
REASON_AREA = "outside study area"
REASON_ACCURACY = "reported accuracy too low"
REASON_DUPLICATE = "duplicate timestamp"

RULES = [
    REASON_TIMESTAMP,
    REASON_COORDINATE,
    REASON_RANGE,
    REASON_ZERO,
    REASON_AREA,
    REASON_ACCURACY,
    REASON_DUPLICATE,
]


def load_records(path_or_buffer: Source, name: str = "csv") -> pd.DataFrame:
    """Read a CSV of GPS records. Values are kept as text for cleaning."""
    return read_csv(path_or_buffer, name=name)


def parse_values(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp, coordinates and accuracy. Unreadable values become NaN/NaT."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if "accuracy_m" in df.columns:
        df["accuracy_m"] = pd.to_numeric(df["accuracy_m"], errors="coerce")
    if "vehicle_id" not in df.columns:
        df["vehicle_id"] = "vehicle"
    return df


def rejection_reasons(df: pd.DataFrame, config: CleaningConfig) -> pd.Series:
    """Return the first failed rule per row (NA where the record is valid)."""
    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    def flag(mask: pd.Series, text: str) -> None:
        reason[mask & reason.isna()] = text

    lat, lon = df["latitude"], df["longitude"]
    flag(df["timestamp"].isna(), REASON_TIMESTAMP)
    flag(lat.isna() | lon.isna(), REASON_COORDINATE)
    flag(~lat.between(-90, 90) | ~lon.between(-180, 180), REASON_RANGE)
    if config.reject_zero_coordinates:
        flag((lat == 0) & (lon == 0), REASON_ZERO)
    if config.bbox is not None:
        b = config.bbox
        flag(
            ~lat.between(b["lat_min"], b["lat_max"]) | ~lon.between(b["lon_min"], b["lon_max"]),
            REASON_AREA,
        )
    if config.max_accuracy_m is not None and "accuracy_m" in df.columns:
        flag(df["accuracy_m"] > config.max_accuracy_m, REASON_ACCURACY)
    if config.drop_duplicate_timestamps:
        flag(
            df.duplicated(subset=["vehicle_id", "timestamp"], keep="first")
            & df["timestamp"].notna(),
            REASON_DUPLICATE,
        )
    return reason


def clean_records(
    raw: pd.DataFrame,
    bbox: dict[str, float] | None = SEVILLE_BBOX,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove invalid records.

    Returns (clean, removed). `clean` has parsed values, sorted by vehicle and
    time. `removed` holds the original rows with `reason` and `stage` columns.
    `bbox` is used only when no config is given (older call style).
    """
    config = config or CleaningConfig(bbox=bbox)
    df = parse_values(raw)
    reason = rejection_reasons(df, config)
    bad = reason.notna()

    removed = raw.loc[bad].assign(reason=reason[bad], stage="record checks")
    clean = (
        df.loc[~bad].sort_values(["vehicle_id", "timestamp"], kind="stable").reset_index(drop=True)
    )
    return clean, removed
