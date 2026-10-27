"""Load raw GPS records and remove invalid ones.

The record-level checks (missing values, coordinate ranges, area, duplicates)
are simple pandas filters written for this project. Removal of GPS jumps is
done later with MovingPandas' OutlierCleaner (see analysis.py).
"""

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "latitude", "longitude"]

# Bounding box around Seville and its nearest suburbs.
SEVILLE_BBOX = {"lat_min": 37.30, "lat_max": 37.45, "lon_min": -6.08, "lon_max": -5.88}


def load_records(path_or_buffer):
    """Read a CSV of GPS records. Values are kept as text for cleaning."""
    df = pd.read_csv(path_or_buffer, dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if "vehicle_id" not in df.columns:
        df["vehicle_id"] = "vehicle"
    return df


def clean_records(raw, bbox=SEVILLE_BBOX):
    """Remove invalid records.

    Returns (clean, removed). `clean` has parsed timestamp/latitude/longitude
    columns sorted by time. `removed` holds the dropped rows with a `reason`.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    def flag(mask, text):
        reason[mask & reason.isna()] = text

    flag(df[REQUIRED_COLUMNS].isna().any(axis=1), "missing or unreadable value")
    flag(~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180),
         "outside valid latitude/longitude range")
    flag((df["latitude"] == 0) & (df["longitude"] == 0), "zero coordinates (0, 0)")
    flag(~df["latitude"].between(bbox["lat_min"], bbox["lat_max"])
         | ~df["longitude"].between(bbox["lon_min"], bbox["lon_max"]),
         "outside Seville area")
    flag(df.duplicated(subset=["vehicle_id", "timestamp"], keep="first"), "duplicate timestamp")

    removed = raw.loc[reason.notna()].assign(reason=reason[reason.notna()])
    clean = df.loc[reason.isna()].sort_values("timestamp").reset_index(drop=True)
    return clean, removed
