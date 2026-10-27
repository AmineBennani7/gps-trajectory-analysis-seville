"""Generate a synthetic GPS log for one vehicle driving around Seville.

The route follows a list of waypoints placed on main streets and bridges, with
straight lines between them (no road network). Coordinates are approximate.
Positions get a small random error, the vehicle stops at a few places, and
some deliberately invalid records are inserted so the cleaning step has
something to do.

Usage:
    python scripts/generate_data.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
INTERVAL_S = 5
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "vehicle_gps_seville.csv"

# (name, lat, lon, stop after arriving in seconds, speed on the leg to here in km/h)
ROUTE = [
    ("Santa Justa station", 37.3917, -5.9757, 0, None),
    ("Avenida Luis de Morales", 37.3870, -5.9735, 0, 35),
    ("Nervión (Nervión Plaza)", 37.3838, -5.9718, 12 * 60, 30),
    ("Avenida Eduardo Dato", 37.3830, -5.9800, 0, 35),
    ("San Bernardo", 37.3808, -5.9850, 0, 30),
    ("Prado de San Sebastián", 37.3800, -5.9862, 0, 30),
    ("City centre (Puerta de Jerez)", 37.3821, -5.9933, 8 * 60, 25),
    ("Paseo de Colón", 37.3850, -5.9985, 0, 30),
    ("Puente de Triana", 37.3857, -6.0012, 0, 25),
    ("Triana (Plaza del Altozano)", 37.3851, -6.0034, 15 * 60, 20),
    ("Calle Pureza", 37.3825, -6.0010, 0, 25),
    ("Plaza de Cuba", 37.3785, -5.9990, 0, 25),
    ("Los Remedios (Calle Asunción)", 37.3745, -6.0012, 90, 25),
    ("Plaza de Cuba", 37.3785, -5.9990, 0, 25),
    ("Puente de los Remedios", 37.3778, -5.9930, 0, 30),
    ("Plaza de España", 37.3775, -5.9875, 0, 30),
    ("Prado de San Sebastián", 37.3800, -5.9862, 0, 25),
    ("Avenida Menéndez Pelayo", 37.3870, -5.9840, 0, 35),
    ("Calle Luis Montoto", 37.3893, -5.9790, 0, 30),
    ("Santa Justa station", 37.3917, -5.9757, 0, 30),
]

METERS_PER_DEG_LAT = 111_320.0


def _distance_m(lat1, lon1, lat2, lon2):
    """Approximate distance, only used to decide how many points a leg gets."""
    mean_lat = np.radians((lat1 + lat2) / 2)
    dy = (lat2 - lat1) * METERS_PER_DEG_LAT
    dx = (lon2 - lon1) * METERS_PER_DEG_LAT * np.cos(mean_lat)
    return float(np.hypot(dx, dy))


def _jitter(rng, lat, noise_m):
    dlat = rng.normal(0, noise_m) / METERS_PER_DEG_LAT
    dlon = rng.normal(0, noise_m) / (METERS_PER_DEG_LAT * np.cos(np.radians(lat)))
    return dlat, dlon


def build_clean_track(start=datetime(2026, 3, 10, 8, 0, 0), seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    t = start

    def add(lat, lon, noise_m):
        dlat, dlon = _jitter(rng, lat, noise_m)
        rows.append((t, round(lat + dlat, 6), round(lon + dlon, 6)))

    _, lat0, lon0, _, _ = ROUTE[0]
    add(lat0, lon0, 3)

    for (_, lat_a, lon_a, _, _), (_, lat_b, lon_b, stop_s, speed_kmh) in zip(ROUTE, ROUTE[1:]):
        leg_m = _distance_m(lat_a, lon_a, lat_b, lon_b)
        step_m = speed_kmh / 3.6 * INTERVAL_S
        n_steps = max(1, int(leg_m / step_m))
        for i in range(1, n_steps + 1):
            t += timedelta(seconds=INTERVAL_S)
            f = i / n_steps
            add(lat_a + f * (lat_b - lat_a), lon_a + f * (lon_b - lon_a), 3)
        for _ in range(stop_s // INTERVAL_S):
            t += timedelta(seconds=INTERVAL_S)
            add(lat_b, lon_b, 1)

    df = pd.DataFrame(rows, columns=["timestamp", "latitude", "longitude"])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def insert_invalid_records(df):
    """Insert invalid records at fixed positions. Returns a new DataFrame."""
    df = df.astype({"latitude": object, "longitude": object})
    ts = df["timestamp"]

    def between(i):
        # timestamp halfway between row i and i+1 (they are 5 s apart)
        t = datetime.strptime(ts.iloc[i], "%Y-%m-%d %H:%M:%S") + timedelta(seconds=2)
        return t.strftime("%Y-%m-%d %H:%M:%S")

    bad = [
        (40, {"timestamp": between(40), "latitude": 0.0, "longitude": 0.0}),  # null island
        (120, {"timestamp": between(120), "latitude": 91.5, "longitude": -5.99}),  # invalid latitude
        (200, {"timestamp": between(200), "latitude": 37.39, "longitude": None}),  # missing value
        (260, {"timestamp": between(260), "latitude": "n/a", "longitude": -5.98}),  # unreadable value
        (300, {"timestamp": "not-a-time", "latitude": 37.38, "longitude": -5.99}),  # bad timestamp
        (450, {"timestamp": between(450), "latitude": 40.4168, "longitude": -3.7038}),  # Madrid
        (380, {"timestamp": between(380), "latitude": 37.4180, "longitude": -5.8931}),  # GPS jump to the airport
        (560, {"timestamp": between(560), "latitude": 37.4040, "longitude": -6.0060}),  # GPS jump to La Cartuja
    ]
    parts, last = [], 0
    for pos, rec in sorted(bad, key=lambda b: b[0]):
        parts.append(df.iloc[last : pos + 1])
        parts.append(pd.DataFrame([rec]))
        last = pos + 1
    parts.append(df.iloc[last:])
    out = pd.concat(parts, ignore_index=True)

    # exact duplicate of an existing record
    dup = out.iloc[[600]]
    out = pd.concat([out.iloc[:601], dup, out.iloc[601:]], ignore_index=True)

    out.insert(0, "vehicle_id", "SEV-VEH-01")
    return out


def main():
    df = insert_invalid_records(build_clean_track())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
