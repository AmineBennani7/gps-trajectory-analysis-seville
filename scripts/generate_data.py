"""Generate synthetic GPS logs for vehicles driving around Seville.

The route follows a list of waypoints placed on main streets and bridges, with
straight lines between them (no road network). Coordinates are approximate.
Positions get a small random error, the vehicle stops at a few places, and
some deliberately invalid records are inserted so the cleaning step has
something to do.

Outputs:
    data/vehicle_gps_seville.csv   one vehicle, one trip (the original dataset)
    data/fleet_gps_seville.csv     three vehicles over two days, with an accuracy_m column
    data/samples/*.gpx, *.geojson  clean tracks in other formats, for testing uploads

Usage:
    python scripts/generate_data.py
"""

import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

SEED = 42
INTERVAL_S = 5
DATA = Path(__file__).resolve().parents[1] / "data"
OUTPUT = DATA / "vehicle_gps_seville.csv"
FLEET_OUTPUT = DATA / "fleet_gps_seville.csv"

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


def build_clean_track(start=datetime(2026, 3, 10, 8, 0, 0), seed=SEED, route=None):
    route = ROUTE if route is None else route
    rng = np.random.default_rng(seed)
    rows = []
    t = start

    def add(lat, lon, noise_m):
        dlat, dlon = _jitter(rng, lat, noise_m)
        rows.append((t, round(lat + dlat, 6), round(lon + dlon, 6)))

    _, lat0, lon0, _, _ = route[0]
    add(lat0, lon0, 3)

    for (_, lat_a, lon_a, _, _), (_, lat_b, lon_b, stop_s, speed_kmh) in itertools.pairwise(route):
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
        (120, {"timestamp": between(120), "latitude": 91.5, "longitude": -5.99}),  # bad latitude
        (200, {"timestamp": between(200), "latitude": 37.39, "longitude": None}),  # missing value
        (260, {"timestamp": between(260), "latitude": "n/a", "longitude": -5.98}),  # unreadable
        (300, {"timestamp": "not-a-time", "latitude": 37.38, "longitude": -5.99}),  # bad timestamp
        (450, {"timestamp": between(450), "latitude": 40.4168, "longitude": -3.7038}),  # Madrid
        (380, {"timestamp": between(380), "latitude": 37.4180, "longitude": -5.8931}),  # jump
        (560, {"timestamp": between(560), "latitude": 37.4040, "longitude": -6.0060}),  # jump
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


# Second vehicle: a van from La Macarena along the river to Triana, then east to Nervión.
ROUTE_VAN = [
    ("Basílica de la Macarena", 37.4010, -5.9893, 0, None),
    ("Calle Torneo (north)", 37.3990, -5.9985, 0, 35),
    ("Plaza de Armas", 37.3925, -6.0010, 0, 70),  # speeding, for anomaly tests
    ("Puente de Triana", 37.3857, -6.0012, 0, 30),
    ("Triana (Plaza del Altozano)", 37.3851, -6.0034, 10 * 60, 20),
    ("Calle Pureza", 37.3825, -6.0010, 0, 25),
    ("Puente de San Telmo", 37.3815, -5.9965, 0, 25),
    ("Puerta de Jerez", 37.3821, -5.9933, 0, 25),
    ("Prado de San Sebastián", 37.3800, -5.9862, 0, 30),
    ("Avenida Eduardo Dato", 37.3830, -5.9800, 0, 35),
    ("Nervión (Nervión Plaza)", 37.3838, -5.9718, 8 * 60, 30),
    ("Avenida Luis de Morales", 37.3870, -5.9735, 0, 30),
    ("Santa Justa station", 37.3917, -5.9757, 0, 30),
]

# Third vehicle: a taxi from Santa Justa to Plaza de España, Los Remedios and La Macarena.
ROUTE_TAXI = [
    ("Santa Justa station", 37.3917, -5.9757, 0, None),
    ("Calle Luis Montoto", 37.3893, -5.9790, 0, 30),
    ("Avenida Menéndez Pelayo", 37.3870, -5.9840, 0, 35),
    ("Prado de San Sebastián", 37.3800, -5.9862, 0, 30),
    ("Plaza de España", 37.3775, -5.9875, 6 * 60, 25),
    ("Puente de los Remedios", 37.3778, -5.9930, 0, 30),
    ("Plaza de Cuba", 37.3785, -5.9990, 0, 25),
    ("Los Remedios (Calle Asunción)", 37.3745, -6.0012, 9 * 60, 25),
    ("Plaza de Cuba", 37.3785, -5.9990, 0, 25),
    ("Calle Pureza", 37.3825, -6.0010, 0, 25),
    ("Triana (Plaza del Altozano)", 37.3851, -6.0034, 0, 20),
    ("Puente de Triana", 37.3857, -6.0012, 0, 25),
    ("Plaza de Armas", 37.3925, -6.0010, 0, 30),
    ("Calle Torneo (north)", 37.3990, -5.9985, 0, 40),
    ("Basílica de la Macarena", 37.4010, -5.9893, 5 * 60, 30),
]

# (vehicle_id, route, start time, random seed, signal gap as (offset s, duration s) or None)
FLEET = [
    ("SEV-VEH-01", ROUTE, datetime(2026, 3, 10, 8, 0, 0), SEED, None),
    ("SEV-VAN-02", ROUTE_VAN, datetime(2026, 3, 10, 9, 10, 0), 101, (960, 240)),
    ("SEV-VEH-01", ROUTE, datetime(2026, 3, 11, 17, 30, 0), 7, None),
    ("SEV-TAXI-03", ROUTE_TAXI, datetime(2026, 3, 11, 10, 0, 0), 202, None),
]


def add_accuracy(df, rng, degraded=None):
    """Add a reported accuracy (m) and, for the degraded rows, a matching larger error.

    Receivers report an estimated horizontal accuracy. Here it is 3-8 m normally;
    `degraded` is a (start row, n rows) block where it is 40-70 m and positions
    get an error of the same size, as under trees or between tall buildings.
    """
    df = df.copy()
    df["accuracy_m"] = rng.uniform(3, 8, len(df)).round(1)
    if degraded:
        start, n = degraded
        idx = df.index[start : start + n]
        acc = rng.uniform(40, 70, len(idx)).round(1)
        df.loc[idx, "accuracy_m"] = acc
        for i, a in zip(idx, acc, strict=True):
            lat = df.at[i, "latitude"]
            dlat, dlon = _jitter(rng, lat, a / 2)
            df.at[i, "latitude"] = round(lat + dlat, 6)
            df.at[i, "longitude"] = round(df.at[i, "longitude"] + dlon, 6)
    return df


def drop_window(df, offset_s, duration_s):
    """Remove records in a time window to simulate a loss of GPS signal."""
    t = pd.to_datetime(df["timestamp"])
    t0 = t.iloc[0] + pd.Timedelta(seconds=offset_s)
    keep = (t < t0) | (t >= t0 + pd.Timedelta(seconds=duration_s))
    return df[keep].reset_index(drop=True)


def build_fleet_tracks():
    """Return {(vehicle_id, date): clean DataFrame} for the fleet dataset."""
    tracks = {}
    for vehicle_id, route, start, seed, gap in FLEET:
        df = build_clean_track(start=start, seed=seed, route=route)
        if gap:
            df = drop_window(df, *gap)
        degraded = (150, 8) if vehicle_id == "SEV-TAXI-03" else None
        df = add_accuracy(df, np.random.default_rng(seed + 1), degraded=degraded)
        df.insert(0, "vehicle_id", vehicle_id)
        tracks[(vehicle_id, start.date().isoformat())] = df
    return tracks


def insert_fleet_invalid_records(df):
    """A smaller set of invalid records than the single-vehicle file, spread over vehicles."""
    df = df.astype({"latitude": object, "longitude": object, "timestamp": object})
    bad = [
        # missing timestamp
        {
            "vehicle_id": "SEV-VAN-02",
            "timestamp": "",
            "latitude": 37.39,
            "longitude": -5.99,
            "accuracy_m": 5.0,
        },
        # zero coordinates
        {
            "vehicle_id": "SEV-VAN-02",
            "timestamp": "2026-03-10 09:20:02",
            "latitude": 0.0,
            "longitude": 0.0,
            "accuracy_m": 5.0,
        },
        # Cádiz, outside the study area
        {
            "vehicle_id": "SEV-TAXI-03",
            "timestamp": "2026-03-11 10:05:02",
            "latitude": 36.5297,
            "longitude": -6.2925,
            "accuracy_m": 6.0,
        },
        # GPS jump to the airport
        {
            "vehicle_id": "SEV-TAXI-03",
            "timestamp": "2026-03-11 10:30:02",
            "latitude": 37.4180,
            "longitude": -5.8931,
            "accuracy_m": 6.0,
        },
        # GPS jump to La Cartuja
        {
            "vehicle_id": "SEV-VEH-01",
            "timestamp": "2026-03-11 17:40:02",
            "latitude": 37.4040,
            "longitude": -6.0060,
            "accuracy_m": 6.0,
        },
    ]
    out = pd.concat([df, pd.DataFrame(bad)], ignore_index=True)
    dup = out[out["vehicle_id"] == "SEV-TAXI-03"].iloc[[100, 101]]
    out = pd.concat([out, dup], ignore_index=True)
    order = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.assign(_t=order).sort_values(["_t", "vehicle_id"], na_position="last", kind="stable")
    return out.drop(columns="_t")


def write_gpx(df, path, name):
    """GPX 1.1 track. The first point is written without a time to test that case."""
    pts = []
    for i, r in enumerate(df.itertuples()):
        time = "" if i == 0 else f"<time>{r.timestamp.replace(' ', 'T')}Z</time>"
        pts.append(f'      <trkpt lat="{r.latitude}" lon="{r.longitude}">{time}</trkpt>')
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="gps-seville generate_data.py" '
        'xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"  <trk><name>{escape(name)}</name><trkseg>\n"
    )
    path.write_text(header + "\n".join(pts) + "\n  </trkseg></trk>\n</gpx>\n", encoding="utf-8")


def write_geojson_points(df, path):
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
            "properties": {
                "vehicle_id": r.vehicle_id,
                "timestamp": r.timestamp,
                "accuracy_m": r.accuracy_m,
            },
        }
        for r in df.itertuples()
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )


def main():
    df = insert_invalid_records(build_clean_track())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} records to {OUTPUT}")

    tracks = build_fleet_tracks()
    fleet = insert_fleet_invalid_records(pd.concat(tracks.values(), ignore_index=True))
    fleet.to_csv(FLEET_OUTPUT, index=False)
    print(f"Wrote {len(fleet)} records to {FLEET_OUTPUT}")

    samples = DATA / "samples"
    samples.mkdir(exist_ok=True)
    write_gpx(
        tracks[("SEV-TAXI-03", "2026-03-11")], samples / "taxi03_2026-03-11.gpx", "SEV-TAXI-03"
    )
    write_geojson_points(tracks[("SEV-VAN-02", "2026-03-10")], samples / "van02_2026-03-10.geojson")
    print(f"Wrote sample GPX and GeoJSON files to {samples}")


if __name__ == "__main__":
    main()
