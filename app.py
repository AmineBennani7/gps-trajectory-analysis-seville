"""Streamlit dashboard: GPS Trajectory Analysis - Seville.

Run with:
    streamlit run app.py
    streamlit run app.py -- --config my_settings.toml

This file only builds the interface. Processing is in gps_seville/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from gps_seville import analytics, charts, db, roads, spatial
from gps_seville.analysis import TRAJ_ID, position_at
from gps_seville.config import SEVILLE_BBOX, AppConfig, load_config
from gps_seville.env import load_env_file
from gps_seville.exceptions import GpsAnalysisError
from gps_seville.export import to_csv, to_geojson
from gps_seville.logging_config import configure_logging, get_logger
from gps_seville.mapping import MapLayers, available_basemaps, build_analysis_map
from gps_seville.pipeline import PipelineResult, run_pipeline
from gps_seville.readers import read_files
from gps_seville.views import Filters, apply_filters, raw_track

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SAMPLES = {
    "Sample: 3 vehicles, 2 days": [DATA / "fleet_gps_seville.csv"],
    "Sample: 1 vehicle (original file)": [DATA / "vehicle_gps_seville.csv"],
}
UPLOAD = "Upload files"
CRS_OPTIONS = {
    "EPSG:25830": "ETRS89 / UTM 30N (EPSG:25830)",
    "EPSG:32630": "WGS 84 / UTM 30N (EPSG:32630)",
    "auto": "Automatic UTM zone",
}

load_env_file(ROOT / ".env")
configure_logging()
log = get_logger("app")
st.set_page_config(page_title="GPS Trajectory Analysis - Seville", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    [data-testid="stSidebar"] h3 {font-size: 0.95rem; margin-top: 0.6rem;}
    </style>""",
    unsafe_allow_html=True,
)


def base_config() -> AppConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return load_config(args.config) if args.config else AppConfig()


@st.cache_data(show_spinner="Reading and cleaning GPS records...", max_entries=8)
def process(files: tuple[tuple[str, bytes], ...], config: AppConfig):
    result = run_pipeline(read_files(files), config)
    return result, raw_track(result)


@st.cache_data(max_entries=4)
def load_zone_file(name: str, data: bytes) -> gpd.GeoDataFrame:
    return spatial.load_zones(data, name)


@st.cache_data(max_entries=1)
def load_pois() -> gpd.GeoDataFrame:
    return spatial.load_pois(DATA / "poi_seville.geojson")


def fmt_minutes(minutes: float) -> str:
    h, m = divmod(round(minutes), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def table(df: pd.DataFrame, height: int | str = "auto", **kwargs) -> None:
    st.dataframe(
        df.drop(columns="geometry", errors="ignore"), hide_index=True, height=height, **kwargs
    )


# --- sidebar: data and settings ---------------------------------------------------------

defaults = base_config()
sb = st.sidebar
sb.markdown("## GPS Trajectory Analysis")
sb.caption("Seville · synthetic GPS data")

sb.markdown("### Data")
source = sb.selectbox("Source", [*SAMPLES, UPLOAD], label_visibility="collapsed")
if source == UPLOAD:
    uploads = sb.file_uploader(
        "CSV, GPX or GeoJSON files",
        type=["csv", "gpx", "geojson", "json"],
        accept_multiple_files=True,
    )
    files = tuple((u.name, u.getvalue()) for u in uploads or [])
else:
    files = tuple((p.name, p.read_bytes()) for p in SAMPLES[source])
zone_upload = sb.file_uploader("Zones (GeoJSON polygons, optional)", type=["geojson", "json"])

filter_box = sb.container()
layer_box = sb.container()

with sb.expander("Cleaning rules"):
    c = defaults.cleaning
    use_bbox = st.toggle("Reject records outside the Seville area", c.bbox is not None)
    max_speed = st.number_input(
        "GPS jump threshold (km/h)",
        50.0,
        1000.0,
        c.max_speed_kmh,
        10.0,
        help="A fix that needs a higher speed to reach is a GPS jump "
        "(MovingPandas OutlierCleaner).",
    )
    use_acc = st.toggle(
        "Reject fixes with low reported accuracy",
        c.max_accuracy_m is not None,
        help="Only applies if the input has an accuracy_m column.",
    )
    max_acc = st.number_input(
        "Maximum accuracy value (m)",
        5.0,
        500.0,
        c.max_accuracy_m or 30.0,
        5.0,
        disabled=not use_acc,
    )
    drop_dup = st.toggle("Drop duplicate timestamps", c.drop_duplicate_timestamps)
    zero = st.toggle("Reject (0, 0) coordinates", c.reject_zero_coordinates)

with sb.expander("Stops and journeys"):
    s = defaults.stops
    stop_d = st.number_input("Stop: maximum diameter (m)", 10.0, 500.0, s.max_diameter_m, 10.0)
    stop_t = st.number_input("Stop: minimum duration (min)", 1.0, 120.0, s.min_duration_min, 1.0)
    gap = st.number_input(
        "Split journeys at gaps longer than (s)", 30.0, 3600.0, defaults.analysis.max_gap_s, 30.0
    )

with sb.expander("Analysis settings"):
    a = defaults.analysis
    z = st.number_input("Unusual movement: robust z-score above", 2.0, 10.0, a.anomaly_z, 0.5)
    eps = st.number_input(
        "Frequent locations: cluster radius (m)", 20.0, 1000.0, a.cluster_eps_m, 10.0
    )
    buffer_m = st.number_input(
        "Route buffer / proximity distance (m)",
        10.0,
        1000.0,
        defaults.spatial.route_buffer_m,
        10.0,
    )
    crs_default = defaults.spatial.projected_crs
    crs_key = st.selectbox(
        "Projected CRS for distances",
        list(CRS_OPTIONS),
        format_func=CRS_OPTIONS.get,
        index=list(CRS_OPTIONS).index(crs_default) if crs_default in CRS_OPTIONS else 0,
    )

config = replace(
    defaults,
    cleaning=replace(
        defaults.cleaning,
        bbox=dict(SEVILLE_BBOX) if use_bbox else None,
        max_speed_kmh=max_speed,
        max_accuracy_m=max_acc if use_acc else None,
        drop_duplicate_timestamps=drop_dup,
        reject_zero_coordinates=zero,
    ),
    stops=replace(defaults.stops, max_diameter_m=stop_d, min_duration_min=stop_t),
    analysis=replace(defaults.analysis, max_gap_s=gap, anomaly_z=z, cluster_eps_m=eps),
    spatial=replace(defaults.spatial, projected_crs=crs_key, route_buffer_m=buffer_m),
)

# --- processing -------------------------------------------------------------------------

st.title("GPS Trajectory Analysis - Seville")
if not files:
    st.info("Upload one or more CSV, GPX or GeoJSON files in the sidebar.")
    st.stop()

try:
    result, raw_fixes = process(files, config)
    if zone_upload:
        zones = load_zone_file(zone_upload.name, zone_upload.getvalue())
    else:
        zones = load_zone_file(
            "zones_seville.geojson", (DATA / "zones_seville.geojson").read_bytes()
        )
    pois = load_pois()
except GpsAnalysisError as exc:
    st.error(str(exc))
    st.stop()
result: PipelineResult

# --- sidebar: filters and layers (need the processed data) -------------------------------

with filter_box:
    st.markdown("### Filters")
    vehicles = st.multiselect("Vehicles", result.vehicles, placeholder="All vehicles")
    first = result.points["timestamp"].min().date()
    last = result.points["timestamp"].max().date()
    dates = st.date_input("Dates", (first, last), first, last)
    hours = st.slider(
        "Time of day", time(0, 0), time(23, 59), (time(0, 0), time(23, 59)), format="HH:mm"
    )
    zone = st.selectbox("Zone", ["All areas", *zones["name"]])

with layer_box, st.expander("Map layers", expanded=True):
    basemap = st.selectbox("Basemap", available_basemaps())
    if len(available_basemaps()) == 2:
        st.caption("CARTO Positron and Voyager need a free CARTO API key (CARTO_API_KEY).")
    layers = MapLayers(
        trajectories=st.checkbox("Trajectories", True),
        start_end=st.checkbox("Start and end markers", True),
        stops=st.checkbox("Detected stops", True),
        zones=st.checkbox("Geofence zones", True),
        rejected=st.checkbox("Rejected GPS records", False),
        raw_track=st.checkbox("Raw track before cleaning", False),
        buffers=st.checkbox("Route buffers", False),
        frequent_locations=st.checkbox("Frequent locations", False),
        anomalies=st.checkbox("Unusual movement", False),
        heatmap=st.checkbox("Heatmap", False),
    )

filters = Filters(
    vehicles=tuple(vehicles),
    dates=tuple(dates) if isinstance(dates, tuple) and len(dates) == 2 else None,
    hours=None if hours == (time(0, 0), time(23, 59)) else hours,
    zone=None if zone == "All areas" else zone,
)
view = apply_filters(result, filters, zones)
crs = result.projected_crs
if view.empty:
    st.warning("No GPS fixes match the current filters.")
    st.stop()

anomalies = analytics.detect_anomalies(view.points, config.analysis.anomaly_z)
locations = analytics.frequent_locations(
    view.stops, crs, config.analysis.cluster_eps_m, config.analysis.cluster_min_visits
)
events = spatial.geofence_events(view.points, zones, config.analysis.max_gap_s)
q = result.quality

overview, map_tab, quality_tab, movement_tab, spatial_tab, db_tab, export_tab = st.tabs(
    ["Overview", "Map", "Data quality", "Movement", "Spatial analysis", "Database", "Export"]
)

# --- overview ---------------------------------------------------------------------------

with overview:
    t = view.trajectories
    cols = st.columns(6)
    cols[0].metric("Vehicles", t["vehicle_id"].nunique())
    cols[1].metric("Trajectories", len(t), help="One per vehicle and day")
    cols[2].metric(
        "Distance",
        f"{t['distance_km'].sum():.1f} km",
        help="Cleaned GPS distance, including parked jitter (see Data quality)",
    )
    cols[3].metric("Moving time", fmt_minutes(t["moving_min"].sum()))
    cols[4].metric("Stops", len(view.stops))
    cols[5].metric(
        "Valid records",
        f"{100 * q.valid_share:.1f} %",
        help=f"{q.valid_records} of {q.total_records} records passed cleaning",
    )
    st.markdown("#### Trajectories")
    table(
        t.drop(columns=["start", "end"]),
        column_config={
            "distance_km": st.column_config.NumberColumn("km", format="%.2f"),
            "avg_moving_speed_kmh": "avg moving km/h",
            "max_speed_kmh": "max km/h",
            "max_accel_ms2": "max accel m/s²",
            "max_decel_ms2": "max decel m/s²",
        },
    )
    st.caption(
        f"Data source: {', '.join(n for n, _ in files)}. Distances computed by MovingPandas "
        f"(geodesic, WGS84) and checked in {crs.name}. The GPS data is synthetic."
    )

# --- map --------------------------------------------------------------------------------

with map_tab:
    with st.expander("Position at a given time"):
        c1, c2 = st.columns([1, 2])
        tid = c1.selectbox("Trajectory", sorted(view.lines[TRAJ_ID]))
        traj = result.collection.get_trajectory(tid)
        start = pd.Timestamp(traj.get_start_time()).to_pydatetime()
        end = pd.Timestamp(traj.get_end_time()).to_pydatetime()
        when = c2.slider(
            "Time",
            start,
            end,
            start + (end - start) / 2,
            step=timedelta(seconds=1),
            format="HH:mm:ss",
        )
        lat, lon, observed = position_at(traj, pd.Timestamp(when))
        kind = "observed GPS fix" if observed else "estimated, interpolated between two fixes"
        st.caption(f"{when:%H:%M:%S}: {lat:.5f}, {lon:.5f} ({kind}).")
        show = st.checkbox("Show on the map (hollow ring)", False)
        estimate = (lat, lon, f"{tid} {when:%H:%M:%S} ({kind})") if show else None

    buffers = (
        spatial.route_buffers(view.lines, crs, config.spatial.route_buffer_m)
        if layers.buffers
        else None
    )
    fmap = build_analysis_map(
        view,
        layers,
        zones,
        result.rejected,
        raw_fixes,
        buffers,
        locations,
        anomalies,
        estimate,
        basemap=basemap,
    )
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[], key="main_map")
    st.caption(
        "Base map © OpenStreetMap contributors. Zones are approximate outlines drawn "
        "for this demo, not official boundaries."
    )

# --- data quality -----------------------------------------------------------------------

with quality_tab:
    cols = st.columns(5)
    cols[0].metric("Records read", q.total_records)
    cols[1].metric("Valid", q.valid_records)
    cols[2].metric("Rejected", q.rejected_records)
    cols[3].metric("Valid share", f"{100 * q.valid_share:.2f} %")
    cols[4].metric("Median sampling interval", f"{q.sampling['median_interval_s'].median():.0f} s")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Rejected records by reason")
        if q.rejections.empty:
            st.write("No records were rejected.")
        else:
            st.altair_chart(charts.rejection_bars(q.rejections), width="stretch")
    with right:
        st.markdown("#### Per vehicle")
        table(q.per_vehicle)
        st.markdown("#### Sampling")
        table(q.sampling)

    st.markdown("#### Distance before and after cleaning")
    st.altair_chart(charts.distance_stages(q.distances), width="stretch")
    table(q.distances)
    with st.expander("How to read these numbers"):
        st.markdown(
            "- **raw_km** joins every record that has a time and valid coordinates, in time "
            "order, including zero coordinates, far-away points and GPS jumps.\n"
            "- **cleaned_km** is the distance after both cleaning stages.\n"
            "- **stationary_km** is the part of cleaned_km collected while a stop was "
            "detected. A parked receiver still reports positions a few metres apart, and "
            "joining them adds distance that was never driven.\n"
            "- **moving_km** (cleaned minus stationary) is an estimate of the distance driven."
            "\n\nThe GPS positions are synthetic. Their random error is a few metres (more in "
            "one degraded block of the fleet file), similar to a consumer receiver in a city. "
            "Results are not more precise than that: distances are shown to the metre only "
            "so the stages can be compared."
        )

    st.markdown("#### Rejected records")
    table(result.rejected, height=280)

# --- movement ---------------------------------------------------------------------------

with movement_tab:
    st.markdown("#### Speed profile")
    st.altair_chart(
        charts.motion_profile(view.points, view.colours, "speed", "Speed (km/h)"),
        width="stretch",
    )
    st.markdown("#### Acceleration profile")
    st.altair_chart(
        charts.motion_profile(view.points, view.colours, "acceleration", "Acceleration (m/s²)"),
        width="stretch",
    )
    st.caption(
        "Speed and acceleration are computed by MovingPandas from consecutive fixes. "
        "With 5-second sampling and a few metres of position error they are noisy."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Journeys")
        st.caption("Trajectories split at stops and at signal gaps (MovingPandas splitters).")
        table(
            view.segments.drop(columns=["start_lat", "start_lon", "end_lat", "end_lon"]),
            height=300,
        )
    with right:
        st.markdown("#### Stops and dwell time")
        if view.stops.empty:
            st.write("No stops with the current settings.")
        else:
            stops_t = view.stops.assign(minutes=(view.stops["duration_s"] / 60).round(1))
            table(stops_t[["vehicle_id", "start_time", "end_time", "minutes"]], height=300)

    st.markdown("#### Unusual movement")
    st.caption(
        f"Fixes with a robust z-score above {config.analysis.anomaly_z:g} for that vehicle: "
        "high speed, or harsh acceleration and braking. The synthetic vehicles change speed "
        "instantly after stops, so many flags here are those transitions."
    )
    table(anomalies, height=220)

    st.markdown("#### Route similarity")
    st.caption(
        "Hausdorff and discrete Fréchet distances (Shapely) and the share of each route "
        f"within {config.analysis.similarity_buffer_m:g} m of the other."
    )
    table(analytics.route_similarity(view.lines, crs, config.analysis.similarity_buffer_m))

# --- spatial analysis -------------------------------------------------------------------

with spatial_tab:
    st.markdown("#### Geofence events")
    st.caption(
        "Entry and exit times are the first and last GPS fix inside a zone. "
        "entered/exited = False means the trajectory started or ended inside."
    )
    table(events, height=280)

    summary = spatial.zone_summary(events)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Time inside zones")
        if not summary.empty:
            st.altair_chart(charts.zone_time(summary), width="stretch")
    with right:
        st.markdown("#### Route length inside zones")
        table(spatial.route_zone_intersections(view.lines, zones, crs), height=300)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Stops by zone (spatial join)")
        table(spatial.stops_by_zone(view.stops, zones))
        st.markdown("#### Frequent locations")
        table(locations)
    with right:
        st.markdown(f"#### Points of interest within {config.spatial.route_buffer_m:g} m")
        table(spatial.pois_near_routes(view.lines, pois, crs, config.spatial.route_buffer_m))
        st.markdown("#### Nearest point of interest to each stop")
        table(spatial.nearest_poi_to_stops(view.stops, pois, crs))

    st.markdown("#### Road network (OpenStreetMap, optional)")
    if not roads.osmnx_available():
        st.write("Install OSMnx to enable this section: `pip install osmnx`.")
    else:
        st.caption(
            "Downloads the drivable road network around the routes from OpenStreetMap "
            "(needs internet) and assigns each fix to the nearest road. This is "
            "nearest-edge snapping, not full map matching."
        )
        if st.button("Load road network"):
            # bounding box of the routes + 150 m: one simple Overpass query
            area = view.lines.to_crs(crs).union_all().buffer(150).envelope
            area_wgs = gpd.GeoSeries([area], crs=crs).to_crs(4326).iloc[0]
            try:
                with st.spinner("Downloading roads from OpenStreetMap..."):
                    st.session_state["roads"] = roads.download_roads(area_wgs)
            except GpsAnalysisError as exc:
                st.error(str(exc))
        if "roads" in st.session_state:
            snapped = roads.snap_to_roads(view.points, st.session_state["roads"], crs)
            c1, c2 = st.columns(2)
            c1.metric(
                "Fixes within 30 m of a road", f"{100 * snapped['road'].notna().mean():.1f} %"
            )
            c2.metric("Median distance to road", f"{snapped['snap_m'].median():.1f} m")
            table(roads.road_usage(snapped), height=260)

# --- database ---------------------------------------------------------------------------

with db_tab:
    if not db.database_url():
        st.info(
            "No database configured. The app works without one. To store results in "
            "PostGIS, start the database with `docker compose up -d db` and set "
            "DATABASE_URL (see .env.example), then restart the app."
        )
    else:
        try:
            engine = db.get_engine()
            c1, c2, c3 = st.columns(3)
            if c1.button("Check connection"):
                version = db.check_connection(engine)
                st.success(f"Connected. {version.split(' ')[0]}")
            if c2.button("Create tables and indexes"):
                db.init_schema(engine)
                st.success("Schema ready")
            if c3.button("Store current results"):
                db.init_schema(engine)
                n = db.store_points(engine, result.points)
                db.store_trajectories(engine, result.lines)
                db.store_zones(engine, zones)
                st.success(
                    f"Stored {n} new points, {len(result.lines)} trajectories and "
                    f"{len(zones)} zones"
                )
            db.init_schema(engine)
            st.write("Rows stored:", db.table_counts(engine))

            st.markdown("#### Query stored points")
            q1, q2, q3 = st.columns(3)
            qv = q1.selectbox("Vehicle", ["Any", *result.vehicles])
            qd = q2.date_input("Date", first, key="db_date")
            qz = q3.selectbox("Within zone (bounding box)", ["Anywhere", *zones["name"]])
            bbox = None
            if qz != "Anywhere":
                bbox = tuple(zones[zones["name"] == qz].total_bounds)
            day = datetime.combine(qd, time(0))
            found = db.query_points(
                engine, None if qv == "Any" else qv, day, day + timedelta(days=1), bbox
            )
            st.caption(f"{len(found)} points (ST_Intersects with ST_MakeEnvelope, GiST index)")
            table(found.drop(columns="geom"), height=260)
        except GpsAnalysisError as exc:
            st.error(str(exc))

# --- export -----------------------------------------------------------------------------

with export_tab:
    st.caption(
        "GeoJSON files are in WGS84 (EPSG:4326). Exports follow the current filters, except "
        "the cleaned and rejected records, which are always complete."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Records**")
        st.download_button(
            "Cleaned records (CSV)", to_csv(result.cleaned), "cleaned_records.csv", "text/csv"
        )
        st.download_button(
            "Rejected records (CSV)", to_csv(result.rejected), "rejected_records.csv", "text/csv"
        )
        st.download_button(
            "Processed points (CSV)", to_csv(view.points), "processed_points.csv", "text/csv"
        )
    with c2:
        st.markdown("**Geometries**")
        st.download_button(
            "Trajectories (GeoJSON)",
            to_geojson(view.lines),
            "trajectories.geojson",
            "application/geo+json",
        )
        st.download_button(
            "Stops (GeoJSON)", to_geojson(view.stops), "stops.geojson", "application/geo+json"
        )
        st.download_button(
            "Zones (GeoJSON)", to_geojson(zones), "zones.geojson", "application/geo+json"
        )
    with c3:
        st.markdown("**Analysis tables**")
        st.download_button(
            "Trajectory summary (CSV)", to_csv(view.trajectories), "trajectories.csv", "text/csv"
        )
        st.download_button(
            "Geofence events (CSV)", to_csv(events), "geofence_events.csv", "text/csv"
        )
        st.download_button(
            "Distance comparison (CSV)",
            to_csv(q.distances),
            "distance_comparison.csv",
            "text/csv",
        )
