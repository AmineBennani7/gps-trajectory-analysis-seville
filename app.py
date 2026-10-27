"""Streamlit app: GPS Trajectory Analysis - Seville.

Run with:
    streamlit run app.py
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from gps_seville.analysis import analyse
from gps_seville.cleaning import clean_records, load_records
from gps_seville.mapping import build_map

SAMPLE_FILE = Path(__file__).parent / "data" / "vehicle_gps_seville.csv"


def fmt_td(td):
    total = int(pd.Timedelta(td).total_seconds())
    hours, rest = divmod(total, 3600)
    if hours == 0:
        return f"{rest // 60} min"
    return f"{hours} h {rest // 60:02d} min"


@st.cache_data
def run(file_bytes_or_path, max_speed, stop_diameter, stop_minutes):
    source = file_bytes_or_path
    if isinstance(source, bytes):
        source = BytesIO(source)
    raw = load_records(source)
    clean, removed = clean_records(raw)
    result = analyse(clean, max_speed, stop_diameter, stop_minutes)
    return raw, removed, result


st.set_page_config(page_title="GPS Trajectory Analysis - Seville", layout="wide")
st.title("GPS Trajectory Analysis - Seville")
st.caption("Synthetic vehicle GPS data. Cleaning with pandas, trajectory analysis with "
           "MovingPandas, map with Folium.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("CSV with timestamp, latitude, longitude", type="csv")
    st.header("Parameters")
    max_speed = st.slider("Max plausible speed (km/h)", 80, 400, 200, 10,
                          help="Points that need a higher speed to reach are treated as GPS jumps.")
    stop_diameter = st.slider("Stop: max diameter (m)", 10, 300, 60, 10)
    stop_minutes = st.slider("Stop: min duration (min)", 1, 30, 5)

source = upload.getvalue() if upload else str(SAMPLE_FILE)
st.write(f"Data source: **{upload.name if upload else SAMPLE_FILE.name}**")

try:
    raw, removed, result = run(source, max_speed, stop_diameter, stop_minutes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

s = result.summary
c = st.columns(4)
c[0].metric("Distance", f"{s['distance_km']:.1f} km")
c[1].metric("Total time", fmt_td(s["duration"]))
c[2].metric("Moving time", fmt_td(s["moving_time"]))
c[3].metric("Stops", s["stops"], help=fmt_td(s["stop_time"]) + " stopped")
c = st.columns(4)
c[0].metric("Avg moving speed", f"{s['avg_moving_speed_kmh']:.1f} km/h")
c[1].metric("Max speed", f"{s['max_speed_kmh']:.1f} km/h")
c[2].metric("Records in file", len(raw))
c[3].metric("Records used", s["points"])

st.subheader("Map")
st_folium(build_map(result, pd.concat([removed, result.outliers])),
          height=550, use_container_width=True, returned_objects=[])

left, right = st.columns(2)
with left:
    st.subheader("Stops")
    if len(result.stops):
        stops = pd.DataFrame({
            "start": result.stops["start_time"].dt.strftime("%H:%M:%S"),
            "end": result.stops["end_time"].dt.strftime("%H:%M:%S"),
            "minutes": (result.stops["duration_s"] / 60).round(1),
            "latitude": result.stops.geometry.y.round(5),
            "longitude": result.stops.geometry.x.round(5),
        }).reset_index(drop=True)
        st.dataframe(stops, hide_index=True)
    else:
        st.write("No stops found with the current parameters.")
with right:
    st.subheader("Removed records")
    st.dataframe(
        pd.concat([removed.astype(str), result.outliers.astype(str)])[
            ["timestamp", "latitude", "longitude", "reason"]
        ],
        hide_index=True,
    )

st.subheader("Speed over time (km/h)")
st.line_chart(result.trajectory.df["speed"])

cleaned = result.trajectory.df[["vehicle_id", "latitude", "longitude", "speed"]].reset_index()
st.download_button("Download cleaned trajectory (CSV)", cleaned.to_csv(index=False),
                   file_name="cleaned_trajectory.csv", mime="text/csv")
