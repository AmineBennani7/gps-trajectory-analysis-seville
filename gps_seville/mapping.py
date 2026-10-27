"""Folium map of the analysed trajectory."""

import folium


def _fmt_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes} min {seconds:02d} s"


def build_map(result, removed=None):
    """Return a folium.Map with the route, start/end, stops and removed points."""
    df = result.trajectory.df
    coords = list(zip(df.geometry.y, df.geometry.x))
    minx, miny, maxx, maxy = df.geometry.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=14, tiles="OpenStreetMap")

    route = folium.FeatureGroup(name="Route")
    folium.PolyLine(coords, color="#1f77b4", weight=4, opacity=0.8).add_to(route)
    folium.Marker(coords[0], tooltip=f"Start {df.index[0]:%H:%M:%S}",
                  icon=folium.Icon(color="green", icon="play")).add_to(route)
    folium.Marker(coords[-1], tooltip=f"End {df.index[-1]:%H:%M:%S}",
                  icon=folium.Icon(color="red", icon="stop")).add_to(route)
    route.add_to(m)

    stops = folium.FeatureGroup(name="Stops")
    for i, row in enumerate(result.stops.itertuples(), start=1):
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            tooltip=f"Stop {i}: {row.start_time:%H:%M} - {row.end_time:%H:%M} "
                    f"({_fmt_duration(row.duration_s)})",
            icon=folium.Icon(color="orange", icon="pause"),
        ).add_to(stops)
    stops.add_to(m)

    if removed is not None and len(removed):
        group = folium.FeatureGroup(name="Removed points (inside map area)", show=False)
        for row in removed.itertuples():
            try:
                lat, lon = float(row.latitude), float(row.longitude)
            except (TypeError, ValueError):
                continue
            if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                folium.CircleMarker([lat, lon], radius=5, color="black", fill=True,
                                    fill_color="black", tooltip=row.reason).add_to(group)
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
