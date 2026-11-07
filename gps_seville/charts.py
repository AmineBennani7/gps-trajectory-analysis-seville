"""Altair charts for the dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd

from gps_seville.analysis import TRAJ_ID


def _colour(colours: dict[str, str]) -> alt.Color:
    ids = sorted(colours)
    return alt.Color(
        f"{TRAJ_ID}:N",
        title="Trajectory",
        scale=alt.Scale(domain=ids, range=[colours[i] for i in ids]),
        legend=alt.Legend(orient="bottom", columns=2),
    )


def motion_profile(
    points: pd.DataFrame, colours: dict[str, str], column: str, title: str
) -> alt.Chart:
    """Speed or acceleration against time since the start of each trajectory."""
    df = points[[TRAJ_ID, "timestamp", column]].copy()
    df["minutes"] = df.groupby(TRAJ_ID)["timestamp"].transform(
        lambda t: (t - t.min()).dt.total_seconds() / 60
    )
    df["time"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("minutes:Q", title="Minutes since start"),
            y=alt.Y(f"{column}:Q", title=title),
            color=_colour(colours),
            tooltip=[TRAJ_ID, "time", alt.Tooltip(f"{column}:Q", format=".2f")],
        )
        .properties(height=260)
    )


def rejection_bars(rejections: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(rejections)
        .mark_bar()
        .encode(
            x=alt.X("records:Q", title="Records"),
            y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "stage:N", title="Stage", scale=alt.Scale(range=["#0072B2", "#D55E00"])
            ),
            tooltip=["stage", "reason", "records", "share_pct"],
        )
        .properties(height=alt.Step(26))
    )


def distance_stages(distances: pd.DataFrame) -> alt.Chart:
    """Distance per trajectory at each cleaning stage (grouped bars)."""
    cols = {
        "after_record_checks_km": "1. after record checks",
        "cleaned_km": "2. after jump removal",
        "moving_km": "3. excluding parked jitter (estimate)",
    }
    long = distances.melt(id_vars=TRAJ_ID, value_vars=list(cols), var_name="stage", value_name="km")
    long["stage"] = long["stage"].map(cols)
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y(f"{TRAJ_ID}:N", title=None, axis=alt.Axis(labelLimit=260)),
            yOffset="stage:N",
            x=alt.X("km:Q", title="Distance (km)"),
            color=alt.Color(
                "stage:N",
                title=None,
                legend=alt.Legend(orient="bottom", labelLimit=320),
                scale=alt.Scale(range=["#999999", "#0072B2", "#009E73"]),
            ),
            tooltip=[TRAJ_ID, "stage", alt.Tooltip("km:Q", format=".3f")],
        )
        .properties(height=alt.Step(14))
    )


def zone_time(summary: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("total_min:Q", title="Minutes inside zone"),
            y=alt.Y("zone:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("vehicle_id:N", title="Vehicle", legend=alt.Legend(orient="bottom")),
            tooltip=["zone", "vehicle_id", "visits", "total_min", "longest_min"],
        )
        .properties(height=alt.Step(26))
    )
