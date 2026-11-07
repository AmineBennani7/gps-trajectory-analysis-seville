"""Run the Streamlit script headless with streamlit.testing (no browser).

These tests check that every tab renders without an exception for the sample
datasets and some settings; they do not check the visual layout.
"""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Run without PostGIS. An empty value also stops the app reading a local .env."""
    monkeypatch.setenv("DATABASE_URL", "")


def _run(**widgets) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for key, value in widgets.items():
        kind, label = key.split("__")
        element = next(e for e in getattr(at, kind) if e.label == label.replace("_", " "))
        element.set_value(value)
    if widgets:
        at.run()
    return at


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_fleet_sample_renders_all_tabs():
    at = _run()
    _no_errors(at)
    assert at.title[0].value == "GPS Trajectory Analysis - Seville"
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Map",
        "Data quality",
        "Movement",
        "Spatial analysis",
        "Database",
        "Export",
    ]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Vehicles"] == "3"
    assert metrics["Trajectories"] == "4"
    assert "No database configured" in " ".join(i.value for i in at.info)


def test_single_vehicle_sample():
    at = _run(selectbox__Source="Sample: 1 vehicle (original file)")
    _no_errors(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Distance"] == "9.7 km"
    assert metrics["Stops"] == "3"


@pytest.mark.parametrize(
    "widget,value",
    [
        ("multiselect__Vehicles", ["SEV-VAN-02"]),
        ("selectbox__Zone", "Triana"),
        ("selectbox__Projected_CRS_for_distances", "auto"),
    ],
)
def test_filters_and_settings(widget, value):
    _no_errors(_run(**{widget: value}))


def test_filter_with_no_matches_shows_warning():
    at = _run(multiselect__Vehicles=["SEV-TAXI-03"])
    dates = next(d for d in at.date_input if d.label == "Dates")
    dates.set_value((date(2026, 3, 10), date(2026, 3, 10)))  # taxi drives on 11 March
    at.run()
    assert not at.exception
    assert "No GPS fixes match" in at.warning[0].value
