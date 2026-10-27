from io import StringIO

import pytest

from gps_seville.cleaning import clean_records, load_records

CSV = """vehicle_id,timestamp,latitude,longitude
V1,2026-03-10 08:00:00,37.3890,-5.9845
V1,2026-03-10 08:00:05,37.3891,-5.9847
V1,2026-03-10 08:00:10,0,0
V1,2026-03-10 08:00:15,91.5,-5.98
V1,2026-03-10 08:00:20,37.389,
V1,2026-03-10 08:00:25,abc,-5.98
V1,not-a-time,37.389,-5.98
V1,2026-03-10 08:00:30,40.4168,-3.7038
V1,2026-03-10 08:00:05,37.3891,-5.9847
V1,2026-03-10 08:00:35,37.3893,-5.9850
"""


def test_invalid_records_are_removed_with_reason():
    clean, removed = clean_records(load_records(StringIO(CSV)))
    assert len(clean) == 3
    assert sorted(removed["reason"]) == sorted([
        "zero coordinates (0, 0)",
        "outside valid latitude/longitude range",
        "missing or unreadable value",
        "missing or unreadable value",
        "missing or unreadable value",
        "outside Seville area",
        "duplicate timestamp",
    ])


def test_clean_output_is_sorted_and_typed():
    clean, _ = clean_records(load_records(StringIO(CSV)))
    assert clean["timestamp"].is_monotonic_increasing
    assert clean["latitude"].dtype.kind == "f"
    assert clean["longitude"].dtype.kind == "f"


def test_missing_column_raises():
    with pytest.raises(ValueError, match="longitude"):
        load_records(StringIO("timestamp,latitude\n2026-03-10 08:00:00,49.6\n"))
