from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from gps_seville.config import AppConfig, load_config
from gps_seville.crs import path_length_m, resolve_projected_crs
from gps_seville.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_default_toml_matches_dataclass_defaults():
    assert load_config(ROOT / "config" / "default.toml") == AppConfig()


def test_partial_override(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[stops]\nmin_duration_min = 2.0\n")
    cfg = load_config(f)
    assert cfg.stops.min_duration_min == 2.0
    assert cfg.stops.max_diameter_m == AppConfig().stops.max_diameter_m


@pytest.mark.parametrize(
    "text,match",
    [
        ("[stops]\nmax_diam = 1\n", "Unknown setting"),
        ("[nonsense]\na = 1\n", "Unknown section"),
        ("[cleaning]\nmax_speed_kmh = -5\n", "positive"),
        ("[cleaning.bbox]\nlat_min = 40\nlat_max = 37\nlon_min = -6\nlon_max = -5\n", "smaller"),
        ("not toml = = =", "not valid TOML"),
    ],
)
def test_invalid_config(tmp_path, text, match):
    f = tmp_path / "c.toml"
    f.write_text(text)
    with pytest.raises(ConfigError, match=match):
        load_config(f)


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.toml")


def _seville():
    return gpd.GeoDataFrame(geometry=[Point(-5.99, 37.38), Point(-5.98, 37.39)], crs=4326)


def test_auto_crs_is_utm_30n():
    assert resolve_projected_crs(_seville(), "auto").to_epsg() == 32630


def test_explicit_crs_and_rejections():
    assert resolve_projected_crs(_seville(), "EPSG:25830").to_epsg() == 25830
    with pytest.raises(ConfigError, match="not a projected"):
        resolve_projected_crs(_seville(), "EPSG:4326")
    with pytest.raises(ConfigError, match="Unknown CRS"):
        resolve_projected_crs(_seville(), "EPSG:999999")


def test_projected_length_matches_geodesic():
    # 1 degree of latitude near 37.4 N is about 110.98 km on the WGS84 ellipsoid.
    pts = gpd.GeoSeries([Point(-5.99, 37.0), Point(-5.99, 38.0)], crs=4326).to_crs(25830)
    assert path_length_m(pts) == pytest.approx(110_980, rel=0.002)
    assert path_length_m(pts.iloc[:1]) == 0.0
