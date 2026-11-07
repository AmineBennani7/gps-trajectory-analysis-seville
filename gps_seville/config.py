"""Analysis settings.

Defaults live in the dataclasses below. `config/default.toml` documents the
same values and can be copied and edited; `load_config` reads such a file and
overrides only the keys it contains.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from gps_seville.exceptions import ConfigError

# Bounding box around Seville and its nearest suburbs (WGS84).
SEVILLE_BBOX: dict[str, float] = {
    "lat_min": 37.30,
    "lat_max": 37.45,
    "lon_min": -6.08,
    "lon_max": -5.88,
}

# ETRS89 / UTM zone 30N: the official projected CRS for mainland Spain west of 0°.
SEVILLE_PROJECTED_CRS = "EPSG:25830"


@dataclass(frozen=True)
class CleaningConfig:
    """Rules for rejecting GPS records."""

    bbox: dict[str, float] | None = field(default_factory=lambda: dict(SEVILLE_BBOX))
    reject_zero_coordinates: bool = True
    drop_duplicate_timestamps: bool = True
    max_speed_kmh: float = 200.0  # GPS jumps, removed with MovingPandas OutlierCleaner
    max_accuracy_m: float | None = 30.0  # used only if the input has an accuracy_m column
    min_points_per_trajectory: int = 2


@dataclass(frozen=True)
class StopConfig:
    max_diameter_m: float = 60.0
    min_duration_min: float = 5.0


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for segmentation, anomalies and clustering."""

    max_gap_s: float = 180.0  # an observation gap longer than this starts a new segment
    anomaly_z: float = 3.5  # robust z-score threshold (median / MAD)
    cluster_eps_m: float = 150.0  # DBSCAN radius for frequent locations
    cluster_min_visits: int = 2
    similarity_buffer_m: float = 50.0


@dataclass(frozen=True)
class SpatialConfig:
    projected_crs: str = SEVILLE_PROJECTED_CRS  # "auto" picks a UTM zone from the data
    route_buffer_m: float = 100.0


@dataclass(frozen=True)
class AppConfig:
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    stops: StopConfig = field(default_factory=StopConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(section: Any, values: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(section)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown setting(s) in [{name}]: {', '.join(sorted(unknown))}")
    return replace(section, **values)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Return defaults, overridden by the TOML file at `path` if given."""
    config = AppConfig()
    if path is None:
        return config
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc

    sections = {f.name: getattr(config, f.name) for f in fields(config)}
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown section(s): {', '.join(sorted(unknown))}")
    updated = {name: _apply(sections[name], data[name], name) for name in data}
    config = replace(config, **updated)
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    c = config.cleaning
    if c.max_speed_kmh <= 0:
        raise ConfigError("cleaning.max_speed_kmh must be positive")
    if c.bbox is not None:
        if set(c.bbox) != set(SEVILLE_BBOX):
            raise ConfigError("cleaning.bbox needs lat_min, lat_max, lon_min, lon_max")
        if c.bbox["lat_min"] >= c.bbox["lat_max"] or c.bbox["lon_min"] >= c.bbox["lon_max"]:
            raise ConfigError("cleaning.bbox minimums must be smaller than maximums")
    if config.stops.max_diameter_m <= 0 or config.stops.min_duration_min <= 0:
        raise ConfigError("stop settings must be positive")
    if config.analysis.anomaly_z <= 0:
        raise ConfigError("analysis.anomaly_z must be positive")
