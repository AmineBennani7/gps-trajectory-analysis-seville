"""Processing pipeline: raw records -> cleaned records -> processed trajectories.

The three datasets are kept separate in PipelineResult:

- raw: records exactly as read from the input files (plus row_id and source);
- cleaned: valid records after both cleaning stages, WGS84 points, no derived values;
- processed: MovingPandas trajectories with speed and acceleration, stops,
  journey segments and summary tables.

Rejected records from both stages are in `rejected`, each with a stage and reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import movingpandas as mpd
import pandas as pd
from pyproj import CRS

from gps_seville import analysis as an
from gps_seville.cleaning import parse_values, rejection_reasons
from gps_seville.config import AppConfig, validate_config
from gps_seville.crs import resolve_projected_crs
from gps_seville.logging_config import get_logger
from gps_seville.quality import (
    QualityReport,
    distance_comparison,
    per_vehicle_quality,
    rejection_summary,
    sampling_quality,
)

log = get_logger(__name__)

REJECTED_COLUMNS = [
    "row_id",
    "vehicle_id",
    "timestamp",
    "latitude",
    "longitude",
    "accuracy_m",
    "source",
    "stage",
    "reason",
]


@dataclass
class PipelineResult:
    config: AppConfig
    raw: pd.DataFrame
    rejected: pd.DataFrame
    cleaned: gpd.GeoDataFrame
    collection: mpd.TrajectoryCollection
    points: gpd.GeoDataFrame
    lines: gpd.GeoDataFrame
    stops: gpd.GeoDataFrame
    segments: mpd.TrajectoryCollection
    segment_table: pd.DataFrame
    trajectory_table: pd.DataFrame
    quality: QualityReport
    projected_crs: CRS

    @property
    def vehicles(self) -> list[str]:
        return sorted(self.points["vehicle_id"].unique())


def run_pipeline(raw: pd.DataFrame, config: AppConfig | None = None) -> PipelineResult:
    config = config or AppConfig()
    validate_config(config)
    raw = raw.copy()
    if "row_id" not in raw.columns:
        raw.insert(0, "row_id", range(len(raw)))
    if "source" not in raw.columns:
        raw["source"] = "input"
    log.info("Pipeline start: %d raw records", len(raw))

    # Stage 1: record checks (my code).
    parsed = parse_values(raw)
    reason = rejection_reasons(parsed, config.cleaning)
    parsed["_reason"] = reason
    bad = reason.notna()
    stage1 = raw.loc[bad].assign(stage="record checks", reason=reason[bad])
    after_checks = (
        parsed.loc[~bad]
        .drop(columns="_reason")
        .sort_values(["vehicle_id", "timestamp"], kind="stable")
    )
    after_checks = an.assign_trajectory_ids(after_checks)

    # Stage 2: GPS jumps (MovingPandas OutlierCleaner), per vehicle and day.
    tc = an.build_collection(after_checks, config.cleaning.min_points_per_trajectory)
    tc, jumps = an.remove_collection_outliers(tc, config.cleaning.max_speed_kmh)
    if not jumps.empty:
        jumps = jumps.merge(
            raw[["row_id", "timestamp"]], on="row_id", how="left", suffixes=("_parsed", "")
        )
    rejected = pd.concat([stage1, jumps], ignore_index=True)
    rejected = rejected.reindex(columns=REJECTED_COLUMNS)

    cleaned = an.points_gdf(tc)
    keep = [
        c
        for c in [
            "row_id",
            "vehicle_id",
            an.TRAJ_ID,
            "timestamp",
            "latitude",
            "longitude",
            "accuracy_m",
            "source",
            "geometry",
        ]
        if c in cleaned
    ]
    cleaned = cleaned[keep]

    # Processed data: motion values, stops, segments.
    tc = an.add_motion(tc)
    points = an.points_gdf(tc)
    stops = an.collection_stops(tc, config.stops)
    segments = an.segment_journeys(tc, config.stops, config.analysis.max_gap_s)

    crs = resolve_projected_crs(cleaned, config.spatial.projected_crs)
    quality = QualityReport(
        total_records=len(raw),
        valid_records=len(cleaned),
        rejections=rejection_summary(rejected, len(raw)),
        per_vehicle=per_vehicle_quality(raw, rejected, cleaned),
        sampling=sampling_quality(points, config.analysis.max_gap_s),
        distances=distance_comparison(parsed, after_checks, cleaned, stops, crs),
    )
    log.info(
        "Pipeline done: %d valid, %d rejected, %d trajectories, %d stops, %d segments",
        len(cleaned),
        len(rejected),
        len(tc),
        len(stops),
        len(segments),
    )
    return PipelineResult(
        config=config,
        raw=raw,
        rejected=rejected,
        cleaned=cleaned,
        collection=tc,
        points=points,
        lines=an.lines_gdf(tc),
        stops=stops,
        segments=segments,
        segment_table=an.segment_table(segments),
        trajectory_table=an.trajectory_table(tc, stops),
        quality=quality,
        projected_crs=crs,
    )
