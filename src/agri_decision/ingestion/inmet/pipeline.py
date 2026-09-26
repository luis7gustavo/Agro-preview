from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.ingestion.inmet.client import InmetClient
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_file,
    write_json_atomic,
)
from agri_decision.transforms.inmet import (
    build_station_season_features,
    map_seasons_to_municipalities,
    parse_annual_zip,
)

LOGGER = logging.getLogger(__name__)


def _cached_zip(directory: Path) -> Path | None:
    return next(iter(sorted(directory.glob("*.zip"))), None)


def _obtain_year(
    year: int,
    *,
    settings: Settings,
    client: InmetClient,
    refresh: bool,
) -> Path:
    directory = settings.paths.bronze / "inmet" / "automatic" / f"year={year}"
    cached = None if refresh else _cached_zip(directory)
    if cached is not None:
        return cached
    downloaded = client.fetch_year(year)
    path, _ = preserve_content_addressed_artifact(
        directory=directory,
        stem=f"inmet_automatic_{year}",
        suffix=".zip",
        content=downloaded.content,
        manifest={
            "source": "INMET annual automatic-station historical data",
            "download_url": downloaded.final_url,
            "requested_url": downloaded.requested_url,
            "download_timestamp": downloaded.retrieved_at.isoformat(),
            "reference_period": str(year),
            "pipeline_version": settings.pipeline_version,
            "content_type": downloaded.content_type,
            "archive_policy": "ZIP preserved intact; station CSVs parsed in memory",
        },
    )
    return path


def run_inmet_pipeline(
    years: Iterable[int] | None = None,
    *,
    settings: Settings | None = None,
    refresh: bool = False,
    client: InmetClient | None = None,
) -> tuple[Path, Path, Path]:
    resolved = settings or load_settings()
    requested = sorted(
        set(years or range(resolved.climate.start_year, resolved.climate.end_year + 1))
    )
    if not requested:
        raise ValueError("At least one INMET year is required")
    source_client = client or InmetClient()
    daily_frames: list[pl.DataFrame] = []
    source_checksums: list[str] = []
    source_files: list[str] = []
    LOGGER.info("pipeline_started", extra={"pipeline": "inmet_climate", "years": requested})
    for year in requested:
        raw_path = _obtain_year(year, settings=resolved, client=source_client, refresh=refresh)
        checksum = sha256_file(raw_path)
        frame = parse_annual_zip(raw_path.read_bytes(), source_checksum=checksum)
        daily_frames.append(frame)
        source_checksums.append(checksum)
        source_files.append(str(raw_path.relative_to(resolved.project_root)))
        LOGGER.info(
            "source_processed",
            extra={
                "pipeline": "inmet_climate",
                "year": year,
                "rows_valid": frame.height,
                "stations": frame.get_column("station_code").n_unique(),
                "checksum": checksum,
            },
        )
    daily = (
        pl.concat(daily_frames, how="diagonal_relaxed")
        .unique(subset=["station_code", "data"], keep="last")
        .sort(["station_code", "data"])
    )
    daily_path = resolved.paths.silver / "inmet_station_daily.parquet"
    _write_parquet_atomic(daily, daily_path)
    return _materialize_climate(daily, resolved, requested, source_checksums, source_files)


def rebuild_inmet_climate(*, settings: Settings | None = None) -> tuple[Path, Path, Path]:
    """Rebuild seasonal features from validated Silver without repeating downloads."""
    resolved = settings or load_settings()
    daily_path = resolved.paths.silver / "inmet_station_daily.parquet"
    manifest = json.loads(daily_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if sha256_file(daily_path) != manifest["checksum_sha256"]:
        raise ValueError("INMET Silver checksum mismatch")
    daily = pl.read_parquet(daily_path)
    years = sorted(daily.get_column("data").dt.year().unique().to_list())
    return _materialize_climate(
        daily, resolved, years, manifest["source_checksums_sha256"], manifest["source_files"]
    )


def _materialize_climate(
    daily: pl.DataFrame,
    resolved: Settings,
    requested: list[int],
    source_checksums: list[str],
    source_files: list[str],
) -> tuple[Path, Path, Path]:
    daily_path = resolved.paths.silver / "inmet_station_daily.parquet"
    station_seasons = build_station_season_features(daily)
    station_path = resolved.paths.gold / "inmet_station_season.parquet"
    _write_parquet_atomic(station_seasons, station_path)
    location_path = resolved.paths.silver / "dim_location.parquet"
    if not location_path.exists():
        raise FileNotFoundError("dim_location.parquet is required before INMET mapping")
    municipal = map_seasons_to_municipalities(
        station_seasons,
        pl.read_parquet(location_path),
        max_distance_km=resolved.climate.max_station_distance_km,
        minimum_coverage=resolved.climate.minimum_season_coverage,
    )
    municipal_path = resolved.paths.gold / "inmet_location_season.parquet"
    _write_parquet_atomic(municipal, municipal_path)
    report = {
        "years_requested": requested,
        "daily_rows": daily.height,
        "stations": daily.get_column("station_code").n_unique(),
        "station_season_rows": station_seasons.height,
        "municipality_season_rows": municipal.height,
        "season_year_min": municipal.get_column("season_year").min(),
        "season_year_max": municipal.get_column("season_year").max(),
        "municipalities": municipal.get_column("codigo_ibge").n_unique(),
        "within_radius_rows": municipal.filter(pl.col("quality_flag").is_null()).height,
        "outside_radius_rows": municipal.filter(
            pl.col("quality_flag") == "NO_NEARBY_INMET_STATION"
        ).height,
        "max_station_distance_km": resolved.climate.max_station_distance_km,
        "minimum_season_coverage": resolved.climate.minimum_season_coverage,
        "source_checksums_sha256": sorted(source_checksums),
    }
    for path, dataset, frame in [
        (daily_path, "inmet_station_daily", daily),
        (station_path, "inmet_station_season", station_seasons),
        (municipal_path, "inmet_location_season", municipal),
    ]:
        checksum = sha256_file(path)
        write_json_atomic(
            path.with_suffix(".manifest.json"),
            {
                "dataset": dataset,
                "dataset_version": f"inmet-{requested[0]}-{requested[-1]}-{checksum[:12]}",
                "checksum_sha256": checksum,
                "pipeline_version": resolved.pipeline_version,
                "rows": frame.height,
                "source_checksums_sha256": sorted(source_checksums),
                "source_files": sorted(source_files),
                "quality": report,
            },
        )
    write_json_atomic(resolved.paths.reports / "inmet_climate_quality.json", report)
    from agri_decision.ingestion.era5.backfill import materialize_combined_climate

    combined_path = materialize_combined_climate(resolved)
    LOGGER.info("pipeline_finished", extra={"pipeline": "inmet_climate", **report})
    return daily_path, station_path, combined_path
