"""UTC-day ERA5-Land features, explicitly retaining reanalysis provenance.

The ERA5-Land hourly catalogue uses forecast-origin accumulations. The value
at 00 UTC is the *previous* day's 24-hour total, for both tp and ssrd. This is
not the convention of the separate, de-accumulated ERA5-Land timeseries product:
ARCO increments describe the hour ending at each timestamp, so daily totals
require all 24 increments from 01 UTC through next-day 00 UTC.
Reference: https://confluence.ecmwf.int/pages/viewpage.action?pageId=402639006
ARCO: https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=536218894
"""

from __future__ import annotations

import io
import zipfile
from calendar import monthrange
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import polars as pl
import xarray as xr

from agri_decision.transforms.inmet import _haversine_matrix_km, add_season_key

SOURCE = "Copernicus ERA5-Land reanalysis"
ARCO_SOURCE = "Copernicus ERA5-Land ARCO reanalysis"
AccumulationMode = Literal["forecast_origin", "hourly_increments"]
LOCATION_COLUMNS = ["codigo_ibge", "municipio", "uf", "latitude_centroid", "longitude_centroid"]
GRID_COLUMNS = ["grid_latitude", "grid_longitude", "distance_to_grid_km"]
VALUES = [
    "rain_mm",
    "mean_temp_c",
    "max_temp_c",
    "min_temp_c",
    "mean_humidity_pct",
    "radiation_kj_m2",
    "mean_wind_m_s",
]
COUNTS = [
    "rain_observations",
    "temp_observations",
    "humidity_observations",
    "radiation_observations",
    "wind_observations",
    "rain_negative_increments",
    "radiation_negative_increments",
]
ALIASES = {
    "t2m": ("t2m", "2m_temperature"),
    "d2m": ("d2m", "2m_dewpoint_temperature"),
    "u10": ("u10", "10u", "10m_u_component_of_wind"),
    "v10": ("v10", "10v", "10m_v_component_of_wind"),
    "tp": ("tp", "total_precipitation"),
    "ssrd": ("ssrd", "surface_solar_radiation_downwards"),
}


@contextmanager
def _datasets(content: bytes) -> Iterator[list[xr.Dataset]]:
    buffers: list[io.BytesIO] = []
    datasets: list[xr.Dataset] = []
    try:
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                payloads = [
                    archive.read(n)
                    for n in sorted(archive.namelist())
                    if n.lower().endswith((".nc", ".netcdf"))
                ]
        else:
            payloads = [content]
        if not payloads:
            raise ValueError("ERA5 archive contains no NetCDF files")
        for payload in payloads:
            buffer = io.BytesIO(payload)
            buffers.append(buffer)
            engine = "h5netcdf" if payload.startswith(b"\x89HDF") else "scipy"
            datasets.append(xr.open_dataset(buffer, engine=engine))
        yield datasets
    finally:
        for dataset in datasets:
            dataset.close()
        for buffer in buffers:
            buffer.close()


def _validate_units(variable: str, array: xr.DataArray) -> None:
    units = str(array.attrs.get("units", "")).lower().replace(" ", "").replace("**", "^")
    acceptable = {
        "t2m": {"k", "kelvin"},
        "d2m": {"k", "kelvin"},
        "u10": {"ms^-1", "ms-1", "m/s"},
        "v10": {"ms^-1", "ms-1", "m/s"},
        "tp": {"m", "mofwaterequivalent"},
        "ssrd": {"jm^-2", "jm-2", "j/m^2", "jm^(-2)"},
    }
    if units not in acceptable[variable]:
        raise ValueError(f"Unsupported ERA5 {variable} units: {array.attrs.get('units')!r}")


def _extract_hourly(
    dataset: xr.Dataset,
    locations: pl.DataFrame,
    *,
    max_grid_distance_km: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    time_name = "valid_time" if "valid_time" in dataset.coords else "time"
    lat_name = "latitude" if "latitude" in dataset.coords else "lat"
    lon_name = "longitude" if "longitude" in dataset.coords else "lon"
    if any(name not in dataset.coords for name in (time_name, lat_name, lon_name)):
        raise ValueError("ERA5 NetCDF must have time, latitude and longitude coordinates")
    for name in (time_name, lat_name, lon_name):
        if dataset.coords[name].ndim != 1:
            raise ValueError(f"ERA5 coordinate {name} must be one-dimensional")
    latitudes = locations.get_column("latitude_centroid").to_numpy().astype(float)
    longitudes = locations.get_column("longitude_centroid").to_numpy().astype(float)
    source_lons = np.asarray(dataset.coords[lon_name].values, dtype=float)
    request_lons = longitudes % 360 if source_lons.min() >= 0 else longitudes
    selected = dataset.sel(
        {
            lat_name: xr.DataArray(latitudes, dims="location"),
            lon_name: xr.DataArray(request_lons, dims="location"),
        },
        method="nearest",
    )
    grid_lats = np.asarray(selected.coords[lat_name].values, dtype=float)
    grid_lons = (np.asarray(selected.coords[lon_name].values, dtype=float) + 180) % 360 - 180
    distances = np.diag(_haversine_matrix_km(latitudes, longitudes, grid_lats, grid_lons))
    if np.any(distances > max_grid_distance_km):
        raise ValueError("ERA5 downloaded grid is too far from one or more municipality centroids")
    times = np.asarray(selected.coords[time_name].values, dtype="datetime64[us]")
    if len(np.unique(times)) != len(times) or np.any(np.isnat(times)):
        raise ValueError("ERA5 time coordinate contains duplicate or invalid timestamps")
    if np.any(times != times.astype("datetime64[h]")):
        raise ValueError("ERA5 time coordinate must contain exact hourly timestamps")
    columns: dict[str, Any] = {
        "codigo_ibge": np.repeat(locations.get_column("codigo_ibge").to_numpy(), len(times)),
        "valid_time": np.tile(times, locations.height),
    }
    recognized = 0
    for canonical, names in ALIASES.items():
        variable_name = next(
            (candidate for candidate in names if candidate in selected.data_vars), None
        )
        if variable_name is None:
            columns[canonical] = np.full(locations.height * len(times), np.nan)
            continue
        array = selected[variable_name]
        _validate_units(canonical, array)
        for dimension in set(array.dims) - {"location", time_name}:
            if array.sizes[dimension] != 1:
                raise ValueError(f"Unsupported ERA5 variable dimension: {dimension}")
            array = array.isel({dimension: 0}, drop=True)
        columns[canonical] = np.asarray(array.transpose("location", time_name).values).ravel()
        recognized += 1
    if not recognized:
        raise ValueError("ERA5 NetCDF has no recognized climate variables")
    hourly = pl.DataFrame(columns).with_columns(
        [
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in ALIASES
        ]
    )
    mapping = locations.with_columns(
        pl.Series("grid_latitude", grid_lats),
        pl.Series("grid_longitude", grid_lons),
        pl.Series("distance_to_grid_km", distances),
    )
    return hourly, mapping


def _merge_non_null(frame: pl.DataFrame, keys: list[str], columns: list[str]) -> pl.DataFrame:
    conflicts = (
        frame.group_by(keys)
        .agg([pl.col(column).drop_nulls().n_unique().alias(column) for column in columns])
        .filter(pl.any_horizontal([pl.col(column) > 1 for column in columns]))
    )
    if not conflicts.is_empty():
        raise ValueError("Conflicting overlapping ERA5 values; source data must be reconciled")
    return frame.group_by(keys).agg([pl.col(column).drop_nulls().first() for column in columns])


def parse_era5_daily(
    content: bytes,
    locations: pl.DataFrame,
    *,
    source_checksum: str,
    max_grid_distance_km: float = 20.0,
    accumulation_mode: AccumulationMode = "forecast_origin",
) -> pl.DataFrame:
    """Parse one CDS download into UTC days; accumulated 00 UTC belongs to D-1.

    Accepts either one NetCDF or a ZIP of NetCDFs split by stepType. Missing
    variables remain null and can be supplied by ``combine_era5_daily`` later.
    ARCO requires ``hourly_increments`` and complete UTC days (01-24 UTC);
    partial daily precipitation/radiation totals are withheld. Their observation
    counts still report the actual valid hours. Never infer the accumulation mode.
    """
    if accumulation_mode not in ("forecast_origin", "hourly_increments"):
        raise ValueError("Unsupported ERA5 accumulation mode")
    locations = locations.select(LOCATION_COLUMNS)
    if locations.is_empty() or locations.get_column("codigo_ibge").n_unique() != locations.height:
        raise ValueError("ERA5 locations must be nonempty and unique by codigo_ibge")
    if locations.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("ERA5 locations have missing coordinates or identity")
    hourly_frames: list[pl.DataFrame] = []
    mappings: list[pl.DataFrame] = []
    with _datasets(content) as datasets:
        for dataset in datasets:
            hourly, mapping = _extract_hourly(
                dataset,
                locations,
                max_grid_distance_km=max_grid_distance_km,
            )
            hourly_frames.append(hourly)
            mappings.append(mapping)
    mapping = _merge_non_null(
        pl.concat(mappings),
        ["codigo_ibge"],
        LOCATION_COLUMNS[1:] + GRID_COLUMNS,
    )
    hourly = _merge_non_null(pl.concat(hourly_frames), ["codigo_ibge", "valid_time"], list(ALIASES))
    vapor_pressure_dewpoint = (17.625 * pl.col("dewpoint") / (243.04 + pl.col("dewpoint"))).exp()
    vapor_pressure_temperature = (
        17.625 * pl.col("temperature") / (243.04 + pl.col("temperature"))
    ).exp()
    hourly = hourly.with_columns(
        (pl.col("t2m") - 273.15).alias("temperature"),
        (pl.col("d2m") - 273.15).alias("dewpoint"),
        pl.col("valid_time").dt.date().alias("data"),
        (pl.col("u10").pow(2) + pl.col("v10").pow(2)).sqrt().alias("wind"),
    ).with_columns(
        (100 * vapor_pressure_dewpoint / vapor_pressure_temperature).clip(0, 100).alias("humidity"),
    )
    instant = (
        hourly.filter(
            pl.any_horizontal([pl.col(name).is_not_null() for name in ("t2m", "d2m", "u10", "v10")])
        )
        .group_by("codigo_ibge", "data")
        .agg(
            pl.col("temperature").mean().alias("mean_temp_c"),
            pl.col("temperature").max().alias("max_temp_c"),
            pl.col("temperature").min().alias("min_temp_c"),
            pl.col("temperature").count().alias("temp_observations"),
            pl.col("humidity").mean().alias("mean_humidity_pct"),
            pl.col("humidity").count().alias("humidity_observations"),
            pl.col("wind").mean().alias("mean_wind_m_s"),
            pl.col("wind").count().alias("wind_observations"),
        )
    )
    accum = hourly.select(
        "codigo_ibge",
        "valid_time",
        (pl.col("tp").cast(pl.Float64) * 1000).alias("rain_mm"),
        (pl.col("ssrd").cast(pl.Float64) / 1000).alias("radiation_kj_m2"),
    )
    if accumulation_mode == "forecast_origin":
        # Only midnight represents a full day in forecast-origin accumulations.
        accum = (
            accum.with_columns(
                pl.when(pl.col("rain_mm") >= -1e-5)
                .then(pl.col("rain_mm").clip(lower_bound=0))
                .otherwise(None),
                pl.when(pl.col("radiation_kj_m2") >= -1e-6)
                .then(pl.col("radiation_kj_m2").clip(lower_bound=0))
                .otherwise(None),
            )
            .filter(
                (pl.col("valid_time").dt.hour() == 0)
                & pl.any_horizontal(
                    pl.col("rain_mm").is_not_null(), pl.col("radiation_kj_m2").is_not_null()
                )
            )
            .with_columns(
                (pl.col("valid_time").dt.date() - pl.duration(days=1)).alias("data"),
                (pl.col("rain_mm").is_not_null().cast(pl.UInt32) * 24).alias("rain_observations"),
                (pl.col("radiation_kj_m2").is_not_null().cast(pl.UInt32) * 24).alias(
                    "radiation_observations"
                ),
                pl.lit(0, dtype=pl.UInt32).alias("rain_negative_increments"),
                pl.lit(0, dtype=pl.UInt32).alias("radiation_negative_increments"),
            )
            .drop("valid_time")
        )
        source = SOURCE
        temporal_method = "UTC; hourly instantaneous; previous-day midnight 24h accumulations"
    else:
        # At 00 UTC ARCO contains 23-24 UTC of D-1, not a 24-hour accumulation.
        # GRIB packing can produce signed hourly differences. Sum them BEFORE
        # validating the daily total, to reconstruct the 24h accumulation without
        # an upward clipping bias or falsely treating a finite hour as missing.
        accum = (
            accum.with_columns(
                (pl.col("valid_time") - pl.duration(hours=1)).dt.date().alias("data")
            )
            .group_by("codigo_ibge", "data")
            .agg(
                pl.when(pl.col("rain_mm").count() == 24)
                .then(pl.col("rain_mm").sum())
                .otherwise(None)
                .alias("rain_mm"),
                pl.col("rain_mm").count().alias("rain_observations"),
                pl.when(pl.col("radiation_kj_m2").count() == 24)
                .then(pl.col("radiation_kj_m2").sum())
                .otherwise(None)
                .alias("radiation_kj_m2"),
                pl.col("radiation_kj_m2").count().alias("radiation_observations"),
                (pl.col("rain_mm") < 0).sum().alias("rain_negative_increments"),
                (pl.col("radiation_kj_m2") < 0).sum().alias("radiation_negative_increments"),
            )
            .with_columns(
                pl.when(pl.col("rain_mm") >= -1e-5)
                .then(pl.col("rain_mm").clip(lower_bound=0))
                .otherwise(None),
                pl.when(pl.col("radiation_kj_m2") >= -1e-6)
                .then(pl.col("radiation_kj_m2").clip(lower_bound=0))
                .otherwise(None),
            )
        )
        source = ARCO_SOURCE
        temporal_method = (
            "UTC; hourly instantaneous; ARCO hourly increments assigned to hour start; "
            "24 finite signed increments required per daily total; "
            "negative increments counted, not clipped; daily nonnegative validation"
        )
    # Preserve all-null grid days as unavailable, rather than manufacturing dry days.
    calendar = hourly.select("codigo_ibge", "data").unique()
    daily = instant.join(accum, on=["codigo_ibge", "data"], how="full", coalesce=True)
    daily = daily.join(calendar, on=["codigo_ibge", "data"], how="full", coalesce=True)
    return (
        daily.join(mapping, on="codigo_ibge")
        .with_columns(
            pl.col(COUNTS).fill_null(0),
            pl.lit("estimated").alias("data_nature"),
            pl.lit(source).alias("source"),
            pl.lit(temporal_method).alias("temporal_method"),
            pl.lit([source_checksum], dtype=pl.List(pl.String)).alias("source_checksums"),
        )
        .sort("codigo_ibge", "data")
    )


def combine_era5_daily(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Coalesce separately downloaded variables/months, rejecting conflicting overlaps."""
    if not frames:
        raise ValueError("No ERA5 daily frames supplied")
    all_days = pl.concat(frames, how="diagonal_relaxed")
    keys = ["codigo_ibge", "data"]
    values = (
        VALUES + LOCATION_COLUMNS[1:] + GRID_COLUMNS + ["source", "data_nature", "temporal_method"]
    )
    merged = _merge_non_null(all_days, keys, values)
    counts = all_days.group_by(keys).agg(
        [pl.col(column).max() for column in COUNTS]
        + [pl.col("source_checksums").explode(empty_as_null=False).unique().sort()],
    )
    return merged.join(counts, on=keys).sort(keys)


def _sum_if_present(column: str, *, months: list[int] | None = None) -> pl.Expr:
    values = pl.col(column)
    if months is not None:
        values = values.filter(pl.col("data").dt.month().is_in(months))
    return pl.when(values.count() > 0).then(values.sum()).otherwise(None)


def _weighted_mean(column: str, weight: str) -> pl.Expr:
    return (
        pl.when(pl.col(weight).sum() > 0)
        .then((pl.col(column) * pl.col(weight)).sum() / pl.col(weight).sum())
        .otherwise(None)
        .alias(column)
    )


def build_era5_season_features(
    daily: pl.DataFrame,
    *,
    minimum_coverage: float = 0.70,
) -> pl.DataFrame:
    """Create Sep-Apr municipality-season rows compatible with climate Gold.

    Missing rain is never interpreted as zero. Incomplete seasons are flagged
    and all agronomic features withheld below the configured coverage minimum.
    Counts reflect available days only, with coverage kept alongside them.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    daily = combine_era5_daily([daily])
    season = add_season_key(daily)
    result = season.group_by(
        LOCATION_COLUMNS + GRID_COLUMNS + ["season_year", "source", "temporal_method"]
    ).agg(
        _sum_if_present("rain_mm").alias("rain_crop_cycle_mm"),
        _sum_if_present("rain_mm", months=[4]).alias("rain_30d_mm"),
        _sum_if_present("rain_mm", months=[3, 4]).alias("rain_60d_mm"),
        _sum_if_present("rain_mm", months=[2, 3, 4]).alias("rain_90d_mm"),
        pl.when(pl.col("rain_mm").count() > 0)
        .then((pl.col("rain_mm") < 1).sum())
        .otherwise(None)
        .alias("dry_days"),
        pl.when(pl.col("max_temp_c").count() > 0)
        .then((pl.col("max_temp_c") > 35).sum())
        .otherwise(None)
        .alias("days_temp_gt_35"),
        pl.when(pl.col("min_temp_c").count() > 0)
        .then((pl.col("min_temp_c") < 10).sum())
        .otherwise(None)
        .alias("days_temp_lt_10"),
        _weighted_mean("mean_temp_c", "temp_observations"),
        pl.col("max_temp_c").max(),
        pl.col("min_temp_c").min(),
        _weighted_mean("mean_humidity_pct", "humidity_observations"),
        _sum_if_present("radiation_kj_m2").alias("radiation_sum_kj_m2"),
        _weighted_mean("mean_wind_m_s", "wind_observations"),
        pl.col("temp_observations").sum().alias("valid_temp_hours"),
        # Partial ARCO days have known hourly observations but no daily total.
        # They must not inflate coverage of the precipitation totals used below.
        pl.col("rain_observations")
        .filter(pl.col("rain_mm").is_not_null())
        .sum()
        .alias("valid_rain_hours"),
        pl.col("data").n_unique().alias("calendar_days_present"),
        pl.col("rain_negative_increments").sum(),
        pl.col("radiation_negative_increments").sum(),
        pl.col("source_checksums").explode(empty_as_null=False).unique().sort(),
    )
    expected = pl.col("season_year").map_elements(
        lambda year: sum(
            monthrange(year - 1 if month >= 9 else year, month)[1] * 24
            for month in [9, 10, 11, 12, 1, 2, 3, 4]
        ),
        return_dtype=pl.Int32,
    )
    result = (
        result.with_columns(expected.alias("expected_hours"))
        .with_columns(
            (pl.col("valid_temp_hours") / pl.col("expected_hours")).alias("temp_coverage"),
            (pl.col("valid_rain_hours") / pl.col("expected_hours")).alias("rain_coverage"),
        )
        .with_columns(
            pl.min_horizontal("temp_coverage", "rain_coverage").alias("season_coverage"),
        )
    )
    features = [
        "rain_crop_cycle_mm",
        "rain_30d_mm",
        "rain_60d_mm",
        "rain_90d_mm",
        "dry_days",
        "days_temp_gt_35",
        "days_temp_lt_10",
        "mean_temp_c",
        "max_temp_c",
        "min_temp_c",
        "mean_humidity_pct",
        "radiation_sum_kj_m2",
        "mean_wind_m_s",
    ]
    return result.with_columns(
        [
            pl.when(pl.col("season_coverage") >= minimum_coverage)
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
            for name in features
        ]
        + [
            pl.lit("estimated").alias("data_nature"),
            pl.concat_str(
                pl.lit("nearest ERA5-Land reanalysis grid cell; UTC Sep-Apr; "),
                pl.col("temporal_method"),
            ).alias("method"),
            pl.lit(None, dtype=pl.String).alias("station_code"),
            pl.lit(None, dtype=pl.String).alias("station_name"),
            pl.lit(None, dtype=pl.String).alias("station_uf"),
            pl.lit(None, dtype=pl.Float64).alias("distance_to_station_km"),
            pl.when(pl.col("season_coverage") < minimum_coverage)
            .then(pl.lit("ERA5_INSUFFICIENT_SEASON_COVERAGE"))
            .otherwise(pl.lit("ERA5_REANALYSIS_FALLBACK"))
            .alias("quality_flag"),
        ],
    ).sort("codigo_ibge", "season_year")
