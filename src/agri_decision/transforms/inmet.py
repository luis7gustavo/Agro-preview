from __future__ import annotations

import io
import unicodedata
import zipfile
from calendar import monthrange
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _ascii_upper(value: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    ).upper()
    return normalized.replace("ESTAC?O", "ESTACAO").replace("REGI?O", "REGIAO")


def _metadata_value(lines: list[str], key: str) -> str:
    prefix = f"{key}:;"
    for line in lines[:10]:
        normalized = _ascii_upper(line)
        if normalized.startswith(prefix) or normalized.startswith(f"{key} "):
            return line.split(";", 1)[1].strip()
    raise ValueError(f"INMET station metadata missing {key}")


def _decimal(value: str) -> float:
    return float(value.replace(",", "."))


def _number_at(index: int, alias: str) -> pl.Expr:
    return (
        pl.col(f"column_{index + 1}")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
        .alias(alias)
    )


def parse_station_csv(content: bytes, *, source_checksum: str) -> pl.DataFrame:
    decoded = content.decode("latin1")
    lines = decoded.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _ascii_upper(line).startswith("DATA") and ";HORA" in _ascii_upper(line)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("INMET CSV has no hourly header")
    station_code = _metadata_value(lines, "CODIGO (WMO)")
    station_name = _metadata_value(lines, "ESTACAO")
    uf = _metadata_value(lines, "UF")
    latitude = _decimal(_metadata_value(lines, "LATITUDE"))
    longitude = _decimal(_metadata_value(lines, "LONGITUDE"))
    altitude = _decimal(_metadata_value(lines, "ALTITUDE"))
    body = ("\n".join(lines[header_index + 1 :]) + "\n").encode("utf-8")
    raw = pl.read_csv(
        io.BytesIO(body),
        separator=";",
        has_header=False,
        infer_schema=False,
        truncate_ragged_lines=True,
    )
    if raw.width < 19:
        raise ValueError(f"INMET hourly schema has {raw.width} columns; expected at least 19")
    frame = raw.select(
        pl.col("column_1")
        .cast(pl.Utf8)
        .str.replace_all("/", "-")
        .str.to_date(strict=False)
        .alias("data"),
        _number_at(2, "precipitacao_mm"),
        _number_at(6, "radiacao_kj_m2"),
        _number_at(7, "temperatura_c"),
        _number_at(9, "temperatura_max_c"),
        _number_at(10, "temperatura_min_c"),
        _number_at(15, "umidade_pct"),
        _number_at(18, "vento_m_s"),
    )
    valid = frame.filter(pl.col("data").is_not_null()).with_columns(
        pl.when(pl.col("precipitacao_mm").is_between(0, 200))
        .then(pl.col("precipitacao_mm"))
        .otherwise(None)
        .alias("precipitacao_mm"),
        pl.when(pl.col("temperatura_c").is_between(-30, 60))
        .then(pl.col("temperatura_c"))
        .otherwise(None)
        .alias("temperatura_c"),
        pl.when(pl.col("temperatura_max_c").is_between(-30, 65))
        .then(pl.col("temperatura_max_c"))
        .otherwise(None)
        .alias("temperatura_max_c"),
        pl.when(pl.col("temperatura_min_c").is_between(-40, 60))
        .then(pl.col("temperatura_min_c"))
        .otherwise(None)
        .alias("temperatura_min_c"),
        pl.when(pl.col("umidade_pct").is_between(0, 100))
        .then(pl.col("umidade_pct"))
        .otherwise(None)
        .alias("umidade_pct"),
        pl.when(pl.col("radiacao_kj_m2") >= 0)
        .then(pl.col("radiacao_kj_m2"))
        .otherwise(None)
        .alias("radiacao_kj_m2"),
        pl.when(pl.col("vento_m_s").is_between(0, 75))
        .then(pl.col("vento_m_s"))
        .otherwise(None)
        .alias("vento_m_s"),
    )
    return (
        valid.group_by("data")
        .agg(
            pl.col("precipitacao_mm").sum().alias("rain_mm"),
            pl.col("precipitacao_mm").count().alias("rain_observations"),
            pl.col("temperatura_c").mean().alias("mean_temp_c"),
            pl.col("temperatura_c").count().alias("temp_observations"),
            pl.col("temperatura_max_c").max().alias("max_temp_c"),
            pl.col("temperatura_min_c").min().alias("min_temp_c"),
            pl.col("umidade_pct").mean().alias("mean_humidity_pct"),
            pl.col("umidade_pct").count().alias("humidity_observations"),
            pl.col("radiacao_kj_m2").sum().alias("radiation_kj_m2"),
            pl.col("radiacao_kj_m2").count().alias("radiation_observations"),
            pl.col("vento_m_s").mean().alias("mean_wind_m_s"),
            pl.col("vento_m_s").count().alias("wind_observations"),
        )
        .with_columns(
            pl.lit(station_code).alias("station_code"),
            pl.lit(station_name).alias("station_name"),
            pl.lit(uf).alias("station_uf"),
            pl.lit(latitude).alias("station_latitude"),
            pl.lit(longitude).alias("station_longitude"),
            pl.lit(altitude).alias("station_altitude_m"),
            pl.lit("observed").alias("data_nature"),
            pl.lit("INMET automatic station annual archive").alias("source"),
            pl.lit(source_checksum).alias("source_dataset_checksum"),
        )
        .sort("data")
    )


def parse_annual_zip(content: bytes, *, source_checksum: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            try:
                frame = parse_station_csv(archive.read(name), source_checksum=source_checksum)
            except (UnicodeError, ValueError, pl.exceptions.PolarsError):
                continue
            if not frame.is_empty():
                frames.append(frame)
    if not frames:
        raise ValueError("INMET annual archive produced no valid automatic-station rows")
    return pl.concat(frames, how="diagonal_relaxed")


def add_season_key(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.filter(pl.col("data").dt.month().is_in([1, 2, 3, 4, 9, 10, 11, 12])).with_columns(
        pl.when(pl.col("data").dt.month() >= 9)
        .then(pl.col("data").dt.year() + 1)
        .otherwise(pl.col("data").dt.year())
        .cast(pl.Int32)
        .alias("season_year")
    )


def _weighted_mean(value: str, weight: str, alias: str) -> pl.Expr:
    return (
        (pl.col(value) * pl.col(weight)).sum() / pl.col(weight).sum()
    ).alias(alias)


STATION_METADATA_TOLERANCE_KM = 1.0
STATION_METADATA_AUDIT_COLUMNS = [
    "station_metadata_versions",
    "station_metadata_max_shift_km",
    "station_metadata_tolerance_km",
    "station_metadata_policy",
]


def _canonicalize_season_metadata(
    seasonal: pl.DataFrame, *, tolerance_km: float
) -> pl.DataFrame:
    """Join annual metadata revisions without merging materially displaced stations.

    Station codes identify the observation stream, but annual INMET archives can
    revise names, altitude and coordinate precision. Exact metadata grouping split
    complete seasons into incomplete half-seasons, especially in 2019. Only merge
    revisions whose *maximum pairwise* coordinate distance is within the tolerance
    and whose UF is unchanged. Larger shifts retain their original segments.
    The daily source and every source checksum remain untouched.
    """
    if not np.isfinite(tolerance_km) or tolerance_km < 0:
        raise ValueError("Station metadata tolerance must be finite and nonnegative")
    keys = ["station_code", "season_year"]
    metadata = [
        "station_name",
        "station_uf",
        "station_latitude",
        "station_longitude",
        "station_altitude_m",
    ]
    versions = seasonal.select(*keys, *metadata).unique()
    policies: list[dict[str, object]] = []
    for group in versions.partition_by(keys, maintain_order=True):
        latitudes = group.get_column("station_latitude").to_numpy()
        longitudes = group.get_column("station_longitude").to_numpy()
        max_shift_km = float(
            _haversine_matrix_km(latitudes, longitudes, latitudes, longitudes).max()
        )
        uf_stable = group.get_column("station_uf").n_unique() == 1
        can_merge = np.isfinite(max_shift_km) and max_shift_km <= tolerance_km and uf_stable
        if group.height == 1:
            policy = "stable"
        elif not uf_stable:
            policy = "segmented_uf_change"
        elif can_merge:
            policy = "canonicalized_latest_within_tolerance"
        else:
            policy = "segmented_coordinate_shift_over_tolerance"
        policies.append(
            {
                "station_code": group.item(0, "station_code"),
                "season_year": group.item(0, "season_year"),
                "station_metadata_versions": group.height,
                "station_metadata_max_shift_km": max_shift_km,
                "station_metadata_tolerance_km": tolerance_km,
                "station_metadata_policy": policy,
                "_metadata_can_merge": bool(can_merge),
            }
        )
    policy_frame = pl.DataFrame(policies).with_columns(
        pl.col("season_year").cast(pl.Int32),
        pl.col("station_metadata_versions").cast(pl.UInt32),
    )
    latest = seasonal.sort("data").group_by(keys).agg(
        [pl.col(column).last().alias(f"{column}_latest") for column in metadata]
    )
    return (
        seasonal.join(policy_frame, on=keys, validate="m:1")
        .join(latest, on=keys, validate="m:1")
        .with_columns(
            [
                pl.when(pl.col("_metadata_can_merge"))
                .then(pl.col(f"{column}_latest"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in metadata
            ]
        )
        .drop("_metadata_can_merge", *[f"{column}_latest" for column in metadata])
    )


def build_station_season_features(
    daily: pl.DataFrame, *, metadata_tolerance_km: float = STATION_METADATA_TOLERANCE_KM
) -> pl.DataFrame:
    seasonal = _canonicalize_season_metadata(
        add_season_key(daily), tolerance_km=metadata_tolerance_km
    ).with_columns(
        (pl.col("rain_observations") > 0).alias("rain_valid_day"),
        ((pl.col("rain_observations") > 0) & (pl.col("rain_mm") < 1.0)).alias("dry_day"),
    )
    base = seasonal.group_by(
        [
            "station_code",
            "station_name",
            "station_uf",
            "station_latitude",
            "station_longitude",
            "station_altitude_m",
            "season_year",
        ]
    ).agg(
        pl.col("rain_mm").sum().alias("rain_crop_cycle_mm"),
        pl.col("rain_mm").filter(pl.col("data").dt.month() == 4).sum().alias("rain_30d_mm"),
        pl.col("rain_mm")
        .filter(pl.col("data").dt.month().is_in([3, 4]))
        .sum()
        .alias("rain_60d_mm"),
        pl.col("rain_mm")
        .filter(pl.col("data").dt.month().is_in([2, 3, 4]))
        .sum()
        .alias("rain_90d_mm"),
        pl.col("dry_day").sum().alias("dry_days"),
        (pl.col("max_temp_c") > 35).sum().alias("days_temp_gt_35"),
        (pl.col("min_temp_c") < 10).sum().alias("days_temp_lt_10"),
        _weighted_mean("mean_temp_c", "temp_observations", "mean_temp_c"),
        pl.col("max_temp_c").max().alias("max_temp_c"),
        pl.col("min_temp_c").min().alias("min_temp_c"),
        _weighted_mean("mean_humidity_pct", "humidity_observations", "mean_humidity_pct"),
        pl.col("radiation_kj_m2").sum().alias("radiation_sum_kj_m2"),
        _weighted_mean("mean_wind_m_s", "wind_observations", "mean_wind_m_s"),
        pl.col("temp_observations").sum().alias("valid_temp_hours"),
        pl.col("rain_observations").sum().alias("valid_rain_hours"),
        pl.col("data").n_unique().alias("calendar_days_present"),
        pl.col("source_dataset_checksum").unique().sort().alias("source_checksums"),
        *[pl.col(column).first() for column in STATION_METADATA_AUDIT_COLUMNS],
    )
    expected_hours = pl.col("season_year").map_elements(
        lambda year: sum(
            monthrange(year - 1 if month >= 9 else year, month)[1] * 24
            for month in [9, 10, 11, 12, 1, 2, 3, 4]
        ),
        return_dtype=pl.Int32,
    )
    return (
        base.with_columns(expected_hours.alias("expected_hours"))
        .with_columns(
            (pl.col("valid_temp_hours") / pl.col("expected_hours")).alias("temp_coverage"),
            (pl.col("valid_rain_hours") / pl.col("expected_hours")).alias("rain_coverage"),
            pl.lit("observed").alias("data_nature"),
            pl.lit("INMET automatic station annual archives").alias("source"),
        )
        .with_columns(
            pl.min_horizontal("temp_coverage", "rain_coverage").alias("season_coverage")
        )
        .sort(["season_year", "station_code"])
    )


def _haversine_matrix_km(
    latitudes_a: NDArray[np.float64],
    longitudes_a: NDArray[np.float64],
    latitudes_b: NDArray[np.float64],
    longitudes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    lat_a = np.radians(latitudes_a)[:, None]
    lon_a = np.radians(longitudes_a)[:, None]
    lat_b = np.radians(latitudes_b)[None, :]
    lon_b = np.radians(longitudes_b)[None, :]
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(
        delta_lon / 2
    ) ** 2
    result = 6_371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))
    return cast(NDArray[np.float64], result)


def map_seasons_to_municipalities(
    station_seasons: pl.DataFrame,
    locations: pl.DataFrame,
    *,
    max_distance_km: float,
    minimum_coverage: float,
) -> pl.DataFrame:
    municipalities = locations.filter(
        pl.col("latitude_centroid").is_not_null() & pl.col("longitude_centroid").is_not_null()
    ).select("codigo_ibge", "municipio", "uf", "latitude_centroid", "longitude_centroid")
    municipality_latitudes = municipalities.get_column("latitude_centroid").to_numpy()
    municipality_longitudes = municipalities.get_column("longitude_centroid").to_numpy()
    outputs: list[pl.DataFrame] = []
    feature_columns = [
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
        "season_coverage",
    ]
    for season_year in sorted(station_seasons.get_column("season_year").unique().to_list()):
        stations = station_seasons.filter(
            (pl.col("season_year") == season_year)
            & (pl.col("season_coverage") >= minimum_coverage)
        )
        if stations.is_empty():
            continue
        distances = _haversine_matrix_km(
            municipality_latitudes,
            municipality_longitudes,
            stations.get_column("station_latitude").to_numpy(),
            stations.get_column("station_longitude").to_numpy(),
        )
        nearest_indices = distances.argmin(axis=1)
        nearest_distances = distances[np.arange(distances.shape[0]), nearest_indices]
        selected = stations[nearest_indices.tolist()].select(
            "station_code",
            "station_name",
            "station_uf",
            *feature_columns,
            "source_checksums",
            *[
                column
                for column in STATION_METADATA_AUDIT_COLUMNS
                if column in stations.columns
            ],
        )
        output = pl.concat([municipalities, selected], how="horizontal_extend").with_columns(
            pl.lit(season_year).cast(pl.Int32).alias("season_year"),
            pl.Series("distance_to_station_km", nearest_distances),
        )
        output = output.with_columns(
            pl.when(pl.col("distance_to_station_km") <= max_distance_km)
            .then(pl.lit("interpolated"))
            .otherwise(pl.lit("estimated"))
            .alias("data_nature"),
            pl.when(pl.col("distance_to_station_km") <= max_distance_km)
            .then(pl.lit("nearest INMET automatic station within configured radius"))
            .otherwise(pl.lit("nearest station exceeds configured radius; features withheld"))
            .alias("method"),
            pl.when(pl.col("distance_to_station_km") <= max_distance_km)
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(pl.lit("NO_NEARBY_INMET_STATION"))
            .alias("quality_flag"),
            pl.lit("INMET automatic station annual archives").alias("source"),
        )
        output = output.with_columns(
            [
                pl.when(pl.col("distance_to_station_km") <= max_distance_km)
                .then(pl.col(column))
                .otherwise(None)
                .alias(column)
                for column in feature_columns
            ]
        )
        outputs.append(output)
    if not outputs:
        raise ValueError("No station season met the configured climate coverage threshold")
    return pl.concat(outputs, how="diagonal_relaxed").sort(["codigo_ibge", "season_year"])
