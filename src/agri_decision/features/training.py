from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.features.history import add_leakage_safe_history_features
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.provenance.artifacts import sha256_file, write_json_atomic

CLIMATE_FEATURES = [
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
    "distance_to_station_km",
]

CLIMATE_METADATA = ["source", "data_nature", "method", "quality_flag", "distance_to_grid_km"]


def _lagged_climate_columns(climate: pl.DataFrame) -> pl.DataFrame:
    """Carry provenance alongside features, without adding it to the model contract."""
    defaults: dict[str, pl.Expr] = {
        "source": pl.lit("INMET automatic station annual archives"),
        "data_nature": pl.when(pl.col("rain_crop_cycle_mm").is_not_null())
        .then(pl.lit("interpolated"))
        .otherwise(pl.lit("not_available")),
        "method": pl.lit(None, dtype=pl.Utf8),
        "quality_flag": pl.lit(None, dtype=pl.Utf8),
        "distance_to_grid_km": pl.lit(None, dtype=pl.Float64),
    }
    normalized = climate.with_columns(
        [
            expression.alias(name)
            for name, expression in defaults.items()
            if name not in climate.columns
        ]
    )
    return normalized.select(
        "codigo_ibge", "season_year", *CLIMATE_FEATURES, *CLIMATE_METADATA
    ).rename(
        {column: f"climate_lag1_{column}" for column in [*CLIMATE_FEATURES, *CLIMATE_METADATA]}
    )


MODEL_FEATURES = [
    "ano",
    "latitude_centroid",
    "longitude_centroid",
    "yield_mean_3y",
    "yield_mean_5y",
    "yield_std_5y",
    "yield_cv_5y",
    "yield_trend_5y",
    "yield_trend_10y",
    "area_mean_5y",
    "production_mean_5y",
    "crop_presence_years",
    "baseline_municipal_median_5y",
    "baseline_regional_median_lag1",
    *[f"climate_lag1_{column}" for column in CLIMATE_FEATURES],
]


def build_soy_yield_inference_frame(
    history: pl.DataFrame,
    climate: pl.DataFrame,
    locations: pl.DataFrame,
    *,
    codigo_ibge: str,
    target_year: int,
) -> pl.DataFrame:
    """Build one future row with only information available before ``target_year``."""
    local = history.filter(
        (pl.col("codigo_ibge") == codigo_ibge) & (pl.col("cultura") == "soja")
    ).sort("ano")
    if local.is_empty():
        raise ValueError(f"No soybean history for municipality {codigo_ibge}")
    latest_year: Any = local.get_column("ano").max()
    if latest_year is None or target_year <= int(latest_year):
        raise ValueError("Inference target year must be after the latest production year")

    placeholder = local.tail(1).with_columns(
        pl.lit(target_year).cast(local.schema["ano"]).alias("ano"),
        pl.lit(None).cast(pl.Float64).alias("area_plantada_ha"),
        pl.lit(None).cast(pl.Float64).alias("area_colhida_ha"),
        pl.lit(None).cast(pl.Float64).alias("producao_t"),
        pl.lit(None).cast(pl.Float64).alias("produtividade_kg_ha"),
        pl.lit(None).cast(pl.Float64).alias("valor_producao_brl"),
    )
    featured = add_leakage_safe_history_features(pl.concat([local, placeholder])).filter(
        pl.col("ano") == target_year
    )

    prior_yields = (
        local.filter(
            (pl.col("ano") < target_year)
            & pl.col("produtividade_kg_ha").is_not_null()
            & (pl.col("produtividade_kg_ha") > 0)
        )
        .sort("ano")
        .tail(5)
        .get_column("produtividade_kg_ha")
    )
    municipal_median = prior_yields.median() if len(prior_yields) >= 3 else None
    uf = str(local.item(-1, "uf"))
    regional_median = (
        history.filter(
            (pl.col("cultura") == "soja")
            & (pl.col("uf") == uf)
            & (pl.col("ano") == target_year - 1)
            & pl.col("produtividade_kg_ha").is_not_null()
            & (pl.col("produtividade_kg_ha") > 0)
        )
        .get_column("produtividade_kg_ha")
        .median()
    )
    climate_row = (
        _lagged_climate_columns(climate)
        .filter((pl.col("codigo_ibge") == codigo_ibge) & (pl.col("season_year") == target_year - 1))
        .rename({"season_year": "climate_reference_season"})
    )
    location_row = locations.filter(pl.col("codigo_ibge") == codigo_ibge).select(
        "codigo_ibge", "latitude_centroid", "longitude_centroid"
    )
    if location_row.is_empty():
        raise ValueError(f"No centroid for municipality {codigo_ibge}")

    return (
        featured.with_columns(
            pl.lit(municipal_median).cast(pl.Float64).alias("baseline_municipal_median_5y"),
            pl.lit(regional_median).cast(pl.Float64).alias("baseline_regional_median_lag1"),
        )
        .join(climate_row, on="codigo_ibge", how="left")
        .join(location_row, on="codigo_ibge", how="left")
    )


def build_soy_yield_training_frame(
    history: pl.DataFrame,
    climate: pl.DataFrame,
    locations: pl.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pl.DataFrame:
    ordered = history.filter(pl.col("cultura") == "soja").sort(["codigo_ibge", "ano"])
    ordered = ordered.with_columns(
        pl.col("produtividade_kg_ha")
        .shift(1)
        .rolling_median(window_size=5, min_samples=3)
        .over("codigo_ibge")
        .alias("baseline_municipal_median_5y")
    )
    regional = (
        ordered.filter(pl.col("produtividade_kg_ha").is_not_null())
        .group_by(["uf", "ano"])
        .agg(pl.col("produtividade_kg_ha").median().alias("baseline_regional_median_lag1"))
        .with_columns((pl.col("ano") + 1).alias("ano"))
    )
    climate_lag = (
        _lagged_climate_columns(climate)
        .with_columns((pl.col("season_year") + 1).alias("ano"))
        .rename({"season_year": "climate_reference_season"})
    )
    location_features = locations.select("codigo_ibge", "latitude_centroid", "longitude_centroid")
    return (
        ordered.filter(
            pl.col("ano").is_between(start_year, end_year)
            & pl.col("produtividade_kg_ha").is_not_null()
            & (pl.col("produtividade_kg_ha") > 0)
        )
        .join(regional, on=["uf", "ano"], how="left")
        .join(climate_lag, on=["codigo_ibge", "ano"], how="left")
        .join(location_features, on="codigo_ibge", how="left")
        .with_columns(
            pl.lit("produtividade_kg_ha").alias("target_name"),
            pl.lit("all features available before target harvest year").alias("leakage_contract"),
            pl.when(pl.col("climate_lag1_rain_crop_cycle_mm").is_not_null())
            .then(pl.col("climate_lag1_data_nature"))
            .otherwise(pl.lit("not_available"))
            .alias("climate_data_nature"),
        )
        .sort(["ano", "codigo_ibge"])
    )


def materialize_soy_yield_training_dataset(
    *, settings: Settings | None = None
) -> tuple[Path, Path]:
    resolved = settings or load_settings()
    history_path = resolved.paths.gold / "crop_location_year.parquet"
    climate_path = resolved.paths.gold / "climate_location_season.parquet"
    location_path = resolved.paths.silver / "dim_location.parquet"
    missing = [
        path.name for path in [history_path, climate_path, location_path] if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required ML inputs missing: {', '.join(missing)}")
    frame = build_soy_yield_training_frame(
        pl.read_parquet(history_path),
        pl.read_parquet(climate_path),
        pl.read_parquet(location_path),
        start_year=resolved.ml.training_start_year,
        end_year=resolved.ml.training_end_year,
    )
    if frame.is_empty():
        raise ValueError("Soybean ML dataset is empty")
    output = resolved.paths.gold / "ml_soy_yield_dataset.parquet"
    _write_parquet_atomic(frame, output)
    checksum = sha256_file(output)
    missing_by_feature = {
        feature: frame.get_column(feature).null_count() for feature in MODEL_FEATURES
    }
    target = frame.get_column("produtividade_kg_ha")
    report = {
        "dataset": "ml_soy_yield_dataset",
        "dataset_version": f"ml-soy-{checksum[:12]}",
        "checksum_sha256": checksum,
        "shape": [frame.height, frame.width],
        "rows": frame.height,
        "columns": frame.width,
        "features": MODEL_FEATURES,
        "target": "produtividade_kg_ha",
        "year_min": frame.get_column("ano").min(),
        "year_max": frame.get_column("ano").max(),
        "years": frame.get_column("ano").unique().sort().to_list(),
        "municipalities": frame.get_column("codigo_ibge").n_unique(),
        "ufs": frame.get_column("uf").n_unique(),
        "missing_by_feature": missing_by_feature,
        "climate_sources": frame.group_by("climate_lag1_source")
        .len()
        .sort("climate_lag1_source")
        .to_dicts(),
        "climate_data_natures": frame.group_by("climate_data_nature")
        .len()
        .sort("climate_data_nature")
        .to_dicts(),
        "target_distribution": {
            "min": target.min(),
            "p10": target.quantile(0.10),
            "median": target.median(),
            "mean": target.mean(),
            "p90": target.quantile(0.90),
            "max": target.max(),
        },
        "leakage_contract": {
            "history": "rolling history is shifted one observation before target year",
            "regional_baseline": "UF median from year Y-1",
            "climate": "completed crop season Y-1, never target season Y",
            "contemporary_area_production": "excluded from model features",
        },
        "source_inputs": {
            history_path.name: sha256_file(history_path),
            climate_path.name: sha256_file(climate_path),
            location_path.name: sha256_file(location_path),
        },
    }
    report_path = resolved.paths.reports / "ml_soy_dataset_quality.json"
    write_json_atomic(report_path, report)
    write_json_atomic(
        output.with_suffix(".manifest.json"),
        {
            "dataset": report["dataset"],
            "dataset_version": report["dataset_version"],
            "checksum_sha256": checksum,
            "pipeline_version": resolved.pipeline_version,
            "rows": frame.height,
            "feature_contract_version": "soy-yield-v1-lagged-climate",
            "climate_sources": frame.filter(pl.col("climate_lag1_rain_crop_cycle_mm").is_not_null())
            .get_column("climate_lag1_source")
            .drop_nulls()
            .unique()
            .sort()
            .to_list(),
            "quality_report": str(report_path.relative_to(resolved.project_root)),
            "source_inputs": report["source_inputs"],
        },
    )
    return output, report_path
