from __future__ import annotations

import polars as pl

HISTORY_KEYS = ["codigo_ibge", "cultura"]


def _rolling_slope(window: int) -> pl.Expr:
    x = pl.col("_history_year")
    y = pl.col("_history_yield")
    mean_x = x.rolling_mean(window_size=window, min_samples=2).over(HISTORY_KEYS)
    mean_y = y.rolling_mean(window_size=window, min_samples=2).over(HISTORY_KEYS)
    mean_xy = (x * y).rolling_mean(window_size=window, min_samples=2).over(HISTORY_KEYS)
    mean_x2 = (x * x).rolling_mean(window_size=window, min_samples=2).over(HISTORY_KEYS)
    denominator = mean_x2 - (mean_x * mean_x)
    return (
        pl.when(denominator.abs() > 1e-12)
        .then((mean_xy - (mean_x * mean_y)) / denominator)
        .otherwise(None)
    )


def add_leakage_safe_history_features(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "codigo_ibge",
        "cultura",
        "ano",
        "produtividade_kg_ha",
        "area_plantada_ha",
        "producao_t",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing production columns for feature engineering: {missing}")

    ordered = frame.sort([*HISTORY_KEYS, "ano"])
    with_history = ordered.with_columns(
        pl.col("ano").cast(pl.Float64).shift(1).over(HISTORY_KEYS).alias("_history_year"),
        pl.col("produtividade_kg_ha").shift(1).over(HISTORY_KEYS).alias("_history_yield"),
        pl.col("area_plantada_ha").shift(1).over(HISTORY_KEYS).alias("_history_area"),
        pl.col("producao_t").shift(1).over(HISTORY_KEYS).alias("_history_production"),
    )

    features = with_history.with_columns(
        pl.col("_history_yield")
        .rolling_mean(window_size=3, min_samples=1)
        .over(HISTORY_KEYS)
        .alias("yield_mean_3y"),
        pl.col("_history_yield")
        .rolling_mean(window_size=5, min_samples=1)
        .over(HISTORY_KEYS)
        .alias("yield_mean_5y"),
        pl.col("_history_yield")
        .rolling_std(window_size=5, min_samples=2)
        .over(HISTORY_KEYS)
        .alias("yield_std_5y"),
        pl.col("_history_area")
        .rolling_mean(window_size=5, min_samples=1)
        .over(HISTORY_KEYS)
        .alias("area_mean_5y"),
        pl.col("_history_production")
        .rolling_mean(window_size=5, min_samples=1)
        .over(HISTORY_KEYS)
        .alias("production_mean_5y"),
        pl.col("_history_yield")
        .is_not_null()
        .cast(pl.Int32)
        .cum_sum()
        .over(HISTORY_KEYS)
        .alias("crop_presence_years"),
        _rolling_slope(5).alias("yield_trend_5y"),
        _rolling_slope(10).alias("yield_trend_10y"),
    ).with_columns(
        pl.when(pl.col("yield_mean_5y").abs() > 1e-12)
        .then(pl.col("yield_std_5y") / pl.col("yield_mean_5y"))
        .otherwise(None)
        .alias("yield_cv_5y")
    )

    return features.drop(
        "_history_year",
        "_history_yield",
        "_history_area",
        "_history_production",
    )
