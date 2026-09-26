from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import mlflow
import numpy as np
import polars as pl
from catboost import CatBoostRegressor  # type: ignore[import-untyped]
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.metrics import r2_score  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from xgboost import XGBRegressor

from agri_decision.config import Settings, load_settings
from agri_decision.features.training import MODEL_FEATURES
from agri_decision.provenance.artifacts import sha256_file, write_json_atomic

TARGET = "produtividade_kg_ha"
BASELINE_COLUMNS = {
    "historical_municipal_median": "baseline_municipal_median_5y",
    "moving_average_5y": "yield_mean_5y",
    "regional_median_lag1": "baseline_regional_median_lag1",
}
CLIMATE_SOURCE_COLUMN = "climate_lag1_source"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    observed = y_true[valid]
    predicted = y_pred[valid]
    if observed.size == 0:
        raise ValueError("No finite observations available for metric calculation")
    errors = predicted - observed
    denominator = np.abs(observed) + np.abs(predicted)
    smape_terms = np.divide(
        2 * np.abs(errors),
        denominator,
        out=np.zeros_like(errors),
        where=denominator > 1e-9,
    )
    return {
        "rows": int(observed.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "smape_pct": float(np.mean(smape_terms) * 100),
        "r2": float(r2_score(observed, predicted)) if observed.size > 1 else float("nan"),
    }


def _arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = frame.select(MODEL_FEATURES).to_numpy().astype(np.float64)
    target = frame.get_column(TARGET).to_numpy().astype(np.float64)
    return features, target


def _baseline_metrics(frame: pl.DataFrame) -> dict[str, dict[str, float | int]]:
    target = frame.get_column(TARGET).to_numpy().astype(np.float64)
    return {
        name: regression_metrics(
            target,
            frame.get_column(column).to_numpy().astype(np.float64),
        )
        for name, column in BASELINE_COLUMNS.items()
    }


def _climate_source_metrics(
    frame: pl.DataFrame, predictions: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Describe each source domain using identical finite rows for all comparisons.

    These are subgroup diagnostics, not additional model-selection/promotion gates.
    Missing provenance remains visible instead of silently dropping those rows.
    """
    if predictions.ndim != 1 or predictions.size != frame.height:
        raise ValueError("Predictions must have exactly one value per evaluation row")
    source_values = (
        frame.get_column(CLIMATE_SOURCE_COLUMN).to_list()
        if CLIMATE_SOURCE_COLUMN in frame.columns
        else [None] * frame.height
    )
    sources = np.asarray(
        [
            str(value).strip() if value is not None and str(value).strip() else "unknown"
            for value in source_values
        ],
        dtype=object,
    )
    target = frame.get_column(TARGET).to_numpy().astype(np.float64)
    baseline_predictions = {
        name: frame.get_column(column).to_numpy().astype(np.float64)
        for name, column in BASELINE_COLUMNS.items()
    }
    common_finite = np.isfinite(target) & np.isfinite(predictions)
    for baseline in baseline_predictions.values():
        common_finite &= np.isfinite(baseline)
    result: dict[str, dict[str, Any]] = {}
    for source in sorted(set(sources)):
        source_mask = sources == source
        evaluated = source_mask & common_finite
        total_rows = int(np.sum(source_mask))
        evaluated_rows = int(np.sum(evaluated))
        record: dict[str, Any] = {
            "status": "evaluated" if evaluated_rows else "no_common_finite_rows",
            "total_rows": total_rows,
            "evaluated_rows": evaluated_rows,
            "excluded_rows": total_rows - evaluated_rows,
            "model": None,
            "baselines": {},
            "best_baseline": None,
            "beats_best_baseline": None,
        }
        if evaluated_rows:
            model_metrics = regression_metrics(target[evaluated], predictions[evaluated])
            baselines = {
                name: regression_metrics(target[evaluated], baseline[evaluated])
                for name, baseline in baseline_predictions.items()
            }
            best_name, best_metrics = min(
                baselines.items(), key=lambda item: float(item[1]["mae"])
            )
            record.update(
                {
                    "model": model_metrics,
                    "baselines": baselines,
                    "best_baseline": {"name": best_name, "metrics": best_metrics},
                    "beats_best_baseline": float(model_metrics["mae"])
                    < float(best_metrics["mae"]),
                }
            )
        result[source] = record
    return result


def _geographic_holdout(code: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{code}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def _factory(name: str, seed: int) -> Callable[[], Any]:
    if name not in {"catboost", "lightgbm", "xgboost"}:
        raise ValueError(f"Unknown model candidate: {name}")

    def build() -> Any:
        if name == "catboost":
            estimator: Any = CatBoostRegressor(
                iterations=350,
                learning_rate=0.04,
                depth=7,
                loss_function="RMSE",
                random_seed=seed,
                l2_leaf_reg=5,
                verbose=False,
                allow_writing_files=False,
                thread_count=4,
            )
        elif name == "lightgbm":
            estimator = LGBMRegressor(
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
        else:
            estimator = XGBRegressor(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=4,
                tree_method="hist",
            )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator),
            ]
        )

    return build


def _fit_metrics(
    factory: Callable[[], Any], train: pl.DataFrame, test: pl.DataFrame
) -> tuple[Any, dict[str, float | int], dict[str, dict[str, Any]]]:
    train_x, train_y = _arrays(train)
    test_x, test_y = _arrays(test)
    model = factory()
    model.fit(train_x, train_y)
    predictions = np.asarray(model.predict(test_x), dtype=np.float64)
    return (
        model,
        regression_metrics(test_y, predictions),
        _climate_source_metrics(test, predictions),
    )


def _conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    finite = np.abs(residuals[np.isfinite(residuals)])
    if finite.size == 0:
        raise ValueError("Conformal calibration has no finite residuals")
    adjusted = min(1.0, math.ceil((finite.size + 1) * coverage) / finite.size)
    return float(np.quantile(finite, adjusted, method="higher"))


def _evaluate_intervals(
    factory: Callable[[], Any], frame: pl.DataFrame, *, split_year: int, coverage: float
) -> dict[str, Any]:
    fit = frame.filter(pl.col("ano") < split_year)
    calibration = frame.filter(pl.col("ano") == split_year)
    evaluation = frame.filter(pl.col("ano") > split_year)
    if fit.is_empty() or calibration.is_empty() or evaluation.is_empty():
        raise ValueError("Temporal interval evaluation needs fit, calibration and evaluation years")
    model = factory()
    fit_x, fit_y = _arrays(fit)
    calibration_x, calibration_y = _arrays(calibration)
    evaluation_x, evaluation_y = _arrays(evaluation)
    model.fit(fit_x, fit_y)
    calibration_prediction = np.asarray(model.predict(calibration_x), dtype=np.float64)
    radius = _conformal_radius(calibration_y - calibration_prediction, coverage)
    evaluation_prediction = np.asarray(model.predict(evaluation_x), dtype=np.float64)
    lower = np.maximum(0.0, evaluation_prediction - radius)
    upper = evaluation_prediction + radius
    empirical = np.mean((evaluation_y >= lower) & (evaluation_y <= upper))
    return {
        "method": "split_conformal_absolute_residual",
        "nominal_coverage": coverage,
        "calibration_year": split_year,
        "evaluation_years": sorted(evaluation.get_column("ano").unique().to_list()),
        "calibration_rows": calibration.height,
        "evaluation_rows": evaluation.height,
        "evaluation_radius_kg_ha": radius,
        "empirical_coverage": float(empirical),
        "mean_interval_width_kg_ha": float(np.mean(upper - lower)),
        "point_metrics": regression_metrics(evaluation_y, evaluation_prediction),
    }


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    manifest = dataset_path.with_suffix(".manifest.json")
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    checksum = sha256_file(dataset_path)
    return {"dataset_version": f"ml-soy-{checksum[:12]}", "checksum_sha256": checksum}


def _mlflow_tracking_uri(settings: Settings) -> str:
    database = (settings.paths.models / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def _log_candidate(
    *,
    settings: Settings,
    name: str,
    record: dict[str, Any],
    dataset: dict[str, Any],
    git_commit: str,
) -> str:
    mlflow.set_tracking_uri(_mlflow_tracking_uri(settings))
    mlflow.set_experiment("soy-yield")
    with mlflow.start_run(run_name=f"soy-yield-{name}") as run:
        mlflow.set_tags(
            {
                "crop": "soja",
                "target": TARGET,
                "dataset_version": str(dataset["dataset_version"]),
                "git_commit": git_commit,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        mlflow.log_params(
            {
                "model": name,
                "feature_count": len(MODEL_FEATURES),
                "temporal_test_start_year": settings.ml.temporal_test_start_year,
                "geographic_holdout_fraction": settings.ml.geographic_holdout_fraction,
                "random_seed": settings.ml.random_seed,
            }
        )
        for split in ["temporal", "geographic"]:
            metrics = record[split]
            for metric in ["mae", "rmse", "smape_pct", "r2"]:
                mlflow.log_metric(f"{split}_{metric}", float(metrics[metric]))
        mlflow.log_dict({"features": MODEL_FEATURES, **record}, "evaluation.json")
        return str(run.info.run_id)


def train_and_evaluate_soy_yield(
    *, settings: Settings | None = None
) -> tuple[Path, Path | None]:
    resolved = settings or load_settings()
    dataset_path = resolved.paths.gold / "ml_soy_yield_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("ML dataset missing; run `agri build ml-dataset` first")
    frame = pl.read_parquet(dataset_path).sort(["ano", "codigo_ibge"])
    temporal_train = frame.filter(pl.col("ano") < resolved.ml.temporal_test_start_year)
    temporal_test = frame.filter(pl.col("ano") >= resolved.ml.temporal_test_start_year)
    holdout_codes = {
        str(code)
        for code in frame.get_column("codigo_ibge").unique().to_list()
        if _geographic_holdout(
            str(code),
            seed=resolved.ml.random_seed,
            fraction=resolved.ml.geographic_holdout_fraction,
        )
    }
    geographic_test = frame.filter(pl.col("codigo_ibge").is_in(holdout_codes))
    geographic_train = frame.filter(~pl.col("codigo_ibge").is_in(holdout_codes))
    if any(
        split.is_empty()
        for split in [temporal_train, temporal_test, geographic_train, geographic_test]
    ):
        raise ValueError("Configured temporal or geographic split is empty")

    baselines = {
        "temporal": _baseline_metrics(temporal_test),
        "geographic": _baseline_metrics(geographic_test),
    }
    best_baseline = {
        split: min(values.items(), key=lambda item: float(item[1]["mae"]))
        for split, values in baselines.items()
    }
    dataset = _dataset_metadata(dataset_path)
    git_commit = _git_commit(resolved.project_root)
    candidates: dict[str, dict[str, Any]] = {}
    mlflow_runs: dict[str, str] = {}
    for name in ["catboost", "lightgbm", "xgboost"]:
        factory = _factory(name, resolved.ml.random_seed)
        temporal_model, temporal_metrics, temporal_source_metrics = _fit_metrics(
            factory, temporal_train, temporal_test
        )
        del temporal_model
        geographic_model, geographic_metrics, geographic_source_metrics = _fit_metrics(
            factory, geographic_train, geographic_test
        )
        del geographic_model
        baseline_temporal_mae = float(best_baseline["temporal"][1]["mae"])
        baseline_geographic_mae = float(best_baseline["geographic"][1]["mae"])
        beats_temporal = float(temporal_metrics["mae"]) < baseline_temporal_mae
        beats_geographic = float(geographic_metrics["mae"]) < baseline_geographic_mae
        record = {
            "model": name,
            "temporal": temporal_metrics,
            "geographic": geographic_metrics,
            "by_climate_source": {
                "temporal": temporal_source_metrics,
                "geographic": geographic_source_metrics,
            },
            "baseline_comparison": {
                "temporal_best_baseline": best_baseline["temporal"][0],
                "temporal_best_baseline_mae": baseline_temporal_mae,
                "temporal_mae_ratio": float(temporal_metrics["mae"])
                / baseline_temporal_mae,
                "geographic_best_baseline": best_baseline["geographic"][0],
                "geographic_best_baseline_mae": baseline_geographic_mae,
                "geographic_mae_ratio": float(geographic_metrics["mae"])
                / baseline_geographic_mae,
                "beats_temporal": beats_temporal,
                "beats_geographic": beats_geographic,
                "beats_baseline": beats_temporal and beats_geographic,
            },
            "selection_score": (
                float(temporal_metrics["mae"]) / baseline_temporal_mae
                + float(geographic_metrics["mae"]) / baseline_geographic_mae
            )
            / 2,
        }
        candidates[name] = record
        mlflow_runs[name] = _log_candidate(
            settings=resolved,
            name=name,
            record=record,
            dataset=dataset,
            git_commit=git_commit,
        )

    selected_name, selected = min(
        candidates.items(), key=lambda item: float(item[1]["selection_score"])
    )
    promoted = bool(selected["baseline_comparison"]["beats_baseline"])
    interval_evaluation = _evaluate_intervals(
        _factory(selected_name, resolved.ml.random_seed),
        frame,
        split_year=resolved.ml.temporal_test_start_year,
        coverage=resolved.ml.prediction_interval_coverage,
    )
    model_path: Path | None = None
    production_interval: dict[str, Any] | None = None
    version: str | None = None
    if promoted:
        latest_year_value: Any = frame.get_column("ano").max()
        if latest_year_value is None:
            raise ValueError("ML dataset has no training year")
        latest_year = int(latest_year_value)
        production_train = frame.filter(pl.col("ano") < latest_year)
        production_calibration = frame.filter(pl.col("ano") == latest_year)
        final_model = _factory(selected_name, resolved.ml.random_seed)()
        production_x, production_y = _arrays(production_train)
        calibration_x, calibration_y = _arrays(production_calibration)
        final_model.fit(production_x, production_y)
        calibration_prediction = np.asarray(
            final_model.predict(calibration_x), dtype=np.float64
        )
        production_radius = _conformal_radius(
            calibration_y - calibration_prediction,
            resolved.ml.prediction_interval_coverage,
        )
        production_interval = {
            "method": "split_conformal_absolute_residual",
            "coverage": resolved.ml.prediction_interval_coverage,
            "calibration_year": latest_year,
            "calibration_rows": production_calibration.height,
            "radius_kg_ha": production_radius,
        }
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        version = (
            f"soja-{selected_name}-{timestamp}-"
            f"{str(dataset['checksum_sha256'])[:8]}"
        )
        artifact_dir = resolved.paths.models / "yield" / "soja" / version
        artifact_dir.mkdir(parents=True, exist_ok=False)
        model_path = artifact_dir / "model.joblib"
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(final_model, temporary)
        temporary.replace(model_path)
        average_improvement = 1.0 - float(selected["selection_score"])
        coverage_gap = abs(
            float(interval_evaluation["empirical_coverage"])
            - resolved.ml.prediction_interval_coverage
        )
        performance_confidence = max(
            0.4, min(0.95, 0.65 + average_improvement - coverage_gap)
        )
        metadata = {
            "model_version": version,
            "crop": "soja",
            "model": selected_name,
            "promoted": True,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "features": MODEL_FEATURES,
            "target": TARGET,
            "training_years": sorted(production_train.get_column("ano").unique().to_list()),
            "training_rows": production_train.height,
            "production_interval": production_interval,
            "interval_evaluation": interval_evaluation,
            "performance": selected,
            "performance_confidence": performance_confidence,
            "git_commit": git_commit,
            "pipeline_version": resolved.pipeline_version,
            "config_version": resolved.config_version,
            "mlflow_run_id": mlflow_runs[selected_name],
            "artifact_checksum_sha256": sha256_file(model_path),
        }
        write_json_atomic(artifact_dir / "metadata.json", metadata)
        write_json_atomic(
            resolved.paths.models / "yield" / "soja" / "latest.json",
            {
                "model_version": version,
                "artifact": str(model_path.relative_to(resolved.project_root)),
                "metadata": str(
                    (artifact_dir / "metadata.json").relative_to(resolved.project_root)
                ),
                "promoted": True,
            },
        )
        mlflow.set_tracking_uri(_mlflow_tracking_uri(resolved))
        with mlflow.start_run(run_id=mlflow_runs[selected_name]):
            mlflow.log_artifact(str(model_path), artifact_path="promoted_model")
            mlflow.log_artifact(
                str(artifact_dir / "metadata.json"), artifact_path="promoted_model"
            )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "crop": "soja",
        "target": TARGET,
        "dataset": dataset,
        "features": MODEL_FEATURES,
        "git_commit": git_commit,
        "splits": {
            "temporal": {
                "train_years": sorted(temporal_train.get_column("ano").unique().to_list()),
                "test_years": sorted(temporal_test.get_column("ano").unique().to_list()),
                "train_rows": temporal_train.height,
                "test_rows": temporal_test.height,
            },
            "geographic": {
                "train_municipalities": geographic_train.get_column("codigo_ibge").n_unique(),
                "test_municipalities": geographic_test.get_column("codigo_ibge").n_unique(),
                "train_rows": geographic_train.height,
                "test_rows": geographic_test.height,
                "holdout_fraction_configured": resolved.ml.geographic_holdout_fraction,
            },
        },
        "baselines": baselines,
        "climate_source_evaluation": {
            "column": CLIMATE_SOURCE_COLUMN,
            "comparison_rows": (
                "common finite target, model prediction and all baseline predictions"
            ),
            "missing_source_label": "unknown",
            "descriptive_only": True,
            "promotion_gate_applied": False,
            "limitation": "Subgroup metrics do not establish performance in unrepresented domains",
        },
        "candidates": candidates,
        "winner": selected_name,
        "promotion_gate": {
            "rule": "winner MAE must be below the best baseline in temporal and geographic tests",
            "passed": promoted,
            "model_version": version,
        },
        "interval_evaluation": interval_evaluation,
        "production_interval": production_interval,
        "mlflow": {
            "tracking_uri": _mlflow_tracking_uri(resolved),
            "experiment": "soy-yield",
            "run_ids": mlflow_runs,
        },
        "leakage_contract": {
            "history": "strictly shifted before the target year",
            "regional": "UF median from target year minus one",
            "climate": "completed crop season from target year minus one",
            "current_area_and_production": "excluded from model features",
        },
    }
    report_path = resolved.paths.reports / "model_evaluation_soja.json"
    write_json_atomic(report_path, report)
    return report_path, model_path
