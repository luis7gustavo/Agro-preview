import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from agri_decision.config import load_settings
from agri_decision.features.training import (
    CLIMATE_FEATURES,
    MODEL_FEATURES,
    build_soy_yield_inference_frame,
)
from agri_decision.models.yield_model.training import (
    BASELINE_COLUMNS,
    _climate_source_metrics,
    regression_metrics,
    train_and_evaluate_soy_yield,
)
from agri_decision.provenance.artifacts import write_json_atomic


def _history() -> pl.DataFrame:
    codes = ["1"] * 3 + ["2"] * 3
    years = [2020, 2021, 2022] * 2
    yields = [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 2200.0]
    return pl.DataFrame(
        {
            "codigo_ibge": codes,
            "municipio": ["A"] * 3 + ["B"] * 3,
            "uf": ["MT"] * 6,
            "ano": years,
            "cultura": ["soja"] * 6,
            "sistema_produtivo": ["rainfed"] * 6,
            "area_plantada_ha": [100.0] * 6,
            "area_colhida_ha": [100.0] * 6,
            "producao_t": [value / 10 for value in yields],
            "produtividade_kg_ha": yields,
            "valor_producao_brl": [1000.0] * 6,
        }
    )


def test_future_inference_features_use_only_prior_information() -> None:
    climate_payload: dict[str, list[object]] = {
        "codigo_ibge": ["1"],
        "season_year": [2022],
    }
    for column in CLIMATE_FEATURES:
        climate_payload[column] = [20.0]
    inference = build_soy_yield_inference_frame(
        _history(),
        pl.DataFrame(climate_payload),
        pl.DataFrame(
            {
                "codigo_ibge": ["1"],
                "latitude_centroid": [-12.0],
                "longitude_centroid": [-55.0],
            }
        ),
        codigo_ibge="1",
        target_year=2023,
    )
    assert inference.item(0, "produtividade_kg_ha") is None
    assert inference.item(0, "yield_mean_3y") == 1100.0
    assert inference.item(0, "baseline_municipal_median_5y") == 1100.0
    assert inference.item(0, "baseline_regional_median_lag1") == 1700.0
    assert inference.item(0, "climate_reference_season") == 2022
    assert inference.item(0, "climate_lag1_rain_crop_cycle_mm") == 20.0


def test_regression_metrics_report_mae_rmse_smape_and_r2() -> None:
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert metrics["rows"] == 2
    assert metrics["mae"] == 15.0
    assert np.isclose(float(metrics["rmse"]), np.sqrt(250.0))
    assert 0 < float(metrics["smape_pct"]) < 20
    assert float(metrics["r2"]) > 0


def test_source_metrics_reveal_era5_regression_hidden_in_pooled_score() -> None:
    frame = pl.DataFrame(
        {
            "climate_lag1_source": ["INMET", "ERA5", "INMET", "ERA5"],
            "produtividade_kg_ha": [100.0, 100.0, 200.0, 200.0],
            "baseline_municipal_median_5y": [120.0, 110.0, 220.0, 210.0],
            "yield_mean_5y": [130.0, 120.0, 230.0, 220.0],
            "baseline_regional_median_lag1": [140.0, 130.0, 240.0, 230.0],
        }
    )
    predictions = np.array([100.0, 115.0, 200.0, 215.0])
    grouped = _climate_source_metrics(frame, predictions)
    assert grouped["INMET"]["model"]["mae"] == 0.0
    assert grouped["INMET"]["beats_best_baseline"] is True
    era5 = grouped["ERA5"]
    assert era5["total_rows"] == era5["evaluated_rows"] == 2
    assert era5["excluded_rows"] == 0
    assert era5["model"]["mae"] == era5["model"]["rmse"] == 15.0
    assert era5["best_baseline"]["name"] == "historical_municipal_median"
    assert era5["best_baseline"]["metrics"]["mae"] == 10.0
    assert era5["beats_best_baseline"] is False
    assert set(era5["baselines"]) == set(BASELINE_COLUMNS)
    # Pooled MAE favors this model (7.5 vs 15), yet ERA5 rows favor the baseline.
    assert regression_metrics(np.array([100.0, 100.0, 200.0, 200.0]), predictions)["mae"] == 7.5


def test_source_metrics_use_paired_finite_rows_and_preserve_missing_provenance() -> None:
    frame = pl.DataFrame(
        {
            "climate_lag1_source": ["ERA5", "ERA5", None, " "],
            "produtividade_kg_ha": [100.0, 200.0, 100.0, 200.0],
            "baseline_municipal_median_5y": [100.0, 200.0, None, 200.0],
            "yield_mean_5y": [105.0, None, 100.0, 200.0],
            "baseline_regional_median_lag1": [105.0, 200.0, 100.0, 200.0],
        }
    )
    grouped = _climate_source_metrics(frame, np.array([110.0, 999.0, 100.0, np.nan]))
    era5 = grouped["ERA5"]
    assert era5["total_rows"] == 2
    assert era5["evaluated_rows"] == era5["excluded_rows"] == 1
    assert era5["model"]["mae"] == 10.0
    assert all(metrics["rows"] == 1 for metrics in era5["baselines"].values())
    assert era5["best_baseline"]["metrics"]["mae"] == 0.0
    assert era5["beats_best_baseline"] is False
    unknown = grouped["unknown"]
    assert unknown["status"] == "no_common_finite_rows"
    assert unknown["total_rows"] == unknown["excluded_rows"] == 2
    assert unknown["evaluated_rows"] == 0
    assert unknown["model"] is unknown["best_baseline"] is None
    assert unknown["beats_best_baseline"] is None


def test_source_metrics_support_legacy_dataset_and_reject_misaligned_predictions() -> None:
    frame = pl.DataFrame(
        {
            "produtividade_kg_ha": [100.0],
            **{column: [110.0] for column in BASELINE_COLUMNS.values()},
        }
    )
    grouped = _climate_source_metrics(frame, np.array([105.0]))
    assert list(grouped) == ["unknown"]
    assert grouped["unknown"]["evaluated_rows"] == 1
    with pytest.raises(ValueError, match="one value per evaluation row"):
        _climate_source_metrics(frame, np.array([105.0, 110.0]))


def test_training_report_contains_source_diagnostics_without_changing_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "project_root": tmp_path,
            "paths": settings.paths.model_copy(
                update={
                    "gold": tmp_path / "gold",
                    "models": tmp_path / "models",
                    "reports": tmp_path / "reports",
                }
            ),
            "ml": settings.ml.model_copy(update={"temporal_test_start_year": 2023}),
        }
    )
    settings.paths.gold.mkdir()
    payload: dict[str, list[object]] = {feature: [10.0] * 6 for feature in MODEL_FEATURES}
    payload.update(
        {
            "ano": [2022, 2023, 2024] * 2,
            "codigo_ibge": ["1"] * 3 + ["2"] * 3,
            "produtividade_kg_ha": [1000.0] * 6,
            "climate_lag1_source": ["INMET"] * 3 + ["ERA5"] * 3,
            **{column: [990.0] * 6 for column in BASELINE_COLUMNS.values()},
        }
    )
    pl.DataFrame(payload).write_parquet(settings.paths.gold / "ml_soy_yield_dataset.parquet")
    module = "agri_decision.models.yield_model.training"
    monkeypatch.setattr(f"{module}._geographic_holdout", lambda code, **kwargs: code == "2")
    monkeypatch.setattr(f"{module}._git_commit", lambda root: "fixture")
    monkeypatch.setattr(f"{module}._log_candidate", lambda **kwargs: "fixture-run")
    monkeypatch.setattr(
        f"{module}._factory",
        lambda name, seed: lambda: SimpleNamespace(
            fit=lambda features, target: None,
            predict=lambda features: np.zeros(len(features)),
        ),
    )
    report_path, model_path = train_and_evaluate_soy_yield(settings=settings)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert model_path is None
    assert report["promotion_gate"]["passed"] is False
    assert report["climate_source_evaluation"]["descriptive_only"] is True
    assert report["climate_source_evaluation"]["promotion_gate_applied"] is False
    for candidate in report["candidates"].values():
        assert candidate["temporal"]["rows"] == 4
        assert candidate["geographic"]["rows"] == 3
        assert candidate["baseline_comparison"]["beats_baseline"] is False
        grouped = candidate["by_climate_source"]
        assert grouped["temporal"]["INMET"]["model"]["rows"] == 2
        assert grouped["temporal"]["ERA5"]["model"]["rows"] == 2
        assert grouped["geographic"]["ERA5"]["model"]["rows"] == 3
        assert "INMET" not in grouped["geographic"]


@pytest.mark.parametrize("trained_on_era5", [False, True])
def test_inference_labels_era5_and_flags_unseen_training_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained_on_era5: bool
) -> None:
    from agri_decision.models.yield_model.inference import predict_soy_yield

    settings = load_settings()
    settings = settings.model_copy(
        update={"paths": settings.paths.model_copy(update={"models": tmp_path / "models"})}
    )
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    metadata_path = tmp_path / "metadata.json"
    source = "Copernicus CDS ERA5-Land"
    write_json_atomic(
        metadata_path,
        {
            "model": "fixture",
            "model_version": "fixture-v1",
            "dataset": {
                "dataset_version": "fixture",
                "climate_sources": [source] if trained_on_era5 else [],
            },
            "production_interval": {"radius_kg_ha": 100.0, "method": "fixture", "coverage": 0.8},
            "interval_evaluation": {"empirical_coverage": 0.8},
            "performance_confidence": 0.7,
            "performance": {},
        },
    )
    write_json_atomic(
        settings.paths.models / "yield" / "soja" / "latest.json",
        {
            "promoted": True,
            "artifact": str(model_path),
            "metadata": str(metadata_path),
        },
    )
    monkeypatch.setattr(
        "agri_decision.models.yield_model.inference.joblib.load",
        lambda path: SimpleNamespace(predict=lambda values: np.array([1200.0])),
    )
    climate_payload: dict[str, list[object]] = {
        "codigo_ibge": ["1"],
        "season_year": [2022],
        "source": [source],
        "data_nature": ["estimated"],
        "method": ["nearest land grid"],
    }
    for column in CLIMATE_FEATURES:
        climate_payload[column] = [20.0]
    climate_payload["distance_to_station_km"] = [None]
    prediction = predict_soy_yield(
        "1",
        history=_history(),
        climate=pl.DataFrame(climate_payload),
        locations=pl.DataFrame(
            {"codigo_ibge": ["1"], "latitude_centroid": [-12.0], "longitude_centroid": [-55.0]}
        ),
        settings=settings,
    )
    assert prediction is not None
    assert source in prediction["source"]
    assert "INMET" not in prediction["source"]
    assert prediction["features_reference"]["climate_data_nature"] == "estimated"
    assert ("CLIMATE_SOURCE_NOT_IN_MODEL_TRAINING" in prediction["alerts"]) is not trained_on_era5
