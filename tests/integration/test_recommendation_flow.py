from pathlib import Path

import polars as pl
import pytest

from agri_decision.config import Settings, load_settings
from agri_decision.recommendations import (
    RecommendationRequest,
    generate_recommendation,
    load_analysis,
)


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings()
    paths = settings.paths.model_copy(
        update={
            "silver": tmp_path / "silver",
            "gold": tmp_path / "gold",
            "reports": tmp_path / "reports",
            "models": tmp_path / "models",
        }
    )
    for path in [paths.silver, paths.gold, paths.reports]:
        path.mkdir(parents=True)
    return settings.model_copy(update={"paths": paths})


def _write_minimum_datasets(settings: Settings) -> None:
    pl.DataFrame(
        {
            "codigo_ibge": ["5107925", "5100001"],
            "municipio": ["Sorriso", "Nao elegivel"],
            "uf": ["MT", "MT"],
        }
    ).write_parquet(settings.paths.silver / "dim_location.parquet")
    pl.DataFrame(
        {
            "codigo_ibge": ["5107925"] * 3,
            "cultura": ["soja"] * 3,
            "ano": [2022, 2023, 2024],
            "produtividade_kg_ha": [3_000.0, 3_500.0, 4_000.0],
        }
    ).write_parquet(settings.paths.silver / "production_history.parquet")
    pl.DataFrame(
        {
            "codigo_ibge_fonte": [5_107_925],
            "uf": ["MT"],
            "ano": [2026],
            "mes": [3],
            "municipio_fonte": ["SORRISO-MT"],
            "custo_variavel_brl_ha": [4_000.0],
            "custo_operacional_brl_ha": [4_800.0],
            "custo_total_brl_ha": [5_000.0],
        }
    ).write_parquet(settings.paths.silver / "conab_soy_costs.parquet")
    pl.DataFrame(
        {
            "coverage_level": ["municipality"] * 12,
            "codigo_ibge_fonte": [5_107_925] * 12,
            "uf": ["MT"] * 12,
            "ano": [2025] * 12,
            "mes": list(range(1, 13)),
            "nivel_comercializacao": ["PRECO RECEBIDO P/ PRODUTOR"] * 12,
            "preco_brl_kg": [2.0] * 12,
        }
    ).write_parquet(settings.paths.silver / "conab_soy_prices.parquet")
    pl.DataFrame(
        {
            "codigo_ibge": ["5107925", "5100001"],
            "safra_inicio": [2026, 2026],
            "safra_fim": [2027, 2027],
            "zarc_status": ["eligible", "not_eligible"],
            "source": ["MAPA ZARC", "MAPA ZARC"],
            "status_method": [None, "absence from complete table"],
            "melhor_nivel_risco": [20, None],
            "decendios_elegiveis": [[27, 28, 29], None],
            "ciclos_codigos": [[20], None],
            "tipos_solo_codigos": [[11], None],
            "portarias": [["Port. 1"], None],
            "combinacoes_elegiveis": [3, None],
        }
    ).write_parquet(settings.paths.gold / "zarc_soy_municipality.parquet")


def test_gold_to_recommendation_and_audit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_minimum_datasets(settings)
    response = generate_recommendation(
        RecommendationRequest(codigo_ibge="5107925", area_ha=10), settings=settings
    )
    recommendation = response["recommendations"][0]
    assert recommendation["zarc"]["status"] == "eligible"
    assert recommendation["economics"]["economic_profit_brl_ha"] == 2_000
    assert recommendation["risk"]["simulations"] == settings.risk.simulations
    stored = load_analysis(response["analysis_id"], settings=settings)
    assert stored["audit"]["analysis_id"] == response["analysis_id"]
    assert "TEMPORAL_MISMATCH" in recommendation["alerts"]
    assert response["temporal_context"]["aligned_forecast"] is False


def test_not_eligible_zarc_is_excluded_before_economics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_minimum_datasets(settings)
    response = generate_recommendation(
        RecommendationRequest(codigo_ibge="5100001"), settings=settings
    )
    assert response["recommendations"] == []
    assert response["excluded_recommendations"][0]["alerts"] == ["ZARC_NOT_ELIGIBLE"]


def test_era5_recommendation_exposes_source_and_mixed_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _write_minimum_datasets(settings)
    pl.DataFrame(
        {
            "codigo_ibge": ["5107925"],
            "season_year": [2024],
            "rain_crop_cycle_mm": [1700.0],
            "mean_temp_c": [25.0],
            "season_coverage": [1.0],
            "distance_to_station_km": [None],
            "distance_to_grid_km": [4.0],
            "station_code": [None],
            "station_name": [None],
            "source": ["Copernicus CDS ERA5-Land"],
            "data_nature": ["estimated"],
            "method": ["nearest land grid"],
            "quality_flag": ["ERA5_LAND_FALLBACK"],
        }
    ).write_parquet(settings.paths.gold / "climate_location_season.parquet")
    monkeypatch.setattr(
        "agri_decision.models.yield_model.inference.predict_soy_yield",
        lambda *args, **kwargs: {
            "p10": 3000.0,
            "p50": 3500.0,
            "p90": 4000.0,
            "type": "predicted",
            "model": "fixture",
            "model_version": "fixture-v1",
            "target_year": 2025,
            "reference_period": "2025",
            "performance_confidence": 0.8,
            "features_reference": {"climate_season": 2024, "history_through_year": 2024},
            "source": "IBGE/PAM and Copernicus CDS ERA5-Land",
            "alerts": ["CLIMATE_SOURCE_NOT_IN_MODEL_TRAINING"],
        },
    )
    response = generate_recommendation(
        RecommendationRequest(codigo_ibge="5107925", area_ha=10), settings=settings
    )
    recommendation = response["recommendations"][0]
    assert recommendation["climate"]["status"] == "available"
    assert recommendation["climate"]["origin"] == "reanalysis"
    assert recommendation["climate"]["confidence"] == 0.75
    assert "ERA5_LAND_FALLBACK" in recommendation["alerts"]
    assert "MISSING_CLIMATE" not in recommendation["alerts"]
    assert "CLIMATE_SOURCE_NOT_IN_MODEL_TRAINING" in recommendation["alerts"]
    assert "Copernicus CDS ERA5-Land" in response["audit"]["sources_used"]
    temporal = response["temporal_context"]
    assert temporal["scenario_type"] == "hybrid_reference"
    assert temporal["yield_target_year"] == 2025
    assert temporal["climate_reference_season"] == 2024
    assert temporal["cost_reference_period"] == "2026-03"
    assert temporal["price_reference_period"] == "2025-01/2025-12"
    assert temporal["zarc_season"] == "2026/2027"
    assert response["audit"]["temporal_context"] == temporal
    stored = load_analysis(response["analysis_id"], settings=settings)
    assert stored["temporal_context"] == temporal
