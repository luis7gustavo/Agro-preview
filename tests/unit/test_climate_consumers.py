from pathlib import Path

import polars as pl
import pytest

from agri_decision.recommendations import _climate_scenario, _cost_scenario, _price_scenario


@pytest.mark.parametrize(
    ("rain", "coverage", "expected_status"),
    [
        (1500.0, 1.0, "available"),
        (None, 1.0, "not_available"),
        (1500.0, 0.5, "not_available"),
        (1500.0, 1.01, "not_available"),
        (1500.0, float("inf"), "not_available"),
        (1500.0, float("nan"), "not_available"),
        (float("nan"), 1.0, "not_available"),
    ],
)
def test_reanalysis_availability_uses_measurements_and_coverage(
    tmp_path: Path, rain: float | None, coverage: float, expected_status: str
) -> None:
    path = tmp_path / "climate.parquet"
    pl.DataFrame(
        {
            "codigo_ibge": ["5107925"],
            "season_year": [2024],
            "rain_crop_cycle_mm": [rain],
            "mean_temp_c": [25.0],
            "season_coverage": [coverage],
            "distance_to_station_km": [None],
            "distance_to_grid_km": [5.0],
            "source": ["Copernicus CDS ERA5-Land"],
            "data_nature": ["estimated"],
            "method": ["nearest land grid"],
        }
    ).write_parquet(path)
    climate = _climate_scenario(path, "5107925", reference_season=2024, max_distance_km=250)
    assert climate["status"] == expected_status
    assert climate["station_code"] is None
    assert climate["distance_to_grid_km"] == 5.0
    assert climate["confidence"] == (0.75 if expected_status == "available" else 0)


def test_price_period_uses_real_month_year_pairs() -> None:
    prices = pl.DataFrame(
        {
            "coverage_level": ["municipality", "municipality"],
            "codigo_ibge_fonte": [5107925, 5107925],
            "uf": ["MT", "MT"],
            "ano": [2025, 2026],
            "mes": [12, 1],
            "nivel_comercializacao": ["PRODUTOR", "PRODUTOR"],
            "preco_brl_kg": [2.0, 2.1],
        }
    )
    result = _price_scenario(prices, 5107925, "MT")
    assert result["reference_period"] == "2025-12/2026-01"


def test_state_cost_period_retains_each_pole_actual_reference() -> None:
    costs = pl.DataFrame(
        {
            "codigo_ibge_fonte": [5100001, 5100002],
            "uf": ["MT", "MT"],
            "municipio_fonte": ["A", "B"],
            "ano": [2025, 2026],
            "mes": [12, 1],
            "custo_variavel_brl_ha": [100.0, 200.0],
            "custo_operacional_brl_ha": [120.0, 220.0],
            "custo_total_brl_ha": [150.0, 250.0],
        }
    )
    result = _cost_scenario(costs, 5107925, "MT")
    assert result["reference_periods"] == ["2025-12", "2026-01"]
    assert result["reference_period"] == "2025-12/2026-01"
    assert result["total_brl_ha"] == 200.0
