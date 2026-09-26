from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from agri_decision.api.main import app
from agri_decision.config import load_settings

client = TestClient(app)


def test_health_version_and_active_crop_contracts() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    version = client.get("/version")
    assert version.status_code == 200
    assert version.json()["api_version"] == "v1"
    crops = client.get("/v1/crops")
    assert crops.status_code == 200
    assert [crop["canonical_name"] for crop in crops.json()] == ["soja"]


def test_unknown_analysis_is_not_found() -> None:
    response = client.get("/v1/recommendations/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.parametrize("era5_coverage", [1.0, 1.01, float("inf"), float("nan")])
def test_climate_coverage_counts_inmet_and_era5_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, era5_coverage: float
) -> None:
    settings = load_settings()
    paths = settings.paths.model_copy(
        update={"silver": tmp_path, "gold": tmp_path, "models": tmp_path / "models"}
    )
    monkeypatch.setattr(
        "agri_decision.api.main.load_settings", lambda: settings.model_copy(update={"paths": paths})
    )
    pl.DataFrame(
        {
            "codigo_ibge": ["1", "2", "3"],
            "season_year": [2024, 2024, 2024],
            "rain_crop_cycle_mm": [1500.0, 1800.0, None],
            "mean_temp_c": [25.0, 24.0, None],
            "season_coverage": [1.0, era5_coverage, None],
            "distance_to_station_km": [10.0, None, 400.0],
            "quality_flag": [None, "ERA5_LAND_FALLBACK", "NO_NEARBY_INMET_STATION"],
            "source": [
                "INMET automatic station annual archives",
                "Copernicus CDS ERA5-Land",
                "INMET automatic station annual archives",
            ],
        }
    ).write_parquet(paths.gold / "climate_location_season.parquet")
    response = client.get("/v1/data-coverage")
    assert response.status_code == 200
    climate = response.json()["climate"]
    assert climate["inmet_within_radius_rows"] == 1
    era5_available = int(era5_coverage == 1.0)
    assert climate["era5_land_fallback_rows"] == era5_available
    assert climate["available_rows"] == 1 + era5_available
    assert climate["missing_climate_rows"] == 2 - era5_available
    assert climate["outside_inmet_radius_rows"] == 2
    assert climate["unfilled_outside_inmet_radius_rows"] == 1
