import polars as pl

from agri_decision.features.training import (
    CLIMATE_FEATURES,
    MODEL_FEATURES,
    build_soy_yield_training_frame,
)


def test_training_dataset_uses_lagged_climate_and_prior_regional_baseline() -> None:
    history = pl.DataFrame(
        {
            "codigo_ibge": ["1", "1", "1", "1", "2", "2", "2", "2"],
            "cultura": ["soja"] * 8,
            "uf": ["MT"] * 8,
            "ano": [2020, 2021, 2022, 2023] * 2,
            "produtividade_kg_ha": [1000.0, 1100.0, 1200.0, 1300.0, 2000.0, 2100.0, 2200.0, 2300.0],
        }
    )
    climate_payload: dict[str, list[object]] = {
        "codigo_ibge": ["1", "1"],
        "season_year": [2021, 2022],
    }
    for column in CLIMATE_FEATURES:
        climate_payload[column] = [10.0, 20.0]
    climate = pl.DataFrame(climate_payload)
    locations = pl.DataFrame(
        {
            "codigo_ibge": ["1", "2"],
            "latitude_centroid": [-12.0, -13.0],
            "longitude_centroid": [-55.0, -56.0],
        }
    )
    frame = build_soy_yield_training_frame(
        history, climate, locations, start_year=2022, end_year=2023
    )
    row_2023 = frame.filter((pl.col("codigo_ibge") == "1") & (pl.col("ano") == 2023))
    assert row_2023.item(0, "climate_reference_season") == 2022
    assert row_2023.item(0, "climate_lag1_rain_crop_cycle_mm") == 20.0
    assert row_2023.item(0, "baseline_regional_median_lag1") == 1700.0
    assert row_2023.item(0, "baseline_municipal_median_5y") == 1100.0


def test_training_preserves_lagged_era5_provenance_without_changing_model_features() -> None:
    history = pl.DataFrame(
        {
            "codigo_ibge": ["1"] * 4,
            "cultura": ["soja"] * 4,
            "uf": ["MT"] * 4,
            "ano": [2020, 2021, 2022, 2023],
            "produtividade_kg_ha": [1000.0] * 4,
        }
    )
    climate_payload: dict[str, list[object]] = {
        "codigo_ibge": ["1", "1"],
        "season_year": [2022, 2023],
        "source": ["Copernicus CDS ERA5-Land", "INMET automatic station annual archives"],
        "data_nature": ["estimated", "interpolated"],
        "method": ["nearest land grid", "nearest INMET station"],
        "distance_to_grid_km": [4.0, None],
    }
    for column in CLIMATE_FEATURES:
        climate_payload[column] = [20.0, 30.0]
    climate_payload["distance_to_station_km"] = [None, 10.0]
    frame = build_soy_yield_training_frame(
        history,
        pl.DataFrame(climate_payload),
        pl.DataFrame(
            {"codigo_ibge": ["1"], "latitude_centroid": [-12.0], "longitude_centroid": [-55.0]}
        ),
        start_year=2023,
        end_year=2023,
    )
    assert frame.item(0, "climate_reference_season") == 2022
    assert frame.item(0, "climate_lag1_source") == "Copernicus CDS ERA5-Land"
    assert frame.item(0, "climate_data_nature") == "estimated"
    assert frame.item(0, "climate_lag1_distance_to_station_km") is None
    assert frame.item(0, "climate_lag1_distance_to_grid_km") == 4.0
    assert len(MODEL_FEATURES) == 29
    assert "climate_lag1_source" not in MODEL_FEATURES
    assert "climate_lag1_distance_to_grid_km" not in MODEL_FEATURES
