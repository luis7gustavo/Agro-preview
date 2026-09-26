import polars as pl

from agri_decision.features.history import add_leakage_safe_history_features


def test_history_features_never_use_current_or_future_target() -> None:
    # Synthetic chronological fixture used only for leakage testing.
    frame = pl.DataFrame(
        {
            "codigo_ibge": ["1234567", "1234567", "1234567"],
            "cultura": ["soja", "soja", "soja"],
            "ano": [2020, 2021, 2022],
            "produtividade_kg_ha": [100.0, 200.0, 10_000.0],
            "area_plantada_ha": [10.0, 20.0, 30.0],
            "producao_t": [1.0, 4.0, 300.0],
        }
    )

    result = add_leakage_safe_history_features(frame).sort("ano")

    assert result[0, "yield_mean_3y"] is None
    assert result[1, "yield_mean_3y"] == 100.0
    assert result[2, "yield_mean_3y"] == 150.0
    assert result[2, "crop_presence_years"] == 2
