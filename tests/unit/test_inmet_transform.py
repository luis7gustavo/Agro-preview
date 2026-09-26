from datetime import date, timedelta

import polars as pl
import pytest

from agri_decision.transforms.inmet import (
    build_station_season_features,
    map_seasons_to_municipalities,
    parse_station_csv,
)


def _station_csv() -> bytes:
    metadata = (
        "REGIAO:;CO\nUF:;MT\nESTACAO:;TESTE\nCODIGO (WMO):;A999\n"
        "LATITUDE:;-12,5\nLONGITUDE:;-55,7\nALTITUDE:;400,0\n"
        "DATA DE FUNDACAO:;01/01/00\n"
    )
    header = ";".join(["Data", "Hora UTC", *[f"v{i}" for i in range(2, 19)]]) + ";\n"
    rows: list[str] = []
    for offset in range(2):
        current = date(2024, 1, 1) + timedelta(days=offset)
        for hour in range(24):
            values = [""] * 19
            values[0] = current.strftime("%Y/%m/%d")
            values[1] = f"{hour:02d}00 UTC"
            values[2] = "1,0" if hour == 12 else "0"
            values[6] = "100,0"
            values[7] = "25,0"
            values[9] = "30,0"
            values[10] = "20,0"
            values[15] = "70,0"
            values[18] = "2,0"
            rows.append(";".join(values) + ";")
    return (metadata + header + "\n".join(rows) + "\n").encode("latin1")


def test_station_csv_is_aggregated_daily_with_observed_provenance() -> None:
    daily = parse_station_csv(_station_csv(), source_checksum="a" * 64)
    assert daily.height == 2
    assert daily.get_column("rain_mm").to_list() == [1.0, 1.0]
    assert daily.get_column("temp_observations").to_list() == [24, 24]
    assert daily.item(0, "station_code") == "A999"
    assert daily.item(0, "data_nature") == "observed"


def test_station_season_and_nearest_mapping_mark_interpolation() -> None:
    daily = parse_station_csv(_station_csv(), source_checksum="a" * 64).with_columns(
        pl.lit(2024).alias("season_year")
    )
    station = build_station_season_features(daily).with_columns(
        pl.lit(1.0).alias("season_coverage")
    )
    location = pl.DataFrame(
        {
            "codigo_ibge": ["5107925"],
            "municipio": ["Sorriso"],
            "uf": ["MT"],
            "latitude_centroid": [-12.55],
            "longitude_centroid": [-55.72],
        }
    )
    mapped = map_seasons_to_municipalities(
        station, location, max_distance_km=250, minimum_coverage=0.7
    )
    assert mapped.item(0, "data_nature") == "interpolated"
    assert mapped.item(0, "quality_flag") is None
    assert mapped.item(0, "distance_to_station_km") < 10
    assert mapped.item(0, "station_metadata_policy") == "stable"


def _season_with_metadata_change(*, latitude_shift: float) -> pl.DataFrame:
    template = parse_station_csv(_station_csv(), source_checksum="a" * 64).row(0, named=True)
    rows = []
    for offset in range(242):
        current = date(2018, 9, 1) + timedelta(days=offset)
        row = {**template, "data": current}
        if current.year == 2019:
            row.update(
                {
                    "station_latitude": -12.5 + latitude_shift,
                    "station_altitude_m": 399.5,
                    "station_name": "TESTE ATUALIZADO",
                    "source_dataset_checksum": "b" * 64,
                }
            )
        rows.append(row)
    return pl.DataFrame(rows)


def test_coordinate_precision_revision_keeps_complete_observed_season() -> None:
    daily = _season_with_metadata_change(latitude_shift=0.005)
    season = build_station_season_features(daily)
    assert season.height == 1
    assert season.item(0, "season_coverage") == 1.0
    assert season.item(0, "rain_crop_cycle_mm") == 242
    assert season.item(0, "valid_rain_hours") == 242 * 24
    assert season.item(0, "station_name") == "TESTE ATUALIZADO"
    assert season.item(0, "station_latitude") == -12.495
    assert season.item(0, "station_metadata_versions") == 2
    assert 0.5 < season.item(0, "station_metadata_max_shift_km") < 0.6
    assert season.item(0, "station_metadata_tolerance_km") == 1.0
    assert season.item(0, "station_metadata_policy") == "canonicalized_latest_within_tolerance"
    assert season.get_column("source_checksums").to_list() == [["a" * 64, "b" * 64]]
    # Canonicalization is derived; original source metadata remains in Silver.
    assert daily.get_column("station_latitude").n_unique() == 2


def test_station_relocation_does_not_merge_observations_at_distinct_sites() -> None:
    season = build_station_season_features(_season_with_metadata_change(latitude_shift=0.1))
    assert season.height == 2
    assert season.get_column("season_coverage").max() < 0.7
    assert season.get_column("station_metadata_max_shift_km").min() > 10
    assert season.get_column("station_metadata_policy").unique().to_list() == [
        "segmented_coordinate_shift_over_tolerance"
    ]


def test_metadata_tolerance_uses_maximum_pairwise_distance_not_distance_to_latest() -> None:
    daily = _season_with_metadata_change(latitude_shift=0).with_columns(
        pl.when(pl.col("data") < date(2018, 11, 1))
        .then(pl.lit(-12.494))
        .when(pl.col("data") < date(2019, 1, 1))
        .then(pl.lit(-12.506))
        .otherwise(pl.lit(-12.5))
        .alias("station_latitude")
    )
    # Both older points are <1 km from the latest, but >1 km from one another.
    season = build_station_season_features(daily)
    assert season.height == 3
    assert season.get_column("station_metadata_max_shift_km").min() > 1.3
    assert season.get_column("station_metadata_policy").unique().to_list() == [
        "segmented_coordinate_shift_over_tolerance"
    ]


def test_station_uf_change_stays_segmented_even_when_coordinates_match() -> None:
    daily = _season_with_metadata_change(latitude_shift=0).with_columns(
        pl.when(pl.col("data").dt.year() == 2019)
        .then(pl.lit("GO"))
        .otherwise(pl.col("station_uf"))
        .alias("station_uf")
    )
    season = build_station_season_features(daily)
    assert season.height == 2
    assert season.get_column("station_metadata_policy").unique().to_list() == [
        "segmented_uf_change"
    ]


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_metadata_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_station_season_features(
            _season_with_metadata_change(latitude_shift=0), metadata_tolerance_km=tolerance
        )
