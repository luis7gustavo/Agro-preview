from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from agri_decision.transforms.era5 import (
    build_era5_season_features,
    combine_era5_daily,
    parse_era5_daily,
)

UNITS = {"t2m": "K", "d2m": "K", "u10": "m s**-1", "v10": "m s**-1", "tp": "m", "ssrd": "J m**-2"}


def _locations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "codigo_ibge": ["5107925"],
            "municipio": ["Sorriso"],
            "uf": ["MT"],
            "latitude_centroid": [-12.51],
            "longitude_centroid": [-55.71],
        }
    )


def _payload(
    tmp_path: Path, filename: str, times: np.ndarray, **variables: float | list[float]
) -> bytes:
    dataset = xr.Dataset(
        {
            name: (
                ("valid_time", "latitude", "longitude"),
                np.broadcast_to(np.asarray(values, dtype=float), (len(times),)).reshape(-1, 1, 1),
                {"units": UNITS[name]},
            )
            for name, values in variables.items()
        },
        coords={"valid_time": times, "latitude": [-12.5], "longitude": [304.3]},
    )
    path = tmp_path / filename
    dataset.to_netcdf(path, engine="h5netcdf")
    return path.read_bytes()


def test_midnight_accumulations_use_previous_day_and_correct_units(tmp_path: Path) -> None:
    instant = _payload(
        tmp_path,
        "instant.nc",
        np.arange("2024-01-01", "2024-01-02", dtype="datetime64[h]"),
        t2m=298.15,
        d2m=298.15,
        u10=3,
        v10=4,
    )
    accum = _payload(
        tmp_path,
        "accum.nc",
        np.array(["2024-01-02T00"], dtype="datetime64[h]"),
        tp=0.012,
        ssrd=20_000_000,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data_stream-oper_stepType-instant.nc", instant)
        archive.writestr("data_stream-oper_stepType-accum.nc", accum)
    daily = parse_era5_daily(buffer.getvalue(), _locations(), source_checksum="a" * 64)
    row = daily.filter(pl.col("data") == date(2024, 1, 1)).row(0, named=True)
    assert row["rain_mm"] == pytest.approx(12)
    assert row["radiation_kj_m2"] == pytest.approx(20_000)
    assert row["mean_temp_c"] == pytest.approx(25)
    assert row["mean_humidity_pct"] == pytest.approx(100)
    assert row["mean_wind_m_s"] == pytest.approx(5)
    assert row["temp_observations"] == row["rain_observations"] == 24
    assert 0 < row["distance_to_grid_km"] < 2
    assert row["grid_longitude"] == pytest.approx(-55.7)
    assert row["data_nature"] == "estimated"


def test_non_midnight_accumulations_are_not_summed(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "accum.nc",
        np.array(["2024-01-02T00", "2024-01-02T01", "2024-01-02T02"], dtype="datetime64[h]"),
        tp=[0.01, 0.1, 0.2],
    )
    daily = parse_era5_daily(content, _locations(), source_checksum="a")
    assert daily.filter(pl.col("data") == date(2024, 1, 1)).item(0, "rain_mm") == 10
    assert daily.filter(pl.col("data") == date(2024, 1, 2)).item(0, "rain_mm") is None


def test_missing_rain_remains_null_in_daily_and_season(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "missing.nc",
        np.array(["2024-01-02T00"], dtype="datetime64[h]"),
        t2m=298.15,
        tp=float("nan"),
    )
    daily = parse_era5_daily(content, _locations(), source_checksum="a")
    assert daily.item(0, "rain_mm") is None
    season = build_era5_season_features(daily)
    assert season.item(0, "rain_crop_cycle_mm") is None
    assert season.item(0, "dry_days") is None
    assert season.item(0, "season_coverage") == 0
    assert season.item(0, "quality_flag") == "ERA5_INSUFFICIENT_SEASON_COVERAGE"


def test_separate_downloads_merge_without_double_counting(tmp_path: Path) -> None:
    instant = _payload(
        tmp_path,
        "instant.nc",
        np.arange("2024-01-01", "2024-01-02", dtype="datetime64[h]"),
        t2m=298.15,
    )
    accum = _payload(
        tmp_path, "accum.nc", np.array(["2024-01-02T00"], dtype="datetime64[h]"), tp=0.01
    )
    first = parse_era5_daily(instant, _locations(), source_checksum="a")
    second = parse_era5_daily(accum, _locations(), source_checksum="b")
    daily = combine_era5_daily([first, second, second])
    row = daily.filter(pl.col("data") == date(2024, 1, 1)).row(0, named=True)
    assert row["rain_mm"] == 10
    assert row["temp_observations"] == 24
    assert row["rain_observations"] == 24
    assert row["source_checksums"] == ["a", "b"]
    conflict = second.with_columns(pl.lit(999.0).alias("rain_mm"))
    with pytest.raises(ValueError, match="Conflicting"):
        combine_era5_daily([first, second, conflict])


def test_full_leap_season_has_complete_coverage_and_explicit_reanalysis(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "daily.nc",
        np.arange("2023-09-01", "2023-09-02", dtype="datetime64[h]"),
        t2m=298.15,
        d2m=288.15,
        u10=3,
        v10=4,
    )
    template = parse_era5_daily(content, _locations(), source_checksum="a").with_columns(
        pl.lit(10.0).alias("rain_mm"),
        pl.lit(24).alias("rain_observations"),
        pl.lit(20_000.0).alias("radiation_kj_m2"),
        pl.lit(24).alias("radiation_observations"),
    )
    dates = [date(2023, 9, 1) + timedelta(days=i) for i in range(243)]
    daily = pl.concat([template.with_columns(pl.lit(day).alias("data")) for day in dates])
    season = build_era5_season_features(daily)
    assert season.height == 1
    row = season.row(0, named=True)
    assert row["rain_crop_cycle_mm"] == 2430
    assert row["rain_30d_mm"] == 300
    assert row["rain_60d_mm"] == 610
    assert row["rain_90d_mm"] == 900
    assert row["season_coverage"] == 1
    assert row["data_nature"] == "estimated"
    assert row["station_code"] is None
    assert row["distance_to_station_km"] is None
    assert row["distance_to_grid_km"] > 0
    assert row["quality_flag"] == "ERA5_REANALYSIS_FALLBACK"
    assert 50 < row["mean_humidity_pct"] < 60


def test_invalid_units_and_outside_grid_are_rejected(tmp_path: Path) -> None:
    content = _payload(
        tmp_path, "instant.nc", np.array(["2024-01-01T00"], dtype="datetime64[h]"), t2m=298.15
    )
    with pytest.raises(ValueError, match="too far"):
        parse_era5_daily(
            content,
            _locations().with_columns(pl.lit(0.0).alias("latitude_centroid")),
            source_checksum="a",
        )
    with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as source:
        dataset = source.load()
    dataset["t2m"].attrs["units"] = "Celsius"
    invalid = tmp_path / "invalid.nc"
    dataset.to_netcdf(invalid, engine="h5netcdf")
    with pytest.raises(ValueError, match="units"):
        parse_era5_daily(invalid.read_bytes(), _locations(), source_checksum="a")


def test_arco_increments_reproduce_forecast_origin_daily_totals(tmp_path: Path) -> None:
    times = np.arange("2024-01-01T00", "2024-01-03T01", dtype="datetime64[h]")
    increments = [0.0024, *([hour * 0.0001 for hour in range(1, 25)] * 2)]
    accumulated = [0.03, *([hour * (hour + 1) / 2 * 0.0001 for hour in range(1, 25)] * 2)]
    instant = {"t2m": 298.15, "d2m": 288.15, "u10": 3.0, "v10": 4.0}
    standard = _payload(
        tmp_path,
        "standard.nc",
        times,
        tp=accumulated,
        ssrd=[value * 1e8 for value in accumulated],
        **instant,
    )
    arco = _payload(
        tmp_path,
        "arco.nc",
        times,
        tp=increments,
        ssrd=[value * 1e8 for value in increments],
        **instant,
    )
    original = parse_era5_daily(standard, _locations(), source_checksum="standard")
    converted = parse_era5_daily(
        arco, _locations(), source_checksum="arco", accumulation_mode="hourly_increments"
    )
    for day in (date(2024, 1, 1), date(2024, 1, 2)):
        left = original.filter(pl.col("data") == day).row(0, named=True)
        right = converted.filter(pl.col("data") == day).row(0, named=True)
        for name in (
            "rain_mm",
            "radiation_kj_m2",
            "mean_temp_c",
            "mean_humidity_pct",
            "mean_wind_m_s",
            "rain_observations",
            "radiation_observations",
        ):
            assert right[name] == pytest.approx(left[name])
        assert right["rain_mm"] == pytest.approx(30)
        assert right["rain_observations"] == 24
        assert right["source"] == "Copernicus ERA5-Land ARCO reanalysis"
        assert right["data_nature"] == "estimated"


def test_arco_midnight_closes_previous_day_but_instantaneous_temperature_does_not_shift(
    tmp_path: Path,
) -> None:
    content = _payload(
        tmp_path,
        "closing-day.nc",
        np.arange("2024-04-30T00", "2024-05-01T01", dtype="datetime64[h]"),
        tp=[0.099, *([0.001] * 23), 0.010],
        ssrd=[999_000, *([1000.0] * 24)],
        t2m=[*([298.15] * 24), 313.15],
    )
    daily = parse_era5_daily(
        content, _locations(), source_checksum="arco", accumulation_mode="hourly_increments"
    )
    april29 = daily.filter(pl.col("data") == date(2024, 4, 29)).row(0, named=True)
    april30 = daily.filter(pl.col("data") == date(2024, 4, 30)).row(0, named=True)
    may1 = daily.filter(pl.col("data") == date(2024, 5, 1)).row(0, named=True)
    assert april29["rain_mm"] is None
    assert april29["rain_observations"] == 1
    assert april30["rain_mm"] == pytest.approx(33)
    assert april30["radiation_kj_m2"] == pytest.approx(24)
    assert april30["temp_observations"] == april30["rain_observations"] == 24
    assert april30["mean_temp_c"] == pytest.approx(25)
    assert may1["mean_temp_c"] == pytest.approx(40)
    assert may1["temp_observations"] == 1
    assert may1["rain_mm"] is None


def test_arco_missing_hour_withholds_daily_total_and_does_not_inflate_season_coverage(
    tmp_path: Path,
) -> None:
    rain = [0.001] * 25
    rain[12] = float("nan")
    content = _payload(
        tmp_path,
        "missing-hour.nc",
        np.arange("2024-01-01T00", "2024-01-02T01", dtype="datetime64[h]"),
        tp=rain,
        ssrd=1000,
        t2m=298.15,
    )
    daily = parse_era5_daily(
        content, _locations(), source_checksum="arco", accumulation_mode="hourly_increments"
    )
    row = daily.filter(pl.col("data") == date(2024, 1, 1)).row(0, named=True)
    assert row["rain_mm"] is None
    assert row["rain_observations"] == 23
    assert row["radiation_kj_m2"] == pytest.approx(24)
    assert row["radiation_observations"] == 24
    season = build_era5_season_features(daily)
    assert season.item(0, "valid_rain_hours") == 0
    assert season.item(0, "rain_crop_cycle_mm") is None
    assert season.item(0, "season_coverage") == 0
    assert season.item(0, "source") == "Copernicus ERA5-Land ARCO reanalysis"
    assert "ARCO hourly increments" in season.item(0, "method")


def test_arco_all_missing_cell_is_unavailable_not_a_dry_season(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "ocean.nc",
        np.arange("2024-01-01T00", "2024-01-02T01", dtype="datetime64[h]"),
        tp=float("nan"),
        ssrd=float("nan"),
        t2m=float("nan"),
    )
    daily = parse_era5_daily(
        content, _locations(), source_checksum="arco", accumulation_mode="hourly_increments"
    )
    assert daily["rain_mm"].null_count() == daily.height
    assert daily["rain_observations"].sum() == 0
    season = build_era5_season_features(daily)
    assert season.item(0, "rain_crop_cycle_mm") is None
    assert season.item(0, "dry_days") is None
    assert season.item(0, "quality_flag") == "ERA5_INSUFFICIENT_SEASON_COVERAGE"


def test_unknown_accumulation_convention_is_rejected(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "convention.nc",
        np.array(["2024-01-01T00"], dtype="datetime64[h]"),
        tp=0.1,
    )
    with pytest.raises(ValueError, match="accumulation mode"):
        parse_era5_daily(
            content,
            _locations(),
            source_checksum="a",
            accumulation_mode="infer",  # type: ignore[arg-type]
        )


def test_signed_arco_increments_reconstruct_daily_accumulation_without_clipping_bias(
    tmp_path: Path,
) -> None:
    # Real packing artefacts can exceed the old per-hour tolerance. Opposite
    # signed differences cancel when reconstructing the original 24h total.
    rain = [0.001, -2.2e-8, 2.2e-8, *([0.0] * 21)]
    radiation = [1000.0, -2.0, 2.0, *([0.0] * 21)]
    content = _payload(
        tmp_path,
        "signed.nc",
        np.arange("2024-01-01T01", "2024-01-02T01", dtype="datetime64[h]"),
        tp=rain,
        ssrd=radiation,
        t2m=298.15,
    )
    daily = parse_era5_daily(
        content, _locations(), source_checksum="signed", accumulation_mode="hourly_increments"
    )
    row = daily.filter(pl.col("data") == date(2024, 1, 1)).row(0, named=True)
    assert row["rain_mm"] == pytest.approx(1, abs=1e-12)
    assert row["radiation_kj_m2"] == pytest.approx(1, abs=1e-12)
    assert row["rain_observations"] == row["radiation_observations"] == 24
    assert row["rain_negative_increments"] == row["radiation_negative_increments"] == 1
    season = build_era5_season_features(daily)
    assert season.item(0, "valid_rain_hours") == 24
    assert season.item(0, "rain_negative_increments") == 1
    assert season.item(0, "radiation_negative_increments") == 1


def test_materially_negative_arco_daily_total_stays_unavailable(tmp_path: Path) -> None:
    content = _payload(
        tmp_path,
        "negative.nc",
        np.arange("2024-01-01T01", "2024-01-02T01", dtype="datetime64[h]"),
        tp=-0.001,
        ssrd=-1000,
        t2m=298.15,
    )
    daily = parse_era5_daily(
        content, _locations(), source_checksum="negative", accumulation_mode="hourly_increments"
    )
    row = daily.filter(pl.col("data") == date(2024, 1, 1)).row(0, named=True)
    assert row["rain_mm"] is row["radiation_kj_m2"] is None
    assert row["rain_observations"] == 24
    assert row["rain_negative_increments"] == 24
    assert build_era5_season_features(daily).item(0, "valid_rain_hours") == 0
