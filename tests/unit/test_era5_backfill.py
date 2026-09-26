import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import pytest
import xarray as xr
from filelock import FileLock, Timeout
from polars.testing import assert_frame_equal

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.era5.backfill import (
    build_backfill_plan,
    merge_climate_sources,
    run_era5_backfill,
)


def _inmet() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "codigo_ibge": ["1000001", "1000002", "1000003"],
            "municipio": ["Test A", "Test B", "Test C"],
            "uf": ["MT"] * 3,
            "season_year": [2024] * 3,
            "latitude_centroid": [-10.1] * 3,
            "longitude_centroid": [-55.1] * 3,
            "rain_crop_cycle_mm": [1000.0, None, None],
            "mean_temp_c": [25.0, None, None],
            "season_coverage": [1.0, None, None],
            "source": ["INMET"] * 3,
        }
    )


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings()
    paths = settings.paths.model_copy(
        update={name: tmp_path / name for name in ("bronze", "silver", "gold", "reports")}
    )
    for name in ("bronze", "silver", "gold", "reports"):
        getattr(paths, name).mkdir()
    result = settings.model_copy(update={"project_root": tmp_path, "paths": paths})
    _inmet().write_parquet(paths.gold / "inmet_location_season.parquet")
    _inmet().select(
        "codigo_ibge", "municipio", "uf", "latitude_centroid", "longitude_centroid"
    ).write_parquet(paths.silver / "dim_location.parquet")
    return result


class FakeClient:
    def __init__(self) -> None:
        self.submissions = 0
        self.remote_status = "running"

    def submit(self, request: dict[str, Any]) -> str:
        self.submissions += 1
        return f"job-{self.submissions}"

    def status(self, request_id: str) -> str:
        return self.remote_status

    def download(self, request_id: str) -> bytes:
        return b"CDF-test-fixture"


def test_plan_is_deterministic_bounded_and_includes_closing_midnight() -> None:
    plan = build_backfill_plan(_inmet())
    assert plan == build_backfill_plan(_inmet().reverse())
    assert len(plan) == 1
    assert plan[0]["municipalities"] == ["1000002", "1000003"]
    assert len(plan[0]["jobs"]) == 9
    feb = next(j for j in plan[0]["jobs"] if j["request"]["month"] == ["02"])
    assert feb["request"]["day"][-1] == "29"
    closing = plan[0]["jobs"][-1]["request"]
    assert closing["month"] == ["05"] and closing["day"] == ["01"]
    assert closing["time"] == ["00:00"]
    assert closing["variable"] == ["total_precipitation", "surface_solar_radiation_downwards"]
    assert build_backfill_plan(_inmet(), codigo_ibge="1000001") == []


def test_merge_preserves_good_inmet_and_withholds_incomplete_era5() -> None:
    inmet = _inmet()
    era5 = inmet.with_columns(
        pl.lit(2000.0).alias("rain_crop_cycle_mm"),
        pl.lit(30.0).alias("mean_temp_c"),
        pl.Series("season_coverage", [1.0, 1.0, 0.5]),
        pl.lit("ERA5").alias("source"),
    )
    merged = merge_climate_sources(inmet, era5)
    assert merged["source"].to_list() == ["INMET", "ERA5", "INMET"]
    assert merged["rain_crop_cycle_mm"].to_list() == [1000.0, 2000.0, None]
    assert_frame_equal(merged, merge_climate_sources(inmet, era5))
    with pytest.raises(ValueError, match="Duplicate ERA5"):
        merge_climate_sources(inmet, pl.concat([era5, era5]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1.01, -0.01])
def test_nonfinite_or_invalid_coverage_cannot_fill_a_gap(invalid: float) -> None:
    era5 = _inmet().with_columns(
        pl.lit(2000.0).alias("rain_crop_cycle_mm"),
        pl.lit(25.0).alias("mean_temp_c"),
        pl.lit(invalid).alias("season_coverage"),
        pl.lit("ERA5").alias("source"),
    )
    assert_frame_equal(_inmet(), merge_climate_sources(_inmet(), era5))


def test_resume_reuses_remote_ids_and_respects_global_cap(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeClient()
    run_era5_backfill(settings=settings, client=client, max_new_requests=3)  # type: ignore[arg-type]
    assert client.submissions == 3
    path = run_era5_backfill(settings=settings, client=client, max_new_requests=3)  # type: ignore[arg-type]
    report = json.loads(path.read_text())
    assert client.submissions == 3
    assert report["job_status_counts"] == {"running": 3, "planned": 6}
    assert report["remaining_gap_rows"] == 2
    # Downloading three of nine months must not fill a whole season yet.
    client.remote_status = "successful"
    path = run_era5_backfill(settings=settings, client=client, max_new_requests=0)  # type: ignore[arg-type]
    report = json.loads(path.read_text())
    assert report["job_status_counts"] == {"downloaded": 3, "planned": 6}
    assert not (settings.paths.gold / "era5_location_season.parquet").exists()
    assert client.submissions == 3
    run_era5_backfill(settings=settings, client=client, max_new_requests=3)  # type: ignore[arg-type]
    assert client.submissions == 6


def test_ambiguous_submission_is_not_repeated(tmp_path: Path) -> None:
    class BrokenClient(FakeClient):
        def submit(self, request: dict[str, Any]) -> str:
            self.submissions += 1
            raise TimeoutError("Potentially sensitive message not persisted")

    settings, client = _settings(tmp_path), BrokenClient()
    run_era5_backfill(settings=settings, client=client, max_new_requests=1)  # type: ignore[arg-type]
    path = run_era5_backfill(settings=settings, client=client, max_new_requests=0)  # type: ignore[arg-type]
    assert client.submissions == 1
    report = json.loads(path.read_text())
    assert report["job_status_counts"]["submission_unknown"] == 1
    state = next((settings.paths.bronze / "era5" / "jobs").glob("*/state.json"))
    assert "Potentially sensitive" not in state.read_text()


def test_second_writer_is_blocked(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    directory = settings.paths.bronze / "era5"
    directory.mkdir()
    with FileLock(directory / "backfill.lock", timeout=0), pytest.raises(Timeout):
        run_era5_backfill(settings=settings)


def test_expired_cds_results_keep_ids_and_do_not_consume_active_slots(tmp_path: Path) -> None:
    class ExpiredClient(FakeClient):
        def download(self, request_id: str) -> bytes:
            response = httpx.Response(404, request=httpx.Request("GET", "https://cds.example"))
            response.raise_for_status()
            raise AssertionError("unreachable")

    settings, client = _settings(tmp_path), ExpiredClient()
    client.remote_status = "successful"
    run_era5_backfill(settings=settings, client=client, max_new_requests=3)  # type: ignore[arg-type]
    path = run_era5_backfill(settings=settings, client=client, max_new_requests=0)  # type: ignore[arg-type]
    report = json.loads(path.read_text())
    assert report["job_status_counts"] == {"expired": 3, "planned": 6}
    assert client.submissions == 3
    for path in (settings.paths.bronze / "era5" / "jobs").glob("*/state.json"):
        state = json.loads(path.read_text())
        assert state["request_id"].startswith("job-")
        assert state["last_error_http_status"] == 404
    # A later invocation may start unsubmitted jobs, never repeats expired ones automatically.
    run_era5_backfill(settings=settings, client=client, max_new_requests=1)  # type: ignore[arg-type]
    assert client.submissions == 4


@pytest.mark.parametrize("missing_grid", [False, True])
def test_full_synthetic_netcdf_pipeline_and_checksum_guard(
    tmp_path: Path, missing_grid: bool
) -> None:
    class WeatherClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.remote_status = "successful"
            self.requests: dict[str, dict[str, Any]] = {}

        def submit(self, request: dict[str, Any]) -> str:
            request_id = super().submit(request)
            self.requests[request_id] = request
            return request_id

        def download(self, request_id: str) -> bytes:
            request = self.requests[request_id]
            year, month = request["year"][0], request["month"][0]
            times = np.array(
                [
                    f"{year}-{month}-{day}T{hour}"
                    for day in request["day"]
                    for hour in request["time"]
                ],
                dtype="datetime64[ns]",
            )
            values = {
                "t2m": (300.15, "K"),
                "d2m": (290.0, "K"),
                "u10": (3.0, "m s**-1"),
                "v10": (4.0, "m s**-1"),
                "tp": (0.001, "m"),
                "ssrd": (1000.0, "J m**-2"),
            }
            data = {
                name: (
                    ("valid_time", "latitude", "longitude"),
                    np.full((len(times), 1, 1), np.nan if missing_grid else value),
                    {"units": unit},
                )
                for name, (value, unit) in values.items()
            }
            dataset = xr.Dataset(
                data, coords={"valid_time": times, "latitude": [-10.1], "longitude": [-55.1]}
            )
            return bytes(dataset.to_netcdf(engine="scipy"))

    settings, client = _settings(tmp_path), WeatherClient()
    for _ in range(4):
        report_path = run_era5_backfill(
            settings=settings,
            client=client,
            max_new_requests=3,  # type: ignore[arg-type]
        )
    report = json.loads(report_path.read_text())
    assert report["downloads_complete"] is True
    assert report["complete_for_selected_scope"] is not missing_grid
    combined = pl.read_parquet(settings.paths.gold / "climate_location_season.parquet")
    expected_rain = None if missing_grid else 243.0
    expected_nature = None if missing_grid else "estimated"
    assert combined["rain_crop_cycle_mm"].to_list() == [1000.0, expected_rain, expected_rain]
    assert combined["data_nature"].to_list() == [None, expected_nature, expected_nature]
    assert client.submissions == 9
    run_era5_backfill(settings=settings, client=client, max_new_requests=3)  # type: ignore[arg-type]
    assert client.submissions == 9
    assert_frame_equal(
        combined, pl.read_parquet(settings.paths.gold / "climate_location_season.parquet")
    )
    raw = next((settings.paths.bronze / "era5" / "jobs").glob("*/*.nc"))
    raw.write_bytes(b"corrupted-test-fixture")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_era5_backfill(settings=settings, client=client)  # type: ignore[arg-type]
