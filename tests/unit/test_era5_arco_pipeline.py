import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xarray as xr

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.era5.arco_pipeline import run_era5_arco_backfill


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings()
    paths = settings.paths.model_copy(
        update={name: tmp_path / name for name in ("bronze", "silver", "gold", "reports")}
    )
    for name in ("bronze", "silver", "gold", "reports"):
        getattr(paths, name).mkdir()
    settings = settings.model_copy(update={"project_root": tmp_path, "paths": paths})
    data = pl.DataFrame(
        {
            "codigo_ibge": ["1000001", "1000002"],
            "municipio": ["Test A", "Test B"],
            "uf": ["MT", "MT"],
            "latitude_centroid": [-10.1, -10.2],
            "longitude_centroid": [-55.1, -55.1],
            "season_year": [2024, 2024],
            "rain_crop_cycle_mm": [1000.0, None],
            "mean_temp_c": [25.0, None],
            "season_coverage": [1.0, None],
            "source": ["INMET", "INMET"],
        }
    )
    data.write_parquet(paths.gold / "inmet_location_season.parquet")
    data.write_parquet(paths.gold / "climate_location_season.parquet")
    return settings


class FakeArco:
    def __init__(self, *, masked: bool = False) -> None:
        self.calls = 0
        self.masked = masked

    def fetch_point_season(
        self, *, latitude: float, longitude: float, season_year: int
    ) -> tuple[bytes, dict[str, Any]]:
        self.calls += 1
        times = np.arange(
            f"{season_year - 1}-09-01T00", f"{season_year}-05-01T01", dtype="datetime64[h]"
        ).astype("datetime64[ns]")
        values = {
            "t2m": (300.15, "K"),
            "d2m": (290.0, "K"),
            "u10": (3.0, "m s**-1"),
            "v10": (4.0, "m s**-1"),
            "tp": (0.001 / 24, "m"),
            "ssrd": (1000.0, "J m**-2"),
        }
        data = {
            name: (
                ("time", "latitude", "longitude"),
                np.full((len(times), 1, 1), np.nan if self.masked else value),
                {"units": unit},
            )
            for name, (value, unit) in values.items()
        }
        dataset = xr.Dataset(
            data, coords={"time": times, "latitude": [latitude], "longitude": [longitude]}
        )
        return bytes(dataset.to_netcdf(engine="scipy")), {"test_fixture": True}


@pytest.mark.parametrize("masked", [False, True])
def test_arco_backfill_preserves_inmet_and_reuses_downloads(tmp_path: Path, masked: bool) -> None:
    settings, client = _settings(tmp_path), FakeArco(masked=masked)
    path = run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    report = json.loads(path.read_text())
    assert report["complete_for_selected_scope"] is not masked
    assert report["remaining_gap_rows"] == int(masked)
    assert report["collection_complete_for_selected_scope"] is True
    assert report["coverage_status"] == (
        "completed_with_unavailable_data" if masked else "complete"
    )
    result = pl.read_parquet(settings.paths.gold / "climate_location_season.parquet")
    assert result["rain_crop_cycle_mm"][0] == 1000.0
    if not masked:
        assert result["rain_crop_cycle_mm"][1] == pytest.approx(243.0)
        assert "ARCO" in result["source"][1]
        assert result["data_nature"][1] == "estimated"
    else:
        assert result["rain_crop_cycle_mm"][1] is None
    run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    assert client.calls == 1


def test_reprocess_existing_uses_cached_bytes_without_new_downloads(tmp_path: Path) -> None:
    settings, client = _settings(tmp_path), FakeArco()
    run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    path = run_era5_arco_backfill(
        settings=settings, client=client, max_new_points=0, reprocess_existing=True
    )
    report = json.loads(path.read_text())
    assert report["points_processed_this_run"] == 1
    assert report["new_points_budget_used"] == 0
    assert report["coverage_status"] == "complete"
    assert client.calls == 1


def test_cached_subset_cannot_be_reused_after_location_changes(tmp_path: Path) -> None:
    settings, client = _settings(tmp_path), FakeArco()
    run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    baseline_path = settings.paths.gold / "inmet_location_season.parquet"
    changed = pl.read_parquet(baseline_path).with_columns(
        (pl.col("latitude_centroid") + 0.01).alias("latitude_centroid")
    )
    changed.write_parquet(baseline_path)
    changed.write_parquet(settings.paths.gold / "climate_location_season.parquet")
    report_path = run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    report = json.loads(report_path.read_text())
    assert report["remaining_gap_rows"] == 1
    assert report["point_states"][0]["error_type"] == "ValueError"
    assert report["coverage_status"] == "attention_required"
    assert client.calls == 1


def test_zero_budget_and_network_failure_preserve_missing_data(tmp_path: Path) -> None:
    class FailedClient(FakeArco):
        def fetch_point_season(self, **kwargs: Any) -> tuple[bytes, dict[str, Any]]:
            self.calls += 1
            raise TimeoutError("This possibly sensitive text must not be saved")

    settings, client = _settings(tmp_path), FailedClient()
    run_era5_arco_backfill(settings=settings, client=client)
    assert client.calls == 0
    path = run_era5_arco_backfill(settings=settings, client=client, max_new_points=1)
    report = json.loads(path.read_text())
    assert report["remaining_gap_rows"] == 1
    assert report["point_states"][0]["error_type"] == "TimeoutError"
    assert "sensitive text" not in path.read_text()
    assert client.calls == 1
