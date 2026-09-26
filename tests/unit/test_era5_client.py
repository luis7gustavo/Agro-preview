from types import SimpleNamespace
from typing import Any

import pytest

from agri_decision.ingestion.era5.client import build_canary_request, parse_catalogue_metadata


def test_parse_era5_catalogue_metadata_keeps_license_and_retrieve_url() -> None:
    metadata = parse_catalogue_metadata(
        {
            "id": "reanalysis-era5-land",
            "title": "ERA5-Land hourly data from 1950 to present",
            "updated": "2026-08-20T00:00:00Z",
            "license": "CC-BY-4.0",
            "extent": {"temporal": {"interval": [["1950-01-01", "2026-08-13"]]}},
            "links": [
                {
                    "rel": "retrieve",
                    "href": "https://cds.example/retrieve/reanalysis-era5-land",
                }
            ],
        }
    )
    assert metadata.dataset_id == "reanalysis-era5-land"
    assert metadata.license == "CC-BY-4.0"
    assert metadata.temporal_end == "2026-08-13"
    assert metadata.retrieve_url.endswith("reanalysis-era5-land")


def test_era5_canary_request_is_small_and_spatially_bounded() -> None:
    request = build_canary_request(year=2025)
    assert request["variable"] == ["2m_temperature"]
    assert request["year"] == ["2025"]
    assert request["month"] == ["01"]
    assert request["day"] == ["01"]
    assert request["time"] == ["00:00"]
    assert request["area"] == [-12.70, -55.72, -12.80, -55.64]
    assert request["data_format"] == "netcdf"


def test_submission_has_no_automatic_post_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from agri_decision.ingestion.era5 import client as module

    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        jobs = SimpleNamespace(submit=lambda *_: SimpleNamespace(request_id="test-job"))
        return SimpleNamespace(client=jobs)

    monkeypatch.setattr(module, "cds_credentials_configured", lambda: True)
    monkeypatch.setattr(module.cdsapi, "Client", factory)
    assert module.Era5Client().submit({"year": ["2025"]}) == "test-job"
    assert captured["retry_max"] == 1
    assert captured["delete"] is False
