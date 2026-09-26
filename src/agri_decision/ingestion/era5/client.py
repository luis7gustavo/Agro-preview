from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cdsapi  # type: ignore[import-untyped]

from agri_decision.ingestion.http import HttpArtifact, OfficialSourceClient

CATALOGUE_URL = (
    "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-land"
)
DATASET_ID = "reanalysis-era5-land"


def build_canary_request(*, year: int = 2025) -> dict[str, Any]:
    """Return a minimal authenticated request around Sorriso/MT."""
    return {
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [-12.70, -55.72, -12.80, -55.64],
    }


@dataclass(frozen=True, slots=True)
class Era5DatasetMetadata:
    dataset_id: str
    title: str
    updated: str
    license: str
    temporal_start: str
    temporal_end: str
    retrieve_url: str


def parse_catalogue_metadata(payload: dict[str, Any]) -> Era5DatasetMetadata:
    intervals = payload["extent"]["temporal"]["interval"]
    retrieve = next(link for link in payload["links"] if link.get("rel") == "retrieve")
    return Era5DatasetMetadata(
        dataset_id=str(payload["id"]),
        title=str(payload["title"]),
        updated=str(payload["updated"]),
        license=str(payload["license"]),
        temporal_start=str(intervals[0][0]),
        temporal_end=str(intervals[0][1]),
        retrieve_url=str(retrieve["href"]),
    )


def cds_credentials_configured() -> bool:
    environment = bool(os.getenv("CDSAPI_KEY") and os.getenv("CDSAPI_URL"))
    configuration_file = (Path.home() / ".cdsapirc").is_file()
    return environment or configuration_file


class Era5Client:
    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient(timeout_seconds=60, retries=3)
        self._jobs_client: Any = None

    def _jobs(self) -> Any:
        if self._jobs_client is None:
            if not cds_credentials_configured():
                raise RuntimeError("CDS API credentials are not configured")
            self._jobs_client = cdsapi.Client(
                quiet=True,
                wait_until_complete=False,
                delete=False,
                progress=False,
                timeout=60,
                retry_max=1,
                sleep_max=5,
            ).client
        return self._jobs_client

    def submit(self, request: dict[str, Any]) -> str:
        """Submit once; the caller persists this ID before any polling/download."""
        return str(self._jobs().submit(DATASET_ID, request).request_id)

    def status(self, request_id: str) -> str:
        return str(self._jobs().get_remote(request_id).status)

    def download(self, request_id: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="agri-era5-") as directory:
            target = Path(directory) / "era5-result.nc"
            self._jobs().download_results(request_id, str(target))
            content = target.read_bytes()
        if not content.startswith((b"\x89HDF", b"CDF", b"PK\x03\x04")):
            raise ValueError("ERA5-Land returned neither NetCDF nor a ZIP container")
        return content

    def fetch_catalogue(self) -> tuple[Era5DatasetMetadata, HttpArtifact]:
        artifact = self.http.get(CATALOGUE_URL)
        payload = json.loads(artifact.content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ERA5-Land catalogue returned a non-object payload")
        metadata = parse_catalogue_metadata(payload)
        if metadata.dataset_id != DATASET_ID:
            raise ValueError(f"Unexpected ERA5-Land dataset id: {metadata.dataset_id}")
        return metadata, artifact

    def fetch_canary(self, *, year: int = 2025) -> bytes:
        if not cds_credentials_configured():
            raise RuntimeError("CDS API credentials are not configured")
        with tempfile.TemporaryDirectory(prefix="agri-era5-") as directory:
            target = Path(directory) / "era5_land_canary.nc"
            client = cdsapi.Client(quiet=True, wait_until_complete=True, delete=False)
            client.retrieve(DATASET_ID, build_canary_request(year=year), str(target))
            content = target.read_bytes()
        if not content.startswith((b"\x89HDF", b"CDF")):
            raise ValueError("ERA5-Land canary did not return a valid NetCDF container")
        return content
