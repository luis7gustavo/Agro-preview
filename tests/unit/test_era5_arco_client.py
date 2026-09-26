from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import xarray as xr
from numcodecs import Blosc

from agri_decision.ingestion.era5.arco import GROUPS, Era5ArcoClient


def _mock_store(
    *,
    metadata_override: tuple[str, str, dict[str, Any]] | None = None,
    response_status: dict[tuple[str, str], int] | None = None,
) -> tuple[httpx.Client, list[str]]:
    times = np.arange("2024-09-01T00", "2025-05-01T01", dtype="datetime64[h]")
    coords = {
        "time": times.astype(np.int64),
        "latitude": np.array([-12.8, -12.7]),
        "longitude": np.array([-55.8, -55.7]),
    }
    constants = {"t2m": 300.0, "d2m": 290.0, "u10": 3.0, "v10": 4.0, "tp": 0.0001, "ssrd": 1000.0}
    units = {
        "t2m": "K",
        "d2m": "K",
        "u10": "m s**-1",
        "v10": "m s**-1",
        "tp": "m",
        "ssrd": "J m**-2",
    }
    codec = Blosc(cname="lz4", clevel=5, shuffle=1)
    requests: list[str] = []
    stores: dict[str, dict[str, Any]] = {}
    for group, (_, _, variables) in GROUPS.items():
        metadata: dict[str, Any] = {".zgroup": {"zarr_format": 2}, ".zattrs": {}}
        for name in [*coords, *variables]:
            is_coord = name in coords
            shape = list(coords[name].shape) if is_coord else [len(times), 2, 2]
            chunks = [2000] if name == "time" else ([2] if is_coord else [2000, 1, 1])
            dtype = str(coords[name].dtype) if is_coord else "<f4"
            attrs = {"_ARRAY_DIMENSIONS": [name] if is_coord else ["time", "latitude", "longitude"]}
            if name == "time":
                attrs["units"] = "hours since 1970-01-01"
            elif not is_coord:
                attrs.update(units=units[name], long_name=name)
            metadata[f"{name}/.zarray"] = {
                "zarr_format": 2,
                "shape": shape,
                "chunks": chunks,
                "dtype": dtype,
                "compressor": codec.get_config(),
                "filters": None,
                "order": "C",
                "fill_value": "NaN" if name != "time" else None,
            }
            metadata[f"{name}/.zattrs"] = attrs
        stores[group] = metadata
    if metadata_override is not None:
        group, key, changes = metadata_override
        stores[group][key].update(changes)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer unit-test-secret"
        path = request.url.path
        requests.append(path)
        group = next(group for group, info in GROUPS.items() if info[1] in path)
        key = path.split("geoChunked.zarr/", 1)[1]
        if response_status and (group, key) in response_status:
            return httpx.Response(response_status[(group, key)], text="unit-test-secret")
        if key == ".zmetadata":
            return httpx.Response(
                200, json={"zarr_consolidated_format": 1, "metadata": stores[group]}
            )
        variable, indices_text = key.split("/")
        indices = tuple(map(int, indices_text.split(".")))
        spec = stores[group][f"{variable}/.zarray"]
        if variable in coords:
            lo = indices[0] * spec["chunks"][0]
            array = coords[variable][lo : lo + spec["chunks"][0]]
        else:
            array = np.full(spec["chunks"], constants[variable], dtype=np.float32)
        return httpx.Response(200, content=codec.encode(np.ascontiguousarray(array)))

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_point_subset_is_lossless_and_has_deaccumulation_provenance(tmp_path: Path) -> None:
    http, requests = _mock_store()
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    content, manifest = client.fetch_point_season(-12.71, -55.71, 2025)
    with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as dataset:
        assert dataset.sizes == {"time": 5809, "latitude": 1, "longitude": 1}
        assert dataset["latitude"].item() == -12.7
        assert dataset["longitude"].item() == -55.7
        assert dataset["t2m"].dtype == np.dtype("float32")
        assert np.all(dataset["t2m"].values == np.float32(300))
        assert np.all(dataset["tp"].values == np.float32(0.0001))
        assert str(dataset["time"].values[-1])[:13] == "2025-05-01T00"
        assert dataset.attrs["accumulation_convention"] == "hourly_deaccumulated"
    assert manifest["access_method"] == "official_arco_zarr"
    assert manifest["hourly_samples"] == 5809
    chunks = [
        record
        for record in manifest["source_artifacts"]
        if record["zarr_key"].split("/")[0] in {"t2m", "d2m", "u10", "v10", "tp", "ssrd"}
    ]
    assert len(chunks) == 18
    assert all(
        row["artifact"] is None and row["retention"] == "memory_cache_only" for row in chunks
    )
    assert all(len(row["checksum_sha256"]) == 64 for row in chunks)
    count = len(requests)
    client.fetch_point_season(-12.71, -55.71, 2025)
    assert len(requests) == count
    for path in tmp_path.rglob("*.json"):
        assert "unit-test-secret" not in path.read_text(encoding="utf-8")


def test_coordinate_cache_integrity_is_checked_on_restart(tmp_path: Path) -> None:
    http, _ = _mock_store()
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    _, manifest = client.fetch_point_season(-12.71, -55.71, 2025)
    coordinate = next(
        record
        for record in manifest["source_artifacts"]
        if record["group"] == "temperature" and record["zarr_key"] == "time/0"
    )
    path = tmp_path / coordinate["artifact"]
    path.write_bytes(b"corrupt-cache")
    restarted = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    with pytest.raises(ValueError, match="checksum"):
        restarted.fetch_point_season(-12.71, -55.71, 2025)


def test_authorization_failure_is_actionable_and_does_not_leak_secret(tmp_path: Path) -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(403, text="server reflected secret unit-test-secret")
        )
    )
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret", retries=1)
    with pytest.raises(RuntimeError, match="authorization denied") as error:
        client.fetch_point_season(-12.71, -55.71, 2025)
    assert "unit-test-secret" not in str(error.value)


def test_requests_outside_available_time_are_not_filled(tmp_path: Path) -> None:
    http, _ = _mock_store()
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    with pytest.raises(ValueError, match="season unavailable"):
        client.fetch_point_season(-12.71, -55.71, 2024)


@pytest.mark.parametrize(
    ("key", "changes", "message"),
    [
        ("t2m/.zattrs", {"scale_factor": 0.01}, "scale/offset"),
        ("t2m/.zattrs", {"add_offset": 273.15}, "scale/offset"),
        ("t2m/.zarray", {"fill_value": -9999.0}, "fill encoding"),
        ("t2m/.zarray", {"dtype": "<i2"}, "value or fill encoding"),
        ("t2m/.zattrs", {"_FillValue": -9999.0}, "finite missing-value"),
        ("t2m/.zattrs", {"missing_value": 3.4028235e38}, "finite missing-value"),
        ("latitude/.zattrs", {"scale_factor": 0.1}, "scale/offset"),
        ("time/.zarray", {"dtype": "<f8"}, "time value"),
        ("time/.zarray", {"fill_value": 0}, "time value"),
        ("time/.zattrs", {"calendar": "360_day"}, "time calendar"),
    ],
)
def test_unsupported_value_encodings_fail_before_reading_chunks(
    tmp_path: Path, key: str, changes: dict[str, Any], message: str
) -> None:
    http, requests = _mock_store(metadata_override=("temperature", key, changes))
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    with pytest.raises(ValueError, match=message):
        client.fetch_point_season(-12.71, -55.71, 2025)
    assert len(requests) == 1
    assert requests[0].endswith("/.zmetadata")


def test_nan_missing_value_metadata_is_compatible(tmp_path: Path) -> None:
    http, _ = _mock_store(
        metadata_override=("temperature", "t2m/.zattrs", {"missing_value": "NaN"})
    )
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    content, _ = client.fetch_point_season(-12.71, -55.71, 2025)
    with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as dataset:
        assert np.all(dataset["t2m"].values == np.float32(300))


def test_absent_variable_chunks_are_nan_with_explicit_missing_source_provenance(
    tmp_path: Path,
) -> None:
    statuses = {
        (group, f"{variable}/{index}.1.1"): 404
        for group, (_, _, variables) in GROUPS.items()
        for variable in variables
        for index in range(3)
    }
    http, requests = _mock_store(response_status=statuses)
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret", retries=1)
    content, manifest = client.fetch_point_season(-12.71, -55.71, 2025)
    with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as dataset:
        for name in ("t2m", "d2m", "u10", "v10", "tp", "ssrd"):
            assert np.isnan(dataset[name].values).all()
    missing = [row for row in manifest["source_artifacts"] if row.get("missing_chunk")]
    assert len(missing) == 18
    for row in missing:
        assert row["http_status"] == 404
        assert row["fill_value"] == "NaN"
        assert row["artifact"] is None
        assert row["retention"] == "absent_source_chunk"
        assert len(row["metadata_checksum_sha256"]) == 64
        assert row["url"].endswith(row["zarr_key"])
        assert "checksum_sha256" not in row
        assert "bytes" not in row
        assert "unit-test-secret" not in str(row)
    count = len(requests)
    client.fetch_point_season(-12.71, -55.71, 2025)
    assert len(requests) == count


@pytest.mark.parametrize(
    ("key", "status"),
    [
        (".zmetadata", 404),
        ("time/0", 404),
        ("latitude/0", 404),
        ("longitude/0", 404),
        ("t2m/0.1.1", 403),
        ("t2m/0.1.1", 400),
        ("t2m/0.1.1", 429),
        ("t2m/0.1.1", 500),
    ],
)
def test_other_http_failures_are_not_interpreted_as_missing_values(
    tmp_path: Path, key: str, status: int
) -> None:
    http, _ = _mock_store(response_status={("temperature", key): status})
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret", retries=1)
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        client.fetch_point_season(-12.71, -55.71, 2025)
    assert "unit-test-secret" not in str(error.value)


@pytest.mark.parametrize("indices", [(-1, 0, 0), (3, 0, 0), (0, 2, 0), (0, 0)])
def test_invalid_chunk_indices_are_rejected_before_fetch(
    tmp_path: Path, indices: tuple[int, ...]
) -> None:
    http, requests = _mock_store()
    client = Era5ArcoClient(tmp_path, http=http, api_key="unit-test-secret")
    with pytest.raises(ValueError, match="outside declared array"):
        client._chunk("temperature", "t2m", indices)
    assert len(requests) == 1
    assert requests[0].endswith("/.zmetadata")
