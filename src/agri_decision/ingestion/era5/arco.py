"""Read-only access to official ECMWF ERA5-Land geo-chunked Zarr v2 assets.

This deliberately bounded reader supports the published Blosc, C-order,
unfiltered Zarr v2 layout only. Metadata and exact selected values are preserved;
large source chunks use a bounded in-memory cache and retain checksums in manifests.
ARCO tp/ssrd are hourly increments, unlike the CDS hourly archive.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, RLock, Semaphore
from typing import Any

import httpx
import numpy as np
import xarray as xr
import yaml
from filelock import FileLock
from numcodecs import get_codec  # type: ignore[import-untyped]
from numpy.typing import NDArray

from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_bytes,
    write_json_atomic,
)

GUIDE_URL = (
    "https://confluence.ecmwf.int/spaces/CKB/pages/536218894/"
    "ERA5-Land+hourly+Analysis+Ready+Cloud+Optimised+ARCO+data+on+single+levels+"
    "from+1950+to+present+Product+User+Guide+PUG"
)
BASE_URL = "https://arco.datastores.ecmwf.int"
GROUPS = {
    "temperature": ("007", "sfc-2m-temperature", ("t2m", "d2m")),
    "wind": ("008", "sfc-wind", ("u10", "v10")),
    "rain": ("009", "sfc-pressure-precipitation", ("tp",)),
    # Exact geo URL is published in the guide's Advanced Usage notebook.
    "radiation": ("010", "sfc-radiation-heat", ("ssrd",)),
}


class _MissingZarrChunk(RuntimeError):
    """An absent variable chunk, not an authorization or metadata failure."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("ARCO variable chunk is absent (HTTP 404)")
        self.record = record


def _is_nan_fill(value: Any) -> bool:
    """Zarr v2 serializes floating NaN as the string ``NaN``."""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_value_encoding(name: str, spec: dict[str, Any], attrs: dict[str, Any]) -> None:
    """Reject encodings that raw chunk decoding would silently misinterpret.

    The published store uses unpacked floating values with NaN masks and integer
    time. CF packing or finite missing-value sentinels require a different reader;
    never copy them into a NetCDF whose units falsely imply decoded values.
    """
    if "scale_factor" in attrs or "add_offset" in attrs:
        raise ValueError("Unsupported ARCO scale/offset encoding; reader update required")
    dtype = np.dtype(spec["dtype"])
    if name == "time":
        if dtype.kind != "i" or spec.get("fill_value") is not None:
            raise ValueError("Unsupported ARCO time value encoding")
        if attrs.get("calendar", "proleptic_gregorian") != "proleptic_gregorian":
            raise ValueError("Unsupported ARCO time calendar")
        if "_FillValue" in attrs or "missing_value" in attrs:
            raise ValueError("Unsupported ARCO time missing-value encoding")
    elif dtype.kind != "f" or not _is_nan_fill(spec.get("fill_value")):
        raise ValueError("Unsupported ARCO floating value or fill encoding")
    for attribute in ("_FillValue", "missing_value"):
        if attribute in attrs and not _is_nan_fill(attrs[attribute]):
            raise ValueError("Unsupported ARCO finite missing-value encoding")


def _credential() -> str:
    key = os.getenv("CDSAPI_KEY")
    if not key:
        path = Path(os.getenv("CDSAPI_RC", str(Path.home() / ".cdsapirc")))
        try:
            configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            raise RuntimeError("CDS API credentials are not configured") from None
        key = configuration.get("key") if isinstance(configuration, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("CDS API credentials are not configured")
    return key.strip()


class Era5ArcoClient:
    """Fetch small point-seasons without submitting queued CDS jobs.

    Use one instance for the entire run to share metadata and coordinate arrays.
    Small coordinates/metadata use verified disk cache. Data chunks use 128 MiB
    memory cache, so retaining global 4-year chunks cannot exhaust local disk.
    """

    def __init__(
        self,
        cache_directory: Path,
        *,
        http: httpx.Client | None = None,
        api_key: str | None = None,
        retries: int = 3,
    ) -> None:
        self.cache_directory = cache_directory.resolve()
        self._http = http or httpx.Client(timeout=90, follow_redirects=False)
        self._owns_http = http is None
        self._headers = {"Authorization": f"Bearer {api_key or _credential()}"}
        self._retries = max(1, retries)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._coordinates: dict[tuple[str, str], NDArray[Any]] = {}
        self._metadata_records: dict[str, dict[str, Any]] = {}
        self._coordinate_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._initialization_lock = RLock()
        self._network_limit = Semaphore(3)
        self._chunk_locks: dict[str, Lock] = {}
        self._memory_cache: OrderedDict[str, tuple[bytes, dict[str, Any]]] = OrderedDict()
        self._missing_chunks: dict[str, dict[str, Any]] = {}
        self._memory_cache_bytes = 0
        self._memory_limit_bytes = 128 * 1024 * 1024
        self._cache_lock = RLock()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def _url(self, group: str, key: str) -> str:
        identifier, name, _ = GROUPS[group]
        return (
            f"{BASE_URL}/cadl-arco-geo-{identifier}/arco/reanalysis_era5_land/"
            f"{name}/geoChunked.zarr/{key}"
        )

    def _raw(self, group: str, key: str) -> tuple[bytes, dict[str, Any]]:
        persistent = key == ".zmetadata" or key.split("/")[0] in {"time", "latitude", "longitude"}
        if not persistent:
            identity = self._url(group, key)
            with self._cache_lock:
                lock = self._chunk_locks.setdefault(identity, Lock())
            with lock:
                return self._raw_unlocked(group, key, persistent=False)
        directory = self.cache_directory / group / sha256_bytes(key.encode())[:20]
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / "download.lock"), timeout=300):
            return self._raw_unlocked(group, key, persistent=True)

    def _raw_unlocked(
        self,
        group: str,
        key: str,
        *,
        persistent: bool,
    ) -> tuple[bytes, dict[str, Any]]:
        url = self._url(group, key)
        if not persistent:
            with self._cache_lock:
                if url in self._missing_chunks:
                    raise _MissingZarrChunk(self._missing_chunks[url])
                if url in self._memory_cache:
                    self._memory_cache.move_to_end(url)
                    return self._memory_cache[url]
        directory = self.cache_directory / group / sha256_bytes(key.encode())[:20]
        pointer = directory / "cache.json"
        if persistent and pointer.exists():
            record: dict[str, Any] = json.loads(pointer.read_text(encoding="utf-8"))
            path = self.cache_directory / record["artifact"]
            if not path.resolve().is_relative_to(self.cache_directory):
                raise ValueError("ARCO cache artifact escapes cache directory")
            content = path.read_bytes()
            if record["url"] != url or sha256_bytes(content) != record["checksum_sha256"]:
                raise ValueError("ARCO cache checksum or source mismatch")
            return content, record
        response: httpx.Response | None = None
        for attempt in range(self._retries):
            try:
                with self._network_limit:
                    response = self._http.get(url, headers=self._headers)
            except httpx.HTTPError:
                if attempt + 1 == self._retries:
                    raise RuntimeError("ARCO network request failed") from None
            else:
                if response.status_code == 200:
                    break
                if response.status_code in {401, 403}:
                    raise RuntimeError(f"ARCO authorization denied (HTTP {response.status_code})")
                if (
                    response.status_code == 404
                    and not persistent
                    and key.split("/")[0] in GROUPS[group][2]
                ):
                    record = {
                        "url": url,
                        "group": group,
                        "zarr_key": key,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "http_status": 404,
                        "missing_chunk": True,
                        "artifact": None,
                        "retention": "absent_source_chunk",
                    }
                    with self._cache_lock:
                        self._missing_chunks[url] = record
                    raise _MissingZarrChunk(record)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"ARCO source returned HTTP {response.status_code}")
                if attempt + 1 == self._retries:
                    raise RuntimeError(f"ARCO source unavailable (HTTP {response.status_code})")
            time.sleep(min(2**attempt, 5))
        if response is None or response.status_code != 200:
            raise RuntimeError("ARCO request did not complete")
        content = response.content
        record = {
            "url": url,
            "group": group,
            "zarr_key": key,
            "checksum_sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        }
        if not persistent:
            record.update(artifact=None, retention="memory_cache_only")
            with self._cache_lock:
                self._memory_cache[url] = (content, record)
                self._memory_cache_bytes += len(content)
                while self._memory_cache_bytes > self._memory_limit_bytes and self._memory_cache:
                    _, (discarded, _) = self._memory_cache.popitem(last=False)
                    self._memory_cache_bytes -= len(discarded)
            return content, record
        artifact, _ = preserve_content_addressed_artifact(
            directory=directory,
            stem="metadata" if key == ".zmetadata" else "chunk",
            suffix=".json" if key == ".zmetadata" else ".bin",
            content=content,
            manifest={**record, "dataset": "ERA5-Land ARCO", "license": "CC-BY-4.0"},
        )
        record["artifact"] = str(artifact.relative_to(self.cache_directory))
        record["retention"] = "preserved_in_bronze"
        write_json_atomic(pointer, record)
        return content, record

    def _meta(self, group: str) -> dict[str, Any]:
        with self._initialization_lock:
            return self._meta_unlocked(group)

    def _meta_unlocked(self, group: str) -> dict[str, Any]:
        if group not in self._metadata:
            content, record = self._raw(group, ".zmetadata")
            payload = json.loads(content)
            if payload.get("zarr_consolidated_format") != 1:
                raise ValueError("Unsupported ARCO consolidated metadata format")
            metadata = payload["metadata"]
            for name in ["time", "latitude", "longitude", *GROUPS[group][2]]:
                spec = metadata[f"{name}/.zarray"]
                if (
                    spec.get("zarr_format") != 2
                    or spec.get("order") != "C"
                    or spec.get("filters")
                    or spec.get("dimension_separator", ".") != "."
                    or spec.get("compressor", {}).get("id") != "blosc"
                ):
                    raise ValueError("Unsupported ARCO Zarr encoding; reader update required")
                attrs = metadata[f"{name}/.zattrs"]
                _validate_value_encoding(name, spec, attrs)
                dimensions = attrs["_ARRAY_DIMENSIONS"]
                expected = (
                    [name]
                    if name in {"time", "latitude", "longitude"}
                    else ["time", "latitude", "longitude"]
                )
                if dimensions != expected:
                    raise ValueError("Unexpected ARCO array dimensions")
            self._metadata[group] = metadata
            self._metadata_records[group] = record
        return self._metadata[group]

    def _chunk(
        self,
        group: str,
        variable: str,
        indices: tuple[int, ...],
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        metadata = self._meta(group)
        spec = metadata[f"{variable}/.zarray"]
        if len(indices) != len(spec["shape"]) or any(
            index < 0 or index >= math.ceil(total / size)
            for index, total, size in zip(indices, spec["shape"], spec["chunks"], strict=True)
        ):
            raise ValueError("ARCO chunk index outside declared array")
        key = f"{variable}/" + ".".join(map(str, indices))
        try:
            content, record = self._raw(group, key)
        except _MissingZarrChunk as exc:
            # Zarr v2 missing chunks represent fill_value, never zero observations.
            # https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#chunks
            # Metadata and coordinates must exist; _meta has validated NaN floats.
            if variable not in GROUPS[group][2] or not _is_nan_fill(spec.get("fill_value")):
                raise ValueError("Cannot interpret absent ARCO chunk fill value") from None
            missing_record = {
                **exc.record,
                "fill_value": "NaN",
                "metadata_checksum_sha256": self._metadata_records[group]["checksum_sha256"],
            }
            return np.full(spec["chunks"], np.nan, dtype=np.dtype(spec["dtype"])), missing_record
        codec = get_codec(spec["compressor"])
        decoded = codec.decode(content)
        data = np.frombuffer(decoded, dtype=np.dtype(spec["dtype"]))
        full_shape = tuple(spec["chunks"])
        if data.size == math.prod(full_shape):
            shape = full_shape
        else:
            shape = tuple(
                min(size, total - index * size)
                for size, total, index in zip(spec["chunks"], spec["shape"], indices, strict=True)
            )
            if data.size != math.prod(shape):
                raise ValueError("Invalid ARCO decoded chunk size")
        return data.reshape(shape), record

    def _coordinate(self, group: str, variable: str) -> NDArray[Any]:
        with self._initialization_lock:
            return self._coordinate_unlocked(group, variable)

    def _coordinate_unlocked(self, group: str, variable: str) -> NDArray[Any]:
        identity = (group, variable)
        if identity not in self._coordinates:
            spec = self._meta(group)[f"{variable}/.zarray"]
            arrays = []
            records = []
            for index in range(math.ceil(spec["shape"][0] / spec["chunks"][0])):
                array, record = self._chunk(group, variable, (index,))
                arrays.append(array)
                records.append(record)
            result = np.concatenate(arrays)[: spec["shape"][0]]
            if variable == "time":
                attrs = self._meta(group)["time/.zattrs"]
                if attrs.get("units") != "hours since 1970-01-01":
                    raise ValueError("Unsupported ARCO time units")
                if not np.all(np.diff(result) == 1):
                    raise ValueError("ARCO time coordinate is not continuous hourly")
                result = np.datetime64("1970-01-01T00", "h") + result.astype("timedelta64[h]")
            elif not np.all(np.diff(result) > 0) and not np.all(np.diff(result) < 0):
                raise ValueError("ARCO grid coordinate must be strictly monotonic")
            self._coordinates[identity] = result
            self._coordinate_records[identity] = records
        return self._coordinates[identity]

    def fetch_point_season(
        self,
        latitude: float,
        longitude: float,
        season_year: int,
    ) -> tuple[bytes, dict[str, Any]]:
        """Return exact six-variable point subset and auditable raw-source manifest.

        Covers Sep 1 00 UTC through May 1 00 UTC inclusive. The final midnight
        is necessary to close Apr 30 for preceding-hour precipitation/radiation.
        The caller must use the hourly-increment transform, not CDS accumulation.
        """
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude")
        if not 1951 <= season_year <= datetime.now(UTC).year:
            raise ValueError("Invalid ERA5-Land season year")
        start = np.datetime64(f"{season_year - 1}-09-01T00", "h")
        end = np.datetime64(f"{season_year}-05-01T00", "h")
        prepared: dict[str, tuple[int, int, int, int]] = {}
        grid: tuple[float, float] | None = None
        expected_time = np.arange(start, end + np.timedelta64(1, "h"), dtype="datetime64[h]")
        records: list[dict[str, Any]] = []
        for group in GROUPS:
            times = self._coordinate(group, "time")
            lats = self._coordinate(group, "latitude")
            lons = self._coordinate(group, "longitude")
            begin = int(np.searchsorted(times, start))
            finish = int(np.searchsorted(times, end, side="right"))
            if not np.array_equal(times[begin:finish], expected_time):
                raise ValueError("Requested season unavailable in official ARCO time coordinate")
            lat_index = int(np.abs(lats - latitude).argmin())
            lon_index = int(np.abs((lons - longitude + 180) % 360 - 180).argmin())
            selected_grid = (float(lats[lat_index]), float(lons[lon_index]))
            if grid is not None and selected_grid != grid:
                raise ValueError("ARCO variable groups use different selected grid cells")
            grid = selected_grid
            prepared[group] = (begin, finish, lat_index, lon_index)
            records.append(self._metadata_records[group])
            for coordinate in ("time", "latitude", "longitude"):
                records.extend(self._coordinate_records[(group, coordinate)])

        def extract(task: tuple[str, str]) -> tuple[str, NDArray[Any], list[dict[str, Any]]]:
            group, variable = task
            begin, finish, lat_index, lon_index = prepared[group]
            sizes = self._meta(group)[f"{variable}/.zarray"]["chunks"]
            parts = []
            sources = []
            for time_chunk in range(begin // sizes[0], (finish - 1) // sizes[0] + 1):
                data, source = self._chunk(
                    group,
                    variable,
                    (time_chunk, lat_index // sizes[1], lon_index // sizes[2]),
                )
                lo = max(begin - time_chunk * sizes[0], 0)
                hi = min(finish - time_chunk * sizes[0], sizes[0])
                parts.append(data[lo:hi, lat_index % sizes[1], lon_index % sizes[2]])
                sources.append(source)
            return variable, np.concatenate(parts), sources

        variables: dict[str, Any] = {}
        tasks = [(group, variable) for group, info in GROUPS.items() for variable in info[2]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for variable, values, sources in pool.map(extract, tasks):
                group = next(group for group in GROUPS if variable in GROUPS[group][2])
                attrs = self._meta(group)[f"{variable}/.zattrs"]
                variables[variable] = (
                    ("time", "latitude", "longitude"),
                    values.reshape(-1, 1, 1),
                    {"units": attrs["units"], "long_name": attrs["long_name"]},
                )
                records.extend(sources)
        if grid is None:
            raise RuntimeError("No ARCO grid selected")
        dataset = xr.Dataset(
            variables,
            coords={"time": expected_time, "latitude": [grid[0]], "longitude": [grid[1]]},
            attrs={
                "source": "Official ECMWF ERA5-Land ARCO geo-chunked Zarr",
                "accumulation_convention": "hourly_deaccumulated",
                "documentation": GUIDE_URL,
                "extraction": "exact point subset; no interpolation or climate aggregation",
            },
        )
        # NetCDF is a derived container of the exact selected source values.
        buffer = io.BytesIO()
        dataset.to_netcdf(buffer, engine="h5netcdf")
        content = buffer.getvalue()
        unique_sources = {record["url"]: record for record in records}
        manifest: dict[str, Any] = {
            "dataset_id": "reanalysis-era5-land",
            "access_method": "official_arco_zarr",
            "documentation": GUIDE_URL,
            "license": "CC-BY-4.0",
            "data_nature": "estimated",
            "accumulation_convention": "hourly_deaccumulated",
            "requested_latitude": latitude,
            "requested_longitude": longitude,
            "grid_latitude": grid[0],
            "grid_longitude": grid[1],
            "season_year": season_year,
            "time_start_utc": str(start),
            "time_end_utc_inclusive": str(end),
            "hourly_samples": len(expected_time),
            "variables": list(variables),
            "source_artifact_root": str(self.cache_directory),
            "source_artifacts": list(unique_sources.values()),
            "checksum_sha256": sha256_bytes(content),
            "derived_container": "NetCDF exact selected hourly source values",
        }
        return content, manifest
