"""Read-only climate/provenance audit; run after ingestion has stopped.

Usage: .venv\\Scripts\\python.exe scripts/verify_climate_backfill.py
Prints JSON to stdout, never writes artifacts, and exits 1 for inconsistencies.
Unfilled, explicitly reported climate gaps do not by themselves fail the audit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal

from agri_decision.config import load_settings
from agri_decision.ingestion.era5.arco import BASE_URL, GROUPS
from agri_decision.ingestion.era5.backfill import KEY, adequate_climate
from agri_decision.provenance.artifacts import sha256_bytes, sha256_file

ERA5_SOURCES = {"Copernicus ERA5-Land reanalysis", "Copernicus ERA5-Land ARCO reanalysis"}
STATION_COLUMNS = ["station_code", "station_name", "station_uf", "distance_to_station_km"]


class _ArcoSourceAudit:
    """Verify retained bytes; only validate records for non-retained remote chunks."""

    def __init__(
        self,
        *,
        root: Path,
        bronze: Path,
        digest: Callable[[Path], str | None],
        read_json: Callable[[Path], dict[str, Any] | None],
        fail: Callable[..., None],
    ) -> None:
        self.root, self.bronze = root.resolve(), bronze.resolve()
        self.digest, self.read_json, self.fail = digest, read_json, fail
        self.metadata: set[tuple[str, str]] = set()
        self.coordinates: set[tuple[str, str]] = set()
        self.remote: set[tuple[str, str]] = set()
        self.missing: set[tuple[str, str]] = set()
        self.manifests_checked = 0

    def check(self, manifest: dict[str, Any], *, subset: str) -> None:
        self.manifests_checked += 1

        def fail(check: str, **details: Any) -> None:
            self.fail(check, subset=subset, **details)

        raw_root = manifest.get("source_artifact_root")
        records = manifest.get("source_artifacts")
        if not isinstance(raw_root, str) or not isinstance(records, list) or not records:
            fail("arco_source_artifacts_present")
            return
        source_root = (self.root / raw_root).resolve()
        if not source_root.is_relative_to(self.bronze):
            fail("arco_source_root_within_bronze")
            return
        metadata: dict[str, tuple[str, dict[str, Any]]] = {}
        coordinates: set[tuple[str, str]] = set()
        variables: set[tuple[str, str]] = set()
        pending_missing: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                fail("arco_source_record_object")
                continue
            group, key, url = record.get("group"), record.get("zarr_key"), record.get("url")
            if not isinstance(group, str) or group not in GROUPS or not isinstance(key, str):
                fail("arco_source_group_and_key")
                continue
            identifier, name, group_variables = GROUPS[group]
            prefix = (
                f"{BASE_URL}/cadl-arco-geo-{identifier}/arco/reanalysis_era5_land/"
                f"{name}/geoChunked.zarr/"
            )
            is_coordinate = re.fullmatch(r"(time|latitude|longitude)/[0-9]+", key) is not None
            is_variable = any(
                re.fullmatch(rf"{variable}/[0-9]+\.[0-9]+\.[0-9]+", key)
                for variable in group_variables
            )
            if url != prefix + key or not (key == ".zmetadata" or is_coordinate or is_variable):
                # Do not echo malformed URLs, which might contain a credential.
                fail("arco_source_official_url_and_key", group=group)
                continue
            if is_variable:
                variables.add((group, key.split("/")[0]))
            if record.get("missing_chunk"):
                if (
                    not is_variable
                    or record.get("http_status") != 404
                    or record.get("fill_value") != "NaN"
                    or record.get("retention") != "absent_source_chunk"
                    or record.get("artifact") is not None
                    or "checksum_sha256" in record
                    or "bytes" in record
                ):
                    fail("arco_missing_chunk_event", group=group, zarr_key=key)
                    continue
                pending_missing.append(record)
                continue
            checksum = record.get("checksum_sha256")
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                fail("arco_source_checksum_format", group=group, zarr_key=key)
                continue
            size = record.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                fail("arco_source_byte_count", group=group, zarr_key=key)
                continue
            if is_variable:
                if record.get("retention") != "memory_cache_only" or record.get("artifact"):
                    fail("arco_remote_chunk_retention", group=group, zarr_key=key)
                    continue
                self.remote.add((url, checksum))
                continue
            raw_artifact = record.get("artifact")
            if record.get("retention") != "preserved_in_bronze" or not isinstance(
                raw_artifact, str
            ):
                fail("arco_metadata_coordinate_retention", group=group, zarr_key=key)
                continue
            artifact = (source_root / raw_artifact).resolve()
            if not artifact.is_relative_to(source_root) or not artifact.is_relative_to(self.bronze):
                fail("arco_source_artifact_within_bronze", group=group, zarr_key=key)
                continue
            actual = self.digest(artifact)
            if actual != checksum:
                fail("arco_retained_source_checksum", group=group, zarr_key=key)
                continue
            if artifact.stat().st_size != size:
                fail("arco_retained_source_byte_count", group=group, zarr_key=key)
                continue
            sidecar = self.read_json(artifact.with_suffix(artifact.suffix + ".manifest.json"))
            if sidecar is None or any(
                sidecar.get(field) != record.get(field)
                for field in ("checksum_sha256", "group", "zarr_key", "url", "bytes")
            ):
                fail("arco_retained_source_manifest", group=group, zarr_key=key)
                continue
            if key == ".zmetadata":
                payload = self.read_json(artifact)
                if (
                    payload is None
                    or payload.get("zarr_consolidated_format") != 1
                    or not isinstance(payload.get("metadata"), dict)
                ):
                    fail("arco_retained_metadata_format", group=group)
                    continue
                metadata[group] = (checksum, payload["metadata"])
                self.metadata.add((url, checksum))
            else:
                coordinates.add((group, key.split("/")[0]))
                self.coordinates.add((url, checksum))
        if set(metadata) != set(GROUPS):
            fail("arco_all_group_metadata_verified")
        expected_coordinates = {
            (group, name)
            for group in GROUPS
            for name in ("time", "latitude", "longitude")
        }
        if coordinates != expected_coordinates:
            fail("arco_all_group_coordinates_verified")
        if variables != {(group, name) for group, info in GROUPS.items() for name in info[2]}:
            fail("arco_all_variable_sources_recorded")
        for record in pending_missing:
            group, key = record["group"], record["zarr_key"]
            matched = metadata.get(group)
            if matched is None or record.get("metadata_checksum_sha256") != matched[0]:
                fail("arco_missing_chunk_metadata_reference", group=group, zarr_key=key)
                continue
            spec = matched[1].get(f"{key.split('/')[0]}/.zarray", {})
            if spec.get("fill_value") != "NaN":
                fail("arco_missing_chunk_metadata_nan", group=group, zarr_key=key)
                continue
            self.missing.add((record["url"], matched[0]))

    def summary(self) -> dict[str, Any]:
        return {
            "subset_manifests_checked": self.manifests_checked,
            "persisted_metadata_artifacts_verified": len(self.metadata),
            "persisted_coordinate_artifacts_verified": len(self.coordinates),
            "remote_data_chunks_recorded_not_retained": len(self.remote),
            "missing_chunk_events_verified": len(self.missing),
            "remote_chunk_verification": "manifest format only; bytes not retained or downloaded",
        }


def verify(
    project_root: Path,
    *,
    gold: Path,
    bronze: Path,
    expected_rows: int = 44_560,
    minimum_coverage: float = 0.70,
    maximum_grid_distance_km: float = 20.0,
) -> dict[str, Any]:
    """Audit an existing snapshot. Parameters allow isolated temporary-fixture tests."""
    errors: list[dict[str, Any]] = []
    fingerprints: dict[Path, str] = {}
    json_cache: dict[Path, dict[str, Any]] = {}
    root = project_root.resolve()

    def relative(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    def fail(check: str, **details: Any) -> None:
        errors.append({"check": check, **details})

    def read_json(path: Path) -> dict[str, Any] | None:
        path = path.resolve()
        if path in json_cache:
            return json_cache[path]
        try:
            content = path.read_bytes()
            fingerprints[path] = sha256_bytes(content)
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("JSON must be an object")
            json_cache[path] = payload
            return payload
        except (OSError, ValueError):
            fail("readable_json", path=relative(path))
            return None

    def digest(path: Path) -> str | None:
        path = path.resolve()
        if path in fingerprints:
            return fingerprints[path]
        try:
            result = str(sha256_file(path))
            fingerprints[path] = result
            return result
        except OSError:
            fail("readable_artifact", path=relative(path))
            return None

    frames: dict[str, pl.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    datasets = ["inmet_location_season", "climate_location_season", "era5_location_season"]
    for name in datasets:
        path = gold / f"{name}.parquet"
        if name == "era5_location_season" and not path.exists():
            continue
        checksum = digest(path)
        manifest = read_json(path.with_suffix(".manifest.json"))
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            fail("readable_parquet", dataset=name)
            continue
        frames[name] = frame
        if manifest is not None:
            manifests[name] = manifest
            if checksum != manifest.get("checksum_sha256") or checksum is None:
                fail("gold_manifest_checksum", dataset=name)
            if manifest.get("rows") != frame.height:
                fail("gold_manifest_row_count", dataset=name)
        if not set(KEY).issubset(frame.columns):
            fail("key_columns_present", dataset=name)
            continue
        if frame.select(pl.any_horizontal(pl.col(KEY).is_null()).any()).item():
            fail("non_null_keys", dataset=name)
        if frame.select(pl.struct(KEY).n_unique()).item() != frame.height:
            fail("unique_keys", dataset=name)
        if name != "era5_location_season" and frame.height != expected_rows:
            fail("expected_row_count", dataset=name, actual=frame.height, expected=expected_rows)

    baseline = frames.get("inmet_location_season")
    combined = frames.get("climate_location_season")
    if baseline is None or combined is None:
        return {"ok": False, "errors": errors}
    required = {
        *KEY,
        "rain_crop_cycle_mm",
        "mean_temp_c",
        "season_coverage",
        "source",
        "quality_flag",
    }
    for name, frame in frames.items():
        absent = sorted(required - set(frame.columns))
        if absent:
            fail("required_columns", dataset=name, absent=absent)
    if errors and any(e["check"] in {"required_columns", "key_columns_present"} for e in errors):
        return {"ok": False, "errors": errors}

    missing_keys = baseline.select(KEY).join(combined.select(KEY), on=KEY, how="anti")
    added_keys = combined.select(KEY).join(baseline.select(KEY), on=KEY, how="anti")
    if missing_keys.height or added_keys.height:
        fail("same_key_universe", missing=missing_keys.height, added=added_keys.height)
    good_inmet = baseline.filter(adequate_climate(minimum_coverage))
    retained = combined.join(good_inmet.select(KEY), on=KEY, how="inner")
    absent_original_columns = sorted(set(baseline.columns) - set(combined.columns))
    if absent_original_columns:
        fail("inmet_original_columns_preserved", absent=absent_original_columns)
    else:
        try:
            assert_frame_equal(
                good_inmet.sort(KEY),
                retained.select(baseline.columns).sort(KEY),
                check_dtypes=False,
                check_exact=True,
            )
        except AssertionError:
            fail("adequate_inmet_unchanged", rows_checked=good_inmet.height)

    baseline_gaps = baseline.filter(~adequate_climate(minimum_coverage))
    era5 = combined.filter(pl.col("source").is_in(ERA5_SOURCES))
    unknown_sources = combined.filter(
        ~pl.col("source").is_in([*baseline["source"].unique().to_list(), *ERA5_SOURCES])
        | pl.col("source").is_null()
    )
    if unknown_sources.height:
        fail("known_climate_source", rows=unknown_sources.height)
    if era5.join(baseline_gaps.select(KEY), on=KEY, how="anti").height:
        fail("era5_only_replaces_baseline_gaps")
    era5_required = {*STATION_COLUMNS, "distance_to_grid_km", "data_nature", "source_checksums"}
    if era5.height and not era5_required.issubset(era5.columns):
        fail("era5_provenance_columns", absent=sorted(era5_required - set(era5.columns)))
    elif era5.height:
        constraints = {
            "era5_adequate_coverage": adequate_climate(minimum_coverage),
            "era5_estimated": pl.col("data_nature").eq("estimated"),
            "era5_no_station_claim": pl.all_horizontal(pl.col(STATION_COLUMNS).is_null()),
            "era5_grid_distance": pl.col("distance_to_grid_km").is_finite()
            & pl.col("distance_to_grid_km").is_between(0, maximum_grid_distance_km),
            "era5_nonnegative_rain": pl.col("rain_crop_cycle_mm").is_finite()
            & (pl.col("rain_crop_cycle_mm") >= 0),
            "era5_source_checksums_present": pl.col("source_checksums").list.len() > 0,
        }
        for check, valid in constraints.items():
            invalid = era5.filter(~valid.fill_null(False))
            if invalid.height:
                fail(check, rows=invalid.height, sample=invalid.select(KEY).head(10).to_dicts())

    fallback = frames.get("era5_location_season")
    if fallback is not None and era5.height:
        selected = fallback.join(era5.select(KEY), on=KEY)
        try:
            assert_frame_equal(
                selected.sort(KEY),
                era5.select(fallback.columns).sort(KEY),
                check_dtypes=False,
                check_exact=True,
            )
        except (AssertionError, pl.exceptions.PolarsError):
            fail("era5_gold_matches_fallback")
    elif era5.height:
        fail("era5_fallback_present")

    combined_manifest = manifests.get("climate_location_season", {})
    source_files = combined_manifest.get("source_files", [])
    source_hashes = combined_manifest.get("source_checksums_sha256", [])
    expected_sources = {relative(gold / f"{n}.parquet") for n in frames if n != datasets[1]}
    found_sources: set[str] = set()
    if not source_files or len(source_files) != len(source_hashes):
        fail("combined_manifest_source_list")
    for source, checksum in zip(source_files, source_hashes, strict=False):
        path = (root / source).resolve()
        if not path.is_relative_to(gold.resolve()):
            fail("combined_manifest_source_within_gold")
            continue
        found_sources.add(relative(path))
        if digest(path) != checksum:
            fail("combined_manifest_source_checksum", path=relative(path))
    if expected_sources != found_sources:
        fail("combined_manifest_exact_sources")

    states: dict[tuple[str, int], dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    verified_checksums: set[str] = set()
    artifacts_checked = 0
    arco_sources = _ArcoSourceAudit(
        root=root, bronze=bronze, digest=digest, read_json=read_json, fail=fail
    )
    state_paths = sorted((bronze / "era5" / "arco_points").glob("*/*/state.json"))
    state_paths += sorted((bronze / "era5" / "jobs").glob("*/state.json"))
    for state_path in state_paths:
        state = read_json(state_path)
        if state is None:
            continue
        status = state.get("status", "unknown")
        state_counts[str(status)] += 1
        is_point = state_path.parent.parent.parent.name == "arco_points"
        if is_point:
            try:
                key = (str(state[KEY[0]]), int(state[KEY[1]]))
            except (KeyError, ValueError, TypeError):
                fail("bronze_point_identity", path=relative(state_path))
                continue
            states[key] = state
            if key != (state_path.parent.parent.name, int(state_path.parent.name)):
                fail("bronze_point_path_identity", path=relative(state_path))
        artifact_name = state.get("artifact")
        if not artifact_name:
            if status in {"available", "unavailable", "downloaded"}:
                fail("bronze_state_artifact_present", path=relative(state_path))
            continue
        artifact = (root / artifact_name).resolve()
        if not artifact.is_relative_to(state_path.parent.resolve()):
            fail("bronze_artifact_within_state_directory", path=relative(state_path))
            continue
        checksum = digest(artifact)
        manifest = read_json(artifact.with_suffix(artifact.suffix + ".manifest.json"))
        artifacts_checked += 1
        if checksum is None or checksum != state.get("checksum_sha256"):
            fail("bronze_state_checksum", path=relative(state_path))
        if manifest is None or manifest.get("checksum_sha256") != checksum or checksum is None:
            fail("bronze_manifest_checksum", path=relative(state_path))
        elif checksum == state.get("checksum_sha256"):
            verified_checksums.add(checksum)
        if is_point and manifest is not None:
            if any(manifest.get(k) != state.get(k) for k in KEY):
                fail("bronze_manifest_point_identity", path=relative(state_path))
            if manifest.get("data_nature") != "estimated":
                fail("bronze_manifest_estimated", path=relative(state_path))
            arco_sources.check(manifest, subset=relative(artifact))

    for name, frame in frames.items():
        if name == "inmet_location_season" or "source_checksums" not in frame.columns:
            continue
        era5_rows = frame.filter(pl.col("source").is_in(ERA5_SOURCES))
        checksums = set(
            era5_rows["source_checksums"].explode(empty_as_null=False).drop_nulls().to_list()
        )
        unknown = checksums - verified_checksums
        if unknown:
            fail("era5_gold_bronze_provenance_chain", dataset=name, unknown_checksums=len(unknown))
        if name == "era5_location_season":
            recorded = set(manifests.get(name, {}).get("source_checksums", []))
            if checksums != recorded:
                fail("era5_manifest_exact_source_checksums")

    remaining = combined.filter(~adequate_climate(minimum_coverage)).sort(KEY)
    available_keys = set(era5.select(KEY).iter_rows())
    arco_keys = set(
        era5.filter(pl.col("source") == "Copernicus ERA5-Land ARCO reanalysis")
        .select(KEY)
        .iter_rows()
    )
    baseline_gap_set = set(baseline_gaps.select(KEY).iter_rows())
    for key, state in states.items():
        if key not in baseline_gap_set:
            continue
        if state.get("status") == "available" and key not in available_keys:
            fail("available_point_materialized", codigo_ibge=key[0], season_year=key[1])
        if state.get("status") == "unavailable" and key in arco_keys:
            fail("unavailable_point_not_promoted", codigo_ibge=key[0], season_year=key[1])

    # Detect a concurrent checkpoint rather than certify a mixture of two snapshots.
    for path, checksum in fingerprints.items():
        try:
            if sha256_file(path) != checksum:
                fail("snapshot_changed_during_audit", path=relative(path))
        except OSError:
            fail("snapshot_changed_during_audit", path=relative(path))
    missing = []
    for row in remaining.select(KEY + ["quality_flag"]).iter_rows(named=True):
        state = states.get((row[KEY[0]], row[KEY[1]]), {})
        missing.append(
            {
                **row,
                "era5_status": state.get("status", "not_attempted"),
                "era5_quality_flag": state.get("quality_flag"),
            }
        )
    return {
        "ok": not errors,
        "rows": combined.height,
        "unique_keys": combined.select(pl.struct(KEY).n_unique()).item(),
        "adequate_inmet_rows_checked": good_inmet.height,
        "baseline_gap_rows": baseline_gaps.height,
        "era5_filled_rows": era5.height,
        "remaining_gap_rows": remaining.height,
        "source_counts": combined.group_by("source").len().sort("source").to_dicts(),
        "bronze": {
            "states_by_status": dict(sorted(state_counts.items())),
            "artifacts_checked": artifacts_checked,
            "arco_sources": arco_sources.summary(),
        },
        "missing": missing,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-rows", type=int, default=44_560)
    args = parser.parse_args()
    try:
        settings = load_settings(args.project_root)
        report = verify(
            settings.project_root,
            gold=settings.paths.gold,
            bronze=settings.paths.bronze,
            expected_rows=args.expected_rows,
            minimum_coverage=settings.climate.minimum_season_coverage,
        )
    except Exception as exc:
        report = {
            "ok": False,
            "errors": [{"check": "audit_failed", "error_type": type(exc).__name__}],
        }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
