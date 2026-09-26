"""Bounded, restartable CDS jobs. INMET remains the primary climate source."""

from __future__ import annotations

import calendar
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from filelock import FileLock

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.era5.client import DATASET_ID, Era5Client
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

KEY = ["codigo_ibge", "season_year"]
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]


def adequate_climate(minimum_coverage: float) -> pl.Expr:
    return (
        pl.col("rain_crop_cycle_mm").is_finite()
        & pl.col("mean_temp_c").is_finite()
        & pl.col("season_coverage").is_finite()
        & pl.col("season_coverage").is_between(minimum_coverage, 1.0)
    ).fill_null(False)


def merge_climate_sources(
    inmet: pl.DataFrame, era5: pl.DataFrame, *, minimum_coverage: float = 0.70
) -> pl.DataFrame:
    """Replace an entire inadequate row; never mix daily values or overwrite good INMET."""
    if inmet.select(pl.struct(KEY).n_unique()).item() != inmet.height:
        raise ValueError("Duplicate INMET municipality-season keys")
    if era5.is_empty():
        return inmet
    if era5.select(pl.struct(KEY).n_unique()).item() != era5.height:
        raise ValueError("Duplicate ERA5 municipality-season keys")
    eligible = inmet.filter(~adequate_climate(minimum_coverage)).select(KEY)
    replacement = era5.filter(adequate_climate(minimum_coverage)).join(eligible, on=KEY)
    kept = inmet.join(replacement.select(KEY), on=KEY, how="anti")
    return pl.concat([kept, replacement], how="diagonal_relaxed").sort(KEY)


def materialize_combined_climate(settings: Settings) -> Path:
    baseline = settings.paths.gold / "inmet_location_season.parquet"
    inmet = pl.read_parquet(baseline)
    fallback = settings.paths.gold / "era5_location_season.parquet"
    era5 = pl.read_parquet(fallback) if fallback.exists() else pl.DataFrame()
    combined = merge_climate_sources(
        inmet, era5, minimum_coverage=settings.climate.minimum_season_coverage
    )
    target = settings.paths.gold / "climate_location_season.parquet"
    _write_parquet_atomic(combined, target)
    checksum = sha256_file(target)
    write_json_atomic(
        target.with_suffix(".manifest.json"),
        {
            "dataset": "climate_location_season",
            "dataset_version": f"climate-{checksum[:12]}",
            "checksum_sha256": checksum,
            "pipeline_version": settings.pipeline_version,
            "rows": combined.height,
            "source_files": [
                str(p.relative_to(settings.project_root))
                for p in [baseline, fallback]
                if p.exists()
            ],
            "source_checksums_sha256": [sha256_file(p) for p in [baseline, fallback] if p.exists()],
            "source_counts": combined.group_by("source").len().to_dicts(),
            "mixing_policy": "adequate INMET first; whole-season ERA5 fallback only",
        },
    )
    return target


def build_backfill_plan(
    inmet: pl.DataFrame,
    *,
    minimum_coverage: float = 0.70,
    season_year: int | None = None,
    codigo_ibge: str | None = None,
) -> list[dict[str, Any]]:
    gaps = inmet.filter(~adequate_climate(minimum_coverage))
    if season_year is not None:
        gaps = gaps.filter(pl.col("season_year") == season_year)
    if codigo_ibge is not None:
        gaps = gaps.filter(pl.col("codigo_ibge") == codigo_ibge)
    groups: dict[tuple[int, int, int], list[str]] = {}
    for row in gaps.iter_rows(named=True):
        lat, lon = row["latitude_centroid"], row["longitude_centroid"]
        if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
            continue
        key = int(row["season_year"]), math.floor(lat / 2), math.floor(lon / 2)
        groups.setdefault(key, []).append(row["codigo_ibge"])
    plan: list[dict[str, Any]] = []
    for (year, lat_tile, lon_tile), codes in sorted(groups.items(), reverse=True):
        # Include one grid-cell halo to retain the nearest point at tile boundaries.
        area = [lat_tile * 2 + 2.1, lon_tile * 2 - 0.1, lat_tile * 2 - 0.1, lon_tile * 2 + 2.1]
        jobs = []
        periods = [(year - 1, m) for m in range(9, 13)] + [(year, m) for m in range(1, 6)]
        for request_year, month in periods:
            closing_day = month == 5
            request = {
                "variable": VARIABLES if not closing_day else VARIABLES[-2:],
                "year": [str(request_year)],
                "month": [f"{month:02d}"],
                "day": ["01"]
                if closing_day
                else [
                    f"{d:02d}" for d in range(1, calendar.monthrange(request_year, month)[1] + 1)
                ],
                "time": ["00:00"] if closing_day else [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            }
            digest = sha256_bytes(
                json.dumps({"dataset": DATASET_ID, "request": request}, sort_keys=True).encode()
            )
            jobs.append({"id": digest, "request": request})
        plan.append(
            {
                "season_year": year,
                "tile": [lat_tile, lon_tile],
                "municipalities": sorted(codes),
                "jobs": jobs,
            }
        )
    return plan


def _load_state(directory: Path, job: dict[str, Any]) -> dict[str, Any]:
    path = directory / job["id"] / "state.json"
    if not path.exists():
        return {**job, "status": "planned"}
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if state["request"] != job["request"] or state["id"] != job["id"]:
        raise ValueError("ERA5 job fingerprint collision")
    return state


def _save_state(directory: Path, state: dict[str, Any]) -> None:
    write_json_atomic(
        directory / state["id"] / "state.json",
        {**state, "updated_at": datetime.now(UTC).isoformat()},
    )


def _materialize_completed_tiles(
    plan: list[dict[str, Any]], states: dict[str, dict[str, Any]], settings: Settings
) -> int:
    from agri_decision.transforms.era5 import (
        build_era5_season_features,
        combine_era5_daily,
        parse_era5_daily,
    )

    locations = pl.read_parquet(settings.paths.silver / "dim_location.parquet")
    seasons = []
    for tile in plan:
        if not all(states[job["id"]]["status"] == "downloaded" for job in tile["jobs"]):
            continue
        subset = locations.filter(pl.col("codigo_ibge").is_in(tile["municipalities"]))
        frames = []
        for job in tile["jobs"]:
            state = states[job["id"]]
            raw = settings.project_root / state["artifact"]
            if not raw.resolve().is_relative_to(settings.paths.bronze.resolve()):
                raise ValueError("ERA5 artifact path escapes Bronze")
            if sha256_file(raw) != state["checksum_sha256"]:
                raise ValueError("ERA5 Bronze checksum mismatch")
            frames.append(
                parse_era5_daily(raw.read_bytes(), subset, source_checksum=state["checksum_sha256"])
            )
        daily = combine_era5_daily(frames)
        daily_path = (
            settings.paths.silver
            / "era5"
            / (f"{tile['season_year']}-{tile['tile'][0]}-{tile['tile'][1]}.parquet")
        )
        _write_parquet_atomic(daily, daily_path)
        features = build_era5_season_features(
            daily, minimum_coverage=settings.climate.minimum_season_coverage
        ).filter(pl.col("season_year") == tile["season_year"])
        seasons.append(features)
    if not seasons:
        return 0
    target = settings.paths.gold / "era5_location_season.parquet"
    new = pl.concat(seasons, how="diagonal_relaxed").unique(subset=KEY, keep="last")
    if target.exists():
        old = pl.read_parquet(target).join(new.select(KEY), on=KEY, how="anti")
        new = pl.concat([old, new], how="diagonal_relaxed")
    _write_parquet_atomic(new.sort(KEY), target)
    write_json_atomic(
        target.with_suffix(".manifest.json"),
        {
            "dataset": "era5_location_season",
            "rows": new.height,
            "checksum_sha256": sha256_file(target),
            "pipeline_version": settings.pipeline_version,
            "data_nature": "estimated",
            "dataset_id": DATASET_ID,
            "calendar": "September-April, UTC; next-day 00 UTC accumulated totals",
            "license": "CC-BY-4.0",
            "source_checksums": new["source_checksums"]
            .explode(empty_as_null=False)
            .unique()
            .sort()
            .to_list(),
        },
    )
    return new.height


def _run_era5_backfill(
    *,
    settings: Settings | None = None,
    client: Era5Client | None = None,
    season_year: int | None = None,
    codigo_ibge: str | None = None,
    max_new_requests: int = 0,
    max_inflight: int = 3,
) -> Path:
    """One bounded pass: resume pending IDs, download finished jobs, submit up to the cap.

    Zero new requests is a read-only plan if there are no existing jobs. Failed/rejected
    requests are retained, not automatically resubmitted; no remote jobs are deleted.
    """
    if max_new_requests < 0 or not 1 <= max_inflight <= 3:
        raise ValueError("Invalid ERA5 request limits")
    resolved = settings or load_settings()
    baseline = resolved.paths.gold / "inmet_location_season.parquet"
    if not baseline.exists():
        raise FileNotFoundError("Rebuild INMET first to preserve its independent baseline")
    inmet = pl.read_parquet(baseline)
    plan = build_backfill_plan(
        inmet,
        minimum_coverage=resolved.climate.minimum_season_coverage,
        season_year=season_year,
        codigo_ibge=codigo_ibge,
    )
    directory = resolved.paths.bronze / "era5" / "jobs"
    jobs = {job["id"]: job for tile in plan for job in tile["jobs"]}
    states = {key: _load_state(directory, job) for key, job in jobs.items()}
    active_client = client or Era5Client()
    polling_states = dict(states)
    for path in directory.glob("*/state.json"):
        other = json.loads(path.read_text(encoding="utf-8"))
        polling_states.setdefault(other["id"], other)
    for state in polling_states.values():
        if state["status"] not in {"accepted", "running", "successful"}:
            continue
        try:
            state["status"] = active_client.status(state["request_id"])
            _save_state(directory, state)
            if state["status"] == "successful":
                content = active_client.download(state["request_id"])
                artifact, _ = preserve_content_addressed_artifact(
                    directory=directory / state["id"],
                    stem="era5",
                    suffix=".zip" if content.startswith(b"PK") else ".nc",
                    content=content,
                    manifest={
                        "dataset_id": DATASET_ID,
                        "request": state["request"],
                        "request_id": state["request_id"],
                        "data_nature": "estimated",
                        "license": "CC-BY-4.0",
                        "pipeline_version": resolved.pipeline_version,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                    },
                )
                state.update(
                    status="downloaded",
                    checksum_sha256=sha256_bytes(content),
                    artifact=str(artifact.relative_to(resolved.project_root)),
                )
                state.pop("last_error_type", None)
                _save_state(directory, state)
        except Exception as exc:
            # Do not persist exception text that could contain credential-bearing URLs.
            state["last_error_type"] = type(exc).__name__
            response = getattr(exc, "response", None)
            http_status = getattr(response, "status_code", None)
            if state["status"] == "successful" and http_status in {404, 410}:
                state["status"] = "expired"
                state["last_error_http_status"] = http_status
            _save_state(directory, state)
    # Count every persisted active job, including jobs outside this invocation's filter.
    inflight = 0
    for path in directory.glob("*/state.json"):
        status = json.loads(path.read_text(encoding="utf-8"))["status"]
        inflight += status in {"accepted", "running", "successful", "submission_unknown"}
    submitted = 0
    for state in states.values():
        if submitted >= max_new_requests or inflight >= max_inflight:
            break
        if state["status"] != "planned":
            continue
        # If interrupted during POST, do not blindly create a duplicate on restart.
        state["status"] = "submission_unknown"
        _save_state(directory, state)
        try:
            state["request_id"] = active_client.submit(state["request"])
            state["status"] = "accepted"
        except Exception as exc:
            state["last_error_type"] = type(exc).__name__
            _save_state(directory, state)
            break
        _save_state(directory, state)
        submitted += 1
        inflight += 1
    materialized = _materialize_completed_tiles(plan, states, resolved)
    target = materialize_combined_climate(resolved)
    combined = pl.read_parquet(target)
    selected = combined
    if season_year is not None:
        selected = selected.filter(pl.col("season_year") == season_year)
    if codigo_ibge is not None:
        selected = selected.filter(pl.col("codigo_ibge") == codigo_ibge)
    selected_gaps = selected.filter(
        ~adequate_climate(resolved.climate.minimum_season_coverage)
    ).height
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state["status"]] = counts.get(state["status"], 0) + 1
    report = resolved.paths.reports / "era5_backfill.json"
    write_json_atomic(
        report,
        {
            "checked_at": datetime.now(UTC).isoformat(),
            "dataset_id": DATASET_ID,
            "filters": {"season_year": season_year, "codigo_ibge": codigo_ibge},
            "tile_seasons": len(plan),
            "planned_requests": len(jobs),
            "municipality_seasons_targeted": sum(len(t["municipalities"]) for t in plan),
            "job_status_counts": counts,
            "new_requests_submitted": submitted,
            "era5_rows_materialized": materialized,
            "remaining_gap_rows": combined.filter(
                ~adequate_climate(resolved.climate.minimum_season_coverage)
            ).height,
            "source_counts": combined.group_by("source").len().to_dicts(),
            "jobs_requiring_attention": [
                {"id": s["id"], "status": s["status"], "error_type": s.get("last_error_type")}
                for s in states.values()
                if s["status"] in {"failed", "rejected", "submission_unknown"}
                or s.get("last_error_type")
            ],
            "downloads_complete": all(s["status"] == "downloaded" for s in states.values()),
            "selected_scope_remaining_gaps": selected_gaps,
            "complete_for_selected_scope": selected.height > 0 and selected_gaps == 0,
        },
    )
    return report


def run_era5_backfill(
    *,
    settings: Settings | None = None,
    client: Era5Client | None = None,
    season_year: int | None = None,
    codigo_ibge: str | None = None,
    max_new_requests: int = 0,
    max_inflight: int = 3,
) -> Path:
    """Run one bounded pass under a process lock, released even on process termination."""
    resolved = settings or load_settings()
    directory = resolved.paths.bronze / "era5"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "backfill.lock", timeout=0):
        return _run_era5_backfill(
            settings=resolved,
            client=client,
            season_year=season_year,
            codigo_ibge=codigo_ibge,
            max_new_requests=max_new_requests,
            max_inflight=max_inflight,
        )
