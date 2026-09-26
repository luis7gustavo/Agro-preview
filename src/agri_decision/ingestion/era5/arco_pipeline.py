"""Resumable official ARCO point subsets with explicit hourly-increment semantics."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from filelock import FileLock

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.era5.backfill import (
    KEY,
    adequate_climate,
    materialize_combined_climate,
)
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_file,
    write_json_atomic,
)
from agri_decision.transforms.era5 import build_era5_season_features, parse_era5_daily

LOGGER = logging.getLogger(__name__)


class ArcoSource(Protocol):
    def fetch_point_season(
        self, *, latitude: float, longitude: float, season_year: int
    ) -> tuple[bytes, dict[str, Any]]: ...


def _point(
    row: dict[str, Any], *, settings: Settings, client: ArcoSource
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    code, year = row["codigo_ibge"], int(row["season_year"])
    identity = {"codigo_ibge": code, "season_year": year}
    directory = settings.paths.bronze / "era5" / "arco_points" / code / str(year)
    state_path = directory / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        if state is None or "artifact" not in state:
            content, metadata = client.fetch_point_season(
                latitude=row["latitude_centroid"],
                longitude=row["longitude_centroid"],
                season_year=year,
            )
            artifact, _ = preserve_content_addressed_artifact(
                directory=directory,
                stem="arco-subset",
                suffix=".nc",
                content=content,
                manifest={
                    **metadata,
                    **identity,
                    "data_nature": "estimated",
                    "pipeline_version": settings.pipeline_version,
                    "representation": "locally serialized subset of official ARCO Zarr",
                    "accumulation_mode": "hourly_increments",
                    "requested_latitude": row["latitude_centroid"],
                    "requested_longitude": row["longitude_centroid"],
                },
            )
            state = {
                **identity,
                "artifact": str(artifact.relative_to(settings.project_root)),
                "checksum_sha256": sha256_file(artifact),
                "status": "downloaded",
            }
            write_json_atomic(state_path, state)
        artifact = settings.project_root / state["artifact"]
        if not artifact.resolve().is_relative_to(directory.resolve()):
            raise ValueError("ARCO artifact escapes its point directory")
        if sha256_file(artifact) != state["checksum_sha256"]:
            raise ValueError("ARCO subset checksum mismatch")
        manifest = json.loads(artifact.with_suffix(".nc.manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["checksum_sha256"] != state["checksum_sha256"]
            or manifest["requested_latitude"] != row["latitude_centroid"]
            or manifest["requested_longitude"] != row["longitude_centroid"]
            or manifest["season_year"] != year
        ):
            raise ValueError("ARCO cached subset provenance does not match the requested point")
        locations = pl.DataFrame([row]).select(
            "codigo_ibge", "municipio", "uf", "latitude_centroid", "longitude_centroid"
        )
        daily = parse_era5_daily(
            artifact.read_bytes(),
            locations,
            source_checksum=state["checksum_sha256"],
            accumulation_mode="hourly_increments",
        )
        _write_parquet_atomic(daily, settings.paths.silver / "era5_arco" / code / f"{year}.parquet")
        features = build_era5_season_features(
            daily, minimum_coverage=settings.climate.minimum_season_coverage
        ).filter(pl.col("season_year") == year)
        if features.height != 1:
            raise ValueError("ARCO subset must produce exactly one selected municipality-season")
        valid = (
            features.filter(adequate_climate(settings.climate.minimum_season_coverage)).height == 1
        )
        state.update(
            status="available" if valid else "unavailable",
            quality_flag=features["quality_flag"][0],
            checked_at=datetime.now(UTC).isoformat(),
        )
        state.pop("error_type", None)
        write_json_atomic(state_path, state)
        return features, state
    except Exception as exc:
        # Preserve successful bytes and their checksum on parse errors for safe resumption.
        previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        state = {
            **previous,
            **identity,
            "status": "error",
            "error_type": type(exc).__name__,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        write_json_atomic(state_path, state)
        return None, state


def _persist_features(frames: list[pl.DataFrame], settings: Settings) -> None:
    if not frames:
        return
    target = settings.paths.gold / "era5_location_season.parquet"
    new = pl.concat(frames, how="diagonal_relaxed").unique(subset=KEY, keep="last")
    if target.exists():
        old = pl.read_parquet(target).join(new.select(KEY), on=KEY, how="anti")
        new = pl.concat([old, new], how="diagonal_relaxed")
    new = new.sort(KEY)
    _write_parquet_atomic(new, target)
    write_json_atomic(
        target.with_suffix(".manifest.json"),
        {
            "dataset": "era5_location_season",
            "rows": new.height,
            "checksum_sha256": sha256_file(target),
            "pipeline_version": settings.pipeline_version,
            "data_nature": "estimated",
            "source_counts": new.group_by("source").len().to_dicts(),
            "source_checksums": new["source_checksums"]
            .explode(empty_as_null=False)
            .unique()
            .sort()
            .to_list(),
            "license": "CC-BY-4.0",
            "calendar": "UTC September-April; semantics documented per row temporal_method",
        },
    )
    materialize_combined_climate(settings)


def run_era5_arco_backfill(
    *,
    settings: Settings | None = None,
    client: ArcoSource | None = None,
    season_year: int | None = None,
    codigo_ibge: str | None = None,
    max_new_points: int = 0,
    workers: int = 3,
    reprocess_existing: bool = False,
) -> Path:
    """Fetch only missing climate, checkpoint every ten points, never invent masked data."""
    if max_new_points < 0 or not 1 <= workers <= 3:
        raise ValueError("Invalid ARCO execution bounds")
    resolved = settings or load_settings()
    directory = resolved.paths.bronze / "era5"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "backfill.lock", timeout=0), ExitStack() as stack:
        if client is None:
            from agri_decision.ingestion.era5.arco import Era5ArcoClient

            client = Era5ArcoClient(cache_directory=directory / "arco_raw")
            stack.callback(client.close)
        baseline = pl.read_parquet(resolved.paths.gold / "inmet_location_season.parquet")
        gaps = baseline.filter(~adequate_climate(resolved.climate.minimum_season_coverage))
        if season_year is not None:
            gaps = gaps.filter(pl.col("season_year") == season_year)
        if codigo_ibge is not None:
            gaps = gaps.filter(pl.col("codigo_ibge") == codigo_ibge)
        # Put points used by the existing supervised dataset first, without widening scope.
        training_path = resolved.paths.gold / "ml_soy_yield_dataset.parquet"
        if training_path.exists():
            training = (
                pl.read_parquet(training_path)
                .select("codigo_ibge", (pl.col("ano") - 1).cast(pl.Int32).alias("season_year"))
                .unique()
                .with_columns(pl.lit(True).alias("training_priority"))
            )
            gaps = gaps.join(training, on=KEY, how="left")
        else:
            gaps = gaps.with_columns(pl.lit(False).alias("training_priority"))
        gaps = gaps.with_columns(pl.col("training_priority").fill_null(False)).sort(
            ["training_priority", "codigo_ibge", "season_year"], descending=[True, False, True]
        )
        existing = pl.read_parquet(resolved.paths.gold / "climate_location_season.parquet")
        filled_keys = existing.filter(adequate_climate(resolved.climate.minimum_season_coverage))
        pending = (
            gaps if reprocess_existing else gaps.join(filled_keys.select(KEY), on=KEY, how="anti")
        )
        selected: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        new_count = 0
        for row in pending.iter_rows(named=True):
            state_path = (
                directory
                / "arco_points"
                / row["codigo_ibge"]
                / str(row["season_year"])
                / "state.json"
            )
            state = (
                json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            )
            cached = state is not None and "artifact" in state
            if not cached and new_count >= max_new_points:
                continue
            new_count += int(not cached)
            selected.append(row)
        frames: list[pl.DataFrame] = []
        report = resolved.paths.reports / "era5_arco_backfill.json"

        def checkpoint() -> None:
            _persist_features(frames, resolved)
            frames.clear()
            combined = pl.read_parquet(resolved.paths.gold / "climate_location_season.parquet")
            selected_result = combined.join(gaps.select(KEY), on=KEY)
            missing = selected_result.filter(
                ~adequate_climate(resolved.climate.minimum_season_coverage)
            )
            unavailable = [s for s in states if s["status"] == "unavailable"]
            errors = [s for s in states if s["status"] == "error"]
            processed_scope = not errors and missing.height == len(unavailable)
            write_json_atomic(
                report,
                {
                    "checked_at": datetime.now(UTC).isoformat(),
                    "source": "Copernicus ERA5-Land ARCO",
                    "accumulation_mode": "hourly_increments",
                    "filters": {"season_year": season_year, "codigo_ibge": codigo_ibge},
                    "targeted_rows": gaps.height,
                    "points_selected_this_run": len(selected),
                    "points_processed_this_run": len(states),
                    "new_points_budget_used": new_count,
                    "reprocess_existing": reprocess_existing,
                    "source_counts": combined.group_by("source").len().sort("source").to_dicts(),
                    "remaining_gap_rows": combined.filter(
                        ~adequate_climate(resolved.climate.minimum_season_coverage)
                    ).height,
                    "selected_scope_remaining_gaps": missing.height,
                    "complete_for_selected_scope": selected_result.height > 0
                    and missing.is_empty(),
                    "collection_complete_for_selected_scope": processed_scope,
                    "coverage_status": (
                        "attention_required"
                        if errors
                        else "complete"
                        if missing.is_empty()
                        else "completed_with_unavailable_data"
                        if processed_scope
                        else "in_progress"
                    ),
                    "point_states": [
                        {k: s.get(k) for k in (*KEY, "status", "quality_flag", "error_type")}
                        for s in states
                    ],
                },
            )

        checkpoint()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_point, row, settings=resolved, client=client) for row in selected
            ]
            for future in as_completed(futures):
                frame, state = future.result()
                states.append(state)
                if frame is not None:
                    frames.append(frame)
                LOGGER.info(
                    "arco_point_processed",
                    extra={
                        **{k: state[k] for k in KEY},
                        "status": state["status"],
                        "completed": len(states),
                        "selected": len(selected),
                    },
                )
                if len(states) % 10 == 0:
                    checkpoint()
        checkpoint()
        return report
