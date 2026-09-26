from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.features.history import add_leakage_safe_history_features
from agri_decision.ingestion.ibge.client import (
    BRAZILIAN_UFS,
    IbgeTerritoryClient,
    JsonArtifact,
    SidraPamClient,
)
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_file,
    write_json_atomic,
)
from agri_decision.quality.reports import production_quality_report
from agri_decision.taxonomy import CropTaxonomy
from agri_decision.transforms.ibge_locations import (
    build_location_records,
    location_coverage_report,
    location_records_frame,
)
from agri_decision.transforms.ibge_pam import parse_pam_rows, production_records_frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PamYearResult:
    year: int
    frame: pl.DataFrame | None
    checksum: str
    artifact_path: Path
    rows_received: int
    rows_valid: int


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        frame.write_parquet(temporary_path, compression="zstd", statistics=True)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _cached_json(directory: Path) -> Path | None:
    candidates = sorted(
        path for path in directory.glob("*.json") if not path.name.endswith(".manifest.json")
    )
    return candidates[-1] if candidates else None


def _obtain_json(
    *,
    directory: Path,
    stem: str,
    fetch: Callable[[], JsonArtifact],
    source: str,
    reference_period: str,
    pipeline_version: str,
    refresh: bool,
) -> tuple[Any, Path, str]:
    cached = None if refresh else _cached_json(directory)
    if cached is not None:
        content = cached.read_bytes()
        return json.loads(content.decode("utf-8-sig")), cached, sha256_file(cached)

    downloaded = fetch()
    artifact_path, _ = preserve_content_addressed_artifact(
        directory=directory,
        stem=stem,
        suffix=".json",
        content=downloaded.http.content,
        manifest={
            "source": source,
            "download_url": downloaded.http.final_url,
            "requested_url": downloaded.http.requested_url,
            "download_timestamp": downloaded.http.retrieved_at.isoformat(),
            "reference_period": reference_period,
            "pipeline_version": pipeline_version,
            "content_type": downloaded.http.content_type,
        },
    )
    return downloaded.rows, artifact_path, sha256_file(artifact_path)


def run_locations_pipeline(
    *,
    settings: Settings | None = None,
    refresh: bool = False,
    client: IbgeTerritoryClient | None = None,
) -> Path:
    resolved = settings or load_settings()
    source_client = client or IbgeTerritoryClient()
    bronze_root = resolved.paths.bronze / "ibge" / "territory"

    LOGGER.info("pipeline_started", extra={"pipeline": "ibge_locations"})
    municipalities, municipality_path, municipality_checksum = _obtain_json(
        directory=bronze_root / "municipalities",
        stem="municipalities",
        fetch=source_client.fetch_municipalities,
        source="IBGE Localidades API v1",
        reference_period="latest_available_at_retrieval",
        pipeline_version=resolved.pipeline_version,
        refresh=refresh,
    )

    meshes: dict[str, dict[str, Any]] = {}
    source_checksums = [municipality_checksum]
    source_files = [str(municipality_path.relative_to(resolved.project_root))]
    for uf in BRAZILIAN_UFS:
        mesh, mesh_path, mesh_checksum = _obtain_json(
            directory=bronze_root / "meshes" / f"uf={uf}",
            stem=f"municipal_mesh_{uf.lower()}",
            fetch=partial(source_client.fetch_state_mesh, uf),
            source="IBGE Malhas API v3",
            reference_period="latest_available_at_retrieval",
            pipeline_version=resolved.pipeline_version,
            refresh=refresh,
        )
        if not isinstance(mesh, dict):
            raise ValueError(f"Invalid cached IBGE mesh for {uf}")
        meshes[uf] = mesh
        source_checksums.append(mesh_checksum)
        source_files.append(str(mesh_path.relative_to(resolved.project_root)))

    if not isinstance(municipalities, list):
        raise ValueError("Invalid cached IBGE municipality payload")
    records = build_location_records(municipalities, meshes)
    frame = location_records_frame(records)
    coverage = location_coverage_report(municipalities, meshes)
    if frame.get_column("codigo_ibge").n_unique() != frame.height:
        raise ValueError("Duplicate codigo_ibge in dim_location")

    output = resolved.paths.silver / "dim_location.parquet"
    _write_parquet_atomic(frame, output)
    checksum = sha256_file(output)
    write_json_atomic(
        output.with_suffix(".manifest.json"),
        {
            "dataset": "dim_location",
            "dataset_version": f"ibge-location-{checksum[:12]}",
            "rows": frame.height,
            "checksum_sha256": checksum,
            "pipeline_version": resolved.pipeline_version,
            "source_checksums": sorted(source_checksums),
            "source_files": sorted(source_files),
            "quality": coverage,
            "centroid_method": "Shapely centroid over IBGE simplified municipal mesh",
            "geometry_format": "GeoJSON geometry serialized as UTF-8 string",
        },
    )
    write_json_atomic(resolved.paths.reports / "ibge_locations_quality.json", coverage)
    LOGGER.info(
        "pipeline_finished",
        extra={"pipeline": "ibge_locations", "rows_valid": frame.height, "output": str(output)},
    )
    return output


def run_pam_pipeline(
    years: Iterable[int],
    *,
    crop: str = "soja",
    settings: Settings | None = None,
    refresh: bool = False,
    client: SidraPamClient | None = None,
) -> tuple[Path, Path, Path]:
    resolved = settings or load_settings()
    source_client = client or SidraPamClient()
    taxonomy = CropTaxonomy.from_file(resolved.project_root / "configs" / "crops.yml")
    canonical_crop = taxonomy.resolve(crop)
    crop_record = next(item for item in taxonomy.crops if item.canonical_name == canonical_crop)
    if not crop_record.active:
        raise ValueError(f"Crop vertical is not active yet: {canonical_crop}")

    requested_years = sorted(set(years))
    if not requested_years:
        raise ValueError("At least one PAM year is required")

    LOGGER.info(
        "pipeline_started",
        extra={"pipeline": "ibge_pam", "crop": canonical_crop, "years": requested_years},
    )

    def load_year(year: int) -> PamYearResult:
        rows, artifact_path, checksum = _obtain_json(
            directory=(
                resolved.paths.bronze
                / "ibge"
                / "pam"
                / "table=5457"
                / f"crop={canonical_crop}"
                / f"year={year}"
            ),
            stem=f"sidra_5457_{canonical_crop}_{year}",
            fetch=partial(source_client.fetch_year, year, canonical_crop),
            source="IBGE/PAM/SIDRA table 5457",
            reference_period=str(year),
            pipeline_version=resolved.pipeline_version,
            refresh=refresh,
        )
        if not isinstance(rows, list):
            raise ValueError(f"Invalid cached SIDRA payload for {year}")
        records = parse_pam_rows(rows, taxonomy=taxonomy, source_checksum=checksum)
        frame = production_records_frame(records) if records else None
        return PamYearResult(
            year=year,
            frame=frame,
            checksum=checksum,
            artifact_path=artifact_path,
            rows_received=max(len(rows) - 1, 0),
            rows_valid=len(records),
        )

    new_frames: list[pl.DataFrame] = []
    source_checksums: set[str] = set()
    with ThreadPoolExecutor(
        max_workers=resolved.ingestion.max_concurrent_requests,
        thread_name_prefix="ibge-pam",
    ) as executor:
        year_results = executor.map(load_year, requested_years)
        for result in year_results:
            if result.frame is not None:
                new_frames.append(result.frame)
            source_checksums.add(result.checksum)
            LOGGER.info(
                "source_processed",
                extra={
                    "pipeline": "ibge_pam",
                    "year": result.year,
                    "rows_received": result.rows_received,
                    "rows_valid": result.rows_valid,
                    "artifact": str(result.artifact_path),
                    "checksum": result.checksum,
                },
            )

    if not new_frames:
        raise ValueError("SIDRA returned no production records for requested years")
    incoming = pl.concat(new_frames, how="diagonal_relaxed")
    silver_path = resolved.paths.silver / "production_history.parquet"
    if silver_path.exists():
        existing = pl.read_parquet(silver_path)
        merged = pl.concat([existing, incoming], how="diagonal_relaxed")
    else:
        merged = incoming
    key = ["codigo_ibge", "cultura", "ano", "sistema_produtivo"]
    merged = merged.unique(subset=key, keep="last").sort(key)
    report = production_quality_report(merged)
    if report["duplicate_keys"]:
        raise ValueError(f"Duplicate production keys after upsert: {report['duplicate_keys']}")

    _write_parquet_atomic(merged, silver_path)
    silver_checksum = sha256_file(silver_path)
    all_source_checksums = sorted(
        set(merged.get_column("source_dataset_checksum").unique().to_list()) | source_checksums
    )
    write_json_atomic(
        silver_path.with_suffix(".manifest.json"),
        {
            "dataset": "production_history",
            "dataset_version": f"ibge-pam-5457-{report['year_max']}-{silver_checksum[:12]}",
            "checksum_sha256": silver_checksum,
            "pipeline_version": resolved.pipeline_version,
            "source_table": "SIDRA 5457",
            "source_checksums": all_source_checksums,
            "quality": report,
        },
    )

    report_path = resolved.paths.reports / "ibge_pam_quality.json"
    write_json_atomic(report_path, report)

    gold = add_leakage_safe_history_features(merged)
    gold_path = resolved.paths.gold / "crop_location_year.parquet"
    _write_parquet_atomic(gold, gold_path)
    gold_checksum = sha256_file(gold_path)
    write_json_atomic(
        gold_path.with_suffix(".manifest.json"),
        {
            "dataset": "crop_location_year",
            "dataset_version": f"gold-history-v1-{gold_checksum[:12]}",
            "checksum_sha256": gold_checksum,
            "input_dataset": "production_history",
            "input_checksum_sha256": silver_checksum,
            "feature_contract_version": "history-v1",
            "leakage_rule": "all historical features are shifted by one observation",
            "pipeline_version": resolved.pipeline_version,
            "rows": gold.height,
        },
    )
    LOGGER.info(
        "pipeline_finished",
        extra={
            "pipeline": "ibge_pam",
            "rows_valid": merged.height,
            "silver": str(silver_path),
            "gold": str(gold_path),
        },
    )
    return silver_path, gold_path, report_path
