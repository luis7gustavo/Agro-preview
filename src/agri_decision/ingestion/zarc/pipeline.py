from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.ingestion.zarc.client import ZarcClient
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_file,
    write_json_atomic,
)
from agri_decision.transforms.zarc import (
    build_zarc_municipality_coverage,
    parse_soy_zarc_windows,
)

LOGGER = logging.getLogger(__name__)


def run_zarc_pipeline(
    *,
    settings: Settings | None = None,
    refresh: bool = False,
    client: ZarcClient | None = None,
) -> tuple[Path, Path]:
    resolved = settings or load_settings()
    bronze_dir = resolved.paths.bronze / "mapa" / "zarc" / "season=2026-2027"
    cached = None if refresh else next(iter(sorted(bronze_dir.glob("*.csv"))), None)
    if cached is None:
        downloaded = (client or ZarcClient()).fetch_current_table()
        raw_path, _ = preserve_content_addressed_artifact(
            directory=bronze_dir,
            stem="zarc_risk_table_2026_2027",
            suffix=".csv",
            content=downloaded.content,
            manifest={
                "source": "MAPA Open Data - ZARC risk table",
                "download_url": downloaded.final_url,
                "requested_url": downloaded.requested_url,
                "download_timestamp": downloaded.retrieved_at.isoformat(),
                "reference_period": "2026/2027",
                "pipeline_version": resolved.pipeline_version,
                "content_type": downloaded.content_type,
                "license": "Creative Commons Attribution",
            },
        )
    else:
        raw_path = cached
    checksum = sha256_file(raw_path)
    windows = parse_soy_zarc_windows(raw_path.read_bytes(), source_checksum=checksum)
    location_path = resolved.paths.silver / "dim_location.parquet"
    if not location_path.exists():
        raise FileNotFoundError("dim_location.parquet is required before ZARC")
    coverage = build_zarc_municipality_coverage(windows, pl.read_parquet(location_path))

    windows_path = resolved.paths.silver / "zarc_soy_windows.parquet"
    coverage_path = resolved.paths.gold / "zarc_soy_municipality.parquet"
    _write_parquet_atomic(windows, windows_path)
    _write_parquet_atomic(coverage, coverage_path)
    report = {
        "season": "2026/2027",
        "crop": "soja",
        "window_rows": windows.height,
        "municipalities_eligible": coverage.filter(pl.col("zarc_status") == "eligible").height,
        "municipalities_not_eligible": coverage.filter(
            pl.col("zarc_status") == "not_eligible"
        ).height,
        "municipalities_not_available": coverage.filter(
            pl.col("zarc_status") == "not_available"
        ).height,
        "ufs_covered": windows.get_column("uf").n_unique(),
        "risk_levels": windows.get_column("nivel_risco").unique().sort().to_list(),
        "source_checksum_sha256": checksum,
    }
    for path, dataset in [
        (windows_path, "zarc_soy_windows"),
        (coverage_path, "zarc_soy_municipality"),
    ]:
        output_checksum = sha256_file(path)
        write_json_atomic(
            path.with_suffix(".manifest.json"),
            {
                "dataset": dataset,
                "dataset_version": f"mapa-zarc-2026-2027-{output_checksum[:12]}",
                "checksum_sha256": output_checksum,
                "source_checksum_sha256": checksum,
                "pipeline_version": resolved.pipeline_version,
                "rows": pl.read_parquet(path).height,
                "quality": report,
            },
        )
    write_json_atomic(resolved.paths.reports / "zarc_quality.json", report)
    LOGGER.info("pipeline_finished", extra={"pipeline": "mapa_zarc", **report})
    return windows_path, coverage_path

