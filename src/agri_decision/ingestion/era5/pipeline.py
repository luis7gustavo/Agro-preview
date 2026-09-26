from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.era5.client import Era5Client, cds_credentials_configured
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_bytes,
    write_json_atomic,
)


def run_era5_readiness_check(
    *, settings: Settings | None = None, client: Era5Client | None = None
) -> Path:
    resolved = settings or load_settings()
    metadata, artifact = (client or Era5Client()).fetch_catalogue()
    _, manifest = preserve_content_addressed_artifact(
        directory=resolved.paths.bronze / "era5",
        stem="catalogue-reanalysis-era5-land",
        suffix=".json",
        content=artifact.content,
        manifest={
            "source": "Copernicus Climate Data Store",
            "dataset_id": metadata.dataset_id,
            "requested_url": artifact.requested_url,
            "final_url": artifact.final_url,
            "retrieved_at": artifact.retrieved_at.isoformat(),
            "reference_period": f"{metadata.temporal_start}/{metadata.temporal_end}",
            "license": metadata.license,
            "pipeline_version": resolved.pipeline_version,
        },
    )
    credentials = cds_credentials_configured()
    report = resolved.paths.reports / "era5_readiness.json"
    write_json_atomic(
        report,
        {
            "checked_at": datetime.now(UTC).isoformat(),
            "dataset_id": metadata.dataset_id,
            "title": metadata.title,
            "updated": metadata.updated,
            "license": metadata.license,
            "temporal_coverage": [metadata.temporal_start, metadata.temporal_end],
            "retrieve_url": metadata.retrieve_url,
            "catalogue_manifest": str(manifest.relative_to(resolved.project_root)),
            "credentials_configured": credentials,
            "ready_for_download": credentials,
            "status": "ready" if credentials else "blocked_credentials",
            "blocked_reason": (
                None
                if credentials
                else "CDS API credentials and dataset terms acceptance are required"
            ),
            "mixing_policy": (
                "ERA5-Land may fill only rows without adequate INMET coverage; "
                "source and data_nature must remain explicit per municipality-season"
            ),
        },
    )
    return report


def run_era5_canary(*, settings: Settings | None = None, client: Era5Client | None = None) -> Path:
    resolved = settings or load_settings()
    active_client = client or Era5Client()
    content = active_client.fetch_canary(year=2025)
    artifact, manifest = preserve_content_addressed_artifact(
        directory=resolved.paths.bronze / "era5",
        stem="canary-sorriso-20250101",
        suffix=".nc",
        content=content,
        manifest={
            "source": "Copernicus Climate Data Store",
            "dataset_id": "reanalysis-era5-land",
            "reference_period": "2025-01-01T00:00:00Z",
            "variable": "2m_temperature",
            "spatial_canary": "Sorriso/MT bounding box",
            "data_nature": "estimated",
            "license": "CC-BY-4.0",
            "pipeline_version": resolved.pipeline_version,
        },
    )
    report = resolved.paths.reports / "era5_canary.json"
    write_json_atomic(
        report,
        {
            "checked_at": datetime.now(UTC).isoformat(),
            "status": "authenticated_download_validated",
            "dataset_id": "reanalysis-era5-land",
            "reference_period": "2025-01-01T00:00:00Z",
            "variable": "2m_temperature",
            "spatial_canary": "Sorriso/MT bounding box",
            "bytes": len(content),
            "checksum_sha256": sha256_bytes(content),
            "artifact": str(artifact.relative_to(resolved.project_root)),
            "manifest": str(manifest.relative_to(resolved.project_root)),
            "full_backfill_materialized": False,
            "next_step": (
                "design bounded tile requests for municipality-seasons without "
                "adequate INMET coverage"
            ),
        },
    )
    return report
