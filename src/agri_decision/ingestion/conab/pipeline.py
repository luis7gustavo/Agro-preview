from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.ingestion.conab.client import ConabClient
from agri_decision.ingestion.http import HttpArtifact
from agri_decision.ingestion.ibge.pipeline import _write_parquet_atomic
from agri_decision.provenance.artifacts import (
    preserve_content_addressed_artifact,
    sha256_file,
    write_json_atomic,
)
from agri_decision.transforms.conab import parse_soy_costs, parse_soy_prices

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawTextArtifact:
    content: bytes
    path: Path
    checksum: str


def _cached_text(directory: Path) -> Path | None:
    candidates = sorted(
        path for path in directory.glob("*.txt") if not path.name.endswith(".manifest.txt")
    )
    return candidates[-1] if candidates else None


def _obtain_text(
    *,
    directory: Path,
    stem: str,
    source_name: str,
    pipeline_version: str,
    refresh: bool,
    fetch: object,
) -> RawTextArtifact:
    cached = None if refresh else _cached_text(directory)
    if cached is not None:
        return RawTextArtifact(cached.read_bytes(), cached, sha256_file(cached))
    if not callable(fetch):
        raise TypeError("fetch must be callable")
    downloaded = fetch()
    if not isinstance(downloaded, HttpArtifact):
        raise TypeError("Conab client returned an invalid HTTP artifact")
    path, _ = preserve_content_addressed_artifact(
        directory=directory,
        stem=stem,
        suffix=".txt",
        content=downloaded.content,
        manifest={
            "source": source_name,
            "download_url": downloaded.final_url,
            "requested_url": downloaded.requested_url,
            "download_timestamp": downloaded.retrieved_at.isoformat(),
            "reference_period": "all_periods_in_current_export",
            "pipeline_version": pipeline_version,
            "content_type": downloaded.content_type,
            "encoding_used_by_parser": "Windows-1252",
        },
    )
    return RawTextArtifact(downloaded.content, path, sha256_file(path))


def _manifest(frame: pl.DataFrame, path: Path, *, dataset: str, source: RawTextArtifact) -> None:
    checksum = sha256_file(path)
    write_json_atomic(
        path.with_suffix(".manifest.json"),
        {
            "dataset": dataset,
            "dataset_version": f"conab-{checksum[:12]}",
            "checksum_sha256": checksum,
            "source_checksum_sha256": source.checksum,
            "source_file": source.path.name,
            "rows": frame.height,
            "data_nature": "observed",
            "key_rule": "source rows are preserved; no implicit aggregation",
        },
    )


def run_conab_pipeline(
    *,
    settings: Settings | None = None,
    refresh: bool = False,
    client: ConabClient | None = None,
) -> tuple[Path, Path]:
    resolved = settings or load_settings()
    source_client = client or ConabClient()
    bronze = resolved.paths.bronze / "conab"
    LOGGER.info("pipeline_started", extra={"pipeline": "conab_economics"})

    costs_raw = _obtain_text(
        directory=bronze / "costs",
        stem="CustoProducao",
        source_name="CONAB Portal de Informacoes - Custo de Producao",
        pipeline_version=resolved.pipeline_version,
        refresh=refresh,
        fetch=source_client.fetch_costs,
    )
    municipal_raw = _obtain_text(
        directory=bronze / "prices" / "municipality",
        stem="PrecosMensalMunicipio",
        source_name="CONAB Portal de Informacoes - Precos Mensais Municipio",
        pipeline_version=resolved.pipeline_version,
        refresh=refresh,
        fetch=source_client.fetch_municipal_prices,
    )
    state_raw = _obtain_text(
        directory=bronze / "prices" / "state",
        stem="PrecosMensalUF",
        source_name="CONAB Portal de Informacoes - Precos Mensais UF",
        pipeline_version=resolved.pipeline_version,
        refresh=refresh,
        fetch=source_client.fetch_state_prices,
    )

    costs = parse_soy_costs(costs_raw.content, source_checksum=costs_raw.checksum)
    prices = pl.concat(
        [
            parse_soy_prices(
                municipal_raw.content,
                coverage_level="municipality",
                source_checksum=municipal_raw.checksum,
            ),
            parse_soy_prices(
                state_raw.content,
                coverage_level="state",
                source_checksum=state_raw.checksum,
            ),
        ],
        how="diagonal_relaxed",
    )
    if costs.is_empty() or prices.is_empty():
        raise ValueError("Current Conab exports contain no valid soybean cost/price rows")

    cost_path = resolved.paths.silver / "conab_soy_costs.parquet"
    price_path = resolved.paths.silver / "conab_soy_prices.parquet"
    _write_parquet_atomic(costs, cost_path)
    _write_parquet_atomic(prices, price_path)
    _manifest(costs, cost_path, dataset="conab_soy_costs", source=costs_raw)
    # Price manifest records both territorial exports.
    price_checksum = sha256_file(price_path)
    write_json_atomic(
        price_path.with_suffix(".manifest.json"),
        {
            "dataset": "conab_soy_prices",
            "dataset_version": f"conab-{price_checksum[:12]}",
            "checksum_sha256": price_checksum,
            "source_checksums_sha256": [municipal_raw.checksum, state_raw.checksum],
            "source_files": [municipal_raw.path.name, state_raw.path.name],
            "rows": prices.height,
            "data_nature": "observed",
            "unit": "BRL/kg",
        },
    )
    report = {
        "crop": "soja",
        "cost_rows": costs.height,
        "cost_year_min": costs.get_column("ano").min(),
        "cost_year_max": costs.get_column("ano").max(),
        "cost_reference_municipalities": costs.get_column("codigo_ibge_fonte").n_unique(),
        "cost_ufs": costs.get_column("uf").n_unique(),
        "price_rows": prices.height,
        "price_year_min": prices.get_column("ano").min(),
        "price_year_max": prices.get_column("ano").max(),
        "price_ufs": prices.get_column("uf").n_unique(),
        "invalid_nonpositive_cost_rows": costs.filter(pl.col("custo_total_brl_ha") <= 0).height,
        "invalid_nonpositive_price_rows": prices.filter(pl.col("preco_brl_kg") <= 0).height,
    }
    write_json_atomic(resolved.paths.reports / "conab_economics_quality.json", report)
    LOGGER.info("pipeline_finished", extra={"pipeline": "conab_economics", **report})
    return cost_path, price_path
