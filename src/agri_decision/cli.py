from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from agri_decision import __version__
from agri_decision.config import load_settings, load_yaml
from agri_decision.dimensions import build_dim_crop
from agri_decision.features.training import materialize_soy_yield_training_dataset
from agri_decision.ingestion.conab import run_conab_pipeline
from agri_decision.ingestion.era5 import run_era5_canary, run_era5_readiness_check
from agri_decision.ingestion.era5.backfill import run_era5_backfill
from agri_decision.ingestion.ibge.pipeline import run_locations_pipeline, run_pam_pipeline
from agri_decision.ingestion.inmet import run_inmet_pipeline
from agri_decision.ingestion.inmet.pipeline import rebuild_inmet_climate
from agri_decision.ingestion.zarc import run_zarc_pipeline
from agri_decision.observability import configure_logging
from agri_decision.ranking import load_ranking_config
from agri_decision.taxonomy import CropTaxonomy

app = typer.Typer(help="Motor auditavel de decisao agricola do Brasil.", no_args_is_help=True)
config_app = typer.Typer(help="Valida configuracoes versionadas.")
crops_app = typer.Typer(help="Consulta a taxonomia canonica de culturas.")
dimensions_app = typer.Typer(help="Materializa e valida dimensoes canonicas.")
ingest_app = typer.Typer(help="Ingere fontes publicas oficiais em Bronze/Silver.")
build_app = typer.Typer(help="Materializa datasets analiticos Gold.")
train_app = typer.Typer(help="Treina, avalia e promove modelos sob gates objetivos.")
evaluate_app = typer.Typer(help="Consulta relatorios reproduziveis de avaliacao.")
app.add_typer(config_app, name="config")
app.add_typer(crops_app, name="crops")
app.add_typer(dimensions_app, name="dimensions")
app.add_typer(ingest_app, name="ingest")
app.add_typer(build_app, name="build")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")


@app.command()
def version() -> None:
    typer.echo(__version__)


@config_app.command("validate")
def validate_config() -> None:
    settings = load_settings()
    taxonomy = CropTaxonomy.from_file()
    ranking = load_ranking_config()
    source_config = load_yaml(settings.project_root / "configs" / "data_sources.yml")
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "settings_config_version": settings.config_version,
                "crop_config_version": taxonomy.config.config_version,
                "ranking_config_version": ranking.config_version,
                "source_config_version": source_config.get("config_version"),
            },
            ensure_ascii=False,
        )
    )


@crops_app.command("list")
def list_crops(
    include_planned: Annotated[
        bool, typer.Option("--all", help="Inclui culturas ainda nao habilitadas.")
    ] = False,
) -> None:
    taxonomy = CropTaxonomy.from_file()
    crops = [crop for crop in taxonomy.crops if crop.active or include_planned]
    typer.echo(json.dumps([crop.model_dump(mode="json") for crop in crops], ensure_ascii=False))


@crops_app.command("resolve")
def resolve_crop(
    label: str,
    source: Annotated[str | None, typer.Option(help="Nome da fonte, por exemplo ibge.")] = None,
) -> None:
    typer.echo(CropTaxonomy.from_file().resolve(label, source))


@dimensions_app.command("crop")
def materialize_crop_dimension(
    output: Annotated[
        Path | None,
        typer.Option(help="Arquivo JSON de saida; omita para imprimir no terminal."),
    ] = None,
) -> None:
    records = [crop.model_dump(mode="json") for crop in build_dim_crop(CropTaxonomy.from_file())]
    serialized = json.dumps(records, ensure_ascii=False, indent=2)
    if output is None:
        typer.echo(serialized)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    typer.echo(str(output.resolve()))


@ingest_app.command("ibge-locations")
def ingest_ibge_locations(
    refresh: Annotated[
        bool,
        typer.Option(help="Baixa novamente e preserva uma nova versao se o conteudo mudou."),
    ] = False,
) -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    output = run_locations_pipeline(settings=settings, refresh=refresh)
    typer.echo(str(output.resolve()))


@ingest_app.command("pam")
def ingest_pam(
    crop: Annotated[str, typer.Option(help="Cultura canonica ativa.")] = "soja",
    start_year: Annotated[int, typer.Option(min=1974, max=2200)] = 1974,
    end_year: Annotated[int, typer.Option(min=1974, max=2200)] = 2024,
    refresh: Annotated[
        bool,
        typer.Option(help="Consulta novamente a API mesmo quando ha Bronze local."),
    ] = False,
) -> None:
    if end_year < start_year:
        raise typer.BadParameter("end-year must be greater than or equal to start-year")
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    silver, gold, report = run_pam_pipeline(
        range(start_year, end_year + 1),
        crop=crop,
        settings=settings,
        refresh=refresh,
    )
    typer.echo(
        json.dumps(
            {
                "silver": str(silver.resolve()),
                "gold": str(gold.resolve()),
                "quality_report": str(report.resolve()),
            },
            ensure_ascii=False,
        )
    )


@ingest_app.command("conab")
def ingest_conab(
    refresh: Annotated[
        bool,
        typer.Option(help="Baixa novamente os exports oficiais de custos e precos."),
    ] = False,
) -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    costs, prices = run_conab_pipeline(settings=settings, refresh=refresh)
    typer.echo(
        json.dumps(
            {"costs": str(costs.resolve()), "prices": str(prices.resolve())},
            ensure_ascii=False,
        )
    )


@ingest_app.command("zarc")
def ingest_zarc(
    refresh: Annotated[
        bool,
        typer.Option(help="Baixa novamente a tabua oficial ZARC 2026/2027."),
    ] = False,
) -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    windows, coverage = run_zarc_pipeline(settings=settings, refresh=refresh)
    typer.echo(
        json.dumps(
            {"windows": str(windows.resolve()), "coverage": str(coverage.resolve())},
            ensure_ascii=False,
        )
    )


@ingest_app.command("all")
def ingest_all(
    refresh: Annotated[
        bool,
        typer.Option(help="Ignora o cache Bronze e consulta novamente todas as fontes."),
    ] = False,
) -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    location = run_locations_pipeline(settings=settings, refresh=refresh)
    production, history, production_report = run_pam_pipeline(
        range(1974, 2025), settings=settings, refresh=refresh
    )
    costs, prices = run_conab_pipeline(settings=settings, refresh=refresh)
    zarc_windows, zarc_coverage = run_zarc_pipeline(settings=settings, refresh=refresh)
    climate_daily, climate_station, climate_location = run_inmet_pipeline(
        settings=settings, refresh=refresh
    )
    typer.echo(
        json.dumps(
            {
                "location": str(location.resolve()),
                "production": str(production.resolve()),
                "history_features": str(history.resolve()),
                "production_report": str(production_report.resolve()),
                "costs": str(costs.resolve()),
                "prices": str(prices.resolve()),
                "zarc_windows": str(zarc_windows.resolve()),
                "zarc_coverage": str(zarc_coverage.resolve()),
                "climate_daily": str(climate_daily.resolve()),
                "climate_station": str(climate_station.resolve()),
                "climate_location": str(climate_location.resolve()),
            },
            ensure_ascii=False,
        )
    )


@ingest_app.command("inmet")
def ingest_inmet(
    start_year: Annotated[int, typer.Option(min=2000, max=2200)] = 2017,
    end_year: Annotated[int, typer.Option(min=2000, max=2200)] = 2025,
    refresh: Annotated[
        bool,
        typer.Option(help="Baixa novamente os arquivos anuais oficiais do INMET."),
    ] = False,
) -> None:
    if end_year < start_year:
        raise typer.BadParameter("end-year must be greater than or equal to start-year")
    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    daily, station, location = run_inmet_pipeline(
        range(start_year, end_year + 1), settings=settings, refresh=refresh
    )
    typer.echo(
        json.dumps(
            {
                "daily": str(daily.resolve()),
                "station_season": str(station.resolve()),
                "location_season": str(location.resolve()),
            },
            ensure_ascii=False,
        )
    )


@ingest_app.command("era5-status")
def check_era5_status() -> None:
    report = run_era5_readiness_check()
    typer.echo(report.read_text(encoding="utf-8"))


@ingest_app.command("era5-canary")
def check_era5_authenticated_download() -> None:
    report = run_era5_canary()
    typer.echo(report.read_text(encoding="utf-8"))


@build_app.command("ml-dataset")
def build_ml_dataset() -> None:
    dataset, report = materialize_soy_yield_training_dataset()
    typer.echo(
        json.dumps(
            {"dataset": str(dataset.resolve()), "quality_report": str(report.resolve())},
            ensure_ascii=False,
        )
    )


@ingest_app.command("era5-backfill")
def ingest_era5_backfill(
    season_year: Annotated[int | None, typer.Option(min=2001, max=2200)] = None,
    codigo_ibge: Annotated[str | None, typer.Option(help="Restringe a um municipio.")] = None,
    max_new_requests: Annotated[int, typer.Option(min=0, max=100)] = 0,
    max_inflight: Annotated[int, typer.Option(min=1, max=3)] = 3,
    wait_seconds: Annotated[int, typer.Option(min=0, max=3600)] = 0,
) -> None:
    """Planeja/retoma lotes ERA5; zero novas requisicoes por padrao. Reexecute para avancar."""
    deadline = time.monotonic() + wait_seconds
    remaining_budget = max_new_requests
    while True:
        report = run_era5_backfill(
            season_year=season_year,
            codigo_ibge=codigo_ibge,
            max_new_requests=remaining_budget,
            max_inflight=max_inflight,
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        remaining_budget -= payload["new_requests_submitted"]
        typer.echo(json.dumps(payload, ensure_ascii=False))
        active = sum(
            payload["job_status_counts"].get(s, 0) for s in ["accepted", "running", "successful"]
        )
        if (
            payload["downloads_complete"]
            or payload["jobs_requiring_attention"]
            or time.monotonic() >= deadline
            or (remaining_budget == 0 and active == 0)
        ):
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))


@build_app.command("climate")
def rebuild_climate() -> None:
    """Recalcula clima Gold usando Silver INMET verificado e preserva fallback ERA5."""
    _, _, path = rebuild_inmet_climate()
    typer.echo(str(path.resolve()))


@ingest_app.command("era5-arco")
def ingest_era5_arco(
    season_year: Annotated[int | None, typer.Option(min=2001, max=2200)] = None,
    codigo_ibge: Annotated[str | None, typer.Option()] = None,
    max_new_points: Annotated[int, typer.Option(min=0, max=10000)] = 0,
    workers: Annotated[int, typer.Option(min=1, max=3)] = 3,
    reprocess_existing: Annotated[
        bool, typer.Option(help="Reprocessa tambem os subsets ja usados.")
    ] = False,
) -> None:
    """Consulta pontos ERA5-Land ARCO oficiais; preserva incrementos horarios e cache."""
    from agri_decision.ingestion.era5.arco_pipeline import run_era5_arco_backfill

    settings = load_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.format == "json")
    report = run_era5_arco_backfill(
        settings=settings,
        season_year=season_year,
        codigo_ibge=codigo_ibge,
        max_new_points=max_new_points,
        workers=workers,
        reprocess_existing=reprocess_existing,
    )
    typer.echo(report.read_text(encoding="utf-8"))


@train_app.command("yield")
def train_yield_model(
    crop: Annotated[str, typer.Option(help="Cultura canonica ativa.")] = "soja",
) -> None:
    if crop != "soja":
        raise typer.BadParameter("Somente soja esta habilitada nesta fase")
    from agri_decision.models.yield_model.training import train_and_evaluate_soy_yield

    report, model = train_and_evaluate_soy_yield()
    typer.echo(
        json.dumps(
            {
                "evaluation_report": str(report.resolve()),
                "promoted_model": str(model.resolve()) if model is not None else None,
            },
            ensure_ascii=False,
        )
    )


@evaluate_app.command("yield")
def evaluate_yield_model(
    crop: Annotated[str, typer.Option(help="Cultura canonica ativa.")] = "soja",
) -> None:
    if crop != "soja":
        raise typer.BadParameter("Somente soja esta habilitada nesta fase")
    settings = load_settings()
    report = settings.paths.reports / "model_evaluation_soja.json"
    if not report.exists():
        raise FileNotFoundError("Relatorio ausente; execute `agri train yield --crop soja`")
    typer.echo(report.read_text(encoding="utf-8"))
