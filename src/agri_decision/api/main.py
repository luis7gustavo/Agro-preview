from __future__ import annotations

import json
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Query

from agri_decision import __version__
from agri_decision.config import load_settings
from agri_decision.recommendations import (
    DataUnavailableError,
    RecommendationRequest,
    generate_recommendation,
    load_analysis,
)
from agri_decision.taxonomy import CropTaxonomy

app = FastAPI(
    title="Motor de Aptidao, Rentabilidade e Risco Agricola",
    version=__version__,
    description="API auditavel de apoio a decisao. V1: vertical de soja.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__, "api_version": "v1"}


@app.get("/v1/locations/search")
def search_locations(
    q: str = Query(min_length=2, max_length=80), limit: int = Query(default=20, ge=1, le=100)
) -> list[dict[str, Any]]:
    path = load_settings().paths.silver / "dim_location.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail="dim_location ainda nao foi materializada")
    frame = pl.read_parquet(path)
    normalized = q.strip().lower()
    result = (
        frame.filter(
            pl.col("municipio").str.to_lowercase().str.contains(normalized, literal=True)
            | pl.col("codigo_ibge").cast(pl.Utf8).str.contains(normalized, literal=True)
        )
        .select("codigo_ibge", "municipio", "uf")
        .sort(["municipio", "uf"])
        .head(limit)
    )
    return result.to_dicts()


@app.get("/v1/crops")
def crops(include_planned: bool = False) -> list[dict[str, Any]]:
    taxonomy = CropTaxonomy.from_file()
    return [
        crop.model_dump(mode="json") for crop in taxonomy.crops if crop.active or include_planned
    ]


@app.post("/v1/recommendations", status_code=201)
def recommend(request: RecommendationRequest) -> dict[str, Any]:
    try:
        return generate_recommendation(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(
            status_code=422, detail={"message": exc.detail, "alerts": exc.alerts}
        ) from exc


@app.get("/v1/recommendations/{analysis_id}")
def get_recommendation(analysis_id: str) -> dict[str, Any]:
    try:
        return load_analysis(analysis_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/recommendations/{analysis_id}/audit")
def get_audit(analysis_id: str) -> dict[str, Any]:
    try:
        return dict(load_analysis(analysis_id)["audit"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/data-coverage")
def data_coverage() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    datasets = {
        "locations": paths.silver / "dim_location.parquet",
        "production": paths.silver / "production_history.parquet",
        "costs": paths.silver / "conab_soy_costs.parquet",
        "prices": paths.silver / "conab_soy_prices.parquet",
    }
    result: dict[str, Any] = {}
    for name, path in datasets.items():
        if not path.exists():
            result[name] = {"available": False, "rows": 0}
            continue
        frame = pl.read_parquet(path)
        result[name] = {"available": True, "rows": frame.height}
    result["zarc"] = {"available": False, "status": "adapter_pending"}
    zarc_path = paths.gold / "zarc_soy_municipality.parquet"
    if zarc_path.exists():
        zarc = pl.read_parquet(zarc_path)
        result["zarc"] = {
            "available": True,
            "rows": zarc.height,
            "eligible_municipalities": zarc.filter(pl.col("zarc_status") == "eligible").height,
            "season": "2026/2027",
        }
    climate_path = paths.gold / "climate_location_season.parquet"
    result["climate"] = {"available": False, "status": "not_materialized"}
    if climate_path.exists():
        climate = pl.read_parquet(climate_path)
        available = (
            pl.col("rain_crop_cycle_mm").is_not_null()
            & pl.col("rain_crop_cycle_mm").is_finite()
            & pl.col("mean_temp_c").is_not_null()
            & pl.col("mean_temp_c").is_finite()
            & pl.col("season_coverage").is_finite()
            & pl.col("season_coverage").is_between(settings.climate.minimum_season_coverage, 1.0)
        )
        is_inmet = pl.col("source").str.contains("INMET", literal=True).fill_null(False)
        is_era5 = pl.col("source").str.contains("ERA5", literal=True).fill_null(False)
        result["climate"] = {
            "available": True,
            "rows": climate.height,
            "municipalities": climate.get_column("codigo_ibge").n_unique(),
            "season_year_min": climate.get_column("season_year").min(),
            "season_year_max": climate.get_column("season_year").max(),
            "inmet_within_radius_rows": climate.filter(
                available
                & is_inmet
                & (pl.col("distance_to_station_km") <= settings.climate.max_station_distance_km)
            ).height,
            "era5_land_fallback_rows": climate.filter(available & is_era5).height,
            "available_rows": climate.filter(available).height,
            "missing_climate_rows": climate.filter(~available.fill_null(False)).height,
            "source_counts": climate.group_by("source").len().sort("source").to_dicts(),
            "outside_inmet_radius_rows": climate.filter(
                (pl.col("quality_flag") == "NO_NEARBY_INMET_STATION") | is_era5
            ).height,
            "unfilled_outside_inmet_radius_rows": climate.filter(
                (pl.col("quality_flag") == "NO_NEARBY_INMET_STATION") & ~is_era5
            ).height,
        }
    model_pointer = paths.models / "yield" / "soja" / "latest.json"
    result["ml_model"] = {"available": False, "status": "not_promoted"}
    if model_pointer.exists():
        pointer = json.loads(model_pointer.read_text(encoding="utf-8"))
        result["ml_model"] = {
            "available": pointer.get("promoted") is True,
            "status": "promoted" if pointer.get("promoted") is True else "not_promoted",
            "model_version": pointer.get("model_version"),
        }
    return result
