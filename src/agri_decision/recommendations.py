from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from agri_decision.config import Settings, load_settings
from agri_decision.economics import EconomicInputs, calculate_economics
from agri_decision.provenance.artifacts import sha256_file, write_json_atomic
from agri_decision.ranking import RankingComponents, load_ranking_config, weighted_score
from agri_decision.risk import MonteCarloInputs, TriangularRange, simulate_profit
from agri_decision.taxonomy import CropTaxonomy


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    codigo_ibge: str = Field(pattern=r"^\d{7}$")
    area_ha: float | None = Field(default=None, gt=0)
    production_system: str = Field(default="rainfed", min_length=2)
    profile: str = Field(default="balanced", min_length=2)


class DataUnavailableError(RuntimeError):
    def __init__(self, detail: str, *, alerts: list[str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.alerts = alerts or []


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _as_int(value: Any) -> int:
    return int(value)


def _as_float(value: Any) -> float:
    return float(value)


def _latest_window(frame: pl.DataFrame, months: int = 12) -> pl.DataFrame:
    with_period = frame.with_columns((pl.col("ano") * 12 + pl.col("mes")).alias("_period"))
    maximum: Any = with_period.get_column("_period").max()
    if maximum is None:
        return with_period.head(0)
    return with_period.filter(pl.col("_period") >= _as_int(maximum) - months + 1).drop("_period")


def _yield_scenario(production: pl.DataFrame, code: str) -> dict[str, Any]:
    history = (
        production.filter(
            (pl.col("codigo_ibge") == code)
            & (pl.col("cultura") == "soja")
            & pl.col("produtividade_kg_ha").is_not_null()
            & (pl.col("produtividade_kg_ha") > 0)
        )
        .sort("ano")
        .tail(10)
    )
    values = [float(value) for value in history.get_column("produtividade_kg_ha").to_list()]
    if len(values) < 3:
        raise DataUnavailableError(
            "Historico municipal insuficiente para estimar produtividade de soja.",
            alerts=["INSUFFICIENT_HISTORY"],
        )
    return {
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "unit": "kg/ha",
        "type": "estimated",
        "method": "quantis empiricos dos ultimos 10 anos municipais disponiveis",
        "source": "IBGE/PAM/SIDRA table 5457",
        "reference_period": f"{history.item(0, 'ano')}-{history.item(-1, 'ano')}",
        "sample_size": len(values),
    }


def _cost_scenario(costs: pl.DataFrame, code: int, uf: str) -> dict[str, Any]:
    exact = costs.filter(pl.col("codigo_ibge_fonte") == code)
    if not exact.is_empty():
        maximum = exact.select((pl.col("ano") * 12 + pl.col("mes")).max()).item()
        selected = exact.filter((pl.col("ano") * 12 + pl.col("mes")) == maximum)
        nature, method, confidence = "observed", None, 1.0
    else:
        state = costs.filter(pl.col("uf") == uf)
        if state.is_empty():
            raise DataUnavailableError(
                "A Conab nao possui polo de custo de soja no estado solicitado.",
                alerts=["MISSING_COST_COVERAGE"],
            )
        latest_by_pole = (
            state.with_columns((pl.col("ano") * 12 + pl.col("mes")).alias("_period"))
            .sort("_period")
            .group_by("codigo_ibge_fonte", maintain_order=True)
            .last()
        )
        selected = latest_by_pole
        nature = "estimated"
        method = "mediana dos valores mais recentes dos polos Conab na mesma UF"
        confidence = 0.65 if selected.height > 1 else 0.55

    def median(column: str) -> float:
        value = selected.get_column(column).median()
        if value is None:
            raise DataUnavailableError("Componente de custo ausente no export oficial da Conab.")
        return _as_float(value)

    periods = sorted(
        f"{int(year)}-{int(month):02d}" for year, month in selected.select("ano", "mes").rows()
    )
    reference_periods = sorted(set(periods))

    return {
        "variable_brl_ha": median("custo_variavel_brl_ha"),
        "operational_brl_ha": median("custo_operacional_brl_ha"),
        "total_brl_ha": median("custo_total_brl_ha"),
        "type": nature,
        "method": method,
        "confidence": confidence,
        "source": "CONAB CustoProducao.txt",
        "reference_period": (
            periods[0] if periods[0] == periods[-1] else f"{periods[0]}/{periods[-1]}"
        ),
        "reference_periods": reference_periods,
        "source_locations": sorted(selected.get_column("municipio_fonte").unique().to_list()),
        "cost_definition": {
            "operational": "variable + fixed",
            "total": "variable + fixed + factor income",
        },
    }


def _price_scenario(prices: pl.DataFrame, code: int, uf: str) -> dict[str, Any]:
    exact = prices.filter(
        (pl.col("coverage_level") == "municipality") & (pl.col("codigo_ibge_fonte") == code)
    )
    if not exact.is_empty():
        producer = exact.filter(pl.col("nivel_comercializacao").str.contains("PRODUTOR"))
        selected = _latest_window(producer if not producer.is_empty() else exact)
        nature, method, confidence = "observed", None, min(1.0, selected.height / 12)
        coverage = "municipality"
    else:
        state = prices.filter((pl.col("coverage_level") == "state") & (pl.col("uf") == uf))
        if state.is_empty():
            raise DataUnavailableError(
                "A Conab nao possui serie de preco de soja na UF solicitada.",
                alerts=["MISSING_PRICE_COVERAGE"],
            )
        producer = state.filter(pl.col("nivel_comercializacao").str.contains("PRODUTOR"))
        selected = _latest_window(producer if not producer.is_empty() else state)
        nature = "estimated"
        method = "quantis da serie mensal estadual Conab como fallback municipal"
        confidence = min(0.75, 0.45 + selected.height / 40)
        coverage = "state"
    values = [float(value) for value in selected.get_column("preco_brl_kg").to_list()]
    if not values:
        raise DataUnavailableError("Serie Conab sem valores positivos de preco.")
    periods = sorted(
        f"{int(year)}-{int(month):02d}" for year, month in selected.select("ano", "mes").rows()
    )
    market_levels = sorted(selected.get_column("nivel_comercializacao").unique().to_list())
    return {
        "p20": _quantile(values, 0.20),
        "p50": _quantile(values, 0.50),
        "p80": _quantile(values, 0.80),
        "unit": "BRL/kg",
        "type": nature,
        "method": method,
        "confidence": confidence,
        "coverage_level": coverage,
        "market_levels": market_levels,
        "source": "CONAB precos mensais",
        "reference_period": f"{periods[0]}/{periods[-1]}",
        "sample_size": len(values),
    }


def _zarc_scenario(path: Path, code: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_available",
            "source": None,
            "method": "ZARC dataset not materialized",
            "confidence": 0.0,
        }
    frame = pl.read_parquet(path).filter(pl.col("codigo_ibge") == code)
    if frame.is_empty():
        return {
            "status": "not_available",
            "source": "MAPA ZARC open risk table 2026/2027",
            "method": "municipality absent from materialized coverage dimension",
            "confidence": 0.0,
        }
    row = frame.row(0, named=True)
    result: dict[str, Any] = {
        "status": row["zarc_status"],
        "season": f"{row['safra_inicio']}/{row['safra_fim']}",
        "source": row["source"],
        "method": row["status_method"],
        "confidence": 1.0 if row["zarc_status"] != "not_available" else 0.0,
    }
    if row["zarc_status"] == "eligible":
        result.update(
            {
                "best_risk_level": row["melhor_nivel_risco"],
                "eligible_decendios": row["decendios_elegiveis"],
                "cycle_codes": row["ciclos_codigos"],
                "soil_type_codes": row["tipos_solo_codigos"],
                "ordinances": row["portarias"],
                "eligible_combinations": row["combinacoes_elegiveis"],
                "window_note": (
                    "decendios are the municipal union; exact windows depend on soil and cycle"
                ),
            }
        )
    return result


def _climate_scenario(
    path: Path,
    code: str,
    *,
    reference_season: int | None = None,
    max_distance_km: float,
    minimum_coverage: float = 0.70,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_available",
            "type": "not_available",
            "method": "municipality-season climate dataset not materialized",
            "source": None,
            "confidence": 0.0,
        }
    municipality = pl.read_parquet(path).filter(pl.col("codigo_ibge") == code)
    if reference_season is not None:
        municipality = municipality.filter(pl.col("season_year") == reference_season)
    elif not municipality.is_empty():
        latest: Any = municipality.get_column("season_year").max()
        municipality = municipality.filter(pl.col("season_year") == latest)
    if municipality.is_empty():
        return {
            "status": "not_available",
            "type": "not_available",
            "method": "no municipality-season climate row for the requested reference",
            "source": None,
            "reference_season": reference_season,
            "confidence": 0.0,
        }
    row = municipality.row(0, named=True)
    distance = row.get("distance_to_station_km")
    coverage = row.get("season_coverage")
    is_reanalysis = "ERA5" in str(row.get("source", ""))
    available = (
        all(
            row.get(column) is not None and math.isfinite(float(row[column]))
            for column in ["rain_crop_cycle_mm", "mean_temp_c", "season_coverage"]
        )
        and minimum_coverage <= float(coverage or 0.0) <= 1.0
    )
    if is_reanalysis:
        spatial_quality = 0.75
        confidence_method = "season coverage times 0.75 reanalysis quality factor; heuristic"
    else:
        spatial_quality = (
            max(0.5, 1.0 - float(distance) / max_distance_km)
            if distance is not None and 0 <= float(distance) <= max_distance_km
            else 0.0
        )
        available = available and spatial_quality > 0
        confidence_method = "season coverage times station-distance quality factor; heuristic"
    confidence = min(1.0, float(coverage or 0.0) * spatial_quality) if available else 0.0
    return {
        "status": "available" if available else "not_available",
        "type": row["data_nature"],
        "method": row["method"],
        "source": row["source"],
        "reference_season": row["season_year"],
        "origin": "reanalysis" if is_reanalysis else "station",
        "station_code": row.get("station_code"),
        "station_name": row.get("station_name"),
        "distance_to_station_km": distance,
        "distance_to_grid_km": row.get("distance_to_grid_km"),
        "season_coverage": coverage,
        "rain_crop_cycle_mm": row["rain_crop_cycle_mm"],
        "dry_days": row.get("dry_days"),
        "mean_temp_c": row["mean_temp_c"],
        "days_temp_gt_35": row.get("days_temp_gt_35"),
        "quality_flag": row.get("quality_flag"),
        "confidence": confidence,
        "confidence_method": confidence_method,
        "alerts": ["ERA5_LAND_FALLBACK"] if is_reanalysis and available else [],
    }


def _temporal_context(
    yield_data: dict[str, Any],
    climate: dict[str, Any],
    cost: dict[str, Any],
    price: dict[str, Any],
    zarc: dict[str, Any],
) -> dict[str, Any]:
    """Expose the periods actually used without implying a forecast of one aligned season."""
    target_year = yield_data.get("target_year")
    cost_periods = cost.get("reference_periods", [cost["reference_period"]])
    cost_years = {int(period[:4]) for period in cost_periods}
    price_periods = str(price["reference_period"]).split("/")
    price_years = {int(period[:4]) for period in price_periods}
    zarc_season = zarc.get("season")
    zarc_harvest_year = int(str(zarc_season).split("/")[-1]) if zarc_season else None
    mismatches: list[str] = []
    if target_year is None:
        mismatches.append("yield uses historical quantiles without a single forecast harvest year")
    else:
        if cost_years != {target_year}:
            mismatches.append("cost reference years differ from the yield target year")
        if price_years != {target_year}:
            mismatches.append("price reference years differ from the yield target year")
        if zarc_harvest_year is not None and zarc_harvest_year != target_year:
            mismatches.append("ZARC harvest season differs from the yield target year")
        if climate.get("reference_season") not in (None, target_year - 1):
            mismatches.append("climate reference does not match the model's prior-season contract")
    return {
        "scenario_type": "hybrid_reference" if mismatches else "reference_scenario",
        "aligned_forecast": False,
        "yield_target_year": target_year,
        "yield_reference_period": yield_data["reference_period"],
        "history_through_year": yield_data.get("features_reference", {}).get(
            "history_through_year"
        ),
        "climate_reference_season": climate.get("reference_season"),
        "climate_usage": "lagged_model_feature"
        if target_year is not None
        else "historical_context",
        "cost_reference_period": cost["reference_period"],
        "cost_reference_periods": cost_periods,
        "price_reference_period": price["reference_period"],
        "zarc_season": zarc_season,
        "mismatches": mismatches,
        "note": (
            "Cenario economico com referencias de periodos diferentes; nao e uma previsao "
            "alinhada para a safra ZARC. O clima defasado e uma entrada historica do modelo."
            if mismatches
            else "Cenario com referencias no mesmo ano; custos e precos sao historicos, "
            "e o clima defasado e uma entrada historica do modelo."
        ),
    }


def _confidence(
    yield_data: dict[str, Any],
    cost: dict[str, Any],
    price: dict[str, Any],
    zarc: dict[str, Any],
    climate: dict[str, Any],
) -> float:
    if yield_data["type"] == "predicted":
        yield_quality = float(yield_data["performance_confidence"]) * 25
    else:
        yield_quality = min(float(yield_data["sample_size"]) / 10, 1.0) * 25
    climate_quality = float(climate["confidence"]) * 15
    cost_quality = float(cost["confidence"]) * 20
    price_quality = float(price["confidence"]) * 20
    zarc_quality = float(zarc["confidence"]) * 20
    return round(
        yield_quality + climate_quality + cost_quality + price_quality + zarc_quality,
        2,
    )


def _label(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _manifest_versions(settings: Settings, paths: list[Path]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in paths:
        manifest = path.with_suffix(".manifest.json")
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            version = payload.get("dataset_version", payload.get("checksum_sha256"))
            versions[path.stem] = str(version)
        elif path.exists():
            versions[path.stem] = sha256_file(path)
    return versions


def generate_recommendation(
    request: RecommendationRequest,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or load_settings()
    ranking = load_ranking_config(resolved.project_root / "configs" / "ranking.yml")
    if request.profile not in ranking.profiles:
        raise ValueError(f"Unknown ranking profile: {request.profile}")

    location_path = resolved.paths.silver / "dim_location.parquet"
    production_path = resolved.paths.silver / "production_history.parquet"
    cost_path = resolved.paths.silver / "conab_soy_costs.parquet"
    price_path = resolved.paths.silver / "conab_soy_prices.parquet"
    zarc_path = resolved.paths.gold / "zarc_soy_municipality.parquet"
    climate_path = resolved.paths.gold / "climate_location_season.parquet"
    ml_dataset_path = resolved.paths.gold / "ml_soy_yield_dataset.parquet"
    required = [location_path, production_path, cost_path, price_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise DataUnavailableError(f"Datasets obrigatorios ausentes: {', '.join(missing)}")

    code = request.codigo_ibge
    numeric_code = int(code)
    locations = pl.read_parquet(location_path)
    location_rows = locations.filter(pl.col("codigo_ibge") == code)
    if location_rows.is_empty():
        raise KeyError(f"Municipio IBGE nao encontrado: {request.codigo_ibge}")
    location = location_rows.row(0, named=True)
    uf = str(location["uf"])
    zarc = _zarc_scenario(zarc_path, code)
    if zarc["status"] == "not_eligible":
        analysis_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        excluded_audit = {
            "analysis_id": analysis_id,
            "timestamp": now,
            "dataset_versions": _manifest_versions(resolved, [*required, zarc_path]),
            "model_versions": {},
            "ranking_config_version": ranking.config_version,
            "pipeline_version": resolved.pipeline_version,
            "input_parameters": request.model_dump(mode="json"),
            "sources_used": [zarc["source"]],
            "decision": "excluded before economic ranking by official ZARC status",
        }
        excluded_response = {
            "analysis_id": analysis_id,
            "timestamp": now,
            "location": {
                "codigo_ibge": request.codigo_ibge,
                "municipio": location["municipio"],
                "uf": uf,
            },
            "area_ha": request.area_ha,
            "production_system": request.production_system,
            "profile": request.profile,
            "recommendations": [],
            "excluded_recommendations": [
                {
                    "crop": "soja",
                    "eligible": False,
                    "zarc": zarc,
                    "reason": "Municipio nao elegivel na tabela oficial ZARC da safra.",
                    "alerts": ["ZARC_NOT_ELIGIBLE"],
                }
            ],
            "audit": excluded_audit,
        }
        write_json_atomic(
            resolved.paths.gold / "analyses" / f"{analysis_id}.json", excluded_response
        )
        return excluded_response
    production = pl.read_parquet(production_path)
    historical_yield = _yield_scenario(production, code)
    yield_data = historical_yield
    model_alert: str | None = "MODEL_NOT_AVAILABLE"
    model_prediction_issue: str | None = None
    if climate_path.exists():
        try:
            from agri_decision.models.yield_model.inference import predict_soy_yield

            predicted_yield = predict_soy_yield(
                code,
                history=production,
                climate=pl.read_parquet(climate_path),
                locations=locations,
                settings=resolved,
            )
            if predicted_yield is not None:
                yield_data = predicted_yield
                model_alert = None
            else:
                evaluation_path = resolved.paths.reports / "model_evaluation_soja.json"
                if evaluation_path.exists():
                    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                    if not evaluation.get("promotion_gate", {}).get("passed", False):
                        model_alert = "MODEL_BELOW_BASELINE"
        except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
            model_alert = "MODEL_PREDICTION_UNAVAILABLE"
            model_prediction_issue = str(exc)
    climate_reference = yield_data.get("features_reference", {}).get("climate_season")
    climate = _climate_scenario(
        climate_path,
        code,
        reference_season=int(climate_reference) if climate_reference is not None else None,
        max_distance_km=resolved.climate.max_station_distance_km,
        minimum_coverage=resolved.climate.minimum_season_coverage,
    )
    cost = _cost_scenario(pl.read_parquet(cost_path), numeric_code, uf)
    price = _price_scenario(pl.read_parquet(price_path), numeric_code, uf)
    temporal_context = _temporal_context(yield_data, climate, cost, price, zarc)
    crop_record = next(
        crop for crop in CropTaxonomy.from_file().crops if crop.canonical_name == "soja"
    )
    cycle_days = round((crop_record.cycle_days_min + crop_record.cycle_days_max) / 2)

    economics = calculate_economics(
        EconomicInputs(
            expected_yield_kg_ha=yield_data["p50"],
            price_brl_kg=price["p50"],
            operational_cost_brl_ha=cost["operational_brl_ha"],
            total_cost_brl_ha=cost["total_brl_ha"],
            area_ha=request.area_ha,
            days_to_first_revenue=cycle_days,
        )
    )
    risk = simulate_profit(
        MonteCarloInputs(
            yield_kg_ha=TriangularRange(
                low=yield_data["p10"], mode=yield_data["p50"], high=yield_data["p90"]
            ),
            price_brl_kg=TriangularRange(low=price["p20"], mode=price["p50"], high=price["p80"]),
            cost_brl_ha=TriangularRange(
                low=cost["total_brl_ha"] * 0.90,
                mode=cost["total_brl_ha"],
                high=cost["total_brl_ha"] * 1.10,
            ),
            simulations=resolved.risk.simulations,
            seed=resolved.risk.seed,
        )
    )
    confidence = _confidence(yield_data, cost, price, zarc, climate)
    return_component = max(0.0, min(100.0, (economics.roi + 0.5) / 1.5 * 100))
    components = RankingComponents(
        **{
            "return": return_component,
            "risk": risk.probability_profit * 100,
            "agronomic": (
                100.0
                if zarc.get("best_risk_level") == 20
                else 75.0
                if zarc.get("best_risk_level") == 30
                else 50.0
            ),
            "speed": max(0.0, min(100.0, 100 - (cycle_days - 90) / 275 * 100)),
            "confidence": confidence,
        }
    )
    score = weighted_score(components, ranking.profiles[request.profile])
    alerts: list[str] = [*yield_data.get("alerts", []), *climate.get("alerts", [])]
    if temporal_context["mismatches"]:
        alerts.append("TEMPORAL_MISMATCH")
    if model_alert is not None:
        alerts.append(model_alert)
    if climate["status"] == "not_available":
        alerts.append("MISSING_CLIMATE")
    if climate.get("quality_flag") == "NO_NEARBY_INMET_STATION":
        alerts.append("NO_NEARBY_INMET_STATION")
    if (
        yield_data["type"] == "predicted"
        and (float(yield_data["p90"]) - float(yield_data["p10"]))
        / max(float(yield_data["p50"]), 1.0)
        > 0.5
    ):
        alerts.append("HIGH_YIELD_UNCERTAINTY")
    if zarc["status"] == "not_available":
        alerts.append("MISSING_ZARC")
    if price["sample_size"] < 12:
        alerts.append("LOW_PRICE_SAMPLE")
    if not any("PRODUTOR" in level for level in price["market_levels"]):
        alerts.append("PRICE_MARKET_LEVEL_NOT_PRODUCER")
    if cost["type"] != "observed":
        alerts.append("LOW_COST_DATA_CONFIDENCE")
    yield_reason = (
        f"Produtividade prevista pelo modelo {yield_data['model']} "
        f"({yield_data['model_version']}) para {yield_data['target_year']}, com intervalo "
        "conformal calibrado."
        if yield_data["type"] == "predicted"
        else f"Produtividade baseada em {yield_data['sample_size']} anos municipais do IBGE/PAM."
    )
    reasons = [
        yield_reason,
        f"Cenario base usa preco mediano de R$ {price['p50']:.2f}/kg da Conab.",
        f"Margem economica base de R$ {economics.economic_profit_brl_ha:,.2f}/ha.",
    ]
    if climate["status"] == "available":
        if climate["origin"] == "reanalysis":
            reasons.append(
                f"Clima da safra {climate['reference_season']} estimado pela reanalise "
                "ERA5-Land como fallback para a cobertura INMET indisponivel."
            )
        else:
            reasons.append(
                f"Clima da safra {climate['reference_season']} interpolado da estacao INMET "
                f"{climate['station_code']} a {climate['distance_to_station_km']:.1f} km."
            )
    if temporal_context["mismatches"]:
        reasons.append(temporal_context["note"])
    if zarc["status"] == "eligible":
        reasons.append(
            f"ZARC oficial {zarc['season']} indica elegibilidade, com melhor risco "
            f"de {zarc['best_risk_level']}%."
        )
    else:
        reasons.append("ZARC indisponivel: ausencia reduz a confianca, mas nao exclui.")
    analysis_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    dataset_versions = _manifest_versions(
        resolved, [*required, zarc_path, climate_path, ml_dataset_path]
    )
    model_versions = (
        {"soy_yield": str(yield_data["model_version"])} if yield_data["type"] == "predicted" else {}
    )
    sources_used = list(
        dict.fromkeys(
            source
            for source in [
                yield_data["source"],
                cost["source"],
                price["source"],
                zarc["source"],
                climate["source"],
            ]
            if source
        )
    )
    yield_assumption = (
        "yield is a versioned model prediction using lagged history and completed "
        "prior-season climate"
        if yield_data["type"] == "predicted"
        else "yield uses empirical municipal historical quantiles because no promoted "
        "model prediction was available"
    )
    audit: dict[str, Any] = {
        "analysis_id": analysis_id,
        "timestamp": now,
        "dataset_versions": dataset_versions,
        "model_versions": model_versions,
        "ranking_config_version": ranking.config_version,
        "pipeline_version": resolved.pipeline_version,
        "input_parameters": request.model_dump(mode="json"),
        "sources_used": sources_used,
        "assumptions": [
            yield_assumption,
            "Monte Carlo cost range is a documented +/-10% sensitivity around the selected cost",
        ],
        "model_prediction_issue": model_prediction_issue,
        "temporal_context": temporal_context,
    }
    response = {
        "analysis_id": analysis_id,
        "timestamp": now,
        "location": {
            "codigo_ibge": request.codigo_ibge,
            "municipio": location["municipio"],
            "uf": uf,
        },
        "area_ha": request.area_ha,
        "production_system": request.production_system,
        "profile": request.profile,
        "temporal_context": temporal_context,
        "recommendations": [
            {
                "crop": "soja",
                "eligible": True,
                "zarc": zarc,
                "climate": climate,
                "temporal_context": temporal_context,
                "yield": yield_data,
                "cost": cost,
                "price": price,
                "economics": economics.model_dump(mode="json"),
                "risk": risk.model_dump(mode="json"),
                "cycle": {
                    "cycle_days": cycle_days,
                    "days_to_first_revenue": cycle_days,
                    "harvests_per_year": math.floor(365 / cycle_days),
                    "type": "estimated",
                    "method": "midpoint of versioned crop taxonomy cycle range",
                },
                "ranking_components": components.model_dump(mode="json", by_alias=True),
                "score": score,
                "confidence_score": confidence,
                "confidence_label": _label(confidence),
                "reasons": reasons,
                "alerts": alerts,
            }
        ],
        "audit": audit,
    }
    analysis_path = resolved.paths.gold / "analyses" / f"{analysis_id}.json"
    write_json_atomic(analysis_path, response)
    return response


def load_analysis(analysis_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or load_settings()
    try:
        normalized = str(UUID(analysis_id))
    except ValueError as exc:
        raise KeyError("analysis_id invalido") from exc
    path = resolved.paths.gold / "analyses" / f"{normalized}.json"
    if not path.exists():
        raise KeyError(f"Analise nao encontrada: {analysis_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stored analysis payload is invalid")
    return payload
