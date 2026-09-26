from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl

from agri_decision.config import Settings, load_settings
from agri_decision.features.training import (
    MODEL_FEATURES,
    build_soy_yield_inference_frame,
)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def predict_soy_yield(
    codigo_ibge: str,
    *,
    history: pl.DataFrame,
    climate: pl.DataFrame,
    locations: pl.DataFrame,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    resolved = settings or load_settings()
    pointer_path = resolved.paths.models / "yield" / "soja" / "latest.json"
    if not pointer_path.exists():
        return None
    pointer = _read_mapping(pointer_path)
    if pointer.get("promoted") is not True:
        return None
    model_path = resolved.project_root / str(pointer["artifact"])
    metadata_path = resolved.project_root / str(pointer["metadata"])
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Promoted yield model pointer references missing artifacts")
    metadata = _read_mapping(metadata_path)
    local_years = history.filter(
        (pl.col("codigo_ibge") == codigo_ibge)
        & (pl.col("cultura") == "soja")
        & pl.col("produtividade_kg_ha").is_not_null()
    ).get_column("ano")
    latest_year: Any = local_years.max()
    if latest_year is None:
        return None
    target_year = int(latest_year) + 1
    inference = build_soy_yield_inference_frame(
        history,
        climate,
        locations,
        codigo_ibge=codigo_ibge,
        target_year=target_year,
    )
    model = joblib.load(model_path)
    features = inference.select(MODEL_FEATURES).to_numpy().astype(np.float64)
    p50 = max(0.0, float(np.asarray(model.predict(features), dtype=float)[0]))
    interval = metadata["production_interval"]
    radius = float(interval["radius_kg_ha"])
    sample_size = int(local_years.len())
    climate_source = inference.item(0, "climate_lag1_source")
    climate_nature = inference.item(0, "climate_lag1_data_nature")
    climate_available = inference.item(0, "climate_lag1_rain_crop_cycle_mm") is not None
    training_climate_sources = metadata.get("dataset", {}).get("climate_sources", [])
    source_not_in_training = (
        climate_available
        and climate_source is not None
        and "ERA5" in str(climate_source)
        and climate_source not in training_climate_sources
    )
    return {
        "p10": max(0.0, p50 - radius),
        "p50": p50,
        "p90": p50 + radius,
        "unit": "kg/ha",
        "type": "predicted",
        "method": "point model with calibrated split-conformal absolute residual interval",
        "source": (
            f"IBGE/PAM history and lagged {climate_source} climate features"
            if climate_available
            else "IBGE/PAM history; unavailable climate features imputed"
        ),
        "reference_period": str(target_year),
        "target_year": target_year,
        "sample_size": sample_size,
        "model": metadata["model"],
        "model_version": metadata["model_version"],
        "dataset_version": metadata["dataset"]["dataset_version"],
        "interval_method": interval["method"],
        "interval_nominal_coverage": interval["coverage"],
        "interval_radius_kg_ha": radius,
        "interval_empirical_coverage": metadata["interval_evaluation"]["empirical_coverage"],
        "performance_confidence": metadata["performance_confidence"],
        "performance": metadata["performance"],
        "alerts": ["CLIMATE_SOURCE_NOT_IN_MODEL_TRAINING"] if source_not_in_training else [],
        "training_climate_sources": training_climate_sources,
        "features_reference": {
            "history_through_year": target_year - 1,
            "climate_season": target_year - 1,
            "climate_source": climate_source,
            "climate_data_nature": climate_nature,
            "climate_method": inference.item(0, "climate_lag1_method"),
            "climate_quality_flag": inference.item(0, "climate_lag1_quality_flag"),
            "climate_available": climate_available,
        },
    }
