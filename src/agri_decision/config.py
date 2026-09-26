from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    bronze: Path
    silver: Path
    gold: Path
    external: Path
    models: Path
    reports: Path


class LoggingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: str = "INFO"
    format: str = "json"


class QualitySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    fail_on_duplicate_keys: bool = True
    preserve_rejected_rows: bool = True
    require_complete_provenance_in_gold: bool = True


class RiskSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    simulations: int = Field(default=10_000, ge=100)
    seed: int = 20_260_820


class IngestionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_concurrent_requests: int = Field(default=3, ge=1, le=8)


class ClimateSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_year: int = Field(default=2017, ge=2000, le=2200)
    end_year: int = Field(default=2025, ge=2000, le=2200)
    max_station_distance_km: float = Field(default=250.0, gt=0, le=1_000)
    minimum_season_coverage: float = Field(default=0.70, ge=0, le=1)

    @model_validator(mode="after")
    def valid_year_range(self) -> ClimateSettings:
        if self.end_year < self.start_year:
            raise ValueError("climate end_year must be >= start_year")
        return self


class MlSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    training_start_year: int = Field(default=2019, ge=1974, le=2200)
    training_end_year: int = Field(default=2024, ge=1974, le=2200)
    temporal_test_start_year: int = Field(default=2023, ge=1974, le=2200)
    geographic_holdout_fraction: float = Field(default=0.20, gt=0, lt=0.5)
    random_seed: int = 20_260_820
    prediction_interval_coverage: float = Field(default=0.80, gt=0.5, lt=1)

    @model_validator(mode="after")
    def valid_ranges(self) -> MlSettings:
        if self.training_end_year < self.training_start_year:
            raise ValueError("ML training_end_year must be >= training_start_year")
        if not self.training_start_year < self.temporal_test_start_year <= self.training_end_year:
            raise ValueError("temporal_test_start_year must be inside the training range")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_root: Path
    environment: str = "development"
    config_version: str
    pipeline_version: str
    paths: DataPaths
    logging: LoggingSettings
    quality: QualitySettings
    risk: RiskSettings
    ingestion: IngestionSettings
    climate: ClimateSettings
    ml: MlSettings

    @model_validator(mode="after")
    def paths_must_stay_inside_project(self) -> Settings:
        root = self.project_root.resolve()
        for name, path in self.paths:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Configured path {name} escapes project root: {resolved}")
        return self


def find_project_root(start: Path | None = None) -> Path:
    override = os.getenv("AGRI_PROJECT_ROOT")
    candidate = Path(override) if override else (start or Path.cwd())
    candidate = candidate.expanduser().resolve()

    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
            return directory
    raise FileNotFoundError(
        "Project root not found. Set AGRI_PROJECT_ROOT or run inside the repository."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_settings(project_root: Path | None = None) -> Settings:
    root = find_project_root(project_root)
    payload = load_yaml(root / "configs" / "settings.yml")
    raw_paths = payload.get("paths", {})
    resolved_paths = {
        name: (root / value).resolve()
        for name, value in raw_paths.items()
        if isinstance(value, str)
    }
    payload["paths"] = resolved_paths
    payload["project_root"] = root
    payload["environment"] = os.getenv("AGRI_ENVIRONMENT", "development")
    payload.setdefault("logging", {})["level"] = os.getenv(
        "AGRI_LOG_LEVEL", payload.get("logging", {}).get("level", "INFO")
    )
    return Settings.model_validate(payload)
