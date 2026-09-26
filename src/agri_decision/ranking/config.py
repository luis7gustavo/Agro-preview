from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agri_decision.config import find_project_root, load_yaml


class RankingWeights(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    return_weight: float = Field(alias="return", ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    agronomic: float = Field(ge=0, le=1)
    speed: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> RankingWeights:
        total = self.return_weight + self.risk + self.agronomic + self.speed + self.confidence
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Ranking weights must sum to 1.0, got {total}")
        return self


class RankingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_version: str = Field(min_length=1)
    profiles: dict[str, RankingWeights]


def load_ranking_config(path: Path | None = None) -> RankingConfig:
    source = path or find_project_root() / "configs" / "ranking.yml"
    return RankingConfig.model_validate(load_yaml(source))
