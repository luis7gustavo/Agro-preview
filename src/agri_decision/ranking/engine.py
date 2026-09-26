from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agri_decision.ranking.config import RankingWeights


class RankingComponents(BaseModel):
    """Direction-correct component scores, each already normalized to 0..100."""

    model_config = ConfigDict(frozen=True)

    return_score: float = Field(alias="return", ge=0, le=100)
    risk: float = Field(ge=0, le=100, description="100 means lowest risk")
    agronomic: float = Field(ge=0, le=100)
    speed: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)


def weighted_score(components: RankingComponents, weights: RankingWeights) -> float:
    score = (
        components.return_score * weights.return_weight
        + components.risk * weights.risk
        + components.agronomic * weights.agronomic
        + components.speed * weights.speed
        + components.confidence * weights.confidence
    )
    return round(score, 6)


def min_max(values: list[float], *, higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return [50.0] * len(values)
    normalized = [(value - minimum) / (maximum - minimum) * 100 for value in values]
    if not higher_is_better:
        normalized = [100 - value for value in normalized]
    return normalized
