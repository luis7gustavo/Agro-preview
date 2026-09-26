from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TriangularRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    low: float = Field(ge=0)
    mode: float = Field(ge=0)
    high: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> TriangularRange:
        if not self.low <= self.mode <= self.high:
            raise ValueError("triangular range must satisfy low <= mode <= high")
        return self


class MonteCarloInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    yield_kg_ha: TriangularRange
    price_brl_kg: TriangularRange
    cost_brl_ha: TriangularRange
    simulations: int = Field(default=10_000, ge=100)
    seed: int = 20_260_820


class RiskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    simulations: int
    seed: int
    probability_profit: float
    probability_loss: float
    profit_p10_brl_ha: float
    profit_p50_brl_ha: float
    profit_p90_brl_ha: float
    expected_profit_brl_ha: float
    expected_downside_brl_ha: float


def _sample(rng: np.random.Generator, interval: TriangularRange, size: int) -> np.ndarray:
    if interval.low == interval.high:
        return np.full(size, interval.low, dtype=float)
    return rng.triangular(interval.low, interval.mode, interval.high, size=size)


def simulate_profit(inputs: MonteCarloInputs) -> RiskResult:
    rng = np.random.default_rng(inputs.seed)
    yields = _sample(rng, inputs.yield_kg_ha, inputs.simulations)
    prices = _sample(rng, inputs.price_brl_kg, inputs.simulations)
    costs = _sample(rng, inputs.cost_brl_ha, inputs.simulations)
    profits = yields * prices - costs
    loss = profits < 0
    downside = np.minimum(profits, 0.0)
    p10, p50, p90 = np.quantile(profits, [0.1, 0.5, 0.9])
    return RiskResult(
        simulations=inputs.simulations,
        seed=inputs.seed,
        probability_profit=float(np.mean(profits > 0)),
        probability_loss=float(np.mean(loss)),
        profit_p10_brl_ha=float(p10),
        profit_p50_brl_ha=float(p50),
        profit_p90_brl_ha=float(p90),
        expected_profit_brl_ha=float(np.mean(profits)),
        expected_downside_brl_ha=float(np.mean(downside)),
    )
