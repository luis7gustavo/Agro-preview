from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EconomicInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_yield_kg_ha: float = Field(gt=0)
    price_brl_kg: float = Field(gt=0)
    operational_cost_brl_ha: float = Field(ge=0)
    total_cost_brl_ha: float = Field(gt=0)
    area_ha: float | None = Field(default=None, gt=0)
    days_to_first_revenue: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def total_cannot_be_below_operational(self) -> EconomicInputs:
        if self.total_cost_brl_ha < self.operational_cost_brl_ha:
            raise ValueError("total cost cannot be lower than operational cost")
        return self


class EconomicResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    revenue_brl_ha: float
    operating_margin_brl_ha: float
    economic_profit_brl_ha: float
    roi: float
    break_even_price_brl_kg: float
    break_even_yield_kg_ha: float
    revenue_total_brl: float | None
    profit_total_brl: float | None
    capital_required_brl: float | None
    profit_per_month_brl: float | None


def calculate_economics(inputs: EconomicInputs) -> EconomicResult:
    revenue = inputs.expected_yield_kg_ha * inputs.price_brl_kg
    operating_margin = revenue - inputs.operational_cost_brl_ha
    profit = revenue - inputs.total_cost_brl_ha
    area = inputs.area_ha
    revenue_total = revenue * area if area is not None else None
    profit_total = profit * area if area is not None else None
    capital = inputs.total_cost_brl_ha * area if area is not None else None
    months = (
        inputs.days_to_first_revenue / (365.25 / 12)
        if inputs.days_to_first_revenue is not None
        else None
    )
    profit_per_month = (
        profit_total / months
        if profit_total is not None and months is not None and months > 0
        else None
    )
    return EconomicResult(
        revenue_brl_ha=revenue,
        operating_margin_brl_ha=operating_margin,
        economic_profit_brl_ha=profit,
        roi=profit / inputs.total_cost_brl_ha,
        break_even_price_brl_kg=inputs.total_cost_brl_ha / inputs.expected_yield_kg_ha,
        break_even_yield_kg_ha=inputs.total_cost_brl_ha / inputs.price_brl_kg,
        revenue_total_brl=revenue_total,
        profit_total_brl=profit_total,
        capital_required_brl=capital,
        profit_per_month_brl=profit_per_month,
    )

