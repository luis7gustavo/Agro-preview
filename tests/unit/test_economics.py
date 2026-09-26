import pytest

from agri_decision.economics import EconomicInputs, calculate_economics


def test_required_economic_example_is_exact() -> None:
    result = calculate_economics(
        EconomicInputs(
            expected_yield_kg_ha=4_000,
            price_brl_kg=2,
            operational_cost_brl_ha=4_500,
            total_cost_brl_ha=5_000,
        )
    )
    assert result.revenue_brl_ha == 8_000
    assert result.economic_profit_brl_ha == 3_000
    assert result.roi == pytest.approx(0.6)
    assert result.break_even_price_brl_kg == pytest.approx(1.25)
    assert result.break_even_yield_kg_ha == pytest.approx(2_500)


def test_property_totals_and_time_are_computed_only_when_given() -> None:
    result = calculate_economics(
        EconomicInputs(
            expected_yield_kg_ha=4_000,
            price_brl_kg=2,
            operational_cost_brl_ha=4_500,
            total_cost_brl_ha=5_000,
            area_ha=10,
            days_to_first_revenue=180,
        )
    )
    assert result.revenue_total_brl == 80_000
    assert result.profit_total_brl == 30_000
    assert result.capital_required_brl == 50_000
    assert result.profit_per_month_brl is not None

