from agri_decision.risk import MonteCarloInputs, TriangularRange, simulate_profit


def test_monte_carlo_is_reproducible_and_probabilities_are_consistent() -> None:
    inputs = MonteCarloInputs(
        yield_kg_ha=TriangularRange(low=3_000, mode=4_000, high=4_500),
        price_brl_kg=TriangularRange(low=1.5, mode=2.0, high=2.2),
        cost_brl_ha=TriangularRange(low=4_500, mode=5_000, high=5_800),
        simulations=1_000,
        seed=42,
    )
    first = simulate_profit(inputs)
    second = simulate_profit(inputs)
    assert first == second
    assert first.probability_profit + first.probability_loss <= 1.0
    assert first.profit_p10_brl_ha <= first.profit_p50_brl_ha <= first.profit_p90_brl_ha

