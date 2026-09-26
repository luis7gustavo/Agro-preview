from agri_decision.ranking import RankingComponents, min_max, weighted_score
from agri_decision.ranking.config import RankingWeights


def test_min_max_handles_direction_and_constant_values() -> None:
    assert min_max([10, 20, 30]) == [0, 50, 100]
    assert min_max([10, 20, 30], higher_is_better=False) == [100, 50, 0]
    assert min_max([5, 5]) == [50, 50]


def test_weighted_score_uses_external_weights() -> None:
    components = RankingComponents(
        **{"return": 100, "risk": 80, "agronomic": 60, "speed": 40, "confidence": 20}
    )
    weights = RankingWeights(
        **{"return": 0.45, "risk": 0.25, "agronomic": 0.15, "speed": 0.10, "confidence": 0.05}
    )
    assert weighted_score(components, weights) == 79.0

