from pathlib import Path

from agri_decision.ranking import load_ranking_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_balanced_weights_match_product_specification() -> None:
    config = load_ranking_config(PROJECT_ROOT / "configs" / "ranking.yml")
    balanced = config.profiles["balanced"]

    assert balanced.return_weight == 0.45
    assert balanced.risk == 0.25
    assert balanced.agronomic == 0.15
    assert balanced.speed == 0.10
    assert balanced.confidence == 0.05


def test_all_ranking_profiles_are_normalized() -> None:
    config = load_ranking_config(PROJECT_ROOT / "configs" / "ranking.yml")

    for weights in config.profiles.values():
        total = (
            weights.return_weight
            + weights.risk
            + weights.agronomic
            + weights.speed
            + weights.confidence
        )
        assert total == 1.0
