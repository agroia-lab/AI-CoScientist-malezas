"""Tests for ai_coscientist.elo module.

Covers:
- calculate_elo_update pure function
- random_pairs, round_robin_pairs, swiss_pairs helpers
- swiss_rounds count
- validate_tournament_mode
- tournament_mode parameter on AIScientistFramework
"""

import random
from unittest.mock import MagicMock, patch

import pytest

from ai_coscientist.elo import (
    calculate_elo_update,
    random_pairs,
    round_robin_pairs,
    swiss_pairs,
    swiss_rounds,
    validate_tournament_mode,
)


# -- calculate_elo_update -----------------------------------------


def test_elo_win_equal_ratings():
    """Win vs equal opponent: +16 from default k=32."""
    assert calculate_elo_update(1200, 1200, True) == 1216


def test_elo_loss_equal_ratings():
    """Loss vs equal opponent: -16 from default k=32."""
    assert calculate_elo_update(1200, 1200, False) == 1184


def test_elo_win_higher_opponent():
    """Win vs stronger opponent gives larger gain."""
    gain_vs_strong = calculate_elo_update(1200, 1600, True) - 1200
    gain_vs_equal = calculate_elo_update(1200, 1200, True) - 1200
    assert gain_vs_strong > gain_vs_equal


def test_elo_loss_weaker_opponent():
    """Loss vs weaker opponent gives larger penalty."""
    loss_vs_weak = 1200 - calculate_elo_update(1200, 800, False)
    loss_vs_equal = 1200 - calculate_elo_update(1200, 1200, False)
    assert loss_vs_weak > loss_vs_equal


def test_elo_custom_k_factor():
    """Custom k-factor scales the update."""
    delta_k16 = calculate_elo_update(1200, 1200, True, 16)
    delta_k64 = calculate_elo_update(1200, 1200, True, 64)
    assert delta_k16 - 1200 == 8
    assert delta_k64 - 1200 == 32


def test_elo_symmetry():
    """Winner gain + loser loss roughly cancel out."""
    new_winner = calculate_elo_update(1200, 1200, True)
    new_loser = calculate_elo_update(1200, 1200, False)
    assert (new_winner - 1200) + (new_loser - 1200) == 0


def test_elo_returns_int():
    """Result is always int."""
    assert isinstance(calculate_elo_update(1200, 1300, True), int)


# -- hypothesis delegation ----------------------------------------


def test_hypothesis_delegates_to_calculate_elo_update():
    """Hypothesis.update_elo delegates to the pure fn."""
    with patch("ai_coscientist.main.Agent"):
        from ai_coscientist.types import Hypothesis

        h = Hypothesis(text="test")
        h.update_elo(1200, True, k_factor=32)
        assert h.elo_rating == 1216


# -- random_pairs --------------------------------------------------


def test_random_pairs_count():
    """Returns exactly the requested number of rounds."""
    items = list(range(5))
    pairs = random_pairs(items, 10)
    assert len(pairs) == 10


def test_random_pairs_valid_indices():
    """All indices in range, a != b."""
    items = list(range(4))
    for a, b in random_pairs(items, 20):
        assert 0 <= a < 4
        assert 0 <= b < 4
        assert a != b


def test_random_pairs_too_few():
    """< 2 items returns empty list."""
    assert random_pairs([1], 5) == []
    assert random_pairs([], 5) == []


def test_random_pairs_deterministic_with_seed():
    """Same seed produces same pairings."""
    items = list(range(6))
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert random_pairs(items, 10, rng1) == random_pairs(
        items, 10, rng2
    )


# -- round_robin_pairs ---------------------------------------------


def test_round_robin_count():
    """n items produce n*(n-1)/2 pairs."""
    items = list(range(5))
    pairs = round_robin_pairs(items)
    assert len(pairs) == 10  # 5*4/2


def test_round_robin_all_unique():
    """Every pair is unique."""
    items = list(range(4))
    pairs = round_robin_pairs(items)
    assert len(pairs) == len(set(pairs))


def test_round_robin_too_few():
    """< 2 items returns empty list."""
    assert round_robin_pairs([1]) == []


def test_round_robin_covers_all_indices():
    """Every index appears in at least one pairing."""
    items = list(range(5))
    pairs = round_robin_pairs(items)
    seen = set()
    for a, b in pairs:
        seen.add(a)
        seen.add(b)
    assert seen == set(range(5))


# -- swiss_pairs ---------------------------------------------------


def test_swiss_pairs_adjacent_ratings():
    """Pairs adjacent-rated items."""
    items = ["a", "b", "c", "d"]
    ratings = [1500, 1200, 1300, 1100]
    # sorted desc by rating: 0(1500), 2(1300), 1(1200), 3(1100)
    pairs = swiss_pairs(items, ratings)
    assert (0, 2) in pairs
    assert (1, 3) in pairs


def test_swiss_pairs_odd_count():
    """Odd number of items: last gets a bye."""
    items = list(range(5))
    ratings = [r * 100 for r in range(5)]
    pairs = swiss_pairs(items, ratings)
    assert len(pairs) == 2  # 5 items -> 2 pairs + 1 bye


def test_swiss_pairs_too_few():
    """< 2 items returns empty."""
    assert swiss_pairs([1], [1200]) == []


# -- swiss_rounds --------------------------------------------------


def test_swiss_rounds_values():
    """Known values for swiss round counts."""
    assert swiss_rounds(2) == 1
    assert swiss_rounds(4) == 2
    assert swiss_rounds(8) == 3
    assert swiss_rounds(10) == 4
    assert swiss_rounds(1) == 0


# -- validate_tournament_mode -------------------------------------


def test_validate_valid_modes():
    """All valid modes accepted."""
    for mode in ("random", "round_robin", "swiss"):
        assert validate_tournament_mode(mode) == mode


def test_validate_invalid_mode():
    """Invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="tournament_mode"):
        validate_tournament_mode("elimination")


# -- Framework integration ----------------------------------------


def test_framework_accepts_tournament_mode():
    """Framework __init__ accepts tournament_mode param."""
    with patch("ai_coscientist.main.Agent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_name = "MockAgent"
        mock_instance.run = MagicMock(return_value="{}")
        MockAgent.return_value = mock_instance

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(
            model_name="test-model",
            tournament_mode="round_robin",
        )
        assert fw.tournament_mode == "round_robin"


def test_framework_rejects_bad_tournament_mode():
    """Invalid tournament_mode raises ValueError."""
    with patch("ai_coscientist.main.Agent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_name = "MockAgent"
        mock_instance.run = MagicMock(return_value="{}")
        MockAgent.return_value = mock_instance

        from ai_coscientist import AIScientistFramework

        with pytest.raises(ValueError):
            AIScientistFramework(
                model_name="test-model",
                tournament_mode="knockout",
            )


def test_framework_default_tournament_mode():
    """Default tournament_mode is 'random'."""
    with patch("ai_coscientist.main.Agent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_name = "MockAgent"
        mock_instance.run = MagicMock(return_value="{}")
        MockAgent.return_value = mock_instance

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(
            model_name="test-model",
        )
        assert fw.tournament_mode == "random"
