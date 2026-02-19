"""Tests for the Hypothesis dataclass.

Covers Elo updates, win/loss tracking, to_dict serialisation,
and edge cases like zero-division and invalid types.
"""

from unittest.mock import patch


# We import Hypothesis inside the Agent mock context so the module-level
# _check_api_keys() and Agent import don't trip up.
def _make_hypothesis(text="Test hypothesis", **kwargs):
    """Helper to create a Hypothesis with Agent mocked at import time."""
    from ai_coscientist.main import Hypothesis

    return Hypothesis(text=text, **kwargs)


# -- update_elo -----------------------------------------------------------


def test_elo_win_increases_rating():
    """Winning against an equal-rated opponent increases Elo."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        original = h.elo_rating
        h.update_elo(opponent_elo=1200, win=True)
        assert h.elo_rating > original


def test_elo_loss_decreases_rating():
    """Losing against an equal-rated opponent decreases Elo."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        original = h.elo_rating
        h.update_elo(opponent_elo=1200, win=False)
        assert h.elo_rating < original


def test_elo_equal_ratings_win_expected_value():
    """Win against equal opponent (both 1200): expected new rating ~1216."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        h.update_elo(opponent_elo=1200, win=True, k_factor=32)
        # Expected score = 0.5 for equal ratings
        # Delta = 32 * (1.0 - 0.5) = 16
        assert h.elo_rating == 1216


def test_elo_equal_ratings_loss_expected_value():
    """Loss against equal opponent (both 1200): expected new rating ~1184."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        h.update_elo(opponent_elo=1200, win=False, k_factor=32)
        assert h.elo_rating == 1184


def test_elo_win_increments_win_count():
    """Win increments win_count by 1."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        assert h.win_count == 0
        h.update_elo(opponent_elo=1200, win=True)
        assert h.win_count == 1


def test_elo_loss_increments_loss_count():
    """Loss increments loss_count by 1."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        assert h.loss_count == 0
        h.update_elo(opponent_elo=1200, win=False)
        assert h.loss_count == 1


def test_elo_invalid_opponent_type_no_crash():
    """Non-integer opponent_elo does not crash; rating stays unchanged."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        original = h.elo_rating
        h.update_elo(opponent_elo="not_a_number", win=True)
        assert h.elo_rating == original


def test_elo_invalid_win_type_no_crash():
    """Non-bool win flag does not crash; rating stays unchanged."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        original = h.elo_rating
        h.update_elo(opponent_elo=1200, win="yes")
        assert h.elo_rating == original


def test_elo_multiple_matches():
    """Multiple sequential matches accumulate correctly."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        h.update_elo(opponent_elo=1200, win=True)
        h.update_elo(opponent_elo=1200, win=True)
        h.update_elo(opponent_elo=1200, win=False)
        assert h.win_count == 2
        assert h.loss_count == 1


# -- to_dict ---------------------------------------------------------------


def test_to_dict_has_all_expected_keys():
    """to_dict() must contain all required keys."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        d = h.to_dict()
        expected_keys = {
            "text",
            "elo_rating",
            "elo_scientific",
            "elo_practical",
            "elo_impact",
            "elo_communication",
            "score",
            "reviews",
            "similarity_cluster_id",
            "evolution_history",
            "win_count",
            "loss_count",
            "total_matches",
            "win_rate",
        }
        assert expected_keys.issubset(set(d.keys()))


def test_to_dict_zero_matches_no_division_error():
    """to_dict with 0 matches: win_rate is 0, no ZeroDivisionError."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        d = h.to_dict()
        assert d["win_rate"] == 0
        assert d["total_matches"] == 0


def test_to_dict_win_rate_after_matches():
    """After 3 wins 1 loss, win_rate should be 75.0."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        for _ in range(3):
            h.update_elo(opponent_elo=1200, win=True)
        h.update_elo(opponent_elo=1200, win=False)
        d = h.to_dict()
        assert d["win_count"] == 3
        assert d["loss_count"] == 1
        assert d["total_matches"] == 4
        assert d["win_rate"] == 75.0


def test_to_dict_text_matches():
    """to_dict text field matches the hypothesis text."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis(text="My specific hypothesis")
        d = h.to_dict()
        assert d["text"] == "My specific hypothesis"


def test_to_dict_default_elo():
    """Default Elo rating in to_dict is 1200."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        d = h.to_dict()
        assert d["elo_rating"] == 1200


def test_to_dict_default_score():
    """Default score in to_dict is 0.0."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        h = _make_hypothesis()
        d = h.to_dict()
        assert d["score"] == 0.0
