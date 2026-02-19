"""Tests for T2-C: multi-criterion tournament scoring.

Covers dimension Elo defaults, update_dimension_elos,
composite Elo, to_dict, tournament_weights validation,
legacy fallback, and full phase integration.
"""

import json
from unittest.mock import MagicMock, patch

from ai_coscientist.types import Hypothesis
from ai_coscientist.main import AIScientistFramework

DEFAULT_WEIGHTS = {
    "scientific_merit": 0.25,
    "practical_value": 0.35,
    "impact": 0.25,
    "communication": 0.15,
}


# ---- Hypothesis dimension Elo unit tests -------------------------


def test_dimension_elo_defaults():
    """All 4 dimension Elos start at 1200."""
    h = Hypothesis(text="T")
    assert h.elo_scientific == 1200
    assert h.elo_practical == 1200
    assert h.elo_impact == 1200
    assert h.elo_communication == 1200


def test_update_dimension_elos_winner():
    """h_a higher on all dims → all elo_* increase."""
    h_a = Hypothesis(text="A")
    h_b = Hypothesis(text="B")
    dim = {
        "scientific_merit": {"h_a": 9, "h_b": 3},
        "practical_value": {"h_a": 8, "h_b": 4},
        "impact": {"h_a": 7, "h_b": 5},
        "communication": {"h_a": 8, "h_b": 4},
    }
    h_a.update_dimension_elos(h_b, dim, DEFAULT_WEIGHTS, is_h_a=True)
    assert h_a.elo_scientific > 1200
    assert h_a.elo_practical > 1200
    assert h_a.elo_impact > 1200
    assert h_a.elo_communication > 1200


def test_update_dimension_elos_mixed():
    """h_a wins some, h_b wins others."""
    h_a = Hypothesis(text="A")
    h_b = Hypothesis(text="B")
    dim = {
        "scientific_merit": {"h_a": 9, "h_b": 3},
        "practical_value": {"h_a": 3, "h_b": 9},
        "impact": {"h_a": 7, "h_b": 5},
        "communication": {"h_a": 4, "h_b": 8},
    }
    h_a.update_dimension_elos(h_b, dim, DEFAULT_WEIGHTS, is_h_a=True)
    assert h_a.elo_scientific > 1200
    assert h_a.elo_practical < 1200
    assert h_a.elo_impact > 1200
    assert h_a.elo_communication < 1200


def test_update_dimension_elos_tie():
    """Tied dimension → no change for that dimension."""
    h_a = Hypothesis(text="A")
    h_b = Hypothesis(text="B")
    dim = {
        "scientific_merit": {"h_a": 7, "h_b": 7},
        "practical_value": {"h_a": 7, "h_b": 7},
        "impact": {"h_a": 7, "h_b": 7},
        "communication": {"h_a": 7, "h_b": 7},
    }
    h_a.update_dimension_elos(h_b, dim, DEFAULT_WEIGHTS, is_h_a=True)
    assert h_a.elo_scientific == 1200
    assert h_a.elo_practical == 1200
    assert h_a.elo_impact == 1200
    assert h_a.elo_communication == 1200
    assert h_a.elo_rating == 1200


def test_composite_elo_is_weighted_sum():
    """Composite elo_rating = weighted sum of dimension Elos."""
    h_a = Hypothesis(text="A")
    h_b = Hypothesis(text="B")
    dim = {
        "scientific_merit": {"h_a": 10, "h_b": 1},
        "practical_value": {"h_a": 10, "h_b": 1},
        "impact": {"h_a": 10, "h_b": 1},
        "communication": {"h_a": 10, "h_b": 1},
    }
    h_a.update_dimension_elos(h_b, dim, DEFAULT_WEIGHTS, is_h_a=True)
    expected = round(
        h_a.elo_scientific * 0.25
        + h_a.elo_practical * 0.35
        + h_a.elo_impact * 0.25
        + h_a.elo_communication * 0.15
    )
    assert h_a.elo_rating == expected


def test_to_dict_includes_dimension_elos():
    """to_dict includes all 4 dimension Elo keys."""
    h = Hypothesis(text="T")
    d = h.to_dict()
    assert d["elo_scientific"] == 1200
    assert d["elo_practical"] == 1200
    assert d["elo_impact"] == 1200
    assert d["elo_communication"] == 1200


# ---- Framework constructor tests ---------------------------------


def test_tournament_weights_default():
    """Default weights match expected values."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        fw = AIScientistFramework(model_name="test")
        assert fw.tournament_weights == DEFAULT_WEIGHTS


def test_tournament_weights_custom():
    """Custom weights are stored and used."""
    custom = {
        "scientific_merit": 0.4,
        "practical_value": 0.3,
        "impact": 0.2,
        "communication": 0.1,
    }
    with patch("ai_coscientist.main.DirectLLMAgent"):
        fw = AIScientistFramework(
            model_name="test",
            tournament_weights=custom,
        )
        assert fw.tournament_weights == custom


def test_tournament_weights_validation():
    """Rejects weights that don't sum to ~1.0."""
    import pytest

    bad = {
        "scientific_merit": 0.5,
        "practical_value": 0.5,
        "impact": 0.5,
        "communication": 0.5,
    }
    with patch("ai_coscientist.main.DirectLLMAgent"):
        with pytest.raises(ValueError, match="sum to"):
            AIScientistFramework(
                model_name="test",
                tournament_weights=bad,
            )


# ---- Tournament phase tests --------------------------------------


def _build_tournament_fw():
    """Build framework with mocked agents for tournament tests."""
    with patch("ai_coscientist.main.DirectLLMAgent") as MockCls:
        agents = []

        def _make(**kw):
            a = MagicMock()
            a.agent_name = kw.get("agent_name", "Unknown")
            a.run = MagicMock(return_value="{}")
            agents.append(a)
            return a

        MockCls.side_effect = _make
        fw = AIScientistFramework(
            model_name="test",
            max_iterations=1,
            tournament_mode="round_robin",
        )
        agent_map = {a.agent_name: a for a in agents}
        return fw, agent_map


def test_fallback_to_legacy_winner():
    """Old format with "winner" still works."""
    fw, agents = _build_tournament_fw()
    legacy_resp = json.dumps(
        {
            "winner": "a",
            "judgment_explanation": {},
            "decision_summary": "A wins",
            "confidence_level": "High",
        }
    )
    agents["TournamentJudge"].run.return_value = legacy_resp

    h1 = Hypothesis(text="H1")
    h2 = Hypothesis(text="H2")
    result = fw._run_tournament_phase([h1, h2])

    assert len(result) == 2
    assert result[0].win_count + result[1].win_count >= 1


def test_dimension_scores_update_in_phase():
    """Full phase with dimension scores updates dimension Elos."""
    fw, agents = _build_tournament_fw()
    dim_resp = json.dumps(
        {
            "dimension_scores": {
                "scientific_merit": {
                    "h_a": 8,
                    "h_b": 4,
                },
                "practical_value": {
                    "h_a": 7,
                    "h_b": 5,
                },
                "impact": {"h_a": 6, "h_b": 6},
                "communication": {
                    "h_a": 9,
                    "h_b": 3,
                },
            },
            "judgment_explanation": {},
            "decision_summary": "A wins overall",
            "confidence_level": "High",
        }
    )
    agents["TournamentJudge"].run.return_value = dim_resp

    h1 = Hypothesis(text="H1")
    h2 = Hypothesis(text="H2")
    result = fw._run_tournament_phase([h1, h2])

    assert len(result) == 2
    # h1 should have won (higher weighted scores)
    winner = result[0]
    assert winner.win_count == 1
    # Dimension Elos should have changed
    assert (
        winner.elo_scientific != 1200 or winner.elo_practical != 1200
    )
