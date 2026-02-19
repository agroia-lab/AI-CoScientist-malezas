"""Tests for T2-A: ensemble review aggregation.

Covers _average_review_scores helper, ensemble pass count,
partial failures, and total failure fallback.
"""

import json
from unittest.mock import MagicMock, patch

from ai_coscientist.main import AIScientistFramework
from ai_coscientist.types import Hypothesis

# ---- helpers -------------------------------------------------------


def _review(score, ss=4, nov=3):
    """Build a minimal review dict."""
    return {
        "overall_score": score,
        "review_summary": "ok",
        "scores": {
            "scientific_soundness": ss,
            "novelty": nov,
            "relevance": 4,
            "testability": 4,
            "clarity": 4,
            "potential_impact": 3,
            "statistical_rigor": 3,
            "field_feasibility": 3,
            "spatial_scalability": 3,
            "environmental_sustainability": 3,
            "agronomic_practicality": 3,
        },
        "safety_ethical_concerns": "None",
        "detailed_feedback": {},
        "constructive_feedback": "ok",
    }


# ---- _average_review_scores unit tests ----------------------------


def test_average_review_scores_single():
    """Single review returns itself."""
    r = _review(0.8)
    avg = AIScientistFramework._average_review_scores([r])
    assert avg["overall_score"] == 0.8
    assert avg["scores"]["scientific_soundness"] == 4


def test_average_review_scores_multiple():
    """Three reviews with different scores averaged."""
    r1 = _review(0.6, ss=3, nov=2)
    r2 = _review(0.8, ss=5, nov=4)
    r3 = _review(0.7, ss=4, nov=3)
    avg = AIScientistFramework._average_review_scores([r1, r2, r3])
    assert abs(avg["overall_score"] - 0.7) < 0.01
    assert avg["scores"]["scientific_soundness"] == 4
    assert avg["scores"]["novelty"] == 3


def test_average_review_scores_empty():
    """Empty list returns empty dict."""
    assert AIScientistFramework._average_review_scores([]) == {}


def test_average_review_scores_no_scores_key():
    """Reviews without 'scores' key still average overall_score."""
    r1 = {"overall_score": 0.6, "review_summary": "ok"}
    r2 = {"overall_score": 0.8, "review_summary": "ok"}
    avg = AIScientistFramework._average_review_scores([r1, r2])
    assert abs(avg["overall_score"] - 0.7) < 0.01


def test_average_review_scores_text_from_first():
    """Text fields come from the first review."""
    r1 = _review(0.6)
    r1["review_summary"] = "first"
    r2 = _review(0.8)
    r2["review_summary"] = "second"
    avg = AIScientistFramework._average_review_scores([r1, r2])
    assert avg["review_summary"] == "first"


# ---- Ensemble integration tests -----------------------------------


def _make_framework(ensemble_count=3):
    """Build a framework with mocked agents."""
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
            hypotheses_per_generation=3,
            ensemble_review_count=ensemble_count,
        )
        agent_map = {a.agent_name: a for a in agents}
        return fw, agent_map


def test_ensemble_count_parameter():
    """Reflection agent is called N times per hypothesis."""
    fw, agents = _make_framework(ensemble_count=5)
    review = json.dumps(_review(0.7))
    agents["HypothesisReflector"].run.return_value = review
    agents["AdversarialReflector"].run.return_value = review

    h = Hypothesis(text="Test h")
    result = fw._run_reflection_phase([h])

    assert len(result) == 1
    assert agents["HypothesisReflector"].run.call_count == 5
    # Adversarial is always 1 call
    assert agents["AdversarialReflector"].run.call_count == 1


def test_ensemble_averages_scores():
    """Ensemble of 3 reviews produces averaged score."""
    fw, agents = _make_framework(ensemble_count=3)
    responses = [
        json.dumps(_review(0.6)),
        json.dumps(_review(0.8)),
        json.dumps(_review(0.7)),
    ]
    agents["HypothesisReflector"].run.side_effect = responses
    agents["AdversarialReflector"].run.return_value = json.dumps(
        _review(0.4)
    )

    h = Hypothesis(text="Test h")
    result = fw._run_reflection_phase([h])

    assert len(result) == 1
    # opt avg = 0.7, adv = 0.4 → overall = 0.55
    assert abs(result[0].score - 0.55) < 0.05


def test_ensemble_partial_failure():
    """1 of 3 passes fails; average of 2 used."""
    fw, agents = _make_framework(ensemble_count=3)
    responses = [
        json.dumps(_review(0.6)),
        "",  # empty / failed
        json.dumps(_review(0.8)),
    ]
    agents["HypothesisReflector"].run.side_effect = responses
    agents["AdversarialReflector"].run.return_value = json.dumps(
        _review(0.5)
    )

    h = Hypothesis(text="Test h")
    result = fw._run_reflection_phase([h])

    assert len(result) == 1
    # opt avg = (0.6+0.8)/2 = 0.7, adv = 0.5 → 0.6
    assert abs(result[0].score - 0.6) < 0.05


def test_ensemble_all_fail():
    """All ensemble passes fail → score=0.0."""
    fw, agents = _make_framework(ensemble_count=3)
    agents["HypothesisReflector"].run.return_value = ""
    agents["AdversarialReflector"].run.return_value = json.dumps(
        _review(0.5)
    )

    h = Hypothesis(text="Test h")
    result = fw._run_reflection_phase([h])

    assert len(result) == 1
    assert result[0].score == 0.0


def test_ensemble_review_count_default():
    """Default ensemble_review_count is 3."""
    with patch("ai_coscientist.main.DirectLLMAgent"):
        fw = AIScientistFramework(model_name="test")
        assert fw.ensemble_review_count == 3


def test_ensemble_review_count_validation():
    """Invalid ensemble_review_count raises ValueError."""
    import pytest

    with patch("ai_coscientist.main.DirectLLMAgent"):
        with pytest.raises(ValueError):
            AIScientistFramework(
                model_name="test",
                ensemble_review_count=0,
            )
        with pytest.raises(ValueError):
            AIScientistFramework(
                model_name="test",
                ensemble_review_count=-1,
            )
