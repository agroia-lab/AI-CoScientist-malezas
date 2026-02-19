"""Tests for T2-B: directed regeneration from meta-review gaps.

Covers gap detection, hypothesis generation, dedup,
self-score filter, and empty/failure cases.
"""

import json
from unittest.mock import MagicMock, patch

from ai_coscientist.main import AIScientistFramework
from ai_coscientist.types import Hypothesis

# ---- helpers -------------------------------------------------------


def _fw():
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
            hypotheses_per_generation=10,
        )
        agent_map = {a.agent_name: a for a in agents}
        return fw, agent_map


# ---- Tests ---------------------------------------------------------


def test_produces_hypotheses_from_gaps():
    """Mocked generation returns 2 gap-filling hypotheses."""
    fw, agents = _fw()
    agents["HypothesisGenerator"].run.return_value = json.dumps(
        {
            "hypotheses": [
                {"text": "New H1 from gap"},
                {"text": "New H2 from gap"},
            ]
        }
    )
    meta = {
        "weaknesses": ["Missing biological control"],
        "strategic_recommendations": [
            {
                "focus_area": "biocontrol",
                "recommendation": "Add biocontrol hypothesis",
            }
        ],
    }
    existing = [Hypothesis(text="Existing H")]

    result = fw._run_directed_regeneration(
        meta, existing, "Test goal"
    )
    assert len(result) == 2
    assert result[0].text == "New H1 from gap"


def test_returns_empty_when_no_gaps():
    """Empty weaknesses + recommendations → []."""
    fw, agents = _fw()
    meta = {"weaknesses": [], "strategic_recommendations": []}
    result = fw._run_directed_regeneration(meta, [], "Test goal")
    assert result == []
    agents["HypothesisGenerator"].run.assert_not_called()


def test_returns_empty_on_empty_meta_review():
    """{} → []."""
    fw, _ = _fw()
    result = fw._run_directed_regeneration({}, [], "Test goal")
    assert result == []


def test_returns_empty_on_non_dict():
    """Non-dict meta_review_data → []."""
    fw, _ = _fw()
    result = fw._run_directed_regeneration(
        "not a dict", [], "Test goal"
    )
    assert result == []


def test_deduplicates_against_existing():
    """Duplicate text is filtered."""
    fw, agents = _fw()
    agents["HypothesisGenerator"].run.return_value = json.dumps(
        {
            "hypotheses": [
                {"text": "Already exists"},
                {"text": "Brand new"},
            ]
        }
    )
    meta = {"weaknesses": ["gap"]}
    existing = [Hypothesis(text="Already exists")]

    result = fw._run_directed_regeneration(
        meta, existing, "Test goal"
    )
    assert len(result) == 1
    assert result[0].text == "Brand new"


def test_caps_count_at_half_generation_size():
    """8 recs with N=10 → asks for 5."""
    fw, agents = _fw()
    fw.hypotheses_per_generation = 10
    recs = [
        {
            "focus_area": f"area{i}",
            "recommendation": f"rec{i}",
        }
        for i in range(8)
    ]
    agents["HypothesisGenerator"].run.return_value = json.dumps(
        {"hypotheses": [{"text": f"H{i}"} for i in range(5)]}
    )
    meta = {"strategic_recommendations": recs}

    fw._run_directed_regeneration(meta, [], "Test goal")
    # The prompt should ask for exactly 5
    call_args = agents["HypothesisGenerator"].run.call_args
    assert "5" in call_args[0][0]


def test_handles_agent_empty_response():
    """Agent returns "" → []."""
    fw, agents = _fw()
    agents["HypothesisGenerator"].run.return_value = ""
    meta = {"weaknesses": ["gap"]}

    result = fw._run_directed_regeneration(meta, [], "Test goal")
    assert result == []


def test_applies_self_score_filter():
    """Low self-scores are filtered."""
    fw, agents = _fw()
    agents["HypothesisGenerator"].run.return_value = json.dumps(
        {
            "hypotheses": [
                {
                    "text": "Good h",
                    "self_scores": {
                        "interestingness": 7,
                        "field_feasibility": 8,
                        "testability": 7,
                        "parsimony": 6,
                    },
                },
                {
                    "text": "Bad h",
                    "self_scores": {
                        "interestingness": 2,
                        "field_feasibility": 3,
                        "testability": 2,
                        "parsimony": 1,
                    },
                },
            ]
        }
    )
    meta = {"weaknesses": ["gap"]}

    result = fw._run_directed_regeneration(meta, [], "Test goal")
    assert len(result) == 1
    assert result[0].text == "Good h"


def test_workflow_integrates_directed_regeneration():
    """End-to-end: directed regen adds hypotheses in iteration loop."""
    from tests.test_workflow import (
        GENERATION_RESPONSE,
        REFLECTION_RESPONSE,
        RANKING_RESPONSE,
        TOURNAMENT_RESPONSE,
        EVOLUTION_RESPONSE,
        PROXIMITY_RESPONSE,
        SUPERVISOR_RESPONSE,
    )

    # Meta-review with gaps
    meta_with_gaps = json.dumps(
        {
            "meta_review_summary": "Gaps found",
            "recurring_themes": [],
            "strengths": [],
            "weaknesses": ["No biocontrol hypothesis"],
            "process_assessment": {
                "generation_process": "ok",
                "review_process": "ok",
                "evolution_process": "ok",
            },
            "strategic_recommendations": [
                {
                    "focus_area": "biocontrol",
                    "recommendation": "Add one",
                    "justification": "Missing pillar",
                }
            ],
            "potential_connections": [],
        }
    )

    regen_response = json.dumps(
        {"hypotheses": [{"text": "Gap-fill: biocontrol hypothesis"}]}
    )

    with patch("ai_coscientist.main.DirectLLMAgent") as MockCls:
        agents_created = []

        def _make(**kw):
            a = MagicMock()
            a.agent_name = kw.get("agent_name", "Unknown")
            a.run = MagicMock(return_value="{}")
            agents_created.append(a)
            return a

        MockCls.side_effect = _make
        fw = AIScientistFramework(
            model_name="test",
            max_iterations=1,
            hypotheses_per_generation=3,
            tournament_size=4,
            evolution_top_k=2,
        )
        am = {a.agent_name: a for a in agents_created}

        am["Supervisor"].run.return_value = SUPERVISOR_RESPONSE
        # Generation agent: first call = normal gen,
        # second call = directed regen
        am["HypothesisGenerator"].run.side_effect = [
            GENERATION_RESPONSE,  # initial generation
            GENERATION_RESPONSE,  # fallback (if needed)
            regen_response,  # directed regen
        ]
        am["HypothesisReflector"].run.return_value = (
            REFLECTION_RESPONSE
        )
        am["AdversarialReflector"].run.return_value = (
            REFLECTION_RESPONSE
        )
        am["HypothesisRanker"].run.return_value = RANKING_RESPONSE
        am["TournamentJudge"].run.return_value = TOURNAMENT_RESPONSE
        am["HypothesisEvolver"].run.return_value = EVOLUTION_RESPONSE
        am["MetaReviewer"].run.return_value = meta_with_gaps
        am["ProximityAnalyzer"].run.return_value = PROXIMITY_RESPONSE

        result = fw.run_research_workflow("Test research goal for AI")
        assert isinstance(result, dict)
        assert len(result["top_ranked_hypotheses"]) > 0
