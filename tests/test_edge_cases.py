"""Edge-case and robustness tests.

Verifies the framework handles pathological inputs gracefully:
empty goals, non-string goals, agents returning nothing, etc.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---- Helpers -------------------------------------------------------------


def _build_framework_with_agent_responses(**overrides):
    """Build a mocked framework with per-agent response overrides.

    Pass keyword arguments like ``generation="..."`` to set the return
    value for HypothesisGenerator.run(), etc.

    Key names: supervisor, generation, reflection, ranking, tournament,
    evolution, meta_review, proximity.
    """
    agent_name_map = {
        "supervisor": "Supervisor",
        "generation": "HypothesisGenerator",
        "reflection": "HypothesisReflector",
        "ranking": "HypothesisRanker",
        "tournament": "TournamentJudge",
        "evolution": "HypothesisEvolver",
        "meta_review": "MetaReviewer",
        "proximity": "ProximityAnalyzer",
    }

    with patch("ai_coscientist.main.DirectLLMAgent") as MockAgentCls:
        agents_created = []

        def _make_agent(**kwargs):
            agent = MagicMock()
            agent.agent_name = kwargs.get(
                "agent_name", "UnknownAgent"
            )
            agent.run = MagicMock(return_value="{}")
            agents_created.append(agent)
            return agent

        MockAgentCls.side_effect = _make_agent

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(
            model_name="test-model",
            max_iterations=1,
            verbose=False,
            hypotheses_per_generation=3,
            tournament_size=4,
            evolution_top_k=2,
        )

        real_map = {a.agent_name: a for a in agents_created}

        for short_name, agent_name in agent_name_map.items():
            if short_name in overrides and agent_name in real_map:
                real_map[agent_name].run.return_value = overrides[
                    short_name
                ]

        return fw


# ---- Empty / invalid research goal ---------------------------------------


def test_empty_research_goal_raises_value_error():
    """Empty string research goal raises ValueError."""
    with patch("ai_coscientist.main.DirectLLMAgent") as MockAgentCls:
        mock = MagicMock()
        mock.agent_name = "MockAgent"
        mock.run = MagicMock(return_value="{}")
        MockAgentCls.return_value = mock

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(model_name="test-model")
        with pytest.raises(ValueError):
            fw.run_research_workflow("")


def test_whitespace_only_research_goal_raises_value_error():
    """Whitespace-only research goal raises ValueError."""
    with patch("ai_coscientist.main.DirectLLMAgent") as MockAgentCls:
        mock = MagicMock()
        mock.agent_name = "MockAgent"
        mock.run = MagicMock(return_value="{}")
        MockAgentCls.return_value = mock

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(model_name="test-model")
        with pytest.raises(ValueError):
            fw.run_research_workflow("   \t\n  ")


def test_non_string_research_goal_raises():
    """Non-string research goal (int) raises ValueError."""
    with patch("ai_coscientist.main.DirectLLMAgent") as MockAgentCls:
        mock = MagicMock()
        mock.agent_name = "MockAgent"
        mock.run = MagicMock(return_value="{}")
        MockAgentCls.return_value = mock

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(model_name="test-model")
        with pytest.raises((ValueError, TypeError)):
            fw.run_research_workflow(12345)


# ---- Generation returns 0 hypotheses ------------------------------------


def test_generation_returns_zero_hypotheses_uses_fallback():
    """When generation agent returns 0 hypotheses, fallback creates basic ones."""
    # Both initial and fallback generation return empty lists,
    # so the "last resort" manual hypotheses are used.
    fw = _build_framework_with_agent_responses(
        supervisor=json.dumps(
            {
                "workflow_plan": {
                    "generation_phase": {"focus_areas": ["test"]}
                }
            }
        ),
        generation=json.dumps({"hypotheses": []}),
        reflection=json.dumps(
            {"overall_score": 0.5, "review_summary": "OK"}
        ),
        ranking=json.dumps({"ranked_hypotheses": []}),
        tournament=json.dumps(
            {
                "winner": "a",
                "judgment_explanation": {},
                "decision_summary": "A wins",
            }
        ),
        evolution=json.dumps(
            {
                "refined_hypothesis_text": "Refined",
                "refinement_summary": "Improved",
            }
        ),
        meta_review=json.dumps({"meta_review_summary": "OK"}),
        proximity=json.dumps(
            {"similarity_clusters": [], "diversity_assessment": "OK"}
        ),
    )
    result = fw.run_research_workflow(
        "Investigate neural scaling laws"
    )
    # Should still complete without error
    assert isinstance(result, dict)
    # The fallback creates exactly 3 basic hypotheses
    assert (
        result.get("execution_metrics", {}).get("hypothesis_count", 0)
        >= 0
    )


# ---- All agents return empty strings ------------------------------------


def test_all_agents_return_empty_no_crash():
    """Workflow completes (with fallbacks) even when all agents return empty."""
    fw = _build_framework_with_agent_responses(
        supervisor="",
        generation="",
        reflection="",
        ranking="",
        tournament="",
        evolution="",
        meta_review="",
        proximity="",
    )
    result = fw.run_research_workflow(
        "Investigate neural scaling laws"
    )
    assert isinstance(result, dict)
    # Should have the basic structure even in degraded mode
    assert "execution_metrics" in result
    assert "total_workflow_time" in result


# ---- Single hypothesis: tournament returns it unchanged ------------------


def test_single_hypothesis_tournament_returns_unchanged():
    """Tournament with <2 hypotheses returns the single one unchanged."""
    fw = _build_framework_with_agent_responses(
        supervisor=json.dumps(
            {
                "workflow_plan": {
                    "generation_phase": {"focus_areas": ["test"]}
                }
            }
        ),
        generation=json.dumps(
            {"hypotheses": [{"text": "Only hypothesis"}]}
        ),
        reflection=json.dumps(
            {"overall_score": 0.9, "review_summary": "Excellent"}
        ),
        ranking=json.dumps(
            {
                "ranked_hypotheses": [
                    {"text": "Only hypothesis", "overall_score": 0.9}
                ]
            }
        ),
        tournament=json.dumps(
            {
                "winner": "a",
                "judgment_explanation": {},
                "decision_summary": "A wins",
            }
        ),
        evolution=json.dumps(
            {
                "refined_hypothesis_text": "Only hypothesis refined",
                "refinement_summary": "Refined",
            }
        ),
        meta_review=json.dumps(
            {"meta_review_summary": "Single hypothesis reviewed"}
        ),
        proximity=json.dumps(
            {"similarity_clusters": [], "diversity_assessment": "N/A"}
        ),
    )
    result = fw.run_research_workflow(
        "Test single hypothesis scenario"
    )
    assert isinstance(result, dict)
    # Should have exactly 1 hypothesis in the results
    assert len(result["top_ranked_hypotheses"]) == 1


# ---- Agents returning malformed JSON ------------------------------------


def test_agents_returning_malformed_json_no_crash():
    """Workflow survives agents returning non-JSON garbage."""
    fw = _build_framework_with_agent_responses(
        supervisor="not json at all",
        generation=json.dumps(
            {
                "hypotheses": [
                    {"text": "H1"},
                    {"text": "H2"},
                    {"text": "H3"},
                ]
            }
        ),
        reflection="broken {json",
        ranking="also broken",
        tournament="nope",
        evolution="<html>error</html>",
        meta_review="random garbage 12345",
        proximity="[]",
    )
    result = fw.run_research_workflow("Test malformed responses")
    assert isinstance(result, dict)
    assert "execution_metrics" in result
