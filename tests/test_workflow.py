"""Tests for the full run_research_workflow pipeline.

Every agent's .run() is mocked to return phase-appropriate JSON so
the workflow completes end-to-end without any LLM or network calls.
"""

import json
from unittest.mock import MagicMock, patch


# ---- Mock response helpers ------------------------------------------------

GENERATION_RESPONSE = json.dumps(
    {
        "hypotheses": [
            {"text": "H1: Hypothesis one"},
            {"text": "H2: Hypothesis two"},
            {"text": "H3: Hypothesis three"},
        ]
    }
)

REFLECTION_RESPONSE = json.dumps(
    {
        "hypothesis_text": "placeholder",
        "overall_score": 0.8,
        "review_summary": "Good hypothesis",
        "scores": {
            "scientific_soundness": 4,
            "novelty": 3,
            "relevance": 5,
            "testability": 4,
            "clarity": 5,
            "potential_impact": 4,
            "statistical_rigor": 4,
            "field_feasibility": 3,
            "spatial_scalability": 3,
            "environmental_sustainability": 4,
            "agronomic_practicality": 3,
        },
        "safety_ethical_concerns": "None identified",
        "detailed_feedback": {
            "scientific_soundness": "Solid foundation",
            "novelty": "Moderately novel",
            "relevance": "Highly relevant",
            "testability": "Testable",
            "clarity": "Clear",
            "potential_impact": "High",
            "statistical_rigor": "Appropriate statistical design",
            "field_feasibility": "Feasible with standard equipment",
            "spatial_scalability": "Scalable to field level",
            "environmental_sustainability": "Good environmental profile",
            "agronomic_practicality": "Practical for farmers",
        },
        "constructive_feedback": "Consider additional controls",
    }
)

RANKING_RESPONSE = json.dumps(
    {
        "ranked_hypotheses": [
            {"text": "H1: Hypothesis one", "overall_score": 0.9},
            {"text": "H2: Hypothesis two", "overall_score": 0.85},
            {"text": "H3: Hypothesis three", "overall_score": 0.8},
        ]
    }
)

TOURNAMENT_RESPONSE = json.dumps(
    {
        "winner": "a",
        "judgment_explanation": {
            "scientific_soundness_comparison": "A is stronger",
            "novelty_comparison": "Comparable",
            "relevance_comparison": "A is more relevant",
            "testability_comparison": "Both testable",
            "clarity_comparison": "A is clearer",
            "impact_comparison": "A has higher impact",
            "feasibility_comparison": "Both feasible",
            "statistical_rigor_comparison": "A has stronger design",
            "field_feasibility_comparison": "Both feasible",
            "spatial_scalability_comparison": "A scales better",
            "environmental_sustainability_comparison": "Comparable",
            "agronomic_practicality_comparison": "Both practical",
        },
        "decision_summary": "A wins on scientific merit",
        "confidence_level": "High",
    }
)

EVOLUTION_RESPONSE = json.dumps(
    {
        "original_hypothesis_text": "H1: Hypothesis one",
        "refined_hypothesis_text": "H1 refined: Improved hypothesis one",
        "refinement_summary": "Improved clarity and testability",
        "specific_refinements": [
            {
                "aspect": "clarity",
                "change": "Reworded",
                "justification": "Clearer",
            },
        ],
    }
)

META_REVIEW_RESPONSE = json.dumps(
    {
        "meta_review_summary": "Good overall quality",
        "recurring_themes": [],
        "strengths": ["Well-formulated"],
        "weaknesses": ["Could be bolder"],
        "process_assessment": {
            "generation_process": "Adequate",
            "review_process": "Thorough",
            "evolution_process": "Effective",
        },
        "strategic_recommendations": [],
        "potential_connections": [],
    }
)

PROXIMITY_RESPONSE = json.dumps(
    {
        "similarity_clusters": [],
        "diversity_assessment": "Diverse set of hypotheses",
        "redundancy_assessment": "Minimal redundancy",
    }
)

SUPERVISOR_RESPONSE = json.dumps(
    {
        "research_goal_analysis": {
            "goal_summary": "Test research",
            "key_areas": ["test"],
            "constraints_identified": [],
            "success_criteria": ["quality"],
        },
        "workflow_plan": {
            "generation_phase": {
                "focus_areas": ["test area"],
                "diversity_targets": "high",
                "quantity_target": 3,
            },
            "review_phase": {
                "critical_criteria": ["soundness"],
                "review_depth": "standard",
            },
            "ranking_phase": {
                "ranking_approach": "composite",
                "selection_criteria": ["score"],
            },
            "evolution_phase": {
                "refinement_priorities": ["clarity"],
                "iteration_strategy": "top-k",
            },
        },
    }
)


def _build_framework():
    """Build a framework with per-agent mock routing."""
    with patch(
        "ai_coscientist.main.DirectLLMAgent"
    ) as MockAgentCls:
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

        # Map agent_name -> mock object for targeted response setup
        agent_map = {a.agent_name: a for a in agents_created}

        # Wire up per-agent responses
        agent_map["Supervisor"].run.return_value = SUPERVISOR_RESPONSE
        agent_map["HypothesisGenerator"].run.return_value = (
            GENERATION_RESPONSE
        )
        agent_map["HypothesisReflector"].run.return_value = (
            REFLECTION_RESPONSE
        )
        agent_map["HypothesisRanker"].run.return_value = (
            RANKING_RESPONSE
        )
        agent_map["TournamentJudge"].run.return_value = (
            TOURNAMENT_RESPONSE
        )
        agent_map["HypothesisEvolver"].run.return_value = (
            EVOLUTION_RESPONSE
        )
        agent_map["MetaReviewer"].run.return_value = (
            META_REVIEW_RESPONSE
        )
        agent_map["ProximityAnalyzer"].run.return_value = (
            PROXIMITY_RESPONSE
        )

        return fw


# ---- Tests ---------------------------------------------------------------


def test_workflow_returns_dict_with_expected_keys():
    """run_research_workflow returns a dict with required top-level keys."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    assert isinstance(result, dict)
    for key in (
        "top_ranked_hypotheses",
        "meta_review_insights",
        "execution_metrics",
        "total_workflow_time",
    ):
        assert key in result, f"Missing key: {key}"


def test_workflow_has_hypotheses():
    """Workflow produces at least one top-ranked hypothesis."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    assert len(result["top_ranked_hypotheses"]) > 0


def test_workflow_execution_metrics_structure():
    """execution_metrics has the correct sub-keys."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    metrics = result["execution_metrics"]
    for key in (
        "total_time",
        "hypothesis_count",
        "reviews_count",
        "tournaments_count",
        "evolutions_count",
        "agent_execution_times",
    ):
        assert key in metrics, f"Missing metrics key: {key}"


def test_workflow_total_time_positive():
    """total_workflow_time is a positive number."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    assert result["total_workflow_time"] > 0


def test_workflow_meta_review_present():
    """meta_review_insights is populated."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    assert isinstance(result["meta_review_insights"], dict)
    assert "meta_review_summary" in result["meta_review_insights"]


def test_workflow_hypothesis_dicts_have_text():
    """Each hypothesis dict in the output contains a 'text' key."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    for h in result["top_ranked_hypotheses"]:
        assert "text" in h


def test_workflow_conversation_history_present():
    """conversation_history key is present in the result."""
    fw = _build_framework()
    result = fw.run_research_workflow("Test research goal for AI")
    assert "conversation_history" in result
