"""Tests verifying agronomy-specific content in default prompts."""

from ai_coscientist.prompts import (
    _default_generation_prompt,
    _default_reflection_prompt,
    _default_ranking_prompt,
    _default_evolution_prompt,
    _default_meta_review_prompt,
    _default_proximity_prompt,
    _default_tournament_prompt,
    _default_supervisor_prompt,
)


# Helper to check multiple keywords in a prompt
def _assert_keywords(prompt_text, keywords, prompt_name):
    lower = prompt_text.lower()
    for kw in keywords:
        assert (
            kw.lower() in lower
        ), f"'{kw}' not found in {prompt_name}"


# --- Generation prompt ---


def test_generation_prompt_iwm_pillars():
    text = _default_generation_prompt()
    _assert_keywords(
        text,
        [
            "weed",
            "IWM",
            "cover crop",
            "herbicide",
            "RCBD",
            "precision",
        ],
        "generation",
    )


def test_generation_prompt_statistical_methods():
    text = _default_generation_prompt()
    _assert_keywords(
        text,
        ["GLMM", "Bayesian", "spatial"],
        "generation",
    )


# --- Reflection prompt ---


def test_reflection_prompt_eleven_criteria():
    text = _default_reflection_prompt()
    _assert_keywords(
        text,
        [
            "scientific_soundness",
            "novelty",
            "relevance",
            "testability",
            "clarity",
            "potential_impact",
            "statistical_rigor",
            "field_feasibility",
            "spatial_scalability",
            "environmental_sustainability",
            "agronomic_practicality",
        ],
        "reflection",
    )


def test_reflection_prompt_agronomy_content():
    text = _default_reflection_prompt()
    _assert_keywords(
        text,
        [
            "herbicide resistance",
            "spatial autocorrelation",
            "overall_score",
        ],
        "reflection",
    )


# --- Ranking prompt ---


def test_ranking_prompt_iwm_weighting():
    text = _default_ranking_prompt()
    _assert_keywords(
        text,
        [
            "field_feasibility",
            "agronomic_practicality",
            "multi-tactic",
        ],
        "ranking",
    )


# --- Tournament prompt ---


def test_tournament_prompt_grouped_dimensions():
    text = _default_tournament_prompt()
    _assert_keywords(
        text,
        [
            "statistical_rigor",
            "field_feasibility",
            "environmental_sustainability",
            "winner",
        ],
        "tournament",
    )


# --- Evolution prompt ---


def test_evolution_prompt_agronomy_refinement():
    text = _default_evolution_prompt()
    _assert_keywords(
        text,
        [
            "RCBD",
            "remote sensing",
            "NDVI",
            "variable-rate",
        ],
        "evolution",
    )


# --- Meta-review prompt ---


def test_meta_review_prompt_iwm_coverage():
    text = _default_meta_review_prompt()
    _assert_keywords(
        text,
        [
            "cultural",
            "mechanical",
            "chemical",
            "biological",
            "technological",
            "spatial",
        ],
        "meta_review",
    )


# --- Proximity prompt ---


def test_proximity_prompt_agronomy_clustering():
    text = _default_proximity_prompt()
    _assert_keywords(
        text,
        [
            "weed",
            "cropping system",
            "IWM",
        ],
        "proximity",
    )


# --- Supervisor prompt ---


def test_supervisor_prompt_iwm_orchestration():
    text = _default_supervisor_prompt()
    _assert_keywords(
        text,
        [
            "IWM",
            "herbicide resistance",
            "growing season",
            "statistical rigor",
        ],
        "supervisor",
    )
