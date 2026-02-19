"""Type definitions for the AI-CoScientist framework.

Contains all enums, TypedDicts, and dataclasses used across the system.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
)

from loguru import logger

from .elo import calculate_elo_update


class AgentRole(Enum):
    """Define the possible roles for agents in the AI co-scientist system."""

    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"
    PROXIMITY = "proximity"
    SUPERVISOR = "supervisor"
    TOURNAMENT = "tournament"


# Type definitions for better type safety
class ReviewScores(TypedDict):
    """Type definition for review scores."""

    scientific_soundness: int
    novelty: int
    relevance: int
    testability: int
    clarity: int
    potential_impact: int
    statistical_rigor: int
    field_feasibility: int
    spatial_scalability: int
    environmental_sustainability: int
    agronomic_practicality: int


class DetailedFeedback(TypedDict):
    """Type definition for detailed feedback."""

    scientific_soundness: str
    novelty: str
    relevance: str
    testability: str
    clarity: str
    potential_impact: str
    statistical_rigor: str
    field_feasibility: str
    spatial_scalability: str
    environmental_sustainability: str
    agronomic_practicality: str


class HypothesisReview(TypedDict):
    """Type definition for hypothesis review."""

    hypothesis_text: str
    review_summary: str
    scores: ReviewScores
    safety_ethical_concerns: str
    detailed_feedback: DetailedFeedback
    constructive_feedback: str
    overall_score: float


class AgentExecutionMetrics(TypedDict):
    """Type definition for agent execution metrics."""

    total_time: float
    calls: int
    avg_time: float


class ExecutionMetrics(TypedDict):
    """Type definition for execution metrics."""

    total_time: float
    hypothesis_count: int
    reviews_count: int
    tournaments_count: int
    evolutions_count: int
    agent_execution_times: Dict[str, AgentExecutionMetrics]


class SimilarHypothesis(TypedDict):
    """Type definition for similar hypothesis in clustering."""

    text: str
    similarity_degree: str


class SimilarityCluster(TypedDict):
    """Type definition for similarity cluster."""

    cluster_id: str
    cluster_name: str
    central_theme: str
    similar_hypotheses: List[SimilarHypothesis]
    synthesis_potential: str


class ProximityAnalysisResult(TypedDict):
    """Type definition for proximity analysis result."""

    similarity_clusters: List[SimilarityCluster]
    diversity_assessment: str
    redundancy_assessment: str


class TournamentDimensionScores(TypedDict):
    """Per-dimension scores for a tournament match."""

    scientific_merit: Dict[str, int]
    practical_value: Dict[str, int]
    impact: Dict[str, int]
    communication: Dict[str, int]


class TournamentJudgment(TypedDict):
    """Type definition for tournament judgment."""

    research_goal: str
    hypothesis_a: str
    hypothesis_b: str
    winner: str
    dimension_scores: TournamentDimensionScores
    judgment_explanation: Dict[str, str]
    decision_summary: str
    confidence_level: str


class WorkflowResult(TypedDict):
    """Type definition for workflow result."""

    top_ranked_hypotheses: List[Dict[str, Any]]
    meta_review_insights: Dict[str, Any]
    conversation_history: str
    execution_metrics: ExecutionMetrics
    total_workflow_time: float


@dataclass
class Hypothesis:
    """
    Represents a research hypothesis.

    Attributes:
        text (str): The text of the hypothesis.
        elo_rating (int): Elo rating for ranking (initially 1200).
        reviews (List[HypothesisReview]): List of review feedback for the hypothesis.
        score (float): Overall score based on reviews (0.0-1.0).
        similarity_cluster_id (Optional[str]): ID of the similarity cluster.
        evolution_history (List[str]): History of evolutions for this hypothesis.
        generation_timestamp (float): When the hypothesis was generated.
        win_count (int): Number of tournament wins.
        loss_count (int): Number of tournament losses.
    """

    text: str
    elo_rating: int = 1200
    reviews: List[HypothesisReview] = field(default_factory=list)
    score: float = 0.0
    similarity_cluster_id: Optional[str] = None
    evolution_history: List[str] = field(default_factory=list)
    generation_timestamp: float = field(default_factory=time.time)
    win_count: int = 0
    loss_count: int = 0
    elo_scientific: int = 1200
    elo_practical: int = 1200
    elo_impact: int = 1200
    elo_communication: int = 1200

    # Map dimension names to elo attribute names
    _DIM_TO_ATTR = {
        "scientific_merit": "elo_scientific",
        "practical_value": "elo_practical",
        "impact": "elo_impact",
        "communication": "elo_communication",
    }

    def update_dimension_elos(
        self,
        opponent: "Hypothesis",
        dimension_scores: Dict[str, Dict[str, int]],
        weights: Dict[str, float],
        k_factor: int = 32,
        is_h_a: bool = True,
    ) -> None:
        """Update per-dimension Elo ratings from tournament scores.

        *dimension_scores* maps dimension name to
        ``{"h_a": score, "h_b": score}`` (1-10 each).
        *is_h_a* indicates whether ``self`` is hypothesis A.
        """
        my_key = "h_a" if is_h_a else "h_b"
        opp_key = "h_b" if is_h_a else "h_a"

        for dim, attr in self._DIM_TO_ATTR.items():
            scores = dimension_scores.get(dim)
            if not isinstance(scores, dict):
                continue
            my_score = scores.get(my_key)
            opp_score = scores.get(opp_key)
            if not isinstance(
                my_score, (int, float)
            ) or not isinstance(opp_score, (int, float)):
                continue
            if my_score == opp_score:
                continue  # tie → no change
            win = my_score > opp_score
            my_elo = getattr(self, attr)
            opp_elo = getattr(opponent, attr)
            new_elo = calculate_elo_update(
                my_elo, opp_elo, win, k_factor
            )
            setattr(self, attr, new_elo)

        # Recompute composite elo_rating as weighted sum
        self.elo_rating = round(
            sum(
                getattr(self, attr) * weights.get(dim, 0.25)
                for dim, attr in self._DIM_TO_ATTR.items()
            )
        )

    def update_elo(
        self, opponent_elo: int, win: bool, k_factor: int = 32
    ) -> None:
        """
        Update the Elo rating based on a tournament match outcome.

        Args:
            opponent_elo (int): The Elo rating of the opponent.
            win (bool): Whether this hypothesis won the match.
            k_factor (int): K-factor for Elo calculation, controlling update magnitude.
        """
        if not isinstance(opponent_elo, int) or not isinstance(
            win, bool
        ):
            logger.error(
                f"Invalid types for Elo update: opponent_elo={type(opponent_elo)}, win={type(win)}"
            )
            return

        self.elo_rating = calculate_elo_update(
            self.elo_rating, opponent_elo, win, k_factor
        )

        # Update win/loss count
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert the hypothesis to a dictionary representation."""
        return {
            "text": self.text,
            "elo_rating": self.elo_rating,
            "elo_scientific": self.elo_scientific,
            "elo_practical": self.elo_practical,
            "elo_impact": self.elo_impact,
            "elo_communication": self.elo_communication,
            "score": self.score,
            "reviews": self.reviews,
            "similarity_cluster_id": self.similarity_cluster_id,
            "evolution_history": self.evolution_history,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_matches": self.win_count + self.loss_count,
            "win_rate": round(
                self.win_count
                / max(1, (self.win_count + self.loss_count))
                * 100,
                2,
            ),
        }
