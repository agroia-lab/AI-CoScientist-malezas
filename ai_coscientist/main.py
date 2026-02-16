"""
AIScientistFramework: A multi-agent system for AI co-scientist based on
"Towards an AI co-scientist" research paper.
Implements hypothesis generation, review, ranking, and evolution using a tournament approach.
"""

import json
import random
import re
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
import os

from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.conversation import Conversation
from loguru import logger

from .types import (
    AgentRole,
    ReviewScores,
    DetailedFeedback,
    HypothesisReview,
    AgentExecutionMetrics,
    ExecutionMetrics,
    SimilarHypothesis,
    SimilarityCluster,
    ProximityAnalysisResult,
    TournamentJudgment,
    WorkflowResult,
    Hypothesis,
)
from .prompts import (
    get_generation_prompt,
    get_reflection_prompt,
    get_ranking_prompt,
    get_evolution_prompt,
    get_meta_review_prompt,
    get_proximity_prompt,
    get_tournament_prompt,
    get_supervisor_prompt,
)
from .json_parser import safely_parse_json

load_dotenv()

_API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
]


def _check_api_keys() -> None:
    """Warn if no LLM API keys are found in environment."""
    found = [k for k in _API_KEY_ENV_VARS if os.getenv(k)]
    if not found:
        logger.warning(
            f"No LLM API keys found in environment. "
            f"Set at least one of: {', '.join(_API_KEY_ENV_VARS)}"
        )


_check_api_keys()


class AIScientistFramework:
    """
    A multi-agent system framework for AI co-scientist, designed to generate
    and refine research hypotheses using tournament-based evolution.

    Attributes:
        model_name (str): Name of the LLM model to use for agents.
        max_iterations (int): Maximum number of iterations for the research workflow.
        base_path (Path): Base path for saving agent states.
        verbose (bool): Enable verbose logging.
        conversation (Conversation): Tracks the conversation history.
        hypotheses (List[Hypothesis]): List to store generated hypotheses.
        tournament_size (int): Number of hypotheses to include in each tournament round.
        hypotheses_per_generation (int): Number of hypotheses to generate initially.
        evolution_top_k (int): Number of top hypotheses to evolve in each iteration.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        max_iterations: int = 3,
        base_path: Optional[str] = None,
        verbose: bool = False,
        tournament_size: int = 8,
        hypotheses_per_generation: int = 10,
        evolution_top_k: int = 3,
        random_seed: Optional[int] = None,
        max_conversation_history: int = 500,
    ) -> None:
        """Initialize the AIScientistFramework system with configuration parameters."""
        # Type validation
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be str, got {type(model_name)}"
            )
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be positive int, got {max_iterations}"
            )
        if not isinstance(verbose, bool):
            raise TypeError(
                f"verbose must be bool, got {type(verbose)}"
            )

        self.model_name: str = model_name
        self.max_iterations: int = max_iterations
        self.base_path: Path = (
            Path(base_path)
            if base_path
            else Path("./ai_coscientist_states")
        )
        self.base_path.mkdir(
            exist_ok=True, parents=True, mode=0o700
        )
        self.verbose: bool = verbose
        self.conversation: Conversation = Conversation()
        self.hypotheses: List[Hypothesis] = []

        # Tournament and evolution parameters
        self.tournament_size: int = tournament_size
        self.hypotheses_per_generation: int = (
            hypotheses_per_generation
        )
        self.evolution_top_k: int = evolution_top_k

        # Reproducibility
        self.random_seed: Optional[int] = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(
                f"Random seed set to {random_seed}"
            )

        # Conversation history bounding
        self.max_conversation_history: int = (
            max_conversation_history
        )

        # Execution metrics
        self.start_time: Optional[float] = None
        self.execution_metrics: ExecutionMetrics = {
            "total_time": 0.0,
            "hypothesis_count": 0,
            "reviews_count": 0,
            "tournaments_count": 0,
            "evolutions_count": 0,
            "agent_execution_times": {},
        }

        # Initialize agents
        self._init_agents()
        logger.info(
            f"AIScientistFramework initialized with model: {model_name}"
        )

    def _init_agents(self) -> None:
        """Initialize all specialized agents with their roles and prompts."""
        try:
            self.generation_agent: Agent = Agent(
                agent_name="HypothesisGenerator",
                system_prompt=get_generation_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "generation_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.reflection_agent: Agent = Agent(
                agent_name="HypothesisReflector",
                system_prompt=get_reflection_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "reflection_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.ranking_agent: Agent = Agent(
                agent_name="HypothesisRanker",
                system_prompt=get_ranking_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "ranking_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.evolution_agent: Agent = Agent(
                agent_name="HypothesisEvolver",
                system_prompt=get_evolution_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "evolution_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.meta_review_agent: Agent = Agent(
                agent_name="MetaReviewer",
                system_prompt=get_meta_review_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "meta_review_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.proximity_agent: Agent = Agent(
                agent_name="ProximityAnalyzer",
                system_prompt=get_proximity_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "proximity_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.tournament_agent: Agent = Agent(
                agent_name="TournamentJudge",
                system_prompt=get_tournament_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "tournament_agent_state.json"
                ),
                verbose=self.verbose,
            )
            self.supervisor_agent: Agent = Agent(
                agent_name="Supervisor",
                system_prompt=get_supervisor_prompt(),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(
                    self.base_path / "supervisor_agent_state.json"
                ),
                verbose=self.verbose,
            )
            logger.success("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON string, handling potential errors.

        Delegates to the standalone :func:`safely_parse_json` utility.

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed JSON as dictionary or error dictionary
        """
        return safely_parse_json(json_str)

    def _time_execution(
        self, agent_name: str, start_time: float
    ) -> None:
        """
        Track execution time for an agent.

        Args:
            agent_name: Name of the agent
            start_time: Start time of execution
        """
        if not isinstance(agent_name, str):
            logger.error(
                f"agent_name must be str, got {type(agent_name)}"
            )
            return
        if not isinstance(start_time, (int, float)):
            logger.error(
                f"start_time must be numeric, got {type(start_time)}"
            )
            return

        execution_time = time.time() - start_time

        if (
            agent_name
            not in self.execution_metrics["agent_execution_times"]
        ):
            self.execution_metrics["agent_execution_times"][
                agent_name
            ] = AgentExecutionMetrics(
                total_time=0.0, calls=0, avg_time=0.0
            )

        metrics = self.execution_metrics["agent_execution_times"][
            agent_name
        ]
        metrics["total_time"] += execution_time
        metrics["calls"] += 1
        metrics["avg_time"] = metrics["total_time"] / metrics["calls"]

        logger.debug(
            f"Agent {agent_name} execution time: {execution_time:.2f}s (avg: {metrics['avg_time']:.2f}s)"
        )

    def _prune_conversation(self) -> None:
        """Prune conversation history if it exceeds the max size."""
        try:
            history = self.conversation.conversation_history
            if len(history) > self.max_conversation_history:
                excess = (
                    len(history) - self.max_conversation_history
                )
                del history[:excess]
                logger.debug(
                    f"Pruned {excess} old conversation entries"
                    f" (kept {self.max_conversation_history})"
                )
        except AttributeError:
            pass  # Conversation impl may differ

    def _run_generation_phase(
        self, research_goal: str
    ) -> List[Hypothesis]:
        """
        Run the hypothesis generation phase.

        Args:
            research_goal: The research goal to generate hypotheses for

        Returns:
            List of generated hypotheses
        """
        if (
            not isinstance(research_goal, str)
            or not research_goal.strip()
        ):
            raise ValueError(
                f"research_goal must be non-empty string, got: {research_goal}"
            )

        start_time = time.time()
        logger.info(
            f"Starting generation phase for goal: {research_goal[:100]}..."
        )

        # Get research plan from supervisor
        supervisor_input = {
            "task": "plan_research",
            "research_goal": research_goal,
            "phase": "generation",
            "parameters": {
                "hypotheses_count": self.hypotheses_per_generation,
                "diversity_target": "high",
            },
        }
        logger.debug("Requesting research plan from supervisor")
        supervisor_response = self.supervisor_agent.run(
            json.dumps(supervisor_input)
        )

        # Handle empty responses from supervisor agent
        if not supervisor_response or not supervisor_response.strip():
            logger.warning(
                "Supervisor agent returned empty response, using default plan"
            )
            supervisor_response = '{"workflow_plan": {"generation_phase": {"focus_areas": ["general research"], "diversity_targets": "high", "quantity_target": 10}}}'

        self.conversation.add(
            role=self.supervisor_agent.agent_name,
            content=supervisor_response,
        )
        supervisor_data = self._safely_parse_json(supervisor_response)

        # Run generation agent with supervisor guidance
        generation_input = {
            "research_goal": research_goal,
            "supervisor_guidance": supervisor_data,
            "required_hypotheses_count": (
                self.hypotheses_per_generation
            ),
        }
        logger.debug(
            "Running hypothesis generation with supervisor guidance"
        )
        generation_response = self.generation_agent.run(
            json.dumps(generation_input)
        )

        # Handle empty responses from agent
        if not generation_response or not generation_response.strip():
            logger.warning(
                "Generation agent returned empty response, using fallback"
            )
            generation_response = '{"hypotheses": []}'

        self.conversation.add(
            role=self.generation_agent.agent_name,
            content=generation_response,
        )

        generation_data = self._safely_parse_json(generation_response)
        initial_hypotheses_data = generation_data.get(
            "hypotheses", []
        )

        if not initial_hypotheses_data:
            logger.warning(
                "Generation Agent returned no hypotheses. Using fallback generation."
            )
            # Fallback to simpler generation prompt
            fallback_input = {
                "research_goal": research_goal,
                "count": self.hypotheses_per_generation,
            }
            fallback_response = self.generation_agent.run(
                json.dumps(fallback_input)
            )

            # Handle empty fallback response
            if not fallback_response or not fallback_response.strip():
                logger.warning(
                    "Fallback generation also returned empty response"
                )
                fallback_response = '{"hypotheses": []}'

            fallback_data = self._safely_parse_json(fallback_response)
            initial_hypotheses_data = fallback_data.get(
                "hypotheses", []
            )

            # Last resort: create basic hypotheses manually
            if not initial_hypotheses_data:
                logger.warning(
                    "All generation attempts failed. Creating basic hypotheses manually."
                )
                initial_hypotheses_data = [
                    {
                        "text": (
                            f"Investigate the relationship between {research_goal.split()[-2] if len(research_goal.split()) > 1 else 'variables'} and performance metrics."
                        )
                    },
                    {
                        "text": (
                            f"Develop novel approaches to improve {research_goal.split()[0] if research_goal.split() else 'system'} efficiency."
                        )
                    },
                    {
                        "text": (
                            f"Analyze the impact of different parameters on {research_goal.lower()}."
                        )
                    },
                ]
                logger.info(
                    f"Created {len(initial_hypotheses_data)} basic hypotheses as fallback"
                )

        # Convert to Hypothesis objects
        hypotheses: List[Hypothesis] = []
        for i, hy_data in enumerate(initial_hypotheses_data):
            try:
                if isinstance(hy_data, dict) and "text" in hy_data:
                    hypothesis_text = hy_data["text"]
                else:
                    hypothesis_text = str(hy_data)

                if not hypothesis_text.strip():
                    logger.warning(
                        f"Empty hypothesis text at index {i}, skipping"
                    )
                    continue

                hypotheses.append(
                    Hypothesis(text=hypothesis_text.strip())
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create hypothesis from data at index {i}: {e}"
                )
                continue

        self._time_execution("generation", start_time)
        self.execution_metrics["hypothesis_count"] += len(hypotheses)
        logger.success(
            f"Generated {len(hypotheses)} initial hypotheses."
        )
        return hypotheses

    def _run_reflection_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run the hypothesis reflection (review) phase.

        Args:
            hypotheses: List of hypotheses to review

        Returns:
            List of reviewed hypotheses
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if not hypotheses:
            logger.warning(
                "No hypotheses provided for reflection phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting reflection phase for {len(hypotheses)} hypotheses"
        )

        reviewed_hypotheses: List[Hypothesis] = []

        for i, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, Hypothesis):
                logger.error(
                    f"Invalid hypothesis type at index {i}: {type(hypothesis)}"
                )
                continue

            try:
                review_input = {"hypothesis_text": hypothesis.text}
                logger.debug(
                    f"Reviewing hypothesis {i+1}/{len(hypotheses)}"
                )
                review_response = self.reflection_agent.run(
                    json.dumps(review_input)
                )

                # Handle empty responses from reflection agent
                if not review_response or not review_response.strip():
                    logger.warning(
                        f"Reflection agent returned empty response for hypothesis {i+1}"
                    )
                    review_response = '{"overall_score": 0.5, "review_summary": "No review available"}'

                self.conversation.add(
                    role=self.reflection_agent.agent_name,
                    content=review_response,
                )
                review_data = self._safely_parse_json(review_response)

                if review_data and "overall_score" in review_data:
                    overall_score = review_data.get(
                        "overall_score", 0.0
                    )
                    try:
                        hypothesis.score = float(overall_score)
                        # Validate the review data structure before appending
                        if isinstance(review_data, dict):
                            hypothesis.reviews.append(
                                review_data
                            )  # Store full review data
                        reviewed_hypotheses.append(hypothesis)
                        logger.debug(
                            f"Successfully reviewed hypothesis {i+1} with score {overall_score}"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid score format for hypothesis {i+1}: {overall_score}, error: {e}"
                        )
                        hypothesis.score = 0.0
                        reviewed_hypotheses.append(hypothesis)
                else:
                    logger.warning(
                        f"No valid review score found for hypothesis {i+1}: {hypothesis.text[:50]}..."
                    )
                    reviewed_hypotheses.append(
                        hypothesis
                    )  # Keep hypothesis even if review fails but log warning

            except Exception as e:
                logger.error(f"Error reviewing hypothesis {i+1}: {e}")
                reviewed_hypotheses.append(
                    hypothesis
                )  # Keep hypothesis even if review fails

        self._time_execution("reflection", start_time)
        self.execution_metrics["reviews_count"] += len(
            reviewed_hypotheses
        )
        logger.success(
            f"Hypotheses reviewed. Total reviews: {len(reviewed_hypotheses)}"
        )
        return reviewed_hypotheses

    def _run_ranking_phase(
        self, reviewed_hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run the hypothesis ranking phase.

        Args:
            reviewed_hypotheses: List of reviewed hypotheses to rank

        Returns:
            List of ranked hypotheses
        """
        if not isinstance(reviewed_hypotheses, list):
            raise TypeError(
                f"reviewed_hypotheses must be list, got {type(reviewed_hypotheses)}"
            )
        if not reviewed_hypotheses:
            logger.warning("No hypotheses provided for ranking phase")
            return []

        start_time = time.time()
        logger.info(
            f"Starting ranking phase for {len(reviewed_hypotheses)} hypotheses"
        )

        ranking_input = [
            {"text": h.text, "overall_score": h.score}
            for h in reviewed_hypotheses
        ]
        logger.debug("Running hypothesis ranking agent")
        ranking_response = self.ranking_agent.run(
            json.dumps({"hypotheses_for_ranking": ranking_input})
        )

        if not ranking_response or not ranking_response.strip():
            logger.warning(
                "Ranking agent returned empty response, using score-based fallback"
            )
            ranking_response = '{"ranked_hypotheses": []}'

        self.conversation.add(
            role=self.ranking_agent.agent_name,
            content=ranking_response,
        )
        ranking_data = self._safely_parse_json(ranking_response)
        ranked_hypothesis_data = ranking_data.get(
            "ranked_hypotheses", []
        )

        ranked_hypotheses: List[Hypothesis] = []
        hypothesis_map: Dict[str, Hypothesis] = {
            h.text: h for h in reviewed_hypotheses
        }  # For efficient lookup

        for i, ranked_hy_data in enumerate(ranked_hypothesis_data):
            if not isinstance(ranked_hy_data, dict):
                logger.warning(
                    f"Invalid ranked hypothesis data at index {i}: {type(ranked_hy_data)}"
                )
                continue

            hypothesis_text = ranked_hy_data.get("text")
            if hypothesis_text and hypothesis_text in hypothesis_map:
                ranked_hypotheses.append(
                    hypothesis_map[hypothesis_text]
                )
                logger.debug(
                    f"Successfully ranked hypothesis {i+1}: {hypothesis_text[:50]}..."
                )
            else:
                logger.warning(
                    f"Ranked hypothesis data missing text or text not found in original hypotheses at index {i}"
                )

        # If ranking failed, fall back to original order
        if not ranked_hypotheses:
            logger.warning(
                "Ranking agent returned no valid rankings, using score-based fallback order"
            )
            ranked_hypotheses = sorted(
                reviewed_hypotheses,
                key=lambda h: h.score,
                reverse=True,
            )

        self._time_execution("ranking", start_time)
        logger.success(
            f"Hypotheses ranked. Final count: {len(ranked_hypotheses)}"
        )
        return ranked_hypotheses

    def _run_evolution_phase(
        self,
        top_hypotheses: List[Hypothesis],
        meta_review_data: Dict[str, Any],
    ) -> List[Hypothesis]:
        """
        Run the hypothesis evolution phase.

        Args:
            top_hypotheses: List of top hypotheses to evolve
            meta_review_data: Meta-review insights for evolution guidance

        Returns:
            List of evolved hypotheses
        """
        if not isinstance(top_hypotheses, list):
            raise TypeError(
                f"top_hypotheses must be list, got {type(top_hypotheses)}"
            )
        if not isinstance(meta_review_data, dict):
            logger.warning(
                f"meta_review_data should be dict, got {type(meta_review_data)}"
            )
            meta_review_data = {}
        if not top_hypotheses:
            logger.warning(
                "No hypotheses provided for evolution phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting evolution phase for {len(top_hypotheses)} hypotheses"
        )

        evolved_hypotheses: List[Hypothesis] = []

        for i, hypothesis in enumerate(top_hypotheses):
            if not isinstance(hypothesis, Hypothesis):
                logger.error(
                    f"Invalid hypothesis type at index {i}: {type(hypothesis)}"
                )
                continue

            try:
                evolution_input = {
                    "original_hypothesis_text": hypothesis.text,
                    "review_feedback": (
                        hypothesis.reviews[-1]
                        if hypothesis.reviews
                        else {}
                    ),  # Use latest review
                    "meta_review_insights": meta_review_data,
                }
                logger.debug(
                    f"Evolving hypothesis {i+1}/{len(top_hypotheses)}"
                )
                evolution_response = self.evolution_agent.run(
                    json.dumps(evolution_input)
                )

                # Fallback if evolution agent returns nothing
                if (
                    not evolution_response
                    or not evolution_response.strip()
                ):
                    logger.warning(
                        f"Evolution agent returned empty response for hypothesis {i+1}"
                    )
                    evolution_response = json.dumps(
                        {
                            "original_hypothesis_text": (
                                hypothesis.text
                            ),
                            "refined_hypothesis_text": (
                                hypothesis.text + " [refined]"
                            ),
                            "refinement_summary": (
                                "Automatic minimal refinement – agent returned no content"
                            ),
                        }
                    )

                self.conversation.add(
                    role=self.evolution_agent.agent_name,
                    content=evolution_response,
                )
                evolution_data = self._safely_parse_json(
                    evolution_response
                )
                refined_hypothesis_text = evolution_data.get(
                    "refined_hypothesis_text"
                )

                if (
                    refined_hypothesis_text
                    and refined_hypothesis_text.strip()
                ):
                    hypothesis.text = refined_hypothesis_text.strip()
                    refinement_summary = evolution_data.get(
                        "refinement_summary", "Evolution completed"
                    )
                    hypothesis.evolution_history.append(
                        refinement_summary
                    )  # Track evolution
                    evolved_hypotheses.append(hypothesis)
                    logger.debug(
                        f"Hypothesis {i+1} evolved successfully: {hypothesis.text[:50]}..."
                    )
                else:
                    evolved_hypotheses.append(
                        hypothesis
                    )  # Keep original if no refinement
                    logger.warning(
                        f"Hypothesis {i+1} evolution failed or returned no refined text"
                    )

            except Exception as e:
                logger.error(f"Error evolving hypothesis {i+1}: {e}")
                evolved_hypotheses.append(
                    hypothesis
                )  # Keep original on error

        self._time_execution("evolution", start_time)
        self.execution_metrics["evolutions_count"] += len(
            evolved_hypotheses
        )
        logger.success(
            f"Evolution phase completed. {len(evolved_hypotheses)} hypotheses processed"
        )
        return evolved_hypotheses

    def _run_meta_review_phase(
        self, reviewed_hypotheses: List[Hypothesis]
    ) -> Dict[str, Any]:
        """
        Run the meta-review phase to synthesize insights from reviews.

        Args:
            reviewed_hypotheses: List of hypotheses with reviews

        Returns:
            Meta-review insights and recommendations
        """
        if not isinstance(reviewed_hypotheses, list):
            raise TypeError(
                f"reviewed_hypotheses must be list, got {type(reviewed_hypotheses)}"
            )
        if not reviewed_hypotheses:
            logger.warning(
                "No hypotheses provided for meta-review phase"
            )
            return {}

        start_time = time.time()
        logger.info(
            f"Starting meta-review phase for {len(reviewed_hypotheses)} hypotheses"
        )

        # Extract latest reviews, handling missing reviews gracefully
        all_reviews_for_meta = []
        for i, h in enumerate(reviewed_hypotheses):
            if h.reviews:
                all_reviews_for_meta.append(h.reviews[-1])
            else:
                logger.debug(
                    f"Hypothesis {i+1} has no reviews, using empty review"
                )
                all_reviews_for_meta.append({})

        logger.debug(
            f"Collected {len(all_reviews_for_meta)} reviews for meta-analysis"
        )
        meta_review_response = self.meta_review_agent.run(
            json.dumps({"reviews": all_reviews_for_meta})
        )

        if (
            not meta_review_response
            or not meta_review_response.strip()
        ):
            logger.warning(
                "Meta-review agent returned empty response"
            )
            meta_review_response = (
                '{"meta_review_summary": "No meta-review available"}'
            )

        self.conversation.add(
            role=self.meta_review_agent.agent_name,
            content=meta_review_response,
        )
        meta_review_data = self._safely_parse_json(
            meta_review_response
        )

        # Validate meta-review data structure
        if not isinstance(meta_review_data, dict):
            logger.warning(
                f"Meta-review returned invalid data type: {type(meta_review_data)}"
            )
            meta_review_data = {
                "error": "Invalid meta-review response",
                "content": str(meta_review_data),
            }

        self._time_execution("meta_review", start_time)
        logger.success("Meta-review phase completed")
        return meta_review_data

    def _run_proximity_analysis_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run proximity analysis to cluster similar hypotheses.

        Args:
            hypotheses: List of hypotheses to analyze for similarity

        Returns:
            List of hypotheses with cluster assignments
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if not hypotheses:
            logger.warning(
                "No hypotheses provided for proximity analysis phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting proximity analysis phase for {len(hypotheses)} hypotheses"
        )

        hypothesis_texts = [
            h.text for h in hypotheses if isinstance(h, Hypothesis)
        ]
        if len(hypothesis_texts) != len(hypotheses):
            logger.warning(
                f"Filtered out {len(hypotheses) - len(hypothesis_texts)} invalid hypotheses"
            )

        logger.debug(
            f"Analyzing similarity for {len(hypothesis_texts)} hypothesis texts"
        )
        proximity_response = self.proximity_agent.run(
            json.dumps({"hypotheses_texts": hypothesis_texts})
        )

        if (
            not proximity_response
            or not proximity_response.strip()
        ):
            logger.warning(
                "Proximity agent returned empty response"
            )
            proximity_response = (
                '{"similarity_clusters": []}'
            )

        self.conversation.add(
            role=self.proximity_agent.agent_name,
            content=proximity_response,
        )
        proximity_data = self._safely_parse_json(proximity_response)

        if not isinstance(proximity_data, dict):
            logger.error(
                f"Invalid proximity data type: {type(proximity_data)}"
            )
            return hypotheses

        similarity_clusters = proximity_data.get(
            "similarity_clusters", []
        )
        logger.debug(
            f"Found {len(similarity_clusters)} similarity clusters"
        )

        # Assign cluster IDs to hypotheses
        clusters_assigned = 0
        for cluster in similarity_clusters:
            if not isinstance(cluster, dict):
                logger.warning(
                    f"Invalid cluster data type: {type(cluster)}"
                )
                continue

            cluster_id = cluster.get("cluster_id", "no_cluster_id")
            similar_hypotheses = cluster.get("similar_hypotheses", [])

            for hy_text_data in similar_hypotheses:
                # Handle different formats for hypothesis text
                if isinstance(hy_text_data, dict):
                    hy_text = hy_text_data.get("text")
                else:
                    hy_text = str(hy_text_data)

                if hy_text:
                    # Find matching hypothesis and assign cluster
                    for hy in self.hypotheses:
                        if (
                            isinstance(hy, Hypothesis)
                            and hy.text == hy_text
                        ):
                            hy.similarity_cluster_id = cluster_id
                            clusters_assigned += 1
                            logger.debug(
                                f"Assigned cluster {cluster_id} to hypothesis: {hy_text[:50]}..."
                            )
                            break

        self._time_execution("proximity_analysis", start_time)
        logger.success(
            f"Proximity analysis completed. {clusters_assigned} cluster assignments made"
        )
        return hypotheses

    def _run_tournament_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run tournament selection and Elo rating update.

        Args:
            hypotheses: List of hypotheses to compete in tournament

        Returns:
            List of hypotheses sorted by Elo rating
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if len(hypotheses) < 2:
            logger.warning(
                f"Need at least 2 hypotheses for tournament, got {len(hypotheses)}"
            )
            return hypotheses

        if self.random_seed is not None:
            random.seed(self.random_seed)

        start_time = time.time()
        tournament_rounds = (
            len(hypotheses) * 3
        )  # 3 rounds per hypothesis
        k_factor = 32  # Standard Elo K-factor (matches Hypothesis.update_elo default)

        logger.info(
            f"Starting tournament phase: {len(hypotheses)} hypotheses, {tournament_rounds} rounds"
        )

        valid_rounds = 0
        skipped_rounds = 0

        for round_num in range(tournament_rounds):
            try:
                # Randomly select two different hypotheses for a match
                h1, h2 = random.sample(hypotheses, 2)

                # Double-check they're different (random.sample should guarantee this)
                if h1 is h2 or h1.text == h2.text:
                    logger.debug(
                        f"Skipping round {round_num+1}: identical hypotheses selected"
                    )
                    skipped_rounds += 1
                    continue

                tournament_input = {
                    "research_goal": (
                        "Compare hypotheses for tournament"
                    ),  # General goal context
                    "hypothesis_a": h1.text,
                    "hypothesis_b": h2.text,
                }

                logger.debug(
                    f"Tournament round {round_num+1}/{tournament_rounds}"
                )
                tournament_response = self.tournament_agent.run(
                    json.dumps(tournament_input)
                )

                if (
                    not tournament_response
                    or not tournament_response.strip()
                ):
                    logger.warning(
                        f"Tournament agent returned empty response in round {round_num+1}"
                    )
                    skipped_rounds += 1
                    continue

                self.conversation.add(
                    role=self.tournament_agent.agent_name,
                    content=tournament_response,
                )
                tournament_data = self._safely_parse_json(
                    tournament_response
                )

                winner_choice = tournament_data.get("winner")
                if winner_choice not in {"a", "b"}:
                    # Attempt regex extraction as fallback
                    match = re.search(
                        r'"winner"\s*:\s*"?([ab])"?',
                        tournament_response,
                        re.IGNORECASE,
                    )
                    if match:
                        winner_choice = match.group(1).lower()
                    else:
                        winner_choice = None

                if winner_choice == "a":
                    winner, loser = h1, h2
                elif winner_choice == "b":
                    winner, loser = h2, h1
                else:
                    logger.warning(
                        f"Round {round_num+1}: Invalid winner choice '{winner_choice}', skipping Elo update"
                    )
                    skipped_rounds += 1
                    continue

                # Update Elo ratings
                old_winner_elo = winner.elo_rating
                old_loser_elo = loser.elo_rating

                winner.update_elo(
                    loser.elo_rating, win=True, k_factor=k_factor
                )
                loser.update_elo(
                    old_winner_elo, win=False, k_factor=k_factor
                )  # Use old winner elo

                valid_rounds += 1
                logger.debug(
                    f"Round {round_num+1}: Winner Elo: {old_winner_elo} -> {winner.elo_rating}, "
                    f"Loser Elo: {old_loser_elo} -> {loser.elo_rating}"
                )

            except Exception as e:
                logger.error(
                    f"Error in tournament round {round_num+1}: {e}"
                )
                skipped_rounds += 1
                continue

        self._time_execution("tournament", start_time)
        self.execution_metrics["tournaments_count"] += valid_rounds
        logger.success(
            f"Tournament phase completed: {valid_rounds} valid rounds, {skipped_rounds} skipped"
        )

        # Rank hypotheses by Elo rating
        try:
            hypotheses.sort(key=lambda h: h.elo_rating, reverse=True)
            logger.debug(
                f"Hypotheses sorted by Elo rating. Top rating: {hypotheses[0].elo_rating}"
            )
        except Exception as e:
            logger.error(
                f"Error sorting hypotheses by Elo rating: {e}"
            )

        return hypotheses

    def run_research_workflow(
        self, research_goal: str
    ) -> WorkflowResult:
        """
        Execute the AI co-scientist research workflow to generate and refine hypotheses.

        Args:
            research_goal: The research goal provided by the scientist.

        Returns:
            A dictionary containing the final results, including top-ranked hypotheses,
            meta-review insights, and conversation history.
        """
        if (
            not isinstance(research_goal, str)
            or not research_goal.strip()
        ):
            raise ValueError(
                f"research_goal must be non-empty string, got: {research_goal}"
            )

        logger.info(
            f"Starting research workflow for goal: '{research_goal}'"
        )
        self.start_time = time.time()
        self.hypotheses = []  # Reset hypotheses list for a new run

        # Reset metrics while preserving structure
        self.execution_metrics = ExecutionMetrics(
            total_time=0.0,
            hypothesis_count=0,
            reviews_count=0,
            tournaments_count=0,
            evolutions_count=0,
            agent_execution_times={},
        )

        try:
            # --- Generation Phase ---
            self.hypotheses = self._run_generation_phase(
                research_goal
            )

            # Quality gate: abort early if generation failed
            if not self.hypotheses:
                logger.error(
                    "Generation phase produced 0 hypotheses even after fallbacks"
                )
                total_time = time.time() - self.start_time
                self.execution_metrics["total_time"] = (
                    total_time
                )
                return {
                    "top_ranked_hypotheses": [],
                    "meta_review_insights": {},
                    "conversation_history": (
                        self.conversation.return_history_as_string()
                    ),
                    "execution_metrics": (
                        self.execution_metrics
                    ),
                    "total_workflow_time": total_time,
                }

            # --- Reflection Phase ---
            self.hypotheses = self._run_reflection_phase(
                self.hypotheses
            )

            # Quality gate: warn if all scores are zero
            if self.hypotheses and all(
                h.score == 0.0 for h in self.hypotheses
            ):
                logger.warning(
                    "All hypotheses scored 0.0 after reflection"
                    " — results may be unreliable"
                )

            # --- Ranking Phase (Initial Ranking based on Reviews) ---
            self.hypotheses = self._run_ranking_phase(self.hypotheses)

            # --- Tournament Phase (Elo-based Ranking) ---
            self.hypotheses = self._run_tournament_phase(
                self.hypotheses
            )

            # --- Iterative Refinement Cycle ---
            meta_review_data: Dict[str, Any] = {}
            for iteration in range(self.max_iterations):
                logger.info(
                    f"Starting Iteration {iteration + 1} of {self.max_iterations}"
                )

                # --- Meta-Review ---
                meta_review_data = self._run_meta_review_phase(
                    self.hypotheses
                )

                # --- Evolution ---
                evo_count = min(
                    self.evolution_top_k, len(self.hypotheses)
                )
                top_hypotheses_for_evolution = (
                    self.hypotheses[:evo_count]
                )
                remaining_hypotheses = (
                    self.hypotheses[evo_count:]
                )
                logger.debug(
                    f"Evolving top {len(top_hypotheses_for_evolution)} hypotheses, preserving {len(remaining_hypotheses)} others"
                )
                evolved = self._run_evolution_phase(
                    top_hypotheses_for_evolution,
                    meta_review_data,
                )
                self.hypotheses = evolved + remaining_hypotheses

                # Re-run Reflection and Ranking on evolved hypotheses
                self.hypotheses = self._run_reflection_phase(
                    self.hypotheses
                )
                self.hypotheses = self._run_ranking_phase(
                    self.hypotheses
                )
                self.hypotheses = self._run_tournament_phase(
                    self.hypotheses
                )  # Tournament after evolution too

                # --- Proximity Analysis (after evolution and ranking each iteration) ---
                self.hypotheses = self._run_proximity_analysis_phase(
                    self.hypotheses
                )

                # Prune conversation history to bound memory
                self._prune_conversation()

                logger.success(f"Completed iteration {iteration + 1}")

            # --- Final Output ---
            top_ranked_hypotheses = self.hypotheses[
                : min(10, len(self.hypotheses))
            ]  # Return top 10 or fewer
            final_output_hypotheses = [
                h.to_dict() for h in top_ranked_hypotheses
            ]  # Convert to dict for output

            total_time = time.time() - self.start_time
            self.execution_metrics["total_time"] = total_time

            final_output: WorkflowResult = {
                "top_ranked_hypotheses": final_output_hypotheses,
                "meta_review_insights": meta_review_data,
                "conversation_history": (
                    self.conversation.return_history_as_string()
                ),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": total_time,
            }
            logger.success(
                f"Research workflow completed successfully in {total_time:.2f} seconds"
            )
            return final_output

        except Exception as e:
            total_time = (
                time.time() - self.start_time
                if self.start_time
                else 0.0
            )
            logger.error(f"Error in research workflow: {e}")
            logger.exception("Full traceback:")

            # Ensure execution metrics are properly structured
            if not isinstance(self.execution_metrics, dict):
                self.execution_metrics = {
                    "total_time": total_time,
                    "hypothesis_count": 0,
                    "reviews_count": 0,
                    "tournaments_count": 0,
                    "evolutions_count": 0,
                    "agent_execution_times": {},
                }

            # Return error response with proper typing (though not strictly WorkflowResult)
            error_response = {
                "error": str(e),
                "conversation_history": (
                    self.conversation.return_history_as_string()
                ),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": total_time,
                "top_ranked_hypotheses": [],
                "meta_review_insights": {},
            }
            return error_response  # type: ignore

    def save_state(self) -> None:
        """Save the state of all agents (if supported by the Agent implementation)."""
        agents = [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]

        saved_count = 0
        for agent in agents:
            if hasattr(agent, "save_state") and callable(
                getattr(agent, "save_state")
            ):
                try:
                    agent.save_state()  # type: ignore[attr-defined]
                    saved_count += 1
                    logger.debug(
                        f"State saved for {agent.agent_name}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Error saving state for {agent.agent_name}: {exc}"
                    )
            else:
                logger.warning(
                    f"Agent {agent.agent_name} does not implement save_state(); skipping"
                )

        logger.success(
            f"Successfully saved state for {saved_count}/{len(agents)} agents"
        )

    def load_state(self) -> None:
        """Load the saved state of all agents (if supported)."""
        agents = [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]

        loaded_count = 0
        for agent in agents:
            if hasattr(agent, "load_state") and callable(
                getattr(agent, "load_state")
            ):
                try:
                    agent.load_state()  # type: ignore[attr-defined]
                    loaded_count += 1
                    logger.debug(
                        f"State loaded for {agent.agent_name}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Error loading state for {agent.agent_name}: {exc}"
                    )
            else:
                logger.warning(
                    f"Agent {agent.agent_name} does not implement load_state(); skipping"
                )

        logger.success(
            f"Successfully loaded state for {loaded_count}/{len(agents)} agents"
        )
