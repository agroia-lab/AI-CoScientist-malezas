"""Shared pytest fixtures for the AI-CoScientist test suite.

All fixtures mock DirectLLMAgent so tests NEVER call real LLMs.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_agent():
    """Create a mock agent that returns predefined JSON."""
    agent = MagicMock()
    agent.agent_name = "MockAgent"
    agent.run = MagicMock(
        return_value='{"hypotheses": [{"text": "Test hypothesis 1"}, {"text": "Test hypothesis 2"}]}'
    )
    return agent


@pytest.fixture
def framework():
    """Create AIScientistFramework with all agents mocked."""
    with patch(
        "ai_coscientist.main.DirectLLMAgent"
    ) as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_name = "MockAgent"
        mock_instance.run = MagicMock(return_value="{}")
        MockAgent.return_value = mock_instance

        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(
            model_name="test-model",
            max_iterations=1,
            verbose=False,
            hypotheses_per_generation=3,
            tournament_size=4,
            evolution_top_k=2,
        )
        yield fw
