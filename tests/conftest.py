"""Shared pytest fixtures for the AI-CoScientist test suite.

All fixtures mock swarms.Agent so tests NEVER call real LLMs.
The swarms package may not be installed in test environments, so we
inject a fake module into sys.modules before anything imports it.
"""

import sys
from unittest.mock import MagicMock

# ---- Inject fake swarms module BEFORE any ai_coscientist imports --------
# This must happen at module level (not inside a fixture) because Python
# caches module imports -- the first import wins.

_mock_swarms = MagicMock()
_mock_swarms.Agent = MagicMock()
_mock_swarms_structs = MagicMock()
_mock_swarms_structs_conversation = MagicMock()

sys.modules.setdefault("swarms", _mock_swarms)
sys.modules.setdefault("swarms.structs", _mock_swarms_structs)
sys.modules.setdefault(
    "swarms.structs.conversation", _mock_swarms_structs_conversation
)

# Now it is safe to import from ai_coscientist
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_agent():
    """Create a mock swarms.Agent that returns predefined JSON."""
    agent = MagicMock()
    agent.agent_name = "MockAgent"
    agent.run = MagicMock(
        return_value='{"hypotheses": [{"text": "Test hypothesis 1"}, {"text": "Test hypothesis 2"}]}'
    )
    return agent


@pytest.fixture
def framework():
    """Create AIScientistFramework with all agents mocked."""
    with patch("ai_coscientist.main.Agent") as MockAgent:
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
