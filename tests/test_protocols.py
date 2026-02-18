"""Tests for the AgentInterface protocol and
from_custom_agents factory.
"""

from unittest.mock import patch

import pytest

from ai_coscientist.protocols import AgentInterface


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


class _StubAgent:
    """Minimal agent that satisfies AgentInterface."""

    def __init__(self, name: str = "stub"):
        self.agent_name = name

    def run(self, input: str) -> str:
        return f"echo: {input}"


class _BadAgent:
    """Agent missing the ``run`` method."""

    def __init__(self):
        self.agent_name = "bad"


def _all_stub_agents():
    """Return a dict mapping every role to a _StubAgent."""
    roles = [
        "generation",
        "reflection",
        "ranking",
        "evolution",
        "meta_review",
        "proximity",
        "tournament",
        "supervisor",
    ]
    return {r: _StubAgent(r) for r in roles}


# -----------------------------------------------------------
# Protocol structural checks
# -----------------------------------------------------------


def test_stub_agent_satisfies_protocol():
    """_StubAgent is recognised by the runtime check."""
    assert isinstance(_StubAgent(), AgentInterface)


def test_bad_agent_does_not_satisfy_protocol():
    """An object missing ``run`` fails the check."""
    assert not isinstance(_BadAgent(), AgentInterface)


def test_plain_object_fails_protocol():
    """A plain object fails the protocol check."""
    assert not isinstance(object(), AgentInterface)


def test_protocol_is_runtime_checkable():
    """AgentInterface has __protocol_attrs__."""
    assert hasattr(AgentInterface, "__protocol_attrs__")


# -----------------------------------------------------------
# from_custom_agents
# -----------------------------------------------------------


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_creates_framework(
    _mock_agent,
):
    """Factory returns a working framework instance."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    fw = AIScientistFramework.from_custom_agents(agents)
    assert isinstance(fw, AIScientistFramework)


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_assigns_all_agents(
    _mock_agent,
):
    """Every agent attribute is the stub we supplied."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    fw = AIScientistFramework.from_custom_agents(agents)
    assert fw.generation_agent is agents["generation"]
    assert fw.reflection_agent is agents["reflection"]
    assert fw.ranking_agent is agents["ranking"]
    assert fw.evolution_agent is agents["evolution"]
    assert fw.meta_review_agent is agents["meta_review"]
    assert fw.proximity_agent is agents["proximity"]
    assert fw.tournament_agent is agents["tournament"]
    assert fw.supervisor_agent is agents["supervisor"]


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_missing_role_raises(
    _mock_agent,
):
    """Omitting a required role raises ValueError."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    del agents["supervisor"]
    with pytest.raises(ValueError, match="Missing"):
        AIScientistFramework.from_custom_agents(agents)


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_bad_agent_raises(
    _mock_agent,
):
    """An agent that does not satisfy the protocol
    raises TypeError."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    agents["ranking"] = _BadAgent()
    with pytest.raises(TypeError, match="AgentInterface"):
        AIScientistFramework.from_custom_agents(agents)


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_kwargs_forwarded(
    _mock_agent,
):
    """Extra kwargs like max_iterations are honoured."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    fw = AIScientistFramework.from_custom_agents(
        agents, max_iterations=5, verbose=True
    )
    assert fw.max_iterations == 5
    assert fw.verbose is True


@patch("ai_coscientist.main.DirectLLMAgent")
def test_from_custom_agents_defaults(
    _mock_agent,
):
    """Default config values are set when omitted."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    fw = AIScientistFramework.from_custom_agents(agents)
    assert fw.max_iterations == 3
    assert fw.verbose is False
    assert fw.tournament_size == 8
    assert fw.evolution_top_k == 3


@patch("ai_coscientist.main.DirectLLMAgent")
def test_custom_agent_run_is_callable(
    _mock_agent,
):
    """Injected agents can actually be called."""
    from ai_coscientist.main import AIScientistFramework

    agents = _all_stub_agents()
    fw = AIScientistFramework.from_custom_agents(agents)
    result = fw.generation_agent.run("hello")
    assert result == "echo: hello"


# -----------------------------------------------------------
# AgentInterface import from package root
# -----------------------------------------------------------


def test_agent_interface_importable_from_package():
    """AgentInterface is accessible via the top-level
    package import."""
    from ai_coscientist import AgentInterface as AI

    assert AI is AgentInterface
