"""Protocol definitions for agent abstraction.

Defines :class:`AgentInterface` so that the framework is not
locked to a specific agent implementation (e.g. ``swarms.Agent``).
Any class that exposes ``agent_name: str`` and
``def run(self, input: str) -> str`` satisfies the protocol.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentInterface(Protocol):
    """Minimal interface every agent must satisfy.

    Attributes:
        agent_name: Human-readable name for the agent.
    """

    agent_name: str

    def run(self, input: str) -> str:
        """Execute the agent on *input* and return a
        response string."""
        ...
