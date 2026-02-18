"""Lightweight LLM agent that calls litellm directly.

Replaces swarms.Agent to avoid conversation-history pollution
in the user message that triggers content filters.
"""

from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import litellm
from loguru import logger

load_dotenv()


class SimpleConversation:
    """Lightweight conversation log replacing swarms.Conversation.

    Stores (role, content) pairs and supports the same API surface
    used by AIScientistFramework: .add(), .conversation_history,
    and .return_history_as_string().
    """

    def __init__(self) -> None:
        self.conversation_history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {"role": role, "content": content}
        )

    def return_history_as_string(self) -> str:
        parts: List[str] = []
        for entry in self.conversation_history:
            parts.append(
                f"{entry['role']}: {entry['content']}"
            )
        return "\n\n".join(parts)


class DirectLLMAgent:
    """Minimal agent: system prompt + user task -> LLM response.

    Satisfies AgentInterface (agent_name + run method).
    Sends clean [system, user] messages to litellm — no history
    accumulation, no conversation pollution.
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model_name: str,
        llm_args: Optional[Dict[str, Any]] = None,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> None:
        self.agent_name = agent_name
        self._system_prompt = system_prompt
        self._model_name = model_name
        self._llm_args = llm_args or {}
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(self, input: str) -> str:
        """Call the LLM with system + user message and return text."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": input},
        ]
        params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        params.update(self._llm_args)

        try:
            response = litellm.completion(**params)
            content = response.choices[0].message.content
            finish = response.choices[0].finish_reason
            if finish == "refusal" or content is None:
                logger.warning(
                    f"Agent {self.agent_name}: LLM refused or "
                    f"returned no content "
                    f"(finish_reason={finish})"
                )
                return ""
            return content
        except Exception as e:
            logger.error(
                f"Agent {self.agent_name}: "
                f"LLM call failed: {e}"
            )
            return ""
