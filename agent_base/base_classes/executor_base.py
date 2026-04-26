from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from agents.tools.auto.generic_tools import GenericMonologueTools


class BaseToolExecutor(ABC):
    """Abstract base class for tool executors."""

    def __init__(
        self,
        logger: logging.Logger,
        observation: Dict[str, Any] = None,
        initial_observation: Dict[str, Any] = None,
        dynamic_environment_model: Dict[str, Any] = None,
        provider=None,
        provider_parameters=None,
        action_space: Dict[str, Any] = None,
        question_budget: int = 2,
        enabled_generic_tools: List[str] = None,
    ):
        self.logger = logger
        self.observation = observation
        self.initial_observation = initial_observation
        self.dynamic_environment_model = dynamic_environment_model
        self.provider = provider
        self.provider_parameters = provider_parameters
        self.action_space = action_space
        self.tools: Dict[str, Any] = {}

        # Initialize and register generic tools
        self.generic_tools = GenericMonologueTools(question_budget)
        all_generic_tools = self.generic_tools.get_tools()

        if enabled_generic_tools is not None:
            # Filter tools based on the provided list
            filtered_tools = {
                name: tool
                for name, tool in all_generic_tools.items()
                if name in enabled_generic_tools
            }
            self.tools.update(filtered_tools)
        else:
            # Register all tools if no filter is provided (backward compatibility)
            self.tools.update(all_generic_tools)

    def __call__(self, tool_name: str, tool_input: str) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name](tool_input)
        else:
            if not self.tools:
                raise ValueError(
                    f"Unknown tool '{tool_name}'. There are no external tools available. You MUST provide your Answer in the correct format. Try again!"
                )
            available_tools = ", ".join(self.tools.keys())
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available tools: [{available_tools}]"
            )
