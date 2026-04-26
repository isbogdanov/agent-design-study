import logging
import yaml
import os
from typing import Dict, Any
from agents.tools.analyst import (
    get_host_baseline_state,
    get_host_current_state,
)
from base_classes.executor_base import BaseToolExecutor
from logs.config.log_config import get_context_logger
import utils.settings as settings

# Configure logging
logger = logging.getLogger("ToolExecutor")


class AnalystToolExecutor(BaseToolExecutor):

    def __init__(
        self,
        logger: logging.Logger,
        observation: Dict[str, Any],
        initial_observation: Dict[str, Any],
        dynamic_environment_model: Dict[str, Any],
        baseline_overrides: Dict[str, Any],
        topology,
        provider,
        provider_parameters,
        question_budget: int = 2,
    ):
        # Load configuration to determine enabled generic tools
        enabled_generic_tools = []
        config_path = os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/analyst/core.yaml")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                
                if config.get("include_tool_raise_a_question", False):
                    enabled_generic_tools.append("raise_a_question")
                if config.get("include_tool_critique_the_answer", False):
                    enabled_generic_tools.append("critique_the_answer")
                if config.get("include_tool_improve_based_on_critique", False):
                    enabled_generic_tools.append("improve_based_on_critique")
            else:
                # If config not found, fall back to None (enable all) or empty list?
                # Based on intent to fix "unconditionally registering", likely want to default to None if file missing to avoid breaking legacy?
                # But here we know the file structure.
                # If file missing, maybe we should log warning and default to None (all tools) to be safe.
                enabled_generic_tools = None
        except Exception as e:
            logger.warning(f"Failed to load analyst config: {e}")
            enabled_generic_tools = None

        super().__init__(
            logger,
            observation,
            initial_observation,
            dynamic_environment_model,
            provider,
            provider_parameters,
            question_budget=question_budget,
            enabled_generic_tools=enabled_generic_tools,
        )
        self.topology = topology
        self.baseline_overrides = baseline_overrides
        # Map of tool names to their execution methods
        analyst_tools = {
            "get_host_baseline_state": self._execute_get_host_baseline_state,
            "get_host_current_state": self._execute_get_host_current_state,
        }
        # Combine generic tools from base with specific tools
        self.tools.update(analyst_tools)

    def _execute_get_host_baseline_state(self, tool_input: str) -> str:
        return get_host_baseline_state(
            self.logger,
            tool_input,
            self.initial_observation,
            self.topology,
            self.baseline_overrides,
        )

    def _execute_get_host_current_state(self, tool_input: str) -> str:
        return get_host_current_state(
            self.logger,
            tool_input,
            self.observation,
            self.topology,
            self.dynamic_environment_model,
            self.initial_observation,
        )
