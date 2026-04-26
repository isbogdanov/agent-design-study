import logging
import yaml
import os
from typing import Dict, Any, List
from agents.tools.planner import (
    get_analysis_of_host_update,
    get_suggestion_for_next_action,
    get_updated_hosts,
)
from base_classes.executor_base import BaseToolExecutor
from utils.helpers.action_processing import get_blue_actions

from logs.config.log_config import get_context_logger
import utils.settings as settings


class PlannerToolExecutor(BaseToolExecutor):
    def __init__(
        self,
        parent_logger: logging.Logger,
        observation: Dict[str, Any],
        initial_observation: Dict[str, Any],
        dynamic_environment_model: Dict[str, Any],
        baseline_overrides: Dict[str, Any],
        action_space: Dict[str, Any],
        episode_memory,
        topology: List[str],
        provider,
        provider_parameters,
        question_budget: int = 2,
        dynamic_actions: bool = False,
        reward: float = 0.0,
    ):
        self.logger = get_context_logger(parent_logger, "Executor")
        
        # Load configuration to determine enabled generic tools
        enabled_generic_tools = []
        config_path = os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/planner/core.yaml")
        
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
                enabled_generic_tools = None
        except Exception as e:
            self.logger.warning(f"Failed to load planner config: {e}")
            enabled_generic_tools = None

        super().__init__(
            self.logger,
            observation,
            initial_observation,
            dynamic_environment_model,
            provider,
            provider_parameters,
            action_space,
            question_budget=question_budget,
            enabled_generic_tools=enabled_generic_tools,
        )
        self.episode_memory = episode_memory
        self.topology = topology
        self.baseline_overrides = baseline_overrides
        self.dynamic_actions = dynamic_actions
        self.reward = reward
        # Map of tool names to their execution methods
        planner_tools = {
            "get_analysis_of_host_update": self._execute_get_analysis_of_host_update,
            "get_suggestion_for_next_action": self._execute_get_suggestion_for_next_action,
            "get_updated_hosts": self._execute_get_updated_hosts,
        }
        self.tools.update(planner_tools)

    def _execute_get_analysis_of_host_update(self, tool_input_query: str) -> str:
        # tool_input_query is the hostname string from the LLM
        analyst_logger = get_context_logger(self.logger, "Analyst")
        return get_analysis_of_host_update(
            logger=analyst_logger,
            hostname=tool_input_query,
            observation=self.observation,
            initial_observation=self.initial_observation,
            dynamic_environment_model=self.dynamic_environment_model,
            topology=self.topology,
            provider=self.provider,
            provider_parameters=self.provider_parameters,
            episode_step=self.episode_memory.episode_step,
            analyses_history=self.episode_memory.analyses_received,
            baseline_overrides=self.baseline_overrides,
            episode_memory=self.episode_memory,
            reward=self.reward,
        )

    def _execute_get_suggestion_for_next_action(
        self, context_and_actions_str: str
    ) -> str:
        valid_actions = []
        if self.dynamic_actions:
            # Get the list of valid actions using action_processing utility
            valid_actions = get_blue_actions(self.action_space, include_decoys=True)
        else:
            # Fallback to a default or placeholder if not dynamic
            pass

        # Pass the valid actions list and the context string to choose_action
        action_chooser_logger = get_context_logger(self.logger, "ActionChooser")
        return get_suggestion_for_next_action(
            action_chooser_logger,
            context_and_actions_str,
            valid_actions,
            self.provider,
            self.provider_parameters,
            self.observation,
            self.initial_observation,
            self.dynamic_environment_model,
            self.episode_memory.episode_step,
            self.episode_memory.analyses_received,
            reward=self.reward,
            episode_memory=self.episode_memory,
        )

    def _execute_get_updated_hosts(self, tool_input_query: str) -> str:
        # tool_input_query is the string from the LLM like "Get the updated hosts..."
        updated_hosts_logger = get_context_logger(self.logger, "UpdatedHosts")
        return get_updated_hosts(
            updated_hosts_logger,
            self.observation,
            self.dynamic_environment_model,
        )
