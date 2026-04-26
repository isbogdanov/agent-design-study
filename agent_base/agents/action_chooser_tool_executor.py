import logging
import yaml
import os
from typing import Dict, Any
from base_classes.executor_base import BaseToolExecutor
from logs.config.log_config import get_context_logger
import utils.settings as settings
import os


class ActionChooserToolExecutor(BaseToolExecutor):
    def __init__(
        self,
        parent_logger: logging.Logger,
        observation: Dict[str, Any],
        initial_observation: Dict[str, Any],
        dynamic_environment_model: Dict[str, Any],
        provider,
        provider_parameters,
        question_budget: int = 2,
    ):
        self.logger = get_context_logger(parent_logger, "Executor")
        
        # Load configuration to determine enabled generic tools
        enabled_generic_tools = []
        config_path = os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/core.yaml")
        
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
            self.logger.warning(f"Failed to load action_chooser config: {e}")
            enabled_generic_tools = None

        super().__init__(
            self.logger,
            observation,
            initial_observation,
            dynamic_environment_model,
            provider,
            provider_parameters,
            question_budget=question_budget,
            enabled_generic_tools=enabled_generic_tools,
        )
