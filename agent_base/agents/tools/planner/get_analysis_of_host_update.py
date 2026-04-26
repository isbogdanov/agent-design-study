import logging
import json
import re
import os
from typing import Dict, Any, List
import yaml

from utils.helpers.data_serialization import to_json_serializable

from base_classes.react_agent_base import ReActAgent
from agents.analyst_tool_executor import AnalystToolExecutor
from agents.prompts.analyst import get_dynamic_analyst_system_prompt


from utils.settings import ANALYST_PROVIDER_PARAMETERS, AGENT_BASE_DIR

# Regular expression for parsing actions
TOOL_INVOCATION_PATTERN = re.compile(r"^Tool: (\w+)(?::\s*(.*))?$", re.MULTILINE)

ANALYST_QUESTION_BUDGET = 2
ANALYST_MAX_TURNS = 5


def _create_initial_prompt(hostname: str, analyses_history: Dict[int, str]) -> str:
    # Load and format the initial prompt from the YAML file
    with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/analyst/initial_prompt.yaml"), "r") as f:
        prompt_data = yaml.safe_load(f).get("prompt", {})
        prompt_opening = prompt_data.get("opening", "")
        prompt_closing = prompt_data.get("closing", "")

    # Format the opening with the hostname
    prompt_opening = prompt_opening.format(hostname=hostname)

    # Add analyses history if available
    analyses_section = ""
    if analyses_history:
        analyses_section = f"Step History: {json.dumps(analyses_history, indent=2)}"

    # Construct the full prompt via concatenation
    return f"{prompt_opening}\n\n{analyses_section}\n\n{prompt_closing}"


def get_analysis_of_host_update(
    logger: logging.Logger,
    hostname: str,
    observation: Dict[str, Any],
    initial_observation: Dict[str, Any],
    dynamic_environment_model: Dict[str, Any],
    topology: List[str],
    provider: tuple[str, str],
    provider_parameters: dict,
    episode_step: int,
    analyses_history: Dict[int, str],
    baseline_overrides: Dict[str, Any],
    episode_memory,
    reward: float,
) -> str:
    agent = ReActAgent(
        get_dynamic_analyst_system_prompt(),
        provider,
        provider_parameters,
        logger,
        log_name=logger.name,
        episode_step=episode_step,
        max_turns=ANALYST_MAX_TURNS,
        episode_memory=episode_memory,
        previous_step_reward=reward,
        log_reward=True,
    )

    analyst_executor = AnalystToolExecutor(
        logger,
        observation,
        initial_observation,
        dynamic_environment_model,
        baseline_overrides,
        topology,
        provider,
        provider_parameters,
        question_budget=ANALYST_QUESTION_BUDGET,
    )

    initial_prompt = _create_initial_prompt(hostname, analyses_history)

    # Run the agent's reasoning loop
    final_analysis = agent.run(initial_prompt, analyst_executor, finalize=False)

    return final_analysis
