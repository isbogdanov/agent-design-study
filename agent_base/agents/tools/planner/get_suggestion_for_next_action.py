import logging
import json
import re
import os
from typing import Dict, Any, List
import yaml
from llm_connector import chat_completion

from agents.action_chooser_tool_executor import ActionChooserToolExecutor
from base_classes.react_agent_base import ReActAgent

# This tool should not know about the prompt's internal configuration.
# The import will be removed.

from agents.prompts.action_chooser import get_dynamic_action_chooser_system_prompt
from utils.settings import ACTION_CHOOSER_PARAMETERS, AGENT_BASE_DIR


def _create_initial_prompt(
    reasoning: str,
    analyses_history: Dict[int, str],
) -> str:
    # Load the prompt template from the YAML file
    with open(
        os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/initial_prompt.yaml"), "r"
    ) as f:
        prompt_template = yaml.safe_load(f).get("prompt", "")

    # Dynamically construct the context sections
    analyses_section = ""
    if analyses_history:
        analyses_section = f"Step History: {json.dumps(analyses_history, indent=2)}"

    # Format the template with the situational understanding
    full_prompt = prompt_template.format(situational_understanding=reasoning)

    # Insert the analyses section if present
    if analyses_section:
        full_prompt = f"{full_prompt}\n\n{analyses_section}"

    return full_prompt


def strip_markdown_code_blocks(response_str: str) -> str:
    """Strips markdown code blocks from a string."""
    match = re.search(r"```(json)?\s*(.*?)\s*```", response_str, re.DOTALL)
    if match:
        return match.group(2)
    return response_str


def extract_json_from_response(response: str) -> list:
    """
    Extract JSON array from LLM response, handling various formats:
    - Direct JSON array
    - Answer: [JSON]
    - Answer:\n```json\n[JSON]\n```
    - Thought:...\nAnswer: [JSON]
    
    Returns the parsed list of suggestions.
    Raises ValueError if no valid JSON array can be extracted.
    """
    candidate = response
    
    # Step 1: Look for "Answer:" prefix and extract everything after it
    answer_match = re.search(r'Answer:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).strip()
    
    # Step 2: Strip markdown code blocks (handles ```json...```)
    candidate = strip_markdown_code_blocks(candidate)
    
    # Step 3: Try to parse directly
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
        # Handle dict wrapper formats
        if isinstance(parsed, dict):
            for key in ["actions", "action_suggestions", "suggestions"]:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass
    
    # Step 4: Fallback - search for JSON array pattern in the candidate
    json_match = re.search(r'\[[\s\S]*\]', candidate)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Step 5: Last resort - search in original response (in case Answer: extraction failed)
    if candidate != response:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
    
    raise ValueError("No valid JSON array found in response")


def get_suggestion_for_next_action(
    logger: logging.Logger,
    reasoning: str,
    valid_actions: List[str],
    provider: tuple[str, str],
    provider_parameters,
    observation: Dict[str, Any],
    initial_observation: Dict[str, Any],
    dynamic_environment_model: Dict[str, Any],
    episode_step: int,
    analyses_history: Dict[int, str],
    reward: float,
    episode_memory,
) -> str:
    # 1. Create the Agent
    # Append the dynamic valid actions to the static system prompt
    base_prompt = get_dynamic_action_chooser_system_prompt()
    system_prompt_with_actions = (
        f"{base_prompt}\n\n"
        f"<AVAILABLE_ACTIONS_LIST>\n{json.dumps(valid_actions, indent=2)}</AVAILABLE_ACTIONS_LIST>"
    )
    agent = ReActAgent(
        system_prompt_with_actions,
        provider,
        provider_parameters,
        logger,
        log_name=logger.name,
        episode_step=episode_step,
        max_turns=5,  # Action chooser should be quick
        previous_step_reward=reward,
        log_reward=True,
        episode_memory=episode_memory,
    )

    # 2. Create the Executor
    executor = ActionChooserToolExecutor(
        parent_logger=logger,
        observation=observation,
        initial_observation=initial_observation,
        dynamic_environment_model=dynamic_environment_model,
        provider=provider,
        provider_parameters=provider_parameters,
        question_budget=2,
    )

    # 3. Construct the Initial Prompt
    initial_prompt = _create_initial_prompt(reasoning, analyses_history)

    # 4. Run the Agent
    final_answer = agent.run(initial_prompt, executor, finalize=False)

    # 5. Validate and Repair JSON
    try:
        # Use robust extraction that handles Answer: prefix, markdown, and regex fallback
        suggestions = extract_json_from_response(final_answer)

        # Validate it's actually a list before iterating
        if not isinstance(suggestions, list):
            raise ValueError(f"Expected list of suggestions, got {type(suggestions).__name__}")

        # Validate hostname consistency if structured input was provided
        try:
            situation_data = json.loads(reasoning)
            target_host = situation_data.get("target_host")
            if target_host:
                # Check that all suggested actions target the specified host
                for suggestion in suggestions:
                    action_str = suggestion.get("action", "")
                    if "hostname=" in action_str:
                        action_host = action_str.split("hostname=")[1].split()[0]
                        if action_host != target_host:
                            logger.warning(
                                f"HOSTNAME MISMATCH: ActionChooser suggested '{action_str}' but target_host was '{target_host}'. "
                                f"This is a critical error in following instructions."
                            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Reasoning wasn't JSON or missing fields - skip validation (backward compatibility)
            pass

        return json.dumps(suggestions, indent=2)

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(
            f"Failed to parse suggestions from LLM on the first attempt: {e}\nResponse was:\n{final_answer}"
        )
        # Fallback to JSON repair logic with strict instructions
        json_formatter_system_prompt = (
            "You are a JSON extractor. Extract action suggestions from the input and output ONLY valid JSON.\n\n"
            "STRICT RULES:\n"
            "1. NO MARKDOWN - Do not wrap in ```json blocks\n"
            "2. NO COMMENTS - JSON does not support // or /* */\n"
            "3. NO EXPLANATIONS - Output ONLY the JSON array, nothing else\n\n"
            "REQUIRED FORMAT:\n"
            '[{"action": "ActionName hostname=HostName", "confidence": 0.X}, ...]\n\n'
            "If multiple suggestions exist, include them all. If unclear, default to:\n"
            '[{"action": "Monitor", "confidence": 0.5}]'
        )
        user_message = f"Extract the action suggestions from this response as a JSON array:\n\n{final_answer}"

        repaired_response, _, _, _, _ = chat_completion(
            messages=[
                {"role": "system", "content": json_formatter_system_prompt},
                {"role": "user", "content": user_message},
            ],
            provider=provider,
            **ACTION_CHOOSER_PARAMETERS,
        )
        try:
            suggestions = json.loads(repaired_response)
            return json.dumps(suggestions, indent=2)
        except (json.JSONDecodeError, ValueError):
            logger.error("JSON repair failed. Falling back to default action.")
            return json.dumps([{"action": "Monitor", "confidence": 0.1}], indent=2)
