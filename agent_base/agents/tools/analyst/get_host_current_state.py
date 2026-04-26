import logging
from typing import Dict, Any, List
import json

from utils.helpers.data_serialization import to_json_serializable


def _summarize_host_state(host_data: dict) -> str:
    state = to_json_serializable(host_data)

    if "activity" in state and isinstance(state["activity"], list):
        activity_list = state["activity"]
        if len(activity_list) > 10:
            unique_patterns = {}
            for entry in activity_list:
                if " connected to " in entry:
                    parts = entry.split(" connected to ")
                    if len(parts) == 2:
                        target = parts[1].split(" on port ")[0]
                        if target not in unique_patterns:
                            unique_patterns[target] = {"count": 0, "ports": []}
                        unique_patterns[target]["count"] += 1
                        if " on port " in entry:
                            port = entry.split(" on port ")[1]
                            unique_patterns[target]["ports"].append(port)

            if unique_patterns:
                summary = []
                for target, data in unique_patterns.items():
                    port_range = (
                        f"{min(data['ports'])}-{max(data['ports'])}"
                        if len(data["ports"]) > 1
                        else data["ports"][0]
                    )
                    summary.append(
                        f"{data['count']} connections to {target} (ports: {port_range})"
                    )
                state["activity"] = summary + [
                    "(Activity summarized - extensive port scanning detected)"
                ]

    return json.dumps(state, indent=2)


def get_host_current_state(
    logger: logging.Logger,
    tool_input_query: str,
    observation: Dict[str, Any],
    topology: List[str],
    dynamic_environment_model: Dict[str, Any],
    initial_observation: Dict[str, Any],
) -> str:
    hostname = tool_input_query

    logger.info(f"Tool 'get_host_current_state' called for hostname: '{hostname}'")
    logger.info(
        f"Available keys in initial_observation: {list(initial_observation.keys())}"
    )
    logger.info(
        f"Available keys in dynamic_environment_model: {list(dynamic_environment_model.keys())}"
    )

    if hostname in observation and observation[hostname]:
        logger.info(f"Found state for '{hostname}' in the current observation.")
        return _summarize_host_state(observation[hostname])

    if hostname in dynamic_environment_model:
        host_model = dynamic_environment_model[hostname]
        status = host_model.get("status")
        history = host_model.get("history", {})

        if status == "unknown":
            return f"Host '{hostname}''s current state is unknown."

        if status == "changed" and history:
            latest_step = max(history.keys())
            latest_update = history[latest_step].get("update")
            if latest_update:
                logger.info(
                    f"Found state for '{hostname}' in dynamic model (status: changed, step: {latest_step})."
                )
                return f"Current state obtained at step {latest_step}:\n{_summarize_host_state(latest_update)}"

    if hostname in initial_observation:
        logger.info(f"Found state for '{hostname}' in initial observation (fallback).")
        return f"No evidence of change from the baseline is available at this time.\n{_summarize_host_state(initial_observation[hostname])}"

    logger.warning(
        f"No state found for the given hostname '{hostname}' in any available source."
    )
    return f"No current state found for the given hostname '{hostname}'"
