import logging
import json
from typing import Dict, Any, List
from utils.helpers.obs_processing import get_topology


def get_updated_hosts(
    logger: logging.Logger,
    observation: Dict[str, Any],
    dynamic_environment_model: Dict[str, Any],
) -> str:
    updated_hosts_info = []

    # Check hosts from the current observation first
    hosts_in_current_obs = get_topology(logger, observation)

    for hostname in dynamic_environment_model:
        host_model = dynamic_environment_model[hostname]
        status = host_model.get("status")

        if status in ["changed", "unknown"]:
            # Format the action history list into a string
            action_list = host_model.get("applied_actions_so_far", [])
            action_history_str = "->".join(action_list)

            host_info = {
                "host_name": hostname,
                "current_status": status,
                "time_of_update": "Past",
                "applied_actions_so_far": action_history_str,
            }

            if hostname in hosts_in_current_obs:
                host_info["time_of_update"] = "Current"

            updated_hosts_info.append(host_info)

    if not updated_hosts_info:
        return "No hosts have been updated in this step."

    return json.dumps(updated_hosts_info, indent=2)
