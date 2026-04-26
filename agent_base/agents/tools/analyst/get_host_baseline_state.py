import logging
from typing import Dict, Any, List
from utils.helpers.data_serialization import to_json_serializable
import copy
import json


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
                    port_range = f"{min(data['ports'])}-{max(data['ports'])}" if len(data['ports']) > 1 else data['ports'][0]
                    summary.append(f"{data['count']} connections to {target} (ports: {port_range})")
                state["activity"] = summary + ["(Activity summarized - extensive port scanning detected)"]
    
    return json.dumps(state, indent=2)


def get_host_baseline_state(
    logger: logging.Logger,
    tool_input_query: str,
    initial_observation: Dict[str, Any],
    topology: List[str],
    baseline_overrides: Dict[str, Any],
) -> str:
    hostname = tool_input_query

    if hostname not in initial_observation:
        return f"No baseline state found for the given hostname '{hostname}'"

    effective_baseline = copy.deepcopy(initial_observation[hostname])

    if hostname in baseline_overrides:
        overrides = baseline_overrides[hostname]
        if "Processes" in overrides:
            if "Processes" not in effective_baseline:
                effective_baseline["Processes"] = []
            effective_baseline["Processes"].extend(overrides["Processes"])
            logger.info(
                f"Applied {len(overrides['Processes'])} process overrides to baseline for {hostname}."
            )

    return _summarize_host_state(effective_baseline)
