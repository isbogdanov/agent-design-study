import logging
from typing import Dict, Any, List


def get_topology(logger: logging.Logger, observation: Dict[str, Any]) -> List[str]:
    hostnames = []
    if not isinstance(observation, dict):
        logger.warning(
            f"Input to extract_host_name is not a dictionary: {type(observation)}"
        )
        return []

    for key in observation.keys():
        if isinstance(key, str) and key.lower() != "success":
            hostnames.append(key)

    if not hostnames:
        logger.warning(f"Could not extract any hostnames: {observation.keys()}")

    return hostnames
