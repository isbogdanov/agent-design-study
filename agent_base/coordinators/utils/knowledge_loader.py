import os
import time
import yaml
from typing import List, Dict, Any

_tactical_knowledge_cache: Dict[str, Dict[str, Any]] = {}


def get_tactical_knowledge(agent_name: str) -> List[str]:
    knowledge_path = (
        f"agents/prompts/definitions/{agent_name.lower()}/tactical_knowledge.yaml"
    )

    if not os.path.exists(knowledge_path):
        return []

    try:
        current_mtime = os.path.getmtime(knowledge_path)
        agent_cache = _tactical_knowledge_cache.get(agent_name)

        if agent_cache and agent_cache["timestamp"] == current_mtime:
            return agent_cache["knowledge"]

        # File is new or has been modified, so we need to read it
        knowledge = _load_tactical_yaml_safe(knowledge_path)
        _tactical_knowledge_cache[agent_name] = {
            "knowledge": knowledge,
            "timestamp": current_mtime,
        }
        return knowledge

    except FileNotFoundError:
        return []


def _load_tactical_yaml_safe(yaml_path: str, max_retries: int = 3) -> List[str]:
    last_exception = None
    for attempt in range(max_retries):
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if not data:
                    return []
                knowledge = data.get("reflection_knowledge")
                return knowledge if knowledge is not None else []
        except (yaml.YAMLError, IOError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(0.05)  # Brief wait for file lock to be released
            else:
                # Log this error? For now, we fail silently.
                print(
                    f"Warning: Could not load tactical knowledge from {yaml_path} after {max_retries} attempts. Last error: {last_exception}"
                )
                return []
    return []
