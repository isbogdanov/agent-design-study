import json
import logging
import re
from typing import List, Dict, Any
from CybORG.Shared.Actions import Monitor, Sleep, Remove, Restore, Analyse


def format_blue_action_space(action_space: dict) -> dict:
    """
    Format the raw CybORG action space dictionary specifically for Blue agent actions.
    Uses sensible defaults for Blue agent.

    Args:
        action_space: The raw action space from CybORG

    Returns:
        A formatted dictionary with Blue actions and hostnames
    """
    formatted = {
        "actions": [],
        "hostnames": [],
        # Blue agent uses sensible defaults
        "session": 0,
        "agent": "Blue",
    }

    # Extract action names - focus on Blue actions
    blue_actions = ["Monitor", "Sleep", "Analyse", "Remove", "Restore"]
    decoy_actions = []

    if "action" in action_space:
        for action_cls in action_space["action"]:
            if hasattr(action_cls, "__name__"):
                action_name = action_cls.__name__
                # Categorize actions
                if action_name in blue_actions:
                    formatted["actions"].append(action_name)
                elif action_name.startswith("Decoy"):
                    decoy_actions.append(action_name)

    # Add decoy actions separately if needed
    if decoy_actions:
        formatted["decoy_actions"] = decoy_actions

    # Extract hostnames
    if "hostname" in action_space:
        # Handle both dictionary and list formats
        if isinstance(action_space["hostname"], dict):
            formatted["hostnames"] = list(action_space["hostname"].keys())
        else:
            formatted["hostnames"] = [str(h) for h in action_space["hostname"]]

    return formatted


def get_blue_actions(action_space: dict, include_decoys: bool = False) -> List[str]:
    """
    Get all possible Blue agent actions from the action space.

    Args:
        action_space: The raw action space from CybORG
        include_decoys: Whether to include decoy actions

    Returns:
        List of Blue agent actions in string format
    """
    # Format the action space for Blue agent
    formatted_space = format_blue_action_space(action_space)
    actions = formatted_space["actions"]
    hostnames = formatted_space["hostnames"]

    # Include decoys if requested
    if include_decoys and "decoy_actions" in formatted_space:
        actions.extend(formatted_space["decoy_actions"])

    # Generate all valid combinations
    action_combinations = []

    for action_name in actions:
        # Actions that don't require hostname
        if action_name in ["Monitor", "Sleep"]:
            action_combinations.append(action_name)
        # Actions that require hostname
        else:
            for hostname in hostnames:
                action_combinations.append(f"{action_name} hostname={hostname}")

    # print(f"Action combinations: {action_combinations}")
    return action_combinations


def parse_action_string(logger: logging.Logger, action_str: str) -> Dict[str, Any]:
    """
    Parse an action string into its components.

    Args:
        action_str: Action string (e.g., "Analyse hostname=\"User0\"")

    Returns:
        Dictionary containing action components
    """
    try:
        # Strip whitespace and surrounding quotes (handles LLM output variations)
        action_str = action_str.strip().strip('"').strip("'").strip()
        
        parts = action_str.split(" ", 1)
        action_name = parts[0].strip()
        result = {"action": action_name}

        # Use regex to find all key=value pairs
        if len(parts) > 1:
            param_str = parts[1].strip()
            # This regex handles quoted and unquoted values
            matches = re.findall(r'(\w+)="([^"]+)"|(\w+)=([\w\d_-]+)', param_str)
            for match in matches:
                # The regex returns a tuple of 4, but only 2 are non-empty
                # e.g., ('hostname', 'User0', '', '') or ('', '', 'hostname', 'User0')
                key = match[0] or match[2]
                value = match[1] or match[3]
                result[key] = value

        return result

    except Exception as e:
        logger.error(f"Error parsing action string: {e}")
        return {"action": "Monitor"}  # Fallback to Monitor


def validate_action_combination(
    action_combo: Dict[str, Any], action_space: dict
) -> bool:
    """
    Validate if an action combination is valid given the action space.

    Args:
        action_combo: The action combination to validate
        action_space: The action space

    Returns:
        True if valid, False otherwise
    """
    try:
        formatted_space = format_blue_action_space(action_space)

        # Check if action exists
        if action_combo["action"] not in formatted_space["actions"]:
            return False

        # Check if hostname is required and valid
        if action_combo["action"] not in ["Monitor", "Sleep"]:
            if "hostname" not in action_combo:
                return False

            if action_combo["hostname"] not in formatted_space["hostnames"]:
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating action combination: {e}")
        return False


if __name__ == "__main__":
    # Test with a simple mock action space
    sample_action_space = {
        "action": [
            type("Monitor", (), {"__name__": "Monitor"}),
            type("Sleep", (), {"__name__": "Sleep"}),
            type("Analyse", (), {"__name__": "Analyse"}),
            type("Remove", (), {"__name__": "Remove"}),
            type("Restore", (), {"__name__": "Restore"}),
            type("DecoyApache", (), {"__name__": "DecoyApache"}),
            type("DecoySSHD", (), {"__name__": "DecoySSHD"}),
        ],
        "hostname": {
            "User0": True,
            "User1": True,
            "Defender": True,
            "Enterprise0": True,
        },
        "session": {0: True},
        "agent": {"Blue": True},
    }

    # Test format_blue_action_space
    formatted = format_blue_action_space(sample_action_space)
    logging.info("Formatted action space:")
    logging.info(f"Actions: {formatted['actions']}")
    logging.info(f"Hostnames: {formatted['hostnames']}")
    if "decoy_actions" in formatted:
        logging.info(f"Decoy actions: {formatted['decoy_actions']}")

    # Test get_blue_actions
    actions = get_blue_actions(sample_action_space)
    logging.info("\nBlue actions (without decoys):")
    for action in actions:
        logging.info(f"  {action}")

    # Test with decoys included
    actions_with_decoys = get_blue_actions(sample_action_space, include_decoys=True)
    logging.info(f"\nBlue actions (with decoys): {len(actions_with_decoys)} total")

    # Test parse_action_string
    test_action = 'Analyse hostname="User0"'
    parsed = parse_action_string(logging.getLogger(), test_action)
    logging.info(f"\nParsed action: {parsed}")

    # Test validation
    valid = validate_action_combination(parsed, sample_action_space)
    logging.info(f"Action valid: {valid}")


def convert_action_string_to_object(
    logger, action_string: str, action_space: Dict[str, Any]
) -> Any:
    """
    Convert an action string to a CybORG Action object.

    Args:
        action_string: String representation of action (e.g., "Monitor", "Sleep", "Remove hostname=\"User0\"")

    Returns:
        CybORG Action object
    """
    # Parse the action string to extract components
    parsed = parse_action_string(logger, action_string)
    action_name = parsed.get("action")

    logger.debug(f"Parsed action string: {parsed}")
    sessions = list(action_space["session"].keys())
    session = sessions[0]
    logger.debug(f"Available sessions: {sessions}. Using session: {session}")

    # Create the appropriate Action object based on the action name
    action_name_lower = action_name.lower()
    if action_name_lower == "monitor":
        return Monitor(session=0, agent="Blue")
    elif action_name_lower == "sleep":
        return Sleep()
    elif action_name_lower == "analyse" or action_name_lower == "analyze":
        hostname = parsed.get("hostname")
        return Analyse(hostname=hostname, session=0, agent="Blue")
    elif action_name_lower == "remove":
        hostname = parsed.get("hostname")
        return Remove(hostname=hostname, session=0, agent="Blue")
    elif action_name_lower == "restore":
        hostname = parsed.get("hostname")
        return Restore(hostname=hostname, session=0, agent="Blue")
    else:
        # Handle decoy actions dynamically
        if "decoy" in action_name_lower:
            # Find the correct decoy action class from the action space
            for action_cls in action_space.get("action", []):
                if action_cls.__name__.lower() == action_name_lower:
                    hostname = parsed.get("hostname")
                    if hostname:
                        return action_cls(
                            hostname=hostname, session=session, agent="Blue"
                        )
                    else:
                        # Raise Error: If hostname is missing but required, raise ValueError
                        # so the coordinator can catch it and ask the agent to correct itself.
                        raise ValueError(
                            f"Action {action_name} selected without a hostname. "
                            f"This action likely requires a hostname (e.g., {action_name} hostname='User0')."
                        )

        # Default to Monitor check but raise error if clearly unknown to avoid silent failure 
        # However, for safety, if we really don't recognize it at all, Monitor is safer than crashing 
        # unless we are sure we want to force format. 
        # STICKING TO USER REQUEST: "agent should be asked to provide correct action".
        # So we raise ValueError for completely unknown actions too.
        logger.warning(f"Unknown action: {action_string}")
        raise ValueError(f"Unknown action: {action_string}. Please provide a valid Action string.")
