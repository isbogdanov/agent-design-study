import json
from enum import Enum
from ipaddress import IPv4Address, IPv4Network


def recursively_serialize(data):
    """
    Recursively iterates through a data structure and converts non-serializable
    objects (like enums and IP addresses) into strings.
    """
    if isinstance(data, dict):
        return {key: recursively_serialize(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursively_serialize(item) for item in data]
    elif isinstance(data, Enum):
        return data.name
    elif isinstance(data, (IPv4Address, IPv4Network)):
        return str(data)
    # Add other type conversions as needed
    return data


def to_json_serializable(observation: dict) -> dict:
    """
    Converts a CybORG observation dictionary into a JSON-serializable format.
    """
    if not isinstance(observation, dict):
        return {}
    return recursively_serialize(observation)
