#from utils.llm_connector.connector.connector_settings import (
#    DEFAULT_PROVIDER,
#    DEFAULT_MODEL,
#)

import os

# Define the base directory of the agent (agent_base folder)
AGENT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROVIDER = "openrouter"
# DEFAULT_MODEL = "qwen/qwen-plus-2025-07-28"
# DEFAULT_MODEL = "qwen/qwen-plus-2025-07-28"

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# DEFAULT_MODEL = "meta-llama/llama-4-maverick"

# DEFAULT_PROVIDER = "google"
# DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"

# DEFAULT_MODEL = "openai/gpt-oss-20b"

# DEFAULT_MODEL = "meta-llama/llama-4-scout"

# DEFAULT_MODEL = "google/gemini-flash-1.5-8b"

# DEFAULT_MODEL = "mistralai/ministral-3b"

# DEFAULT_MODEL = "meta-llama/llama-4-maverick"

# DEFAULT_PROVIDER = "ollama"
# DEFAULT_MODEL = "llama3.1:8b"


PROVIDER = (DEFAULT_PROVIDER, DEFAULT_MODEL)


TOOLS_PROVIDER = PROVIDER


PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

ANALYST_PROVIDER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

ACTION_CHOOSER_PARAMETERS = {"temperature": 0, "max_tokens": 10000, "top_p": 0.7}

# Configuration for the Reflector/Exemplifier has been removed.
# This codebase runs in evaluation-only mode.

def update_provider_settings(provider_name, model_name):
    global PROVIDER, TOOLS_PROVIDER
    new_provider = (provider_name, model_name)
    PROVIDER = new_provider
    TOOLS_PROVIDER = new_provider


TOOLS_PROVIDER_PARAMETERS = {
    "temperature": 0,
    "max_tokens": 10000,
    "top_p": 0.7,
}

CONTEXT_LENGTH = 100000
