import yaml
from utils.PTE.builder import PromptBuilder
import os
from coordinators.utils.knowledge_loader import get_tactical_knowledge

# Load the base agent definition
from utils.settings import AGENT_BASE_DIR
with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/core.yaml"), "r") as f:
    agent_definition = yaml.safe_load(f)

# Load common knowledge if it exists
common_knowledge = []
common_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/common_knowledge.yaml")
if os.path.exists(common_knowledge_path):
    with open(common_knowledge_path, "r") as f:
        common_knowledge_data = yaml.safe_load(f)
        common_knowledge = (
            common_knowledge_data.get("reflection_knowledge", [])
            if common_knowledge_data
            else []
        )

# Ensure common_knowledge is a list, defaulting to an empty list if it's None
if common_knowledge is None:
    common_knowledge = []

# Load agent-specific persistent knowledge
persistent_knowledge = []
persistent_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/persistent_knowledge.yaml")
if os.path.exists(persistent_knowledge_path):
    with open(persistent_knowledge_path, "r") as f:
        persistent_knowledge_data = yaml.safe_load(f)
        if (
            persistent_knowledge_data
            and "reflection_knowledge" in persistent_knowledge_data
        ):
            persistent_knowledge = (
                persistent_knowledge_data.get("reflection_knowledge", []) or []
            )

# Load and merge agent-specific reflection knowledge if it exists
agent_specific_knowledge = []
reflection_knowledge_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/reflection_knowledge.yaml")
if os.path.exists(reflection_knowledge_path):
    with open(reflection_knowledge_path, "r") as f:
        specific_knowledge_data = yaml.safe_load(f)
        if (
            specific_knowledge_data
            and "reflection_knowledge" in specific_knowledge_data
        ):
            agent_specific_knowledge = (
                specific_knowledge_data.get("reflection_knowledge", []) or []
            )

# Combine knowledge and update the definition
combined_knowledge = common_knowledge + persistent_knowledge + agent_specific_knowledge
if combined_knowledge:
    agent_definition["reflection_knowledge"] = combined_knowledge

# Load and merge examples only if the 'add_examples' flag is true
if agent_definition.get("add_examples", False):
    with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/examples.yaml"), "r") as f:
        examples_definition = yaml.safe_load(f)

    # Merge the examples into the main definition
    if examples_definition and "examples" in examples_definition:
        agent_definition.setdefault("examples", []).extend(
            examples_definition["examples"]
        )

# Load and merge reflection examples if the 'add_reflection_examples' flag is true
if agent_definition.get("add_reflection_examples", False):
    reflection_examples_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/reflection_examples.yaml")
    if os.path.exists(reflection_examples_path):
        with open(reflection_examples_path, "r") as f:
            reflection_examples_data = yaml.safe_load(f)
        
        if reflection_examples_data and reflection_examples_data.get("examples"):
             agent_definition.setdefault("examples", []).extend(
                reflection_examples_data["examples"]
            )

# The PromptBuilder needs a path, so we'll pass the original and then
# overwrite its internal template with our merged one.
_builder = PromptBuilder(
    agent_definition_path=os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/core.yaml")
)
_builder.agent_template = agent_definition


# Build the prompt, injecting the valid actions at build time.
# The `valid_actions` key is specific to the action_chooser.yaml and will be
# substituted into the system_message.
ACTION_AGENT_SYSTEM_PROMPT = _builder.build()


def get_dynamic_action_chooser_system_prompt() -> str:
    # This function rebuilds the prompt from scratch to include tactical knowledge
    # inside the reflection_knowledge block, rather than appending it externally.

    # Start with a fresh copy of the base agent definition
    # 1. Start with base core.yaml
    with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/core.yaml"), "r") as f:
        dynamic_agent_def = yaml.safe_load(f)

    # 2. Add Knowledge
    current_agent_specific_knowledge = []
    if os.path.exists(reflection_knowledge_path):
        with open(reflection_knowledge_path, "r") as f:
            specific_knowledge_data = yaml.safe_load(f)
            if (
                specific_knowledge_data
                and "reflection_knowledge" in specific_knowledge_data
            ):
                current_agent_specific_knowledge = (
                    specific_knowledge_data.get("reflection_knowledge", []) or []
                )

    current_combined_knowledge = common_knowledge + persistent_knowledge + current_agent_specific_knowledge
    tactical_knowledge = get_tactical_knowledge("action_chooser")
    full_knowledge = (current_combined_knowledge or []) + (tactical_knowledge or [])

    if full_knowledge:
        dynamic_agent_def["reflection_knowledge"] = full_knowledge

    # 3. Add Examples (Static + Dynamic)
    if dynamic_agent_def.get("add_examples", False):
        with open(os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/examples.yaml"), "r") as f:
            examples_definition = yaml.safe_load(f)
        if examples_definition and "examples" in examples_definition:
            dynamic_agent_def.setdefault("examples", []).extend(examples_definition["examples"])

    if dynamic_agent_def.get("add_reflection_examples", False):
        reflection_examples_path = os.path.join(AGENT_BASE_DIR, "agents/prompts/definitions/action_chooser/reflection_examples.yaml")
        if os.path.exists(reflection_examples_path):
            with open(reflection_examples_path, "r") as f:
                reflection_examples_data = yaml.safe_load(f)
            if reflection_examples_data and reflection_examples_data.get("examples"):
                dynamic_agent_def.setdefault("examples", []).extend(reflection_examples_data["examples"])

    # Use the same builder instance but temporarily overwrite its template
    _builder.agent_template = dynamic_agent_def
    prompt = _builder.build()
    _builder.agent_template = agent_definition  # Restore original template

    return prompt
