import os
import sys
import datetime

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import the fully constituted prompt constants
from agents.prompts.planner import PLANNER_SYSTEM_PROMPT
from agents.prompts.analyst import ANALYST_SYSTEM_PROMPT
from agents.prompts.action_chooser import ACTION_AGENT_SYSTEM_PROMPT


def generate_prompts():
    """
    Generates the full text of each agent's system prompt and saves them
    to a timestamped log directory for debugging and analysis.
    """
    # Create a timestamped directory for this generation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("agents", "prompts", "logs", f"prompt_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    prompts_to_generate = {
        "planner_prompt.log": PLANNER_SYSTEM_PROMPT,
        "analyst_prompt.log": ANALYST_SYSTEM_PROMPT,
        "action_chooser_prompt.log": ACTION_AGENT_SYSTEM_PROMPT,
    }

    print(f"Generating full prompts in: {log_dir}")

    for filename, content in prompts_to_generate.items():
        output_content = content
        # The planner prompt is the only one with a runtime placeholder.
        # Use .replace() for safety, to avoid KeyError from JSON examples.
        if filename == "planner_prompt.log":
            output_content = content.replace(
                "{history}", "<RUNTIME_HISTORY_WILL_BE_INSERTED_HERE>"
            )

        file_path = os.path.join(log_dir, filename)
        with open(file_path, "w") as f:
            f.write(output_content)
        print(f"  - Wrote {filename}")

    print("\nPrompt generation complete.")


if __name__ == "__main__":
    generate_prompts()
