import logging
import re
from typing import Dict, Any, List
from dataclasses import dataclass, field
from llm_connector import chat_completion
import os
import json
from utils.settings import CONTEXT_LENGTH
from logs.config.log_config import get_current_run_dir


@dataclass
class AgentState:
    messages: List[Dict[str, str]]
    system_prompt: str


class ReActAgent:

    def __init__(
        self,
        system_prompt: str,
        llm_provider: str,
        llm_provider_parameters: dict,
        logger: logging.Logger,
        log_name: str,
        episode_step: int,
        max_turns: int = 5,
        previous_step_reward: float = 0.0,
        log_reward: bool = False,
        episode_memory=None,
        user_message_history_length: int = 10,
        assistant_message_history_length: int = 10,
        context_length: int = CONTEXT_LENGTH,
        trajectory_log_path: str = None,
    ):
        self.logger = logger
        self.max_turns = max_turns
        self.log_name = log_name
        self.episode_step = episode_step if episode_step > 0 else 1
        self.previous_step_reward = previous_step_reward
        self.log_reward = log_reward
        self.episode_memory = episode_memory
        self.run_dir = get_current_run_dir()
        if trajectory_log_path:
            self.trajectory_log_path = trajectory_log_path
        else:
            self.trajectory_log_path = os.path.join(
                self.run_dir, "trajectories", f"{self.log_name}.json"
            )
        os.makedirs(os.path.dirname(self.trajectory_log_path), exist_ok=True)

        self.state = AgentState(
            messages=[{"role": "system", "content": system_prompt}],
            system_prompt=system_prompt,
        )

        # Initialize the trajectory log file with the new structure
        if not os.path.exists(self.trajectory_log_path):
            initial_trajectory = {
                "agent_name": self.log_name,
                "system_prompt": self.state.system_prompt,
                "steps": {},
            }
            with open(self.trajectory_log_path, "w") as f:
                json.dump(initial_trajectory, f, indent=2)

        self.tool_invocation_pattern = re.compile(
            r"^Tool: ([^\n:]+)(?::\s*(.*))?", re.MULTILINE
        )
        self.llm_provider = llm_provider
        self.llm_provider_parameters = llm_provider_parameters
        self.user_message_history_length = user_message_history_length
        self.assistant_message_history_length = assistant_message_history_length
        self.context_length = context_length
        self.last_prompt_token_count = 0

        # Log the reward for the previous step if applicable
        if self.log_reward and self.episode_step > 1:
            self._log_reward_for_previous_step()

    def run(self, initial_prompt: str, executor, finalize: bool = False) -> str:
        next_prompt = initial_prompt
        final_answer = "No answer found."
        agent_name = self.logger.name.split(".")[-1]

        # If finalize=True, force exactly one iteration to finalize trajectory logging
        max_iterations = 1 if finalize else self.max_turns

        # Handle finalization mode - purely programmatic, no LLM calls
        if finalize:
            print("\n" + "#" * 50 + f" {agent_name} FINALIZING TRAJECTORY " + "#" * 50)
            print("-" * 50 + f"{agent_name}: Programmatic Finalization" + "-" * 50)
            print("Triggering reward logging for previous step...")
            print("-" * 10 + f"{agent_name}: Finalization Complete" + "-" * 10 + "\n")

            # The __init__ method already called _log_reward_for_previous_step()
            # so trajectory logging is complete. No LLM calls needed.
            self.logger.info("Trajectory finalization complete.")
            return "Trajectory finalized"

        # Normal mode with LLM calls
        for turn in range(max_iterations):
            print("\n" + "#" * 50 + f" {agent_name} Turn:{turn + 1} " + "#" * 50)

            result, prompt_tokens = self(next_prompt)

            # Guard against None responses from the LLM (API errors, rate limits, empty responses)
            if result is None:
                self.logger.warning("LLM returned None response. Treating as empty string.")
                result = ""

            print(
                "-" * 50 + f"{agent_name}: Thought Start for Turn:{turn + 1}" + "-" * 50
            )
            print(f"\n{result}\n")
            print(f"Context length: {prompt_tokens}")
            print("-" * 10 + f"{agent_name}: Thought End {turn + 1}" + "-" * 10 + "\n")

            # Check for a final answer first
            answer_match = re.search(r"Answer: (.*)", result, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                self.logger.info(f"Final answer found: {final_answer}")
                return final_answer

            # Check for tool calls
            tool_match = self.tool_invocation_pattern.search(result)
            if tool_match:
                tool_name, tool_input = tool_match.groups()

                # Handle cases where the LLM appends PAUSE to the tool name or input
                if tool_name.endswith("PAUSE"):
                    tool_name = tool_name[:-5]  # Strip "PAUSE"

                tool_input = tool_input.strip() if tool_input else ""
                if tool_input.endswith("PAUSE"):
                    tool_input = tool_input[
                        :-5
                    ].strip()  # Strip "PAUSE" and any trailing whitespace
                self.logger.info(f"Tool call: {tool_name} with input: '{tool_input}'")

                print("-" * 50 + f"Tool Call Start for Turn:{turn + 1}" + "-" * 50)

                try:
                    observation = executor(tool_name, tool_input)
                    next_prompt = f"Observation: {observation}"
                except ValueError as e:
                    self.logger.error(f"Tool execution error: {e}")
                    next_prompt = f"Observation: {e}"
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    next_prompt = (
                        f"Observation: An unexpected error occurred. Please try again."
                    )
                print(f"TOOL OUTPUT: {next_prompt}")
                print("-" * 10 + "Tool Call End" + "-" * 10 + "\n")

            else:
                self.logger.warning(
                    "No tool call or final answer found. Ending reasoning loop."
                )
                return result  # Fallback to the last raw output

            print("#" * 10 + f" End Of {agent_name} Turn:{turn + 1} " + "#" * 10)

        return final_answer

    def _log_reward_for_previous_step(self):
        """Finds the log entry for the previous step and appends the reward."""
        try:
            with open(self.trajectory_log_path, "r+") as f:
                trajectory = json.load(f)
                previous_step_key = str(self.episode_step - 1)

                # Find the previous step and check if this agent was active
                if previous_step_key in trajectory.get("steps", {}):
                    # Look up the correct reward from the central log
                    reward_to_log = self.episode_memory.rewards.get(
                        int(previous_step_key), 0.0
                    )
                    trajectory["steps"][previous_step_key]["reward"] = reward_to_log
                    f.seek(0)
                    f.truncate()
                    json.dump(trajectory, f, indent=2)
                else:
                    self.logger.debug(
                        f"Agent '{self.log_name}' was not active in step '{previous_step_key}', skipping reward logging."
                    )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            self.logger.error(
                "Failed to log reward for previous step due to file issue."
            )

    def _log_trajectory(self, message: Dict[str, str]):
        """Append the latest message to the trajectory log file, organized by step."""
        # Do not log the system prompt in the steps, as it's at the root level
        if message.get("role") == "system":
            return

        try:
            with open(self.trajectory_log_path, "r+") as f:
                trajectory = json.load(f)
                step_key = str(self.episode_step)

                # Ensure the 'steps' dictionary exists
                if "steps" not in trajectory:
                    trajectory["steps"] = {}

                # Ensure the entry for the current step exists
                if step_key not in trajectory["steps"]:
                    trajectory["steps"][step_key] = {"messages": []}

                # Now it's safe to append the message
                trajectory["steps"][step_key]["messages"].append(message)
                f.seek(0)
                f.truncate()
                json.dump(trajectory, f, indent=2)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # If file is corrupt or missing, re-initialize it
            step_content = {"messages": [message]}
            initial_trajectory = {
                "agent_name": self.log_name,
                "system_prompt": self.state.system_prompt,
                "steps": {str(self.episode_step): step_content},
            }
            with open(self.trajectory_log_path, "w") as f:
                json.dump(initial_trajectory, f, indent=2)

    def add_message(self, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self.state.messages.append(message)
        self._log_trajectory(message)

    def execute(self) -> str:
        if self.last_prompt_token_count > self.context_length:
            self.logger.debug(
                f"Last prompt token count ({self.last_prompt_token_count}) exceeded context length ({self.context_length}). Trimming history."
            )
            # Remove the oldest user/assistant message pair (after the system prompt)
            if len(self.state.messages) > 2:
                self.state.messages.pop(1)
                self.state.messages.pop(1)

        completion, prompt_tokens, _, _, _ = chat_completion(
            messages=self.state.messages,
            provider=self.llm_provider,
            **self.llm_provider_parameters,
        )
        self.last_prompt_token_count = prompt_tokens
        return (completion, prompt_tokens)

    def __call__(self, message: str) -> str:
        self.add_message("user", message)
        completion, prompt_tokens = self.execute()
        self.add_message("assistant", completion)
        return completion, prompt_tokens
