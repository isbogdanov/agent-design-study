import logging
import json
import re
import os
import shutil
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from base_classes.react_agent_base import ReActAgent

# from tools.agents.planner_agentic_tools import analyze_observation, choose_action
from agents.planner_tool_executor import PlannerToolExecutor
from datetime import datetime

from CybORG.Shared.Actions import Monitor, Sleep, Remove, Restore, Analyse
from utils.helpers.action_processing import (
    parse_action_string,
    convert_action_string_to_object,
)
from utils.helpers.obs_processing import get_topology

from agents.prompts.planner import get_dynamic_planner_system_prompt
from base_classes.react_agent_base import ReActAgent


def clean_agent_answer(answer: str) -> str:
    """Remove prefixes like 'Answer:', 'Thought:', and 'PAUSE' from the agent's final output."""
    if not isinstance(answer, str):
        return ""

    # Remove prefixes
    answer = answer.strip()
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:") :].strip()
    if answer.startswith("Thought:"):
        answer = answer[len("Thought:") :].strip()

    # Remove PAUSE keyword
    answer = answer.replace("PAUSE", "").strip()

    return answer


# from prompts.agents.planner.parts import OBSERVATION_PROMPT
import utils.settings as settings
from utils.settings import (
    PROVIDER_PARAMETERS,
    TOOLS_PROVIDER_PARAMETERS,
)
from logs.config.log_config import get_current_run_dir

import pprint
import yaml
import copy


@dataclass
class EpisodeMemory:
    """Represents the current state of the agent."""

    analyses_received: Dict[int, Dict[str, str]]
    messages: Dict[str, str]
    initial_prompt: str
    observations: Dict[int, str]  # Changed to store string representations
    rewards: Dict[int, float]
    episode_step: int


class CybORGAgentCoordinator(BaseAgent):
    """ReAct agent for cybersecurity that integrates with CybORG."""

    def __init__(self, log_summary: bool = True):
        super().__init__()
        self.log_summary = log_summary
        self.logger = logging.getLogger("Planner")
        self.logger.info(f"Initializing the agent, usingLLM: {settings.PROVIDER[1]}")
        self.planner = None
        self.episode_memory = EpisodeMemory(
            analyses_received={},
            messages={},
            initial_prompt="",
            observations={},
            rewards={},
            episode_step=0,
        )
        self.episode_step = 1
        self.current_episode = 0  # Track current episode number
        self.action_space = None
        self.initial_observation = None
        self._previous_observation = None
        self.topology = None
        self.dynamic_environment_model = {}
        self.baseline_overrides = {}
        self.local_step_history_lookup_size = 0
        self.local_analyses_history_lookup_size = 0
        self.question_budget = 3

    def set_initial_values(self, action_space, observation):

        self.action_space = action_space
        self.initial_observation = observation
        self._previous_observation = observation
        self.topology = get_topology(self.logger, observation)
        for hostname in self.topology:
            self.dynamic_environment_model[hostname] = {
                "status": "baseline",
                "history": {},
                "applied_actions_so_far": [],
            }
        self.logger.debug(f"Topology: {self.topology}")
        self.logger.debug(f"EnvModel: {self.dynamic_environment_model}")
        self.logger.info(f"Topology and Environment Model set.\n\n")

        # Clear trajectory logs from previous runs at the start of a new session
        from logs.config.log_config import get_current_run_dir

        run_dir = get_current_run_dir()
        trajectories_dir = os.path.join(run_dir, "trajectories")
        if not os.path.exists(trajectories_dir):
            os.makedirs(trajectories_dir)

    def _is_host_state_baseline(
        self, current_state: dict, effective_baseline: dict
    ) -> bool:
        # 1. Compare keys (excluding 'Connections' if present at top level)
        curr_keys = set(current_state.keys()) - {"Connections"}
        base_keys = set(effective_baseline.keys()) - {"Connections"}

        if curr_keys != base_keys:
            return False

        # 2. Compare values
        for key in curr_keys:
            val_curr = current_state[key]
            val_base = effective_baseline.get(key)

            # Special handling for 'Processes' list
            if (
                key == "Processes"
                and isinstance(val_curr, list)
                and isinstance(val_base, list)
            ):

                def get_process_signature(proc_item):
                    # Create a signature that ignores Connections and handles dicts
                    if not isinstance(proc_item, dict):
                        return str(proc_item)

                    # If this item is ONLY a connections container (no PID/Service), ignore it
                    # (Based on the log showing {'Connections': ...} as separate items)
                    if (
                        "Connections" in proc_item
                        and "PID" not in proc_item
                        and "Service Name" not in proc_item
                    ):
                        return None

                    # Make a copy to strip volatile fields
                    item_clean = proc_item.copy()
                    item_clean.pop("Connections", None)

                    # Return a sorted JSON string for deterministic comparison
                    return json.dumps(item_clean, sort_keys=True)

                # Transform both lists to multisets of signatures
                curr_sigs = []
                for p in val_curr:
                    sig = get_process_signature(p)
                    if sig is not None:
                        curr_sigs.append(sig)

                base_sigs = []
                for p in val_base:
                    sig = get_process_signature(p)
                    if sig is not None:
                        base_sigs.append(sig)

                # Compare sorted signatures (multiset equality)
                if sorted(curr_sigs) != sorted(base_sigs):
                    return False

            # Default strict comparison for other fields
            elif val_curr != val_base:
                return False

        return True

    def _update_dynamic_environment_model(self, current_observation):
        if self.episode_step == 1:
            return

        previous_action_string = (
            self.episode_memory.analyses_received.get(self.episode_step - 1, {}).get(
                "action"
            )
            if self.episode_memory.analyses_received
            else None
        )

        for hostname in self.topology:
            # First, handle decoy updates based on the previous action
            if previous_action_string and "Decoy" in previous_action_string:
                action_parts = parse_action_string(self.logger, previous_action_string)
                if action_parts.get("hostname") == hostname:
                    self._update_baseline_with_decoy(
                        hostname, self._previous_observation, current_observation
                    )

            current_host_state = current_observation.get(hostname, {})

            # Construct an "effective" previous state by applying overrides
            effective_previous_state = copy.deepcopy(
                self._previous_observation.get(hostname, {})
            )
            if hostname in self.baseline_overrides:
                overrides = self.baseline_overrides[hostname]
                if "Processes" in overrides:
                    if "Processes" not in effective_previous_state:
                        effective_previous_state["Processes"] = []
                    # Get PIDs of existing effective processes to avoid duplicates
                    effective_pids = {
                        p.get("PID")
                        for p in effective_previous_state.get("Processes", [])
                    }
                    for override_proc in overrides.get("Processes", []):
                        if override_proc.get("PID") not in effective_pids:
                            effective_previous_state["Processes"].append(override_proc)

            # An update is only real if the current observation contains the host
            # Use strict inequality here to detect ANY change for history logging purposes?
            # Or use robust equality?
            # If we use strict, we log "update" even if it's just connections.
            # If we use robust, we only log "update" if significant stuff changed.
            # Let's stick to strict for history, but robust for STATUS.
            if (
                hostname in current_observation
                and current_host_state != effective_previous_state
            ):
                # A change was detected, so we must record it
                host_model = self.dynamic_environment_model.get(
                    hostname, {"status": "baseline", "history": {}}
                )
                current_history = host_model["history"]

                action_for_this_host = None
                if previous_action_string:
                    action_parts = parse_action_string(
                        self.logger, previous_action_string
                    )
                    if action_parts.get("hostname") == hostname:
                        action_for_this_host = previous_action_string

                current_history[self.episode_step - 1] = {
                    "action": action_for_this_host,
                    "update": current_host_state,
                }

                # Now, determine the NEW status based on the CURRENT state
                # by comparing against an effective baseline that includes our overrides
                effective_baseline = copy.deepcopy(
                    self.initial_observation.get(hostname, {})
                )
                if hostname in self.baseline_overrides:
                    overrides = self.baseline_overrides[hostname]
                    if "Processes" in overrides:
                        if "Processes" not in effective_baseline:
                            effective_baseline["Processes"] = []
                        effective_baseline["Processes"].extend(overrides["Processes"])

                is_baseline = self._is_host_state_baseline(
                    current_host_state, effective_baseline
                )
                new_status = "baseline" if is_baseline else "changed"

                # Update the model
                host_model["status"] = new_status
                self.dynamic_environment_model[hostname] = host_model
                self.logger.info(
                    f"Host {hostname} state updated at step {self.episode_step}. New status: {new_status}"
                )

    def _handle_restore_action(self, action_string: str):
        if "Restore" in action_string:
            action_parts = parse_action_string(self.logger, action_string)
            hostname_to_restore = action_parts.get("hostname")
            if (
                hostname_to_restore
                and hostname_to_restore in self.dynamic_environment_model
            ):
                self.dynamic_environment_model[hostname_to_restore][
                    "status"
                ] = "baseline"

                if hostname_to_restore in self.baseline_overrides:
                    del self.baseline_overrides[hostname_to_restore]
                    self.logger.info(
                        f"Host {hostname_to_restore} status reset to 'baseline' and baseline overrides (decoys) cleared due to Restore action. History preserved."
                    )
                else:
                    self.logger.info(
                        f"Host {hostname_to_restore} status reset to 'baseline' due to Restore action. History preserved."
                    )

    def _handle_remove_action(self, action_string: str):
        """Set the host status to 'unknown' if a remove action is taken."""
        if "Remove" in action_string:
            action_parts = parse_action_string(self.logger, action_string)
            hostname_to_remove = action_parts.get("hostname")
            if (
                hostname_to_remove
                and hostname_to_remove in self.dynamic_environment_model
            ):
                self.dynamic_environment_model[hostname_to_remove]["status"] = "unknown"
                self.logger.info(
                    f"Host {hostname_to_remove} status set to 'unknown' due to Remove action."
                )

    def _handle_analyse_action(self, action_string: str):
        if "Analyse" in action_string:
            action_parts = parse_action_string(self.logger, action_string)
            hostname_to_analyse = action_parts.get("hostname")
            if (
                hostname_to_analyse
                and hostname_to_analyse in self.dynamic_environment_model
            ):
                current_status = self.dynamic_environment_model[
                    hostname_to_analyse
                ].get("status")
                if current_status == "unknown":
                    self.dynamic_environment_model[hostname_to_analyse][
                        "status"
                    ] = f"analysed at step {self.episode_step}"
                    self.logger.info(
                        f"Host {hostname_to_analyse} status updated from 'unknown' to 'analysed at step {self.episode_step}' due to Analyse action."
                    )

    def _handle_decoy_action(self, action_string: str):
        if "Decoy" in action_string:
            action_parts = parse_action_string(self.logger, action_string)
            hostname = action_parts.get("hostname")
            action_type = action_parts.get("action")

            if hostname and action_type:
                service_map = {
                    "DecoyApache": "apache2",
                    "DecoyFemitter": "femitter",
                    "DecoyHarakaSMPT": "haraka-smtp",
                    "DecoySmss": "smss",
                    "DecoySSHD": "sshd",
                    "DecoySvchost": "svchost",
                    "DecoyTomcat": "tomcat",
                    "DecoyVsftpd": "vsftpd",
                }
                service_name = action_type.replace("Decoy", "")
                expected_process_name = service_map.get(f"Decoy{service_name}")

                if expected_process_name:
                    if hostname not in self.baseline_overrides:
                        self.baseline_overrides[hostname] = {}
                    self.baseline_overrides[hostname][
                        "expected_decoy"
                    ] = expected_process_name
                    self.logger.info(
                        f"Expecting decoy '{expected_process_name}' on host {hostname} in the next step."
                    )

    def _update_baseline_with_decoy(
        self, hostname, previous_observation, current_observation
    ):
        # Check for an expected decoy from the previous step
        expected_decoy = self.baseline_overrides.get(hostname, {}).get("expected_decoy")
        if not expected_decoy:
            return

        prev_procs = {
            p.get("Service Name")
            for p in previous_observation.get(hostname, {}).get("Processes", [])
        }
        curr_procs = {
            p.get("Service Name")
            for p in current_observation.get(hostname, {}).get("Processes", [])
        }

        new_proc_names = curr_procs - prev_procs

        # STRICT FILTERING: Only look for the exact expected process name.
        # Ignore any other new processes that might have appeared (e.g. malware).
        matched_proc_obj = None
        if expected_decoy in new_proc_names:
            matched_proc_obj = next(
                (
                    p
                    for p in current_observation[hostname].get("Processes", [])
                    if p.get("Service Name") == expected_decoy
                ),
                None,
            )

        if matched_proc_obj:
            if "Processes" not in self.baseline_overrides[hostname]:
                self.baseline_overrides[hostname]["Processes"] = []

            # Add ONLY the matched decoy process to the baseline override
            self.baseline_overrides[hostname]["Processes"].append(matched_proc_obj)
            self.logger.info(
                f"Validated and added decoy process '{matched_proc_obj.get('Service Name')}' to baseline overrides for {hostname}."
            )
            # Clean up the expectation now that it has been met
            del self.baseline_overrides[hostname]["expected_decoy"]
        else:
            self.logger.warning(
                f"Expected decoy '{expected_decoy}' on {hostname} but it did not appear in the new processes: {new_proc_names}. "
                "Baseline override NOT applied."
            )

    def _get_network_status_message(self) -> str:
        updated_hosts_info = []
        hosts_in_current_obs = get_topology(self.logger, self.initial_observation)

        for hostname in self.dynamic_environment_model:
            host_model = self.dynamic_environment_model[hostname]
            status = host_model.get("status")

            if status != "baseline":
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
            return "Network Status: No hosts have been updated in this step. All hosts are in baseline state."

        return f"Network Status: The following hosts have updates or are in a non-baseline state:\n{json.dumps(updated_hosts_info, indent=2)}"

    def _get_summarized_history_message(self) -> str:
        sorted_steps = sorted(self.episode_memory.analyses_received.keys())
        recent_history = {
            k: self.episode_memory.analyses_received[k]
            for k in sorted_steps[-self.local_analyses_history_lookup_size :]
        }

        summarized_history = []
        if recent_history:
            temp_steps = []

            for step in sorted_steps:
                step_data = self.episode_memory.analyses_received[step]
                action = step_data.get("action", "")

                # Check if this is a "quiet" step
                if action in ["Monitor", "No action needed", "Sleep"]:
                    temp_steps.append(step)
                else:
                    # Flush any pending quiet steps as a summary range
                    if temp_steps:
                        if len(temp_steps) == 1:
                            summarized_history.append(
                                f"Step {temp_steps[0]}: Action: {self.episode_memory.analyses_received[temp_steps[0]].get('action')}"
                            )
                        else:
                            summarized_history.append(
                                f"Steps {temp_steps[0]}-{temp_steps[-1]}: Action: Monitor/No action needed. (No state changes observed)"
                            )
                        temp_steps = []

                    # Add the interesting step in full detail
                    summarized_history.append(
                        f"Step {step}: {json.dumps(step_data, indent=2)}"
                    )

            # Flush any remaining quiet steps at the end
            if temp_steps:
                if len(temp_steps) == 1:
                    summarized_history.append(
                        f"Step {temp_steps[0]}: Action: {self.episode_memory.analyses_received[temp_steps[0]].get('action')}"
                    )
                else:
                    summarized_history.append(
                        f"Steps {temp_steps[0]}-{temp_steps[-1]}: Action: Monitor/No action needed. (No state changes observed)"
                    )

        if not summarized_history:
            return ""

        return "Here is a history of the previous steps:\n" + "\n".join(
            summarized_history
        )

    def _create_initial_prompt(self, observation_str: str = ""):
        # Load the static part of the prompt from the YAML file
        with open(os.path.join(settings.AGENT_BASE_DIR, "agents/prompts/definitions/planner/initial_prompt.yaml"), "r") as f:
            prompt_data = yaml.safe_load(f).get("prompt", {})
            prompt_opening = prompt_data.get("opening", "")
            prompt_closure = prompt_data.get("closing", "")

        network_status = self._get_network_status_message()
        history_section = self._get_summarized_history_message()

        # network_baseline_status is shown only when all hosts are in baseline state
        all_baseline = all(
            host.get("status") == "baseline"
            for host in self.dynamic_environment_model.values()
        )
        network_baseline_status = (
            "Network Status: No hosts have been updated in this step. All hosts are in baseline state."
            if all_baseline
            else ""
        )

        # Format the prompt with the dynamic content
        prompt_opening = prompt_opening.format(
            step_number=self.episode_step,
            network_status=network_status,
            network_baseline_status=network_baseline_status,
            history=history_section,
            observation=observation_str,
        )

        # Combine into the initial prompt
        return f"{prompt_opening}\n\n{prompt_closure}"

    def get_action(
        self,
        current_observation_object: Dict[str, Any],
        action_space: Dict[str, Any],
        reward: float,
        debug=False,
        finalize: bool = False,
    ) -> Any:
        # Store the reward for the previous step as soon as it's received
        if self.episode_step > 1:
            self.episode_memory.rewards[self.episode_step - 1] = reward

        # Update episode_memory with current step BEFORE creating sub-agents
        self.episode_memory.episode_step = self.episode_step

        self.logger.info(
            f"--- Starting Planner decision process for Step {self.episode_step} ---"
        )

        current_observation = current_observation_object.observation
        self._update_dynamic_environment_model(current_observation)

        if self.episode_step == 1:
            current_observation_object.observation = {}
        else:
            # Store the observation as string representation for this step
            self.episode_memory.observations[self.episode_step - 1] = str(
                current_observation_object.observation
            )
            print("=" * 100)
            print(
                f"STEP STARTS: Observation: {self.episode_memory.observations[self.episode_step-1]}"
            )
            print("=" * 100)

        current_observation = current_observation_object.observation

        self.logger.debug(
            f"RAW_OBSERVATION:\n{pprint.pformat(current_observation_object.observation)}"
        )

        planner_agent = ReActAgent(
            get_dynamic_planner_system_prompt(),
            settings.PROVIDER,
            PROVIDER_PARAMETERS,
            logger=self.logger,
            log_name="Planner",
            episode_step=self.episode_step,
            max_turns=10,
            previous_step_reward=reward,
            log_reward=True,
            episode_memory=self.episode_memory,
        )

        executor = PlannerToolExecutor(
            self.logger,
            current_observation,
            self.initial_observation,
            self.dynamic_environment_model,
            self.baseline_overrides,
            self.action_space,
            self.episode_memory,
            self.topology,
            settings.TOOLS_PROVIDER,
            TOOLS_PROVIDER_PARAMETERS,
            question_budget=self.question_budget,
            dynamic_actions=True,
            reward=reward,
        )

        # Build observation string: empty on step 1 (full initial status), raw observation on subsequent steps
        observation_str = "" if self.episode_step == 1 else str(current_observation_object.observation)
        initial_prompt = self._create_initial_prompt(observation_str=observation_str)

        # If this is a finalization call, skip action validation entirely
        # We only need to complete trajectory logging, not produce a valid action
        if finalize:
            planner_agent.run(initial_prompt, executor, finalize=True)
            self.logger.info(
                f"--- Trajectory finalization completed for Step {self.episode_step} ---"
            )
            self._finalize_sub_agent_rewards()
            return None

        # Retry loop for invalid actions (only for non-finalize calls)
        max_retries = 3
        retry_count = 0
        current_prompt = initial_prompt
        final_action = "Monitor"  # Default safety

        while retry_count < max_retries:
            try:
                # Run the agent's reasoning loop
                final_action = planner_agent.run(current_prompt, executor, finalize=False)
                
                # Attempt to convert to object - this will raise ValueError if invalid
                action_obj = convert_action_string_to_object(self.logger, final_action, action_space)
                
                # If successful, we have our valid object. We need to set final_action to the valid string for logging?
                # Actually, we need to return the OBJECT at the end. 
                # But we also do logging below based on 'final_action' string.
                # So we break here.
                break
                
            except ValueError as e:
                retry_count += 1
                self.logger.warning(f"Agent produced invalid action '{final_action}': {e}. Retrying ({retry_count}/{max_retries})...")
                
                # Update prompt with error message
                error_feedback = f"\n\nERROR: The action '{final_action}' was invalid: {e}\nPlease provide the Correct Action with all required parameters (e.g. Action hostname='Target')."
                current_prompt = f"The previous attempt resulted in an error:: {error_feedback}\n\nTry again:"
                
                # If we hit max retries, we might want to default or just let the last error slide (which will crash or be caught?)
                # We should probably default to Monitor if all retries fail to avoid crash.
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached. Defaulting to Monitor.")
                    final_action = "Monitor"
                    action_obj = convert_action_string_to_object(self.logger, final_action, action_space)

        # Clean and store the planner's final thought process as an analysis
        analysis_from_planner = clean_agent_answer(
            planner_agent.state.messages[-1].get("content", "")
        )
        if analysis_from_planner:
            if self.episode_step not in self.episode_memory.analyses_received:
                self.episode_memory.analyses_received[self.episode_step] = {}
            self.episode_memory.analyses_received[self.episode_step][
                "analysis"
            ] = analysis_from_planner

        # Store messages for this step
        self.episode_memory.messages[self.episode_step - 1] = (
            planner_agent.state.messages
        )
        # If we have a final action, store it in actions_taken
        if final_action and "No answer found" not in final_action:
            if self.episode_step not in self.episode_memory.analyses_received:
                self.episode_memory.analyses_received[self.episode_step] = {}
            self.episode_memory.analyses_received[self.episode_step][
                "action"
            ] = final_action

            # Log the action sequence for the targeted host
            action_parts = parse_action_string(self.logger, final_action)
            hostname = action_parts.get("hostname")
            action_name = action_parts.get("action")
            if hostname and action_name and hostname in self.dynamic_environment_model:
                self.dynamic_environment_model[hostname][
                    "applied_actions_so_far"
                ].append(action_name)
                self.logger.info(
                    f"Logged action '{action_name}' for host '{hostname}'. New history: {'->'.join(self.dynamic_environment_model[hostname]['applied_actions_so_far'])}"
                )

            self._handle_restore_action(final_action)
            self._handle_remove_action(final_action)
            self._handle_analyse_action(final_action)
            self._handle_decoy_action(final_action)

        # Increment the episode step before logging the end, but use the original step number for clarity
        original_step = self.episode_step
        self._previous_observation = current_observation
        self.episode_step += 1

        self.logger.info(
            f"--- Ending Planner decision process for Step {original_step}. Final Action: {final_action} ---"
        )
        # Return the final action object we already validated
        return action_obj

    def train(self, results):
        pass

    def end_episode(self, total_reward, episode_number):
        """Ends the episode and logs a summary."""
        self.logger.info(f"Total reward for Episode {episode_number}: {total_reward}")
        self.current_episode = episode_number

        # Log the detailed summary of the episode
        if self.log_summary:
            self._log_episode_summary()

        # Reset episode memory completely
        self.episode_memory = EpisodeMemory(
            analyses_received={},
            messages={},
            initial_prompt="",
            observations={},
            rewards={},
            episode_step=0,
        )
        self.dynamic_environment_model = {}
        self.baseline_overrides = {}
        self.episode_step = 1

    def create_learning_snapshot(
        self,
        failed_step: int,
        reward: float,
        total_reward: float,
        reward_threshold: float,
        agents_to_improve: List[str],
        helper_agents: List[str],
        agent_log_map: Dict[str, str],
        max_reflection_rules: int,
        disable_trajectory_pruning: bool,
        disable_system_prompt_pruning: bool,
        session_learning_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snapshot = {
            "failed_step": failed_step,
            "reward": reward,
            "total_reward": total_reward,
            "reward_threshold": reward_threshold,
            "agents_to_improve": agents_to_improve,
            "helper_agents": helper_agents,
            "max_reflection_rules": max_reflection_rules,
            "disable_trajectory_pruning": disable_trajectory_pruning,
            "disable_system_prompt_pruning": disable_system_prompt_pruning,
            "session_learning_history": session_learning_history or [],
            "all_trajectories": {},
            "knowledge": {},
            "dynamic_environment_model": copy.deepcopy(self.dynamic_environment_model),
            "episode_memory": copy.deepcopy(self.episode_memory),
        }

        run_dir = get_current_run_dir()
        all_agent_names = set(agents_to_improve + helper_agents)
        for agent_name in all_agent_names:
            snapshot["knowledge"][agent_name] = {}
            log_filename = agent_log_map.get(agent_name)
            if not log_filename:
                continue

            trajectory_path = os.path.join(run_dir, "trajectories", log_filename)
            try:
                with open(trajectory_path, "r") as f:
                    trajectory_content = json.load(f)
                snapshot["all_trajectories"][agent_name] = trajectory_content
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(
                    f"Could not read trajectory for {agent_name} during snapshot creation: {e}"
                )

            # Load all three types of knowledge
            for knowledge_type in ["persistent", "reflection", "tactical"]:
                knowledge_path = os.path.join(settings.AGENT_BASE_DIR, f"agents/prompts/definitions/{agent_name.lower()}/{knowledge_type}_knowledge.yaml")
                try:
                    with open(knowledge_path, "r") as f:
                        knowledge_data = yaml.safe_load(f)
                    # Ensure we always get a list, handling empty files (None) or keys with null values.
                    snapshot["knowledge"][agent_name][knowledge_type] = (
                        knowledge_data.get("reflection_knowledge") or []
                        if knowledge_data
                        else []
                    )
                except FileNotFoundError:
                    snapshot["knowledge"][agent_name][knowledge_type] = []

        return snapshot

    def _finalize_sub_agent_rewards(self):
        """Ensure rewards are logged for sub-agents from the previous step."""
        if self.episode_step <= 1:
            return

        previous_step = self.episode_step - 1
        self.logger.info(f"Finalizing sub-agent rewards for step {previous_step}")

        # Create dummy sub-agents to trigger reward logging
        # This ensures ActionChooser and Analyst rewards are logged
        from agents.prompts.action_chooser import ACTION_AGENT_SYSTEM_PROMPT
        from agents.prompts.analyst import ANALYST_SYSTEM_PROMPT

        run_dir = get_current_run_dir()

        # Finalize ActionChooser reward if it was active in previous step
        action_chooser_trajectory_path = os.path.join(
            run_dir, "trajectories", "Planner.Executor.ActionChooser.json"
        )
        if os.path.exists(action_chooser_trajectory_path):
            try:
                with open(action_chooser_trajectory_path, "r") as f:
                    trajectory = json.load(f)
                if str(previous_step) in trajectory.get("steps", {}):
                    self.logger.info(
                        f"Finalizing ActionChooser reward for step {previous_step}"
                    )
                    dummy_action_chooser = ReActAgent(
                        ACTION_AGENT_SYSTEM_PROMPT,
                        settings.PROVIDER,
                        PROVIDER_PARAMETERS,
                        logger=self.logger,
                        log_name="Planner.Executor.ActionChooser",
                        episode_step=self.episode_step,
                        max_turns=1,
                        previous_step_reward=self.episode_memory.rewards.get(
                            previous_step, 0.0
                        ),
                        log_reward=True,
                        episode_memory=self.episode_memory,
                    )
            except Exception as e:
                self.logger.warning(f"Failed to finalize ActionChooser reward: {e}")

        # Finalize Analyst reward if it was active in previous step
        analyst_trajectory_path = os.path.join(
            run_dir, "trajectories", "Planner.Executor.Analyst.json"
        )
        if os.path.exists(analyst_trajectory_path):
            try:
                with open(analyst_trajectory_path, "r") as f:
                    trajectory = json.load(f)
                if str(previous_step) in trajectory.get("steps", {}):
                    self.logger.info(
                        f"Finalizing Analyst reward for step {previous_step}"
                    )
                    dummy_analyst = ReActAgent(
                        ANALYST_SYSTEM_PROMPT,
                        PROVIDER,
                        PROVIDER_PARAMETERS,
                        logger=self.logger,
                        log_name="Planner.Executor.Analyst",
                        episode_step=self.episode_step,
                        max_turns=1,
                        previous_step_reward=self.episode_memory.rewards.get(
                            previous_step, 0.0
                        ),
                        log_reward=True,
                        episode_memory=self.episode_memory,
                    )
            except Exception as e:
                self.logger.warning(f"Failed to finalize Analyst reward: {e}")

    def _log_episode_summary(self):
        """Logs a formatted summary of the entire episode."""
        summary_parts = ["\n\n\n" + "=" * 50 + " EPISODE SUMMARY " + "=" * 50]

        # Log Actions Taken
        summary_parts.append("\n--- Step History ---")

        if self.episode_memory.analyses_received:
            summary_parts.append(pprint.pformat(self.episode_memory.analyses_received))
        else:
            summary_parts.append("No steps were recorded.")

        # Log Final Delta States
        summary_parts.append("\n--- Final Delta-Derived Host States ---")
        if self.dynamic_environment_model:
            summary_parts.append(pprint.pformat(self.dynamic_environment_model))
        else:
            summary_parts.append("No delta-derived states were created.")

        summary_parts.append("\n" + "=" * 50 + " END SUMMARY " + "=" * 50)
        self.logger.info("\n".join(summary_parts))
