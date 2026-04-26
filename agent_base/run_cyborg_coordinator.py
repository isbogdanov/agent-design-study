import logging
import inspect
import argparse
import os
from datetime import datetime
from typing import List

# from llm_cage_agent import LLMCageAgent
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents import B_lineAgent

from CybORG.Agents import SleepAgent

from CybORG.Agents.SimpleAgents.BlueReactAgent import (
    BlueReactRemoveAgent,
    BlueReactRestoreAgent,
)
from CybORG.Agents.SimpleAgents.KeyboardAgent import KeyboardAgent

# Import action processing functions
from utils.helpers.action_processing import format_blue_action_space, get_blue_actions
from logs.config.log_config import setup_logging
import utils.settings as settings

# Import our custom ReactAgent
# from react_agent import ReactAgent

# from my_agent import MyAgent

# from blue_random_agent import BlueRandomAgent

from coordinators.cyborg_agent_coordinator import CybORGAgentCoordinator


def main():
    parser = argparse.ArgumentParser(
        description="Run CybORG agent evaluation."
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of steps per episode."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override the LLM provider (e.g., 'google', 'openai', 'openrouter').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model name.",
    )
    args = parser.parse_args()

    # Apply provider/model override if arguments are provided
    if args.provider or args.model:
        current_provider, current_model = settings.PROVIDER
        new_provider = args.provider if args.provider else current_provider
        new_model = args.model if args.model else current_model
        settings.update_provider_settings(new_provider, new_model)
        print(f"CLI Override: Settings updated to Provider='{new_provider}', Model='{new_model}'")

    # Configure logging — single run directory under 'evaluating/'
    eval_dir = os.path.join(settings.AGENT_BASE_DIR, "logs", "runs", "evaluating")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(eval_dir, f"run_{timestamp}")
    setup_logging(attempt_dir=run_dir, session_dir=run_dir)

    logger = logging.getLogger(__name__)

    # Get path to scenario file
    cyborg_module_path = os.path.dirname(inspect.getfile(CybORG))
    scenario_path = os.path.join(cyborg_module_path, "Shared", "Scenarios", "Scenario2.yaml")

    # Initialize the environment
    logger.info(f"Initializing CAGE environment with scenario: {scenario_path}")
    cyborg = CybORG(
        scenario_path,
        "sim",
        agents={"Red": B_lineAgent},
    )

    agent = CybORGAgentCoordinator(log_summary=True)
    run_evaluation_session(cyborg, agent, steps=args.steps)


def run_evaluation_session(
    cyborg: CybORG,
    agent: CybORGAgentCoordinator,
    n_episodes: int = 1,
    steps: int = 30,
):
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Evaluation Session ---")
    episode_rewards = []
    agent_name = agent.__class__.__name__
    logger.info(f"Running {n_episodes} episode(s) with {agent_name}...")

    for episode in range(n_episodes):
        logger.info(f"Starting Episode {episode + 1}/{n_episodes}")
        total_reward = run_single_episode(
            cyborg,
            agent,
            episode_num=episode + 1,
            steps=steps,
        )
        episode_rewards.append(total_reward)

    # Print summary of all episode rewards
    logger.info("===== EPISODE REWARDS SUMMARY =====")
    logger.info(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    logger.info(f"Total Cumulative Reward: {sum(episode_rewards):.2f}")
    logger.info(
        f"Best Episode: #{episode_rewards.index(max(episode_rewards))+1} with reward {max(episode_rewards):.2f}"
    )
    logger.info(
        f"Worst Episode: #{episode_rewards.index(min(episode_rewards))+1} with reward {min(episode_rewards):.2f}"
    )
    logger.info("Evaluation complete.")


def run_single_episode(
    cyborg: CybORG,
    agent: CybORGAgentCoordinator,
    episode_num: int,
    steps: int,
) -> float:
    """Runs a single evaluation episode and returns the total reward."""
    total_reward = 0
    observation = cyborg.reset()
    action_space = cyborg.get_action_space("Blue")
    agent.set_initial_values(action_space, observation.observation)
    reward = 0
    logger = logging.getLogger(__name__)

    for step in range(steps):
        print("\n\n\n\n\n")
        print("=" * 120)
        print(f" CAGE STEP {step + 1}/{steps} (Episode #{episode_num}) START ")
        print("=" * 120)

        action = agent.get_action(observation, action_space, reward, False)
        results = cyborg.step("Blue", action)
        total_reward += results.reward
        reward = results.reward

        logger.info(
            f"MAIN_LOOP| Episode={episode_num} | Step={step+1} | Action={action} | Reward={results.reward} | TotalReward={total_reward}"
        )

        print(f"============== CAGE STEP {step + 1}/{steps} (Episode #{episode_num}) END ==============\n")
        observation = results

    agent.end_episode(total_reward, episode_num)
    return total_reward


if __name__ == "__main__":
    main()
