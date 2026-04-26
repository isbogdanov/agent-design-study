#!/usr/bin/env python3
"""
Utility script for plotting learning metrics from continual learning sessions.

Usage:
    python utils/plot_learning_metrics.py path/to/learning_metrics.json
    python utils/plot_learning_metrics.py logs/runs/learning/learning_session_*/learning_metrics.json
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_metrics(metrics_file: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        sys.exit(1)


def plot_learning_progress(metrics_data: Dict[str, Any], output_dir: str = None):
    """Create comprehensive learning progress plots."""

    if not metrics_data.get("attempts"):
        print("No attempt data found in metrics.")
        return

    attempts = []
    for attempt_data in metrics_data["attempts"].values():
        attempts.append(
            {
                "attempt": attempt_data["attempt_number"],
                "steps": attempt_data["steps_completed"],
                "total_reward": attempt_data["total_reward"],
                "average_reward": attempt_data["total_reward"]
                / max(attempt_data["steps_completed"], 1),
                "reflection_triggered": attempt_data["reflection_triggered"],
                "reflection_step": attempt_data.get("reflection_step"),
                "status": attempt_data["attempt_status"],
                "step_rewards": [sr["reward"] for sr in attempt_data["step_rewards"]],
                "cumulative_rewards": [
                    sr["cumulative_reward"] for sr in attempt_data["step_rewards"]
                ],
            }
        )

    # Sort by attempt number
    attempts.sort(key=lambda x: x["attempt"])

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Continual Learning Progress", fontsize=16, fontweight="bold")

    # Extract data for plotting
    attempt_nums = [a["attempt"] for a in attempts]
    total_rewards = [a["total_reward"] for a in attempts]
    avg_rewards = [a["average_reward"] for a in attempts]
    steps_completed = [a["steps"] for a in attempts]

    # Color code by success/failure
    colors = ["green" if a["status"] == "success" else "red" for a in attempts]

    # Plot 1: Total Reward per Attempt
    bars1 = ax1.bar(attempt_nums, total_rewards, color=colors, alpha=0.7)
    ax1.set_xlabel("Attempt Number")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward per Attempt")
    ax1.grid(True, alpha=0.3)

    # Add reflection markers
    for i, attempt in enumerate(attempts):
        if attempt["reflection_triggered"]:
            ax1.annotate(
                f'R@{attempt["reflection_step"]}',
                xy=(attempt["attempt"], attempt["total_reward"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="blue",
            )

    # Plot 2: Steps Completed per Attempt
    bars2 = ax2.bar(attempt_nums, steps_completed, color=colors, alpha=0.7)
    ax2.set_xlabel("Attempt Number")
    ax2.set_ylabel("Steps Completed")
    ax2.set_title("Steps Completed per Attempt")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Reward per Step
    ax3.plot(attempt_nums, avg_rewards, "o-", color="blue", linewidth=2, markersize=6)
    ax3.set_xlabel("Attempt Number")
    ax3.set_ylabel("Average Reward per Step")
    ax3.set_title("Learning Efficiency (Avg Reward/Step)")
    ax3.grid(True, alpha=0.3)

    # Highlight successful attempts
    for i, attempt in enumerate(attempts):
        if attempt["status"] == "success":
            ax3.plot(
                attempt["attempt"],
                attempt["average_reward"],
                "go",
                markersize=10,
                alpha=0.7,
            )

    # Plot 4: Step-by-step rewards for recent attempts (last 3)
    recent_attempts = attempts[-3:] if len(attempts) >= 3 else attempts
    for i, attempt in enumerate(recent_attempts):
        step_nums = list(range(1, len(attempt["step_rewards"]) + 1))
        ax4.plot(
            step_nums,
            attempt["step_rewards"],
            "o-",
            label=f'Attempt {attempt["attempt"]}',
            alpha=0.8,
        )

    ax4.set_xlabel("Step Number")
    ax4.set_ylabel("Step Reward")
    ax4.set_title("Step-by-Step Rewards (Recent Attempts)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add overall legend
    success_patch = mpatches.Patch(color="green", alpha=0.7, label="Successful")
    failed_patch = mpatches.Patch(color="red", alpha=0.7, label="Failed")
    reflection_patch = mpatches.Patch(color="blue", label="Reflection Triggered")
    fig.legend(
        handles=[success_patch, failed_patch, reflection_patch],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, "learning_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Learning progress plot saved to: {plot_file}")

    plt.show()


def print_session_summary(metrics_data: Dict[str, Any]):
    """Print a text summary of the learning session."""
    session_info = metrics_data.get("session_info", {})
    attempts = list(metrics_data.get("attempts", {}).values())

    print("\n" + "=" * 60)
    print("LEARNING SESSION SUMMARY")
    print("=" * 60)

    print(f"Session Start: {session_info.get('session_start_time', 'Unknown')}")
    print(f"Total Attempts: {len(attempts)}")

    if attempts:
        successful = [a for a in attempts if a["attempt_status"] == "success"]
        failed = [a for a in attempts if a["attempt_status"] != "success"]

        print(f"Successful Attempts: {len(successful)}")
        print(f"Failed Attempts: {len(failed)}")
        print(f"Success Rate: {len(successful)/len(attempts)*100:.1f}%")

        total_rewards = [a["total_reward"] for a in attempts]
        steps_completed = [a["steps_completed"] for a in attempts]

        print(f"Best Reward: {max(total_rewards):.2f}")
        print(f"Worst Reward: {min(total_rewards):.2f}")
        print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
        print(
            f"Average Steps per Attempt: {sum(steps_completed)/len(steps_completed):.1f}"
        )

        reflections = sum(1 for a in attempts if a["reflection_triggered"])
        print(f"Reflections Triggered: {reflections}")

        print("\nPER-ATTEMPT BREAKDOWN:")
        print("-" * 60)
        for attempt in sorted(attempts, key=lambda x: x["attempt_number"]):
            status = "✓" if attempt["attempt_status"] == "success" else "✗"
            reflection = (
                f" (R@{attempt.get('reflection_step', '?')})"
                if attempt["reflection_triggered"]
                else ""
            )
            print(
                f"Attempt {attempt['attempt_number']:2d}: {status} "
                f"Steps={attempt['steps_completed']:2d} "
                f"Reward={attempt['total_reward']:6.2f}"
                f"{reflection}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning metrics from continual learning sessions"
    )
    parser.add_argument("metrics_file", help="Path to learning_metrics.json file")
    parser.add_argument(
        "--output-dir", help="Directory to save plots (default: same as metrics file)"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Only print summary, don't show plots"
    )

    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        print(f"Metrics file not found: {args.metrics_file}")
        sys.exit(1)

    metrics_data = load_metrics(args.metrics_file)

    # Print summary
    print_session_summary(metrics_data)

    # Create plots unless disabled
    if not args.no_plot:
        try:
            output_dir = args.output_dir or os.path.dirname(args.metrics_file)
            plot_learning_progress(metrics_data, output_dir)
        except ImportError:
            print(
                "\nWarning: matplotlib not available. Install with: pip install matplotlib"
            )
            print("Showing text summary only.")
        except Exception as e:
            print(f"\nError creating plots: {e}")
            print("Showing text summary only.")


if __name__ == "__main__":
    main()




