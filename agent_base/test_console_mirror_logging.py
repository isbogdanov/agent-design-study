#!/usr/bin/env python3
"""
Test script to verify that console mirror logging works correctly.
This script tests the new console mirror functionality without running the full agent.
"""

import logging
import os
import sys
from logs.config.log_config import setup_logging, get_current_run_dir


def test_console_mirror_logging():
    """Test that console mirror logging captures all console output."""
    print("Testing console mirror logging functionality...")

    # Setup logging
    run_dir = setup_logging()
    print(f"Created run directory: {run_dir}")

    # Get various loggers to test
    root_logger = logging.getLogger()
    planner_logger = logging.getLogger("Planner")
    analyst_logger = logging.getLogger("Planner.Executor.Analyst")
    action_logger = logging.getLogger("Planner.Executor.ActionChooser")

    # Test messages at different levels
    print("\n=== Testing different log levels ===")

    root_logger.info(
        "ROOT: This is an info message that should appear in console and console mirror"
    )
    root_logger.warning("ROOT: This is a warning message")
    root_logger.error("ROOT: This is an error message")

    planner_logger.info("PLANNER: Planning phase started")
    planner_logger.debug(
        "PLANNER: This debug message should only be in files, not console"
    )

    analyst_logger.info("ANALYST: Analyzing host User4")
    analyst_logger.warning("ANALYST: Suspicious activity detected")

    action_logger.info("ACTION_CHOOSER: Recommending Remove action")
    action_logger.error("ACTION_CHOOSER: Failed to parse action")

    # Check that files were created
    console_mirror_file = None
    main_log_file = None

    for filename in os.listdir(run_dir):
        if filename.endswith("_console_mirror.log"):
            console_mirror_file = os.path.join(run_dir, filename)
        elif filename.endswith("_main.log"):
            main_log_file = os.path.join(run_dir, filename)

    print(f"\n=== File Check Results ===")
    print(f"Console mirror file created: {console_mirror_file is not None}")
    print(f"Main log file created: {main_log_file is not None}")

    if console_mirror_file:
        print(f"Console mirror file: {console_mirror_file}")
        with open(console_mirror_file, "r") as f:
            mirror_content = f.read()
            line_count = (
                len(mirror_content.strip().split("\n")) if mirror_content.strip() else 0
            )
            print(f"Console mirror log lines: {line_count}")

    if main_log_file:
        print(f"Main log file: {main_log_file}")
        with open(main_log_file, "r") as f:
            main_content = f.read()
            line_count = (
                len(main_content.strip().split("\n")) if main_content.strip() else 0
            )
            print(f"Main log lines: {line_count}")

    print(f"\n=== Test completed ===")
    print(f"Check the console mirror file to verify it matches console output")
    print(f"Run directory: {run_dir}")

    return run_dir


if __name__ == "__main__":
    test_console_mirror_logging()
