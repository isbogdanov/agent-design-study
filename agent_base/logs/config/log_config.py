import logging
import sys
from datetime import datetime
import os
import atexit


class TeeOutput:
    """A class that writes to multiple outputs simultaneously."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def write(self, text):
        for output in self.outputs:
            output.write(text)
            output.flush()

    def flush(self):
        for output in self.outputs:
            output.flush()

    def add_output(self, output):
        self.outputs.append(output)

    def remove_output(self, output):
        if output in self.outputs:
            self.outputs.remove(output)


# Global variables
_current_run_dir = None
_original_stdout = None
_console_mirror_file = None


class IndentingFormatter(logging.Formatter):
    """A custom formatter to add indentation based on logger name hierarchy."""

    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # Calculate indentation level based on the number of dots in the logger name
        indent_level = record.name.count(".")
        indent = "    " * indent_level  # 4 spaces per level

        # Prepend indentation to the message
        record.msg = indent + str(record.msg)

        # Handle multiline messages
        if "\n" in record.msg:
            # Add indentation to each line of the message
            record.msg = record.msg.replace("\n", "\n" + indent)

        return super().format(record)


def add_dynamic_logger(logger_name: str, log_file_path: str):
    """
    Dynamically adds a file handler to a logger at runtime.
    This is useful for creating dedicated logs for concurrent tasks.
    """
    dynamic_logger = logging.getLogger(logger_name)
    dynamic_logger.propagate = False  # Prevent logs from going to the root logger

    # Clear existing handlers to avoid duplication if this is called more than once
    if dynamic_logger.hasHandlers():
        dynamic_logger.handlers.clear()

    # Create a file handler for the new log file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Use the same indenting formatter for consistency
    log_format = IndentingFormatter(
        "%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s"
    )
    file_handler.setFormatter(log_format)
    dynamic_logger.addHandler(file_handler)

    # Also add the console handler to see these logs in real-time
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    dynamic_logger.addHandler(console_handler)

    logging.info(
        f"Dynamically configured logger '{logger_name}' to output to {log_file_path}"
    )


def setup_logging(attempt_dir: str, session_dir: str):
    global _current_run_dir, _original_stdout, _console_mirror_file

    os.makedirs(attempt_dir, exist_ok=True)
    _current_run_dir = (
        attempt_dir  # The current run dir is always the specific attempt dir
    )

    log_format = IndentingFormatter(
        "%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- Clear existing handlers before setting up new ones ---
    # Clear root logger handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Clear specific agent logger handlers to prevent duplicate logging
    loggers_to_clear = [
        "Planner",
        "Planner.Executor.Analyst",
        "Planner.Executor.ActionChooser",
        "ReflectionCoordinator",
        "LearningCoordinator",
    ]
    for logger_name in loggers_to_clear:
        logger = logging.getLogger(logger_name)
        if logger.hasHandlers():
            logger.handlers.clear()
            logger.propagate = False  # Ensure we control propagation state

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(root_logger.level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    timestamp = (
        os.path.basename(attempt_dir).replace("run_", "").replace("attempt_", "")
    )
    main_log_path = os.path.join(attempt_dir, f"run_{timestamp}_main.log")
    main_file_handler = logging.FileHandler(main_log_path)
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(log_format)
    root_logger.addHandler(main_file_handler)

    # --- Console Mirroring ---
    # The console mirror is now ALWAYS at the session level.
    mirror_timestamp = (
        os.path.basename(session_dir)
        .replace("run_", "")
        .replace("learning_session_", "")
    )
    console_mirror_path = os.path.join(
        session_dir, f"{mirror_timestamp}_console_mirror.log"
    )

    if _console_mirror_file is None:
        # This part runs only once per session
        _console_mirror_file = open(console_mirror_path, "a")
        if _original_stdout is None:
            _original_stdout = sys.stdout
        sys.stdout = TeeOutput(_original_stdout, _console_mirror_file)
        atexit.register(cleanup_stdout_redirect)

    # This handler is created fresh for each attempt, but always points to the same session-level file.
    console_mirror_handler = logging.FileHandler(console_mirror_path)
    console_mirror_handler.setLevel(root_logger.level)
    console_mirror_handler.setFormatter(log_format)
    root_logger.addHandler(console_mirror_handler)

    log_files_for = {
        "Planner": "planner",
        "Planner.Executor.Analyst": "analyst",
        "Planner.Executor.ActionChooser": "action_chooser",
        "ReflectionCoordinator": "reflection_coordinator",
        "LearningCoordinator": "learning_coordinator",
    }

    for logger_name, file_suffix in log_files_for.items():
        specific_logger = logging.getLogger(logger_name)
        specific_logger.propagate = False

        specific_log_path = os.path.join(
            attempt_dir, f"run_{timestamp}_{file_suffix}.log"
        )
        file_handler = logging.FileHandler(specific_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        specific_logger.addHandler(file_handler)
        specific_logger.addHandler(console_handler)
        specific_logger.addHandler(console_mirror_handler)

    logging.info(f"Logging configured for attempt directory: {attempt_dir}")
    return attempt_dir


def restore_stdout():
    """Manually restore original stdout (useful for debugging or testing)."""
    cleanup_stdout_redirect()


def get_context_logger(
    parent_logger: logging.Logger, child_name: str
) -> logging.Logger:
    """
    Creates a child logger with a hierarchical name.
    """
    return logging.getLogger(f"{parent_logger.name}.{child_name}")


def cleanup_stdout_redirect():
    """Restore original stdout and close console mirror file."""
    global _original_stdout, _console_mirror_file

    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None

    if _console_mirror_file is not None:
        _console_mirror_file.close()
        _console_mirror_file = None


def get_current_run_dir():
    """
    Get the current run directory path for saving additional files.
    This assumes setup_logging has already been called.
    """
    global _current_run_dir
    if _current_run_dir is None:
        raise RuntimeError(
            "setup_logging() must be called first to initialize run directory"
        )
    return _current_run_dir
