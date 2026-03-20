"""
Centralized logging configuration for the application.
"""

import glob
import logging
import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

# Thread-local storage for per-task context
_thread_local = threading.local()


def set_task_id(task_id: str) -> None:
    """
    Set the task ID for the current thread.
    All log records emitted by this thread will include this task_id.
    Call with 'main' (or '') to clear the context.
    """
    _thread_local.task_id = task_id


class ThreadTaskFilter(logging.Filter):
    """
    Injects task_id from thread-local storage into every log record.
    This makes the task_id visible in ALL log messages produced by a thread,
    including those from third-party libraries (httpx, google-genai, etc.).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.task_id = getattr(_thread_local, "task_id", "main")
        return True


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Configure logging based on provided config or environment variables.

    Args:
        config: Optional dictionary with logging configuration.
            If not provided, will read from environment variables.
    """
    config = config or {}

    # Get config from parameter or environment
    log_level_name = config.get("log_level") or os.getenv("LOG_LEVEL", "INFO")
    log_format = config.get("log_format") or os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(threadName)s [%(task_id)s] - %(name)s - %(levelname)s - %(message)s"
    )
    log_dir = config.get("log_dir") or os.getenv("LOG_DIR", "logs/")
    enable_console = config.get("enable_console", None)

    if enable_console is None:
        enable_console_str = os.getenv("ENABLE_CONSOLE_LOGGING", "true")
        enable_console = enable_console_str.lower() == "true"

    # Map string level to logging constant
    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(log_level_name, logging.INFO)

    # Create directory if needed
    os.makedirs(log_dir, exist_ok=True)

    # ── Garbage-collect old log files: keep the 5 most recent ──────────────
    _cleanup_old_logs(log_dir, keep=5)

    # Set up file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/carmina_{timestamp}.log"

    # Set up handlers
    task_filter = ThreadTaskFilter()
    handlers: List[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    # # Especificamos que la lista acepta cualquier "logging.Handler"
    if enable_console:
        handlers.append(logging.StreamHandler())

    # Apply task_id filter to every handler so ALL log records carry it
    for h in handlers:
        h.addFilter(task_filter)

    # Configure logging
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level_name}")

    # Return the configured logger
    return logger


def _cleanup_old_logs(log_dir: str, keep: int = 5) -> None:
    """
    Delete all but the *keep* most-recent carmina_*.log files in *log_dir*.
    This prevents logs/ from growing without bound across long benchmark runs.
    """
    pattern = os.path.join(log_dir, "carmina_*.log")
    log_files = sorted(glob.glob(pattern))  # lexicographic = chronological for YYYYMMDD_HHMMSS names
    to_delete = log_files[:-keep] if len(log_files) >= keep else []
    for path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass  # already gone or locked — not worth crashing over


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a specific module.

    Args:
        name: Name for the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

