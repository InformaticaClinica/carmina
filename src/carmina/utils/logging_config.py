"""
Centralized logging configuration for the application.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


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
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    # Set up file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/carmina_{timestamp}.log"

    # Set up handlers
    handlers: List[logging.Handler] = [logging.FileHandler(log_file)]
    # # Especificamos que la lista acepta cualquier "logging.Handler"
    if enable_console:
        handlers.append(logging.StreamHandler())

    # Configure logging
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level_name}")

    # Return the configured logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a specific module.

    Args:
        name: Name for the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

