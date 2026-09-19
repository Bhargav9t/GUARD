# ==============================================================================
# File: src/utils/logger.py
# Description: Centralized Logging Utility for GUARD.
# Provides unified console and file logging with customizable formats,
# log rotation readiness, and visual styling.
# ==============================================================================

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "GUARD",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Initializes and configures a standard logger instance for the application.

    Parameters:
        name (str): Identifier for the logger. Default is "GUARD".
        log_file (Optional[Path]): Destination path for log file persistence.
                                   If None, logs are only output to console.
        level (int): Minimum logging severity level (e.g. logging.DEBUG, logging.INFO).
        console_output (bool): Whether to stream logs to sys.stdout. Default is True.

    Returns:
        logging.Logger: Fully configured Logger instance.
    """
    # 1. Obtain or instantiate the logger by name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 2. Prevent adding duplicate handlers if setup_logger is invoked multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Define consistent message formatting
    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [LOGGER_NAME] Message
    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 4. Attach Console Stream Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # 5. Attach File Handler if destination path is specified
    if log_file is not None:
        log_file = Path(log_file)
        # Ensure target logging directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(filename=str(log_file), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # Avoid propagation to ancestor root loggers which might duplicate entries
    logger.propagate = False

    return logger


# Default application-wide logger instance
app_logger = setup_logger(name="GUARD")
