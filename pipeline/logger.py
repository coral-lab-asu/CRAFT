"""
Logging setup for CRAFT.

Every logger.info/warning/error call goes to both the terminal and the log
file at the same time, so you always have a timestamped record of what the
pipeline did and when.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(log_file: str = "craft_run.log", name: str = "craft") -> logging.Logger:
    """
    Create (or retrieve) a logger that writes to both stdout and a file.

    Args:
        log_file: Path to the log file.  Parent dirs are created automatically.
        name:     Logger name.  Call with the same name anywhere to get the
                  same logger without setting up handlers twice.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only add handlers once; subsequent calls just return the same logger.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (INFO and above) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # --- File handler (DEBUG and above, so every detail is captured) ---
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logger initialised — writing to %s", log_path.resolve())
    return logger


def get_logger(name: str = "craft") -> logging.Logger:
    """Return the already-configured logger (must call setup_logger first)."""
    return logging.getLogger(name)
