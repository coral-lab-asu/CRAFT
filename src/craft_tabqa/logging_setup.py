"""A logger that writes to both the terminal and a run log file.

The console shows INFO and above; the file captures DEBUG and above so a full
timestamped record of every run is kept on disk.
"""

import logging
import sys
from pathlib import Path


def setup_logger(log_file: str = "craft.log", name: str = "craft") -> logging.Logger:
    """Return a logger writing to stdout and ``log_file`` (idempotent per name)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logging to %s", log_path.resolve())
    return logger
