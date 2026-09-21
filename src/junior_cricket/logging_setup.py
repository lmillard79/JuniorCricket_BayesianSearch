"""Logging setup for junior_cricket processing scripts.

Implements the mandatory logging protocol for data processing
scripts:

* a ``log`` directory is created immediately adjacent to the
  primary output directory,
* log files are stamped with the exact execution time using the
  ``YYYYMMDD_HHMMSS_<script_name>.log`` naming convention,
* all records route to both the console and the log file using
  the standard ``logging`` module (no ``print()`` calls).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(script_name: str, output_dir: Path) -> logging.Logger:
    """Configure dual console and file logging for a script run.

    Args:
        script_name: Name of the calling script, used both as the
            logger name and in the log filename stem.
        output_dir: Primary output directory for the script. A
            sibling ``log`` directory is created next to it.

    Returns:
        A configured ``Logger`` writing to both the console and a
        timestamped log file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir.parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Script: %s", script_name)
    logger.info("Log file: %s", log_file.resolve())
    return logger
