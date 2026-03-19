"""Structured logging setup using Rich."""

import logging
import sys

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure structured logging with rich console output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.

    Returns:
        Configured root logger.
    """
    handlers: list[logging.Handler] = [
        RichHandler(
            level=level,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger("omr")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(f"omr.{name}")
