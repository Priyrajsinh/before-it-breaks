"""Structured logger factory — JSON when ENV=production, plain text otherwise."""

import logging
import os
import sys

from pythonjsonlogger import json as jsonlogger


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. JSON output when ENV=production."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("ENV", "dev") == "production":
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
