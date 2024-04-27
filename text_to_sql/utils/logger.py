"""
Create a custom logger with time and file name
"""

import logging

from text_to_sql.config.settings import settings

# Unify the logging format
LOGGING_FORMAT = "%(asctime)s [%(levelname)-7s] %(filename)20s:%(lineno)-4d %(message)s"

logging.basicConfig(format=LOGGING_FORMAT)


def get_logger(name: str):
    """
    Return a logger with the given name (usually the file name)
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOGGING_LEVEL)
    return logger
