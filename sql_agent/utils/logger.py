import logging

from sql_agent.config.settings import LOGGING_LEVEL

# Unify the logging format
LOGGING_FORMAT = "%(asctime)s [%(levelname)-7s] %(filename)20s:%(lineno)-4d %(message)s"

logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def get_logger(name: str):
    """
    Return a logger with the given name (usually the file name)
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    return logger
