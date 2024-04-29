"""
Create a custom logger with time and file name
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from text_to_sql.config.settings import settings

LOGGING_FORMAT = "%(asctime)s [%(levelname)-7s] %(filename)20s:%(lineno)-4d %(message)s"


def config_logger(file_name: str, console_level: int = settings.LOGGING_LEVEL, file_level: int = logging.DEBUG) -> None:
    """Configure the logger with the given file name and levels

    Args:
        file_name (str): The file name to get the logger
        console_level (int, optional): Level of console to log. Defaults to settings.LOGGING_LEVEL.
        file_level (int, optional): Level of file to log. Defaults to logging.DEBUG.


    Returns:
        None: No return
    """
    # Unify the logging format
    formatter = logging.Formatter(LOGGING_FORMAT)

    file_handler = RotatingFileHandler(file_name, maxBytes=10 * 1024 * 1024, mode="a", backupCount=5, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    logging.basicConfig(handlers=[file_handler, console_handler], level=settings.LOGGING_LEVEL, format=LOGGING_FORMAT)


# create a global logger to log to file
logs_dir = settings.LOGS_DIR
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# log name format: debug_2021-08-01-12-00-00.log
log_name = "debug.log"
global_file_handler = RotatingFileHandler(os.path.join(logs_dir, log_name), maxBytes=10 * 1024 * 1024)
global_file_handler.setLevel(settings.LOGGING_LEVEL)
global_file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))


def get_logger(name: str):
    """
    Return a logger with the given name (usually the file name)
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOGGING_LEVEL)
    # avoid duplicate logs
    logger.propagate = False

    if settings.LOGGING_LEVEL == logging.DEBUG:
        # log to console and file in DEBUG mode
        logger = add_console_handler(logger)
        logger.addHandler(global_file_handler)
        return logger

    if settings.LOGGING_LEVEL == logging.INFO:
        # only log to console in INFO mode
        logger = add_console_handler(logger)
        return logger

    raise ValueError("Invalid logging level, please set to DEBUG or INFO.")


def add_console_handler(logger):
    """
    Add a console handler to the logger
    """
    ch = logging.StreamHandler()
    ch.setLevel(settings.LOGGING_LEVEL)
    ch.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(ch)
    return logger
