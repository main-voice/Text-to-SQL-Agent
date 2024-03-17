"""
Interact with the env variables
Create a .env in the same folder first
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

DEBUG = os.environ.get("DEBUG", "False") == "True"
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO

AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", None)
