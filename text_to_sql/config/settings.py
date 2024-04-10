"""
Interact with the env variables
Create a .env in the same folder first
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# For developer
DEBUG = os.environ.get("DEBUG", "False") == "True"
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO

# LLM Config
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", None)
# if using gpt4 model
AZURE_GPT_4 = os.environ.get("AZURE_GPT_4", "False") == "True"

# Embedding model
AZURE_EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDING_MODEL", None)
HUGGING_FACE_EMBEDDING_MODEL = os.environ.get("HUGGING_FACE_EMBEDDING_MODEL", None)

# Database config
DB_HOST = os.environ.get("DB_HOST", None)
DB_USER = os.environ.get("DB_USER", None)
DB_PASSWORD = os.environ.get("DB_PASSWORD", None)
DB_NAME = os.environ.get("DB_NAME", None)

assert DB_HOST is not None, "Please set DB_HOST variable in .env file under sql_agent/config"
assert DB_USER is not None, "Please set DB_USER variable in .env file under sql_agent/config"
assert DB_PASSWORD is not None, "Please set DB_PASSWORD variable in .env file under sql_agent/config"

# You Dao translate service
YD_APP_ID = os.environ.get("YD_APP_ID", None)
YD_APP_SECRET_KEY = os.environ.get("YD_APP_SECRET_KEY", None)

# TOP K relevant tables
TOP_K = int(os.environ.get("TOP_K", 5))
