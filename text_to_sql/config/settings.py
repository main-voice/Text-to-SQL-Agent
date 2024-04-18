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
# AZURE
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", None)
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", None)

# if using gpt4 model
AZURE_GPT_4 = os.environ.get("AZURE_GPT_4", "False") == "True"

# PER API
PER_API_KEY = os.environ.get("PER_API_KEY", None)
PER_ENDPOINT = os.environ.get("PER_ENDPOINT", None)


# Embedding model
AZURE_EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDING_MODEL", None)
HUGGING_FACE_EMBEDDING_MODEL = os.environ.get("HUGGING_FACE_EMBEDDING_MODEL", None)

# Database config
MY_DB_HOST = os.environ.get("MY_DB_HOST", None)
MY_DB_USER = os.environ.get("MY_DB_USER", None)
MY_DB_PASSWORD = os.environ.get("MY_DB_PASSWORD", None)
MY_DB_PORT = os.environ.get("MY_DB_PORT", None)
MY_DB_NAME = os.environ.get("MY_DB_NAME", None)

PG_DB_HOST = os.environ.get("PG_DB_HOST", None)
PG_DB_USER = os.environ.get("PG_DB_USER", None)
PG_DB_PASSWORD = os.environ.get("PG_DB_PASSWORD", None)
PG_DB_PORT = os.environ.get("PG_DB_PORT", None)
PG_DB_NAME = os.environ.get("PG_DB_NAME", None)

# You Dao translate service
YD_APP_ID = os.environ.get("YD_APP_ID", None)
YD_APP_SECRET_KEY = os.environ.get("YD_APP_SECRET_KEY", None)

# TOP K relevant tables
TOP_K = int(os.environ.get("TOP_K", 5))

# Langchain Smith
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "False") == "True"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", None)
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", None)
