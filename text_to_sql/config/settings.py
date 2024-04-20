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


# Embedding model
AZURE_EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDING_MODEL", None)
HUGGING_FACE_EMBEDDING_MODEL = os.environ.get("HUGGING_FACE_EMBEDDING_MODEL", None)


# You Dao translate service
YD_APP_ID = os.environ.get("YD_APP_ID", None)
YD_APP_SECRET_KEY = os.environ.get("YD_APP_SECRET_KEY", None)

# TOP K relevant tables
TOP_K = int(os.environ.get("TOP_K", 5))

# Langchain Smith
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "False") == "True"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", None)
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", None)

SENTRY_DSN = os.environ.get("SENTRY_DSN", None)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
