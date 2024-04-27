"""
Interact with the env variables
Create a .env in the same folder first
"""

import logging
import os

from pydantic import BaseSettings, Field, SecretStr


class Settings(BaseSettings):
    """Store the settings for the application using pydantic"""

    DEBUG: bool = False
    LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO

    # Embedding model
    AZURE_EMBEDDING_MODEL: str = None
    HUGGING_FACE_EMBEDDING_MODEL: str = None

    # You Dao translate service
    YD_BASE_URL: str = Field(default=None)
    YD_APP_ID: SecretStr = Field(default=None, exclude=True)
    YD_APP_SECRET_KEY: SecretStr = Field(default=None, exclude=True)

    TOP_K: int = 5  # Top K relevant tables

    # Max input length
    MAX_INPUT_LENGTH: int = 256

    # Langchain Smith
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = None
    LANGCHAIN_PROJECT: str = None

    # Sentry
    SENTRY_DSN: SecretStr = Field(default=None, exclude=True)
    ENVIRONMENT: str = "development"

    class Config:
        """Config class for pydantic"""

        # Get the absolute path of the .env file (ATTENTION: relative path maybe failed in some cases)
        # https://github.com/pydantic/pydantic/issues/1368
        current_dir_path = os.path.dirname(__file__)

        env_file = os.path.join(current_dir_path, ".env")


# Singleton settings
settings = Settings()
