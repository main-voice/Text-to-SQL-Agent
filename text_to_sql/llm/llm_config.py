"""Set up the LLM configurations using pydantic."""

from typing import Literal, Optional

from pydantic import BaseSettings, Field, SecretStr


class BaseLLMConfig(BaseSettings):
    """
    The Base LLM configurations class, mapped from .env file
    Avoid using this class directly, use the specific LLMConfig class instead.
    """

    # necessary info
    endpoint: Optional[str] = Field(exclude=True)
    api_key: Optional[SecretStr] = Field(exclude=True)

    # optional info
    model: Optional[str] = Field(default=None)
    llm_source: Optional[str] = Field(default=None)

    # for more accuracy we use lower temperature
    temperature: float = 0.0
    max_tokens: int = 700

    class Config:
        """Config for pydantic class"""

        env_path = "../config/.env"
        env_file_encoding = "utf-8"


class AzureLLMConfig(BaseLLMConfig):
    """The Azure LLM configurations class, mapped from .env file"""

    deployment_name: Literal["gpt-4", "gpt-35-turbo", "gpt-4-turbo"] = "gpt-35-turbo"
    model: Literal["gpt-4", "gpt-35-turbo"] = "gpt-35-turbo"
    llm_source: str = "azure"
    api_version: str = Field(default="2024-03-01-preview", const=True)

    class Config:
        """Config for pydantic class"""

        # add prefix for azure llm related environment variables
        env_prefix = "AZURE_"


class PerplexityLLMConfig(BaseLLMConfig):
    """The Perplexity LLM configurations class, mapped from .env file"""

    model: Literal[
        "sonar-small-chat", "sonar-medium-chat", "mistral-7b-instruct", "mistral-8x7b-instruct"
    ] = "sonar-small-chat"
    llm_source: str = "perplexity"

    class Config:
        """Config for pydantic class"""

        # add prefix for perplexity llm related environment variables
        env_prefix = "PERPLEXITY_"
