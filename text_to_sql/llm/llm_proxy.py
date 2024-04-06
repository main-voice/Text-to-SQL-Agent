"""
Proxy class to supported LLM client
"""

from typing import Any, Literal, Optional

# Seems pylint can't handle import package dynamically
# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from text_to_sql.config.settings import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_GPT_4
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMConfig(BaseModel):
    """
    The Base LLM configurations class
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    model: str


class AzureLLMConfig(BaseLLMConfig):
    """
    Azure LLM configuration class
    """

    azure_endpoint: Optional[str] = Field(exclude=True, default=None)
    deployment_name: Literal["gpt-4", "gpt-35-turbo", "gpt-4-turbo"] = "gpt-35-turbo"
    model: Literal["gpt-4", "gpt-35-turbo"] = "gpt-35-turbo"
    api_version: str = "2023-08-01-preview"
    api_key: Optional[str] = Field(exclude=True, default=None)


class LLMProxy:
    """
    A Proxy class to interact with the LLM
    """

    def __init__(self, llm_name: str = "azure"):
        self.llm_config = None
        self.llm = None

        if llm_name == "azure":
            if AZURE_GPT_4:
                self.llm_config = AzureLLMConfig(
                    azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, model="gpt-4", deployment_name="gpt-4-turbo"
                )
            else:
                self.llm_config = AzureLLMConfig(
                    azure_endpoint=AZURE_ENDPOINT,
                    api_key=AZURE_API_KEY,
                    model="gpt-35-turbo",
                    deployment_name="gpt-35-turbo",
                )
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.llm_config.azure_endpoint,
                deployment_name=self.llm_config.deployment_name,
                model=self.llm_config.model,
                openai_api_type="azure",
                openai_api_version=self.llm_config.api_version,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                openai_api_key=self.llm_config.api_key,
            )
        else:
            raise ValueError("Only Azure LLM supported now!")

    def get_response_from_llm(self, question: Any, verbose: bool = False):
        """
        Get response from the LLM for the given question
        """
        if verbose:
            with get_openai_callback() as cb:
                _response = self.llm.invoke(question)
                logger.info(f"Token usage: {cb.total_tokens}")
                return _response

        return self.llm.invoke(question)
