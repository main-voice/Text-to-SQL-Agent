"""
Proxy class to supported LLM client
"""

from typing import Any, Literal, Optional

# pylint: disable=no-name-in-module
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from text_to_sql.config.settings import AZURE_API_KEY, AZURE_API_VERSION, AZURE_ENDPOINT, PER_API_KEY, PER_ENDPOINT
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMConfig(BaseModel):
    """
    The Base LLM configurations class
    """

    endpoint: Optional[str] = Field(exclude=True, default=None)
    api_key: Optional[SecretStr] = Field(exclude=True, default=None)
    # for more accuracy we use lower temperature
    temperature: float = 0.0
    max_tokens: int = 700
    model: str
    llm_source: str


class AzureLLMConfig(BaseLLMConfig):
    """
    Azure LLM configuration class
    """

    llm_source: str = "azure"
    deployment_name: Literal["gpt-4", "gpt-35-turbo", "gpt-4-turbo"] = "gpt-35-turbo"
    model: Literal["gpt-4", "gpt-35-turbo"] = "gpt-35-turbo"
    api_version: str = AZURE_API_VERSION


class PerplexityLLMConfig(BaseLLMConfig):
    """
    Perplexity LLM configuration class
    """

    llm_source: str = "perplexity"
    model: Literal[
        "sonar-small-chat", "sonar-medium-chat", "mistral-7b-instruct", "mistral-8x7b-instruct"
    ] = "sonar-small-chat"


class AzureLLM:
    """The Azure LLM class"""

    def __init__(self, config: AzureLLMConfig):
        self.llm = self.get_llm(config)

    @staticmethod
    def get_llm(config: AzureLLMConfig) -> AzureChatOpenAI:
        """Static method, will return a Azure LLM instance according to the configuration provided

        Args:
            config (AzureLLMConfig): The config information for the Azure LLM

        Returns:
            AzureChatOpenAI: An azure llm instance defined in langchain_openai
        """
        return AzureChatOpenAI(
            azure_endpoint=config.endpoint or AZURE_ENDPOINT,
            deployment_name=config.deployment_name,
            model=config.model,
            openai_api_type=config.llm_source,
            openai_api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key or AZURE_API_KEY,
        )


class PerplexityLLM:
    """The Perplexity LLM class"""

    def __init__(self, config: PerplexityLLMConfig):
        self.llm = self.get_llm(config)

    @staticmethod
    def get_llm(config: PerplexityLLMConfig) -> ChatOpenAI:
        """Static method, will return a Perplexity LLM instance according to the configuration provided

        Args:
            config (PerplexityLLMConfig): The config information for the Perplexity LLM

        Returns:
            ChatOpenAI: A perplexity llm instance defined in langchain_openai
        """
        return ChatOpenAI(
            base_url=config.endpoint or PER_ENDPOINT,
            api_key=config.api_key or PER_API_KEY,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class LLMProxy:
    """
    A Proxy class to interact with the supported LLMs
    """

    def __init__(self, config: AzureLLMConfig | PerplexityLLMConfig) -> None:
        if config.llm_source == "azure":
            self.llm = AzureLLM(config).llm
        elif config.llm_source == "perplexity":
            self.llm = PerplexityLLM(config).llm
        else:
            raise ValueError("Only Azure LLM and Perplexity LLM supported now!")

    def get_response_from_llm(self, question: Any, verbose: bool = False):
        """method to get response from LLM

        Args:
            question (Any): the question to ask the LLM
            verbose (bool, optional): if to print token usage. Defaults to False.

        Returns:
            BaseMessage: An message object with the response from the LLM
        """
        if verbose:
            with get_openai_callback() as cb:
                _response = self.llm.invoke(question)
                logger.info(f"Token usage: {cb.total_tokens}")
                return _response

        return self.llm.invoke(question)
